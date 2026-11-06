"""Evaluate geometric correction using paired endpoints and initial noise."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import csv
import hashlib
import json
import time

import numpy as np
from scipy.spatial import cKDTree
import torch
import original_closed_coupling as toy

BASE=Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def reverse(model,x1,initial,args,corrected):
    x=initial.clone()
    trajectory=[x.clone()]
    times=torch.linspace(.999,.001,61)
    for t,tn in zip(times[:-1],times[1:]):
        x=toy.ddim_like_bridge_step(model,x,x1,float(t),float(tn),args.sigma_min,
            args.sigma_max,args.gamma_max,.12,corrected)
        trajectory.append(x.clone())
    return torch.stack(trajectory),times


def turning(traj):
    delta=np.diff(traj.astype('float64'),axis=0)
    length=np.linalg.norm(delta,axis=-1)
    valid=(length[:-1]>1e-8)&(length[1:]>1e-8)
    cosine=(delta[:-1]*delta[1:]).sum(axis=-1)/np.maximum(length[:-1]*length[1:],1e-30)
    angles=np.where(valid,np.arccos(np.clip(cosine,-1,1)),0)
    return np.rad2deg(angles).sum(axis=0)


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--reference-archive',type=Path,required=True)
    parser.add_argument('--root',type=Path,default=BASE)
    options=parser.parse_args()
    CHECKPOINT=options.checkpoint.resolve()
    ARCHIVE=options.reference_archive.resolve()
    OUT=options.root.resolve()/'data/case1_smooth_closed'
    RESULTS=options.root.resolve()/'results/case1_smooth_closed'
    torch.set_num_threads(2)
    OUT.mkdir(parents=True,exist_ok=True)
    RESULTS.mkdir(parents=True,exist_ok=True)
    payload=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    args=SimpleNamespace(**payload['args'])
    assert args.kind=='swirl' and args.steps==20000
    model=toy.BPGI2SBToy(hidden=args.hidden,depth=args.depth).eval()
    model.load_state_dict(payload['model'],strict=True)
    old=dict(np.load(ARCHIVE))
    replay={}
    for corrected,key in [(False,'no_projection'),(True,'projection')]:
        original=old['trajectory_'+key]
        trajectory,_=reverse(model,torch.from_numpy(old['degraded']),torch.from_numpy(original[0]),args,corrected)
        difference=float(np.abs(trajectory.numpy()-original).max())
        replay[key]=difference
        assert difference<2e-5,(key,difference)
    # The two samplers share endpoint pairs and initial noise.
    generator=torch.Generator().manual_seed(20261002)
    theta=2*np.pi*torch.rand(1024,1,generator=generator)
    clean,terminal=toy.make_pair(theta,args.kind)
    sigma=toy.sigma_schedule(torch.full((1024,1),.999),args.sigma_min,args.sigma_max)
    initial=terminal+.25*sigma*torch.randn(terminal.shape,generator=generator)
    dense=torch.linspace(0,2*np.pi,1200)[:,None]
    c0,c1=toy.make_pair(dense,args.kind)
    times=torch.linspace(.999,.001,61)
    supports=np.stack([((1-t)*c0+t*c1).numpy() for t in times])
    trees=[cKDTree(c) for c in supports]
    raw=dict(theta=theta.numpy().ravel(),target=clean.numpy(),terminal=terminal.numpy(),
             initial=initial.numpy(),target_curve=c0.numpy(),terminal_curve=c1.numpy(),
             support_curves=supports,times=times.numpy())
    ids=[np.abs(np.angle(np.exp(1j*(raw['theta']-a)))).argmin() for a in np.arange(8)*2*np.pi/8]
    raw['selected_ids']=np.array(ids)
    raw['detail_sample_id']=np.array([ids[5]])
    metrics,rows={},[]
    # Check that sampling is independent of the analytic pair generator.
    pair_generator=toy.make_pair
    def forbidden(*args,**kwargs):
        raise AssertionError('The sampler accessed the analytic pair generator.')
    toy.make_pair=forbidden
    try:
        for key,corrected in [('triad',False),('corrected',True)]:
            start=time.perf_counter()
            trajectory,_=reverse(model,terminal,initial,args,corrected)
            seconds=time.perf_counter()-start
            traj=trajectory.numpy()
            assert np.array_equal(traj[0],raw['initial'])
            endpoint=np.linalg.norm(traj[-1]-raw['target'],axis=-1)
            support_time=np.stack([tree.query(traj[k],workers=2)[0] for k,tree in enumerate(trees)])
            support=support_time.mean(axis=0)
            excess=np.linalg.norm(np.diff(traj,axis=0),axis=-1).sum(axis=0)-np.linalg.norm(traj[-1]-traj[0],axis=-1)
            angles=turning(traj)
            legacy=toy.trajectory_straightness_metrics(trajectory,clean)
            raw['trajectory_'+key]=traj
            raw['endpoint_'+key]=endpoint
            raw['support_'+key]=support
            raw['support_time_'+key]=support_time
            raw['excess_'+key]=excess
            raw['turning_'+key]=angles
            metrics[key]=dict(mean_endpoint_distance=float(endpoint.mean()),
                coordinate_rmse=float(np.sqrt(np.mean((traj[-1]-raw['target'])**2))),
                mean_support_distance=float(support.mean()),mean_excess_length=float(excess.mean()),
                mean_turning_degrees=float(angles.mean()),
                legacy_turning_degrees=legacy['total_turning_angle_deg'],
                inference_seconds=seconds,full_model_calls=120 if corrected else 60)
            for i in range(1024):
                rows.append(dict(method=key,sample_id=i,endpoint_distance=float(endpoint[i]),
                            support_distance=float(support[i]),excess_length=float(excess[i]),
                            turning_degrees=float(angles[i])))
    finally:
        toy.make_pair=pair_generator
    # Compare the 1200-point support distances with a finer reference curve.
    fine=torch.linspace(0,2*np.pi,9600)[:,None]
    f0,f1=toy.make_pair(fine,args.kind)
    discretization={}
    for key in ['triad','corrected']:
        fine_values=[]
        for k,t in enumerate(times):
            curve=((1-t)*f0+t*f1).numpy()
            fine_values.append(cKDTree(curve).query(raw['trajectory_'+key][k],workers=2)[0])
        fine_values=np.stack(fine_values)
        discretization[key]=dict(mean_support_9600=float(fine_values.mean()),
             change_from_1200=float(fine_values.mean()-raw['support_'+key].mean()))
    np.savez_compressed(OUT/'closed_coupling_data.npz',**raw)
    with (OUT/'per_sample_metrics.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report=dict(device='CPU, 2 threads',checkpoint=CHECKPOINT.name,
        case='Smooth closed coupling (swirl)',config={key:value for key,value in vars(args).items() if key!='outdir'},
        training_seed=args.seed,parameters=sum(p.numel() for p in model.parameters()),
        evaluation_seed=20261002,samples=1024,reverse_steps=60,correction_strength=.12,
        same_endpoint_pairs=True,same_initial_noise=True,
        support_curve_points=1200,discretization=discretization,metrics=metrics,
        data_sha256=sha(OUT/'closed_coupling_data.npz'),
        metric_note='Endpoint distance is Euclidean. Coordinate RMSE averages squared errors over samples and coordinates. legacy_turning_degrees uses a denominator offset of 1e-8; mean_turning_degrees uses normalized displacement vectors.')
    (RESULTS/'data_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
