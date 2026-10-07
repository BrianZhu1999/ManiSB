from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path

import numpy as np
import torch


def as_namespace(values: dict):
    class Namespace:
        pass

    obj = Namespace()
    for key, value in values.items():
        setattr(obj, key, value)
    return obj


def to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def export_case1(repo: Path, output: Path, device: torch.device) -> None:
    sys.path.insert(0, str(repo))
    import toy_bp_gi2sb_with_straightness as toy1

    checkpoint = repo / "toy_outputs_gpu_straightness" / "bp_gi2sb_toy_model.pt"
    payload = torch.load(checkpoint, map_location=device)
    args = as_namespace(payload["args"])
    model = toy1.BPGI2SBToy(hidden=args.hidden, depth=args.depth).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()

    theta = torch.linspace(0.0, 2.0 * math.pi, 64, device=device).view(-1, 1)
    theta_dense = torch.linspace(0.0, 2.0 * math.pi, 1200, device=device).view(-1, 1)
    clean_dense, degraded_dense = toy1.make_pair(theta_dense, kind=args.kind)

    torch.manual_seed(20260801)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260801)
    clean, degraded, final_no, traj_no = toy1.sample_trajectories(
        model,
        theta,
        args.kind,
        args.sigma_min,
        args.sigma_max,
        args.gamma_max,
        args.rho_max,
        args.traj_steps,
        use_projection=False,
    )

    torch.manual_seed(20260801)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260801)
    _, _, final_proj, traj_proj = toy1.sample_trajectories(
        model,
        theta,
        args.kind,
        args.sigma_min,
        args.sigma_max,
        args.gamma_max,
        args.rho_max,
        args.traj_steps,
        use_projection=True,
    )

    bridge_times = np.asarray([0.25, 0.50, 0.75], dtype=np.float64)
    clean_np = to_numpy(clean_dense)
    degraded_np = to_numpy(degraded_dense)
    bridge_curves = np.stack(
        [(1.0 - t) * clean_np + t * degraded_np for t in bridge_times], axis=0
    )
    np.savez_compressed(
        output / "toy_case1_checkpoint_trajectories.npz",
        seed=np.asarray([20260801]),
        theta=to_numpy(theta),
        clean=to_numpy(clean),
        degraded=to_numpy(degraded),
        clean_dense=clean_np,
        degraded_dense=degraded_np,
        bridge_times=bridge_times,
        bridge_curves=bridge_curves,
        trajectory_no_projection=to_numpy(traj_no),
        trajectory_projection=to_numpy(traj_proj),
        final_no_projection=to_numpy(final_no),
        final_projection=to_numpy(final_proj),
        rho_max=np.asarray([args.rho_max]),
        nfe=np.asarray([args.traj_steps]),
        training_steps=np.asarray([args.steps]),
        hidden_width=np.asarray([args.hidden]),
        depth=np.asarray([args.depth]),
    )
    print(
        "Case 1 metadata:",
        {"steps": args.steps, "hidden": args.hidden, "depth": args.depth, "kind": args.kind},
    )


def load_case2_model(repo: Path, device: torch.device):
    import train_curved_multibranch_toy_v2 as toy2

    checkpoint = repo / "runs_toy_curved_bp_v2" / "checkpoints" / "latest.pt"
    payload = torch.load(checkpoint, map_location=device)
    args = as_namespace(payload["args"])
    model = toy2.BPModelV2(
        hidden=args.hidden,
        depth=args.depth,
        gamma_max=args.gamma_max,
        detach_m_for_heads=getattr(args, "detach_m_for_heads", False),
    ).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return toy2, model, args, payload


@torch.no_grad()
def export_case2(repo: Path, output: Path, device: torch.device) -> None:
    toy2, model, args, payload = load_case2_model(repo, device)

    torch.manual_seed(2026)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2026)
    batch = toy2.sample_batch(4096, device, args)

    run_args = copy.copy(args)
    final_no, traj_no, times_no = toy2.sample_reverse(
        model,
        batch["x1"],
        run_args,
        nfe=20,
        rho_max=0.0,
        rho_sched="sin2",
        projection_mode="none",
        return_traj=True,
    )
    final_proj, traj_proj, times_proj = toy2.sample_reverse(
        model,
        batch["x1"],
        run_args,
        nfe=20,
        rho_max=0.02,
        rho_sched="sin2",
        projection_mode="adaptive",
        return_traj=True,
    )

    t_values = np.asarray([0.0, 0.25, 0.50, 0.75, 1.0], dtype=np.float64)
    branch_values = (-1.0, 1.0)
    curves = []
    for branch in branch_values:
        branch_curves = []
        for t_value in t_values:
            values = toy2.make_branch_curves(
                2500, device, kappa=args.kappa, t_value=float(t_value)
            )[branch]["mt"]
            branch_curves.append(to_numpy(values))
        curves.append(np.stack(branch_curves, axis=0))

    np.savez_compressed(
        output / "toy_case2_checkpoint_trajectories.npz",
        seed=np.asarray([2026]),
        clean=to_numpy(batch["x0"]),
        degraded=to_numpy(batch["x1"]),
        branch=to_numpy(batch["y"]),
        t_values=t_values,
        branch_values=np.asarray(branch_values),
        bridge_curves=np.stack(curves, axis=0),
        times_no_projection=to_numpy(times_no),
        times_projection=to_numpy(times_proj),
        trajectory_no_projection=to_numpy(traj_no),
        trajectory_projection=to_numpy(traj_proj),
        final_no_projection=to_numpy(final_no),
        final_projection=to_numpy(final_proj),
        rho_max=np.asarray([0.02]),
        nfe=np.asarray([20]),
        kappa=np.asarray([args.kappa]),
        training_step=np.asarray([payload.get("step", -1)]),
        hidden_width=np.asarray([args.hidden]),
        depth=np.asarray([args.depth]),
    )
    print(
        "Case 2 metadata:",
        {
            "checkpoint_step": payload.get("step", -1),
            "steps": args.steps,
            "hidden": args.hidden,
            "depth": args.depth,
            "kappa": args.kappa,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    export_case1(args.repo.resolve(), args.output.resolve(), device)
    export_case2(args.repo.resolve(), args.output.resolve(), device)
    print(f"Exported checkpoint-derived toy data to {args.output.resolve()}")


if __name__ == "__main__":
    main()
