"""Fine-tune a three-head bridge with endpoint loss weight 5."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import json
import torch

import train_curved_multibranch_toy_v2 as toy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    cli = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(456)
    source = cli.source
    payload = torch.load(source, map_location='cpu', weights_only=False)
    args = SimpleNamespace(**payload['args'])
    args.device = 'cpu'
    args.batch_size = 256
    args.steps = 30000
    args.lambda_score = 1.0
    args.lambda_bridge = 0.5
    args.lambda_transport = 0.25
    args.lambda_endpoint = 5.0
    args.lambda_corr = 1e-4
    model = toy.build_model(args)
    model.load_state_dict(payload['model'], strict=True)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
    rows = []
    for step in range(1, args.steps + 1):
        batch = toy.sample_batch(args.batch_size, torch.device('cpu'), args)
        total, detail = toy.compute_losses(model, batch, args)
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 1000 == 0:
            rows.append({'step': step, **detail})
            print(f"step {step:05d} total={detail['total']:.5e} endpoint={detail['endpoint']:.5e}", flush=True)
    training_settings = {'name': 'endpoint_fine_tuning_2',
     'steps': 30000,
     'seed': 456,
     'device': 'cpu',
     'torch_threads': 2,
     'batch_size': 256,
     'optimizer': 'AdamW',
     'learning_rate': 5e-05,
     'weight_decay': 1e-05,
     'gradient_clip': 1.0,
     'loss_weights': {'score': 1.0,
                      'bridge': 0.5,
                      'transport': 0.25,
                      'endpoint': 5.0,
                      'correction_regularization': 0.0001}}
    outdir = cli.outdir
    (outdir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    save = {'model': model.state_dict(), 'args': vars(args), 'parent_checkpoint': str(source), 'training_settings': training_settings, 'losses': rows}
    torch.save(save, outdir / 'checkpoints/latest.pt')
    torch.save(save, outdir / 'checkpoints/ckpt_step_0030000.pt')
    (outdir / 'finetune_config.json').write_text(json.dumps({'parent': str(source), 'args': vars(args), 'training_settings': training_settings, 'losses': rows}, indent=2), encoding='utf-8')
    print(f"saved {outdir}")


if __name__ == '__main__':
    main()
