# Model weights

Model weights are not included in this repository. The published trajectories
and per-sample metrics are in `data/`.

The Case II evaluation notebook reads these weight files:

```text
checkpoints/single_latest.pt
checkpoints/triad_latest.pt
```

The model settings are listed in
[`provenance/case2_source_hashes.json`](../provenance/case2_source_hashes.json).

The three-head checkpoint follows 50000 initial updates and two endpoint-aware
fine-tuning stages of 20000 and 30000 updates. The single-head checkpoint follows
50000 updates. Training commands are given in the
[experiment guide](../docs/experiments.md#training).

After training, copy the final checkpoints to the notebook input paths:

```bash
cp runs/case2_single/checkpoints/latest.pt checkpoints/single_latest.pt
cp runs/case2_triad/checkpoints/latest.pt checkpoints/triad_latest.pt
```
