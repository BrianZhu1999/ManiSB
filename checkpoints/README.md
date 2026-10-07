# Checkpoint storage

Model weights are kept outside the public Git package because the archived
weights are large binary files and the paper figures can be inspected from the
derived trajectory archives.

For the archived Case II evaluation, place the weights using the following
names when reproducing the notebook:

```text
checkpoints/single_latest.pt
checkpoints/triad_latest.pt
```

The original server-side checkpoint paths and hashes are recorded in
`provenance/case2_source_hashes.json`. The integrity check uses the derived
trajectory archives and configuration records, so it runs independently of
the checkpoint files.
