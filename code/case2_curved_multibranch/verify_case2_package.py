"""检查 ManiSB 总包中的案例二数据、哈希和实验配置。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.root.resolve()
    data_path = root / "data" / "case2_curved_multibranch" / "toy_case2_checkpoint_trajectories.npz"
    manifest_path = root / "provenance" / "case2_source_hashes.json"
    required = [
        data_path,
        manifest_path,
        root / "configs" / "case2_single.json",
        root / "configs" / "case2_triad.json",
        root / "configs" / "case2_evaluation.json",
    ]
    missing = [str(path.relative_to(root)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Case II files: " + ", ".join(missing))

    archive_hash = sha256(data_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest["files"]["data/toy_case2_checkpoint_trajectories.npz"]["sha256"]
    if archive_hash != expected:
        raise AssertionError(f"Data SHA-256 mismatch: {archive_hash} != {expected}")

    with np.load(data_path, allow_pickle=False) as values:
        expected_shapes = {
            "clean": (4096, 2),
            "degraded": (4096, 2),
            "trajectory_no_projection": (21, 4096, 2),
            "trajectory_projection": (21, 4096, 2),
            "final_no_projection": (4096, 2),
            "final_projection": (4096, 2),
        }
        for name, shape in expected_shapes.items():
            if values[name].shape != shape:
                raise AssertionError(f"Unexpected shape for {name}: {values[name].shape} != {shape}")
            if not np.isfinite(values[name]).all():
                raise AssertionError(f"Non-finite values in {name}")

        final_checks = {
            "final_no_projection": float(np.max(np.abs(values["trajectory_no_projection"][-1] - values["final_no_projection"]))),
            "final_projection": float(np.max(np.abs(values["trajectory_projection"][-1] - values["final_projection"]))),
        }
        if max(final_checks.values()) > 1e-6:
            raise AssertionError(f"Final trajectory mismatch: {final_checks}")
        if float(values["kappa"][0]) != 0.6:
            raise AssertionError("Unexpected kappa")
        if int(values["nfe"][0]) != 20:
            raise AssertionError("Unexpected NFE")
        if float(values["rho_max"][0]) != 0.02:
            raise AssertionError("Unexpected projection strength")
        shapes = {name: list(values[name].shape) for name in expected_shapes}

    report = {
        "status": "pass",
        "data_sha256": archive_hash,
        "array_shapes": shapes,
        "final_replay_max_abs_error": final_checks,
        "kappa": 0.6,
        "reverse_updates": 20,
        "projection_strength_max": 0.02,
    }
    output = root / "results" / "case2_curved_multibranch" / "package_verification.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
