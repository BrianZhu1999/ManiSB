#!/usr/bin/env python3
"""静态检查 ManiSB 项目包的数据、图和配置。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(path: Path, checks: list[str], root: Path) -> None:
    rel = path.relative_to(root).as_posix()
    if not path.is_file():
        checks.append(f"missing: {rel}")


def check_npz(path: Path, expected: dict[str, tuple[int, ...]], checks: list[str], root: Path) -> dict[str, Any]:
    rel = path.relative_to(root).as_posix()
    if not path.is_file():
        checks.append(f"missing: {rel}")
        return {"path": rel, "status": "missing"}
    with np.load(path, allow_pickle=False) as archive:
        observed = {key: tuple(archive[key].shape) for key in archive.files}
        for key, shape in expected.items():
            if key not in archive.files:
                checks.append(f"missing array: {rel}:{key}")
                continue
            if tuple(archive[key].shape) != shape:
                checks.append(f"shape mismatch: {rel}:{key}={archive[key].shape}, expected={shape}")
            if not np.isfinite(archive[key]).all():
                checks.append(f"non-finite values: {rel}:{key}")
    return {"path": rel, "status": "ok", "arrays": observed, "sha256": sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    checks: list[str] = []

    required_files = [
        "README.md",
        "README_zh.md",
        "LICENSE",
        "requirements.txt",
        "configs/case1_swirl.json",
        "configs/case2_single.json",
        "configs/case2_triad.json",
        "configs/case2_evaluation.json",
        "provenance/PROJECT_MANIFEST.json",
        "figures/paper/fig_smooth_closed_coupling.pdf",
        "figures/paper/fig3_multibranch_linear_v5.pdf",
        "code/case1_smooth_closed/README_zh.md",
        "code/case2_curved_multibranch/README_zh.md",
    ]
    for rel in required_files:
        require(root / rel, checks, root)

    case1 = check_npz(
        root / "data/case1_smooth_closed/closed_coupling_data.npz",
        {"target": (1024, 2), "terminal": (1024, 2), "times": (61,), "trajectory_triad": (61, 1024, 2)},
        checks,
        root,
    )
    case2 = check_npz(
        root / "data/case2_curved_multibranch/toy_case2_checkpoint_trajectories.npz",
        {"clean": (4096, 2), "degraded": (4096, 2), "branch": (4096,), "trajectory_no_projection": (21, 4096, 2)},
        checks,
        root,
    )

    stray = root / "provenance/case2_"
    if stray.exists():
        checks.append("stray file: provenance/case2_")

    manifest_path = root / "provenance/PROJECT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    report = {
        "project": manifest.get("project_name", "ManiSB"),
        "root": str(root),
        "status": "passed" if not checks else "failed",
        "errors": checks,
        "case1": case1,
        "case2": case2,
    }
    output = root / "results/project_verification.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not checks else 1


if __name__ == "__main__":
    raise SystemExit(main())
