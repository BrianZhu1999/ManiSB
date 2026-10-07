"""检查 ManiSB 总包中的案例一图文件和派生数据。"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.root.resolve()
    figure_dir = root / "figures" / "case1_smooth_closed"
    data_path = root / "data" / "case1_smooth_closed" / "closed_coupling_data.npz"
    required = [
        figure_dir / "fig_smooth_closed_coupling.pdf",
        figure_dir / "fig_smooth_closed_coupling.svg",
        figure_dir / "fig_smooth_closed_coupling.png",
        figure_dir / "fig_smooth_closed_coupling.tiff",
        data_path,
    ]
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Case I files: " + ", ".join(missing))

    archive = np.load(data_path, allow_pickle=False)
    if not all(np.isfinite(archive[key]).all() for key in archive.files):
        raise AssertionError("Case I archive contains non-finite values")

    svg = figure_dir / "fig_smooth_closed_coupling.svg"
    tree = ET.parse(svg)
    texts = tree.findall(".//{http://www.w3.org/2000/svg}text")
    if len(texts) == 0:
        raise AssertionError("Case I SVG has no text elements")

    sizes = {}
    for suffix in ("png", "tiff"):
        with Image.open(figure_dir / f"fig_smooth_closed_coupling.{suffix}") as image:
            sizes[suffix] = list(image.size)
    if sizes["png"] != sizes["tiff"]:
        raise AssertionError(f"PNG/TIFF dimensions differ: {sizes}")

    report = {
        "status": "pass",
        "data": str(data_path.relative_to(root)),
        "data_arrays": {key: list(archive[key].shape) for key in archive.files},
        "svg_text_count": len(texts),
        "raster_sizes": sizes,
    }
    output = root / "results" / "case1_smooth_closed" / "export_verification.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
