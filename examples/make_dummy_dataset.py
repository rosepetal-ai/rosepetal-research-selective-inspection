#!/usr/bin/env python3
"""Generate a tiny synthetic OK/NOK dataset + JSONL manifests (smoke test).

OK images are smooth gradient textures with mild noise; NOK images carry a
bright scratch or blob defect. The point is NOT realism — it is a fast,
fully self-contained way to exercise the whole chain
(train -> infer -> ONNX export -> calibrate -> pipeline) without downloading
any real dataset.

Usage::

    python examples/make_dummy_dataset.py --out data/dummy \\
        [--n-train 64] [--n-val 32] [--n-test 16] [--image-size 256] [--seed 0]

Writes ``<out>/images/<split>/*.png`` and ``<out>/<split>.jsonl`` manifests
({"image_id", "image_path", "label"}); each split is half OK / half NOK.
Image paths are written relative to the manifest's directory (portable).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def make_image(rng: np.random.Generator, size: int, nok: bool) -> np.ndarray:
    """One synthetic RGB image: smooth textured background (+ defect if NOK)."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    base = 90.0 + 60.0 * (0.5 * yy + 0.5 * xx) + rng.normal(0.0, 6.0, (size, size))
    img = np.stack([base + rng.normal(0, 2), base + rng.normal(0, 2), base + rng.normal(0, 2)], -1)
    if nok:
        kind = rng.integers(0, 2)
        if kind == 0:  # bright scratch: a thick random line
            x0, y0 = rng.integers(0, size, 2)
            angle = rng.uniform(0, np.pi)
            length = rng.integers(size // 4, size // 2)
            thickness = int(rng.integers(2, 5))
            for t in range(length):
                px = int(x0 + t * np.cos(angle))
                py = int(y0 + t * np.sin(angle))
                if 0 <= px < size and 0 <= py < size:
                    lo_y, hi_y = max(0, py - thickness), min(size, py + thickness)
                    lo_x, hi_x = max(0, px - thickness), min(size, px + thickness)
                    img[lo_y:hi_y, lo_x:hi_x] = 235.0
        else:  # dark blob
            cx, cy = rng.integers(size // 4, 3 * size // 4, 2)
            r = int(rng.integers(size // 16, size // 8))
            dist = (np.mgrid[0:size, 0:size][0] - cy) ** 2 + (np.mgrid[0:size, 0:size][1] - cx) ** 2
            img[dist < r * r] = 25.0
    return np.clip(img, 0, 255).astype(np.uint8)


def write_split(out: Path, split: str, n: int, size: int, rng: np.random.Generator) -> None:
    img_dir = out / "images" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        nok = i % 2 == 1  # half OK / half NOK, interleaved
        label = "nok" if nok else "ok"
        image_id = f"{split}_{label}_{i:04d}"
        rel_path = f"images/{split}/{image_id}.png"
        Image.fromarray(make_image(rng, size, nok)).save(out / rel_path)
        rows.append({"image_id": image_id, "image_path": rel_path, "label": label})
    manifest = out / f"{split}.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    n_nok = sum(1 for r in rows if r["label"] == "nok")
    print(f"[make_dummy_dataset] {split}: {n} images ({n - n_nok} ok / {n_nok} nok) -> {manifest}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--n-train", type=int, default=64)
    p.add_argument("--n-val", type=int, default=32)
    p.add_argument("--n-test", type=int, default=16)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    write_split(out, "train", args.n_train, args.image_size, rng)
    write_split(out, "val", args.n_val, args.image_size, rng)
    write_split(out, "test", args.n_test, args.image_size, rng)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
