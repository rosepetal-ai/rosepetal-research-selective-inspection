"""PyTorch inference: score images with a trained weak scout.

CLI::

    # Score loose image files
    python -m selective_inspection.infer --checkpoint runs/my_run/checkpoint.pt \\
        --images img1.png img2.png [--device auto] [--out scores.jsonl]

    # Score a JSONL manifest (labels are carried through -> calibration-ready)
    python -m selective_inspection.infer --checkpoint runs/my_run/checkpoint.pt \\
        --manifest data/my_dataset/val.jsonl --out runs/my_run/predictions_val.jsonl

Python API::

    from selective_inspection.infer import load_model, score_images
    model, image_size = load_model("runs/my_run/checkpoint.pt", device="cuda")
    scores = score_images(model, ["img1.png", "img2.png"])   # list[float] in [0, 1]

Scores are ``sigmoid(image_head)`` in ``[0, 1]``; higher = more anomalous
(NOK-like). Preprocessing is the frozen letterbox-256 + ``[0, 1]`` scaling of
``selective_inspection.data`` (NO ImageNet mean/std).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

from .data import load_manifest, preprocess_batch, write_jsonl
from .model import WeakScout, load_checkpoint
from .train import resolve_device


def load_model(checkpoint: str | Path, device: str | torch.device = "cpu") -> tuple[WeakScout, int]:
    """Load a checkpoint -> ``(eval-mode model on device, image_size)``."""
    model, _meta = load_checkpoint(str(checkpoint), device=device)
    return model, model.image_size


@torch.no_grad()
def score_images(
    model: WeakScout,
    images: Sequence[str | Path],
    *,
    batch_size: int = 16,
) -> list[float]:
    """Score image files -> list of floats in ``[0, 1]`` (higher = more anomalous)."""
    model.eval()
    device = next(model.parameters()).device
    scores: list[float] = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        x = preprocess_batch(list(batch), model.image_size).to(device)
        scores.extend(float(s) for s in model(x).detach().cpu().numpy())
    return scores


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", nargs="+", help="image files to score")
    src.add_argument("--manifest", help="JSONL manifest to score (labels carried through)")
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out", default=None, help="write predictions JSONL here")
    args = p.parse_args(argv)

    device = resolve_device(args.device)
    model, _image_size = load_model(args.checkpoint, device=device)

    rows: list[dict[str, Any]] = []
    if args.manifest:
        records = load_manifest(args.manifest)
        paths = [r["image_path"] for r in records]
        scores = score_images(model, paths, batch_size=args.batch_size)
        for rec, s in zip(records, scores):
            rows.append(
                {
                    "image_id": rec["image_id"],
                    "image_path": rec["image_path"],
                    "image_score": s,
                    "label": rec["label"],
                }
            )
    else:
        for path in args.images:
            if not Path(path).exists():
                print(f"ERROR: image not found: {path}", file=sys.stderr)
                return 2
        scores = score_images(model, args.images, batch_size=args.batch_size)
        for path, s in zip(args.images, scores):
            rows.append({"image_id": Path(path).stem, "image_path": str(path), "image_score": s})

    for r in rows:
        print(f"{r['image_score']:.6f}  {r['image_path']}")
    if args.out:
        write_jsonl(args.out, rows)
        print(f"[infer] predictions -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
