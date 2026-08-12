"""Train the weak scout from image-level OK/NOK labels (paper recipe).

Reference recipe (paper §3.1 / §4.3, ``configs/train_default.yaml``):
ImageNet-pretrained ResNet-18 @ 256x256, 1500 steps, batch size 16, AdamW
lr 1e-4 / weight decay 1e-4, gradient-norm clip 1.0, seed 1337. Image-level
BCE is the only supervision — no boxes, no masks.

Usage::

    python -m selective_inspection.train \\
        --config configs/train_default.yaml \\
        --train-manifest data/my_dataset/train.jsonl \\
        --val-manifest data/my_dataset/val.jsonl \\
        --out runs/my_run [--device cuda] [--max-steps N] [--seed N] \\
        [--init-checkpoint prior_checkpoint.pt]

Outputs under ``--out``:

* ``checkpoint.pt``          — model weights + architecture params.
* ``predictions_val.jsonl``  — validation scores (input to the calibrators).
* ``metrics_val.json``       — validation image-AUROC + counts.
* ``resolved_config.json``   — the exact resolved configuration of the run.

Device policy: ``--device cuda`` raises if CUDA is unavailable (no silent
CPU fallback); ``--device auto`` (default) picks CUDA when available and
logs the choice. GPU is strongly recommended for training; CPU is fine for
inference.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .data import load_manifest, preprocess_batch, write_jsonl
from .model import WeakScout, load_checkpoint, save_checkpoint, weak_bce_loss


def resolve_device(preference: str) -> torch.device:
    """``cuda`` raises when unavailable (never a silent CPU fallback)."""
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "--device cuda requested but no CUDA device is available. "
                "Refusing to fall back to CPU silently — pass --device cpu "
                "explicitly if that is really what you want."
            )
        return torch.device("cuda")
    if preference == "cpu":
        return torch.device("cpu")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] --device auto resolved to: {dev}")
    return dev


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def binary_auroc(pairs: list[tuple[float, int]]) -> float | None:
    """Image AUROC from ``(score, label01)`` pairs (rank-based, tie-aware).

    Returns None when only one class is present.
    """
    pos = [s for s, y in pairs if y == 1]
    neg = [s for s, y in pairs if y == 0]
    if not pos or not neg:
        return None
    all_scores = sorted(s for s, _ in pairs)
    # midranks
    ranks: dict[float, float] = {}
    i = 0
    while i < len(all_scores):
        j = i
        while j + 1 < len(all_scores) and all_scores[j + 1] == all_scores[i]:
            j += 1
        ranks[all_scores[i]] = (i + j) / 2.0 + 1.0  # average 1-based rank
        i = j + 1
    rank_sum_pos = sum(ranks[s] for s in pos)
    n_pos, n_neg = len(pos), len(neg)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


@torch.no_grad()
def score_manifest(
    model: WeakScout,
    records: list[dict[str, Any]],
    *,
    device: torch.device,
    batch_size: int = 16,
) -> list[dict[str, Any]]:
    """Score every record; returns prediction rows carrying the manifest label."""
    model.eval()
    rows: list[dict[str, Any]] = []
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        x = preprocess_batch([r["image_path"] for r in batch], model.image_size).to(device)
        scores = model(x).detach().cpu().numpy()
        for rec, s in zip(batch, scores):
            rows.append(
                {
                    "image_id": rec["image_id"],
                    "image_path": rec["image_path"],
                    "image_score": float(s),
                    "label": rec["label"],
                }
            )
    return rows


def train(cfg: dict[str, Any], out_dir: Path, device: torch.device) -> dict[str, Any]:
    """Run the training loop; returns a summary dict."""
    tcfg = cfg["training"]
    seed = int(tcfg["seed"])
    batch_size = int(tcfg["batch_size"])
    max_steps = int(tcfg["max_steps"])
    lr = float(tcfg["learning_rate"])
    weight_decay = float(tcfg["weight_decay"])
    grad_clip = float(tcfg.get("grad_clip_norm", 1.0))
    class_weight = float((tcfg.get("loss") or {}).get("class_weight", 1.0))
    image_weight = float((tcfg.get("loss") or {}).get("image_weight", 1.0))
    log_every = int(tcfg.get("log_every", 10))

    train_records = load_manifest(cfg["manifests"]["train"])
    val_records = load_manifest(cfg["manifests"]["val"]) if cfg["manifests"].get("val") else []
    n_nok = sum(1 for r in train_records if r["label"] == "nok")
    print(
        f"[train] train records: {len(train_records)} "
        f"({len(train_records) - n_nok} ok / {n_nok} nok) | val records: {len(val_records)}"
    )
    if n_nok < 200:
        print(
            f"[train] WARNING: only {n_nok} NOK training images. The paper's "
            "supervision-budget curve (§5.2) found <=50 NOK ~= chance and >=200 NOK "
            "required for reliable performance; treat results below that budget "
            "with suspicion."
        )

    set_seed(seed)
    init_ckpt = cfg.get("init_checkpoint")
    if init_ckpt:
        print(f"[train] initializing from checkpoint: {init_ckpt}")
        model, _ = load_checkpoint(init_ckpt, device="cpu")
        model.to(device)
    else:
        model = WeakScout(cfg["model"]).to(device)

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=weight_decay
    )

    rng = random.Random(seed)
    model.train()
    step = 0
    losses: list[float] = []
    t0 = time.time()
    while step < max_steps:
        rng.shuffle(train_records)
        for i in range(0, len(train_records), batch_size):
            batch = train_records[i : i + batch_size]
            if not batch:
                continue
            x = preprocess_batch([r["image_path"] for r in batch], model.image_size).to(device)
            y = torch.tensor(
                [1.0 if r["label"] == "nok" else 0.0 for r in batch],
                dtype=torch.float32,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model.forward_logits(x)
            ld = weak_bce_loss(logits, y, class_weight=class_weight, image_weight=image_weight)
            loss = ld["loss"]
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            if step % log_every == 0:
                print(
                    f"[train] step={step} loss={losses[-1]:.4f} "
                    f"image_bce={float(ld['loss_image']):.4f}"
                )
            step += 1
            if step >= max_steps:
                break
    wall_s = time.time() - t0
    print(f"[train] done: {step} steps in {wall_s:.1f}s ({wall_s / max(1, step):.2f}s/step)")

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "checkpoint.pt"
    save_checkpoint(str(ckpt_path), model, extra={"step": step, "seed": seed})
    print(f"[train] checkpoint -> {ckpt_path}")

    summary: dict[str, Any] = {
        "steps": step,
        "final_train_loss": losses[-1] if losses else None,
        "train_wall_s": wall_s,
        "device": str(device),
        "seed": seed,
        "checkpoint": str(ckpt_path),
    }

    if val_records:
        val_rows = score_manifest(model, val_records, device=device, batch_size=batch_size)
        write_jsonl(out_dir / "predictions_val.jsonl", val_rows)
        pairs = [(r["image_score"], 1 if r["label"] == "nok" else 0) for r in val_rows]
        auroc = binary_auroc(pairs)
        val_metrics = {
            "n": len(val_rows),
            "n_ok": sum(1 for _, y in pairs if y == 0),
            "n_nok": sum(1 for _, y in pairs if y == 1),
            "image_auroc": auroc,
        }
        (out_dir / "metrics_val.json").write_text(json.dumps(val_metrics, indent=2) + "\n")
        summary["val"] = val_metrics
        print(
            f"[train] val: n={val_metrics['n']} image_auroc="
            f"{'n/a (single class)' if auroc is None else f'{auroc:.4f}'} "
            f"-> {out_dir / 'predictions_val.jsonl'}"
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", required=True, help="YAML config (see configs/train_default.yaml)")
    p.add_argument("--out", required=True, help="output run directory")
    p.add_argument("--train-manifest", default=None, help="override manifests.train")
    p.add_argument("--val-manifest", default=None, help="override manifests.val")
    p.add_argument("--max-steps", type=int, default=None, help="override training.max_steps")
    p.add_argument("--batch-size", type=int, default=None, help="override training.batch_size")
    p.add_argument("--seed", type=int, default=None, help="override training.seed")
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument(
        "--init-checkpoint",
        default=None,
        help="initialize weights from a prior checkpoint (fine-tuning); overrides model config",
    )
    p.add_argument(
        "--no-pretrained",
        action="store_true",
        help="do not load ImageNet backbone weights (offline smoke runs)",
    )
    args = p.parse_args(argv)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
        return 2
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg.setdefault("model", {})
    cfg.setdefault("training", {})
    cfg.setdefault("manifests", {})
    if args.train_manifest:
        cfg["manifests"]["train"] = args.train_manifest
    if args.val_manifest:
        cfg["manifests"]["val"] = args.val_manifest
    if args.max_steps is not None:
        cfg["training"]["max_steps"] = args.max_steps
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.seed is not None:
        cfg["training"]["seed"] = args.seed
    if args.init_checkpoint:
        cfg["init_checkpoint"] = args.init_checkpoint
    if args.no_pretrained:
        cfg["model"]["pretrained"] = False
    if not cfg["manifests"].get("train"):
        print("ERROR: no train manifest (config manifests.train or --train-manifest)", file=sys.stderr)
        return 2

    device = resolve_device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved_config.json").write_text(json.dumps(cfg, indent=2, default=str) + "\n")

    summary = train(cfg, out_dir, device)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
