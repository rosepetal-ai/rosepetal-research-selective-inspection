"""Manifest loading + image preprocessing (single source of truth for parity).

Manifest contract
-----------------
A dataset is a JSONL file — one JSON object per line — with at least:

* ``image_path`` (str) — absolute path, or a path relative to the current
  working directory, or relative to the manifest file's own directory
  (resolved in that order).
* ``label`` (str) — ``"ok"`` (normal) or ``"nok"`` (defective),
  case-insensitive. Anything else raises.

Optional: ``image_id`` (str, defaults to the file stem) and any extra
metadata fields (ignored by this package, preserved on predictions where
noted).

Preprocessing (frozen; used identically by training, PyTorch inference and
the ONNX examples)
------------------------------------------------------------------------
1. Load as RGB (PIL).
2. Aspect-preserving resize so the longest side equals ``image_size``
   (bilinear), centered zero-padding to a square canvas ("letterbox").
3. Scale to ``[0, 1]`` float32, channel-first ``(3, S, S)``.

Note: NO ImageNet mean/std normalization is applied — the paper's scout was
trained on ``[0, 1]`` inputs, so any deployment (Python, ONNX, C++) must
reproduce exactly this preprocessing to match the calibrated thresholds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL manifest; resolve each ``image_path``; validate labels."""
    path = Path(path)
    records: list[dict[str, Any]] = []
    with path.open() as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "image_path" not in rec:
                raise ValueError(f"{path}:{lineno}: missing 'image_path'")
            label = str(rec.get("label", "")).lower()
            if label not in ("ok", "nok"):
                raise ValueError(
                    f"{path}:{lineno}: label must be 'ok' or 'nok', got {rec.get('label')!r}"
                )
            rec["label"] = label
            rec["image_path"] = str(resolve_image_path(rec["image_path"], manifest_dir=path.parent))
            rec.setdefault("image_id", Path(rec["image_path"]).stem)
            records.append(rec)
    if not records:
        raise ValueError(f"manifest is empty: {path}")
    return records


def resolve_image_path(image_path: str, *, manifest_dir: Path | None = None) -> Path:
    """Resolve absolute -> cwd-relative -> manifest-dir-relative (first match)."""
    p = Path(image_path)
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    if manifest_dir is not None:
        candidate = (manifest_dir / p).resolve()
        if candidate.exists():
            return candidate
    return p  # let the loader raise a clear FileNotFoundError later


def load_image_rgb(path: str | Path) -> np.ndarray:
    """Load an image as an RGB uint8 HWC array."""
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def letterbox(image: np.ndarray, image_size: int) -> np.ndarray:
    """Aspect-preserving letterbox of an RGB uint8 HWC array to ``(S, S, 3)`` uint8.

    Bilinear resize (PIL) of the image to fit, centered on a zero (black)
    canvas — identical to the paper's training-time preprocessing.
    """
    if image.ndim == 2:
        image = image[..., None]
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    h, w = image.shape[:2]
    scale = float(image_size) / float(max(h, w, 1))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    resized = np.asarray(
        Image.fromarray(image.astype(np.uint8)).resize((new_w, new_h), resample=Image.BILINEAR),
        dtype=np.uint8,
    )
    canvas = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    pad_y = (image_size - new_h) // 2
    pad_x = (image_size - new_w) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas


def preprocess(image: np.ndarray | str | Path, image_size: int) -> torch.Tensor:
    """Path or RGB uint8 HWC array -> ``(3, S, S)`` float32 tensor in ``[0, 1]``."""
    if isinstance(image, (str, Path)):
        image = load_image_rgb(image)
    canvas = letterbox(image, image_size)
    chw = np.transpose(canvas.astype(np.float32) / 255.0, (2, 0, 1))
    return torch.from_numpy(np.ascontiguousarray(chw))


def preprocess_batch(images: Sequence[np.ndarray | str | Path], image_size: int) -> torch.Tensor:
    """Stack :func:`preprocess` over a sequence -> ``(B, 3, S, S)``."""
    return torch.stack([preprocess(im, image_size) for im in images], dim=0)


class ManifestDataset(torch.utils.data.Dataset):
    """Torch ``Dataset`` over a JSONL manifest -> ``(image_tensor, label_float, index)``.

    Provided for API users; the reference training loop in ``train.py``
    iterates records directly (deterministic seeded shuffling, as in the
    paper recipe) and does not require this class.
    """

    def __init__(self, manifest_path: str | Path, image_size: int) -> None:
        self.records = load_manifest(manifest_path)
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        rec = self.records[idx]
        x = preprocess(rec["image_path"], self.image_size)
        y = torch.tensor(1.0 if rec["label"] == "nok" else 0.0, dtype=torch.float32)
        return x, y, idx


# --- prediction-file helpers (shared by the calibration CLIs) ---------------


def load_predictions(
    predictions_path: str | Path,
    manifest_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load a predictions JSONL (``image_score`` + ``image_id``); join labels.

    Labels come from the prediction rows themselves (``label`` field, written
    by ``infer.py --manifest``) or are joined by ``image_id`` from
    ``manifest_path``. Raises if any row ends up unlabeled — calibration
    without labels is meaningless.
    """
    labels: dict[str, str] = {}
    if manifest_path is not None:
        for rec in load_manifest(manifest_path):
            labels[str(rec["image_id"])] = rec["label"]
    rows: list[dict[str, Any]] = []
    with Path(predictions_path).open() as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "image_score" not in r:
                raise ValueError(f"{predictions_path}:{lineno}: missing 'image_score'")
            label = str(r.get("label") or labels.get(str(r.get("image_id")), "")).lower()
            if label not in ("ok", "nok"):
                raise ValueError(
                    f"{predictions_path}:{lineno}: no OK/NOK label for image_id="
                    f"{r.get('image_id')!r} — pass the matching --*-manifest"
                )
            r["label"] = label
            r["image_score"] = float(r["image_score"])
            rows.append(r)
    if not rows:
        raise ValueError(f"predictions file is empty: {predictions_path}")
    return rows


def split_scores(rows: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
    """-> ``(sorted NOK scores ascending, sorted OK scores ascending)``."""
    nok = sorted(r["image_score"] for r in rows if r["label"] == "nok")
    ok = sorted(r["image_score"] for r in rows if r["label"] != "nok")
    return nok, ok


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""))


__all__ = [
    "ManifestDataset",
    "letterbox",
    "load_image_rgb",
    "load_manifest",
    "load_predictions",
    "preprocess",
    "preprocess_batch",
    "resolve_image_path",
    "split_scores",
    "write_jsonl",
]
