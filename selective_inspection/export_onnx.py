"""Export the weak scout to ONNX (opset 17, dynamic batch) + parity check.

The exported graph maps a preprocessed batch ``images (B, 3, S, S)`` float32
in ``[0, 1]`` to ``scores (B,)`` float32 in ``[0, 1]`` — the sigmoid of the
image head, i.e. exactly the score used by the gate and the conformal layer.

CLI::

    python -m selective_inspection.export_onnx \\
        --checkpoint runs/my_run/checkpoint.pt --out runs/my_run/scout.onnx \\
        [--opset 17] [--verify-images img1.png img2.png ...] [--tolerance 1e-4]

``--verify-images`` runs BOTH the PyTorch model and the exported ONNX model
(onnxruntime) on the given images through the SAME preprocessing and asserts
``|torch_score - onnx_score| < tolerance`` per image (exit code 1 on
failure). Always verify after export — an ONNX file that passes the checker
but diverges numerically is worse than no export.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import preprocess_batch
from .model import WeakScout, load_checkpoint


class _ScoreWrapper(nn.Module):
    """Wrap :class:`WeakScout` so the ONNX graph outputs the image score only."""

    def __init__(self, model: WeakScout) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model.forward_logits(images).image_logits.squeeze(-1))


def export(checkpoint: str | Path, out_path: str | Path, *, opset: int = 17) -> Path:
    """Export ``checkpoint`` to ONNX at ``out_path``; returns the output path."""
    model, _meta = load_checkpoint(str(checkpoint), device="cpu")
    model.eval()
    wrapper = _ScoreWrapper(model).eval()
    s = model.image_size
    dummy = torch.zeros((1, 3, s, s), dtype=torch.float32)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(out_path),
        input_names=["images"],
        output_names=["scores"],
        dynamic_axes={"images": {0: "batch"}, "scores": {0: "batch"}},
        opset_version=opset,
    )
    import onnx

    onnx.checker.check_model(onnx.load(str(out_path)))
    print(f"[export_onnx] exported + checked: {out_path} (opset {opset}, dynamic batch)")
    return out_path


def verify(
    checkpoint: str | Path,
    onnx_path: str | Path,
    images: list[str],
    *,
    tolerance: float = 1e-4,
) -> bool:
    """Torch-vs-ONNX parity on ``images``; True iff every |diff| < tolerance."""
    import onnxruntime as ort

    model, _meta = load_checkpoint(str(checkpoint), device="cpu")
    x = preprocess_batch(images, model.image_size)
    with torch.no_grad():
        torch_scores = model(x).numpy().astype(np.float32)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    (onnx_scores,) = sess.run(["scores"], {"images": x.numpy().astype(np.float32)})

    ok = True
    print(f"[export_onnx] parity check (tolerance {tolerance:g}):")
    print(f"{'image':<48} {'torch':>10} {'onnx':>10} {'|diff|':>12}")
    for path, ts, os_ in zip(images, torch_scores, onnx_scores):
        diff = abs(float(ts) - float(os_))
        status = "OK" if diff < tolerance else "FAIL"
        ok = ok and diff < tolerance
        print(f"{Path(path).name:<48} {float(ts):>10.6f} {float(os_):>10.6f} {diff:>12.2e}  {status}")
    print(f"[export_onnx] parity: {'PASS' if ok else 'FAIL'} (max |diff| = "
          f"{max(abs(float(t) - float(o)) for t, o in zip(torch_scores, onnx_scores)):.2e})")
    return ok


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True, help="output .onnx path")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--verify-images", nargs="*", default=None,
                   help="images for the torch-vs-onnx parity assertion")
    p.add_argument("--tolerance", type=float, default=1e-4)
    args = p.parse_args(argv)

    export(args.checkpoint, args.out, opset=args.opset)
    if args.verify_images:
        if not verify(args.checkpoint, args.out, args.verify_images, tolerance=args.tolerance):
            print("ERROR: torch-vs-onnx parity check FAILED", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
