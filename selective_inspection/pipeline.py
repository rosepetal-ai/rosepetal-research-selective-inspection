"""Selective-inspection cascade demo: scout -> Fast-OK-Exit gate -> inspector hook.

One production image traverses the cascade as follows (paper §3, Figure 1):

1. The weak scout scores the image.
2. If ``scout_score < tau_exit`` the image is admitted OK on the spot
   (Fast-OK-Exit) — no further compute is charged.
3. Otherwise the image is forwarded to the INSPECTOR, and the (optional)
   conformal layer turns the inspector's score into the line's
   accept / review / reject decision.

Inspector interface (deliberately external — paper §3.3)
---------------------------------------------------------
Any callable ``inspector(image: np.ndarray) -> float`` can be plugged in:

* input: the ORIGINAL RGB image as an HWC uint8 numpy array (the inspector
  owns its own preprocessing — it may run at a different resolution);
* output: a single float anomaly score, HIGHER = MORE ANOMALOUS.

If a conformal layer is attached, its thresholds MUST have been calibrated
(``calibrate_conformal.py``) on validation scores of the SAME scorer whose
output it gates here — swapping the inspector requires recalibrating the
conformal layer (one validation pass, no retraining).

CLI demo::

    python -m selective_inspection.pipeline \\
        --checkpoint runs/my_run/checkpoint.pt \\
        --gate-json runs/my_run/gate.json [--budget 0.0] \\
        [--conformal-json runs/my_run/conformal.json] \\
        --manifest data/my_dataset/test.jsonl [--device auto] [--out decisions.jsonl]

The CLI demo, lacking a real inspector, reuses the scout score as the
inspector score (a stand-in so the full decision path can be exercised
end-to-end). In production, plug in a real detector via the Python API.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .calibrate_conformal import decide
from .data import load_image_rgb, load_manifest, preprocess, write_jsonl
from .infer import load_model
from .model import WeakScout
from .train import resolve_device

Inspector = Callable[[np.ndarray], float]


class SelectiveInspectionPipeline:
    """Scout -> gate -> inspector (-> conformal decision) cascade.

    Parameters
    ----------
    scout:
        Trained :class:`WeakScout` (eval mode, on its device).
    tau_exit:
        Frozen exit threshold from ``calibrate_gate.py``. Exit iff
        ``scout_score < tau_exit``.
    inspector:
        Optional callable ``(HWC uint8 RGB array) -> float`` (higher = more
        anomalous). ``None`` -> routed images carry no inspector score.
    tau_accept / tau_reject:
        Optional frozen conformal thresholds from ``calibrate_conformal.py``,
        calibrated on the SAME scorer that produces the score they gate
        (the inspector when one is attached, else the scout).
    """

    def __init__(
        self,
        scout: WeakScout,
        tau_exit: float,
        inspector: Inspector | None = None,
        tau_accept: float | None = None,
        tau_reject: float | None = None,
    ) -> None:
        self.scout = scout
        self.tau_exit = float(tau_exit)
        self.inspector = inspector
        if (tau_accept is None) != (tau_reject is None):
            raise ValueError("tau_accept and tau_reject must be provided together")
        self.tau_accept = tau_accept
        self.tau_reject = tau_reject

    @torch.no_grad()
    def scout_score(self, image: np.ndarray) -> float:
        device = next(self.scout.parameters()).device
        x = preprocess(image, self.scout.image_size).unsqueeze(0).to(device)
        return float(self.scout(x).item())

    def run(self, image: np.ndarray | str | Path) -> dict[str, Any]:
        """Run one image through the cascade; returns the full decision trace."""
        if isinstance(image, (str, Path)):
            image = load_image_rgb(image)
        s_scout = self.scout_score(image)
        if s_scout < self.tau_exit:
            return {
                "scout_score": s_scout,
                "exited": True,
                "stage": "fast_ok_exit",
                "inspector_score": None,
                "decision": "accept",
            }
        result: dict[str, Any] = {
            "scout_score": s_scout,
            "exited": False,
            "stage": "inspector",
            "inspector_score": None,
            "decision": None,
        }
        operative = s_scout
        if self.inspector is not None:
            operative = float(self.inspector(image))
            result["inspector_score"] = operative
        if self.tau_accept is not None and self.tau_reject is not None:
            result["decision"] = decide(operative, self.tau_accept, self.tau_reject)
        return result


def _tau_from_gate_json(gate_json: str | Path, budget: float | None) -> float:
    payload = json.loads(Path(gate_json).read_text())
    entries = payload["budgets"]
    if budget is None:
        return float(entries[0]["tau_exit"])
    for e in entries:
        if abs(float(e["budget_missed_nok"]) - budget) < 1e-12:
            return float(e["tau_exit"])
    raise SystemExit(
        f"budget {budget} not found in {gate_json} "
        f"(available: {[e['budget_missed_nok'] for e in entries]})"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True, help="trained scout checkpoint")
    gate = p.add_mutually_exclusive_group(required=True)
    gate.add_argument("--gate-json", help="output of calibrate_gate.py")
    gate.add_argument("--tau-exit", type=float, help="explicit frozen exit threshold")
    p.add_argument("--budget", type=float, default=None,
                   help="which budget entry of --gate-json to use (default: first)")
    p.add_argument("--conformal-json", default=None, help="output of calibrate_conformal.py")
    p.add_argument("--manifest", required=True, help="JSONL manifest of images to run")
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--out", default=None, help="write per-image decisions JSONL here")
    args = p.parse_args(argv)

    device = resolve_device(args.device)
    scout, _ = load_model(args.checkpoint, device=device)
    tau_exit = args.tau_exit if args.tau_exit is not None else _tau_from_gate_json(
        args.gate_json, args.budget
    )

    tau_accept = tau_reject = None
    if args.conformal_json:
        conf = json.loads(Path(args.conformal_json).read_text())
        tau_accept, tau_reject = float(conf["tau_accept"]), float(conf["tau_reject"])

    # Demo inspector: reuse the scout score (stand-in; see module docstring).
    pipe = SelectiveInspectionPipeline(
        scout, tau_exit, inspector=None, tau_accept=tau_accept, tau_reject=tau_reject
    )
    print(f"[pipeline] tau_exit={tau_exit} tau_accept={tau_accept} tau_reject={tau_reject}")
    print("[pipeline] NOTE: demo inspector = scout score reused; plug a real "
          "detector via the Python API for production use.")

    records = load_manifest(args.manifest)
    rows: list[dict[str, Any]] = []
    n_exit = 0
    decisions: dict[str, int] = {}
    for rec in records:
        r = pipe.run(rec["image_path"])
        r["image_id"] = rec["image_id"]
        r["label"] = rec["label"]
        rows.append(r)
        n_exit += int(r["exited"])
        key = r["decision"] or "no_decision_layer"
        decisions[key] = decisions.get(key, 0) + 1
    summary = {
        "n": len(rows),
        "exited": n_exit,
        "ok_exit_stage_rate": round(n_exit / len(rows), 4),
        "decisions": decisions,
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        write_jsonl(args.out, rows)
        print(f"[pipeline] decisions -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
