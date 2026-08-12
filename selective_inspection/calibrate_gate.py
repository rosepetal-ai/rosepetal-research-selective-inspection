"""Calibrate the risk-calibrated Fast-OK-Exit gate (paper §3.2).

The recall-guarantee rule
-------------------------
Gate convention: an image EXITS early (admitted OK, no further compute)
iff ``image_score < tau_exit``; it is ROUTED to the inspector iff
``image_score >= tau_exit``.

Given the validation NOK scores sorted ascending and an operator-chosen
missed-NOK budget ``B`` (fraction, e.g. 0.0 or 0.01), the calibrated
threshold is the LARGEST ``tau_exit`` such that the number of wrongly-exited
validation NOK images stays within the budget:

    k_allowed = floor(B * n_val_nok)
    tau_exit  = val_nok_sorted[k_allowed]        # exits the k strictly-smaller NOK
    (if k_allowed >= n_val_nok, every image may exit: tau_exit = max_score + 1)

For ``B = 0`` this is the minimum validation NOK score — no validation NOK
exits. Choosing the largest admissible threshold MAXIMISES early OK exits
subject to the guarantee.

Val -> test discipline
----------------------
``tau_exit`` is calibrated on VALIDATION predictions only and applied FROZEN
to the test set / production stream. Never tune it on test data. If test
predictions are supplied, the realised test rates are reported unclamped so
the val -> test generalisation gap stays visible.

CLI::

    python -m selective_inspection.calibrate_gate \\
        --val-predictions runs/my_run/predictions_val.jsonl \\
        [--val-manifest data/.../val.jsonl] \\
        --budget 0.0 0.01 \\
        [--test-predictions runs/my_run/predictions_test.jsonl \\
         [--test-manifest data/.../test.jsonl]] \\
        [--out gate.json]

Prediction files are JSONL rows with ``image_score`` (+ ``label``, or join
labels via the matching manifest). Output: JSON with one entry per budget —
``tau_exit`` + expected (validation) stats + realised test stats if given.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .data import load_predictions, split_scores


def tau_exit_for_budget(val_nok_sorted: list[float], budget: float) -> float:
    """Largest ``tau_exit`` keeping wrongly-exited validation NOK within ``budget``."""
    if not 0.0 <= budget < 1.0:
        raise ValueError(f"budget must be in [0, 1); got {budget}")
    n_nok = len(val_nok_sorted)
    if n_nok == 0:
        raise ValueError(
            "validation set contains no NOK images — the recall-guarantee rule "
            "cannot be calibrated without validation NOK scores"
        )
    k_allowed = int(budget * n_nok)  # floor
    if k_allowed >= n_nok:
        return val_nok_sorted[-1] + 1.0
    return val_nok_sorted[k_allowed]


def gate_stats(nok: list[float], ok: list[float], tau_exit: float) -> dict[str, Any]:
    """Exit/miss stats at ``tau_exit`` (exit iff score < tau_exit)."""
    n_nok, n_ok = len(nok), len(ok)
    missed_nok = sum(1 for s in nok if s < tau_exit)  # exited NOK = missed by definition
    ok_exit = sum(1 for s in ok if s < tau_exit)
    return {
        "n_nok": n_nok,
        "n_ok": n_ok,
        "missed_nok": missed_nok,
        "missed_nok_rate": (missed_nok / n_nok) if n_nok else 0.0,
        "ok_exit": ok_exit,
        "ok_exit_rate": (ok_exit / n_ok) if n_ok else 0.0,
        "nok_recall_routed": 1.0 - ((missed_nok / n_nok) if n_nok else 0.0),
    }


def calibrate(
    val_rows: list[dict[str, Any]],
    budgets: list[float],
    test_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Calibrate ``tau_exit`` per budget on val; optionally report realised test stats."""
    val_nok, val_ok = split_scores(val_rows)
    result: dict[str, Any] = {
        "rule": (
            "Fast-OK-Exit recall guarantee: exit iff image_score < tau_exit; "
            "tau_exit = largest threshold keeping wrongly-exited VALIDATION NOK "
            "within the budget B (calibrated on val only, applied frozen)"
        ),
        "n_val_ok": len(val_ok),
        "n_val_nok": len(val_nok),
        "budgets": [],
    }
    test_split = split_scores(test_rows) if test_rows else None
    for b in budgets:
        tau = tau_exit_for_budget(val_nok, b)
        entry: dict[str, Any] = {
            "budget_missed_nok": b,
            "tau_exit": tau,
            "val_expected": gate_stats(val_nok, val_ok, tau),
        }
        if test_split is not None:
            entry["test_realized_unclamped"] = gate_stats(test_split[0], test_split[1], tau)
        result["budgets"].append(entry)
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--val-predictions", required=True)
    p.add_argument("--val-manifest", default=None, help="manifest to join labels from (if the "
                   "prediction rows do not carry a 'label' field)")
    p.add_argument("--budget", type=float, nargs="+", default=[0.0, 0.01],
                   help="missed-NOK budget(s) B as fractions (default: 0.0 0.01)")
    p.add_argument("--test-predictions", default=None,
                   help="optional test predictions for realised-rate reporting (never used "
                        "for calibration)")
    p.add_argument("--test-manifest", default=None)
    p.add_argument("--out", default=None, help="write the calibration JSON here")
    args = p.parse_args(argv)

    val_rows = load_predictions(args.val_predictions, args.val_manifest)
    test_rows = (
        load_predictions(args.test_predictions, args.test_manifest)
        if args.test_predictions
        else None
    )
    result = calibrate(val_rows, list(args.budget), test_rows)
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload + "\n")
        print(f"[calibrate_gate] -> {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
