"""Calibrate the split-conformal accept/review/reject decision layer (paper §3.4).

Scheme (split conformal via order statistics — NOT interpolated quantiles)
--------------------------------------------------------------------------
Calibration scores are the VALIDATION **OK** image scores (``n_cal`` of
them, sorted ascending). For a miss level ``alpha``:

    k          = ceil((n_cal + 1) * (1 - alpha))
    tau(alpha) = k-th smallest calibration score        (certifiable iff k <= n_cal,
                                                         i.e. alpha >= 1 / (n_cal + 1))

Decision rule on an operative scorer's image score ``s``:

    accept  iff s <= tau(alpha_accept)
    reject  iff s >  tau(alpha_reject)
    review  otherwise (the band in between)

Under exchangeability the reject rate carries the marginal finite-sample
guarantee ``P(OK rejected) <= alpha_reject`` — certifiable ONLY when
``alpha_reject >= 1 / (n_cal + 1)``. A requested level below that floor
yields ``tau = +inf`` and ``certifiable: false`` (the guarantee would be
vacuous); size the calibration set before promising a budget (a 0.1%
false-reject budget needs ~1000 OK calibration images).

Val -> test discipline: thresholds are calibrated on validation OK scores
only and applied FROZEN; realised test rates (if test predictions are
supplied) are reported unclamped.

The layer is scorer-agnostic: calibrate it on the validation scores of
WHICHEVER scorer's output it will gate in production (the scout, or the
downstream inspector) — the same rule and guarantee apply.

CLI::

    python -m selective_inspection.calibrate_conformal \\
        --val-predictions runs/my_run/predictions_val.jsonl \\
        [--val-manifest data/.../val.jsonl] \\
        [--alpha-accept 0.10] [--alpha-reject 0.005] \\
        [--test-predictions runs/my_run/predictions_test.jsonl \\
         [--test-manifest data/.../test.jsonl]] \\
        [--out conformal.json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from .data import load_predictions, split_scores


def conformal_tau(cal_sorted: list[float], alpha: float) -> tuple[float, bool]:
    """Order-statistic split-conformal threshold at miss level ``alpha``.

    Returns ``(tau, certifiable)``. ``tau`` is the k-th smallest calibration
    score with ``k = ceil((n+1)(1-alpha))``; if ``k > n`` the level is not
    certifiable from ``n`` calibration points and ``tau = +inf``.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    n = len(cal_sorted)
    if n == 0:
        raise ValueError("no OK calibration scores — cannot calibrate the conformal layer")
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return float("inf"), False
    return cal_sorted[k - 1], True


def decide(score: float, tau_accept: float, tau_reject: float) -> str:
    """Three-way decision for one image score."""
    if score <= tau_accept:
        return "accept"
    if score > tau_reject:
        return "reject"
    return "review"


def realized_rates(
    rows: list[dict[str, Any]], tau_accept: float, tau_reject: float
) -> dict[str, Any]:
    """Realised accept/review/reject rates on a labeled prediction set (unclamped)."""
    nok, ok = split_scores(rows)
    n = len(rows)
    n_ok, n_nok = len(ok), len(nok)
    fa = sum(1 for s in ok if s > tau_reject)  # OK rejected (false alarm)
    mdr = sum(1 for s in nok if s <= tau_accept)  # NOK accepted (missed defect)
    review = sum(1 for r in rows if tau_accept < r["image_score"] <= tau_reject)
    return {
        "n": n,
        "n_ok": n_ok,
        "n_nok": n_nok,
        "ok_rejected": fa,
        "fa_per_10k_ok": round(fa / n_ok * 10_000, 1) if n_ok else None,
        "nok_accepted": mdr,
        "missed_defect_rate": round(mdr / n_nok, 4) if n_nok else None,
        "review_rate": round(review / n, 4) if n else None,
    }


def calibrate(
    val_rows: list[dict[str, Any]],
    *,
    alpha_accept: float = 0.10,
    alpha_reject: float = 0.005,
    test_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Calibrate the accept/review/reject thresholds on validation OK scores."""
    if alpha_accept < alpha_reject:
        raise ValueError(
            f"alpha_accept ({alpha_accept}) must be >= alpha_reject ({alpha_reject}); "
            "otherwise the accept threshold would sit above the reject threshold and "
            "the review band would be inverted"
        )
    _, val_ok = split_scores(val_rows)
    cal = sorted(val_ok)
    n_cal = len(cal)
    floor = 1.0 / (n_cal + 1)
    tau_accept, cert_a = conformal_tau(cal, alpha_accept)
    tau_reject, cert_r = conformal_tau(cal, alpha_reject)
    result: dict[str, Any] = {
        "scheme": (
            "split conformal via order statistics on VALIDATION OK scores: "
            "tau(alpha) = k-th smallest with k = ceil((n_cal+1)(1-alpha)); "
            "accept iff score <= tau_accept; reject iff score > tau_reject; "
            "review in between. Marginal guarantee P(OK rejected) <= alpha_reject "
            "under exchangeability, certifiable iff alpha >= 1/(n_cal+1). "
            "Calibrated on val only, applied frozen."
        ),
        "n_cal_ok": n_cal,
        "certifiable_alpha_floor": round(floor, 6),
        "alpha_accept": alpha_accept,
        "alpha_reject": alpha_reject,
        "tau_accept": tau_accept,
        "tau_reject": tau_reject,
        "certifiable": bool(cert_a and cert_r),
    }
    if not cert_r:
        result["warning"] = (
            f"alpha_reject={alpha_reject} is below the certifiable floor "
            f"1/(n_cal+1)={floor:.6f} for n_cal={n_cal}: tau_reject=+inf (nothing "
            "is ever rejected) and the guarantee is vacuous. Collect more OK "
            "calibration images or raise alpha_reject."
        )
    if test_rows is not None:
        result["test_realized_unclamped"] = realized_rates(test_rows, tau_accept, tau_reject)
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--val-predictions", required=True)
    p.add_argument("--val-manifest", default=None)
    p.add_argument("--alpha-accept", type=float, default=0.10)
    p.add_argument("--alpha-reject", type=float, default=0.005)
    p.add_argument("--test-predictions", default=None,
                   help="optional test predictions for realised-rate reporting")
    p.add_argument("--test-manifest", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    val_rows = load_predictions(args.val_predictions, args.val_manifest)
    test_rows = (
        load_predictions(args.test_predictions, args.test_manifest)
        if args.test_predictions
        else None
    )
    result = calibrate(
        val_rows,
        alpha_accept=args.alpha_accept,
        alpha_reject=args.alpha_reject,
        test_rows=test_rows,
    )
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload + "\n")
        print(f"[calibrate_conformal] -> {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
