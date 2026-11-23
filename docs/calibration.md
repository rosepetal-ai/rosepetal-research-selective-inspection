---
title: "Calibration"
description: "Calibrating the Fast-OK-Exit gate under a missed-NOK budget and the split-conformal accept/review/reject layer on validation scores: rules, command options, output files, realised test rates and production use."
sidebar_position: 2
---

# Calibration

The cascade has three thresholds: `tau_exit` for the exit gate, and `tau_accept` and `tau_reject` for the decision layer. Both tools compute them from validation scores only; the thresholds are then applied unchanged to the test set and to production. Neither tool trains anything: calibrating again takes one pass over a scores file.

## Prediction files

Both tools read JSONL files with one row per image and at least `image_score`. The label comes from the row's `label` field, which `train.py` (validation scores) and `infer.py --manifest` write. For rows without a label, such as those of `infer.py --images` or of your own inspector, pass the manifest with `--val-manifest` or `--test-manifest`: labels are joined by `image_id`. A row that ends up without a label stops the tool with `no OK/NOK label for image_id=...`.

The scores can come from any model, as long as a higher score means more anomalous.

## Exit gate

An image exits (is accepted as OK without further work) when its scout score is **below** `tau_exit`, and is routed to the inspector otherwise.

For a missed-NOK budget `B` and the `n` validation NOK scores sorted in ascending order, the tool allows `k = floor(B × n)` NOK images to exit and sets `tau_exit` to the score of the (k+1)-th lowest one: the largest threshold that lets out no more than `k` validation NOK images, which is also the one that lets out the most OK images. With `B = 0`, `tau_exit` is the lowest validation NOK score, so no validation NOK image exits. With 188 validation NOK images, as in the paper, `B = 0.01` allows one.

```bash
python -m selective_inspection.calibrate_gate \
    --val-predictions runs/scout_seed1337/predictions_val.jsonl \
    --budget 0.0 0.01 \
    --test-predictions runs/scout_seed1337/predictions_test.jsonl \
    --out runs/scout_seed1337/gate.json
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--val-predictions` | required | Validation scores |
| `--val-manifest` | none | Manifest to take the validation labels from |
| `--budget` | `0.0 0.01` | One or more missed-NOK budgets, as fractions from 0 up to (not including) 1 |
| `--test-predictions` | none | Test scores, used only to report the rates of the calibrated thresholds |
| `--test-manifest` | none | Manifest to take the test labels from |
| `--out` | none | File to write the JSON to; it is always printed |

The output has `n_val_ok`, `n_val_nok` and one entry per budget in `budgets`, with `budget_missed_nok`, `tau_exit`, the statistics on the validation set (`val_expected`) and, with test scores, on the test set (`test_realized_unclamped`). The statistics are:

| Field | Meaning |
|-------|---------|
| `n_ok`, `n_nok` | OK and NOK images in the set |
| `ok_exit`, `ok_exit_rate` | OK images that exit, and their share of the OK images |
| `missed_nok`, `missed_nok_rate` | NOK images that exit (missed defects), and their share of the NOK images |
| `nok_recall_routed` | Share of the NOK images routed to the inspector: the exit-stage NOK recall |

The tool stops with an error when the validation set has no NOK image: the rule needs NOK scores.

## Decision layer

The layer judges the score of the last model in the cascade (the inspector, or the scout when there is none) and answers accept, review or reject. It is calibrated on the validation **OK** scores, `n_cal` of them, sorted in ascending order. For a level `alpha`, the threshold is the k-th lowest score, with `k = ceil((n_cal + 1) × (1 - alpha))`. Then:

- **accept** when the score is at most `tau_accept` (level `alpha_accept`);
- **reject** when the score is above `tau_reject` (level `alpha_reject`);
- **review** in between.

If validation and production images are exchangeable, the probability that an OK image is rejected is at most `alpha_reject`. This is a marginal bound, on average over calibration sets and not per category, and a shift in the image distribution voids it. It can only be certified when `alpha ≥ 1 / (n_cal + 1)`: below that floor the threshold is infinite (written `Infinity` in the JSON), `certifiable` is `false`, and for `alpha_reject` a `warning` says that nothing will ever be rejected. With the paper's 614 OK calibration images the floor is 0.001626; certifying a 0.1 % false-reject rate needs about 1,000 OK calibration images.

```bash
python -m selective_inspection.calibrate_conformal \
    --val-predictions runs/scout_seed1337/predictions_val.jsonl \
    --alpha-accept 0.10 --alpha-reject 0.005 \
    --test-predictions runs/scout_seed1337/predictions_test.jsonl \
    --out runs/scout_seed1337/conformal.json
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--val-predictions` | required | Validation scores of the model the layer will judge |
| `--val-manifest` | none | Manifest to take the validation labels from |
| `--alpha-accept` | `0.10` | Level of the accept threshold, between 0 and 1 |
| `--alpha-reject` | `0.005` | Level of the reject threshold, between 0 and 1, and not above `--alpha-accept` |
| `--test-predictions` | none | Test scores, used only to report the rates of the calibrated thresholds |
| `--test-manifest` | none | Manifest to take the test labels from |
| `--out` | none | File to write the JSON to; it is always printed |

The output has `n_cal_ok`, `certifiable_alpha_floor`, `alpha_accept`, `alpha_reject`, `tau_accept`, `tau_reject` and `certifiable` (true only when both levels are certifiable). With test scores it adds `test_realized_unclamped`:

| Field | Meaning |
|-------|---------|
| `n`, `n_ok`, `n_nok` | Images in the test set |
| `ok_rejected`, `fa_per_10k_ok` | OK images rejected (false alarms), and the same per 10,000 OK images |
| `nok_accepted`, `missed_defect_rate` | NOK images accepted, and their share of the NOK images |
| `review_rate` | Share of all test images sent to review |

The layer only works with the scores it was calibrated on. In a cascade with an inspector, calibrate it on the inspector's validation scores, written in the same JSONL format; changing the inspector means calibrating again on the same validation OK images, without retraining anything.

## In production

The paper's deployment guidance (§8), and what the code expects:

- **Apply the thresholds exactly as calibrated.** In the paper, rounding the inspector's threshold at the fourth decimal lost one more defect, with no error anywhere (§6.2). The tools write full-precision values; read them from the JSON files.
- **Choose `B`.** Start at 0. In the paper, relaxing to 1 % or less let 4.1 more points of OK images exit with no measured loss of test recall, but at the resolution of a single missed image; check it on your own validation data before relying on it.
- **Expect throughput, not latency.** The scout runs on every image and every routed image still runs the full path, so the exit lowers the mean cost per image but not the worst case. Size worst-case latency and provisioning against the full path.
- **Recalibrate rather than adjust by hand.** Watch the realised OK-exit rate, review rate, false-reject rate and, from whatever labelled audits you have, the missed-NOK rate. A drifting OK-exit or review rate is the first sign that the OK scores have shifted: calibrate again on a fresh validation sample.
- **Size the calibration set first.** The smallest false-reject level you can certify is `1 / (n_cal + 1)`: count your OK calibration images before promising a budget.
