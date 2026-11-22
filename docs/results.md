---
title: "Paper Results"
description: "What the paper Learning When Not to Inspect measured on MVTec AD: exit rates under a missed-NOK budget, the supervision-budget curve, end-to-end recall and cost, the conformal layer over four scorers, the protocol and its limits."
sidebar_position: 4
---

# Paper Results

Every number on this page comes from the paper ([PDF](./paper.pdf)); the section or table is given in parentheses. They hold for the paper's dataset, splits and protocol, described first.

## What this repository reproduces

The code trains the weak scout with the paper's recipe, calibrates the exit gate and the decision layer with the paper's rules, and reports the same exit and decision-layer rates on your own splits. The repository does not include the paper's split manifests, trained checkpoints, the inspector used for the end-to-end measurements, the baselines (YOLO11 classifiers, PatchCore, RF-DETR-S) or the latency harness, so the end-to-end and latency figures below cannot be reproduced with it alone.

## Protocol

- **Data** (§2, §4.1). MVTec AD, all 15 categories, re-carved from its original unsupervised split into two training pools: a few-NOK pool of 3,088 images (2,361 OK, 727 NOK) and a balanced pool of 1,456 images (729 OK, 727 NOK). Every model is scored on one shared held-out test set of 802 images (614 OK, 188 NOK), built with seed 1337. A validation set of the same construction and composition is the only set used to calibrate thresholds; the test set is never used for tuning.
- **Disjointness** (§4.2). Checked by image identifier, so renamed copies count as overlap. The intersection of every training pool with the test set is empty.
- **Comparability** (§4.1, §4.3). Only the label-free baseline rows are comparable with the original unsupervised MVTec AD leaderboard. The label-free baselines are trained once on the OK images of all 15 categories pooled, so they score below the per-category numbers their authors publish.
- **Training** (§4.3). The scout is trained with a single seed (1337), except for the supervision-budget curve, which adds seeds 2027 and 4242.
- **Hardware and latency** (§4.4). One NVIDIA RTX 4090, fp32, PyTorch eager. Every latency comes from one protocol: batch 1, 256 px input, 50-image warm-up, strict serial execution, five timed sweeps with per-image medians.
- **Statistics** (§4.5). 95 % bootstrap intervals over the test set; exact Clopper–Pearson intervals for the exit-stage recall. AUROC differences below about 0.01 are read as ties.

## The scout against other models

Image AUROC on the shared test set (Table 2):

| Model (supervision) | Few-NOK pool | Balanced pool |
|---------------------|-------------:|--------------:|
| Weak scout (image labels only) | 0.9339 | 0.9110 |
| YOLO11-S classifier (image labels) | 0.8571 | 0.8532 |
| YOLO11-N classifier (image labels) | 0.7250 | 0.8252 |
| PatchCore (OK images only, no NOK labels) | 0.9242 | 0.9242 |

The scout leads the classifiers trained on the same labels. Against the label-free PatchCore, its margin on the few-NOK pool is below 0.01 (a tie under the paper's rule), and on the balanced pool it is lower. The paper's case for the scout is the calibrated exit score it provides, not ranking accuracy (§5.1).

With the OK pool fixed, the scout is close to chance with 50 NOK images or fewer, seed-dependent around 100, and reliable from 200 (§5.2, Table 3); the full curve is in [Data and training](./training.md#how-many-nok-labels). On four other benchmarks, each with its own test set, balanced pools reached 0.9659 (MPDD), 0.8760 (MVTec AD 2), 0.8289 (VisA) and 0.8700 (Real-IAD), while few-NOK pools fell to 0.5451 (VisA) and 0.5933 (Real-IAD) (§5.3).

## Exit gate

Calibrated on validation and applied unchanged to the test set (§6.1):

| Missed-NOK budget | OK images that exit | Exit-stage NOK recall |
|-------------------|--------------------:|----------------------:|
| 0 % | 234 of 614 (38.1 %) | 187 of 188 (0.9947) |
| 1 % or less | 259 of 614 (42.2 %) | 187 of 188 (0.9947) |

The 95 % Clopper–Pearson interval of the recall is 0.9707 to 0.9999. These rates depend on the scout and its calibration only, whatever the inspector. Relaxing the budget bought 4.1 points of exit at no measured recall cost, but at the resolution of a single image the two budgets are only shown to be indistinguishable on this test set.

## End-to-end cascade

The end-to-end figures use the authors' residual-DETR head as the inspector: a small DETR decoder (16 queries, 2 layers) over a frozen OK-only distillation substrate, trained head-only (§3.3). It is not the strongest inspector they measured, and it is not in this repository. Its own image AUROC (0.8216, 95 % CI 0.7862 to 0.8539) does not exceed that of its substrate (0.8424 map-max, 0.8465 standalone).

- **Recall** (§6.2). The cascade detects 0.8670 of the NOK images at both budgets, against 0.8723 when the inspector runs on every image: the exit costs exactly one defect, and the other 24 misses are the inspector's. On average the cascade inspects about 1 % of the pixel area (routed-area ratio 0.0098, exited images counted as zero).
- **Threshold precision** (§6.2). Rounding the inspector's calibrated threshold at the fourth decimal dropped one more defect (recall 0.8617), with no error raised.
- **Cost** (§6.3, Table 4):

| Path | Mean (ms) | p95 (ms) | Mean cost |
|------|----------:|---------:|----------:|
| Full: scout 10.34 + substrate 9.66 + head 1.23 | 21.22 | 22.20 | |
| Selective, budget 0 % | 18.03 | 22.15 | −15.1 % |
| Selective, budget 1 % or less | 17.69 | 22.13 | −16.7 % |

The exit lowers the mean cost per image; the tail latency stays at about 22.2 ms, because every detected defect goes through the full path. Measured as recall within a latency budget, the selective path scores an area under the curve of 0.0931 against 0.3538 for the full path. The benefit is throughput, not detection latency.

## Decision layer over four scorers

The same conformal rule, calibrated on the 614 validation OK scores with `alpha_accept` 0.1 and `alpha_reject` 0.005 (above the certifiable floor of 0.001626), applied to four scorers on the test set (§7, Table 5):

| Scorer | False alarms per 10,000 OK | Missed-defect rate | Review rate |
|--------|---------------------------:|-------------------:|------------:|
| RF-DETR-S (external, fully supervised) | 32.6 | 0.0053 | 8.2 % |
| Distilled student | 65.1 | 0.0798 | 13.3 % |
| Weak scout, few-NOK pool | 65.1 | 0.2074 | 13.5 % |
| PatchCore (label-free) | 32.6 | 0.2713 | 11.3 % |

RF-DETR-S reaches 0.9977 image AUROC on the test set; the distilled student, a student model distilled from it, 0.9614 (§3.3). The student comes from the companion project [Detector-to-Scout Distillation](https://github.com/rosepetal-ai/rosepetal-research-detector-to-scout-distillation); this paper did not measure its latency or size. The layer does not add discrimination: it turns the discrimination of the scorer it judges into a three-way decision with a bounded false-reject rate.

## Limitations

The paper lists them in §10:

- The scout has not been compared with a plain ResNet-18 classifier or with a classic classifier-then-detector cascade, so nothing shows that its DETR-style neck is necessary.
- Every headline number comes from one benchmark and one test set; on two of the four secondary benchmarks the few-NOK operating point did not transfer.
- The supervision-budget curve's transition near 100 NOK images depends on the seed; only the shape of the curve is claimed.
- The benefit is mean cost, measured under one protocol on one GPU; other hardware, batch sizes and resolutions are not characterised.
- The end-to-end recall is limited by the inspector used, and should be read as a lower bound for a stronger one.
- The conformal bound is marginal and requires exchangeability: it does not hold per category, and distribution shift voids it.
