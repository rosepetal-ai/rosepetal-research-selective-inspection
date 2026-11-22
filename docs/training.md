---
title: "Data and Training"
description: "Manifest format and splits, the synthetic smoke dataset, the paper's training recipe for the weak scout, training options and outputs, the NOK-label budget, and fine-tuning from a checkpoint."
sidebar_position: 1
---

# Data and Training

The weak scout learns from image-level OK/NOK labels only. This page covers the input files, the training command and its options, and how many defect labels the paper found you need.

## Manifests

Every tool reads JSONL manifests: one JSON object per line, one image per object.

```json
{"image_path": "images/train/part_0001.png", "label": "ok"}
{"image_path": "images/train/part_0002.png", "label": "nok", "image_id": "part_0002"}
```

| Field | Required | Meaning |
|-------|----------|---------|
| `image_path` | yes | Absolute path, or relative to the current directory, or relative to the manifest's directory, tried in that order. |
| `label` | yes | `ok` (normal) or `nok` (defective), in any letter case. Any other value stops the tool with `label must be 'ok' or 'nok'`. |
| `image_id` | no | Identifier of the image; defaults to the file name without extension. The calibration tools use it to join scores with labels. |

Other fields are allowed and ignored. Empty lines are skipped, and an empty manifest is an error.

### Splits

Build three manifests, `train.jsonl`, `val.jsonl` and `test.jsonl`, each with OK and NOK images. The validation set does double duty: its NOK scores calibrate the exit gate and its OK scores calibrate the decision layer. The test set is scored once, at the end, to report the rates the frozen thresholds give.

The splits must be disjoint at the image level, not only by file name: a renamed or re-exported copy of the same image counts as overlap. An image that has been used to evaluate or calibrate a model must never appear in a training split, and the thresholds are only valid if the model never trained on the images they were calibrated on.

### MVTec AD

The paper uses [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad). Each category folder has `train/good/`, `test/good/`, `test/<defect_type>/` and `ground_truth/<defect_type>/` (the defect masks, which are not inspection images). This builds one manifest of all the images of the 15 categories:

```python
import json
from pathlib import Path

root = Path("mvtec_anomaly_detection")           # extracted dataset root
rows = []
for img in sorted(root.glob("*/*/*/*.png")):
    category, split, kind = img.parts[-4], img.parts[-3], img.parts[-2]
    if split not in ("train", "test"):            # skip ground_truth/ (masks)
        continue
    rows.append({
        "image_id": f"{category}_{split}_{kind}_{img.stem}",
        "image_path": str(img.resolve()),
        "label": "ok" if kind == "good" else "nok",
        "category": category,                     # extra field, ignored by the tools
    })
with open("mvtec_all.jsonl", "w") as f:
    f.writelines(json.dumps(r) + "\n" for r in rows)
```

Do not use the official folders as splits: MVTec AD's `train/` holds OK images only, so a scout cannot learn from it. The paper re-carves all the images into its own pools and disjoint validation and test sets (§4.1 and §4.2); carve your three manifests from `mvtec_all.jsonl` the same way, for example at random with a fixed seed. The paper's own manifests are not in the repository.

### Synthetic dataset

[`examples/make_dummy_dataset.py`](https://github.com/rosepetal-ai/rosepetal-research-selective-inspection/blob/main/examples/make_dummy_dataset.py) writes a small synthetic dataset to try the whole chain without downloading a dataset: OK images are noisy gradients, NOK images carry a bright scratch or a dark blob.

```bash
python examples/make_dummy_dataset.py --out data/dummy
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--out` | required | Output folder |
| `--n-train` | `64` | Images in the training split |
| `--n-val` | `32` | Images in the validation split |
| `--n-test` | `16` | Images in the test split |
| `--image-size` | `256` | Side of the square images, in pixels |
| `--seed` | `0` | Random seed |

Each split is half OK and half NOK. The script writes the images to `<out>/images/<split>/` and the manifests `train.jsonl`, `val.jsonl` and `test.jsonl` to `<out>`, with image paths relative to the manifest.

## Training

```bash
python -m selective_inspection.train \
    --config configs/train_default.yaml \
    --train-manifest data/my_dataset/train.jsonl \
    --val-manifest data/my_dataset/val.jsonl \
    --device cuda \
    --out runs/scout_seed1337
```

### The paper recipe

[`configs/train_default.yaml`](https://github.com/rosepetal-ai/rosepetal-research-selective-inspection/blob/main/configs/train_default.yaml) is the configuration of the paper's weak-scout runs:

| Setting | Value |
|---------|-------|
| Backbone | ResNet-18 from `timm`, ImageNet-pretrained, three feature levels |
| Input | 256×256 letterbox, pixels scaled to `[0, 1]`, no ImageNet mean/std normalization |
| Steps | 1500 |
| Batch size | 16 |
| Optimizer | AdamW, learning rate 1e-4, weight decay 1e-4, gradient-norm clip 1.0 |
| Loss | Binary cross-entropy of the image head against the label, plus a background BCE on the per-query class logits |
| Seed | 1337 (the paper's primary seed; 2027 and 4242 replicate the supervision-budget curve) |

The configuration keys and the values of `train_default.yaml`:

| Key | Value | Meaning |
|-----|-------|---------|
| `model.backbone` | `resnet18` | `timm` model name |
| `model.pretrained` | `true` | Start from ImageNet weights (downloaded by `timm` on first use) |
| `model.image_size` | `256` | Side of the square input |
| `model.feature_dim` | `256` | Width of the transformer |
| `model.num_queries` | `100` | Learnable decoder queries |
| `model.num_heads` | `8` | Attention heads |
| `model.num_encoder_layers` | `2` | Encoder layers |
| `model.num_decoder_layers` | `2` | Decoder layers |
| `model.multi_scale` | `true` | Three backbone levels; `false` uses one |
| `training.max_steps` | `1500` | Optimizer steps (one batch each) |
| `training.batch_size` | `16` | Images per step |
| `training.learning_rate` | `1.0e-4` | AdamW learning rate |
| `training.weight_decay` | `1.0e-4` | AdamW weight decay |
| `training.grad_clip_norm` | `1.0` | Gradient-norm clip |
| `training.seed` | `1337` | Seed of the shuffling and of the initialization |
| `training.log_every` | `10` | Steps between log lines |
| `training.loss.class_weight` | `1.0` | Weight of the background BCE |
| `training.loss.image_weight` | `1.0` | Weight of the image BCE |
| `manifests.train` | `data/my_dataset/train.jsonl` | Training manifest |
| `manifests.val` | `data/my_dataset/val.jsonl` | Validation manifest |

### Options

| Option | Default | Meaning |
|--------|---------|---------|
| `--config` | required | YAML configuration |
| `--out` | required | Run folder, created if missing |
| `--train-manifest` | `manifests.train` | Training manifest |
| `--val-manifest` | `manifests.val` | Validation manifest; without one, the run writes no validation scores |
| `--max-steps` | `training.max_steps` | Number of steps |
| `--batch-size` | `training.batch_size` | Batch size |
| `--seed` | `training.seed` | Seed |
| `--device` | `auto` | `auto` takes CUDA when available and prints its choice; `cuda` stops with an error when no CUDA device is available instead of falling back to the CPU; `cpu` |
| `--init-checkpoint` | none | Start from a checkpoint (see [fine-tuning](#fine-tuning)) |
| `--no-pretrained` | off | Do not download ImageNet weights (smoke runs without network; accuracy suffers) |

### Outputs

| File | Contents |
|------|----------|
| `resolved_config.json` | The configuration after the command-line overrides, written before training starts |
| `checkpoint.pt` | Weights (`model_state`), architecture (`model_params`), `step` and `seed` |
| `predictions_val.jsonl` | One row per validation image: `image_id`, `image_path`, `image_score`, `label`. This is the input of the calibration tools |
| `metrics_val.json` | `n`, `n_ok`, `n_nok` and `image_auroc` of the validation set (`null` when it holds a single class) |

The last two files are written only when there is a validation manifest. The run prints the loss every `log_every` steps and a summary at the end. When the training manifest holds fewer than 200 NOK images it prints a warning (see [how many NOK labels](#how-many-nok-labels)).

On a CPU, training is slow and uses a lot of memory: at 256×256 the encoder attends over about 5,400 tokens. For a CPU smoke run, keep the batch size at 4 or less.

## The model

`WeakScout` (in `selective_inspection/model.py`):

- The backbone returns three feature maps (strides 4, 8 and 16 for ResNet-18), each projected to 256 channels by a 1×1 convolution, given a 2-D sine position embedding and flattened into tokens.
- A 2-layer transformer encoder runs over the tokens; a 2-layer decoder runs 100 learnable queries over the encoder output.
- Three heads: a class logit per query, a box per query (kept so checkpoints match the research code; never supervised), and the **image head**, a LayerNorm and a small MLP over the mean of the encoder output. The sigmoid of the image head is the score every later stage uses.

The loss is `class_weight` × BCE of every query's class logit against "background" + `image_weight` × BCE of the image logit against the label (1 for NOK, 0 for OK). No box or mask is ever read.

## How many NOK labels

The paper fixed the OK pool at 2,361 images and varied only the number of NOK images (§5.2, Table 3), on its shared test set of 802 images:

| NOK images in training | Image AUROC (seed 1337) | Across three seeds |
|-----------------------:|------------------------:|--------------------|
| 25 | 0.583 | Close to chance, replicated |
| 50 | 0.512 (95 % CI 0.464 to 0.560) | Close to chance, replicated |
| 100 | 0.749 | Seed-dependent: 0.569 to 0.810 |
| 200 | 0.860 | Reliable, replicated (cross-seed range at most 0.042) |
| 400 | 0.911 | Reliable, replicated |
| 727 (full pool) | 0.9339 | Reliable, replicated |

Plan for at least 200 image-level NOK labels before trusting a scout; that is why the training script warns below 200.

The balance of the pool matters as well. On MVTec AD, the few-NOK pool (2,361 OK, 727 NOK) gave 0.9339 and a balanced pool (729 OK, 727 NOK) gave 0.9110. On two harder benchmarks the few-NOK recipe collapsed (VisA 0.5451, Real-IAD 0.5933) while balanced pools reached 0.8289 and 0.8700 (§5.3, each on its own test set). The paper's advice: if you can choose the training images, balance OK and NOK (§8). If the validation AUROC is close to 0.5, check the number of NOK images and the balance before the architecture.

## Fine-tuning

To adapt a trained scout to a new production line, collect OK images and at least 200 image-level NOK labels, build the three disjoint manifests, and start from the previous checkpoint:

```bash
python -m selective_inspection.train \
    --config configs/finetune_example.yaml \
    --train-manifest data/my_line/train.jsonl \
    --val-manifest data/my_line/val.jsonl \
    --init-checkpoint checkpoints/scout_prior.pt \
    --device cuda \
    --out runs/my_line_ft
```

With `--init-checkpoint` (or `init_checkpoint:` in the configuration; the option wins), both the weights and the architecture come from the checkpoint and the `model:` block is ignored. A plain state dict, as saved by the research code, is loaded into the default architecture. [`configs/finetune_example.yaml`](https://github.com/rosepetal-ai/rosepetal-research-selective-inspection/blob/main/configs/finetune_example.yaml) keeps the paper recipe with a shorter schedule (500 steps) and a lower learning rate (5e-5). Training from ImageNet weights with `train_default.yaml` remains an option for a large dataset or one far from the previous line.

Then calibrate the new run on its own validation set ([Calibration](./calibration.md)): thresholds of an earlier model do not carry over.
