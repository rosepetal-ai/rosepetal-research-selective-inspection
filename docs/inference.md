---
title: "Inference and Deployment"
description: "Scoring images with a trained weak scout in PyTorch, exporting it to ONNX with a parity check, reproducing its preprocessing elsewhere, and running the cascade with your own inspector."
sidebar_position: 3
---

# Inference and Deployment

A trained scout scores images in PyTorch or, after export, in ONNX Runtime. The cascade class puts the scout, the exit gate, your inspector and the decision layer together.

## Preprocessing

Training, PyTorch inference and the ONNX examples all use the same preprocessing (`preprocess` in `selective_inspection/data.py`):

1. Load the image as RGB.
2. Resize it with bilinear interpolation so its longest side is 256 pixels, keeping the aspect ratio, and center it on a black 256×256 canvas (letterbox).
3. Scale the pixels to `[0, 1]` as float32, channels first: `(3, 256, 256)`.

There is **no ImageNet mean/std normalization**: the scout was trained on `[0, 1]` inputs. Any other preprocessing shifts every score and invalidates the calibrated thresholds.

## Scoring with PyTorch

```bash
# Loose files
python -m selective_inspection.infer --checkpoint runs/scout_seed1337/checkpoint.pt \
    --images part1.png part2.png

# A manifest: the labels are copied to the output, ready for calibration
python -m selective_inspection.infer --checkpoint runs/scout_seed1337/checkpoint.pt \
    --manifest data/my_dataset/val.jsonl --out predictions_val.jsonl
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--checkpoint` | required | `checkpoint.pt` of a training run |
| `--images` | | One or more image files (this or `--manifest`) |
| `--manifest` | | A JSONL manifest (this or `--images`) |
| `--device` | `auto` | `auto`, `cuda` (an error when no CUDA device is available) or `cpu` |
| `--batch-size` | `16` | Images per forward pass |
| `--out` | none | JSONL file for the scores |

The tool prints one line per image, the score and the path. The rows of `--out` have `image_id`, `image_path` and `image_score`, plus `label` when the input is a manifest. Scores are in `[0, 1]`; higher means more anomalous.

From Python:

```python
from selective_inspection.infer import load_model, score_images

model, image_size = load_model("runs/scout_seed1337/checkpoint.pt", device="cuda")
scores = score_images(model, ["part1.png", "part2.png"])   # list of floats in [0, 1]
```

`load_model` puts the model on the CPU unless you pass `device`. Loading a checkpoint never downloads backbone weights: the checkpoint holds them all.

## ONNX export

```bash
python -m selective_inspection.export_onnx \
    --checkpoint runs/scout_seed1337/checkpoint.pt \
    --out runs/scout_seed1337/scout.onnx \
    --verify-images part1.png part2.png part3.png part4.png
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--checkpoint` | required | `checkpoint.pt` of a training run |
| `--out` | required | Path of the `.onnx` file |
| `--opset` | `17` | ONNX opset |
| `--verify-images` | none | Images for the parity check |
| `--tolerance` | `1e-4` | Largest accepted difference between the PyTorch and ONNX scores |

The export runs on the CPU and checks the file with `onnx.checker`. The graph has:

- input `images`: float32 `[batch, 3, 256, 256]`, already preprocessed. The batch size is dynamic; the image size is fixed at the training resolution, so do not feed other sizes.
- output `scores`: float32 `[batch]`, the sigmoid of the image head. It is the same quantity as the PyTorch score, so the calibrated thresholds apply unchanged.

With `--verify-images`, the tool scores the images with both PyTorch and ONNX Runtime through the same preprocessing, prints a table of the two scores and their difference, and exits with code 1 if any difference reaches the tolerance. Run it after every export.

### ONNX Runtime

```python
import numpy as np
import onnxruntime as ort
from selective_inspection.data import preprocess_batch

sess = ort.InferenceSession("runs/scout_seed1337/scout.onnx",
                            providers=["CPUExecutionProvider"])
x = preprocess_batch(["part1.png", "part2.png"], image_size=256).numpy()
(scores,) = sess.run(["scores"], {"images": x.astype(np.float32)})
```

The graph does not contain the preprocessing. A consumer in another language or runtime (ONNX Runtime's C++ API, an edge runtime) must reproduce the letterbox and the `[0, 1]` scaling exactly, including the bilinear resize and the centered black padding, then bind `images` as a float32 NCHW tensor and read `scores`. Check a port by scoring a few images in both stacks and comparing, as `--verify-images` does.

The decisions are then three comparisons: the image exits when `score < tau_exit`; the decision layer accepts when `score <= tau_accept` and rejects when `score > tau_reject`. The paper's measurements are in fp32. If you quantize the model (fp16, int8), run the parity check again and recalibrate the thresholds on validation scores of the quantized model, since its scores are not the same.

## The cascade

`SelectiveInspectionPipeline` runs one image through the scout, the exit gate, your inspector and the decision layer:

```python
import json

from selective_inspection import SelectiveInspectionPipeline
from selective_inspection.infer import load_model

scout, _ = load_model("runs/scout_seed1337/checkpoint.pt", device="cuda")
gate = json.load(open("runs/scout_seed1337/gate.json"))
conformal = json.load(open("runs/my_inspector/conformal.json"))  # inspector's val scores

def my_inspector(image_rgb_uint8) -> float:
    """HWC uint8 RGB array in, anomaly score out (higher = more anomalous)."""
    ...

pipe = SelectiveInspectionPipeline(
    scout,
    tau_exit=gate["budgets"][0]["tau_exit"],   # first budget of the file
    inspector=my_inspector,
    tau_accept=conformal["tau_accept"],
    tau_reject=conformal["tau_reject"],
)
result = pipe.run("part1.png")
```

| Parameter | Meaning |
|-----------|---------|
| `scout` | The scout, as returned by `load_model` |
| `tau_exit` | Exit threshold from `calibrate_gate` |
| `inspector` | Optional. A callable that receives the **original** RGB image as an HWC uint8 NumPy array and returns a float, higher meaning more anomalous. It does its own preprocessing and may work at another resolution |
| `tau_accept`, `tau_reject` | Optional, but both or neither. Thresholds from `calibrate_conformal`, calibrated on validation scores of the model whose score they judge: the inspector when there is one, the scout otherwise |

`run` takes an image path or an HWC uint8 RGB array and returns:

| Key | Value |
|-----|-------|
| `scout_score` | The scout's score |
| `exited` | `True` when the score is below `tau_exit` |
| `stage` | `fast_ok_exit` or `inspector` |
| `inspector_score` | The inspector's score; `None` when the image exited or there is no inspector |
| `decision` | `accept` for an image that exited. For a routed image, `accept`, `review` or `reject` on the inspector's score (the scout's when there is no inspector), or `None` without decision-layer thresholds |

The cascade's measured benefit is mean cost and throughput: the scout runs on every image, and a routed image pays the scout plus the inspector.

### Command-line demo

The module also runs a manifest through the cascade. It has no inspector, so routed images are judged on their scout score, and the decision layer must then be calibrated on the scout's validation scores.

```bash
python -m selective_inspection.pipeline --checkpoint runs/scout_seed1337/checkpoint.pt \
    --gate-json runs/scout_seed1337/gate.json --budget 0.0 \
    --conformal-json runs/scout_seed1337/conformal.json \
    --manifest data/my_dataset/test.jsonl --out runs/scout_seed1337/decisions.jsonl
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--checkpoint` | required | Scout checkpoint |
| `--gate-json` | | Output of `calibrate_gate` (this or `--tau-exit`) |
| `--tau-exit` | | Exit threshold given directly (this or `--gate-json`) |
| `--budget` | first entry | Which budget of `--gate-json` to use; an error if the file does not have it |
| `--conformal-json` | none | Output of `calibrate_conformal`; without it, routed images get no decision |
| `--manifest` | required | Images to run |
| `--device` | `auto` | `auto`, `cuda` or `cpu` |
| `--out` | none | JSONL file with one row per image: the keys of `run` plus `image_id` and `label` |

It prints the thresholds in use and a summary: `n`, `exited`, `ok_exit_stage_rate` (images that exited divided by all the images of the manifest, OK and NOK alike) and the count of each decision (`no_decision_layer` for routed images without a decision layer).
