# Composite MNIST detector

This package trains a small CNN detector from scratch. It produces 20 unordered
prediction slots, each containing object confidence, one of ten digit classes,
and a normalized `xyxy` bounding box.

## Train

From the repository root:

```bash
source .env/bin/activate
python -m src_model.train
```

Defaults are 50 epochs, batch size 64, AdamW with learning rate `1e-3` and
weight decay `1e-4`, automatic CUDA/MPS/CPU selection, and early stopping after
10 epochs without mAP improvement.

Outputs are written to `outputs/mnist_detector/`:

- `best.pt` and `last.pt` checkpoints;
- `history.csv`;
- `training_curves.png`;
- `test_metrics.pt`.

Resume training with:

```bash
python -m src_model.train \
  --resume outputs/mnist_detector/last.pt \
  --epochs 50
```

## Inspect predictions

```bash
python -m src_model.predict \
  --checkpoint outputs/mnist_detector/best.pt \
  --split test \
  --random
```

## Metrics

- **Precision/recall:** class-correct detections at IoU ≥ 0.5.
- **Mean IoU:** mean IoU of correct detections.
- **Classification accuracy:** correct classes among localized predictions.
- **Exact match:** every ground-truth object is correctly detected and there
  are no extra retained predictions.
- **Binary/group match:** fraction of ground-truth objects correctly detected,
  averaged equally across samples.
- **Correct digits per sample:** mean number of correct detections per image.
- **mAP@0.5** and **mAP@0.5:0.95:** per-class interpolated average precision.

Confidence filtering uses objectness multiplied by maximum class probability.
NMS is applied separately per predicted digit class.
