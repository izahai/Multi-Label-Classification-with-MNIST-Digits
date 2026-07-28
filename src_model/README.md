# Composite MNIST detector

This package defaults to an 8×8 YOLO-style grid detector with three prediction
slots per cell (192 total). Grid assignment replaces Hungarian matching during
training. Legacy set-prediction detectors remain available as `small`/`large`.

- `yolo_small` (default): 111,933 parameters.
- `yolo_large`: larger YOLO grid model.

Every convolution block has a residual shortcut before its activation and
downsampling step. The shortcut uses parameter-free channel padding/slicing,
so adding residual paths does not change parameter counts or checkpoint keys.

## Train

From the repository root:

```bash
source .env/bin/activate
pip install tqdm
python -m src_model.train
```

Defaults are 50 epochs, batch size 64, AdamW with learning rate `1e-3` and
weight decay `1e-4`, automatic CUDA/MPS/CPU selection, and early stopping after
10 epochs without mAP improvement. The default architecture is `small`.

Default outputs are separated by architecture:

- YOLO small: `outputs/yolo_mnist_detector_small/`
- YOLO large: `outputs/yolo_mnist_detector_large/`

Each directory contains:

- `best.pt` and `last.pt` checkpoints;
- `history.csv`;
- `training_curves.png`;
- `test_metrics.pt`.

Train the large model explicitly with:

```bash
python -m src_model.train --model-size yolo_large
```

Resume training with:

```bash
python -m src_model.train \
  --resume outputs/yolo_mnist_detector_small/last.pt \
  --epochs 50
```

Resume reads the architecture from the checkpoint. Older checkpoints without
architecture metadata are treated as large.

## Inspect predictions

```bash
python -m src_model.predict \
  --checkpoint outputs/yolo_mnist_detector_small/best.pt \
  --split test \
  --random
```

## Metrics

- **Precision/recall:** class-correct detections at IoU ≥ 0.5.
- **Mean IoU:** mean IoU of correct detections.
- **Classification accuracy:** correct classes among localized predictions.
- **Exact match:** the predicted 10-class digit-presence vector exactly equals
  the ground-truth vector; boxes and duplicate digit instances are ignored.
- **Binary/group match:** fraction of ground-truth objects correctly detected,
  averaged equally across samples.
- **Correct digits per sample:** mean number of correct detections per image.
- **mAP@0.5** and **mAP@0.5:0.95:** per-class interpolated average precision.

Confidence filtering uses objectness multiplied by maximum class probability.
NMS is applied separately per predicted digit class.
