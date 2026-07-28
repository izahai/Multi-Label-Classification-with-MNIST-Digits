# Composite MNIST multi-label classifier

This package trains a classifier that predicts which digits from 0 through 9
are present in a 64×64 composite image. Bounding boxes and repeated instances
of the same digit are ignored.

## Models

- `small`: residual CNN with 125,902 trainable parameters.
- `large`: residual CNN with 1,485,774 trainable parameters.
- `dense_net`: DenseNet-BC-121 with 6,955,146 trainable parameters.

All models produce 10 raw logits. Training uses `BCEWithLogitsLoss`; inference
applies sigmoid and a configurable threshold (0.5 by default).

## Data behavior

Training concatenates:

- `data/train.pt`
- `data/uni_with_bboxes/train.pt`

Validation and test data are never concatenated. The pipeline reports original
and bbox-source results independently.

Only `images` and `labels` are loaded. The target must be a `(N, 10)` multi-hot
digit-presence tensor, so duplicate digit instances have no special effect.

## Train

From the repository root:

```bash
source .env/bin/activate
python -m src_model_cls.train --model-size small
```

Train the large model:

```bash
python -m src_model_cls.train --model-size large
```

Train DenseNet-121 (a smaller batch is recommended because its dense feature
maps use substantially more accelerator memory):

```bash
python -m src_model_cls.train \
  --model-size dense_net \
  --batch-size 32
```

Useful options:

```bash
python -m src_model_cls.train \
  --original-data-dir data \
  --bbox-data-dir data/uni_with_bboxes \
  --model-size small \
  --epochs 50 \
  --batch-size 256 \
  --num-workers 2
```

Outputs default to `outputs/mnist_classifier_<model-size>/`, for example
`outputs/mnist_classifier_dense_net/`:

- `best.pt` and `last.pt`
- `history.csv`
- `training_curves.png`
- `test_metrics.json`

The best checkpoint maximizes the mean exact match across the two validation
sources. A tie is resolved by lower mean validation BCE loss.

Resume a run:

```bash
python -m src_model_cls.train \
  --resume outputs/mnist_classifier_small/last.pt \
  --epochs 50
```

## Evaluate

Evaluate both sources separately on train, validation, and test:

```bash
python -m src_model_cls.evaluate \
  --checkpoint outputs/mnist_classifier_small/best.pt
```

Evaluate only validation and test:

```bash
python -m src_model_cls.evaluate \
  --checkpoint outputs/mnist_classifier_small/best.pt \
  --splits val test
```

Metrics:

- `exact_match`: fraction of samples whose full 10-bit prediction equals the
  target.
- `binary_match`: fraction of correct entries across all 10-bit predictions.

## Colab

Open `train_classifier_colab.ipynb`, edit the repository and Google Drive
paths in the configuration cell, then use **Run all**. The notebook trains one
selected model for up to 50 epochs and stores outputs in Google Drive.
