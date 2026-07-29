# Composite MNIST multi-label classifier

This package trains a classifier that predicts which digits from 0 through 9
are present in a 64×64 composite image. Bounding boxes and repeated instances
of the same digit are ignored.

## Models

- `small`: residual CNN with 125,902 trainable parameters.
- `large`: residual CNN with 1,485,774 trainable parameters.
- `dense_net`: DenseNet-BC-121 with 6,955,146 trainable parameters.
- `densenet_atn_head`: DenseNet-BC-121 whose ten classes each learn an
  independent spatial-attention pooling map.
  The requested alias `denset_net_atn_head` is also accepted.

All models produce 10 raw logits. Training uses `BCEWithLogitsLoss`; inference
applies sigmoid and a configurable threshold (0.5 by default).

## Data behavior

Training concatenates:

- `data/train.pt`
- `data/uni_with_bboxes/train.pt`

Validation and test use only `data/val.pt` and `data/test.pt`; bbox data is
used only for training.

Only `images` and `labels` are loaded. The target must be a `(N, 10)` multi-hot
digit-presence tensor, so duplicate digit instances have no special effect.

## Train

From the repository root:

```bash
source .env/bin/activate
python -m src_model_cls.train --model-name small
```

Train the large model:

```bash
python -m src_model_cls.train --model-name large
```

Train DenseNet-121 (a smaller batch is recommended because its dense feature
maps use substantially more accelerator memory):

```bash
python -m src_model_cls.train \
  --model-name dense_net \
  --batch-size 32
```

Train the attention-head DenseNet:

```bash
python -m src_model_cls.train \
  --model-name densenet_atn_head \
  --batch-size 32
```

Useful options:

```bash
python -m src_model_cls.train \
  --original-data-dir data \
  --bbox-data-dir data/uni_with_bboxes \
  --model-name small \
  --epochs 50 \
  --batch-size 256 \
  --num-workers 2
```

Outputs default to `outputs/mnist_classifier_<model-name>/`, for example
`outputs/mnist_classifier_dense_net/`:

- `best.pt` and `last.pt` (EMA inference weights; `last.pt` also retains raw
  weights for resuming)
- `top_checkpoints/` with the best 3–5 validation checkpoints, controlled by
  `--checkpoint-average-count 3|4|5` (default: 5)
- `averaged_top_<N>.pt`, the arithmetic average of the retained EMA checkpoints
  (written once at least three checkpoints exist)
- `history.csv`
- `training_curves.png`
- `test_metrics.json`

The best checkpoint maximizes exact match on the original validation split. A
tie is resolved by lower validation BCE loss. Validation, checkpoint selection,
and final inference use EMA weights (`--ema-decay`, default `0.999`). After
training, the best checkpoint, each retained top checkpoint, and the averaged
checkpoint are all evaluated on the original test split and written to
`test_metrics.json`.

Resume a run:

```bash
python -m src_model_cls.train \
  --resume outputs/mnist_classifier_small/last.pt \
  --epochs 50
```

## Evaluate

Evaluate the original data on train, validation, and test:

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
