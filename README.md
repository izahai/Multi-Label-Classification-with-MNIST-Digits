# Composite MNIST Multi-Label Classification

Train convolutional neural networks to predict which MNIST digits (`0`–`9`)
are present in a 64×64 composite image. Each image can contain repeated and
overlapping digits; the target is a 10-element multi-hot presence vector, not
a digit count or a bounding-box prediction.

## Project layout

```text
gen_data_src/   Data download, generation, and visualisation scripts
src_model_cls/  Model definitions, training, evaluation, and tests
data/           PyTorch dataset splits (not tracked by Git)
outputs/        Checkpoints, metrics, and training plots (not tracked by Git)
```

## Setup

Use Python 3.10+ and install a PyTorch build suitable for your CPU, CUDA, or
Apple Silicon environment. The remaining dependencies are:

```bash
pip install torch torchvision matplotlib tqdm
```

## Dataset

The classifier expects PyTorch `.pt` dictionaries with:

- `images`: `uint8` tensor of shape `(N, 64, 64)`
- `labels`: `float32` multi-hot tensor of shape `(N, 10)`

Training combines these two splits:

```text
data/train.pt
data/gen_data/train.pt
```

Validation and test use only:

```text
data/val.pt
data/test.pt
```

Place the original composite-MNIST `train.pt`, `val.pt`, and `test.pt` in
`data/`. Then create the additional synthetic training split with:

```bash
python gen_data_src/create_multilabel_mnist_with_boxes.py \
  --mnist-dir data \
  --output-dir data/gen_data \
  --mode train
```

The generator downloads the source MNIST dataset to `data/` if needed and
creates images with 6–8 transformed, potentially overlapping digits. It also
records boxes and generation metadata, although the classification pipeline
uses only `images` and `labels`.

## Train

Run commands from the repository root.

```bash
python -m src_model_cls.train --model-name small
```

Available models:

| Model | Description |
| --- | --- |
| `small` | Compact residual CNN. |
| `large` | Larger residual CNN. |
| `dense_net` | DenseNet-BC-121. |
| `densenet_atn_head` | DenseNet-BC-121 with class-specific spatial attention. |
| `densenet_atn_head_v2` | DenseNet with block 2–4 multi-scale fusion, learnable GeM pooling, and the attention head. |

Use a smaller batch for DenseNet models:

```bash
python -m src_model_cls.train \
  --model-name densenet_atn_head_v2 \
  --batch-size 32 \
  --epochs 50
```

Useful development run:

```bash
python -m src_model_cls.train \
  --model-name small \
  --epochs 1 \
  --batch-size 32 \
  --max-train-samples 3200 \
  --max-val-samples 1000 \
  --max-test-samples 1000
```

Training uses `BCEWithLogitsLoss`, AdamW, gradient clipping, early stopping,
and exponential moving average (EMA) weights for validation and inference.
Outputs are written to `outputs/mnist_classifier_<model-name>/` and include
`best.pt`, `last.pt`, selected top checkpoints, metrics, history, and plots.

Resume a run with:

```bash
python -m src_model_cls.train \
  --resume outputs/mnist_classifier_densenet_atn_head_v2/last.pt \
  --epochs 50
```

## Evaluate

```bash
python -m src_model_cls.evaluate \
  --checkpoint outputs/mnist_classifier_densenet_atn_head_v2/best.pt
```

By default this evaluates train, validation, and test splits. Add
`--splits val test` to skip the training split. The main metrics are
`exact_match` (the full 10-label prediction must match) and `binary_match`
(fraction of correct individual labels).

## Tests

```bash
python -m unittest src_model_cls.test_models
```

## Colab

`src_model_cls/train_classifier_colab.ipynb` provides a Google Colab workflow.
Set the repository and Google Drive paths in its configuration cell, then run
all cells.
