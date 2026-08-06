# Multi-Label Classification with MNIST Digits

This project builds a CNN model that identifies all digits appearing in a
`64 × 64` composite MNIST image. Each image contains 6–8 digits, and its target
is a 10-element multi-label vector representing digits 0–9.

## 1. Setup

Run the following commands from the project root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install torch torchvision matplotlib tqdm
```

## 2. Generate sample data

The following command generates 12 sample images in `data/demo_generated`
without overwriting the main training data:

```bash
python gen_data_src/create_multilabel_mnist_with_boxes.py \
  --mnist-dir data \
  --output-dir data/demo_generated \
  --mode train \
  --train-size 12 \
  --seed 42
```

The original MNIST dataset is downloaded automatically if it is not already
available. Each image is produced by combining 6–8 rotated digits with varying
intensities and added noise. The result is saved as
`data/demo_generated/train.pt`.

## 3. Visualize the generated data

```bash
python gen_data_src/visualize_multilabel_mnist_boxes.py \
  --data-dir data/demo_generated \
  --split train
```

In the visualization window, use `←`/`→` to navigate between images, `R` to
open a random sample, and `Q` to quit. Each colored rectangle is a bounding
box, and the number displayed above it is the corresponding digit label.

## 4. Train and evaluate on Google Colab

Open [`src_model_cls/notebook_pytorch.ipynb`](src_model_cls/notebook_pytorch.ipynb)
or [launch it directly in Google Colab](https://colab.research.google.com/github/izahai/Multi-Label-Classification-with-MNIST-Digits/blob/main/src_model_cls/notebook_pytorch.ipynb).
Then select:

**Runtime → Change runtime type → Hardware accelerator: GPU → GPU type: A100**

Finally, select **Runtime → Run all**. The notebook will automatically:

1. clone the source code from GitHub;
2. download the required datasets from Google Drive;
3. train the model;
4. plot the learning curves and evaluate the model on the validation and test
   sets.

> **Note:** An A100 GPU may require a paid Colab plan and is subject to
> availability. If A100 is unavailable, the notebook can still run on another
> GPU, but training will take longer.
