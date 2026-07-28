# `data/uni/train.pt` structure

`train.pt` is a PyTorch-serialized dictionary containing **50,000** synthetic MNIST-style training examples. Each example is a 64×64 composite image with 6–8 overlapping, transformed digits.

| Key | Type and shape | Meaning |
| --- | --- | --- |
| `images` | `uint8`, `(50000, 64, 64)` | Grayscale composite images. |
| `labels` | `float32`, `(50000, 10)` | Multi-hot digit-presence targets for classes 0–9. |
| `center_labels` | `int64`, `(50000,)` | Class of the centred digit. |
| `count_labels` | `int64`, `(50000, 10)` | Per-class digit counts in each image. |
| `num_digits` | `int64`, `(50000,)` | Total number of digits in each image. |
| `all_digit_labels` | `int64`, `(50000, 8)` | Labels of all digit slots; there are at most 8 digits. |
| `all_positions` | `int64`, `(50000, 8, 2)` | Top-left `(row, column)` position of every 28×28 digit slot. |

The remaining keys are generation metadata: image and digit sizes, placement settings, overlap limits, augmentation/noise settings, and label semantics. The primary target is `labels`, whose declared type is `multi_hot_digit_presence`.

There is no separate bounding-box tensor. A digit box can be derived from `all_positions` and `digit_size=28` as `(row, column, row + 28, column + 28)`; it describes the original placed patch and may extend beyond visible content after overlap/rotation.
