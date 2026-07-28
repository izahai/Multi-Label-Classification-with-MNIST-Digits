# `train.pt` structure with bounding boxes

`train.pt` contains **50,000** synthetic 64×64 MNIST composite images. Every sample contains 6, 7, or 8 digits; the generator balances these three values as evenly as possible.

| Key | Type and shape | Meaning |
| --- | --- | --- |
| `images` | `uint8`, `(50000, 64, 64)` | Grayscale composite images. |
| `labels` | `float32`, `(50000, 10)` | Multi-hot digit-presence target for classes 0–9. |
| `center_labels` | `int64`, `(50000,)` | Legacy compatibility key containing the label in slot 0. |
| `count_labels` | `int64`, `(50000, 10)` | Number of appearances of each class. |
| `num_digits` | `int64`, `(50000,)` | Number of digit instances: 6, 7, or 8. |
| `all_digit_labels` | `int64`, `(50000, 8)` | Class label for each instance slot. |
| `all_positions` | `int64`, `(50000, 8, 2)` | Each patch's top-left `(row, column)`. |
| `all_bboxes` | `int64`, `(50000, 8, 4)` | Tight foreground boxes `(x_min, y_min, x_max, y_max)` after rotation, with exclusive maxima. |
| `bbox_labels` | `int64`, `(50000, 8)` | Class label paired with the box at the same slot in `all_bboxes`. |
| `all_rotation_degrees` | `float32`, `(50000, 8)` | Random rotation angle used for each digit. |

Slots from `num_digits` through slot 7 are padding and use `-1` in the label, position, and box tensors and `NaN` in `all_rotation_degrees`. Every digit is randomly placed. Pairwise overlap is measured using actual rotated foreground masks as `intersection / smaller foreground area`. Later digit foreground pixels overwrite earlier digit pixels.
