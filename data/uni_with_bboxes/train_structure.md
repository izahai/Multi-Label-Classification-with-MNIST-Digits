# `train.pt` structure with bounding boxes

`train.pt` contains **50,000** synthetic 64×64 MNIST composite images. Every sample contains 6, 7, or 8 digits; the generator balances these three values as evenly as possible.

| Key | Type and shape | Meaning |
| --- | --- | --- |
| `images` | `uint8`, `(50000, 64, 64)` | Grayscale composite images. |
| `labels` | `float32`, `(50000, 10)` | Multi-hot digit-presence target for classes 0–9. |
| `center_labels` | `int64`, `(50000,)` | Label of slot 0, the centred digit. |
| `count_labels` | `int64`, `(50000, 10)` | Number of appearances of each class. |
| `num_digits` | `int64`, `(50000,)` | Number of digit instances: 6, 7, or 8. |
| `all_digit_labels` | `int64`, `(50000, 8)` | Class label for each instance slot. |
| `all_positions` | `int64`, `(50000, 8, 2)` | Each patch's top-left `(row, column)`. |
| `all_bboxes` | `int64`, `(50000, 8, 4)` | Tight foreground boxes `(x_min, y_min, x_max, y_max)` after rotation, with exclusive maxima. |
| `bbox_labels` | `int64`, `(50000, 8)` | Class label paired with the box at the same slot in `all_bboxes`. |
| `all_rotation_degrees` | `float32`, `(50000, 8)` | Random rotation angle used for each digit. |
| `all_intensity_scales` | `float32`, `(50000, 8)` | Random bold/dim scale applied to each digit. |
| `salt_pepper_probabilities` | `float32`, `(50000,)` | Salt-and-pepper probability sampled for each full image. |

Slots from `num_digits` through slot 7 are padding and use `-1` in the label, position, and box tensors and `NaN` in the rotation/intensity tensors. Slot 0 is randomly rotated but fixed at the image centre; all other digits are randomly placed. Pairwise overlap is measured using actual rotated foreground masks as `intersection / smaller foreground area`. The centred digit's foreground is drawn last, so it overwrites every distractor pixel where they overlap. Finally, variable-intensity salt-and-pepper noise is added across the entire image.
