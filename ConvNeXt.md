Dưới đây là kiến trúc **ConvNeXt V2** theo implementation chính thức, được tách thành các thành phần để bạn tự viết lại bằng PyTorch từ đầu. Điểm khác biệt chính so với ConvNeXt V1 là **bỏ Layer Scale và thêm GRN — Global Response Normalization** vào phần mở rộng channel. ([arXiv][1])

---

# 1. Kiến trúc tổng thể

ConvNeXt V2 gồm:

```text
Input
  │
  ├── Stem
  │
  ├── Stage 0
  │
  ├── Downsample 1
  │
  ├── Stage 1
  │
  ├── Downsample 2
  │
  ├── Stage 2
  │
  ├── Downsample 3
  │
  ├── Stage 3
  │
  ├── Global Average Pooling
  │
  ├── LayerNorm
  │
  └── Linear classifier
```

Mỗi stage gồm nhiều `ConvNeXtV2Block`.

Với ảnh ImageNet `224×224`, stem chính thức dùng:

```python
Conv2d(
    in_channels=3,
    out_channels=dims[0],
    kernel_size=4,
    stride=4
)
```

Các downsampling tiếp theo dùng:

```python
LayerNorm
Conv2d(
    dims[i],
    dims[i + 1],
    kernel_size=2,
    stride=2
)
```

Do đó kích thước không gian thay đổi như sau:

```text
224×224
→ 56×56
→ 28×28
→ 14×14
→ 7×7
```

Với ảnh `64×64` và giữ nguyên stem:

```text
64×64
→ 16×16
→ 8×8
→ 4×4
→ 2×2
```

Cách tổ chức stem, bốn stage và ba lớp downsampling này khớp với implementation PyTorch chính thức. ([GitHub][2])

---

# 2. ConvNeXt V2 Block

Một block hoàn chỉnh có dạng:

```text
Input: N × C × H × W
        │
        ▼
Depthwise Conv 7×7
        │
        ▼
Permute NCHW → NHWC
        │
        ▼
LayerNorm
        │
        ▼
Linear C → 4C
        │
        ▼
GELU
        │
        ▼
GRN
        │
        ▼
Linear 4C → C
        │
        ▼
Permute NHWC → NCHW
        │
        ▼
DropPath
        │
        ▼
Residual addition
```

Dạng toán học:

[
y=x+\operatorname{DropPath}\left(
W_2\left(
\operatorname{GRN}\left(
\operatorname{GELU}\left(
W_1\left(
\operatorname{LN}\left(
\operatorname{DWConv}_{7\times7}(x)
\right)\right)\right)\right)\right)\right)
]

Trong đó:

* `DWConv`: depthwise convolution để trộn thông tin không gian;
* `Linear C → 4C`: mở rộng channel;
* `GRN`: tạo sự cạnh tranh và chuẩn hóa phản hồi giữa các channel;
* `Linear 4C → C`: đưa số channel về ban đầu;
* residual connection giữ ổn định gradient.

Đây là đúng thứ tự phép toán trong code chính thức. ([GitHub][2])

---

# 3. Depthwise convolution

Depthwise convolution được khai báo bằng:

```python
nn.Conv2d(
    dim,
    dim,
    kernel_size=7,
    padding=3,
    groups=dim,
)
```

Vì `groups=dim`, mỗi channel được convolution độc lập.

Input và output đều có shape:

```text
N × C × H × W
```

Kernel `7×7`, padding `3` giữ nguyên kích thước không gian.

---

# 4. Channel MLP

Sau depthwise convolution, tensor được chuyển từ:

```text
N × C × H × W
```

sang:

```text
N × H × W × C
```

Sau đó áp dụng:

```text
LayerNorm(C)
Linear(C, 4C)
GELU
GRN(4C)
Linear(4C, C)
```

Việc dùng `Linear` trên chiều cuối tương đương với convolution `1×1` về mặt biến đổi channel.

Bạn cũng có thể viết block hoàn toàn theo `NCHW` bằng `Conv2d(kernel_size=1)`, nhưng để bám sát code chính thức thì nên dùng `NHWC + Linear`. ([GitHub][2])

---

# 5. Global Response Normalization

GRN là thành phần quan trọng nhất được thêm trong ConvNeXt V2. Paper giới thiệu nó nhằm tăng sự cạnh tranh giữa các channel và hạn chế hiện tượng các feature trở nên dư thừa. ([arXiv][1])

Input của GRN có dạng:

```text
N × H × W × C
```

## Bước 1: Tính response theo không gian

Với mỗi channel:

[
G_i = \sqrt{\sum_{h,w}x_{h,w,i}^2}
]

Trong PyTorch:

```python
gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
```

Shape:

```text
N × 1 × 1 × C
```

## Bước 2: Chuẩn hóa giữa các channel

[
N_i = \frac{G_i}{\operatorname{mean}_j(G_j)+\epsilon}
]

Trong code:

```python
nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
```

## Bước 3: Calibration và residual

[
y = x+\gamma(x\odot N)+\beta
]

Trong code:

```python
y = x + gamma * (x * nx) + beta
```

`gamma` và `beta` là tham số học được có shape:

```text
1 × 1 × 1 × C
```

Cả hai được khởi tạo bằng 0. Vì vậy lúc bắt đầu huấn luyện:

[
y=x
]

GRN ban đầu hoạt động gần như identity mapping, giúp việc tối ưu ổn định.

---

# 6. Code GRN

```python
import torch
import torch.nn as nn


class GRN(nn.Module):
    """Global Response Normalization.

    Input shape: [N, H, W, C]
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()

        self.gamma = nn.Parameter(
            torch.zeros(1, 1, 1, dim)
        )
        self.beta = nn.Parameter(
            torch.zeros(1, 1, 1, dim)
        )
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global L2 response trên H và W
        gx = torch.norm(
            x,
            p=2,
            dim=(1, 2),
            keepdim=True,
        )

        # Chuẩn hóa response giữa các channel
        nx = gx / (
            gx.mean(dim=-1, keepdim=True) + self.eps
        )

        # Calibration + residual
        return x + self.gamma * (x * nx) + self.beta
```

---

# 7. LayerNorm cho NCHW

Trong block, LayerNorm được áp dụng sau khi chuyển sang `NHWC`, nên có thể dùng trực tiếp:

```python
nn.LayerNorm(dim)
```

Nhưng trong downsampling layer, tensor vẫn ở dạng:

```text
N × C × H × W
```

Bạn cần một `LayerNorm` hỗ trợ cả hai data format.

```python
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        data_format: str = "channels_last",
    ):
        super().__init__()

        if data_format not in {
            "channels_last",
            "channels_first",
        }:
            raise ValueError(
                f"Unsupported data format: {data_format}"
            )

        self.weight = nn.Parameter(
            torch.ones(normalized_shape)
        )
        self.bias = nn.Parameter(
            torch.zeros(normalized_shape)
        )
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = normalized_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(
                x,
                (self.normalized_shape,),
                self.weight,
                self.bias,
                self.eps,
            )

        # x: N, C, H, W
        mean = x.mean(dim=1, keepdim=True)
        variance = (
            x - mean
        ).pow(2).mean(dim=1, keepdim=True)

        x = (x - mean) / torch.sqrt(
            variance + self.eps
        )

        return (
            self.weight[:, None, None] * x
            + self.bias[:, None, None]
        )
```

---

# 8. ConvNeXt V2 Block bằng PyTorch

```python
class ConvNeXtV2Block(nn.Module):
    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
    ):
        super().__init__()

        self.dwconv = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=7,
            padding=3,
            groups=dim,
        )

        self.norm = nn.LayerNorm(
            dim,
            eps=1e-6,
        )

        self.pwconv1 = nn.Linear(
            dim,
            4 * dim,
        )

        self.act = nn.GELU()

        self.grn = GRN(4 * dim)

        self.pwconv2 = nn.Linear(
            4 * dim,
            dim,
        )

        self.drop_path = DropPath(
            drop_path
        ) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # N,C,H,W → N,C,H,W
        x = self.dwconv(x)

        # N,C,H,W → N,H,W,C
        x = x.permute(0, 2, 3, 1)

        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)

        # N,H,W,C → N,C,H,W
        x = x.permute(0, 3, 1, 2)

        x = residual + self.drop_path(x)
        return x
```

Lưu ý: sau `permute`, tensor có thể không contiguous. Các lớp `Linear` vẫn xử lý được, nhưng nếu bạn gặp vấn đề với một backend cụ thể, có thể dùng:

```python
x = x.permute(0, 2, 3, 1).contiguous()
```

---

# 9. DropPath

DropPath loại bỏ toàn bộ residual branch theo từng sample trong lúc huấn luyện.

```python
def drop_path(
    x: torch.Tensor,
    drop_prob: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1.0 - drop_prob

    shape = (
        x.shape[0],
        *([1] * (x.ndim - 1)),
    )

    random_tensor = keep_prob + torch.rand(
        shape,
        dtype=x.dtype,
        device=x.device,
    )

    random_tensor.floor_()

    return x * random_tensor / keep_prob


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(
            x,
            self.drop_prob,
            self.training,
        )
```

DropPath khác Dropout:

* Dropout bỏ từng phần tử;
* DropPath bỏ toàn bộ nhánh residual của một sample.

---

# 10. Stem và downsampling

## Stem chính thức

```python
stem = nn.Sequential(
    nn.Conv2d(
        in_channels,
        dims[0],
        kernel_size=4,
        stride=4,
    ),
    LayerNorm(
        dims[0],
        eps=1e-6,
        data_format="channels_first",
    ),
)
```

Lưu ý thứ tự chính thức là:

```text
Conv → LayerNorm
```

## Downsampling giữa các stage

```python
downsample = nn.Sequential(
    LayerNorm(
        dims[i],
        eps=1e-6,
        data_format="channels_first",
    ),
    nn.Conv2d(
        dims[i],
        dims[i + 1],
        kernel_size=2,
        stride=2,
    ),
)
```

Thứ tự là:

```text
LayerNorm → Conv stride 2
```

Không phải `Conv → LayerNorm`.

---

# 11. Bốn stage

Mỗi stage được xây dựng như sau:

```python
stage = nn.Sequential(
    *[
        ConvNeXtV2Block(
            dim=dims[i],
            drop_path=drop_rates[index],
        )
        for ...
    ]
)
```

Ví dụ với Femto:

```python
depths = [2, 2, 6, 2]
dims = [48, 96, 192, 384]
```

Ta có:

```text
Stage 0: 2 blocks, C=48
Stage 1: 2 blocks, C=96
Stage 2: 6 blocks, C=192
Stage 3: 2 blocks, C=384
```

---

# 12. DropPath schedule

Không nên cho mọi block cùng một DropPath probability.

Implementation thường tăng tuyến tính từ `0` đến `drop_path_rate`:

```python
drop_rates = torch.linspace(
    0,
    drop_path_rate,
    sum(depths),
).tolist()
```

Ví dụ:

```python
depths = [2, 2, 6, 2]
drop_path_rate = 0.1
```

Tổng cộng có 12 block, probability sẽ tăng dần:

```text
Block đầu: 0.000
...
Block cuối: 0.100
```

Block càng sâu thì regularization càng mạnh.

---

# 13. Classification head

Sau stage cuối:

```python
x = x.mean(dim=(-2, -1))
x = self.norm(x)
x = self.head(x)
```

Shape:

```text
N × C × H × W
→ N × C
→ N × C
→ N × num_classes
```

Không cần `AdaptiveAvgPool2d`, dù dùng nó cũng tương đương:

```python
x = F.adaptive_avg_pool2d(x, 1).flatten(1)
```

Head gồm:

```python
self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
self.head = nn.Linear(dims[-1], num_classes)
```

---

# 14. Toàn bộ skeleton ConvNeXt V2

```python
class ConvNeXtV2(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        depths=(2, 2, 6, 2),
        dims=(48, 96, 192, 384),
        drop_path_rate: float = 0.0,
    ):
        super().__init__()

        if len(depths) != 4 or len(dims) != 4:
            raise ValueError(
                "ConvNeXt V2 requires four depths and four dims."
            )

        # -------------------------------------------------
        # Downsampling layers
        # Index 0 là stem.
        # Index 1-3 là downsampling giữa các stage.
        # -------------------------------------------------
        self.downsample_layers = nn.ModuleList()

        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                dims[0],
                kernel_size=4,
                stride=4,
            ),
            LayerNorm(
                dims[0],
                eps=1e-6,
                data_format="channels_first",
            ),
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample = nn.Sequential(
                LayerNorm(
                    dims[i],
                    eps=1e-6,
                    data_format="channels_first",
                ),
                nn.Conv2d(
                    dims[i],
                    dims[i + 1],
                    kernel_size=2,
                    stride=2,
                ),
            )
            self.downsample_layers.append(downsample)

        # -------------------------------------------------
        # DropPath schedule
        # -------------------------------------------------
        drop_rates = torch.linspace(
            0,
            drop_path_rate,
            sum(depths),
        ).tolist()

        # -------------------------------------------------
        # Four stages
        # -------------------------------------------------
        self.stages = nn.ModuleList()

        block_index = 0

        for stage_index in range(4):
            blocks = []

            for _ in range(depths[stage_index]):
                blocks.append(
                    ConvNeXtV2Block(
                        dim=dims[stage_index],
                        drop_path=drop_rates[block_index],
                    )
                )
                block_index += 1

            self.stages.append(nn.Sequential(*blocks))

        # -------------------------------------------------
        # Classification head
        # -------------------------------------------------
        self.norm = nn.LayerNorm(
            dims[-1],
            eps=1e-6,
        )
        self.head = nn.Linear(
            dims[-1],
            num_classes,
        )

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(
                module.weight,
                std=0.02,
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward_features(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        # Global average pooling
        x = x.mean(dim=(-2, -1))

        return self.norm(x)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.forward_features(x)
        return self.head(x)
```

---

# 15. Các cấu hình ConvNeXt V2

Các biến thể chính thức khác nhau ở `depths` và `dims`. Repository chính thức cung cấp Atto, Femto, Pico, Nano, Tiny, Base, Large và Huge. ([GitHub][3])

Đối với yêu cầu dưới 10M tham số:

```python
CONFIGS = {
    "atto": {
        "depths": (2, 2, 6, 2),
        "dims": (40, 80, 160, 320),
    },
    "femto": {
        "depths": (2, 2, 6, 2),
        "dims": (48, 96, 192, 384),
    },
    "pico": {
        "depths": (2, 2, 6, 2),
        "dims": (64, 128, 256, 512),
    },
}
```

Factory function:

```python
def convnextv2_femto(
    num_classes: int,
    in_channels: int = 3,
    drop_path_rate: float = 0.1,
) -> ConvNeXtV2:
    return ConvNeXtV2(
        in_channels=in_channels,
        num_classes=num_classes,
        depths=(2, 2, 6, 2),
        dims=(48, 96, 192, 384),
        drop_path_rate=drop_path_rate,
    )
```

---

# 16. Điều chỉnh cho ảnh 64×64

Với ảnh 64×64, có hai lựa chọn.

## Phương án A: Giữ kiến trúc chính thức

```python
kernel_size=4
stride=4
```

Shape với Femto:

```text
Input:          N × 3   × 64 × 64
Stem:           N × 48  × 16 × 16
Stage 0:        N × 48  × 16 × 16

Downsample 1:   N × 96  × 8 × 8
Stage 1:        N × 96  × 8 × 8

Downsample 2:   N × 192 × 4 × 4
Stage 2:        N × 192 × 4 × 4

Downsample 3:   N × 384 × 2 × 2
Stage 3:        N × 384 × 2 × 2

Global pooling: N × 384
Classifier:     N × num_classes
```

Đây là lựa chọn nên thử trước vì bám sát pretrained architecture.

## Phương án B: Stem dành cho ảnh nhỏ

Đổi stem thành:

```python
stem = nn.Sequential(
    nn.Conv2d(
        in_channels,
        dims[0],
        kernel_size=3,
        stride=2,
        padding=1,
    ),
    LayerNorm(
        dims[0],
        eps=1e-6,
        data_format="channels_first",
    ),
)
```

Shape:

```text
64×64
→ 32×32
→ 16×16
→ 8×8
→ 4×4
```

Phương án này giữ chi tiết không gian tốt hơn nhưng:

* FLOPs tăng đáng kể;
* activation memory tăng;
* không tương thích trực tiếp với trọng số stem pretrained chính thức.

Với dataset kiểu CIFAR, ảnh y tế nhỏ hoặc vật thể nhỏ, stem stride 2 thường đáng thử. Nhưng nên benchmark cả hai thay vì mặc định cho rằng stride 2 luôn tốt hơn.

---

# 17. Kiểm tra shape và số tham số

```python
def count_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


model = convnextv2_femto(
    num_classes=10,
    drop_path_rate=0.1,
)

dummy = torch.randn(4, 3, 64, 64)

with torch.no_grad():
    logits = model(dummy)

print("Output:", logits.shape)
print(
    "Parameters:",
    f"{count_parameters(model) / 1e6:.2f}M",
)
```

Kết quả mong đợi:

```text
Output: torch.Size([4, 10])
Parameters: khoảng 5M
```

Số tham số chính xác thay đổi nhẹ theo `num_classes`.

---

# 18. Những điểm dễ viết sai

1. **GRN nhận NHWC**, không phải NCHW trong implementation này.

2. `GRN` đặt sau `GELU` và trước `Linear(4C, C)`:

```text
Linear → GELU → GRN → Linear
```

3. Downsampling dùng:

```text
LayerNorm → Conv2d
```

4. Block dùng duy nhất một LayerNorm, nằm sau depthwise convolution.

5. ConvNeXt V2 không cần Layer Scale như V1; GRN thực hiện calibration bằng `gamma` và `beta`.

6. `gamma`, `beta` trong GRN có dimension `4C`, vì GRN nằm sau lớp mở rộng channel.

7. Global pooling diễn ra trước LayerNorm cuối:

```text
Spatial mean → LayerNorm → Linear head
```

8. Kernel depthwise convolution là `7×7`, padding `3`, `groups=C`.

---

## Bản mình khuyên bạn triển khai trước

Cho ảnh `64×64`, dưới 10M tham số:

```python
depths = (2, 2, 6, 2)
dims = (48, 96, 192, 384)
stem_stride = 4
drop_path_rate = 0.1
```

Tức là **ConvNeXt V2 Femto nguyên bản**. Sau khi pipeline chạy đúng, bạn mới thử stem stride 2 để đánh giá xem việc giữ độ phân giải có thực sự cải thiện validation accuracy hay không.

[1]: https://arxiv.org/abs/2301.00808?utm_source=chatgpt.com "ConvNeXt V2: Co-designing and Scaling ConvNets with ..."
[2]: https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/convnextv2.py?utm_source=chatgpt.com "ConvNeXt-V2/models/convnextv2.py at main"
[3]: https://github.com/facebookresearch/ConvNeXt-V2?utm_source=chatgpt.com "Code release for ConvNeXt V2 model"
