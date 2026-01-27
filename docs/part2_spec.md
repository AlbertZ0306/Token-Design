# Part 2：Heatmap Encoder + Heatmap Decoder（PyTorch）

本模块用于消费 Part 1 的逐分钟热力图，并进行自回归预训练的编码/解码。输入输出口径与 Part 1 完全对齐。

## 与 Part 1 的对齐点（必须）

- 输入热力图 shape：`(B, T, 2, 32, 32)`
- 通道含义：`C=0` 为 B，`C=1` 为 S
- 时间 slot：`0..238`
- 值域：`[0, 1]`（与 Part 1 的 `pixel_scale=True` 对齐）

## 模块接口

### HeatmapEncoder

- 输入：
  - `x`: `(B, T, 2, 32, 32)` 或 `(B*T, 2, 32, 32)`
  - `slot_id`: `(B, T)` 或与 `(B*T,)` 等价的扁平张量，dtype 为 `torch.long`
- 输出：
  - `z`: `(B, T, 4096)` 或 `(B*T, 4096)`

### HeatmapDecoder

- 输入：
  - `z`: `(B, T, 4096)` 或 `(B*T, 4096)`
- 输出：
  - `x_hat`: `(B, T, 2, 32, 32)`，并通过 `Sigmoid` 保证在 `[0, 1]`

### AutoRegWrapper（示例/测试用）

- 流程：
  - `Z_in = Encoder(X, slot_id)`
  - `H = DummyCausal(Z_in)`（占位模块）
  - `Z_next_hat = Linear4096(H)`
  - `X_next_hat = Decoder(Z_next_hat)`
- Loss 对齐（teacher forcing）：
  - `loss = SmoothL1Loss(X_next_hat[:, 0:T-1], X[:, 1:T])`

## 编码器结构（Encoder）

### 归一化（默认开启）

对每张 `(2, 32, 32)` 图计算 **全局 mean/std**（在 `C,H,W` 上）：

```
x = (x - mean) / (std + 1e-6)
```

### CNN 主干（ResNet-lite）

```
Input:   (N, 2, 32, 32)
Stem:    Conv(2->32,k3,s1,p1) + GN(8) + GELU           -> (N, 32, 32, 32)
Down1:   Conv(32->64,k3,s2,p1) + GN(8) + GELU          -> (N, 64, 16, 16)
         ResBlock(64) x2                               -> (N, 64, 16, 16)
Down2:   Conv(64->128,k3,s2,p1) + GN(8) + GELU         -> (N, 128, 8, 8)
         ResBlock(128) x2                              -> (N, 128, 8, 8)
Down3:   Conv(128->256,k3,s2,p1) + GN(8) + GELU        -> (N, 256, 4, 4)
         ResBlock(256) x2                              -> (N, 256, 4, 4)
GAP:     GlobalAvgPool                                 -> (N, 256)
```

ResBlock(c)：

- Conv(3x3) + GN + GELU
- Conv(3x3) + GN
- Residual add + GELU

### Projector + Alpha + Slot Embedding

```
Linear(256->1024) + GELU + Dropout(0.1)
Linear(1024->4096) + LayerNorm
alpha * z (alpha 可学习，初始 0.1)
Slot Embedding (239, 4096) + LayerNorm
```

## 解码器结构（Decoder）

Upsample 采用 `nearest`：

```
Input:   (N, 4096)
Linear:  4096 -> 256*4*4  reshape -> (N, 256, 4, 4)
Up1:     x2 + Conv(256->128,k3,p1) + GN + GELU -> (N, 128, 8, 8)
Up2:     x2 + Conv(128->64,k3,p1)  + GN + GELU -> (N, 64, 16, 16)
Up3:     x2 + Conv(64->32,k3,p1)   + GN + GELU -> (N, 32, 32, 32)
Out:     Conv(32->2,k1) + Sigmoid              -> (N, 2, 32, 32)
```

## 关键超参数

- slot 数：`239`
- 隐空间维度：`4096`
- Dropout：`0.1`
- Alpha：可学习标量，初始 `0.1`
- Slot Embedding：`(239, 4096)`
- Upsample：`nearest`

## 测试

```
pytest -q
```

包含测试：

- `test_shapes.py`：Encoder/Decoder shape + 值域检查
- `test_slot_embedding.py`：slot embedding 生效检查
- `test_end2end_forward_loss.py`：前向、loss、梯度回传检查

## 常见问题

- `slot_id out of range`：确保 slot_id 在 `[0, 238]`
- `input must be (B,T,2,32,32)`：确保维度和通道正确
- `dtype`：输入建议为 float32，内部会自动转为 float
