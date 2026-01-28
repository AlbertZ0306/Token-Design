# Part 3：热力图可视化工具

本模块用于可视化 Part 1 生成的 `.pt` 格式热力图文件。提供 5 种可视化模式，支持交互显示或批量导出图像。

## 安装依赖

```bash
pip install matplotlib torch numpy
```

或使用 conda 环境：

```bash
conda run -n vllm pip install matplotlib torch numpy
```

## 热力图数据结构

输入文件格式为 `torch.save` 保存的字典，包含：

- **`heatmaps`**: `(239, 2, 32, 32)` 形状的张量
  - `239`: 时间槽位数（交易时段）
  - `2`: 通道数（通道 0 = 买入 B，通道 1 = 卖出 S）
  - `32×32`: 价格×成交量空间网格
- **`metadata`**: 元数据字典，包含：
  - `stock_id`: 股票代码
  - `trade_date`: 交易日期
  - `pref`: 前收盘价
  - `r_max`: 价格对数收益率截断值
  - `stock_board`: 股票板块（主板/创业板/科创板）
  - 其他处理参数...

## 可视化模式

### 1. Grid（网格视图）

显示所有时间槽的网格布局，每个子图展示一个时间槽的买入/卖出热力图。

**适用场景**：快速浏览全天交易模式

```bash
python -m scripts.visualize_heatmaps \
  --input data/processed/heatmaps/2021/01/12/000088_2021-01-12.pt \
  --mode grid \
  --output outputs/visualizations
```

### 2. Single（单槽位视图）

显示单个时间槽的详细视图，便于观察特定时刻的买卖分布。

**适用场景**：分析特定时间点的交易活动

```bash
python -m scripts.visualize_heatmaps \
  --input data/processed/heatmaps/2021/01/12/000088_2021-01-12.pt \
  --mode single \
  --slot 120 \
  --output outputs/visualizations
```

### 3. Animation（时间序列动画）

创建动画展示热力图随时间的演化过程。

**适用场景**：观察交易活动的动态变化

```bash
python -m scripts.visualize_heatmaps \
  --input data/processed/heatmaps/2021/01/12/000088_2021-01-12.pt \
  --mode animation \
  --interval 50 \
  --output outputs/visualizations
```

### 4. Difference（买卖差额）

显示买入与卖出的差额热力图（买入 - 卖出），正值区域偏红表示买入占优，负值区域偏蓝表示卖出占优。

**适用场景**：分析买卖力量对比

```bash
python -m scripts.visualize_heatmaps \
  --input data/processed/heatmaps/2021/01/12/000088_2021-01-12.pt \
  --mode difference \
  --colormap RdBu_r \
  --output outputs/visualizations
```

### 5. Sum（总交易活动）

显示买入与卖出的总和热力图（买入 + 卖出），展示整体交易活跃度。

**适用场景**：识别交易热点区域

```bash
python -m scripts.visualize_heatmaps \
  --input data/processed/heatmaps/2021/01/12/000088_2021-01-12.pt \
  --mode sum \
  --colormap viridis \
  --output outputs/visualizations
```

## CLI 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-i, --input` | 必填 | 输入文件路径（支持通配符批量处理） |
| `-m, --mode` | `grid` | 可视化模式：`grid`/`single`/`animation`/`difference`/`sum` |
| `-o, --output` | `None` | 输出目录（不指定则交互显示） |
| `--slot` | `0` | 时间槽索引（用于 `single` 模式） |
| `--channel` | `both` | 通道选择：`0`=买、`1`=卖、`both`=双通道 |
| `--colormap` | `viridis` | Matplotlib 颜色映射名称 |
| `--figsize` | `12 8` | 图像尺寸（宽 高） |
| `--dpi` | `100` | 输出分辨率（DPI） |
| `--interval` | `50` | 动画间隔（毫秒） |
| `--max-slots` | `None` | 网格模式最大槽位数（不限制则显示全部） |

## 使用示例

### 基础用法

```bash
# 使用 conda 环境运行
conda run -n vllm python -m scripts.visualize_heatmaps \
  --input data/processed/heatmaps/2021/01/12/000088_2021-01-12.pt \
  --mode grid
```

### 仅显示买入通道

```bash
python -m scripts.visualize_heatmaps \
  --input data/processed/heatmaps/2021/01/12/000088_2021-01-12.pt \
  --mode grid \
  --channel 0
```

### 指定时间槽（尾盘）

```bash
python -m scripts.visualize_heatmaps \
  --input data/processed/heatmaps/2021/01/12/000088_2021-01-12.pt \
  --mode single \
  --slot 230
```

### 批量处理多个文件

```bash
# 使用通配符批量处理
python -m scripts.visualize_heatmaps \
  --input "data/processed/heatmaps/2021/01/12/*.pt" \
  --mode grid \
  --output outputs/visualizations
```

### 高分辨率输出

```bash
python -m scripts.visualize_heatmaps \
  --input data/processed/heatmaps/2021/01/12/000088_2021-01-12.pt \
  --mode grid \
  --output outputs/visualizations \
  --dpi 300 \
  --figsize 20 30
```

### 慢速动画

```bash
python -m scripts.visualize_heatmaps \
  --input data/processed/heatmaps/2021/01/12/000088_2021-01-12.pt \
  --mode animation \
  --interval 100 \
  --output outputs/visualizations
```

### 限制网格显示槽数

```bash
# 仅显示前 50 个槽位
python -m scripts.visualize_heatmaps \
  --input data/processed/heatmaps/2021/01/12/000088_2021-01-12.pt \
  --mode grid \
  --max-slots 50 \
  --output outputs/visualizations
```

## 输出格式

- **静态图像**（grid/single/difference/sum）：PNG 格式
- **动画**（animation）：GIF 格式
- **文件命名**：`{stock_id}_{trade_date}_{mode}.png` 或 `.gif`

## 支持的颜色映射

常用 matplotlib colormap：

- `viridis`: 默认，蓝绿色渐变
- `plasma`: 紫黄色渐变
- `inferno`: 黑黄红渐变
- `magma`: 黑紫红渐变
- `cividis`: 色盲友好
- `RdBu_r`: 红蓝反转（适用于 difference 模式）
- `coolwarm`: 冷暖色对比
- `hot`: 黑红黄白（类似热成像）
- `jet`: 彩虹色

## 元数据输出

运行时会在控制台输出热力图元数据：

```
Loaded heatmap from: data/processed/heatmaps/2021/01/12/000088_2021-01-12.pt
  Shape: (239, 2, 32, 32) (t_slots=239, channels=2, size=32x32)

==================================================
Heatmap Metadata
==================================================
  pref: 5.23
  r_max: 0.0953
  stock_board: main
  stock_id: 000088
  trade_date: 2021-01-12
  ...
==================================================
```

## 常见问题

- **`torch is required`**: 请安装 PyTorch (`pip install torch`)
- **`Slot X out of range`**: 槽位索引必须在 `[0, 238]` 范围内
- **`Invalid channel`**: 通道参数必须是 `0`、`1` 或 `both`
- **`No files found matching pattern`**: 检查输入路径和通配符是否正确
- 动画保存失败需要安装 Pillow: `pip install pillow`
