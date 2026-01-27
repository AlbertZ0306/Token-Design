# Tick数据转分钟热力图处理系统

本工程将逐笔成交tick聚合为每分钟热力图。每个`(stock_id, trade_date)`输入文件
生成一个形状为`(239, 2, 32, 32)`的张量，包含B/S双通道、价格桶、成交量桶。

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 基本使用

```bash
python -m scripts.build_heatmaps \
  --input data/raw/tick \
  --output data/processed/heatmaps \
  --pref_map data/raw/pref/pref_map.csv \
  --workers 4
```

## 目录结构

```
Token/
├── data/                           # 数据目录
│   ├── raw/                        # 原始数据
│   │   ├── tick/                   # Tick数据文件
│   │   └── pref/                   # 昨收价数据
│   └── processed/                  # 处理后的数据
│       ├── heatmaps/               # 热力图输出
│       └── analysis/               # 分析结果
├── logs/                           # 日志文件
├── configs/                        # 配置文件
│   └── default.yaml                # 默认配置
├── checkpoints/                    # 模型检查点
├── outputs/                        # 其他输出
├── src/                            # 源代码
│   ├── config.py                   # 配置常量
│   ├── paths.py                    # 统一路径管理
│   ├── stock_board.py              # 股票板块分类
│   ├── io.py                       # 输入输出处理
│   ├── time_binning.py             # 时间分桶逻辑
│   ├── heatmap.py                  # 热力图构建
│   ├── pref.py                     # 昨收价处理
│   └── pipeline.py                 # 数据处理流水线
├── scripts/                        # 工具脚本
│   ├── build_heatmaps.py           # 主构建脚本
│   └── analyze_out_of_session.py   # 分析工具
├── tests/                          # 测试套件
└── docs/                           # 文档
```

## 数据格式要求

### 输入文件格式

Tick数据文件为CSV或TSV格式，支持有表头或无表头。

### 目录结构支持

系统自动解析多种目录结构：

- 标准结构：`data/raw/tick/YYYY/MM/DD/YYYY-MM-DD/<stock_id>.csv`
- 扁平结构：`data/raw/tick/<stock_id>_<trade_date>.csv`
- 自定义路径：自动搜索包含`YYYY-MM-DD`的路径

### 必需列说明

输入文件必须包含以下列（列名大小写不敏感）：

| 列名 | 说明 | 示例 |
|------|------|------|
| Time | 成交时间 | `09:30:01` 或 `9:30:01` |
| Price | 成交价格 | `10.52` |
| Volume | 成交量 | `100` |
| Type | 买卖方向 | `B`(买) 或 `S`(卖) |

完整的原始列（其他列会被忽略）：

```
TranID    Time    Price    Volume    SaleOrderVolume    BuyOrderVolume    Type    SaleOrderID    SaleOrderPrice    BuyOrderID    BuyOrderPrice
```

## Pref（昨收价）配置

Pref必须为**前一交易日收盘价**，用于计算相对收益率。

### 获取方式

1. **推荐：传入pref映射文件**

```bash
python -m scripts.build_heatmaps \
  --pref_map data/raw/pref/pref_map.csv
```

2. **自动发现（可选）**

```bash
python -m scripts.build_heatmaps \
  --auto_pref true
```

系统会扫描`data/raw/pref/`目录，寻找包含`pref`、`close`、`daily`、`kline`等关键词的文件。

3. **兜底机制（默认关闭）**

```bash
python -m scripts.build_heatmaps \
  --allow_fallback_pref true
```

使用文件首个有效成交价作为Pref（不推荐，可能影响分桶准确性）。

### Pref映射文件格式

支持CSV、Parquet、JSON三种格式：

**CSV/Parquet格式：**

```csv
stock_id,trade_date,pref
600000,2024-01-02,10.50
000001,2024-01-02,12.30
300750,2024-01-02,185.20
```

支持的列名：`pref`、`close`、`prev_close`、`preclose`、`pre_close`

**JSON格式：**

```json
[
  {"stock_id": "600000", "trade_date": "2024-01-02", "pref": 10.50},
  {"stock_id": "000001", "trade_date": "2024-01-02", "pref": 12.30}
]
```

或使用复合键：

```json
{
  "600000,2024-01-02": 10.50,
  "000001|2024-01-02": 12.30
}
```

## 时间槽映射

系统将交易时间映射到239个固定的时间槽（slot）：

### 时间槽分布

| 时段 | 时间范围 | Slot范围 | 分钟数 |
|------|----------|----------|--------|
| 集合竞价 | 09:25:00 - 09:25:59 | 0 | 1 |
| 上午交易 | 09:30:00 - 11:29:59 | 1 - 120 | 120 |
| 下午交易 | 13:00:00 - 14:56:59 | 121 - 237 | 117 |
| 收盘竞价 | 15:00:00 - 15:00:59 | 238 | 1 |

**注意**：
- 下午交易结束时间为`14:56:59`（不含14:57和14:58）
- 总时间槽数：**239个**（非240个）
- 落在交易时段外的tick会被丢弃并计入`out_of_session_count`

### 映射规则

```
[09:25:00, 09:25:59)   -> slot 0
[09:30:00, 11:29:59)   -> slots 1-120 (按分钟递增)
[13:00:00, 14:56:59)   -> slots 121-237 (按分钟递增)
[15:00:00, 15:00:59)   -> slot 238
其他时间               -> 丢弃
```

## 股票板块与动态r_max

系统支持根据股票代码自动调整价格涨跌幅限制。

### A股板块分类

A股代码规则及对应涨跌幅限制：

| 市场 | 板块 | 代码规则 | 涨跌幅限制 | r_max值 |
|------|------|----------|------------|---------|
| 沪市 | 主板 | 600xxx, 601xxx, 603xxx, 605xxx | 10% | log(1.1) ≈ 0.0953 |
| 沪市 | 科创板 | 688xxx | 20% | log(1.2) ≈ 0.1823 |
| 深市 | 主板 | 000xxx, 001xxx, 002xxx, 003xxx | 10% | log(1.1) ≈ 0.0953 |
| 深市 | 创业板 | 300xxx, 301xxx | 20% | log(1.2) ≈ 0.1823 |

### 动态r_max功能

**默认行为（推荐）**：

```bash
python -m scripts.build_heatmaps \
  --input data/raw/tick \
  --output data/processed/heatmaps \
  --pref_map data/raw/pref/pref_map.csv
```

当`--r_max`未指定时，系统会：
1. 从文件路径解析股票代码
2. 根据代码判断所属板块
3. 自动使用对应的r_max值

**手动指定（不推荐）**：

```bash
python -m scripts.build_heatmaps \
  --r_max 0.1823215568  # 固定为20%限制
```

强制所有股票使用指定的r_max值。

## 价格与成交量分桶

### 价格分桶（32桶）

价格轴使用Pref相对收益+tanh非线性映射：

```
r = log(P / Pref)
r_clipped = clip(r, -r_max, r_max)
x = tanh(r_clipped / s) / tanh(r_max / s)
x_idx = floor(((x + 1) / 2) * (W - 1))
```

参数说明：
- `r_max`：价格相对收益上限（动态调整或手动指定）
- `s`：价格分桶步长，默认`0.02`
- `W`：价格桶数，固定`32`

### 成交量分桶（32桶）

成交量轴使用log1p与固定上限：

```
u = log(1 + V)
u_max = log(1 + v_cap)
y_idx = floor((min(u, u_max) / u_max) * (H - 1))
```

参数说明：
- `v_cap`：成交量上限，默认`50000`
- `H`：成交量桶数，固定`32`

## 热力图输出

### 输出格式

每个`(stock_id, trade_date)`输出一个PyTorch张量文件：

```
data/processed/heatmaps/YYYY/MM/DD/<stock_id>_<trade_date>.pt
```

### 文件内容

使用`torch.save`保存的字典，包含：

| 键 | 类型 | 说明 |
|---|------|------|
| `heatmaps` | Tensor | 形状为`(239, 2, 32, 32)`的张量 |
| `metadata` | dict | 包含处理参数和统计信息的字典 |

### 元数据字段

```python
metadata = {
    "stock_id": "600000",
    "trade_date": "2024-01-02",
    "pref": 10.50,
    "pixel_scale": true,
    "count_cap": 128,
    "r_max": 0.0953101798,  # 根据板块自动调整
    "s": 0.02,
    "v_cap": 50000,
    "t_slots": 239,
    "channels": 2,
    "height": 32,
    "width": 32,
    "unknown_type_count": 0,
    "out_of_session_count": 123,
    "total_ticks": 50000,
    "used_fallback_pref": false,
    "board_type": "main_board"  # 新增：板块类型
}
```

### 通道说明

- **通道0**：`Type == 'B'`（买入）
- **通道1**：`Type == 'S'`（卖出）
- **其他类型**：累计到`unknown_type_count`，不写入热力图

### 像素缩放

默认开启，将计数缩放到`[0, 1]`范围：

```
v = log1p(count)
pix = min(v, log1p(count_cap)) / log1p(count_cap)
```

- `count_cap`：缩放上限，默认`128`
- 使用`--pixel_scale false`可保存原始整数计数

## CLI使用指南

### 基本用法

```bash
python -m scripts.build_heatmaps \
  --input data/raw/tick \
  --output data/processed/heatmaps \
  --pref_map data/raw/pref/pref_map.csv
```

### 完整参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` | `data/raw/tick` | 输入tick数据目录 |
| `--output` | `data/processed/heatmaps` | 输出热力图目录 |
| `--pref_map` | `data/raw/pref/pref_map.csv` | Pref映射文件路径 |
| `--r_max` | `None` | 价格收益上限（None表示动态调整） |
| `--s` | `0.02` | 价格分桶步长 |
| `--v_cap` | `50000` | 成交量上限 |
| `--pixel_scale` | `true` | 是否启用像素缩放 |
| `--count_cap` | `128` | 像素缩放上限 |
| `--allow_fallback_pref` | `false` | 允许使用首笔价格作为Pref |
| `--auto_pref` | `false` | 自动发现Pref文件 |
| `--workers` | `1` | 并发处理进程数 |
| `--progress` | `true` | 是否显示进度条 |
| `--config` | `configs/default.yaml` | 配置文件路径 |

### 使用示例

**多进程处理：**

```bash
python -m scripts.build_heatmaps \
  --input data/raw/tick \
  --output data/processed/heatmaps \
  --pref_map data/raw/pref/pref_map.csv \
  --workers 8
```

**使用配置文件：**

```bash
python -m scripts.build_heatmaps \
  --config configs/custom.yaml
```

**关闭像素缩放：**

```bash
python -m scripts.build_heatmaps \
  --input data/raw/tick \
  --output data/processed/heatmaps \
  --pref_map data/raw/pref/pref_map.csv \
  --pixel_scale false
```

**手动指定r_max（不推荐）：**

```bash
python -m scripts.build_heatmaps \
  --input data/raw/tick \
  --output data/processed/heatmaps \
  --pref_map data/raw/pref/pref_map.csv \
  --r_max 0.1823215568  # 强制使用20%限制
```

## 测试

运行全部测试：

```bash
pytest -q
```

运行特定测试：

```bash
pytest tests/test_slot_mapping.py -v
```

查看测试覆盖：

```bash
pytest --cov=src tests/
```

## 工具脚本

### analyze_out_of_session

分析交易时段外的tick时间分布：

```bash
python -m scripts.analyze_out_of_session \
  --input data/raw/tick \
  --output_csv data/processed/analysis/out_of_session_minutes.csv \
  --top_k 10
```

输出内容包括：
- 每分钟的tick数量分布
- 最常出现的交易外时间段
- 可用于识别数据质量问题

## 常见问题

### Q1: pref_missing错误

**问题**：`Skipping file: pref missing`

**解决方案**：
1. 提供正确的`--pref_map`文件
2. 或启用`--auto_pref true`自动发现
3. 或临时启用`--allow_fallback_pref true`（不推荐）

### Q2: r_max如何选择？

**问题**：应该手动指定r_max还是使用动态调整？

**答案**：
- **推荐**：使用动态调整（不指定`--r_max`），系统会根据股票代码自动选择合适的值
- **手动指定**：仅在需要统一处理所有股票时使用，但会导致主板股票的分桶精度降低

### Q3: unknown_type_count > 0

**问题**：有tick数据未被计入热力图

**原因**：输入的`Type`列值不是`'B'`或`'S'`

**解决方案**：检查原始数据的`Type`列，确保值为`B`或`S`

### Q4: out_of_session_count > 0

**问题**：部分tick因时间不在交易时段被丢弃

**原因**：tick时间不在239个slot的覆盖范围内

**时间段**：系统只处理以下时间
- `[09:25:00, 09:25:59)`
- `[09:30:00, 11:29:59)`
- `[13:00:00, 14:56:59)`
- `[15:00:00, 15:00:59)`

### Q5: 科创板股票的价格分桶不准确

**问题**：科创板股票涨幅可达20%，但分桶范围不够

**解决方案**：
1. 确保未指定`--r_max`参数
2. 系统会自动为科创板/创业板使用`log(1.2)`
3. 检查`metadata`中的`board_type`字段确认板块识别正确

### Q6: 如何处理涨跌幅限制变化的股票？

**问题**：某些股票可能有特殊的涨跌幅限制（如ST股票5%）

**答案**：当前版本不支持ST股票的特殊处理，如有需要请手动指定`--r_max`参数

### Q7: 输出文件读取方法

**问题**：如何读取生成的.pt文件？

**解决方案**：

```python
import torch

# 加载热力图文件
data = torch.load('data/processed/heatmaps/2024/01/02/600000_2024-01-02.pt')

# 获取张量和元数据
heatmaps = data['heatmaps']  # shape: (239, 2, 32, 32)
metadata = data['metadata']

print(f"股票代码: {metadata['stock_id']}")
print(f"交易日期: {metadata['trade_date']}")
print(f"昨收价: {metadata['pref']}")
print(f"板块类型: {metadata.get('board_type', 'N/A')}")
print(f"r_max: {metadata['r_max']}")
```

### Q8: 多进程处理注意事项

**问题**：使用多进程时可能出现的问题

**解决方案**：
1. 确保输出目录不存在或可写
2. 使用`--workers`参数控制并发数（建议不超过CPU核心数）
3. 处理大量文件时注意内存使用
