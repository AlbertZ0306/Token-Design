# 项目概述：Tick数据转分钟热力图

## 简介

这是一个完整的金融tick数据处理流水线，用于将逐笔成交数据转换为分钟级热力图张量，为后续的深度学习模型（如自回归编码器-解码器）提供训练数据。

每个`(stock_id, trade_date)`生成一个形状为`(239, 2, 32, 32)`的张量，包含：
- **239个时间槽**：覆盖A股主要交易时段
- **2个通道**：买入(B)和卖出(S)
- **32×32网格**：价格×成交量分桶

## 完整项目结构

```
Token/
├── data/                           # 数据目录
│   ├── raw/                        # 原始数据
│   │   ├── tick/                   # Tick数据文件
│   │   │   └── YYYY/MM/DD/YYYY-MM-DD/<stock_id>.csv
│   │   └── pref/                   # 昨收价数据
│   │       └── pref_map.csv
│   └── processed/                  # 处理后的数据
│       ├── heatmaps/               # 热力图输出
│       │   └── YYYY/MM/DD/<stock_id>_<trade_date>.pt
│       └── analysis/               # 分析结果
│           └── out_of_session_minutes.csv
│
├── logs/                           # 日志文件目录
├── configs/                        # 配置文件目录
│   └── default.yaml                # 默认配置文件
├── checkpoints/                    # 模型检查点目录
├── outputs/                        # 其他输出目录
│
├── src/                            # 源代码目录
│   ├── config.py                   # 配置常量
│   ├── paths.py                    # 统一路径管理 [新增]
│   ├── stock_board.py              # 股票板块分类 [新增]
│   ├── io.py                       # 输入输出处理
│   ├── time_binning.py             # 时间分桶逻辑 [更新]
│   ├── heatmap.py                  # 热力图构建核心
│   ├── pref.py                     # 昨收价处理
│   ├── pipeline.py                 # 数据处理流水线 [更新]
│   └── models/                     # 模型组件
│       ├── autoreg_wrapper.py
│       ├── heatmap_encoder.py
│       ├── heatmap_decoder.py
│       └── __init__.py
│
├── scripts/                        # 工具脚本
│   ├── build_heatmaps.py           # 主构建脚本
│   ├── analyze_out_of_session.py   # 分析工具
│   └── __init__.py
│
├── tests/                          # 测试套件
│   ├── conftest.py                 # 测试配置
│   ├── test_binning.py             # 分桶测试
│   ├── test_end2end_forward_loss.py
│   ├── test_end2end_small.py
│   ├── test_path_parsing.py        # 路径解析测试
│   ├── test_shapes.py              # 形状测试
│   ├── test_slot_embedding.py
│   ├── test_slot_mapping.py        # 时间槽映射测试
│   └── test_stock_board.py         # 板块分类测试 [新增]
│
├── docs/                           # 文档目录
│   ├── part2_spec.md               # Part 2规格说明
│   └── project_overview.md         # 本文档
│
├── README.md                       # 项目主文档
└── requirements.txt                # Python依赖
```

## 核心功能模块

### 1. 配置管理 (`src/config.py`)

定义热力图的关键参数：

| 参数 | 值 | 说明 |
|------|-----|------|
| `NUM_SLOTS` | 239 | 时间槽位数 |
| `NUM_BINS` | 32 | 价格/成交量分桶数 |
| `NUM_CHANNELS` | 2 | 通道数（买入/卖出） |
| `R_MAX` | None (动态) | 价格相对收益上限（None表示动态调整） |
| `S` | 0.02 | 价格分桶步长 |
| `V_CAP` | 50000 | 成交量上限 |
| `COUNT_CAP` | 128 | 像素计数缩放上限 |

**HeatmapConfig类**：

```python
@dataclass(frozen=True)
class HeatmapConfig:
    t_slots: int = 239
    channels: int = 2
    height: int = 32
    width: int = 32
    r_max: float | None = None  # None表示动态调整
    s: float = 0.02
    v_cap: int = 50000
    pixel_scale: bool = True
    count_cap: int = 128
    allow_fallback_pref: bool = False
```

### 2. 路径管理 (`src/paths.py`) [新增模块]

统一管理所有项目路径，避免硬编码：

```python
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).resolve().parents[1]

# 数据目录
DATA_DIR = ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# 原始数据子目录
TICK_DATA_DIR = RAW_DATA_DIR / "tick"
PREF_DATA_DIR = RAW_DATA_DIR / "pref"

# 处理后数据子目录
HEATMAPS_DIR = PROCESSED_DATA_DIR / "heatmaps"
ANALYSIS_DIR = PROCESSED_DATA_DIR / "analysis"

# 其他目录
LOGS_DIR = ROOT / "logs"
CONFIGS_DIR = ROOT / "configs"
CHECKPOINTS_DIR = ROOT / "checkpoints"
OUTPUTS_DIR = ROOT / "outputs"

# 默认文件路径
DEFAULT_PREF_MAP = PREF_DATA_DIR / "pref_map.csv"
DEFAULT_CONFIG = CONFIGS_DIR / "default.yaml"
```

### 3. 股票板块分类 (`src/stock_board.py`) [新增模块]

根据股票代码自动判断所属板块并调整涨跌幅限制：

**BoardType枚举**：

```python
class BoardType(Enum):
    MAIN_BOARD = "main_board"      # 主板 (10%)
    GEM = "gem"                    # 创业板 (20%)
    STAR = "star"                  # 科创板 (20%)
    UNKNOWN = "unknown"            # 未知
```

**r_max映射**：

```python
BOARD_R_MAX = {
    BoardType.MAIN_BOARD: math.log(1.1),   # 10% -> 0.0953
    BoardType.GEM: math.log(1.2),          # 20% -> 0.1823
    BoardType.STAR: math.log(1.2),         # 20% -> 0.1823
    BoardType.UNKNOWN: math.log(1.1),      # 默认10% -> 0.0953
}
```

**主要函数**：

```python
def get_board_type(stock_id: str) -> BoardType:
    """根据股票代码判断板块类型"""

def get_r_max_for_stock(stock_id: str) -> float:
    """根据股票代码获取对应的r_max值"""
```

### 4. 输入输出 (`src/io.py`)

- 自动扫描tick数据文件目录
- 解析股票ID和交易日期
- 支持多种编码格式和分隔符
- 自动检测列名
- 鲁棒读取（处理缺失值、异常值）

**支持的数据格式**：
- CSV/TSV文件
- 有表头或无表头
- 大小写不敏感的列名

### 5. 时间分桶 (`src/time_binning.py`) [已更新]

将交易时间映射到**239个**固定时间窗口：

| 时段 | 时间范围 | Slot范围 | 分钟数 |
|------|----------|----------|--------|
| 集合竞价 | 09:25:00 - 09:25:59 | 0 | 1 |
| 上午交易 | 09:30:00 - 11:29:59 | 1 - 120 | 120 |
| 下午交易 | 13:00:00 - 14:56:59 | 121 - 237 | 117 |
| 收盘竞价 | 15:00:00 - 15:00:59 | 238 | 1 |

**关键常量**：

```python
SESSION_SLOT_COUNT = 239

START_0925 = 9 * 3600 + 25 * 60  # 09:25:00
END_0926 = 9 * 3600 + 26 * 60    # 09:26:00
START_0930 = 9 * 3600 + 30 * 60  # 09:30:00
END_1130 = 11 * 3600 + 30 * 60   # 11:30:00
START_1300 = 13 * 3600           # 13:00:00
END_1457 = 14 * 3600 + 57 * 60   # 14:57:00
START_1500 = 15 * 3600           # 15:00:00
END_1501 = 15 * 3600 + 60        # 15:01:00
```

**映射规则**：

```python
def map_times_to_slots(time_series: pd.Series) -> np.ndarray:
    """将时间戳映射到槽位数组"""
    # [09:25:00, 09:25:59)   -> slot 0
    # [09:30:00, 11:29:59)   -> slots 1-120
    # [13:00:00, 14:56:59)   -> slots 121-237
    # [15:00:00, 15:00:59)   -> slot 238
```

### 6. 热力图构建 (`src/heatmap.py`)

核心处理逻辑：

1. **价格分桶**：基于前收盘价计算相对收益，映射到32个价格区间
2. **成交量分桶**：对成交量取对数，映射到32个成交量区间
3. **通道分离**：分别统计买入(B)和卖出(S)的笔数
4. **像素缩放**：将计数缩放到[0, 1]范围

**价格分桶公式**：

```python
r = log(P / Pref)
r_clipped = clip(r, -r_max, r_max)  # r_max根据板块动态调整
x = tanh(r_clipped / s) / tanh(r_max / s)
x_idx = floor(((x + 1) / 2) * (width - 1))
```

**成交量分桶公式**：

```python
u = log(1 + V)
u_max = log(1 + v_cap)
y_idx = floor((min(u, u_max) / u_max) * (height - 1))
```

### 7. 前收盘价处理 (`src/pref.py`)

- 加载前收盘价映射表
- 自动发现前收盘价文件
- 支持多种格式（CSV/Parquet/JSON）
- 回退机制（使用当日第一笔价格）

**支持的Pref列名**：`pref`, `close`, `prev_close`, `preclose`, `pre_close`

**自动发现**：扫描`data/raw/pref/`，寻找包含以下关键词的文件：
- `pref`, `close`, `daily`, `kline`, `ohlc`, `quote`, `summary`, `meta`

### 8. 数据流水线 (`src/pipeline.py`) [已更新]

- 单文件处理逻辑
- 批处理支持
- 进度条显示
- 错误处理和统计
- **动态r_max集成**：根据股票代码自动选择合适的r_max值

**处理流程**：

```python
def process_file(config: HeatmapConfig, file_path: Path, pref_map: dict) -> dict:
    """处理单个tick文件"""
    # 1. 解析股票代码和日期
    # 2. 读取tick数据
    # 3. 获取昨收价
    # 4. 获取板块类型和r_max
    # 5. 时间分桶
    # 6. 价格/成交量分桶
    # 7. 构建热力图
    # 8. 返回结果和元数据
```

## 数据处理流程

```
┌─────────────────┐
│  Tick数据文件    │
│ (逐笔成交数据)   │
└────────┬────────┘
         ▼
┌─────────────────┐
│  解析与验证      │
│ - 股票ID         │
│ - 交易日期       │
│ - 时间/价格/数量 │
│ - 买卖方向       │
└────────┬────────┘
         ▼
┌─────────────────┐
│  获取昨收价      │
│ - 从映射表读取   │
│ - 自动发现       │
│ - 兜底机制       │
└────────┬────────┘
         ▼
┌─────────────────┐
│  判断板块类型    │ [新增]
│ - 解析股票代码   │
│ - 确定r_max值    │
└────────┬────────┘
         ▼
┌─────────────────┐
│  时间分桶        │ [已更新]
│ (239个slot)     │
└────────┬────────┘
         ▼
┌─────────────────┐
│  价格/成交量分桶 │
│ (32×32网格)     │
└────────┬────────┘
         ▼
┌─────────────────┐
│  通道分离        │
│ (买入/卖出)     │
└────────┬────────┘
         ▼
┌─────────────────┐
│  像素缩放        │
│ - log1p归一化    │
└────────┬────────┘
         ▼
┌─────────────────┐
│  输出热力图      │
│ (.pt张量文件)   │
│ + 元数据         │
└─────────────────┘
```

## 股票板块与涨跌幅限制

### A股板块分类规则

A股市场按代码规则分为不同板块，各板块有不同的涨跌幅限制：

#### 代码规则表

| 市场 | 板块 | 代码规则 | 涨跌幅限制 | r_max值 | 示例代码 |
|------|------|----------|------------|---------|----------|
| **沪市** | 主板 | 600xxx, 601xxx, 603xxx, 605xxx | 10% | log(1.1) ≈ 0.0953 | 600000, 601398, 603259 |
| **沪市** | 科创板 | 688xxx | 20% | log(1.2) ≈ 0.1823 | 688012, 688981 |
| **深市** | 主板 | 000xxx, 001xxx, 002xxx, 003xxx | 10% | log(1.1) ≈ 0.0953 | 000001, 002594 |
| **深市** | 创业板 | 300xxx, 301xxx | 20% | log(1.2) ≈ 0.1823 | 300750, 301111 |

#### 特殊情况

- **ST股票**：涨跌幅限制为5%，当前版本不支持自动识别
- **新股上市**：前5日无涨跌幅限制，当前版本不支持
- **退市整理期**：10%限制

### 动态r_max实现

系统根据股票代码自动选择合适的r_max值：

```python
# 示例
get_board_type("600000")  # -> BoardType.MAIN_BOARD
get_r_max_for_stock("600000")  # -> 0.0953101798 (log(1.1))

get_board_type("688012")  # -> BoardType.STAR
get_r_max_for_stock("688012")  # -> 0.1823215568 (log(1.2))

get_board_type("300750")  # -> BoardType.GEM
get_r_max_for_stock("300750")  # -> 0.1823215568 (log(1.2))
```

### 优势

1. **自动适配**：无需手动指定，系统自动识别
2. **提高精度**：主板股票使用更窄的收益范围，提高分桶精度
3. **避免截断**：科创板/创业板使用更宽的范围，避免价格被截断

## 输出格式

每个热力图文件包含：

### 张量形状

`(239, 2, 32, 32)`
- **239**：时间槽位数
- **2**：通道数（通道0=买入，通道1=卖出）
- **32×32**：价格×成交量网格

### 元数据字段

```python
metadata = {
    # 基本信息
    "stock_id": "600000",           # 股票代码
    "trade_date": "2024-01-02",     # 交易日期

    # 昨收价相关
    "pref": 10.50,                  # 前一交易日收盘价
    "used_fallback_pref": false,    # 是否使用了兜底Pref

    # 处理参数
    "pixel_scale": true,            # 是否启用像素缩放
    "count_cap": 128,               # 像素缩放上限
    "r_max": 0.0953101798,          # 价格收益上限（根据板块自动调整）
    "s": 0.02,                      # 价格分桶步长
    "v_cap": 50000,                 # 成交量上限

    # 张量形状
    "t_slots": 239,                 # 时间槽数
    "channels": 2,                  # 通道数
    "height": 32,                   # 成交量桶数
    "width": 32,                    # 价格桶数

    # 统计信息
    "unknown_type_count": 0,        # 非B/S类型的tick数量
    "out_of_session_count": 123,    # 交易时段外的tick数量
    "total_ticks": 50000,           # 总tick数量

    # 板块信息 [新增]
    "board_type": "main_board"      # 板块类型
}
```

## 模型组件

### 编码器 (`src/models/heatmap_encoder.py`)

- 2D CNN架构
- 提取热力图时空特征
- 输出固定维度表示

### 解码器 (`src/models/heatmap_decoder.py`)

- 反卷积网络
- 重建热力图
- 支持多步预测

### 自回归包装器 (`src/models/autoreg_wrapper.py`)

- 封装编码器-解码器
- 实现自回归预测逻辑

## 使用指南

### CLI基本用法

```bash
python -m scripts.build_heatmaps \
  --input data/raw/tick \
  --output data/processed/heatmaps \
  --pref_map data/raw/pref/pref_map.csv \
  --workers 4
```

### CLI参数列表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` | `data/raw/tick` | 输入tick数据目录 |
| `--output` | `data/processed/heatmaps` | 输出热力图目录 |
| `--pref_map` | `data/raw/pref/pref_map.csv` | Pref映射文件路径 |
| `--r_max` | `None` | 价格收益上限（None=动态调整） |
| `--s` | `0.02` | 价格分桶步长 |
| `--v_cap` | `50000` | 成交量上限 |
| `--pixel_scale` | `true` | 是否启用像素缩放 |
| `--count_cap` | `128` | 像素缩放上限 |
| `--allow_fallback_pref` | `false` | 允许使用首笔价格作为Pref |
| `--auto_pref` | `false` | 自动发现Pref文件 |
| `--workers` | `1` | 并发处理进程数 |
| `--progress` | `true` | 是否显示进度条 |
| `--config` | `configs/default.yaml` | 配置文件路径 |

### 使用场景

**场景1：多进程批量处理**

```bash
python -m scripts.build_heatmaps \
  --input data/raw/tick \
  --output data/processed/heatmaps \
  --pref_map data/raw/pref/pref_map.csv \
  --workers 8 \
  --progress true
```

**场景2：使用配置文件**

```bash
python -m scripts.build_heatmaps \
  --config configs/custom.yaml
```

**场景3：分析交易时段外数据**

```bash
python -m scripts.analyze_out_of_session \
  --input data/raw/tick \
  --output_csv data/processed/analysis/out_of_session_minutes.csv \
  --top_k 10
```

## 运行测试

```bash
# 运行全部测试
pytest tests/

# 运行特定测试
pytest tests/test_binning.py
pytest tests/test_slot_mapping.py
pytest tests/test_stock_board.py

# 查看详细输出
pytest tests/ -v

# 查看测试覆盖率
pytest --cov=src tests/

# 快速测试（不显示输出）
pytest -q
```

## 依赖项

主要依赖（见 `requirements.txt`）：

```txt
# 核心依赖
torch                    # PyTorch深度学习框架
pandas                   # 数据处理
numpy                    # 数值计算

# 工具依赖
tqdm                     # 进度条显示
pytest                   # 测试框架

# 可选依赖
pyyaml                   # 配置文件解析
```

安装依赖：

```bash
pip install -r requirements.txt
```

## 与Part 2的衔接

本项目生成的热力图数据作为输入，供Part 2的模型进行：

1. **自回归预训练**：学习金融时序数据的表示
2. **特征提取**：编码器提取特征用于下游任务
3. **预测任务**：解码器生成未来热力图

### 数据流转

```
Part 1 (本项目)              Part 2
┌─────────────┐           ┌─────────────┐
│ Tick数据    │           │             │
│      ↓      │           │             │
│  热力图.pt  │─────────>│  自回归模型  │
│ (239,2,32,32)          │             │
└─────────────┘           └─────────────┘
```

### 输出数据使用示例

```python
import torch

# Part 1 生成的热力图
data = torch.load('data/processed/heatmaps/2024/01/02/600000_2024-01-02.pt')
heatmaps = data['heatmaps']  # (239, 2, 32, 32)

# Part 2 模型使用
from src.models import AutoRegWrapper

model = AutoRegWrapper(...)
output = model(heatmaps)  # 预测未来热力图
```

## 注意事项

1. **昨收价准确性**：确保前收盘价数据正确加载，否则价格分桶可能不准确
2. **多进程处理**：处理大量数据时建议使用多进程（`--workers`），注意内存使用
3. **目录自动创建**：输出目录会自动创建，无需手动创建
4. **日志记录**：处理日志会保存到`logs/`目录
5. **板块识别**：确保股票代码格式正确（6位数字），以便正确识别板块
6. **r_max选择**：推荐使用动态调整（不指定`--r_max`），仅在特殊需求时手动指定
7. **时间范围**：下午交易结束于14:56:59（不含14:57和14:58），这是设计决定

## 更新日志

### v2.0 (当前版本)

- [新增] 统一路径管理模块 (`src/paths.py`)
- [新增] 股票板块分类模块 (`src/stock_board.py`)
- [更新] 时间槽数量从240改为239
- [更新] 下午交易时间调整为13:00-14:56
- [更新] r_max支持动态调整（根据股票代码）
- [更新] 输出元数据增加`board_type`字段
- [重构] 目录结构标准化（data/, logs/, configs/, checkpoints/, outputs/）

### v1.0 (初始版本)

- 基础tick数据处理功能
- 240个时间槽
- 固定r_max值
