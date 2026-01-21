# 项目概述：Tick数据转分钟热力图

## 简介

这是一个完整的金融tick数据处理流水线，用于将逐笔成交数据转换为分钟级热力图张量，为后续的深度学习模型（如自回归编码器-解码器）提供训练数据。

## 项目结构

```
part1/
├── src/                      # 源代码目录
│   ├── config.py            # 配置常量
│   ├── io.py                # 输入输出处理
│   ├── heatmap.py           # 热力图构建核心
│   ├── pipeline.py          # 数据处理流水线
│   ├── pref.py              # 前收盘价处理
│   ├── time_binning.py      # 时间分桶逻辑
│   └── models/              # 模型组件
│       ├── autoreg_wrapper.py
│       ├── heatmap_encoder.py
│       ├── heatmap_decoder.py
│       └── __init__.py
├── scripts/                 # 工具脚本
│   ├── build_heatmaps.py    # 主构建脚本
│   ├── analyze_out_of_session.py
│   └── __init__.py
├── tests/                   # 测试套件
│   ├── conftest.py
│   ├── test_binning.py
│   ├── test_end2end_forward_loss.py
│   ├── test_end2end_small.py
│   ├── test_path_parsing.py
│   ├── test_shapes.py
│   ├── test_slot_embedding.py
│   └── test_slot_mapping.py
├── docs/                    # 文档目录
│   ├── part2_spec.md       # Part 2规格说明
│   └── project_overview.md # 本文档
├── README.md               # 项目主文档
└── requirements.txt        # Python依赖
```

## 核心功能模块

### 1. 配置管理 (`src/config.py`)

定义热力图的关键参数：

| 参数 | 值 | 说明 |
|------|-----|------|
| `NUM_SLOTS` | 240 | 时间槽位数 |
| `NUM_BINS` | 32 | 价格/成交量分桶数 |
| `NUM_CHANNELS` | 2 | 通道数（买入/卖出） |
| `R_MAX` | ln(1.2) | 价格相对收益上限 |
| `S` | 0.02 | 价格分桶步长 |
| `V_CAP` | 50000 | 成交量上限 |
| `COUNT_CAP` | 128 | 像素计数缩放上限 |

### 2. 输入输出 (`src/io.py`)

- 自动扫描tick数据文件目录
- 解析股票ID和交易日期
- 支持多种编码格式和分隔符
- 自动检测列名

### 3. 时间分桶 (`src/time_binning.py`)

将交易时间映射到240个固定时间窗口：

| 时段 | 时间范围 | Slot范围 |
|------|----------|----------|
| 集合竞价 | 9:25-9:26 | 0-1 |
| 上午交易 | 9:30-11:30 | 2-121 |
| 下午交易 | 13:00-14:58 | 122-238 |
| 收盘竞价 | 15:00-15:01 | 239 |

### 4. 热力图构建 (`src/heatmap.py`)

核心处理逻辑：

1. **价格分桶**：基于前收盘价计算相对收益，映射到32个价格区间
2. **成交量分桶**：对成交量取对数，映射到32个成交量区间
3. **通道分离**：分别统计买入(B)和卖出(S)的笔数
4. **像素缩放**：将计数缩放到[0, 1]范围

### 5. 前收盘价处理 (`src/pref.py`)

- 加载前收盘价映射表
- 自动发现前收盘价文件
- 支持多种格式（CSV/Parquet/JSON）
- 回退机制（使用当日第一笔价格）

### 6. 数据流水线 (`src/pipeline.py`)

- 单文件处理逻辑
- 批处理支持
- 进度条显示
- 错误处理和统计

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
│  时间分桶        │
│ (240个slot)     │
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
│ - Min-max归一化  │
└────────┬────────┘
         ▼
┌─────────────────┐
│  输出热力图      │
│ (.pt张量文件)   │
└─────────────────┘
```

## 输出格式

每个热力图文件包含：

- **张量形状**: `(240, 2, 32, 32)`
  - 240: 时间槽位数
  - 2: 通道数（通道0=买入，通道1=卖出）
  - 32×32: 价格×成交量网格

- **元数据**:
  - 股票ID
  - 交易日期
  - 前收盘价
  - 统计信息（总笔数、买入/卖出比例等）

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

## 工具脚本

### `build_heatmaps.py`

主构建脚本，用法示例：

```bash
python scripts/build_heatmaps.py \
    --input_dir /path/to/tick/data \
    --output_dir /path/to/output \
    --pref_dir /path/to/prev_close \
    --num_workers 4
```

### `analyze_out_of_session.py`

分析交易时段外的tick数据分布：

```bash
python scripts/analyze_out_of_session.py \
    --input_dir /path/to/tick/data \
    --output_dir /path/to/output
```

## 运行测试

```bash
# 运行全部测试
pytest tests/

# 运行特定测试
pytest tests/test_binning.py

# 查看详细输出
pytest tests/ -v
```

## 依赖项

主要依赖（见 `requirements.txt`）：

- `torch`: PyTorch深度学习框架
- `pandas`: 数据处理
- `numpy`: 数值计算
- `tqdm`: 进度条显示
- `pytest`: 测试框架

## 与Part 2的衔接

本项目生成的热力图数据作为输入，供Part 2的模型进行：

1. **自回归预训练**：学习金融时序数据的表示
2. **特征提取**：编码器提取特征用于下游任务
3. **预测任务**：解码器生成未来热力图

## 注意事项

1. 确保前收盘价数据正确加载，否则价格分桶可能不准确
2. 处理大量数据时建议使用多进程（`--num_workers`）
3. 输出目录会自动创建，无需手动创建
4. 处理日志会保存到输出目录的日志文件中
