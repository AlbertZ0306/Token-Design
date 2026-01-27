# Part 1：Tick → 分钟热力图（B/S 双通道）

本工程将逐笔成交 tick 聚合为每分钟热力图。每个 `(stock_id, trade_date)` 输入文件
生成一个形状为 `(239, 2, 32, 32)` 的张量，包含 B/S 双通道、价格桶、成交量桶。

## 目录结构

- `src/config.py`: 超参数常量与 `HeatmapConfig`
- `src/io.py`: 文件扫描、股票/日期解析、鲁棒读取
- `src/time_binning.py`: 固定 239 个分钟 slot 映射
- `src/paths.py`: 统一路径管理
- `src/stock_board.py`: 股票板块分类与动态 r_max
- `src/heatmap.py`: 价格/成交量分桶与热力图构造
- `src/pref.py`: Pref 映射加载与自动发现辅助
- `src/pipeline.py`: 单文件处理与批处理流水线
- `scripts/build_heatmaps.py`: CLI 入口
- `tests/`: pytest 测试用例
- `docs/`: 预留文档目录

## 数据布局与解析

新数据根目录示例（与旧结构兼容）：

```
data/raw/tick/2025年/202501/2025-01-09/601599.csv
```

旧结构示例：

```
data/raw/tick/YYYY/MM/DD/YYYY-MM-DD/<stock_id>.csv
```

解析规则（硬编码）：

- `stock_id` 为文件名去后缀（例如 `000001.csv` -> `000001`）。
- `trade_date` 优先解析路径中的 `YYYY-MM-DD`，若不存在则支持 `YYYYMMDD`，或从相邻目录组合 `YYYY/MM/DD`（含中文“年/月/日”也可）。
- 若解析失败，记录错误并跳过该文件。

输入列名（逗号或制表符分隔，支持有表头或无表头）：

```
TranID\tTime\tPrice\tVolume\tSaleOrderVolume\tBuyOrderVolume\tType\tSaleOrderID\tSaleOrderPrice\tBuyOrderID\tBuyOrderPrice
```

仅使用：`Time`、`Price`、`Volume`、`Type`。

## Pref 获取与缺失处理

Pref 必须为**前一交易日收盘价**。

- 推荐传入 `--pref_map`（CSV/Parquet/JSON）。
- 若 Pref 缺失且未提供映射，该文件会被跳过。
- 可选兜底（默认关闭）：`--allow_fallback_pref true` 使用文件首个有效成交价作为 Pref。

Pref 映射文件格式：

- CSV/Parquet：包含 `stock_id`、`trade_date` 以及 Pref 列（列名可为
  `pref`、`close`、`prev_close`、`preclose`、`pre_close`）。
- JSON：记录列表或字典。字典键支持 `"stock_id,trade_date"` 或
  `"stock_id|trade_date"` 的形式。

自动 Pref 发现（`--auto_pref true`）会扫描 `data/`，寻找文件名包含
`pref`、`close`、`daily`、`kline`、`ohlc`、`quote`、`summary`、`meta` 的文件，
找到后使用第一个候选作为 `pref_map`。

## 交易时段与 239 个 slot

Slot 固定且硬编码：

- slot 0：`09:25:00`–`09:25:59`
- slots 1..120：`09:30:00`–`11:29:59`（120 分钟）
- slots 121..237：`13:00:00`–`14:56:59`（117 分钟）
- slot 238：`15:00:00`–`15:00:59`
- 其他时间的 tick 会被丢弃并计入 `out_of_session_count`

映射规则：

```
[09:25:00, 09:26:00) -> 0
[09:30:00, 11:30:00) -> 1 + floor((t - 09:30:00)/60s)
[13:00:00, 14:57:00) -> 121 + floor((t - 13:00:00)/60s)
[15:00:00, 15:01:00) -> 238
```

## 价格与成交量分桶

价格轴（W=32）使用 Pref 相对收益 + tanh：

```
r = log(P / Pref)
r_clipped = clip(r, -r_max, r_max)
x = tanh(r_clipped / s) / tanh(r_max / s)
x_idx = floor(((x + 1) / 2) * (W - 1))
```

默认 `s = 0.02`。

成交量轴（H=32）使用 log1p 与固定上限：

```
u = log(1 + V)
u_max = log(1 + 50000)
y_idx = floor((min(u, u_max) / u_max) * (H - 1))
```

## 热力图累加与像素缩放

- 通道 0：`Type == 'B'`
- 通道 1：`Type == 'S'`
- 其他类型累计到 `unknown_type_count`，不写入热力图

像素缩放（默认开启）：

```
v = log1p(count)
pix = min(v, log1p(count_cap)) / log1p(count_cap)
```

默认 `count_cap = 128`。当 `pixel_scale=false` 时，保存原始整数计数。

## 输出格式

每个 `(stock_id, trade_date)` 输出一个文件（按年月日分层目录存储）：

```
data/processed/heatmaps/YYYY/MM/DD/<stock_id>_<trade_date>.pt
```

内容为 `torch.save` 的字典：

- `heatmaps`：`(239, 2, 32, 32)` 的 `torch.Tensor`
- `metadata`：Python `dict`，字段包括
  `stock_id`、`trade_date`、`pref`、`pixel_scale`、`count_cap`、`r_max`、`s`、
  `v_cap`、`t_slots`、`channels`、`height`、`width`、`unknown_type_count`、
  `out_of_session_count`、`total_ticks`、`used_fallback_pref`、`stock_board`

## 股票板块与动态 r_max

系统支持根据股票代码自动识别所属板块，并动态调整 `r_max` 参数：

### A 股代码规则

- **主板（60xxxx、000xxx）**：涨跌幅限制 10%
- **创业板（30xxxx）**：涨跌幅限制 20%
- **科创板（688xxx）**：涨跌幅限制 20%

### r_max 动态调整

```
r_clipped = clip(r, -r_max, r_max)

其中 r_max 根据股票代码动态调整：
- 主板（10%）：r_max = ln(1.1) ≈ 0.0953
- 创业板/科创板（20%）：r_max = ln(1.2) ≈ 0.1823
```

`r_max` 会自动写入输出的 `metadata` 中，包含 `stock_board` 字段标示所属板块。

### 板块识别逻辑

位于 `src/stock_board.py`，通过股票代码前缀匹配：
- `688` → 科创板
- `30` → 创业板
- `60`、`000` → 主板
- 其他 → 默认主板

## CLI 用法

示例（数据位于 `data/raw/tick`）：

```
python -m scripts.build_heatmaps \
  --input data/raw/tick \
  --output data/processed/heatmaps \
  --pref_map data/pref/pref_map.csv \
  --pixel_scale true \
  --workers 4
```

注意：输出为 `.pt`，需要安装 `torch` 才能保存与读取。

可选参数：

- `--allow_fallback_pref true`
- `--auto_pref true`
- `--count_cap 128`
- `--progress true/false`（是否显示进度条）
- `--s 0.02`
- `--v_cap 50000`
- `--r_max`（可选，默认根据股票代码自动调整）

## 测试

运行全部测试：

```
pytest -q
```

## 工具脚本

分析 out_of_session 的时间分布（支持导出分钟粒度 CSV）：

```
python -m scripts.analyze_out_of_session \
  --input data/raw/tick \
  --output_csv logs/out_of_session_minutes.csv \
  --top_k 10
```

## 常见问题

- `pref_missing`：请提供 `--pref_map` 或开启 `--allow_fallback_pref`。
- `unknown_type_count` > 0：输入 `Type` 不是 `B` 或 `S`。
- `out_of_session_count` > 0：tick 时间落在 239 slot 以外。
