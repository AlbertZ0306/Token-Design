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
