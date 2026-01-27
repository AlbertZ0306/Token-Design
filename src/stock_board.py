"""A股板块分类和涨跌幅限制配置"""
import math
from enum import Enum


class BoardType(Enum):
    """A股板块类型"""
    MAIN_BOARD = "main_board"      # 主板 (10%)
    GEM = "gem"                    # 创业板 (20%)
    STAR = "star"                  # 科创板 (20%)
    UNKNOWN = "unknown"            # 未知


# 板块对应的r_max值
BOARD_R_MAX = {
    BoardType.MAIN_BOARD: math.log(1.1),  # 10%
    BoardType.GEM: math.log(1.2),          # 20%
    BoardType.STAR: math.log(1.2),         # 20%
    BoardType.UNKNOWN: math.log(1.1),      # 默认10%
}


def get_board_type(stock_id: str) -> BoardType:
    """
    根据股票代码判断板块类型

    Args:
        stock_id: 股票代码 (字符串，如 "600000", "000001", "300750", "688012")

    Returns:
        BoardType: 板块类型枚举
    """
    # 标准化股票代码（转为字符串，补齐6位）
    code = str(stock_id).strip().zfill(6)

    # 6开头 - 上交所
    if code.startswith("6"):
        if code.startswith("688"):
            return BoardType.STAR  # 科创板 (20%)
        else:
            # 600, 601, 603, 605 都是沪市主板 (10%)
            return BoardType.MAIN_BOARD

    # 0开头或3开头 - 深交所
    if code.startswith("0") or code.startswith("3"):
        if code.startswith("300") or code.startswith("301"):
            return BoardType.GEM  # 创业板 (20%)
        else:
            # 000, 001, 002, 003 都是深市主板 (10%)
            return BoardType.MAIN_BOARD

    return BoardType.UNKNOWN


def get_r_max_for_stock(stock_id: str) -> float:
    """
    根据股票代码获取对应的r_max值

    Args:
        stock_id: 股票代码

    Returns:
        float: 对应的r_max值 (log(涨跌限制))
    """
    board_type = get_board_type(stock_id)
    return BOARD_R_MAX[board_type]
