from __future__ import annotations

import math

from src.stock_board import get_board_type, get_r_max_for_stock, BoardType


def test_main_board_classification() -> None:
    """测试主板股票分类"""
    # 沪市主板
    assert get_board_type("600000") == BoardType.MAIN_BOARD
    assert get_board_type("601398") == BoardType.MAIN_BOARD
    assert get_board_type("603019") == BoardType.MAIN_BOARD
    assert get_board_type("605000") == BoardType.MAIN_BOARD

    # 深市主板
    assert get_board_type("000001") == BoardType.MAIN_BOARD
    assert get_board_type("001979") == BoardType.MAIN_BOARD
    assert get_board_type("002594") == BoardType.MAIN_BOARD
    assert get_board_type("003816") == BoardType.MAIN_BOARD


def test_gem_board_classification() -> None:
    """测试创业板股票分类"""
    assert get_board_type("300750") == BoardType.GEM  # 宁德时代
    assert get_board_type("301000") == BoardType.GEM


def test_star_board_classification() -> None:
    """测试科创板股票分类"""
    assert get_board_type("688012") == BoardType.STAR
    assert get_board_type("688981") == BoardType.STAR


def test_r_max_by_board() -> None:
    """测试不同板块的r_max值"""
    # 主板 10%
    assert abs(get_r_max_for_stock("600000") - math.log(1.1)) < 1e-6
    assert abs(get_r_max_for_stock("000001") - math.log(1.1)) < 1e-6

    # 创业板 20%
    assert abs(get_r_max_for_stock("300750") - math.log(1.2)) < 1e-6
    assert abs(get_r_max_for_stock("301000") - math.log(1.2)) < 1e-6

    # 科创板 20%
    assert abs(get_r_max_for_stock("688012") - math.log(1.2)) < 1e-6


def test_stock_code_normalization() -> None:
    """测试股票代码标准化处理"""
    # 测试不同格式的股票代码
    assert get_board_type("600000") == get_board_type("60000")  # 短代码补零
    assert get_board_type("000001") == get_board_type("1")  # 短代码补零
    assert get_board_type("300750") == get_board_type("30750")  # 短代码补零
