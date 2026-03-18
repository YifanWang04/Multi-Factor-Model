"""
诊断 DATA_START_OFFSET_DAYS 对回测收益的影响
============================================
运行：python analysis/strategy/debug_offset_impact.py

说明：REBALANCE_DATE_OFFSET 已移除，改为数据起始日偏移（DATA_START_OFFSET_DAYS）。
要对比 offset=0 与 offset=7 的收益差异，需分别执行：
  1. DATA_START_OFFSET_DAYS=0 时：pull → build_factors → data_process → run_composite_factor → 回测
  2. DATA_START_OFFSET_DAYS=7 时：同上（输出至 *_offset7d.xlsx，不覆盖原数据）

本脚本展示当前配置下的调仓日序列与因子对齐情况。
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SF_DIR = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "analysis", "single_factor")
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in [_HERE, _SF_DIR, _ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import strategy_config as cfg
from strategy_backtest import _select_rebalance_dates
from run_multi_factor_test import load_return_data
from run_strategy import load_composite_factor
from data.data_config import DATA_START_OFFSET_DAYS


def main():
    ret_df = load_return_data(cfg.PRICE_FILE, cfg.RETURN_COLUMN)
    factor_df = load_composite_factor(cfg.COMPOSITE_FACTOR_FILE, cfg.COMPOSITE_FACTOR_SHEET)

    factor_index = factor_df.index
    ret_index = ret_df.index
    period = 10

    rb = _select_rebalance_dates(factor_index, ret_index, period)

    print("=" * 60)
    print(f"DATA_START_OFFSET_DAYS = {DATA_START_OFFSET_DAYS}")
    print(f"调仓日数量: {len(rb)}")
    print("=" * 60)
    print("前 5 个调仓日（rb_date vs signal_date）：")
    for i, d in enumerate(rb[:5]):
        sig = d if d in factor_df.index else factor_df.index[factor_df.index <= d][-1]
        match = "[OK]" if d == sig else f"-> fallback {sig.date()}"
        print(f"  rb_date={d.date()}  signal_date={sig.date()}  {match}")
    print()
    print("调仓日历由数据起始日控制，因子与调仓日应对齐。")
    print("=" * 60)


if __name__ == "__main__":
    main()
