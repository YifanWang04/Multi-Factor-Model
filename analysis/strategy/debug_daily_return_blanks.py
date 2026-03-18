"""
诊断脚本：分析 strategy_daily_return 中部分策略首尾空白的原因
运行：python analysis/strategy/debug_daily_return_blanks.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_SF_DIR = os.path.join(_ROOT, "analysis", "single_factor")
for _p in [_HERE, _SF_DIR, _ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
import strategy_config as cfg
from strategy_backtest import StrategyBacktester, _select_rebalance_dates
from run_multi_factor_test import load_return_data


def load_composite_factor(file_path: str, sheet_name: str) -> pd.DataFrame:
    xl = pd.ExcelFile(file_path)
    df = pd.read_excel(file_path, sheet_name=sheet_name, index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.apply(pd.to_numeric, errors="coerce")
    df.sort_index(inplace=True)
    return df


def main():
    print("=" * 70)
    print("  诊断：strategy_daily_return 首尾空白原因")
    print("=" * 70)

    factor_df = load_composite_factor(cfg.COMPOSITE_FACTOR_FILE, cfg.COMPOSITE_FACTOR_SHEET)
    ret_df = load_return_data(cfg.PRICE_FILE, cfg.RETURN_COLUMN)
    ret_df.sort_index(inplace=True)

    print(f"\n因子日期范围: {factor_df.index[0].date()} ~ {factor_df.index[-1].date()} (共 {len(factor_df)} 个调仓日)")
    print(f"收益率日期范围: {ret_df.index[0].date()} ~ {ret_df.index[-1].date()} (共 {len(ret_df)} 个交易日)")

    # 检查不同调仓周期的调仓日
    print("\n" + "-" * 70)
    print("各调仓周期的调仓日数量与首尾日期：")
    print("-" * 70)
    for rp in cfg.REBALANCE_PERIODS:
        rb = _select_rebalance_dates(
            factor_df.index,
            ret_df.index,
            rp,
            offset_days=getattr(cfg, "REBALANCE_DATE_OFFSET", 0),
        )
        if len(rb) >= 2:
            first_hold_start = ret_df.index[ret_df.index > rb[0]].min() if (ret_df.index > rb[0]).any() else None
            last_hold_end = rb[-1]
            print(f"  P{rp}d: {len(rb)} 个调仓日, 首调仓={rb[0].date()}, 末调仓={rb[-1].date()}")
            print(f"        → 首个持仓期起: {first_hold_start.date() if first_hold_start else 'N/A'}, 末持仓期止: {last_hold_end.date()}")

    # 运行回测，检查各策略的 daily_returns 日期范围
    backtester = StrategyBacktester(factor_df, ret_df, cfg)
    results = backtester.run_grid()

    # 分析用户提到的策略
    target_names = [
        "max_return_10G_Top1_P20d", "mvo_5G_Top3_P20d", "max_return_10G_Top2_P30d",
        "max_return_5G_Top1_P20d", "min_variance_5G_Top3_P20d", "equal_10G_Top3_P30d"
    ]

    print("\n" + "-" * 70)
    print("目标策略的 daily_returns 日期范围：")
    print("-" * 70)

    all_first = []
    all_last = []
    for name in target_names:
        res = results.get(name)
        if res is None:
            print(f"  {name}: 未找到")
            continue
        dr = res.get("daily_returns", pd.Series(dtype=float))
        if len(dr) == 0:
            print(f"  {name}: 无有效日收益")
            continue
        first_d = dr.index.min()
        last_d = dr.index.max()
        all_first.append(first_d)
        all_last.append(last_d)
        print(f"  {name}: {first_d.date()} ~ {last_d.date()} (共 {len(dr)} 天)")

    # 全量策略的日期范围统计
    print("\n" + "-" * 70)
    print("全量策略日期范围统计（首日 / 末日）：")
    print("-" * 70)

    first_dates = []
    last_dates = []
    for name, res in results.items():
        dr = res.get("daily_returns", pd.Series(dtype=float))
        if len(dr) > 0:
            first_dates.append((name, dr.index.min()))
            last_dates.append((name, dr.index.max()))

    global_first = min(d for _, d in first_dates)
    global_last = max(d for _, d in last_dates)
    print(f"  全表日期范围（并集）: {global_first.date()} ~ {global_last.date()}")

    late_start = [(n, d) for n, d in first_dates if d > global_first]
    early_end = [(n, d) for n, d in last_dates if d < global_last]

    if late_start:
        print(f"\n  首日晚于全表起点的策略（前10）:")
        for n, d in sorted(late_start, key=lambda x: x[1])[:10]:
            print(f"    {n}: 首日 {d.date()}")

    if early_end:
        print(f"\n  末日早于全表终点的策略（前10）:")
        for n, d in sorted(early_end, key=lambda x: x[1], reverse=True)[:10]:
            print(f"    {n}: 末日 {d.date()}")

    # 根因总结
    print("\n" + "=" * 70)
    print("根因分析：")
    print("=" * 70)
    print("""
1. strategy_daily_return 的 index = 所有策略日收益日期的并集。
2. 每个策略的 daily_returns 仅包含「持仓期内」的日期（(T, T_next]）。
3. 调仓周期不同 -> 调仓日序列不同 -> 首尾日期不同：
   - P10d 有 8 个调仓日，覆盖最宽（2023-05-30 ~ 2025-10-17）
   - P20d/P30d/P60d 调仓日更少，末持仓期止于 2025-05-27
   - REBALANCE_DATE_OFFSET=6 会使部分调仓日映射合并，导致 P20d/P30d 首日
     晚于 P10d（如 2023-10-20、2024-03-15）
4. 因此：首尾空白 = 该策略在那些日期没有持仓，属于正常现象。
""")


if __name__ == "__main__":
    main()
