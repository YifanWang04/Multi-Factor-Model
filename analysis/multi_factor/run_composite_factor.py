"""
因子复合+回测主流程 (run_composite_factor.py)
=============================================
输出1: composite_factors.xlsx  — 每个sheet为一个复合因子(行=日期, 列=股票)
输出2: composite_backtest_report.xlsx — 4个sheet的集中回测报表
"""
import os
import sys
import numpy as np
import pandas as pd
from types import SimpleNamespace

_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_DIR))
for p in [_DIR, os.path.join(_ROOT, "analysis", "single_factor"), _ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from composite_config import (
    PRICE_FILE, RETURN_COLUMN, OUTPUT_DIR, REBALANCE_PERIOD,
    N_WINDOWS, M_WINDOWS, GROUP_NUM, WEIGHT_METHOD, RISK_FREE_RATE,
    TRANSACTION_COST, get_selected_factor_files, get_factor_display_name,
)
from composite_factor import compute_all_composites, compute_selected_composites
from rebalance_manager import RebalancePeriodManager
from run_multi_factor_test import (
    load_return_data, load_factor,
    run_one_factor_one_period,
    build_sheet1_df, build_sheet2_df, build_long_excess_df, build_long_cumret_df,
    write_excel_with_format, filter_factor_ret_by_lookback,
)

try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False


# ---------------------------------------------------------------------------
# 加载选定因子，对齐到调仓期截面
# ---------------------------------------------------------------------------

def load_selected_factors(factor_files):
    """返回 {display_name: DataFrame(date×stock)}（日频原始数据）。"""
    factor_dict = {}
    for path in factor_files:
        name = get_factor_display_name(path)
        df = load_factor(path)
        factor_dict[name] = df
    return factor_dict


def align_to_rebalance_periods(factor_dict, ret, rebalance_period):
    """
    用第一个因子的日期生成调仓期，对所有因子和收益率对齐。
    返回 (factor_periods_dict, ret_periods)
      factor_periods_dict: {name: DataFrame(rebalance_date×stock)}
      ret_periods: DataFrame(rebalance_date×stock)
    """
    # 用第一个因子做调仓日生成
    first_factor = next(iter(factor_dict.values()))
    manager = RebalancePeriodManager(first_factor, ret, rebalance_period)
    _, ret_periods = manager.align_factor_return_by_period()
    common_dates = ret_periods.index

    factor_periods_dict = {}
    for name, fdf in factor_dict.items():
        mgr = RebalancePeriodManager(fdf, ret, rebalance_period)
        fp, _ = mgr.align_factor_return_by_period()
        # 统一对齐到 ret_periods 的日期，缺失日期以 NaN 填充
        fp = fp.reindex(common_dates)
        factor_periods_dict[name] = fp

    return factor_periods_dict, ret_periods


# ---------------------------------------------------------------------------
# 写输出1：复合因子 Excel
# ---------------------------------------------------------------------------

def write_composite_factors_excel(composite_dict, out_path):
    """每个sheet为一个复合因子 DataFrame(date×stock)。"""
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for name, df in composite_dict.items():
            sheet = name[:31]  # Excel sheet名最长31字符
            df.to_excel(writer, sheet_name=sheet)
    print(f"复合因子数据已写入: {out_path}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 加载数据
    factor_files = get_selected_factor_files()
    if not factor_files:
        raise FileNotFoundError("未找到选定因子文件，请检查 SELECTED_FACTOR_NAMES 配置")
    print(f"选定因子文件 ({len(factor_files)} 个):")
    for f in factor_files:
        print(f"  {os.path.basename(f)}")

    ret = load_return_data(PRICE_FILE, RETURN_COLUMN)
    factor_dict = load_selected_factors(factor_files)

    # 2. 对齐到调仓期
    factor_periods_dict, ret_periods = align_to_rebalance_periods(
        factor_dict, ret, REBALANCE_PERIOD
    )
    print(f"调仓期数量: {len(ret_periods)}")

    # 3. 计算复合因子（在调仓期截面上）
    selected_composite = os.environ.get("REBALANCE_SELECTED_COMPOSITE")
    if selected_composite:
        names = [n.strip() for n in selected_composite.split(",") if n.strip()]
        print(f"计算选定复合因子: {names}")
        composite_dict = compute_selected_composites(
            factor_periods_dict, ret_periods, names, N_WINDOWS, M_WINDOWS
        )
    else:
        print("计算复合因子...")
        composite_dict = compute_all_composites(
            factor_periods_dict, ret_periods, N_WINDOWS, M_WINDOWS
        )
    print(f"复合因子数量: {len(composite_dict)}")

    # 4. 输出1：复合因子 Excel
    out1 = os.path.join(OUTPUT_DIR, "composite_factors.xlsx")
    write_composite_factors_excel(composite_dict, out1)

    # 5. 对每个复合因子跑回测（复合因子已在调仓期截面，直接作为 factor_periods 传入）
    config = SimpleNamespace(
        GROUP_NUM=GROUP_NUM,
        WEIGHT_METHOD=WEIGHT_METHOD,
        RISK_FREE_RATE=RISK_FREE_RATE,
        TRANSACTION_COST=TRANSACTION_COST,
    )

    factor_names = []
    records = []
    records_3M, records_6M, records_1Y = [], [], []
    total = len(composite_dict)
    for i, (name, comp_df) in enumerate(composite_dict.items()):
        print(f"[{i+1}/{total}] 回测复合因子: {name}")
        # comp_df 已是调仓期截面 DataFrame，直接用 run_one_factor_one_period
        # 但该函数内部会再做一次 RebalancePeriodManager 对齐，
        # 所以我们传入 rebalance_period=1 并确保 comp_df 的 index 与 ret_periods 一致
        # 更直接：复用 run_one_factor_one_period 的内部逻辑，传入已对齐的数据
        rec = _run_composite_backtest(comp_df, ret_periods, config)
        factor_names.append(name)
        records.append(rec)

        # 近期 3M / 6M / 1Y 的 factor_test_statistics
        for lb_months, rec_list in [(3, records_3M), (6, records_6M), (12, records_1Y)]:
            comp_filt, ret_filt = filter_factor_ret_by_lookback(comp_df, ret_periods, lb_months)
            rec_lb = _run_composite_backtest(comp_filt, ret_filt, config)
            rec_list.append(rec_lb)

    # 6. 输出2：集中回测报表 Excel（前四个 sheet 每行按 long_annual_return 降序）
    def _sort_by_return(df):
        return df.sort_values(by="long_annual_return", ascending=False, na_position="last")

    sheet1_df = _sort_by_return(build_sheet1_df(records, factor_names))
    sheet1_3M_df = _sort_by_return(build_sheet1_df(records_3M, factor_names))
    sheet1_6M_df = _sort_by_return(build_sheet1_df(records_6M, factor_names))
    sheet1_1Y_df = _sort_by_return(build_sheet1_df(records_1Y, factor_names))
    sheet2_df = build_sheet2_df(records, factor_names)
    sheet3_df = build_long_excess_df(records, factor_names)
    sheet4_df = build_long_cumret_df(records, factor_names)

    out2 = os.path.join(OUTPUT_DIR, f"composite_backtest_report_P{REBALANCE_PERIOD}.xlsx")
    write_excel_with_format(
        out2, sheet1_df, sheet2_df, sheet3_df, sheet4_df,
        sheet1_3M_df=sheet1_3M_df, sheet1_6M_df=sheet1_6M_df, sheet1_1Y_df=sheet1_1Y_df,
    )
    print(f"回测报表已写入: {out2}")


def _run_composite_backtest(comp_df, ret_periods, config):
    """
    对已对齐到调仓期的复合因子直接跑回测，复用 run_multi_factor_test 的内部逻辑。
    comp_df: DataFrame(rebalance_date × stock) — 复合因子截面值
    ret_periods: DataFrame(rebalance_date × stock) — 对应期间累计收益
    """
    # 对齐日期
    common_dates = comp_df.index.intersection(ret_periods.index)
    if len(common_dates) == 0:
        from run_multi_factor_test import _empty_factor_record
        return _empty_factor_record(config)

    fp = comp_df.loc[common_dates]
    rp = ret_periods.loc[common_dates]

    # 直接调用 run_one_factor_one_period，传 rebalance_period=1
    # 但该函数会再做 RebalancePeriodManager，会丢失最后一期
    # 所以直接内联核心逻辑（复用已有模块）
    from ic import ICAnalyzerEnhanced
    from grouping import GrouperEnhanced
    from backtest import LongOnlyBacktest
    from performance import PerformanceAnalyzer
    from run_multi_factor_test import _empty_factor_record

    if len(fp) == 0:
        return _empty_factor_record(config)

    ic_analyzer = ICAnalyzerEnhanced(fp, rp)
    ic_df = ic_analyzer.calculate_ic()
    ic_stats = {
        "IC": ic_analyzer.calculate_statistics(ic_df["IC"]),
        "Rank_IC": ic_analyzer.calculate_statistics(ic_df["Rank_IC"]),
    }

    grouper = GrouperEnhanced(fp, config.GROUP_NUM, config.WEIGHT_METHOD)
    group_dict = grouper.split()
    weight_dict = grouper.get_group_weights(group_dict)
    group_returns = grouper.calculate_group_returns(group_dict, rp, weight_dict)

    group_ic_df = ic_analyzer.calculate_group_ic(group_dict, group_returns)
    group_ic_stats = {
        "Group_IC": ic_analyzer.calculate_statistics(group_ic_df["Group_IC"]),
        "Group_Rank_IC": ic_analyzer.calculate_statistics(group_ic_df["Group_Rank_IC"]),
    }

    cols = group_returns.columns.tolist()
    if len(cols) >= 4:
        top2_cols, bottom2_cols = cols[-2:], cols[:2]
    elif len(cols) >= 2:
        # 分组 2~3 组时，根据实际列动态选取，避免与真实组号不一致
        top2_cols, bottom2_cols = cols[-2:], cols[:2]
    else:
        # 分组不足 2 组时无法做多空，用占位零收益
        idx = group_returns.index if len(group_returns) > 0 else rp.index
        group_returns = pd.DataFrame(np.zeros((len(idx), 4)), index=idx, columns=[1, 2, 3, 4])
        top2_cols, bottom2_cols = [4, 3], [1, 2]

    long_combined = group_returns[top2_cols].mean(axis=1)
    short_combined_raw = group_returns[bottom2_cols].mean(axis=1)

    ls_returns = long_combined - short_combined_raw - 2 * config.TRANSACTION_COST
    ls_nav = (1 + ls_returns).cumprod()
    ls_stats = PerformanceAnalyzer(ls_nav, ls_returns, config.RISK_FREE_RATE).calculate_metrics()

    short_ret = -group_returns[bottom2_cols].mean(axis=1) - config.TRANSACTION_COST
    short_nav = (1 + short_ret).cumprod()
    short_combined_stats = PerformanceAnalyzer(short_nav, short_ret, config.RISK_FREE_RATE).calculate_metrics()

    benchmark_returns = rp.mean(axis=1)
    long_ret = long_combined - config.TRANSACTION_COST
    long_nav = (1 + long_ret).cumprod()
    long_metrics = PerformanceAnalyzer(long_nav, long_ret, config.RISK_FREE_RATE).calculate_metrics()

    long_excess_returns = long_ret - benchmark_returns
    long_excess_nav = (1 + long_excess_returns).cumprod()
    long_excess_metrics = PerformanceAnalyzer(long_excess_nav, long_excess_returns, config.RISK_FREE_RATE).calculate_metrics()

    return {
        "ic_df": ic_df,
        "ic_stats": ic_stats,
        "group_ic_stats": group_ic_stats,
        "ls_stats": ls_stats,
        "short_combined_stats": short_combined_stats,
        "long_annual_return": long_metrics["Annual_Return"],
        "long_sharpe": long_metrics["Sharpe"],
        "long_excess_annual": long_excess_metrics["Annual_Return"],
        "long_excess_sharpe": long_excess_metrics["Sharpe"],
        "long_excess_returns": long_excess_returns,
        "long_excess_nav": long_excess_nav,
        "long_nav": long_nav,
        "ls_returns": ls_returns,
        "group_returns": group_returns,
        "ret_periods": rp,
    }


if __name__ == "__main__":
    main()
