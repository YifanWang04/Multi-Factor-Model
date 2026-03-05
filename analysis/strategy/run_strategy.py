"""
策略构建主流程入口 (run_strategy.py)
=====================================
完整运行步骤：
  1. 加载配置（strategy_config.py）
  2. 读取复合因子数据（composite_factors.xlsx，指定 sheet）
  3. 读取日频收益率数据
  4. StrategyBacktester.run_grid() — 网格遍历所有参数组合
  5. compute_all_metrics()        — 计算全部绩效指标
  6. StrategyReporter.write()     — 输出 Excel 报表

用法（在项目根目录下）：
  python analysis/strategy/run_strategy.py

修改参数：
  编辑 analysis/strategy/strategy_config.py 中的各项设置即可。
"""

import os
import sys
import time

import numpy as np
import pandas as pd

# ── 路径注册 ──────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_SF_DIR = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "analysis", "single_factor")
_ROOT = os.path.dirname(os.path.dirname(_HERE))

for _p in [_HERE, _SF_DIR, _ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── 本地模块 ──────────────────────────────────────────────────────────────────
import strategy_config as cfg
from strategy_backtest import StrategyBacktester
from strategy_metrics import compute_all_metrics
from strategy_report import StrategyReporter

# run_multi_factor_test 提供与单因子流程一致的数据加载函数
from run_multi_factor_test import load_return_data


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_composite_factor(file_path: str, sheet_name: str) -> pd.DataFrame:
    """
    从 composite_factors.xlsx 加载指定 sheet 的复合因子数据。
    index = 调仓日（DatetimeIndex），columns = 股票代码。
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"复合因子文件不存在: {file_path}")

    xl = pd.ExcelFile(file_path)
    available = xl.sheet_names
    if sheet_name not in available:
        raise ValueError(
            f"Sheet '{sheet_name}' 不存在于 {os.path.basename(file_path)}。\n"
            f"可用 sheet: {available}"
        )

    df = pd.read_excel(file_path, sheet_name=sheet_name, index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.apply(pd.to_numeric, errors="coerce")
    df.sort_index(inplace=True)
    return df


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    print("=" * 64)
    print("  策略构建与回测")
    print("=" * 64)

    # ── 1. 加载复合因子 ────────────────────────────────────────────────
    print(f"\n[1/4] 加载复合因子: {cfg.COMPOSITE_FACTOR_SHEET}")
    factor_df = load_composite_factor(cfg.COMPOSITE_FACTOR_FILE, cfg.COMPOSITE_FACTOR_SHEET)
    if factor_df.empty or len(factor_df) == 0:
        raise ValueError("复合因子数据为空，请检查 COMPOSITE_FACTOR_FILE 与 COMPOSITE_FACTOR_SHEET")
    print(f"      因子形状: {factor_df.shape}  "
          f"日期范围: {factor_df.index[0].date()} ~ {factor_df.index[-1].date()}")

    # ── 2. 加载日频收益率 ──────────────────────────────────────────────
    print(f"\n[2/4] 加载日频收益率: {os.path.basename(cfg.PRICE_FILE)}")
    ret_df = load_return_data(cfg.PRICE_FILE, cfg.RETURN_COLUMN)
    ret_df.sort_index(inplace=True)
    print(f"      收益率形状: {ret_df.shape}  "
          f"日期范围: {ret_df.index[0].date()} ~ {ret_df.index[-1].date()}")

    # ── 参数组合数预览 ─────────────────────────────────────────────────
    n_combos = (
        len(cfg.GROUP_NUMS)
        * len(cfg.REBALANCE_PERIODS)
        * len(cfg.TARGET_GROUP_RANKS)
        * len(cfg.WEIGHT_METHODS)
    )
    print(f"\n      参数组合数: {n_combos}")
    print(f"      分层数量: {cfg.GROUP_NUMS}")
    print(f"      调仓周期: {cfg.REBALANCE_PERIODS} 天")
    print(f"      目标组排名: {cfg.TARGET_GROUP_RANKS}（1=最高组）")
    print(f"      资产配置方式: {cfg.WEIGHT_METHODS}")

    # ── 3. 网格回测 ────────────────────────────────────────────────────
    print(f"\n[3/4] 开始网格回测（共 {n_combos} 个组合）...")
    backtester = StrategyBacktester(factor_df, ret_df, cfg)
    results = backtester.run_grid()

    # 统计有效策略数
    valid = sum(
        1 for r in results.values() if len(r.get("daily_returns", [])) > 0
    )
    print(f"\n      完成：{valid}/{n_combos} 个策略有效")

    # ── 4. 计算绩效指标 ────────────────────────────────────────────────
    print("\n[4/4] 计算绩效指标并生成报表...")
    all_metrics = compute_all_metrics(results, rf=cfg.RISK_FREE_RATE)

    # ── 5. 输出 Excel ─────────────────────────────────────────────────
    output_path = os.path.join(cfg.OUTPUT_DIR, cfg.OUTPUT_EXCEL_NAME)
    reporter = StrategyReporter(results, all_metrics, cfg)
    reporter.write(output_path)

    elapsed = time.time() - t0
    print(f"\n完成！耗时 {elapsed:.1f}s")
    print(f"报表路径: {output_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()
