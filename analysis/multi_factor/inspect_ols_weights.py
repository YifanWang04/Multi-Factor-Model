"""
查看 ols_m3_M5 每期各因子权重占比。
运行: python analysis/multi_factor/inspect_ols_weights.py
输出: output/composite_factor_reports/ols_m3_M5_weights.xlsx
"""
import os, sys
_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_DIR))
for p in [_DIR, os.path.join(_ROOT, "analysis", "single_factor"), _ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd
from composite_config import (
    PRICE_FILE, RETURN_COLUMN, OUTPUT_DIR, REBALANCE_PERIOD, M_WINDOWS,
    get_selected_factor_files, get_factor_display_name,
)
from composite_factor import _factor_matrix, _ols_betas_per_period
from rebalance_manager import RebalancePeriodManager
from run_multi_factor_test import load_return_data, load_factor
from run_composite_factor import load_selected_factors, align_to_rebalance_periods

# --- 加载数据 ---
ret = load_return_data(PRICE_FILE, RETURN_COLUMN)
factor_files = get_selected_factor_files()
factor_dict = load_selected_factors(factor_files)
factor_periods, ret_periods = align_to_rebalance_periods(factor_dict, ret, REBALANCE_PERIOD)

# --- 计算 ols_m3_M5 权重 ---
M = 5
dates = ret_periods.index
aligned = _factor_matrix(factor_periods, dates)
beta_df = _ols_betas_per_period(aligned, ret_periods)

# 滚动 M 期均值权重
weight_rows = []
for d in dates:
    past = beta_df[beta_df.index <= d].iloc[-M:]
    weight_rows.append(past.mean() if len(past) > 0 else pd.Series(np.nan, index=beta_df.columns))
weight_df = pd.DataFrame(weight_rows, index=dates)

# 归一化为占比（按绝对值之和）
abs_sum = weight_df.abs().sum(axis=1).replace(0, np.nan)
weight_pct = weight_df.div(abs_sum, axis=0) * 100  # 百分比

# --- 输出 ---
out_path = os.path.join(OUTPUT_DIR, "ols_m3_M5_weights.xlsx")
os.makedirs(OUTPUT_DIR, exist_ok=True)
with pd.ExcelWriter(out_path) as writer:
    weight_df.to_excel(writer, sheet_name="raw_weights")
    weight_pct.to_excel(writer, sheet_name="weight_pct(%)")
    beta_df.to_excel(writer, sheet_name="per_period_betas")

print(f"已保存: {out_path}")
print("\n=== 各因子平均权重占比(%) ===")
print(weight_pct.mean().round(2).to_string())
print("\n=== 最新一期权重占比(%) ===")
print(weight_pct.iloc[-1].round(2).to_string())
