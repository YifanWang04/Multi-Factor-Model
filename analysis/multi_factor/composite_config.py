"""
因子复合配置文件 (composite_config.py)
"""
import os
import sys

PROJECT_ROOT = r"D:\qqq"
FACTOR_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "factor_processed")
PRICE_FILE = os.path.join(PROJECT_ROOT, "data", "us_top100_daily_2023_present.xlsx")
RETURN_COLUMN = "Return"

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "composite_factor_reports")

# 选定因子的1-based索引（对应 factor_processed 目录排序后的因子）
SELECTED_FACTOR_INDICES = [2, 10, 1, 8, 6]

# 调仓周期（天）
REBALANCE_PERIOD = 10

# 一元/IC加权滚动窗口列表 N
N_WINDOWS = [5, 10, 20]

# 多元回归滚动窗口列表 M
M_WINDOWS = [5, 10, 20]

# 回测参数
GROUP_NUM = 10
WEIGHT_METHOD = "equal"
RISK_FREE_RATE = 0.02
TRANSACTION_COST = 0.001


def get_all_factor_files(factor_dir=None):
    factor_dir = factor_dir or FACTOR_PROCESSED_DIR
    if not os.path.isdir(factor_dir):
        return []
    out = []
    seen = set()
    for f in sorted(os.listdir(factor_dir)):
        if not f.endswith(".xlsx"):
            continue
        full = os.path.join(factor_dir, f)
        base = f.replace("_processed.xlsx", "").replace(".xlsx", "")
        if base in seen:
            continue
        seen.add(base)
        out.append(full)
    return out


def get_selected_factor_files():
    all_files = get_all_factor_files()
    selected = []
    for idx in SELECTED_FACTOR_INDICES:
        if 1 <= idx <= len(all_files):
            selected.append(all_files[idx - 1])
    return selected


def get_factor_display_name(filepath):
    basename = os.path.splitext(os.path.basename(filepath))[0]
    if basename.endswith("_processed"):
        basename = basename[:-len("_processed")]
    if basename.lower().startswith("factor_"):
        basename = basename[7:]
    return basename.replace("_", " ").strip() or basename
