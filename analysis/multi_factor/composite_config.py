"""
因子复合配置文件 (composite_config.py)
"""
import os

PROJECT_ROOT = r"D:\qqq"
_RUN_DIR = os.environ.get("REBALANCE_RUN_DIR")
if _RUN_DIR:
    FACTOR_PROCESSED_DIR = os.path.join(_RUN_DIR, "factor_processed")
    PRICE_FILE = os.path.join(_RUN_DIR, "data", "us_top100_daily_2023_present.xlsx")
    OUTPUT_DIR = os.path.join(_RUN_DIR, "composite_factor_reports")
else:
    FACTOR_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "factor_processed")
    PRICE_FILE = os.path.join(PROJECT_ROOT, "data", "us_top100_daily_2023_present.xlsx")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "composite_factor_reports")
RETURN_COLUMN = "Return"

# 选定因子：在 config 中直接写因子编号（factor_library 中的编号，如 95 → alpha095）
SELECTED_FACTOR_INDICES = [95, 101, 62, 65, 32]
SELECTED_FACTOR_NAMES = [f"alpha{i:03d}" for i in SELECTED_FACTOR_INDICES]

# 调仓周期（交易日数）：相邻调仓日之间至少相隔 N 个交易日
# ⚠️ 重要：此周期决定复合因子的生成频率
# 建议：策略回测的 REBALANCE_PERIODS 应包含此值，或为此值的整数倍
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
    if _RUN_DIR:
        # 调仓日流程已通过 REBALANCE_SELECTED_FACTORS 只构建了选定因子
        # factor_processed 中即为全部选定因子，直接返回
        return all_files
    name_to_path = {get_factor_display_name(p): p for p in all_files}
    return [name_to_path[n] for n in SELECTED_FACTOR_NAMES if n in name_to_path]


def get_factor_display_name(filepath):
    basename = os.path.splitext(os.path.basename(filepath))[0]
    if basename.endswith("_processed"):
        basename = basename[:-len("_processed")]
    if basename.lower().startswith("factor_"):
        basename = basename[7:]
    return basename.replace("_", " ").strip() or basename
