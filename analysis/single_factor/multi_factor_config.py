"""
多因子集中测试配置文件 (multi_factor_config.py)
=====================================
定义多因子测试的输入输出路径与参数。
"""
import os

# 项目根目录
PROJECT_ROOT = r"D:\新建文件夹\qqq"

# 因子数据目录：多因子测试默认扫描该目录下所有因子 Excel
FACTOR_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "factor_processed")

# 多个因子数据 Excel 路径。留空 [] 表示使用「所有因子」（扫描 FACTOR_PROCESSED_DIR）
# 若需只跑指定因子，可显式列出路径，例如：
# FACTOR_FILES = [os.path.join(FACTOR_PROCESSED_DIR, "factor_price_slope_processed.xlsx")]
FACTOR_FILES = []

# 收益率数据：与单因子一致，从价格文件计算或使用 Return 列
PRICE_FILE = os.path.join(PROJECT_ROOT, "data", "us_top100_daily_2023_present.xlsx")
RETURN_COLUMN = "Return"

# 多因子报告输出路径（与参考报表一致，便于归档）
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "multi_factor_reports")
OUTPUT_EXCEL_NAME = "multi_factor_test_report.xlsx"  # 可改为 factor_test_report.xlsx 等

# 调仓周期列表（天）：每个周期生成一份独立报表，在此处统一配置
REBALANCE_PERIODS = [10]

# 共线性分析输出文件名前缀（调仓周期后缀由代码自动追加，如 _P5.xlsx）
OUTPUT_COLLINEARITY_NAME = "factor_collinearity_report"

# 与单因子一致的参数
GROUP_NUM = 10
WEIGHT_METHOD = "equal"
RISK_FREE_RATE = 0.02
TRANSACTION_COST = 0.001


def get_factor_display_name(filepath):
    """从因子文件路径得到显示名。"""
    basename = os.path.splitext(os.path.basename(filepath))[0]
    if basename.endswith("_processed"):
        basename = basename[: -len("_processed")]
    if basename.lower().startswith("factor_"):
        basename = basename[7:]
    return basename.replace("_", " ").strip() or basename


def get_all_factor_files(factor_dir=None):
    """
    扫描因子目录，返回所有因子 Excel 的绝对路径列表（用于多因子集中测试）。
    优先 *_processed.xlsx，否则 factor_*.xlsx；同一 base 只保留一份。
    """
    factor_dir = factor_dir or FACTOR_PROCESSED_DIR
    if not os.path.isdir(factor_dir):
        return []
    out = []
    seen = set()
    for f in sorted(os.listdir(factor_dir)):
        if not f.endswith(".xlsx"):
            continue
        full = os.path.join(factor_dir, f)
        if not os.path.isfile(full):
            continue
        base = f.replace("_processed.xlsx", "").replace(".xlsx", "")
        if base in seen:
            continue
        if f.endswith("_processed.xlsx"):
            seen.add(base)
            out.append(full)
        elif f.startswith("factor_") and f.endswith(".xlsx"):
            seen.add(base)
            out.append(full)
    return out
