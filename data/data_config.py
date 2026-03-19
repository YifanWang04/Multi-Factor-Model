"""
数据路径与起始日配置 (data/data_config.py)
============================================
集中定义数据起始日偏移（DATA_START_OFFSET_DAYS）及所有相关路径。
按 offset 分子目录，避免切换 offset 时覆盖原数据。

- DATA_START_OFFSET_DAYS: 数据起始日提前的交易日数，0=不提前
- offset=0: 使用默认目录 factor_raw/, factor_processed/, output/composite_factor_reports/
- offset!=0: 使用 factor_raw_offset{N}d/, factor_processed_offset{N}d/, output/composite_factor_reports_offset{N}d/
"""

import os

# 项目根目录（data 的上级）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据起始日提前的交易日数：0=不提前，正数=提前 N 个交易日
# 注：此值只从配置文件读取（不再支持通过环境变量覆盖）
DATA_START_OFFSET_DAYS = 0

# 基准起始日（用于 pull 计算实际 start_date）
DATA_BASE_START_DATE = "2023-01-01"

# 价格文件名（不含路径）
def _price_filename() -> str:
    if DATA_START_OFFSET_DAYS == 0:
        return "us_top100_daily_2023_present.xlsx"
    return f"us_top100_daily_2023_present_offset{DATA_START_OFFSET_DAYS}d.xlsx"

# 目录后缀：offset=0 为空，offset!=0 为 _offset{N}d
def _offset_dir_suffix() -> str:
    if DATA_START_OFFSET_DAYS == 0:
        return ""
    return f"_offset{DATA_START_OFFSET_DAYS}d"

# 默认价格文件路径（项目 data 目录下）
PRICE_FILE = os.path.join(_PROJECT_ROOT, "data", _price_filename())

# 因子目录（按 offset 分子目录，不覆盖）
FACTOR_RAW_DIR = os.path.join(_PROJECT_ROOT, f"factor_raw{_offset_dir_suffix()}")
FACTOR_PROCESSED_DIR = os.path.join(_PROJECT_ROOT, f"factor_processed{_offset_dir_suffix()}")

# 复合因子输出目录（按 offset 分子目录，不覆盖）
COMPOSITE_FACTOR_OUTPUT_DIR = os.path.join(
    _PROJECT_ROOT, "output", f"composite_factor_reports{_offset_dir_suffix()}"
)
COMPOSITE_FACTOR_FILE = os.path.join(COMPOSITE_FACTOR_OUTPUT_DIR, "composite_factors.xlsx")

# 其他输出目录（按 offset 分子目录，不覆盖）
STRATEGY_REPORTS_DIR = os.path.join(_PROJECT_ROOT, "output", f"strategy_reports{_offset_dir_suffix()}")
WALK_FORWARD_REPORTS_DIR = os.path.join(_PROJECT_ROOT, "output", f"walk_forward_reports{_offset_dir_suffix()}")
SINGLE_FACTOR_REPORTS_DIR = os.path.join(_PROJECT_ROOT, "output", f"single_factor_reports{_offset_dir_suffix()}")
MULTI_FACTOR_REPORTS_DIR = os.path.join(_PROJECT_ROOT, "output", f"multi_factor_reports{_offset_dir_suffix()}")
