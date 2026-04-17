"""
数据路径与起始日配置 (data/data_config.py)
============================================
集中定义数据起始日偏移（DATA_START_OFFSET_DAYS）、yfinance 拉取标的列表、
实际拉取起始日计算及所有相关路径。
按 offset 分子目录，避免切换 offset 时覆盖原数据。

- DATA_START_OFFSET_DAYS: 数据起始日提前的交易日数，0=不提前
- offset=0: 使用默认目录 factor_raw/, factor_processed/, output/composite_factor_reports/
- offset!=0: 使用 factor_raw_offset{N}d/, factor_processed_offset{N}d/, output/composite_factor_reports_offset{N}d/
"""

import os

import pandas as pd

# 项目根目录（data 的上级）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据起始日提前的交易日数：0=不提前，正数=提前 N 个交易日
# 注：此值只从配置文件读取（不再支持通过环境变量覆盖）
# 6 = 将调仓日从 3.27 提前至 3.19（约 6 个交易日）
DATA_START_OFFSET_DAYS = 0

# 基准起始日（用于 pull 计算实际 start_date）
DATA_BASE_START_DATE = "2023-01-01"

# yfinance 日频拉取标的（约 100 只美股，与 us_top100 命名一致）
YFINANCE_TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "BRK-B", "TSLA", "JPM", "JNJ",
    "V", "PG", "UNH", "HD", "MA", "XOM", "LLY", "MRK", "ABBV", "PEP",
    "KO", "AVGO", "COST", "WMT", "BAC", "MCD", "CSCO", "ADBE", "CRM", "NFLX",
    "ORCL", "ACN", "TMO", "ABT", "CVX", "DHR", "TXN", "VZ", "NEE", "PM",
    "INTC", "QCOM", "HON", "IBM", "AMD", "LIN", "LOW", "GS", "MS", "UPS",
    "RTX", "SPGI", "CAT", "AMGN", "INTU", "DE", "ISRG", "MDT", "AXP", "BLK",
    "NOW", "LMT", "SCHW", "BA", "CB", "PLD", "BKNG", "CI", "TGT",
    "MO", "GE", "ADI", "GILD", "SYK", "EL", "ZTS", "USB", "PGR", "SO",
    "DUK", "CME", "APD", "BDX", "ITW", "EW", "CSX", "NSC", "CCJ", "SVM",
    "WPM", "PAAS", "TSM", "MU", "PLTR", "WDC", "STX", "VRT",
    "TER", "AEP", "TTMI", "RKLB", "ASTS", "SNDK", "RMBS", "ONDS", "HROW",
    "SANM", "ANET", 
    # 'GENV', 'VRT', 'LRCX', 'AMAT', 'NET'
]

# YFINANCE_TICKERS = [
#     "COHR", "LITE", "FLY", "MRVL", "NBIS", "NVDA", "RDW", "GLW", "AEP", "RMBS", 
#     "HROW", "ONDS", "SANM", "TTMI", "MU"
# ]

# yf.download 参数（与历史 Excel 列含义一致时可保持 auto_adjust=False）
YFINANCE_DOWNLOAD_AUTO_ADJUST = False
YFINANCE_DOWNLOAD_PROGRESS = False


def yfinance_pull_start_date() -> str:
    """根据 DATA_BASE_START_DATE 与 DATA_START_OFFSET_DAYS 得到 yfinance 的 start 参数（YYYY-MM-DD）。
    通过 _resolve_offset() 优先读取环境变量 REBALANCE_OFFSET_DAYS。
    """
    offset = _resolve_offset()
    if offset <= 0:
        return DATA_BASE_START_DATE
    base = pd.Timestamp(DATA_BASE_START_DATE)
    # 用 BDay 回推 N 个交易日；避免 bdate_range(end=非交易日, periods=...) 与预期长度不一致
    start = base - pd.offsets.BDay(offset)
    return start.strftime("%Y-%m-%d")


# 统一 offset 解析：优先读环境变量（subprocess 传播），否则读配置文件常量
def _resolve_offset() -> int:
    env_val = os.environ.get("REBALANCE_OFFSET_DAYS")
    if env_val is not None:
        return int(env_val)
    return DATA_START_OFFSET_DAYS


# 价格文件名（不含路径）
def _price_filename() -> str:
    offset = _resolve_offset()
    if offset == 0:
        return "us_top100_daily_2023_present.xlsx"
    return f"us_top100_daily_2023_present_offset{offset}d.xlsx"

# 目录后缀：offset=0 为空，offset!=0 为 _offset{N}d
def _offset_dir_suffix() -> str:
    offset = _resolve_offset()
    if offset == 0:
        return ""
    return f"_offset{offset}d"

# 默认价格文件路径（项目 data 目录下）
# 当 offset 文件不存在时，回退到基线文件
_BASE_PRICE_FILE = os.path.join(_PROJECT_ROOT, "data", "us_top100_daily_2023_present.xlsx")
_OFFSET_PRICE_FILE = os.path.join(_PROJECT_ROOT, "data", _price_filename())
PRICE_FILE = _OFFSET_PRICE_FILE if os.path.isfile(_OFFSET_PRICE_FILE) else _BASE_PRICE_FILE

# 因子目录（按 offset 分子目录，不覆盖）
FACTOR_RAW_DIR = os.path.join(_PROJECT_ROOT, f"factor_raw{_offset_dir_suffix()}")
FACTOR_PROCESSED_DIR = os.path.join(_PROJECT_ROOT, f"factor_processed{_offset_dir_suffix()}")

# 复合因子输出目录（按 offset 分子目录，不覆盖）
COMPOSITE_FACTOR_OUTPUT_DIR = os.path.join(
    _PROJECT_ROOT, "output", f"composite_factor_reports{_offset_dir_suffix()}"
)

# 复合因子文件（不带后缀，仅指向目录；实际文件名由各调用方根据因子索引推导）
# 保留此变量供向后兼容（如 pipeline 中直接引用），但不从 composite_config 推导后缀
_COMPOSITE_BASE_FILE = os.path.join(
    COMPOSITE_FACTOR_OUTPUT_DIR, "composite_factors.xlsx"
)
_BASE_DIR_FILE = os.path.join(
    _PROJECT_ROOT, "output", "composite_factor_reports", "composite_factors.xlsx"
)
COMPOSITE_FACTOR_FILE = _COMPOSITE_BASE_FILE if os.path.isfile(_COMPOSITE_BASE_FILE) else _BASE_DIR_FILE

# 其他输出目录（按 offset 分子目录，不覆盖）
STRATEGY_REPORTS_DIR = os.path.join(_PROJECT_ROOT, "output", f"strategy_reports{_offset_dir_suffix()}")
WALK_FORWARD_REPORTS_DIR = os.path.join(_PROJECT_ROOT, "output", f"walk_forward_reports{_offset_dir_suffix()}")
SINGLE_FACTOR_REPORTS_DIR = os.path.join(_PROJECT_ROOT, "output", f"single_factor_reports{_offset_dir_suffix()}")
MULTI_FACTOR_REPORTS_DIR = os.path.join(_PROJECT_ROOT, "output", f"multi_factor_reports{_offset_dir_suffix()}")
