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
import pandas_market_calendars as mcal

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
    ## June 8, 2026 add new stocks
    "AMAT", "LRCX", "CRDO", "ARM", 
    "AAOI",
    "MRVL", "NBIS",
    "BN", "FN", "COHR", "FLY", "RDW", "GLW", "DELL",
    "HPE", "ALAB", "CIEN", "LITE", "MTSI", "ASML", "SNPS", "CDNS",
    "ETN", "GEV", "PWR", "CLS", "JBL", "FLEX", "FIX", "DDOG", "NET",
    "MDB", "PANW", "CRWD",
    "KLAC" #June 12, 2026拆股一拆十
]


def configured_ticker_set() -> set[str]:
    """Return the configured ticker universe used by data/factor loaders."""
    return set(YFINANCE_TICKERS)


def should_use_price_sheet(sheet_name: str) -> bool:
    """Whether an Excel sheet belongs to the configured ticker universe."""
    universe = configured_ticker_set()
    return not universe or sheet_name in universe

# yf.download 参数（与历史 Excel 列含义一致时可保持 auto_adjust=False）
YFINANCE_DOWNLOAD_AUTO_ADJUST = False
YFINANCE_DOWNLOAD_PROGRESS = False

# 因子构建 OHLC 口径：
# False = 兼容历史回测：Open/High/Low 使用 yfinance 原始列，close 使用 Adj Close。
# True  = 更严格复权口径：优先使用 Adj Open/Adj High/Adj Low 与 Adj Close。
FACTOR_USE_ADJUSTED_OHLC = False


def yfinance_pull_start_date() -> str:
    """根据 DATA_BASE_START_DATE 与 DATA_START_OFFSET_DAYS 得到 yfinance 的 start 参数（YYYY-MM-DD）。
    通过 _resolve_offset() 优先读取环境变量 REBALANCE_OFFSET_DAYS。
    """
    offset = _resolve_offset()
    if offset <= 0:
        return DATA_BASE_START_DATE
    base = pd.Timestamp(DATA_BASE_START_DATE)
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(
        start_date=(base - pd.Timedelta(days=offset * 3 + 30)).strftime("%Y-%m-%d"),
        end_date=base.strftime("%Y-%m-%d"),
    )
    valid_days = schedule.index.tz_localize(None)
    prior_days = valid_days[valid_days < base.normalize()]
    if len(prior_days) < offset:
        raise ValueError(
            f"NYSE 日历不足以从 {DATA_BASE_START_DATE} 回推 {offset} 个交易日"
        )
    start = pd.Timestamp(prior_days[-offset])
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
# offset 文件不存在时不再回退到基线文件，读取方必须 fail fast。
_BASE_PRICE_FILE = os.path.join(_PROJECT_ROOT, "data", "us_top100_daily_2023_present.xlsx")
_OFFSET_PRICE_FILE = os.path.join(_PROJECT_ROOT, "data", _price_filename())
_RESOLVED_OFFSET = _resolve_offset()
if _RESOLVED_OFFSET == 0:
    PRICE_FILE = _BASE_PRICE_FILE
else:
    PRICE_FILE = _OFFSET_PRICE_FILE


def require_price_file_exists(price_file: str | None = None) -> str:
    """返回价格文件路径；若不存在则报错，禁止 offset 静默回退到基线文件。"""
    path = price_file or PRICE_FILE
    if os.path.isfile(path):
        return path
    offset = _resolve_offset()
    if offset != 0:
        raise FileNotFoundError(
            f"DATA_START_OFFSET_DAYS/REBALANCE_OFFSET_DAYS={offset}，"
            f"但 offset 价格文件不存在: {path}。"
            "请先运行 data/pull_yhfinance_Data.py 生成对应 offset 数据，"
            "避免静默回退到基线价格文件造成回测口径混淆。"
        )
    raise FileNotFoundError(
        f"DATA_START_OFFSET_DAYS/REBALANCE_OFFSET_DAYS={_RESOLVED_OFFSET}，"
        f"但价格文件不存在: {path}。请先运行 data/pull_yhfinance_Data.py。"
    )

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
