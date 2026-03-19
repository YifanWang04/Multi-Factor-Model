"""
Yahoo Finance 行情数据下载脚本 (data/pull_yhfinance_Data.py)
=====================================
本脚本通过 yfinance 下载指定股票列表的日频行情（含复权收盘价、成交量等），并写入单个 Excel 文件，每个标的一个 sheet，便于后续因子构建与回测使用。

行为说明：
- 股票列表：脚本内硬编码为约 100 只美股的 ticker（如 AAPL, MSFT 等）。
- 时间范围：start_date 至当前日期（end_date 使用 datetime.today()）。
- DATA_START_OFFSET_DAYS：数据起始日提前的交易日数，0=不提前；正数=提前 N 个交易日，使调仓日历整体前移。
- 输出：offset=0 时为 us_top100_daily_2023_present.xlsx；offset!=0 时为 us_top100_daily_2023_present_offset{N}d.xlsx，避免覆盖原数据。
"""

import os
import sys

# 确保项目根目录在 path 中，以便 import data.data_config
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd
import yfinance as yf
from datetime import datetime

from data.data_config import (
    DATA_START_OFFSET_DAYS,
    DATA_BASE_START_DATE,
    PRICE_FILE,
    _price_filename,
)

# 1. 参数设置

# 计算实际 start_date：基准日向前推 N 个交易日
if DATA_START_OFFSET_DAYS <= 0:
    start_date = DATA_BASE_START_DATE
else:
    base = pd.Timestamp(DATA_BASE_START_DATE)
    # 向前取 N 个交易日（bdate_range 的 periods 从 base 开始，需反向）
    bd = pd.bdate_range(end=base, periods=DATA_START_OFFSET_DAYS + 1, freq="B")
    start_date = bd[0].strftime("%Y-%m-%d")
    print(f"DATA_START_OFFSET_DAYS={DATA_START_OFFSET_DAYS}，起始日提前至 {start_date}")

end_date = datetime.today().strftime("%Y-%m-%d")

codes = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "BRK-B", "TSLA", "JPM", "JNJ",
    "V", "PG", "UNH", "HD", "MA", "XOM", "LLY", "MRK", "ABBV", "PEP",
    "KO", "AVGO", "COST", "WMT", "BAC", "MCD", "CSCO", "ADBE", "CRM", "NFLX",
    "ORCL", "ACN", "TMO", "ABT", "CVX", "DHR", "TXN", "VZ", "NEE", "PM",
    "INTC", "QCOM", "HON", "IBM", "AMD", "LIN", "LOW", "GS", "MS", "UPS",
    "RTX", "SPGI", "CAT", "AMGN", "INTU", "DE", "ISRG", "MDT", "AXP", "BLK",
    "NOW", "LMT", "SCHW", "BA", "CB", "PLD", "BKNG", "CI", "MMC", "TGT",
    "MO", "GE", "ADI", "GILD", "SYK", "EL", "ZTS", "USB", "PGR", "SO",
    "DUK", "CME", "APD", "BDX", "ITW", "EW", "CSX", "NSC", 'CCJ', 'SVM',
    'WPM', 'PAAS', 'TSM', 'MU', 'PLTR', 'WDC', 'STX', 'VRT',
    'TER', 'AEP', 'TTMI', 'RKLB', 'ASTS', 'SNDK', 'RMBS', 'ONDS', 'HROW',
    'SANM', 'ANET' 
]

# 2. get data

data_dict = {}

for i, code in enumerate(codes, 1):
    print(f"Downloading {i}/{len(codes)}: {code}")
    
    df = yf.download(
        code,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        print(f"{code} has no data")
        continue

    # 单标的 yf.download 可能返回单级或 MultiIndex 列名
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)
    df["Ticker"] = code

    data_dict[code] = df

# 3. write into Excel (multiple sheets)

if not data_dict:
    raise RuntimeError("没有成功下载任何股票数据，请检查网络或股票代码")

_run_dir = os.environ.get("REBALANCE_RUN_DIR")
_price_name = _price_filename()
if _run_dir:
    _out_path = os.path.join(_run_dir, "data", _price_name)
    os.makedirs(os.path.dirname(_out_path), exist_ok=True)
else:
    _out_path = PRICE_FILE
    os.makedirs(os.path.dirname(_out_path), exist_ok=True)

with pd.ExcelWriter(_out_path, engine="xlsxwriter") as writer:
    for sheet_name, df in data_dict.items():
        df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        print(f"Saved sheet: {sheet_name}")

print("Data download completed.")
