"""
Yahoo Finance 行情数据下载脚本 (data/pull_yhfinance_Data.py)
=====================================
本脚本通过 yfinance 下载指定股票列表的日频行情（含复权收盘价、成交量等），并写入单个 Excel 文件，每个标的一个 sheet，便于后续因子构建与回测使用。

行为说明：
- 股票列表：脚本内硬编码为约 100 只美股的 ticker（如 AAPL, MSFT 等）。
- 时间范围：start_date 至当前日期（end_date 使用 datetime.today()）。
- 下载字段：yf.download 返回的 OHLCV；脚本中使用 Adj Close 与 Volume（在 build_factors 中读取）。
- 输出：us_top100_daily_2023_present.xlsx，每个 sheet 名为 ticker（Excel 限制 31 字符），含 Date 与 Ticker 列。保存路径为运行时的当前目录，通常应放在项目 data 目录下并供 config/pipeline 引用。
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime

# 1. 参数设置

start_date = "2023-01-01"
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
if _run_dir:
    _out_path = os.path.join(_run_dir, "data", "us_top100_daily_2023_present.xlsx")
    os.makedirs(os.path.dirname(_out_path), exist_ok=True)
else:
    _out_path = "data/us_top100_daily_2023_present.xlsx"

with pd.ExcelWriter(_out_path, engine="xlsxwriter") as writer:
    for sheet_name, df in data_dict.items():
        df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        print(f"Saved sheet: {sheet_name}")

print("Data download completed.")
