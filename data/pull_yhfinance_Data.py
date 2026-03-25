"""
Yahoo Finance 行情数据下载脚本 (data/pull_yhfinance_Data.py)
=====================================
本脚本通过 yfinance 下载指定股票列表的日频行情（含复权收盘价、成交量等），并写入单个 Excel 文件，每个标的一个 sheet，便于后续因子构建与回测使用。

行为说明：
- 股票列表与时间范围：见 data/data_config.py（YFINANCE_TICKERS、yfinance_pull_start_date、DATA_BASE_START_DATE）。
- end_date 使用运行当日（datetime.today()）。
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
from datetime import datetime, timedelta

from data.data_config import (
    DATA_START_OFFSET_DAYS,
    YFINANCE_DOWNLOAD_AUTO_ADJUST,
    YFINANCE_DOWNLOAD_PROGRESS,
    YFINANCE_TICKERS,
    _price_filename,
    yfinance_pull_start_date,
)

# 1. 运行参数（配置均在 data_config）

start_date = yfinance_pull_start_date()
if DATA_START_OFFSET_DAYS > 0:
    print(f"DATA_START_OFFSET_DAYS={DATA_START_OFFSET_DAYS}，起始日提前至 {start_date}")

# yfinance 的 end 为不包含结束日，+1 天以纳入「运行当日」的日线
end_date = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
codes = YFINANCE_TICKERS

# 2. get data

data_dict = {}

for i, code in enumerate(codes, 1):
    print(f"Downloading {i}/{len(codes)}: {code}")
    
    df = yf.download(
        code,
        start=start_date,
        end=end_date,
        auto_adjust=YFINANCE_DOWNLOAD_AUTO_ADJUST,
        progress=YFINANCE_DOWNLOAD_PROGRESS,
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
    # 必须按当前 offset 写入对应文件名；勿用 PRICE_FILE（offset 文件不存在时会回退到基线路径并覆盖）
    _out_path = os.path.join(_ROOT, "data", _price_filename())
    os.makedirs(os.path.dirname(_out_path), exist_ok=True)

with pd.ExcelWriter(_out_path, engine="xlsxwriter") as writer:
    for sheet_name, df in data_dict.items():
        df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        print(f"Saved sheet: {sheet_name}")

print("Data download completed.")
