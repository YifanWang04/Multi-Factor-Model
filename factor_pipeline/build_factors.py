"""
因子构建流水线 (factor_pipeline/build_factors.py)
=====================================
从 factor_library 读取所有因子配置，用 OHLCV 数据构建原始因子并保存到 factor_raw。
不做回测；回测请使用 run_single_factor_test 或 run_batch_single_factor_tests。

详细说明：
- 数据来源：从 data/us_top100_daily_2023_present.xlsx 读取 OHLCV 数据
  （多 sheet，每 sheet 一只标的，期望列：Date、Open、High、Low、Adj Close、Volume）。
- 衍生数据：returns = close.pct_change()；
  vwap ≈ (high+low+close)/3（典型价格，若无 High/Low 则退化为 close）。
- 因子构建：FACTOR_CONFIGS 中每个因子通过 data_keys 指定所需数据，输出单 sheet（"factor"）的原始因子文件。
- 数据处理：去极值与标准化由 factor_pipeline/process_factors.py 单独执行，输出到 factor_processed。

命令行：python factor_pipeline/build_factors.py。建议在项目根目录运行。
"""

import os
import sys
import time

import numpy as np
import pandas as pd

# 项目根目录
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from data.data_config import (
    PRICE_FILE,
    _price_filename,
    FACTOR_RAW_DIR as _DEFAULT_FACTOR_RAW_DIR,
    FACTOR_USE_ADJUSTED_OHLC,
    require_price_file_exists,
    should_use_price_sheet,
)
from qqq_core.data_cache import load_or_compute, print_cache_summary
from qqq_core.excel_io import atomic_excel_writer
from qqq_core.parallel import get_max_workers, ordered_parallel_map

_FACTOR_WRITE_ATTEMPTS = 3
_FACTOR_WRITE_RETRY_DELAY_SECONDS = 0.2

_RUN_DIR = os.environ.get("REBALANCE_RUN_DIR")
if _RUN_DIR:
    EXCEL_PATH = os.path.join(_RUN_DIR, "data", _price_filename())
    FACTOR_RAW_DIR = os.path.join(_RUN_DIR, "factor_raw")
else:
    EXCEL_PATH = PRICE_FILE
    FACTOR_RAW_DIR = _DEFAULT_FACTOR_RAW_DIR


def load_ohlcv_data(excel_path):
    """
    从 Excel 加载 OHLCV 数据，返回 data_dict：
        'open', 'high', 'low', 'close', 'volume' → DataFrame(index=Date, columns=tickers)
    若某列不存在（如 Open/High/Low），则对应 DataFrame 为空（empty）。
    """
    excel_path = require_price_file_exists(excel_path)
    raw = load_or_compute(
        "price_workbook_raw",
        [excel_path],
        {"loader": "build_factors"},
        lambda: pd.read_excel(excel_path, sheet_name=None),
    )

    frames = {k: {} for k in ('open', 'high', 'low', 'close', 'volume')}

    skipped_extra_sheets = []

    for ticker, df in raw.items():
        if not should_use_price_sheet(ticker):
            skipped_extra_sheets.append(ticker)
            continue
        if "Date" not in df.columns:
            continue
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)

        # 收盘价：优先 Adj Close，其次 Close
        if "Adj Close" in df.columns:
            frames['close'][ticker] = df["Adj Close"]
        elif "Close" in df.columns:
            frames['close'][ticker] = df["Close"]

        if "Volume" in df.columns:
            frames['volume'][ticker] = df["Volume"]
        if FACTOR_USE_ADJUSTED_OHLC and "Adj Open" in df.columns:
            frames['open'][ticker] = df["Adj Open"]
        elif "Open" in df.columns:
            frames['open'][ticker] = df["Open"]
        if FACTOR_USE_ADJUSTED_OHLC and "Adj High" in df.columns:
            frames['high'][ticker] = df["Adj High"]
        elif "High" in df.columns:
            frames['high'][ticker] = df["High"]
        if FACTOR_USE_ADJUSTED_OHLC and "Adj Low" in df.columns:
            frames['low'][ticker] = df["Adj Low"]
        elif "Low" in df.columns:
            frames['low'][ticker] = df["Low"]

    result = {}
    for key, col_dict in frames.items():
        if col_dict:
            result[key] = pd.DataFrame(col_dict)
        else:
            result[key] = pd.DataFrame()

    if skipped_extra_sheets:
        preview = ", ".join(skipped_extra_sheets[:10])
        suffix = "..." if len(skipped_extra_sheets) > 10 else ""
        print(
            f"  [universe] skipped {len(skipped_extra_sheets)} Excel sheets "
            f"outside YFINANCE_TICKERS: {preview}{suffix}"
        )

    return result


def build_data_dict(frames):
    """
    在 OHLCV 帧基础上计算衍生数据并返回完整 data_dict：
        'returns' = close.pct_change()
        'vwap'    = (high+low+close)/3  若缺 high/low 则退化为 close
    """
    close = frames['close']
    data_dict = dict(frames)

    data_dict['returns'] = close.pct_change()

    high = frames.get('high', pd.DataFrame())
    low = frames.get('low', pd.DataFrame())
    if not high.empty and not low.empty:
        data_dict['vwap'] = (high + low + close) / 3.0
    else:
        data_dict['vwap'] = close.copy()

    return data_dict


def build_and_save_all_factors(data_dict):
    """
    根据 factor_library 的 FACTOR_CONFIGS 构建因子并保存到 factor_raw。
    每个因子输出一个单 sheet（"factor"）的 Excel 文件。
    若设置了 REBALANCE_SELECTED_FACTORS（逗号分隔因子名），仅构建指定因子。
    """
    from factor_pipeline.factor_library import FACTOR_CONFIGS

    os.makedirs(FACTOR_RAW_DIR, exist_ok=True)
    built = []

    selected = os.environ.get("REBALANCE_SELECTED_FACTORS")
    if selected:
        factor_names = [n.strip() for n in selected.split(",") if n.strip()]
    else:
        factor_names = None

    factor_specs = []
    for name, cfg in FACTOR_CONFIGS.items():
        if factor_names is not None and name not in factor_names:
            continue
        factor_specs.append((name, cfg.get('data_keys', ['close'])))

    worker_count = get_max_workers(len(factor_specs))
    chunk_size = max(1, int(np.ceil(len(factor_specs) / max(1, worker_count)))) if factor_specs else 1
    chunks = [
        factor_specs[i:i + chunk_size]
        for i in range(0, len(factor_specs), chunk_size)
    ]
    tasks = [(chunk, data_dict, FACTOR_RAW_DIR) for chunk in chunks]
    built_chunks = ordered_parallel_map(
        _build_factor_chunk_worker,
        tasks,
        label="build_factors",
    )
    return [item for chunk in built_chunks for item in chunk]

    for name, cfg in FACTOR_CONFIGS.items():
        if factor_names is not None and name not in factor_names:
            continue
        func = cfg['func']
        data_keys = cfg.get('data_keys', ['close'])
        raw_path = os.path.join(FACTOR_RAW_DIR, f"factor_{name}.xlsx")

        # 检查所需数据是否存在
        missing = [k for k in data_keys if data_dict.get(k) is None or (
            isinstance(data_dict[k], pd.DataFrame) and data_dict[k].empty
        )]
        if missing:
            print(f"  [跳过] {name}: 缺少数据 {missing}")
            continue

        try:
            args = [data_dict[k] for k in data_keys]
            factor_df = func(*args)

            if isinstance(factor_df, pd.Series):
                factor_df = factor_df.to_frame()

            factor_df = factor_df.replace([np.inf, -np.inf], np.nan)
            factor_df.index.name = "Date"
            factor_df.to_excel(raw_path, sheet_name="factor")
            built.append((name, raw_path))
            print(f"  {name} -> {raw_path}")
        except Exception as e:
            import traceback
            print(f"  [错误] {name}: {e}")
            traceback.print_exc()

    return built


def _write_factor_excel(factor_df, raw_path):
    for attempt in range(1, _FACTOR_WRITE_ATTEMPTS + 1):
        try:
            with atomic_excel_writer(raw_path, engine="xlsxwriter") as writer:
                factor_df.to_excel(writer, sheet_name="factor")
            return
        except OSError as exc:
            if attempt >= _FACTOR_WRITE_ATTEMPTS:
                raise
            delay = _FACTOR_WRITE_RETRY_DELAY_SECONDS * attempt
            print(
                f"  [retry] Excel write {attempt}/{_FACTOR_WRITE_ATTEMPTS} "
                f"failed for {raw_path}: {exc}; retrying in {delay:.1f}s",
                flush=True,
            )
            time.sleep(delay)


def _build_factor_chunk_worker(task):
    factor_specs, data_dict, factor_raw_dir = task
    from factor_pipeline.factor_library import FACTOR_CONFIGS

    os.makedirs(factor_raw_dir, exist_ok=True)
    built = []

    for name, data_keys in factor_specs:
        cfg = FACTOR_CONFIGS[name]
        func = cfg['func']
        raw_path = os.path.join(factor_raw_dir, f"factor_{name}.xlsx")
        missing = [k for k in data_keys if data_dict.get(k) is None or (
            isinstance(data_dict[k], pd.DataFrame) and data_dict[k].empty
        )]
        if missing:
            print(f"  [skip] {name}: missing data {missing}", flush=True)
            continue

        try:
            args = [data_dict[k] for k in data_keys]
            factor_df = func(*args)

            if isinstance(factor_df, pd.Series):
                factor_df = factor_df.to_frame()

            factor_df = factor_df.replace([np.inf, -np.inf], np.nan)
            factor_df.index.name = "Date"
            _write_factor_excel(factor_df, raw_path)
            built.append((name, raw_path))
            print(f"  {name} -> {raw_path}", flush=True)
        except Exception as e:
            print(f"  [error] {name}: {e}", flush=True)
            raise RuntimeError(
                f"Failed to build or save {name} to {raw_path}: {e}"
            ) from e

    return built


def main():
    print("=" * 60)
    print("Step 1: 加载 OHLCV 数据")
    print("=" * 60)
    frames = load_ohlcv_data(EXCEL_PATH)
    close = frames['close']
    volume = frames['volume']
    print(f"  收盘价: {close.shape}")
    print(f"  成交量: {volume.shape}")
    for key in ('open', 'high', 'low'):
        df = frames.get(key, pd.DataFrame())
        status = df.shape if not df.empty else "（未找到）"
        print(f"  {key:8s}: {status}")

    data_dict = build_data_dict(frames)

    print("\n" + "=" * 60)
    print("Step 2: 构建并保存所有因子 -> factor_raw")
    print("=" * 60)
    factor_list = build_and_save_all_factors(data_dict)
    print_cache_summary()
    print(f"\n共成功构建 {len(factor_list)} 个因子（数据处理请运行 factor_pipeline/process_factors.py）")
    print("\nFactor pipeline finished.")


def run():
    """可导入入口，供 run_rebalance_day --inline 复用。"""
    return main()


if __name__ == "__main__":
    main()
