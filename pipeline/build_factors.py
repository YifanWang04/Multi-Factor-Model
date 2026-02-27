"""
因子构建流水线 (pipeline/build_factors.py)
=====================================
从 factor_library 读取所有因子配置，用 OHLCV 数据构建原始因子并保存到 factor_raw。
不做回测；回测请使用 run_single_factor_test 或 run_all_factors_backtest。

详细说明：
- 数据来源：从 data/us_top100_daily_2023_present.xlsx 读取 OHLCV 数据
  （多 sheet，每 sheet 一只标的，期望列：Date、Open、High、Low、Adj Close、Volume）。
- 衍生数据：returns = close.pct_change()；
  vwap ≈ (high+low+close)/3（典型价格，若无 High/Low 则退化为 close）。
- 因子构建：FACTOR_CONFIGS 中每个因子通过 data_keys 指定所需数据，输出单 sheet（"factor"）的原始因子文件。
- 数据处理：去极值与标准化由 pipeline/data_process.py 单独执行，输出到 factor_processed。

命令行：python pipeline/build_factors.py。建议在项目根目录运行。
"""

import os
import sys

import numpy as np
import pandas as pd

# 项目根目录
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

EXCEL_PATH = os.path.join(_PROJECT_ROOT, "data", "us_top100_daily_2023_present.xlsx")
FACTOR_RAW_DIR = os.path.join(_PROJECT_ROOT, "factor_raw")


def load_ohlcv_data(excel_path):
    """
    从 Excel 加载 OHLCV 数据，返回 data_dict：
        'open', 'high', 'low', 'close', 'volume' → DataFrame(index=Date, columns=tickers)
    若某列不存在（如 Open/High/Low），则对应 DataFrame 为空（empty）。
    """
    raw = pd.read_excel(excel_path, sheet_name=None)

    frames = {k: {} for k in ('open', 'high', 'low', 'close', 'volume')}

    for ticker, df in raw.items():
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
        if "Open" in df.columns:
            frames['open'][ticker] = df["Open"]
        if "High" in df.columns:
            frames['high'][ticker] = df["High"]
        if "Low" in df.columns:
            frames['low'][ticker] = df["Low"]

    result = {}
    for key, col_dict in frames.items():
        if col_dict:
            result[key] = pd.DataFrame(col_dict)
        else:
            result[key] = pd.DataFrame()

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
    """
    from factors.factor_library import FACTOR_CONFIGS

    os.makedirs(FACTOR_RAW_DIR, exist_ok=True)
    built = []

    for name, cfg in FACTOR_CONFIGS.items():
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
    print(f"\n共成功构建 {len(factor_list)} 个因子（数据处理请运行 pipeline/data_process.py）")
    print("\nFactor pipeline finished.")


if __name__ == "__main__":
    main()
