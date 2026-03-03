"""
Rolling Data Processor for Walk-Forward Validation

防止信息泄露的滚动数据处理器。
确保因子处理只使用训练期内的数据，不包含未来信息。

核心功能：
1. 读取原始因子数据，只保留 <= end_date 的数据
2. 应用横截面去极值和标准化（与pipeline/data_process.py相同逻辑）
3. 返回处理后的DataFrame，而不是写入Excel文件
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, List
from datetime import datetime


def mad_winsorize(df, n=3):
    """
    横截面（逐日）中位数 + MAD 去极值

    与pipeline/data_process.py保持一致
    """
    def winsorize_row(row):
        median = row.median()
        mad = (row - median).abs().median()
        if mad == 0 or np.isnan(mad):
            return row
        bound = n * 1.4826 * mad
        upper = median + bound
        lower = median - bound
        return row.clip(lower=lower, upper=upper)
    return df.apply(winsorize_row, axis=1)


def zscore_standardize(df):
    """
    横截面（逐日）Z-score 标准化

    与pipeline/data_process.py保持一致
    """
    def zscore_row(row):
        std = row.std()
        if std == 0 or np.isnan(std):
            return row * 0.0
        return (row - row.mean()) / std
    return df.apply(zscore_row, axis=1)


def process_factor_df(df):
    """
    去极值 → 标准化

    与pipeline/data_process.py保持一致
    """
    original_index = df.index
    df_numeric = df.select_dtypes(include=[np.number])
    df_numeric = mad_winsorize(df_numeric)
    df_numeric = zscore_standardize(df_numeric)
    df_numeric.index = original_index
    return df_numeric


def load_and_process_factor(
    factor_file: str,
    end_date: datetime,
    sheet_name: str = None
) -> pd.DataFrame:
    """
    加载并处理单个因子文件，只使用 <= end_date 的数据

    Args:
        factor_file: 因子文件路径（factor_raw/*.xlsx）
        end_date: 截止日期（只使用此日期之前的数据）
        sheet_name: sheet名称，如果为None则使用第一个sheet

    Returns:
        pd.DataFrame: 处理后的因子数据（Date索引 × 股票列）
    """
    # 读取因子文件
    if sheet_name is None:
        # 读取第一个sheet
        sheets = pd.read_excel(factor_file, sheet_name=None, index_col=0)
        sheet_name = list(sheets.keys())[0]
        df = sheets[sheet_name]
    else:
        df = pd.read_excel(factor_file, sheet_name=sheet_name, index_col=0)

    # 解析日期索引
    df.index = pd.to_datetime(df.index)
    df.index.name = 'Date'

    # 只保留 <= end_date 的数据（防止look-ahead bias）
    df = df[df.index <= end_date]

    if df.empty:
        raise ValueError(f"No data available before {end_date} in {factor_file}")

    # 处理因子（去极值 + 标准化）
    df_processed = process_factor_df(df)

    return df_processed


def process_factors_rolling(
    factor_files: List[str],
    end_date: datetime,
    sheet_name: str = None,
    verbose: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    批量处理多个因子文件，只使用 <= end_date 的数据

    Args:
        factor_files: 因子文件路径列表
        end_date: 截止日期
        sheet_name: sheet名称，如果为None则使用第一个sheet
        verbose: 是否打印处理信息

    Returns:
        Dict[str, pd.DataFrame]: {因子名: 处理后的DataFrame}
    """
    processed_factors = {}

    for factor_file in factor_files:
        factor_name = os.path.basename(factor_file).replace('.xlsx', '').replace('_processed', '')

        if verbose:
            print(f"  Processing {factor_name}...", end=' ')

        try:
            df = load_and_process_factor(factor_file, end_date, sheet_name)
            processed_factors[factor_name] = df

            if verbose:
                print(f"OK ({df.shape[0]} dates, {df.shape[1]} stocks)")

        except Exception as e:
            if verbose:
                print(f"Error: {e}")
            continue

    return processed_factors


def align_factors_to_dates(
    factors: Dict[str, pd.DataFrame],
    target_dates: pd.DatetimeIndex
) -> Dict[str, pd.DataFrame]:
    """
    将因子数据对齐到目标日期

    Args:
        factors: {因子名: DataFrame}
        target_dates: 目标日期索引

    Returns:
        Dict[str, pd.DataFrame]: 对齐后的因子数据
    """
    aligned_factors = {}

    for name, df in factors.items():
        # 使用reindex对齐，缺失日期填充NaN
        df_aligned = df.reindex(target_dates)
        aligned_factors[name] = df_aligned

    return aligned_factors


def get_common_dates(factors: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """
    获取所有因子的共同日期

    Args:
        factors: {因子名: DataFrame}

    Returns:
        pd.DatetimeIndex: 共同日期索引
    """
    if not factors:
        return pd.DatetimeIndex([])

    # 获取第一个因子的日期作为基准
    common_dates = list(factors.values())[0].index

    # 与其他因子求交集
    for df in list(factors.values())[1:]:
        common_dates = common_dates.intersection(df.index)

    return common_dates.sort_values()


if __name__ == "__main__":
    # 测试代码
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from analysis.walk_forward import walk_forward_config as config

    print("=" * 80)
    print("Testing Rolling Data Processor")
    print("=" * 80)

    # 获取选中的因子文件
    factor_files = config.get_selected_factor_files()
    print(f"\nSelected {len(factor_files)} factors:")
    for f in factor_files:
        print(f"  - {os.path.basename(f)}")

    # 测试：处理到2024-07-15的数据
    test_end_date = pd.Timestamp('2024-07-15')
    print(f"\nProcessing factors up to {test_end_date}...")

    processed = process_factors_rolling(factor_files, test_end_date, verbose=True)

    print(f"\nOK Processed {len(processed)} factors")

    # 获取共同日期
    common_dates = get_common_dates(processed)
    print(f"\nCommon dates: {len(common_dates)}")
    print(f"  Date range: {common_dates.min()} to {common_dates.max()}")

    # 显示第一个因子的统计信息
    if processed:
        first_factor = list(processed.keys())[0]
        df = processed[first_factor]
        print(f"\nSample factor: {first_factor}")
        print(f"  Shape: {df.shape}")
        print(f"  Mean: {df.mean().mean():.6f}")
        print(f"  Std: {df.std().mean():.6f}")
        print(f"  NaN ratio: {df.isna().sum().sum() / (df.shape[0] * df.shape[1]):.2%}")
