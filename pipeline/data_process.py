"""
因子数据处理流水线 (pipeline/data_process.py)
=====================================
本模块对 factor_raw 目录下的因子 Excel 文件做横截面「去极值 + 标准化」，并写入 factor_processed 目录。

处理逻辑：
- mad_winsorize(df, n=3)：逐行（每行一个交易日截面）用中位数与 MAD 做去极值，边界为 median ± n*1.4826*MAD。
- zscore_standardize(df)：逐行 Z-score 标准化（均值 0、标准差 1）。
- process_factor_df(df)：对数值列先 MAD 去极值再 Z-score，保留索引与列名。
- process_factor_excel(input_excel, output_excel, reference_excel=None)：读入多 sheet 因子表，可选用 reference_excel 的日期列修复或对齐索引，再调用 process_factor_df 写回。

直接运行本文件时：遍历 factor_raw 中 factor_*.xlsx，输出到 factor_processed 下同名_processed.xlsx，参考日期默认使用 data/us_top100_daily_2023_present.xlsx 第一 sheet 的 Date 列。若某因子 Excel 的数值全为空/NaN/0，则删除该因子输入及对应输出文件，并在结束时 output 标记。
"""

import numpy as np
import pandas as pd
import os

def mad_winsorize(df, n=3):
    """
    横截面（逐日）中位数 + MAD 去极值
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
    保留原始索引和列名
    """
    # 保存原始索引（但不保存列名，因为 select_dtypes 可能会过滤掉非数值列）
    original_index = df.index
    
    # 只保留数值列
    df_numeric = df.select_dtypes(include=[np.number])
    
    # 去极值和标准化
    df_numeric = mad_winsorize(df_numeric)
    df_numeric = zscore_standardize(df_numeric)
    
    # 恢复索引（列名已经在 select_dtypes 后保持正确）
    df_numeric.index = original_index
    
    return df_numeric

def is_factor_all_empty_nan_or_zero(excel_path):
    """
    检查因子 Excel 是否全为空 / NaN / 0。
    若所有 sheet 的数值列均无有效数据（非空且非零），返回 True。
    """
    try:
        sheets = pd.read_excel(excel_path, sheet_name=None, index_col=0)
    except Exception:
        return False
    if not sheets:
        return True
    for sheet_name, df in sheets.items():
        if df.empty:
            continue
        df_num = df.select_dtypes(include=[np.number])
        if df_num.empty:
            continue
        mask_valid = df_num.notna() & (df_num != 0)
        if mask_valid.any().any():
            return False
    return True


def process_factor_excel(input_excel, output_excel, reference_excel=None):
    """
    处理因子文件，可选使用参考文件修复日期
    """
    sheets = pd.read_excel(input_excel, sheet_name=None, index_col=0)

    # 如果提供了参考文件，读取正确的日期
    if reference_excel:
        print(f"使用参考文件修复日期: {reference_excel}")
        ref_data = pd.read_excel(reference_excel, sheet_name=0)
        ref_dates = pd.to_datetime(ref_data['Date'])
    else:
        ref_dates = None

    with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
        for sheet_name, df in sheets.items():
            print(f"  处理 sheet: {sheet_name}")
            print(f"    原始 shape: {df.shape}")

            # 如果有参考日期，使用 reindex 对齐（更安全的方式）
            if ref_dates is not None:
                # 先尝试解析现有索引为日期
                try:
                    df.index = pd.to_datetime(df.index)
                    print(f"    原始日期范围: {df.index.min()} 到 {df.index.max()}")
                except Exception:
                    print(f"    [警告] 无法解析原始索引为日期")

                # 使用 reindex 对齐到参考日期（缺失日期填充 NaN）
                df_aligned = df.reindex(ref_dates)

                # 统计对齐结果
                n_matched = df_aligned.notna().any(axis=1).sum()
                n_missing = len(ref_dates) - n_matched
                print(f"    对齐结果: {n_matched} 个日期有数据, {n_missing} 个日期缺失")

                # 如果匹配率太低，发出警告
                if n_matched < len(df) * 0.8:
                    print(f"    [警告] 匹配率较低 ({n_matched}/{len(df)} = {n_matched/len(df)*100:.1f}%)")
                    print(f"    可能原因: 因子日期与参考日期不匹配")

                df = df_aligned
            else:
                # 尝试解析现有索引为日期
                try:
                    df.index = pd.to_datetime(df.index)
                    print(f"    日期索引已解析")
                except Exception:
                    print(f"    [警告] 无法解析日期，使用原始索引")

            df.index.name = 'Date'

            # 处理因子
            df_processed = process_factor_df(df)

            # 保存，确保日期索引被保留
            df_processed.to_excel(writer, sheet_name=sheet_name)

            print(f"    处理后 shape: {df_processed.shape}")
            if not df_processed.empty:
                print(f"    日期范围: {df_processed.index.min()} 到 {df_processed.index.max()}")

    print(f"  处理完成，保存到: {output_excel}")

if __name__ == "__main__":

    _run_dir = os.environ.get("REBALANCE_RUN_DIR")
    if _run_dir:
        input_dir = os.path.join(_run_dir, "factor_raw")
        output_dir = os.path.join(_run_dir, "factor_processed")
        reference_file = os.path.join(_run_dir, "data", "us_top100_daily_2023_present.xlsx")
    else:
        input_dir = "factor_raw"
        output_dir = "factor_processed"
        reference_file = "data/us_top100_daily_2023_present.xlsx"

    os.makedirs(output_dir, exist_ok=True)

    selected = os.environ.get("REBALANCE_SELECTED_FACTORS")
    if selected:
        factor_names = [n.strip() for n in selected.split(",") if n.strip()]
        files_to_process = [f"factor_{n}.xlsx" for n in factor_names]
    else:
        files_to_process = None
    
    print("=" * 60)
    print("因子数据处理（去极值 + 标准化）")
    print("=" * 60)

    removed_empty_factors = []

    for file in os.listdir(input_dir):
        if file.startswith("factor_") and file.endswith(".xlsx"):
            if files_to_process is not None and file not in files_to_process:
                continue
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(
                output_dir,
                file.replace(".xlsx", "_processed.xlsx")
            )

            # 检查是否全为空/NaN/0，若是则删除并跳过
            if is_factor_all_empty_nan_or_zero(input_path):
                factor_name = file.replace("factor_", "").replace(".xlsx", "")
                removed_empty_factors.append(factor_name)
                print(f"\n[跳过] {file}：因子值全为空/NaN/0，已删除")
                try:
                    os.remove(input_path)
                    print(f"  已删除: {input_path}")
                except OSError as e:
                    print(f"  删除输入文件失败: {e}")
                try:
                    if os.path.exists(output_path):
                        os.remove(output_path)
                        print(f"  已删除: {output_path}")
                except OSError as e:
                    print(f"  删除输出文件失败: {e}")
                continue

            print(f"\n处理 {file} ...")
            
            try:
                process_factor_excel(
                    input_excel=input_path,
                    output_excel=output_path,
                    reference_excel=reference_file
                )
            except Exception as e:
                print(f"  处理失败: {e}")
                import traceback
                traceback.print_exc()
                continue

    print("\n" + "=" * 60)
    print("所有因子文件处理完成")
    if removed_empty_factors:
        print("-" * 60)
        print("已删除的空因子（全为空/NaN/0）:")
        for name in removed_empty_factors:
            print(f"  - {name}")
        print("-" * 60)
    print("=" * 60)