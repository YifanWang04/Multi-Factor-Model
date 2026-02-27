"""
因子数据处理流水线 (pipeline/data_process.py)
=====================================
本模块对 factor_raw 目录下的因子 Excel 文件做横截面「去极值 + 标准化」，并写入 factor_processed 目录。

处理逻辑：
- mad_winsorize(df, n=3)：逐行（每行一个交易日截面）用中位数与 MAD 做去极值，边界为 median ± n*1.4826*MAD。
- zscore_standardize(df)：逐行 Z-score 标准化（均值 0、标准差 1）。
- process_factor_df(df)：对数值列先 MAD 去极值再 Z-score，保留索引与列名。
- process_factor_excel(input_excel, output_excel, reference_excel=None)：读入多 sheet 因子表，可选用 reference_excel 的日期列修复或对齐索引，再调用 process_factor_df 写回。

直接运行本文件时：遍历 factor_raw 中 factor_*.xlsx，输出到 factor_processed 下同名_processed.xlsx，参考日期默认使用 data/us_top100_daily_2023_present.xlsx 第一 sheet 的 Date 列。
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

# 处理单因子Excel文件
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
    
    writer = pd.ExcelWriter(output_excel, engine='xlsxwriter')
    
    for sheet_name, df in sheets.items():
        print(f"  处理 sheet: {sheet_name}")
        print(f"    原始 shape: {df.shape}")
        
        # 如果有参考日期且行数匹配，使用参考日期
        if ref_dates is not None:
            if len(df) == len(ref_dates):
                print(f"    修复日期索引 (完全匹配)")
                df.index = ref_dates.values
            elif len(df) < len(ref_dates):
                print(f"    修复日期索引 (因子行数较少: {len(df)} < {len(ref_dates)})")
                df.index = ref_dates[:len(df)].values
            else:
                print(f"    ⚠️ 警告: 因子行数多于参考日期 ({len(df)} > {len(ref_dates)})")
                # 截断因子数据
                df = df.iloc[:len(ref_dates)]
                df.index = ref_dates.values
        else:
            # 尝试解析现有索引为日期
            try:
                df.index = pd.to_datetime(df.index)
                print(f"    日期索引已解析")
            except:
                print(f"    ⚠️ 警告: 无法解析日期，使用原始索引")
        
        df.index.name = 'Date'
        
        # 处理因子
        df_processed = process_factor_df(df)
        
        # 保存，确保日期索引被保留
        df_processed.to_excel(writer, sheet_name=sheet_name)
        
        print(f"    处理后 shape: {df_processed.shape}")
        print(f"    日期范围: {df_processed.index.min()} 到 {df_processed.index.max()}")

    writer.close()
    print(f"✓ 处理完成，保存到: {output_excel}")

if __name__ == "__main__":

    input_dir = "factor_raw"
    output_dir = "factor_processed"
    reference_file = "data/us_top100_daily_2023_present.xlsx"

    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("因子数据处理（去极值 + 标准化）")
    print("=" * 60)

    for file in os.listdir(input_dir):
        if file.startswith("factor_") and file.endswith(".xlsx"):
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(
                output_dir,
                file.replace(".xlsx", "_processed.xlsx")
            )

            print(f"\n处理 {file} ...")
            
            try:
                process_factor_excel(
                    input_excel=input_path,
                    output_excel=output_path,
                    reference_excel=reference_file
                )
            except Exception as e:
                print(f"❌ 处理失败: {e}")
                import traceback
                traceback.print_exc()
                continue

    print("\n" + "=" * 60)
    print("所有因子文件处理完成")
    print("=" * 60)