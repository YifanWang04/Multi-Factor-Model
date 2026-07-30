"""
因子数据处理流水线 (factor_pipeline/process_factors.py)
=====================================
本模块对 factor_raw 目录下的因子 Excel 文件做横截面「去极值 + 标准化」，并写入 factor_processed 目录。

处理逻辑：
- mad_winsorize(df, n=3)：逐行（每行一个交易日截面）用中位数与 MAD 做去极值，边界为 median ± n*1.4826*MAD。
- zscore_standardize(df)：逐行 Z-score 标准化（均值 0、标准差 1）。
- process_factor_df(df)：对数值列先 MAD 去极值再 Z-score，保留索引与列名。
- process_factor_excel(input_excel, output_excel, reference_excel=None)：读入多 sheet 因子表，可选用 reference_excel 的日期列修复或对齐索引，再调用 process_factor_df 写回。

直接运行本文件时：遍历 factor_raw 中 factor_*.xlsx，输出到 factor_processed 下同名_processed.xlsx，参考日期使用 data_config.PRICE_FILE。若某因子 Excel 的数值全为空/NaN/0，则跳过该因子并在结束时写出 manifest。
"""

import os
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd

from data.data_config import (
    PRICE_FILE,
    _price_filename,
    FACTOR_RAW_DIR,
    FACTOR_PROCESSED_DIR,
    require_price_file_exists,
)
from qqq_core.excel_io import atomic_excel_writer
from qqq_core.parallel import ordered_parallel_map

_PROCESSED_WRITE_ATTEMPTS = 3
_PROCESSED_WRITE_RETRY_DELAY_SECONDS = 0.2

def mad_winsorize(df, n=3):
    """
    横截面（逐日）中位数 + MAD 去极值
    """
    median = df.median(axis=1)
    mad = df.sub(median, axis=0).abs().median(axis=1)
    invalid_mad = mad.isna() | mad.eq(0)
    bound = n * 1.4826 * mad
    lower = median - bound
    upper = median + bound
    result = df.copy()
    valid_mad = ~invalid_mad
    if valid_mad.any():
        result.loc[valid_mad] = df.loc[valid_mad].clip(
            lower=lower.loc[valid_mad],
            upper=upper.loc[valid_mad],
            axis=0,
        )
    # 统一将 inf 替换为 nan，避免后续统计计算异常
    return result.replace([np.inf, -np.inf], np.nan)

def zscore_standardize(df):
    """
    横截面（逐日）Z-score 标准化
    """
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    invalid_std = std.isna() | std.eq(0)
    result = df.sub(mean, axis=0).div(std.replace(0, np.nan), axis=0)
    if invalid_std.any():
        result.loc[invalid_std] = df.loc[invalid_std] * 0.0
    return result.replace([np.inf, -np.inf], np.nan)

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


def _load_reference_dates(reference_excel):
    ref_data = pd.read_excel(reference_excel, sheet_name=0)
    return pd.DatetimeIndex(pd.to_datetime(ref_data["Date"]))


def _write_processed_sheets(processed_sheets, output_excel):
    for attempt in range(1, _PROCESSED_WRITE_ATTEMPTS + 1):
        try:
            with atomic_excel_writer(output_excel, engine="xlsxwriter") as writer:
                for sheet_name, df_processed in processed_sheets.items():
                    df_processed.to_excel(writer, sheet_name=sheet_name)
            return
        except OSError as exc:
            if attempt >= _PROCESSED_WRITE_ATTEMPTS:
                raise
            delay = _PROCESSED_WRITE_RETRY_DELAY_SECONDS * attempt
            print(
                f"  [retry] Excel write {attempt}/{_PROCESSED_WRITE_ATTEMPTS} "
                f"failed for {output_excel}: {exc}; retrying in {delay:.1f}s",
                flush=True,
            )
            time.sleep(delay)


def process_factor_excel(
    input_excel,
    output_excel,
    reference_excel=None,
    reference_dates=None,
):
    """
    处理因子文件，可选使用参考文件修复日期
    """
    sheets = pd.read_excel(input_excel, sheet_name=None, index_col=0)

    # 如果提供了参考文件，读取正确的日期
    if reference_dates is not None:
        ref_dates = pd.DatetimeIndex(pd.to_datetime(reference_dates))
        if reference_excel:
            print(f"使用参考文件修复日期: {reference_excel}")
    elif reference_excel:
        print(f"使用参考文件修复日期: {reference_excel}")
        ref_dates = _load_reference_dates(reference_excel)
    else:
        ref_dates = None

    processed_sheets = {}
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
        processed_sheets[sheet_name] = df_processed

        print(f"    处理后 shape: {df_processed.shape}")
        if not df_processed.empty:
            print(f"    日期范围: {df_processed.index.min()} 到 {df_processed.index.max()}")

    _write_processed_sheets(processed_sheets, output_excel)

    print(f"  处理完成，保存到: {output_excel}")


def _process_factor_file_task(task):
    file, input_dir, output_dir, reference_file, reference_dates = task
    input_path = os.path.join(input_dir, file)
    output_path = os.path.join(
        output_dir,
        file.replace(".xlsx", "_processed.xlsx")
    )

    if is_factor_all_empty_nan_or_zero(input_path):
        factor_name = file.replace("factor_", "").replace(".xlsx", "")
        print(f"\n[skip] {file}: all empty/NaN/zero", flush=True)
        return {
            "status": "skipped_empty",
            "factor_name": factor_name,
            "input_path": input_path,
            "output_path": output_path,
            "reason": "all_empty_nan_or_zero",
        }

    print(f"\n处理 {file} ...", flush=True)
    try:
        process_factor_excel(
            input_excel=input_path,
            output_excel=output_path,
            reference_excel=reference_file,
            reference_dates=reference_dates,
        )
        return {
            "status": "processed",
            "factor_name": file.replace("factor_", "").replace(".xlsx", ""),
            "input_path": input_path,
            "output_path": output_path,
        }
    except Exception as e:
        import traceback
        print(f"  处理失败: {e}", flush=True)
        traceback.print_exc()
        return {
            "status": "failed",
            "factor_name": file.replace("factor_", "").replace(".xlsx", ""),
            "input_path": input_path,
            "output_path": output_path,
            "reason": str(e),
        }

if __name__ == "__main__":

    _run_dir = os.environ.get("REBALANCE_RUN_DIR")
    if _run_dir:
        input_dir = os.path.join(_run_dir, "factor_raw")
        output_dir = os.path.join(_run_dir, "factor_processed")
        reference_file = os.path.join(_run_dir, "data", _price_filename())
    else:
        input_dir = FACTOR_RAW_DIR
        output_dir = FACTOR_PROCESSED_DIR
        reference_file = require_price_file_exists(PRICE_FILE)

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

    skipped_empty_factors = []
    files = [
        file for file in sorted(os.listdir(input_dir))
        if file.startswith("factor_")
        and file.endswith(".xlsx")
        and (files_to_process is None or file in files_to_process)
    ]
    reference_dates = _load_reference_dates(reference_file)
    tasks = [
        (file, input_dir, output_dir, reference_file, reference_dates)
        for file in files
    ]
    results = ordered_parallel_map(
        _process_factor_file_task,
        tasks,
        label="data_process",
    )
    skipped_empty_factors = [
        {
            "factor_name": rec["factor_name"],
            "input_path": rec["input_path"],
            "output_path": rec["output_path"],
            "reason": rec.get("reason", "all_empty_nan_or_zero"),
        }
        for rec in results
        if rec.get("status") == "skipped_empty"
    ]
    failed_factors = [rec for rec in results if rec.get("status") == "failed"]
    if failed_factors:
        print("-" * 60)
        print("处理失败的因子：")
        for rec in failed_factors:
            print(f"  - {rec['factor_name']}: {rec.get('reason', '')}")
        print("-" * 60)

    print("\n" + "=" * 60)
    if failed_factors:
        print("因子文件处理结束（存在失败）")
    else:
        print("所有因子文件处理完成")
    if skipped_empty_factors:
        print("-" * 60)
        print("已跳过的空因子（全为空/NaN/0，未删除）:")
        for rec in skipped_empty_factors:
            print(f"  - {rec['factor_name']}")
        manifest_path = os.path.join(output_dir, "factor_processing_skip_manifest.csv")
        pd.DataFrame(skipped_empty_factors).to_csv(manifest_path, index=False, encoding="utf-8-sig")
        print(f"Skip manifest: {manifest_path}")
        print("-" * 60)
    print("=" * 60)

    if failed_factors:
        failed_names = ", ".join(rec["factor_name"] for rec in failed_factors)
        raise RuntimeError(
            f"Factor processing failed after retries: {failed_names}"
        )


def main():
    """可导入入口；通过脚本路径执行历史主流程。"""
    import runpy
    return runpy.run_path(__file__, run_name="__main__")


def run():
    """可导入入口，供 run_rebalance_day --inline 复用。"""
    return main()
