"""
批量单因子回测入口 (run_all_factors_backtest.py)
=====================================
本模块扫描 factor_processed 目录下所有因子 Excel 文件（优先 *_processed.xlsx，否则 factor_*.xlsx），
对每个因子调用 SingleFactorTesterOptimized 运行完整单因子测试并生成独立 PDF 报告。

行为说明：
- 因子显示名由文件名推导：如 factor_momentum_processed.xlsx → "momentum"（去掉 factor_ 与 _processed，下划线转空格）。
- 每个因子使用与 SingleFactorConfig 相同的 REBALANCE_PERIODS、GROUP_NUM、WEIGHT_METHOD、交易成本等，仅 FACTOR_FILE 与 FACTOR_NAME 按文件替换。
- 支持环境变量 FACTOR_PROCESSED_DIR 覆盖因子目录；默认为项目下的 factor_processed。

运行方式：在项目根目录执行 python -m analysis.single_factor.run_all_factors_backtest，或在 analysis/single_factor 下执行 python run_all_factors_backtest.py。结束时打印成功/失败汇总。
"""

import os
import sys
from types import SimpleNamespace

# 保证能导入同目录模块与 config
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import SingleFactorConfig
from run_single_factor_test import SingleFactorTesterOptimized


FACTOR_PROCESSED_DIR = os.path.join(_PROJECT_ROOT, "factor_processed")
# 匹配 factor_xxx_processed.xlsx 或 factor_xxx.xlsx
PROCESSED_PATTERN = "_processed.xlsx"
FALLBACK_PATTERN = ".xlsx"


def _factor_display_name(filepath):
    """从文件名推出报告用因子名。如 factor_momentum_processed.xlsx -> Momentum"""
    basename = os.path.splitext(os.path.basename(filepath))[0]
    if basename.endswith("_processed"):
        basename = basename[: -len("_processed")]
    if basename.lower().startswith("factor_"):
        basename = basename[7:]  # 去掉 factor_
    return basename.replace("_", " ").strip() or basename


def get_factor_files(factor_dir):
    """
    扫描 factor_processed 目录，返回 (因子名, 绝对路径) 列表。
    优先匹配 *_processed.xlsx，否则匹配 factor_*.xlsx。
    """
    if not os.path.isdir(factor_dir):
        return []
    out = []
    seen = set()
    for f in sorted(os.listdir(factor_dir)):
        if not f.endswith(".xlsx"):
            continue
        full = os.path.join(factor_dir, f)
        if not os.path.isfile(full):
            continue
        # 避免重复：既有 factor_x_processed.xlsx 又有 factor_x.xlsx 时只取 processed
        base = f.replace("_processed.xlsx", "").replace(".xlsx", "")
        if base in seen:
            continue
        if f.endswith("_processed.xlsx"):
            seen.add(base)
            out.append((_factor_display_name(full), full))
        elif f.startswith("factor_") and f.endswith(".xlsx") and base not in seen:
            seen.add(base)
            out.append((_factor_display_name(full), full))
    return out


def make_config_for_factor(factor_file_path, factor_display_name, project_root=None):
    """基于 SingleFactorConfig 生成单因子回测配置，仅替换因子文件与名称。"""
    root = project_root or _PROJECT_ROOT
    c = SingleFactorConfig
    return SimpleNamespace(
        PROJECT_ROOT=root,
        FACTOR_FILE=os.path.abspath(factor_file_path),
        PRICE_FILE=os.path.join(root, "data", "us_top100_daily_2023_present.xlsx"),
        RETURN_COLUMN=getattr(c, 'RETURN_COLUMN', 'Return'),
        OUTPUT_DIR=os.path.join(root, "output", "single_factor_reports"),
        REBALANCE_PERIODS=c.REBALANCE_PERIODS,
        GROUP_NUM=c.GROUP_NUM,
        WEIGHT_METHOD=c.WEIGHT_METHOD,
        RISK_FREE_RATE=c.RISK_FREE_RATE,
        TRANSACTION_COST=c.TRANSACTION_COST,
        FIGURE_SIZE=c.FIGURE_SIZE,
        FACTOR_NAME=factor_display_name,
    )


def run_one(config):
    """对给定 config 跑一次完整单因子回测。"""
    tester = SingleFactorTesterOptimized(config=config)
    tester.run()


def main():
    factor_dir = os.path.abspath(
        os.environ.get("FACTOR_PROCESSED_DIR", FACTOR_PROCESSED_DIR)
    )
    print("=" * 60)
    print("批量单因子回测（factor_processed）")
    print("=" * 60)
    print(f"因子目录: {factor_dir}")

    factor_list = get_factor_files(factor_dir)
    if not factor_list:
        print("未找到任何因子文件（期望 factor_*_processed.xlsx 或 factor_*.xlsx）")
        return

    print(f"共 {len(factor_list)} 个因子")
    for i, (name, path) in enumerate(factor_list, 1):
        print(f"  [{i}] {name} <- {os.path.basename(path)}")

    print("\n" + "=" * 60)
    results = []
    for i, (display_name, path) in enumerate(factor_list, 1):
        print(f"\n>>> [{i}/{len(factor_list)}] 回测: {display_name}")
        try:
            config = make_config_for_factor(path, display_name, _PROJECT_ROOT)
            run_one(config)
            results.append((display_name, True, None))
        except Exception as e:
            import traceback
            results.append((display_name, False, traceback.format_exc()))
            print(f"  [FAIL] {e}")

    print("\n" + "=" * 60)
    print("汇总")
    print("=" * 60)
    ok = [n for n, s, _ in results if s]
    fail = [(n, e) for n, s, e in results if not s]
    print(f"成功: {len(ok)}")
    for n in ok:
        print(f"  [OK] {n}")
    if fail:
        print(f"失败: {len(fail)}")
        for n, err in fail:
            print(f"  [FAIL] {n}")
            if err:
                print(err[:400] + ("..." if len(err) > 400 else ""))
    print("\n批量回测结束。")


if __name__ == "__main__":
    main()
