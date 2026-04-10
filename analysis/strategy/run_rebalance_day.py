"""
调仓日全流程与报表 (run_rebalance_day.py)
============================================
完整 Pipeline：pull_data → build_factors → data_process → run_composite_factor
使用固定策略参数生成持仓，输出调仓日判定、当前调仓日操作及未来调仓日列表。

所有输出保存至带日期时间的独立文件夹：output/rebalance_day_YYYY-MM-DD_HHMMSS/

职责分工（SRP 原则）：
  - run_rebalance_day.py  — 流程编排器（Pipeline 执行 + 各阶段串联）
  - rebalance/rebalance_operations.py — 调仓日判定、操作明细、实时价格
  - rebalance/market_value.py       — 市值重估（MTM）
  - rebalance/discord_notifier.py    — Discord 通知
  - rebalance/rebalance_report.py    — Excel 报表生成

时序约定（与 README.md 一致）：
  - 交易：T 日收盘执行，买卖价格均使用 Adj Close（收盘价）
  - 调仓日且未收盘时：用当日开盘价（Today_Open）与现价（收盘价估计）替代买入价
  - 持仓区间：(T, T_next]，T 日收益不计入当期持仓
  - 报表/Discord：若下一调仓日尚未到（或卖出价缺失），用 As_Of 日收盘价或实时价作假设卖出价，
    重算 Period_Return / Sell_Value（列 Sell_Price_Source 标明来源）

用法（项目根目录）：
  python analysis/strategy/run_rebalance_day.py
  python analysis/strategy/run_rebalance_day.py --no-discord  # 不发送 Discord 通知
  python analysis/strategy/run_rebalance_day.py --inline       # Pipeline 在同一进程中执行（更快）
"""

from __future__ import annotations

import os
import sys
import io
import subprocess
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

# 设置 UTF-8 输出（Windows 兼容）
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── 路径注册 ─────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ANALYSIS = os.path.dirname(os.path.dirname(_HERE))
_SF_DIR = os.path.join(_ANALYSIS, "single_factor")
_MF_DIR = os.path.join(_ANALYSIS, "multi_factor")
_ROOT = os.path.dirname(_ANALYSIS)

for _p in (_HERE, _SF_DIR, _MF_DIR, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── 子模块导入（职责已拆分）──────────────────────────────────────────
from rebalance import (
    get_rebalance_day_status,
    get_current_rebalance_operations,
    apply_live_prices_to_operations,
    collect_live_prices_for_mtm,
    MarkToMarket,
    patch_period_summary_from_mtm,
    send_discord_notification,
    write_rebalance_day_report,
)

# ── 回测与配置导入 ──────────────────────────────────────────────────
from run_strategy import load_return_data as _load_ret_data
from run_detailed_backtest_report import run_detailed_backtest, parse_strategy_param
from strategy_backtest import _select_rebalance_dates
import strategy_config as cfg
from data.data_config import (
    DATA_START_OFFSET_DAYS,
    _price_filename,
    COMPOSITE_FACTOR_OUTPUT_DIR,
    COMPOSITE_FACTOR_FILE as _COMPOSITE_FACTOR_FILE,
)
from strategy_utils import (
    load_price_data as _load_price_data,
    load_composite_factor_with_fallback,
    composite_factors_path,
)


# ---------------------------------------------------------------------------
# 全局常量
# ---------------------------------------------------------------------------

PIPELINE_SUBPROCESS_TIMEOUT: int = 600


# ---------------------------------------------------------------------------
# 配置（本脚本独立配置，策略相关参数从 strategy_config 派生）
# ---------------------------------------------------------------------------

PROJECT_ROOT = r"D:\qqq"
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "output")

COMPOSITE_FACTOR_SHEET = "ic_m3_N20"

# MANUALLY_SELECTED_FACTOR_INDICES = [95, 101, 62, 65, 32]  # 3/17
MANUALLY_SELECTED_FACTOR_INDICES = [95, 24, 64, 65, 32]  # 3/25 

STRATEGY_PARAM = "max_return_10G_Top1_P20d"  # 3/25
# STRATEGY_PARAM = "max_return_5G_Top1_P10d"  # 3/17

SELECTED_FACTOR_INDICES = MANUALLY_SELECTED_FACTOR_INDICES
SELECTED_FACTOR_NAMES = [f"alpha{i:03d}" for i in SELECTED_FACTOR_INDICES]

_parsed = parse_strategy_param(STRATEGY_PARAM)
STRATEGY_PARAMS = {
    "weight_method": _parsed[0],
    "group_num": _parsed[1],
    "target_rank": _parsed[2],
    "rebalance_period": _parsed[3],
}


# ---------------------------------------------------------------------------
# Pipeline 执行
# ---------------------------------------------------------------------------

def _run_pipeline_inline(run_dir: str, skip_pull: bool = False) -> None:
    """在同一进程中依次执行 pipeline 各步骤（避免 subprocess 进程启动开销）。"""
    from pipeline.build_factors import run as run_build_factors
    from pipeline.data_process import run as run_data_process
    from analysis.multi_factor.run_composite_factor import main as run_composite

    data_dir = os.path.join(run_dir, "data")
    factor_raw_dir = os.path.join(run_dir, "factor_raw")
    factor_processed_dir = os.path.join(run_dir, "factor_processed")
    composite_dir = os.path.join(run_dir, "composite_factor_reports")
    for d in (data_dir, factor_raw_dir, factor_processed_dir, composite_dir):
        os.makedirs(d, exist_ok=True)

    if not skip_pull:
        print("[Pipeline] 拉取行情数据...")
        from data import pull_yhfinance_Data
        pull_yhfinance_Data.main()

    print("[Pipeline] 构建因子...")
    run_build_factors()

    print("[Pipeline] 因子数据处理...")
    run_data_process()

    print("[Pipeline] 因子复合...")
    run_composite()


def _run_pipeline_subprocess(run_dir: str, skip_pull: bool = False) -> None:
    """通过 subprocess 依次调用 pipeline 各步骤（stdout/stderr 实时流式打印）。"""
    import shutil
    import subprocess as sp

    env = os.environ.copy()
    env["REBALANCE_RUN_DIR"] = run_dir
    env["REBALANCE_SELECTED_FACTORS"] = ",".join(SELECTED_FACTOR_NAMES)
    env["REBALANCE_SELECTED_FACTOR_INDICES"] = ",".join(str(i) for i in SELECTED_FACTOR_INDICES)
    env["REBALANCE_SELECTED_COMPOSITE"] = COMPOSITE_FACTOR_SHEET

    data_dir = os.path.join(run_dir, "data")
    for sub_dir in (data_dir, os.path.join(run_dir, "factor_raw"),
                    os.path.join(run_dir, "factor_processed"),
                    os.path.join(run_dir, "composite_factor_reports")):
        os.makedirs(sub_dir, exist_ok=True)

    if skip_pull:
        from data.data_config import PRICE_FILE as _src_price, _price_filename
        src = _src_price
        dst = os.path.join(data_dir, _price_filename())
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            print(f"  已复制数据至: {dst}")
        else:
            raise FileNotFoundError(
                f"skip_pull 时需存在 {src}，请先运行 pull 或确保已在 data/data_config.py 中设好 DATA_START_OFFSET_DAYS"
            )

    steps = []
    if not skip_pull:
        steps.append(("data/pull_yhfinance_Data.py", "拉取行情数据"))
    steps.extend([
        ("pipeline/build_factors.py", "构建因子"),
        ("pipeline/data_process.py", "因子数据处理"),
        ("analysis/multi_factor/run_composite_factor.py", "因子复合"),
    ])

    for i, (script, desc) in enumerate(steps, 1):
        print(f"[Pipeline {i}/{len(steps)}] {desc}...", flush=True)
        proc = sp.Popen(
            [sys.executable, script],
            cwd=PROJECT_ROOT,
            env=env,
            stdout=sp.PIPE,
            stderr=sp.STDOUT,
            text=True,
        )
        if proc.stdout:
            for line in iter(proc.stdout.readline, ""):
                if line:
                    print(line, end="", flush=True)
        returncode = proc.wait()

        if returncode != 0:
            raise RuntimeError(f"Pipeline 步骤失败 [{returncode}]: {script}")
        print(f"  ✅ {desc} 完成\n", flush=True)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _get_run_dir(run_dir_arg: Optional[str], skip_pipeline: bool) -> str:
    """获取本次运行的输出目录。"""
    if run_dir_arg:
        return os.path.abspath(run_dir_arg)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return os.path.join(OUTPUT_BASE, f"rebalance_day_{ts}")


def _sync_composite_factor_to_standard(run_dir: str, sheet: str) -> None:
    """
    将 Pipeline 生成的复合因子同步到标准路径，使 run_detailed_backtest_report.py 使用最新数据。
    原子写保护：先写临时文件，再 os.replace() 原子替换。
    """
    import openpyxl

    src = composite_factors_path(run_dir, SELECTED_FACTOR_INDICES)
    dst = composite_factors_path(COMPOSITE_FACTOR_OUTPUT_DIR, SELECTED_FACTOR_INDICES)

    if not os.path.isfile(src):
        print(f"  [同步跳过] 源文件不存在: {src}")
        return

    try:
        src_df = pd.read_excel(src, sheet_name=sheet, index_col=0)
        src_df.index = pd.to_datetime(src_df.index)
        src_df = src_df.apply(pd.to_numeric, errors="coerce")
        os.makedirs(os.path.dirname(dst), exist_ok=True)

        tmp_path = dst + ".tmp"
        with pd.ExcelWriter(tmp_path, engine="openpyxl") as writer:
            src_df.to_excel(writer, sheet_name=sheet)
        os.replace(tmp_path, dst)

        print(f"  [同步完成] 复合因子 {sheet} 已更新至: {dst}")
        print(f"             因子日期范围: {src_df.index[0].date()} ~ {src_df.index[-1].date()}")
    except Exception as e:
        print(f"  [同步警告] 同步复合因子失败（不影响本次调仓报表）: {e}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main(
    skip_pipeline: bool = False,
    skip_pull: bool = False,
    run_dir_arg: Optional[str] = None,
    send_discord: bool = True,
    inline_pipeline: bool = False,
) -> None:
    """
    调仓日全流程编排器。

    Parameters
    ----------
    skip_pipeline : bool
        若为 True，跳过 pipeline，从 run_dir_arg 或默认路径读取数据
    skip_pull : bool
        pipeline 中是否跳过 pull_data
    run_dir_arg : str, optional
        指定运行目录
    send_discord : bool
        是否发送 Discord 通知
    inline_pipeline : bool
        若为 True，pipeline 在同一进程中执行（更快）
    """
    run_dir = _get_run_dir(run_dir_arg, skip_pipeline)
    os.makedirs(run_dir, exist_ok=True)

    # 确定复合因子文件和价格文件路径
    if skip_pipeline:
        if run_dir_arg:
            composite_file = composite_factors_path(run_dir, SELECTED_FACTOR_INDICES)
            price_file = os.path.join(run_dir, "data", _price_filename())
        else:
            composite_file = _COMPOSITE_FACTOR_FILE
            price_file = os.path.join(PROJECT_ROOT, "data", _price_filename())
    else:
        composite_file = composite_factors_path(run_dir, SELECTED_FACTOR_INDICES)
        price_file = os.path.join(run_dir, "data", _price_filename())

    print("=" * 64)
    print("  调仓日全流程与报表")
    print(f"  输出目录: {run_dir}")
    print(f"  策略参数: {STRATEGY_PARAM} | 价格: Adj Close（收盘价）")
    print(f"  Pipeline 模式: {'内联（inline）' if inline_pipeline else '子进程（subprocess）'}")
    print("=" * 64)

    # 阶段 1：执行 Pipeline
    if not skip_pipeline:
        print("\n[阶段 1] 执行 Pipeline...")
        if inline_pipeline:
            _run_pipeline_inline(run_dir, skip_pull=skip_pull)
        else:
            _run_pipeline_subprocess(run_dir, skip_pull=skip_pull)
        _sync_composite_factor_to_standard(run_dir=run_dir, sheet=COMPOSITE_FACTOR_SHEET)
    else:
        print("\n[阶段 1] 跳过 Pipeline")

    # 阶段 2：加载数据
    print("\n[阶段 2] 加载复合因子与收益率...")
    factor_df = load_composite_factor_with_fallback(composite_file, COMPOSITE_FACTOR_SHEET, _COMPOSITE_FACTOR_FILE)
    ret_df = _load_ret_data(price_file, cfg.RETURN_COLUMN)
    ret_df.sort_index(inplace=True)
    price_df = _load_price_data(price_file, "Adj Close")

    # 阶段 3：运行策略回测
    print(f"\n[阶段 3] 运行策略回测（{STRATEGY_PARAM}）...")
    result = run_detailed_backtest(
        factor_df=factor_df,
        ret_df=ret_df,
        price_df=price_df,
        group_num=STRATEGY_PARAMS["group_num"],
        target_rank=STRATEGY_PARAMS["target_rank"],
        rebalance_period=STRATEGY_PARAMS["rebalance_period"],
        weight_method=STRATEGY_PARAMS["weight_method"],
        config=cfg,
    )

    if "error" in result:
        print(f"错误: {result['error']}")
        return

    result["_factor_df"] = factor_df
    result["_ret_df"] = ret_df
    result["_price_df"] = price_df
    result["_config"] = cfg

    as_of_date = pd.Timestamp(datetime.now().date())

    # 阶段 3b：市值重估（MTM Round 1）
    print("\n[阶段 3b] 市值重估（MTM Round 1）...")
    mtm_live = collect_live_prices_for_mtm(result["operations_df"], price_df, as_of_date)
    mtm = MarkToMarket(result["operations_df"], price_df, as_of_date)
    mtm.apply(live_prices=mtm_live)
    result["operations_df"] = mtm.operations_df
    patch_period_summary_from_mtm(result, mtm.operations_df, as_of_date)
    mtm_applied = mtm.was_applied()

    # 调仓日判定
    rebalance_dates = _select_rebalance_dates(
        factor_df.index,
        ret_df.index,
        STRATEGY_PARAMS["rebalance_period"],
    )
    last_factor_date = factor_df.index[-1]

    status = get_rebalance_day_status(
        rebalance_dates=rebalance_dates,
        rebalance_period=STRATEGY_PARAMS["rebalance_period"],
        as_of_date=as_of_date,
        last_factor_date=last_factor_date,
        trading_dates=ret_df.index.tolist(),
    )

    current_rb_date = status.get("current_rebalance_date")
    next_rb_date = status.get("next_rebalance_date")
    current_ops = pd.DataFrame()
    used_live_prices = False

    # 阶段 3c：获取当前调仓日操作明细（含盘中实时价格 + MTM Round 2）
    if current_rb_date is not None:
        current_ops = get_current_rebalance_operations(
            result, current_rb_date,
            next_rebalance_date=next_rb_date,
            rebalance_dates=rebalance_dates,
            strategy_params=STRATEGY_PARAMS,
        )
        current_ops, used_live_prices = apply_live_prices_to_operations(
            current_ops, price_df, current_rb_date, as_of_date
        )
        # 调仓日盘中更新买入价后，对 Current_Operations 再跑一次 MTM（MTM Round 2）
        if not current_ops.empty:
            live_co = collect_live_prices_for_mtm(current_ops, price_df, as_of_date)
            mtm_co = MarkToMarket(current_ops, price_df, as_of_date)
            mtm_co.apply(live_prices=live_co)
            current_ops = mtm_co.operations_df
            mtm_applied = mtm_applied or mtm_co.was_applied()
        patch_period_summary_from_mtm(result, mtm.operations_df, as_of_date)

    # 阶段 4：写入 Excel 报表
    print("\n[阶段 4] 生成调仓日报表...")
    output_path = os.path.join(run_dir, "rebalance_day_report.xlsx")
    write_rebalance_day_report(
        result=result,
        status=status,
        current_ops=current_ops,
        output_path=output_path,
        used_live_prices=used_live_prices,
        mtm_applied=mtm_applied,
        strategy_params=STRATEGY_PARAMS,
        selected_factor_indices=SELECTED_FACTOR_INDICES,
        selected_factor_names=SELECTED_FACTOR_NAMES,
        composite_factor_sheet=COMPOSITE_FACTOR_SHEET,
        strategy_param=STRATEGY_PARAM,
        rebalance_period=STRATEGY_PARAMS["rebalance_period"],
        data_start_offset_days=DATA_START_OFFSET_DAYS,
        rf_rate=cfg.RISK_FREE_RATE,
    )

    # 打印摘要
    print("\n" + "-" * 64)
    print("策略概要:")
    print(f"  选定因子: {', '.join(SELECTED_FACTOR_NAMES)}")
    print(f"  复合因子: {COMPOSITE_FACTOR_SHEET} (IC加权 M3/N20)")
    print(f"  策略参数: {STRATEGY_PARAM}")
    print(f"    权重方式: {STRATEGY_PARAMS['weight_method']}")
    print(f"    分组数:   {STRATEGY_PARAMS['group_num']}")
    print(f"    目标组:   Top{STRATEGY_PARAMS['target_rank']}")
    print(f"    调仓周期: {STRATEGY_PARAMS['rebalance_period']} 交易日")
    print(f"    数据起始日偏移: {DATA_START_OFFSET_DAYS} 交易日")
    print("调仓日判定:")
    print(f"  今日是否调仓日: {'是' if status['is_rebalance_today'] else '否'}")
    print(f"  当前调仓日: {current_rb_date}")
    print(f"  下一调仓日: {status.get('next_rebalance_date')}")
    print(f"  当前调仓操作数: {len(current_ops)} 条")
    print(f"  全部输出目录: {run_dir}")
    print("=" * 64)

    # 阶段 5：Discord 通知
    if send_discord:
        print("\n[阶段 5] 发送 Discord 通知...")
        send_discord_notification(
            status=status,
            current_ops=current_ops,
            result=result,
            used_live_prices=used_live_prices,
            selected_factor_names=SELECTED_FACTOR_NAMES,
            composite_factor_sheet=COMPOSITE_FACTOR_SHEET,
            strategy_param=STRATEGY_PARAM,
            strategy_params=STRATEGY_PARAMS,
            data_start_offset_days=DATA_START_OFFSET_DAYS,
            rf_rate=cfg.RISK_FREE_RATE,
        )
    else:
        print("\n[阶段 5] 跳过 Discord 通知")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="调仓日全流程与报表")
    parser.add_argument("--skip-pipeline", action="store_true", help="跳过 pipeline，使用已有数据")
    parser.add_argument("--skip-pull", action="store_true", help="pipeline 中跳过 pull_data")
    parser.add_argument("--run-dir", type=str, default=None, help="指定运行目录")
    parser.add_argument("--no-discord", action="store_true", help="不发送 Discord 通知")
    parser.add_argument(
        "--inline",
        action="store_true",
        help="Pipeline 在同一进程中执行（更快，skip-pipeline 时无效）",
    )
    args = parser.parse_args()

    main(
        skip_pipeline=args.skip_pipeline,
        skip_pull=args.skip_pull,
        run_dir_arg=args.run_dir,
        send_discord=not args.no_discord,
        inline_pipeline=args.inline,
    )
