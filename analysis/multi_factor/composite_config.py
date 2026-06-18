"""
复合因子配置兼容导出层 (composite_config.py)

因子选择机制（优先级从高到低）：
  1. 环境变量 REBALANCE_SELECTED_FACTOR_INDICES
     —— 由 run_rebalance_day.py 启动 pipeline 时设置，用于保证单次运行一致。
  2. qqq_config/strategy_profiles.py 中的 active profile
     —— 默认权威配置源，与 strategy_config.py 保持一致。
"""
import os
import sys
from qqq_core.paths import ProjectPaths

PROJECT_ROOT = str(ProjectPaths.from_env().root)
_RUN_DIR = os.environ.get("REBALANCE_RUN_DIR")

# ── 路径注册（从 strategy_utils 导入统一实现前需先注册）─────────────
_STRATEGY_UTILS_DIR = os.path.join(PROJECT_ROOT, "analysis", "strategy")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if os.path.isdir(_STRATEGY_UTILS_DIR) and _STRATEGY_UTILS_DIR not in sys.path:
    sys.path.insert(0, _STRATEGY_UTILS_DIR)

from qqq_config.strategy_profiles import get_active_profile, parse_factor_indices_csv

# ── Active profile 派生配置 ───────────────────────────────────────────────────
ACTIVE_PROFILE = get_active_profile()
# 保留旧变量名供历史调用方兼容；不要在此处手动改因子列表。
MANUALLY_SELECTED_FACTOR_INDICES = list(ACTIVE_PROFILE.factor_indices)

# Research-only override for running run_composite_factor.py directly.
# Set to a list like [95, 99, 27, 46, 19], or leave as None to use the active profile.
# The rebalance-day pipeline still takes precedence through REBALANCE_SELECTED_FACTOR_INDICES.
# COMPOSITE_RESEARCH_FACTOR_INDICES = [95, 101, 62, 65, 32]
COMPOSITE_RESEARCH_FACTOR_INDICES = None #June 16, 2026

# ─────────────────────────────────────────────────────────────────────────────


def _coerce_factor_indices(value):
    if value is None:
        return []
    if isinstance(value, str):
        return list(parse_factor_indices_csv(value))
    return [int(i) for i in value]


def _resolve_selected_factor_indices():
    """
    解析选定的因子编号。
    优先级：
      1. REBALANCE_SELECTED_FACTOR_INDICES（运行时环境变量）
      2. active profile（通过兼容变量 MANUALLY_SELECTED_FACTOR_INDICES 暴露）
      3. 抛出异常
    """
    env_val = os.environ.get("REBALANCE_SELECTED_FACTOR_INDICES")
    if env_val:
        indices = list(parse_factor_indices_csv(env_val))
        if indices:
            return indices

    research_indices = _coerce_factor_indices(COMPOSITE_RESEARCH_FACTOR_INDICES)
    if research_indices:
        return research_indices

    if MANUALLY_SELECTED_FACTOR_INDICES:
        return list(MANUALLY_SELECTED_FACTOR_INDICES)

    raise ValueError(
        "未找到因子配置：请检查 qqq_config/strategy_profiles.py 的 active profile，"
        "或通过 REBALANCE_SELECTED_FACTOR_INDICES 提供运行时覆盖。"
    )


# 内联 offset 目录后缀逻辑（避免从 data_config 导入 _offset_dir_suffix 触发循环导入）
# 支持两层来源：1) 环境变量 REBALANCE_OFFSET_DAYS（subprocess 传递） 2) data_config.DATA_START_OFFSET_DAYS
def _offset_suffix() -> str:
    env_val = os.environ.get("REBALANCE_OFFSET_DAYS")
    if env_val is not None:
        offset = int(env_val)
    else:
        from data.data_config import DATA_START_OFFSET_DAYS
        offset = DATA_START_OFFSET_DAYS
    return f"_offset{offset}d" if offset != 0 else ""

# 路径：根据 data_config 按 offset 分子目录（不覆盖）
from data.data_config import (
    PRICE_FILE as _DEFAULT_PRICE_FILE,
    _price_filename,
    FACTOR_PROCESSED_DIR as _DEFAULT_FACTOR_PROCESSED_DIR,
    COMPOSITE_FACTOR_OUTPUT_DIR as _DEFAULT_COMPOSITE_OUTPUT_DIR,
)

if _RUN_DIR:
    FACTOR_PROCESSED_DIR = os.path.join(_RUN_DIR, "factor_processed")
    PRICE_FILE = os.path.join(_RUN_DIR, "data", _price_filename())
    OUTPUT_DIR = os.path.join(_RUN_DIR, "composite_factor_reports")
else:
    FACTOR_PROCESSED_DIR = _DEFAULT_FACTOR_PROCESSED_DIR
    PRICE_FILE = _DEFAULT_PRICE_FILE
    OUTPUT_DIR = _DEFAULT_COMPOSITE_OUTPUT_DIR
RETURN_COLUMN = "Return"

# 选定因子编号（自动从运行时环境变量或 active profile 解析，勿硬编码）
SELECTED_FACTOR_INDICES = _resolve_selected_factor_indices()
SELECTED_FACTOR_NAMES = [f"alpha{i:03d}" for i in SELECTED_FACTOR_INDICES]

# 调仓周期（交易日数）：相邻调仓日之间至少相隔 N 个交易日。
# REBALANCE_PERIOD 是兼容旧调用方的主周期；多周期研究使用 COMPOSITE_REBALANCE_PERIODS。
REBALANCE_PERIOD = ACTIVE_PROFILE.rebalance_period
COMPOSITE_REBALANCE_PERIODS = [5, 10, 20]

# 一元/IC加权滚动窗口列表 N
N_WINDOWS = [5, 10, 20]

# 多元回归滚动窗口列表 M
M_WINDOWS = [5, 10, 20]

# 回测参数
GROUP_NUM = 10
WEIGHT_METHOD = "equal"
RISK_FREE_RATE = 0.02
TRANSACTION_COST = 0.001


def get_all_factor_files(factor_dir=None):
    factor_dir = factor_dir or FACTOR_PROCESSED_DIR
    if not os.path.isdir(factor_dir):
        return []
    out = []
    seen = set()
    for f in sorted(os.listdir(factor_dir)):
        if not f.endswith(".xlsx"):
            continue
        full = os.path.join(factor_dir, f)
        base = f.replace("_processed.xlsx", "").replace(".xlsx", "")
        if base in seen:
            continue
        seen.add(base)
        out.append(full)
    return out


def get_selected_factor_files():
    all_files = get_all_factor_files()
    if _RUN_DIR:
        name_to_path = {get_factor_display_name(p): p for p in all_files}
        missing = [n for n in SELECTED_FACTOR_NAMES if n not in name_to_path]
        if missing:
            raise FileNotFoundError(
                "调仓日 factor_processed 缺少选定因子: "
                f"{missing}；扫描目录: {FACTOR_PROCESSED_DIR}；"
                f"实际文件: {[os.path.basename(p) for p in all_files]}"
            )
        return [name_to_path[n] for n in SELECTED_FACTOR_NAMES]
    name_to_path = {get_factor_display_name(p): p for p in all_files}
    missing = [n for n in SELECTED_FACTOR_NAMES if n not in name_to_path]
    if missing:
        raise FileNotFoundError(
            "factor_processed 缺少选定因子: "
            f"{missing}；扫描目录: {FACTOR_PROCESSED_DIR}；"
            f"实际文件: {[os.path.basename(p) for p in all_files]}"
        )
    return [name_to_path[n] for n in SELECTED_FACTOR_NAMES]


def get_factor_display_name(filepath):
    basename = os.path.splitext(os.path.basename(filepath))[0]
    if basename.endswith("_processed"):
        basename = basename[:-len("_processed")]
    if basename.lower().startswith("factor_"):
        basename = basename[7:]
    return basename.replace("_", " ").strip() or basename


# ── 从 strategy_utils 统一导入 build_factor_suffix ──────────────────
# strategy_utils.build_factor_suffix 支持无参数调用（返回空字符串）
# composite_config 需要无参数时使用 SELECTED_FACTOR_INDICES，因此用 wrapper
try:
    from strategy_utils import build_factor_suffix as _su_build_factor_suffix

    def build_factor_suffix(factor_indices: list[int] | None = None) -> str:
        if factor_indices is None:
            factor_indices = SELECTED_FACTOR_INDICES
        return _su_build_factor_suffix(factor_indices)
except (ImportError, OSError):
    # 兜底：内联实现（strategy_utils 不可用时）
    def build_factor_suffix(factor_indices: list[int] | None = None) -> str:
        if factor_indices is None:
            factor_indices = SELECTED_FACTOR_INDICES
        if factor_indices is None:
            return ""
        nums = [str(int(i)) for i in factor_indices]
        return "f" + "-".join(nums)
