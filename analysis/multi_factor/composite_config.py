"""
因子复合配置文件 (composite_config.py)

因子选择机制（优先级从高到低）：
  1. 环境变量 REBALANCE_SELECTED_FACTOR_INDICES
     —— 由 run_rebalance_day.py 在启动 pipeline 子进程时设置，
        保证整个流程（build_factors → data_process → run_composite_factor）使用一致的因子
  2. 本文件中的 MANUALLY_SELECTED_FACTOR_INDICES（手动配置）
     —— 直接在本文件修改，适合临时测试其他因子组合，不依赖 strategy_config
  ⚠️ 注意：若需长期换因子，建议同步更新 strategy_config.py 以保持一致性
"""
import os

PROJECT_ROOT = r"D:\qqq"
_RUN_DIR = os.environ.get("REBALANCE_RUN_DIR")

# ── 手动因子配置区 ─────────────────────────────────────────────────────────────
# ⚠️ 如需切换因子，直接修改此列表（如 [95, 101, 62, 65, 32]），无需改其他文件
MANUALLY_SELECTED_FACTOR_INDICES = [95, 24, 64, 65, 32]
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_selected_factor_indices():
    """
    解析选定的因子编号。
    优先级：
      1. REBALANCE_SELECTED_FACTOR_INDICES（环境变量，run_rebalance_day.py 设置）
      2. MANUALLY_SELECTED_FACTOR_INDICES（本文件手动配置）
      3. 抛出异常（必须配置）
    """
    env_val = os.environ.get("REBALANCE_SELECTED_FACTOR_INDICES")
    if env_val:
        indices = [int(x.strip()) for x in env_val.split(",") if x.strip()]
        if indices:
            return indices

    if MANUALLY_SELECTED_FACTOR_INDICES:
        return list(MANUALLY_SELECTED_FACTOR_INDICES)

    raise ValueError(
        "未找到因子配置：请在 composite_config.py 中设置 MANUALLY_SELECTED_FACTOR_INDICES，"
        "或通过 run_rebalance_day.py 启动以自动设置环境变量。"
    )


# 内联 offset 目录后缀逻辑（避免从 data_config 导入 _offset_dir_suffix 触发循环导入）
def _offset_suffix() -> str:
    from data.data_config import DATA_START_OFFSET_DAYS
    return f"_offset{DATA_START_OFFSET_DAYS}d" if DATA_START_OFFSET_DAYS != 0 else ""

# 路径：根据 data_config 按 offset 分子目录（不覆盖）
from data.data_config import (
    PRICE_FILE as _DEFAULT_PRICE_FILE,
    _price_filename,
    FACTOR_PROCESSED_DIR as _DEFAULT_FACTOR_PROCESSED_DIR,
)

if _RUN_DIR:
    FACTOR_PROCESSED_DIR = os.path.join(_RUN_DIR, "factor_processed")
    PRICE_FILE = os.path.join(_RUN_DIR, "data", _price_filename())
    OUTPUT_DIR = os.path.join(_RUN_DIR, "composite_factor_reports")
else:
    _pfx = _offset_suffix()
    FACTOR_PROCESSED_DIR = os.path.join(PROJECT_ROOT, f"factor_processed{_pfx}")
    PRICE_FILE = _DEFAULT_PRICE_FILE
    _comp_out = os.path.join(PROJECT_ROOT, "output", f"composite_factor_reports{_pfx}")
    OUTPUT_DIR = _comp_out
RETURN_COLUMN = "Return"

# 选定因子编号（自动从环境变量或 strategy_config 解析，勿硬编码）
SELECTED_FACTOR_INDICES = _resolve_selected_factor_indices()
SELECTED_FACTOR_NAMES = [f"alpha{i:03d}" for i in SELECTED_FACTOR_INDICES]

# 调仓周期（交易日数）：相邻调仓日之间至少相隔 N 个交易日
# ⚠️ 重要：此周期决定复合因子的生成频率
# 建议：策略回测的 REBALANCE_PERIODS 应包含此值，或为此值的整数倍
REBALANCE_PERIOD = 10

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
        # 调仓日流程已通过 REBALANCE_SELECTED_FACTORS 只构建了选定因子
        # factor_processed 中即为全部选定因子，直接返回
        return all_files
    # 当 offset 子目录（如 factor_processed_offset4d）不存在时，回退到 factor_processed
    if not all_files and "_offset" in FACTOR_PROCESSED_DIR:
        _project_root = os.path.dirname(FACTOR_PROCESSED_DIR)
        _base_factor_dir = os.path.join(_project_root, "factor_processed")
        if os.path.isdir(_base_factor_dir):
            all_files = get_all_factor_files(_base_factor_dir)
    name_to_path = {get_factor_display_name(p): p for p in all_files}
    return [name_to_path[n] for n in SELECTED_FACTOR_NAMES if n in name_to_path]


def get_factor_display_name(filepath):
    basename = os.path.splitext(os.path.basename(filepath))[0]
    if basename.endswith("_processed"):
        basename = basename[:-len("_processed")]
    if basename.lower().startswith("factor_"):
        basename = basename[7:]
    return basename.replace("_", " ").strip() or basename


def build_factor_suffix(factor_indices: list[int] | None = None) -> str:
    """
    基于因子编号列表生成简短后缀，如 f95-24-64-65-32。
    未提供时使用 SELECTED_FACTOR_INDICES。
    """
    if factor_indices is None:
        factor_indices = SELECTED_FACTOR_INDICES
    nums = [str(int(i)) for i in factor_indices]
    return "f" + "-".join(nums)
