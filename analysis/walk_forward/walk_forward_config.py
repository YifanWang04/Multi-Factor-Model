"""
Walk-Forward Validation Configuration

配置walk-forward验证的所有参数，包括时间窗口、因子选择、策略网格搜索参数等。
"""

import os

# ==================== 项目路径 ====================
PROJECT_ROOT = r"D:\qqq"

# ==================== 数据文件 ====================
PRICE_FILE = os.path.join(PROJECT_ROOT, "data", "us_top100_daily_2023_present.xlsx")
FACTOR_RAW_DIR = os.path.join(PROJECT_ROOT, "factor_raw")
FACTOR_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "factor_processed")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "walk_forward_reports")

# ==================== 时间窗口参数 ====================
TRAINING_WINDOW = 400  # 训练窗口：400个交易日 (~1.6年)
TESTING_WINDOW = 60    # 测试窗口：60个交易日 (~3个月)
ROLL_FORWARD_STEP = 30 # 滚动步长：30个交易日 (~1.5个月)
TRAIN_TEST_GAP = 0     # 训练和测试之间的间隔天数（0表示无间隔）

# ==================== 复合因子配置（固定） ====================
# 选择的因子索引（1-based，与multi_factor_config保持一致）
SELECTED_FACTOR_INDICES = [20, 16, 43, 17, 34]

# 复合因子方法：beta_m3 (beta加权，方法3：滚动窗口)
COMPOSITE_METHOD = "beta_m3"
N_WINDOW = 10  # 方法3的滚动窗口大小
COMPOSITE_REBALANCE_PERIOD = 10  # 复合因子的调仓周期（固定10天）

# ==================== 策略网格搜索参数 ====================
# 分组数量
GROUP_NUMS = [5, 10]

# 目标组排名（1=因子值最高的组）
TARGET_GROUP_RANKS = [1, 2, 3]

# 调仓周期（天数）
REBALANCE_PERIODS = [10, 20, 30, 60]

# 调仓日偏移（天数）：正数=提前，负数=延后；0=不偏移
REBALANCE_DATE_OFFSET = 0

# 权重方法
WEIGHT_METHODS = ["equal", "min_variance", "mvo", "max_return", "factor_score"]

# 总策略组合数：2 × 3 × 4 × 5 = 120

# ==================== 投资组合优化参数 ====================
OPTIMIZATION_LOOKBACK = 252  # 协方差估计的历史回溯天数（1年）
MAX_WEIGHT = 0.4            # 单只股票的最大权重
TRANSACTION_COST = 0.001    # 交易成本（单边）
RISK_FREE_RATE = 0.02       # 无风险利率（年化）

# ==================== 输出配置 ====================
# 是否保存每个walk的详细结果
SAVE_WALK_DETAILS = True

# 是否生成可视化图表
GENERATE_PLOTS = True

# 报告中展示的top策略数量
TOP_N_STRATEGIES = 10


def get_strategy_name(group_num, target_rank, rebalance_period, weight_method):
    """
    生成策略名称

    例如：mvo_5G_Top2_P10d
    """
    return f"{weight_method}_{group_num}G_Top{target_rank}_P{rebalance_period}d"


def get_factor_files():
    """
    获取所有原始因子文件路径

    Returns:
        List[str]: 因子文件路径列表
    """
    import glob
    pattern = os.path.join(FACTOR_RAW_DIR, "factor_*.xlsx")
    files = glob.glob(pattern)
    return sorted(files)


def get_selected_factor_files():
    """
    根据SELECTED_FACTOR_INDICES获取选中的因子文件

    Returns:
        List[str]: 选中的因子文件路径列表
    """
    all_files = get_factor_files()
    # 索引从1开始，转换为0-based
    selected_files = [all_files[i-1] for i in SELECTED_FACTOR_INDICES if 0 < i <= len(all_files)]
    return selected_files


if __name__ == "__main__":
    # 测试配置
    print("Walk-Forward Validation Configuration")
    print("=" * 80)
    print(f"Training Window: {TRAINING_WINDOW} days")
    print(f"Testing Window: {TESTING_WINDOW} days")
    print(f"Roll Forward Step: {ROLL_FORWARD_STEP} days")
    print(f"Train-Test Gap: {TRAIN_TEST_GAP} days")
    print()
    print(f"Composite Method: {COMPOSITE_METHOD}")
    print(f"N Window: {N_WINDOW}")
    print(f"Composite Rebalance Period: {COMPOSITE_REBALANCE_PERIOD} days")
    print()
    print(f"Selected Factor Indices: {SELECTED_FACTOR_INDICES}")
    print(f"Selected Factor Files:")
    for f in get_selected_factor_files():
        print(f"  - {os.path.basename(f)}")
    print()
    print(f"Strategy Grid Search:")
    print(f"  Group Numbers: {GROUP_NUMS}")
    print(f"  Target Ranks: {TARGET_GROUP_RANKS}")
    print(f"  Rebalance Periods: {REBALANCE_PERIODS}")
    print(f"  Weight Methods: {WEIGHT_METHODS}")
    print(f"  Total Combinations: {len(GROUP_NUMS) * len(TARGET_GROUP_RANKS) * len(REBALANCE_PERIODS) * len(WEIGHT_METHODS)}")
    print()
    print(f"Output Directory: {OUTPUT_DIR}")
