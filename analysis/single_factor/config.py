"""
单因子测试配置文件 (config.py)
=====================================
本模块定义单因子回测所需的全部配置项，供 run_single_factor_test、run_all_factors_backtest 等使用。

主要职责：
- 路径：项目根目录、因子文件、价格文件、报告输出目录
- 测试参数：调仓周期列表（如 1/5/10 天）、分组数（固定 10）、组内加权方式（等权/因子加权）
- 风险参数：无风险利率、单边交易成本
- 绘图与报告：图表尺寸、因子显示名称

使用方式：通过 SingleFactorConfig 类读取属性；也可用 SimpleNamespace 复制部分属性做批量回测。
"""
import os

class SingleFactorConfig:
    """
    单因子测试配置
    """
    
    # ==================== 路径配置 ====================
    # 项目根目录（手动指定）
    PROJECT_ROOT = r"D:\新建文件夹\qqq"
    
    # 数据路径
    # 指定单因子测试时使用的因子文件（运行 run_single_factor_test.py 时生效）
    # 批量测试（run_all_factors_backtest.py）会自动扫描 factor_processed 目录，忽略此项
    FACTOR_FILE = os.path.join(PROJECT_ROOT, "factor_processed", "factor_alpha001_processed.xlsx")
    PRICE_FILE = os.path.join(PROJECT_ROOT, "data", "us_top100_daily_2023_present.xlsx")
    # 收益率：从 PRICE_FILE 读。若某 sheet 有 RETURN_COLUMN 列（如 "Return"）则直接用，否则用该 sheet 的收盘价 pct_change() 计算
    RETURN_COLUMN = "Return"
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "single_factor_reports")
    
    # ==================== 测试参数 ====================
    # 调仓周期列表（天），单因子回测时会对每个周期分别跑
    REBALANCE_PERIODS = [1, 5, 10]  # 可以修改为你需要的周期
    
    # 分层数量（固定为10）
    GROUP_NUM = 10
    
    # 层内资产配置方式
    WEIGHT_METHOD = 'equal'  # 'equal' 或 'factor_weight'
    
    # ==================== 风险参数 ====================
    # 无风险利率（年化）
    RISK_FREE_RATE = 0.02
    
    # 交易成本（单边，百分比）
    TRANSACTION_COST = 0.001
    
    # ==================== 绘图配置 ====================
    # 图表尺寸
    FIGURE_SIZE = (12, 6)
    
    # ==================== 报告配置 ====================
    # 因子名称（仅影响单因子报告标题；批量测试时由文件名自动推导）
    FACTOR_NAME = "Alpha 001"


# 测试路径
if __name__ == "__main__":
    config = SingleFactorConfig()
    
    print("="*60)
    print("配置路径检查")
    print("="*60)
    
    print(f"\n项目根目录: {config.PROJECT_ROOT}")
    print(f"因子文件路径: {config.FACTOR_FILE}")
    print(f"价格文件路径: {config.PRICE_FILE}")
    print(f"输出目录: {config.OUTPUT_DIR}")
    
    print("\n调仓周期:")
    for period in config.REBALANCE_PERIODS:
        print(f"  - {period} days")
    
    print("\n文件存在性检查:")
    print(f"✓ 因子文件存在: {os.path.exists(config.FACTOR_FILE)}")
    print(f"✓ 价格文件存在: {os.path.exists(config.PRICE_FILE)}")
    
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)
        print(f"✓ 创建输出目录: {config.OUTPUT_DIR}")
    else:
        print(f"✓ 输出目录存在: {config.OUTPUT_DIR}")