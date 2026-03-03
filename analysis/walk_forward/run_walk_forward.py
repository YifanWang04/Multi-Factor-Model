"""
Walk-Forward Validation Main Entry Point

运行walk-forward验证的主程序。

使用方法：
    python analysis/walk_forward/run_walk_forward.py

输出：
    - output/walk_forward_reports/parameter_stability.xlsx
    - output/walk_forward_reports/robust_strategies.xlsx
    - output/walk_forward_reports/parameter_sensitivity.xlsx
    - output/walk_forward_reports/walk_comparison.xlsx
    - output/walk_forward_reports/visualizations/*.png
"""

import sys
import os
import time
from datetime import datetime

# 添加项目路径
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from analysis.walk_forward import walk_forward_config as config
from analysis.walk_forward.walk_forward_engine import WalkForwardEngine
from analysis.walk_forward.walk_forward_analyzer import WalkForwardAnalyzer


def print_header():
    """打印标题"""
    print("\n" + "=" * 80)
    print(" " * 20 + "WALK-FORWARD VALIDATION")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")


def print_configuration():
    """打印配置信息"""
    print("Configuration:")
    print("-" * 80)
    print(f"  Training window:        {config.TRAINING_WINDOW} days")
    print(f"  Testing window:         {config.TESTING_WINDOW} days")
    print(f"  Roll forward step:      {config.ROLL_FORWARD_STEP} days")
    print(f"  Train-test gap:         {config.TRAIN_TEST_GAP} days")
    print()
    print(f"  Composite factor:       {config.COMPOSITE_METHOD}")
    print(f"  N window:               {config.N_WINDOW}")
    print(f"  Composite rebalance:    {config.COMPOSITE_REBALANCE_PERIOD} days")
    print()
    print(f"  Selected factors:       {len(config.SELECTED_FACTOR_INDICES)}")
    print(f"    Indices: {config.SELECTED_FACTOR_INDICES}")
    print()
    print(f"  Strategy grid search:")
    print(f"    Group numbers:        {config.GROUP_NUMS}")
    print(f"    Target ranks:         {config.TARGET_GROUP_RANKS}")
    print(f"    Rebalance periods:    {config.REBALANCE_PERIODS}")
    print(f"    Weight methods:       {config.WEIGHT_METHODS}")
    print(f"    Total combinations:   {len(config.GROUP_NUMS) * len(config.TARGET_GROUP_RANKS) * len(config.REBALANCE_PERIODS) * len(config.WEIGHT_METHODS)}")
    print()
    print(f"  Output directory:       {config.OUTPUT_DIR}")
    print("-" * 80 + "\n")


def print_summary(analyzer: WalkForwardAnalyzer, elapsed_time: float):
    """打印汇总信息"""
    print("\n" + "=" * 80)
    print("WALK-FORWARD VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Total walks:              {analyzer.num_walks}")
    print(f"Total strategies:         {analyzer.num_strategies}")
    print(f"Total backtests:          {len(analyzer.results)}")
    print(f"Elapsed time:             {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print()

    # 获取最稳健的策略
    try:
        top_strategy = analyzer.get_most_robust_strategy()
        print("Most Robust Strategy:")
        print(f"  Parameters:             {top_strategy['params']}")
        print(f"  Average Sharpe:         {top_strategy['avg_sharpe']:.3f}")
        print(f"  Sharpe Std:             {top_strategy['sharpe_std']:.3f}")
        print(f"  Win Rate:               {top_strategy['win_rate']:.1%}")
        print(f"  Average Annual Return:  {top_strategy['avg_return']:.2%}")
        print(f"  Average Max Drawdown:   {top_strategy['avg_mdd']:.2%}")
        print(f"  Consistency Score:      {top_strategy['consistency_score']:.3f}")
    except Exception as e:
        print(f"  ⚠ Could not determine most robust strategy: {e}")

    print()
    print(f"Reports saved to: {config.OUTPUT_DIR}")
    print("=" * 80 + "\n")


def main():
    """主函数"""
    start_time = time.time()

    # 打印标题和配置
    print_header()
    print_configuration()

    # 步骤1：初始化引擎
    print("Step 1/3: Initializing walk-forward engine...")
    print("-" * 80)
    try:
        engine = WalkForwardEngine(verbose=True)
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize engine: {e}")
        import traceback
        traceback.print_exc()
        return

    # 步骤2：运行walk-forward验证
    print("\n" + "=" * 80)
    print("Step 2/3: Running walk-forward validation...")
    print("=" * 80)
    try:
        results = engine.run()
    except Exception as e:
        print(f"\n[ERROR] Walk-forward validation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    if len(results) == 0:
        print("\n[ERROR] No results generated. Please check the logs above.")
        return

    print(f"\n[OK] Walk-forward validation complete: {len(results)} results generated")

    # 步骤3：分析结果
    print("\n" + "=" * 80)
    print("Step 3/3: Analyzing results...")
    print("=" * 80)
    try:
        analyzer = WalkForwardAnalyzer(results, config)
        analyzer.generate_all_reports()
    except Exception as e:
        print(f"\n[ERROR] Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 打印汇总
    elapsed_time = time.time() - start_time
    print_summary(analyzer, elapsed_time)

    print("[OK] Walk-forward validation complete!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
