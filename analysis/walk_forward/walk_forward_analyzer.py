"""
Walk-Forward Results Analyzer

分析和汇总walk-forward验证结果：
1. 参数稳定性分析（每个参数组合在所有walk中的表现）
2. 识别最稳健的策略
3. 参数敏感性分析
4. Walk对比分析
5. 生成报告和可视化
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
from datetime import datetime


class WalkForwardAnalyzer:
    """
    Walk-Forward结果分析器

    输入：所有walk的所有策略结果
    输出：汇总报告、稳健策略排名、可视化图表
    """

    def __init__(self, results: List[Dict], config):
        """
        初始化分析器

        Args:
            results: 所有walk的所有策略结果
            config: walk_forward_config模块
        """
        self.results = results
        self.config = config

        # 转换为DataFrame便于分析
        self.results_df = pd.DataFrame(results)

        # 基本统计
        self.num_walks = self.results_df['walk_id'].nunique()
        self.num_strategies = len(self.results_df.groupby(
            ['group_num', 'target_rank', 'rebalance_period', 'weight_method']
        ))

        # 创建输出目录
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        self.output_dir = config.OUTPUT_DIR

    def generate_all_reports(self):
        """生成所有报告和可视化"""
        print("\nGenerating reports...")

        # 1. 参数稳定性报告
        print("  1/5: Parameter stability report...")
        self.generate_parameter_stability_report()

        # 2. 稳健策略报告
        print("  2/5: Robust strategies report...")
        self.generate_robust_strategies_report()

        # 3. 参数敏感性报告
        print("  3/5: Parameter sensitivity report...")
        self.generate_parameter_sensitivity_report()

        # 4. Walk对比报告
        print("  4/5: Walk comparison report...")
        self.generate_walk_comparison_report()

        # 5. 可视化
        if self.config.GENERATE_PLOTS:
            print("  5/5: Generating visualizations...")
            self.generate_visualizations()

        print(f"\n[OK] All reports saved to: {self.output_dir}")

    def generate_parameter_stability_report(self):
        """
        生成参数稳定性报告

        对每个参数组合，计算：
        - 平均Sharpe
        - Sharpe标准差
        - 胜率（Sharpe>0的walk比例）
        - 平均年化收益
        - 平均最大回撤
        """
        # 按参数组合分组
        grouped = self.results_df.groupby(
            ['group_num', 'target_rank', 'rebalance_period', 'weight_method']
        )

        stability_metrics = []

        for params, group in grouped:
            group_num, target_rank, rebalance_period, weight_method = params

            # 计算稳定性指标
            sharpe_values = group['sharpe'].dropna()
            annual_return_values = group['annual_return'].dropna()
            max_dd_values = group['max_drawdown'].dropna()

            metrics = {
                'group_num': group_num,
                'target_rank': target_rank,
                'rebalance_period': rebalance_period,
                'weight_method': weight_method,
                'strategy_name': f"{weight_method}_{group_num}G_Top{target_rank}_P{rebalance_period}d",
                'avg_sharpe': sharpe_values.mean(),
                'sharpe_std': sharpe_values.std(),
                'win_rate': (sharpe_values > 0).mean(),
                'avg_annual_return': annual_return_values.mean(),
                'avg_max_drawdown': max_dd_values.mean(),
                'num_walks': len(group),
                'consistency_score': self._compute_consistency_score(sharpe_values)
            }

            stability_metrics.append(metrics)

        # 转换为DataFrame并排序
        stability_df = pd.DataFrame(stability_metrics)
        stability_df = stability_df.sort_values('consistency_score', ascending=False)

        # 保存
        output_file = os.path.join(self.output_dir, 'parameter_stability.xlsx')
        stability_df.to_excel(output_file, index=False)

        return stability_df

    def generate_robust_strategies_report(self, top_n: int = None):
        """
        生成稳健策略报告

        Args:
            top_n: 展示前N个策略，默认使用config中的设置
        """
        if top_n is None:
            top_n = self.config.TOP_N_STRATEGIES

        # 读取稳定性报告
        stability_file = os.path.join(self.output_dir, 'parameter_stability.xlsx')
        if not os.path.exists(stability_file):
            self.generate_parameter_stability_report()

        stability_df = pd.read_excel(stability_file)

        # 选择top N
        top_strategies = stability_df.head(top_n)

        # 保存
        output_file = os.path.join(self.output_dir, 'robust_strategies.xlsx')
        top_strategies.to_excel(output_file, index=False)

        return top_strategies

    def generate_parameter_sensitivity_report(self):
        """
        生成参数敏感性报告

        分析各参数对性能的影响：
        - Group number
        - Target rank
        - Rebalance period
        - Weight method
        """
        sensitivity = {}

        # 1. Group number sensitivity
        group_analysis = self.results_df.groupby('group_num')['sharpe'].agg([
            'mean', 'std', 'count'
        ]).reset_index()
        group_analysis.columns = ['group_num', 'avg_sharpe', 'sharpe_std', 'count']
        sensitivity['group_num'] = group_analysis

        # 2. Target rank sensitivity
        rank_analysis = self.results_df.groupby('target_rank')['sharpe'].agg([
            'mean', 'std', 'count'
        ]).reset_index()
        rank_analysis.columns = ['target_rank', 'avg_sharpe', 'sharpe_std', 'count']
        sensitivity['target_rank'] = rank_analysis

        # 3. Rebalance period sensitivity
        period_analysis = self.results_df.groupby('rebalance_period')['sharpe'].agg([
            'mean', 'std', 'count'
        ]).reset_index()
        period_analysis.columns = ['rebalance_period', 'avg_sharpe', 'sharpe_std', 'count']
        sensitivity['rebalance_period'] = period_analysis

        # 4. Weight method sensitivity
        method_analysis = self.results_df.groupby('weight_method')['sharpe'].agg([
            'mean', 'std', 'count'
        ]).reset_index()
        method_analysis.columns = ['weight_method', 'avg_sharpe', 'sharpe_std', 'count']
        sensitivity['weight_method'] = method_analysis

        # 保存到Excel（多个sheet）
        output_file = os.path.join(self.output_dir, 'parameter_sensitivity.xlsx')
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            for param_name, df in sensitivity.items():
                df.to_excel(writer, sheet_name=param_name, index=False)

        return sensitivity

    def generate_walk_comparison_report(self):
        """
        生成Walk对比报告

        对比各个walk的表现：
        - 最佳策略
        - 最差策略
        - 平均表现
        - 难度评估
        """
        walk_comparison = []

        for walk_id in range(self.num_walks):
            walk_data = self.results_df[self.results_df['walk_id'] == walk_id]

            if len(walk_data) == 0:
                continue

            # 检查是否所有sharpe都是NaN
            sharpe_values = walk_data['sharpe'].dropna()
            if len(sharpe_values) == 0:
                # 所有sharpe都是NaN，跳过此walk
                print(f"  [WARN] Walk {walk_id}: All strategies have NaN Sharpe, skipping...")
                continue

            # 找到最佳和最差策略
            best_idx = walk_data['sharpe'].idxmax()
            worst_idx = walk_data['sharpe'].idxmin()

            best_strategy = walk_data.loc[best_idx]
            worst_strategy = walk_data.loc[worst_idx]

            # 统计
            comparison = {
                'walk_id': walk_id,
                'train_start': walk_data.iloc[0]['train_period'][0],
                'train_end': walk_data.iloc[0]['train_period'][1],
                'test_start': walk_data.iloc[0]['test_period'][0],
                'test_end': walk_data.iloc[0]['test_period'][1],
                'best_strategy': best_strategy['strategy_name'],
                'best_sharpe': best_strategy['sharpe'],
                'worst_strategy': worst_strategy['strategy_name'],
                'worst_sharpe': worst_strategy['sharpe'],
                'avg_sharpe': walk_data['sharpe'].mean(),
                'median_sharpe': walk_data['sharpe'].median(),
                'sharpe_range': best_strategy['sharpe'] - worst_strategy['sharpe'],
                'positive_sharpe_ratio': (walk_data['sharpe'] > 0).mean()
            }

            walk_comparison.append(comparison)

        # 转换为DataFrame
        walk_df = pd.DataFrame(walk_comparison)

        # 保存
        output_file = os.path.join(self.output_dir, 'walk_comparison.xlsx')
        walk_df.to_excel(output_file, index=False)

        return walk_df

    def generate_visualizations(self):
        """生成可视化图表"""
        viz_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)

        # 1. Sharpe热力图（按参数组合）
        self._plot_sharpe_heatmap(viz_dir)

        # 2. 稳定性散点图（收益 vs 稳定性）
        self._plot_stability_scatter(viz_dir)

        # 3. 参数箱线图
        self._plot_parameter_boxplots(viz_dir)

        # 4. Walk表现对比
        self._plot_walk_performance(viz_dir)

    def _plot_sharpe_heatmap(self, output_dir):
        """绘制Sharpe热力图"""
        # 读取稳定性报告
        stability_file = os.path.join(self.output_dir, 'parameter_stability.xlsx')
        stability_df = pd.read_excel(stability_file)

        # 检查是否有有效数据
        if stability_df['avg_sharpe'].isna().all():
            print("  [WARN] No valid Sharpe data for heatmap, skipping...")
            return

        # 创建透视表（weight_method × rebalance_period）
        pivot = stability_df.pivot_table(
            values='avg_sharpe',
            index='weight_method',
            columns='rebalance_period',
            aggfunc='mean'
        )

        # 检查pivot是否为空
        if pivot.empty or pivot.isna().all().all():
            print("  [WARN] Pivot table is empty, skipping heatmap...")
            return

        # 绘图
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0)
        plt.title('Average Sharpe Ratio by Weight Method and Rebalance Period')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sharpe_heatmap.png'), dpi=300)
        plt.close()

    def _plot_stability_scatter(self, output_dir):
        """绘制稳定性散点图"""
        stability_file = os.path.join(self.output_dir, 'parameter_stability.xlsx')
        stability_df = pd.read_excel(stability_file)

        # 检查是否有有效数据
        if stability_df['avg_sharpe'].isna().all():
            print("  [WARN] No valid Sharpe data for scatter plot, skipping...")
            return

        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            stability_df['avg_sharpe'],
            stability_df['sharpe_std'],
            c=stability_df['win_rate'],
            s=100,
            alpha=0.6,
            cmap='RdYlGn'
        )
        plt.colorbar(scatter, label='Win Rate')
        plt.xlabel('Average Sharpe Ratio')
        plt.ylabel('Sharpe Std (Lower is Better)')
        plt.title('Strategy Stability: Return vs. Consistency')
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label='Std=0.5')
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.3)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'stability_scatter.png'), dpi=300)
        plt.close()

    def _plot_parameter_boxplots(self, output_dir):
        """绘制参数箱线图"""
        # 检查是否有有效数据
        if self.results_df['sharpe'].isna().all():
            print("  [WARN] No valid Sharpe data for boxplots, skipping...")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Group number
        self.results_df.boxplot(column='sharpe', by='group_num', ax=axes[0, 0])
        axes[0, 0].set_title('Sharpe by Group Number')
        axes[0, 0].set_xlabel('Group Number')
        axes[0, 0].set_ylabel('Sharpe Ratio')

        # 2. Target rank
        self.results_df.boxplot(column='sharpe', by='target_rank', ax=axes[0, 1])
        axes[0, 1].set_title('Sharpe by Target Rank')
        axes[0, 1].set_xlabel('Target Rank')
        axes[0, 1].set_ylabel('Sharpe Ratio')

        # 3. Rebalance period
        self.results_df.boxplot(column='sharpe', by='rebalance_period', ax=axes[1, 0])
        axes[1, 0].set_title('Sharpe by Rebalance Period')
        axes[1, 0].set_xlabel('Rebalance Period (days)')
        axes[1, 0].set_ylabel('Sharpe Ratio')

        # 4. Weight method
        self.results_df.boxplot(column='sharpe', by='weight_method', ax=axes[1, 1])
        axes[1, 1].set_title('Sharpe by Weight Method')
        axes[1, 1].set_xlabel('Weight Method')
        axes[1, 1].set_ylabel('Sharpe Ratio')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.suptitle('Parameter Sensitivity Analysis')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'parameter_boxplots.png'), dpi=300)
        plt.close()

    def _plot_walk_performance(self, output_dir):
        """绘制Walk表现对比"""
        walk_file = os.path.join(self.output_dir, 'walk_comparison.xlsx')
        walk_df = pd.read_excel(walk_file)

        # 检查是否有数据
        if len(walk_df) == 0:
            print("  [WARN] No valid walk data for visualization, skipping walk performance plot...")
            return

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # 1. 平均Sharpe
        axes[0].bar(walk_df['walk_id'], walk_df['avg_sharpe'], alpha=0.7)
        axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0].set_xlabel('Walk ID')
        axes[0].set_ylabel('Average Sharpe')
        axes[0].set_title('Average Sharpe Ratio Across Walks')
        axes[0].grid(True, alpha=0.3)

        # 2. Sharpe范围（最佳-最差）
        axes[1].bar(walk_df['walk_id'], walk_df['sharpe_range'], alpha=0.7, color='orange')
        axes[1].set_xlabel('Walk ID')
        axes[1].set_ylabel('Sharpe Range (Best - Worst)')
        axes[1].set_title('Strategy Performance Dispersion Across Walks')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'walk_performance.png'), dpi=300)
        plt.close()

    def get_most_robust_strategy(self) -> Dict:
        """
        获取最稳健的策略

        Returns:
            Dict: 最稳健策略的信息
        """
        stability_file = os.path.join(self.output_dir, 'parameter_stability.xlsx')
        if not os.path.exists(stability_file):
            self.generate_parameter_stability_report()

        stability_df = pd.read_excel(stability_file)
        top_strategy = stability_df.iloc[0]

        return {
            'params': top_strategy['strategy_name'],
            'avg_sharpe': top_strategy['avg_sharpe'],
            'sharpe_std': top_strategy['sharpe_std'],
            'win_rate': top_strategy['win_rate'],
            'avg_return': top_strategy['avg_annual_return'],
            'avg_mdd': top_strategy['avg_max_drawdown'],
            'consistency_score': top_strategy['consistency_score']
        }

    def _compute_consistency_score(self, sharpe_values: pd.Series) -> float:
        """
        计算一致性得分

        综合考虑：
        - 平均Sharpe（越高越好）
        - Sharpe标准差（越低越好）
        - 胜率（越高越好）

        Returns:
            float: 一致性得分（越高越好）
        """
        if len(sharpe_values) == 0:
            return 0.0

        avg_sharpe = sharpe_values.mean()
        sharpe_std = sharpe_values.std()
        win_rate = (sharpe_values > 0).mean()

        # 归一化并加权
        # 高Sharpe + 低std + 高胜率 = 高一致性
        score = (
            avg_sharpe * 0.4 +  # 40%权重给平均表现
            (1 / (1 + sharpe_std)) * 0.3 +  # 30%权重给稳定性
            win_rate * 0.3  # 30%权重给胜率
        )

        return score


if __name__ == "__main__":
    print("Walk-Forward Analyzer Test")
    print("This module requires results from walk_forward_engine.py")
