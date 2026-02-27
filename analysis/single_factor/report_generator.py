"""
单因子测试 PDF 报告生成器 (report_generator.py)
=====================================
本模块将 IC 分析、多空/多头/空头回测的图表与统计表按固定顺序组装成一份 PDF 报告。

主要类：FactorReportGeneratorOptimized(output_path, factor_name, rebalance_periods)

报告结构：
1. 封面：因子名、调仓周期、生成时间、目录概要。
2. IC 分析：IC 统计表 → 年度 IC 柱状图 → 各周期月度 IC 热力图 → 各周期每日 IC 折线 → 各周期累计 IC。
3. 多空回测：绩效表 → 各周期净值与回撤图 → 各周期月度收益热力图。
4. 多头回测：分层收益柱状图 → 分组统计表 → 各周期分层净值曲线。
5. 空头回测：做空组 1+2 绩效表与净值回撤 → 分层收益柱状图 → 分组统计表 → 分层净值曲线。

generate(figures, ic_stats, ls_stats, long_stats, short_stats) 接收预先生成的 figures 字典与各统计字典，写入 PDF 并关闭图形。
"""

from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


class FactorReportGeneratorOptimized:
    """
    单因子测试报告生成器 - 优化版
    """
    
    def __init__(self, output_path, factor_name, rebalance_periods):
        """
        Parameters:
        -----------
        output_path: str, 输出PDF路径
        factor_name: str, 因子名称
        rebalance_periods: list, 调仓周期列表
        """
        self.output_path = output_path
        self.factor_name = factor_name
        self.rebalance_periods = rebalance_periods
    
    def generate(self, figures, ic_stats, ls_stats, long_stats, short_stats):
        """
        生成完整的PDF报告
        
        PDF结构：
        1. 封面
        2. IC分析
           - 图1: IC统计表
           - 图2: 年度IC柱状图（合并）
           - 图3: 月度IC热力图（每个周期一张）
           - 图4: 每日IC折线图（每个周期一张）
           - 图5: 累计IC曲线（每个周期一张）
        3. 多空回测
           - 图6: Performance统计表
           - 图7: 净值与回撤图（每个周期一张）
           - 图8: 月度收益率热力图（每个周期一张）
        4. 多头回测
           - 图9: 分层收益柱状图
           - 图10: 分组统计表
           - 图11: 分层净值曲线（每个周期一张）
        5. 空头回测
           - 同多头结构
        """
        
        with PdfPages(self.output_path) as pdf:
            
            # ========== 封面 ==========
            self._add_cover_page(pdf)
            
            # ========== IC 分析 ==========
            print("  添加IC分析图表...")
            
            # 图1: IC统计表
            if 'stats_table' in figures['ic']:
                pdf.savefig(figures['ic']['stats_table'], bbox_inches='tight')
                plt.close(figures['ic']['stats_table'])
            
            # 图2: 年度IC柱状图（合并）
            if 'annual_bar' in figures['ic']:
                pdf.savefig(figures['ic']['annual_bar'], bbox_inches='tight')
                plt.close(figures['ic']['annual_bar'])
            
            # 图3: 月度IC热力图（每个周期）
            for period in self.rebalance_periods:
                key = f'monthly_heatmap_P{period}'
                if key in figures['ic']:
                    pdf.savefig(figures['ic'][key], bbox_inches='tight')
                    plt.close(figures['ic'][key])
            
            # 图4: 每日IC折线图（每个周期）
            for period in self.rebalance_periods:
                key = f'daily_ic_P{period}'
                if key in figures['ic']:
                    pdf.savefig(figures['ic'][key], bbox_inches='tight')
                    plt.close(figures['ic'][key])
            
            # 图5: 累计IC曲线（每个周期）
            for period in self.rebalance_periods:
                key = f'cumulative_ic_P{period}'
                if key in figures['ic']:
                    pdf.savefig(figures['ic'][key], bbox_inches='tight')
                    plt.close(figures['ic'][key])
            # Group IC 统计表
            if 'group_ic_stats_table' in figures['ic']:
                pdf.savefig(figures['ic']['group_ic_stats_table'], bbox_inches='tight')
                plt.close(figures['ic']['group_ic_stats_table'])
            # Group IC 时间序列（每个周期）
            for period in self.rebalance_periods:
                key = f'group_ic_ts_P{period}'
                if key in figures['ic']:
                    pdf.savefig(figures['ic'][key], bbox_inches='tight')
                    plt.close(figures['ic'][key])
            
            # ========== 多空回测 ==========
            print("  添加多空回测图表...")
            
            # 图6: Performance统计表
            if 'performance_table' in figures['long_short']:
                pdf.savefig(figures['long_short']['performance_table'], bbox_inches='tight')
                plt.close(figures['long_short']['performance_table'])
            
            # 图7: 净值与回撤图（每个周期）
            for period in self.rebalance_periods:
                key = f'nav_drawdown_P{period}'
                if key in figures['long_short']:
                    pdf.savefig(figures['long_short'][key], bbox_inches='tight')
                    plt.close(figures['long_short'][key])
            
            # 图8: 月度收益率热力图（每个周期）
            for period in self.rebalance_periods:
                key = f'monthly_returns_P{period}'
                if key in figures['long_short']:
                    pdf.savefig(figures['long_short'][key], bbox_inches='tight')
                    plt.close(figures['long_short'][key])
            # 调仓日收益率折线图（每个周期）
            for period in self.rebalance_periods:
                key = f'period_returns_P{period}'
                if key in figures['long_short']:
                    pdf.savefig(figures['long_short'][key], bbox_inches='tight')
                    plt.close(figures['long_short'][key])
            # 多空操作汇总表（调仓周期短时表格很长，已注释不写入 PDF）
            # for period in self.rebalance_periods:
            #     key = f'ops_table_P{period}'
            #     if key in figures['long_short']:
            #         pdf.savefig(figures['long_short'][key], bbox_inches='tight')
            #         plt.close(figures['long_short'][key])
            
            # ========== 多头回测 ==========
            print("  添加多头回测图表...")
            
            # 图9: 分层收益柱状图
            if 'group_returns_bar' in figures['long_only']:
                pdf.savefig(figures['long_only']['group_returns_bar'], bbox_inches='tight')
                plt.close(figures['long_only']['group_returns_bar'])
            
            # 图10: 分组统计表
            if 'stats_table' in figures['long_only']:
                pdf.savefig(figures['long_only']['stats_table'], bbox_inches='tight')
                plt.close(figures['long_only']['stats_table'])
            
            # 图11: 分层净值曲线（每个周期）
            for period in self.rebalance_periods:
                key = f'group_nav_P{period}'
                if key in figures['long_only']:
                    pdf.savefig(figures['long_only'][key], bbox_inches='tight')
                    plt.close(figures['long_only'][key])
            # 多头调仓日收益率折线图（每个周期）
            for period in self.rebalance_periods:
                key = f'period_returns_P{period}'
                if key in figures['long_only']:
                    pdf.savefig(figures['long_only'][key], bbox_inches='tight')
                    plt.close(figures['long_only'][key])
            # 多头操作汇总表（每个周期）- 已注释，不写入 PDF
            # for period in self.rebalance_periods:
            #     key = f'ops_table_P{period}'
            #     if key in figures['long_only']:
            #         pdf.savefig(figures['long_only'][key], bbox_inches='tight')
            #         plt.close(figures['long_only'][key])
            
            # ========== 空头回测（做空组1+2） ==========
            print("  添加空头回测图表...")
            
            # 做空组1+2 绩效表
            if 'performance_table_combined' in figures['short_only']:
                pdf.savefig(figures['short_only']['performance_table_combined'], bbox_inches='tight')
                plt.close(figures['short_only']['performance_table_combined'])
            # 做空组1+2 净值与回撤（每个周期）
            for period in self.rebalance_periods:
                key = f'nav_drawdown_combined_P{period}'
                if key in figures['short_only']:
                    pdf.savefig(figures['short_only'][key], bbox_inches='tight')
                    plt.close(figures['short_only'][key])
            # 图9: 分层收益柱状图
            if 'group_returns_bar' in figures['short_only']:
                pdf.savefig(figures['short_only']['group_returns_bar'], bbox_inches='tight')
                plt.close(figures['short_only']['group_returns_bar'])
            
            # 图10: 分组统计表
            if 'stats_table' in figures['short_only']:
                pdf.savefig(figures['short_only']['stats_table'], bbox_inches='tight')
                plt.close(figures['short_only']['stats_table'])
            
            # 图11: 分层净值曲线（每个周期）
            for period in self.rebalance_periods:
                key = f'group_nav_P{period}'
                if key in figures['short_only']:
                    pdf.savefig(figures['short_only'][key], bbox_inches='tight')
                    plt.close(figures['short_only'][key])
            # 空头调仓日收益率折线图（每个周期）
            for period in self.rebalance_periods:
                key = f'period_returns_P{period}'
                if key in figures['short_only']:
                    pdf.savefig(figures['short_only'][key], bbox_inches='tight')
                    plt.close(figures['short_only'][key])
            # 空头操作汇总表（做空组1+2，每个周期）- 已注释，不写入 PDF
            # for period in self.rebalance_periods:
            #     key = f'ops_table_P{period}'
            #     if key in figures['short_only']:
            #         pdf.savefig(figures['short_only'][key], bbox_inches='tight')
            #         plt.close(figures['short_only'][key])
            
            # 元数据
            d = pdf.infodict()
            d['Title'] = f'{self.factor_name} - Single Factor Test Report'
            d['Author'] = 'Factor Testing System'
            d['Subject'] = f'Rebalance Periods: {", ".join([str(p) for p in self.rebalance_periods])} days'
            d['CreationDate'] = datetime.now()
        
        print(f"Report generated: {self.output_path}")
    
    def _add_cover_page(self, pdf):
        """添加封面页"""
        fig = plt.figure(figsize=(11, 8.5))
        
        fig.text(0.5, 0.7, 'Single Factor Test Report', 
                ha='center', va='center', fontsize=32, fontweight='bold')
        fig.text(0.5, 0.6, f'Factor: {self.factor_name}', 
                ha='center', va='center', fontsize=20)
        fig.text(0.5, 0.5, f'Rebalance Periods: {", ".join([str(p) for p in self.rebalance_periods])} days', 
                ha='center', va='center', fontsize=16)
        fig.text(0.5, 0.4, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                ha='center', va='center', fontsize=14)
        
        # 添加分段标识
        fig.text(0.5, 0.25, 'Report Contents:', 
                ha='center', va='center', fontsize=16, fontweight='bold')
        
        contents = [
            '1. IC Analysis',
            '2. Long-Short Backtest',
            '3. Long-Only Backtest',
            '4. Short-Only Backtest'
        ]
        
        y_start = 0.20
        for i, content in enumerate(contents):
            fig.text(0.5, y_start - i*0.03, content, 
                    ha='center', va='center', fontsize=12)
        
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)