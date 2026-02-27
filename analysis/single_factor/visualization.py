"""
因子测试可视化模块 (visualization.py)
=====================================
本模块为单因子测试报告生成所有图表，使用 matplotlib/seaborn，后端为 Agg（适合无界面环境生成 PDF）。

主要类：FactorVisualizerOptimized(figsize=(12, 6))

图表类型概览：
- IC 分析：IC 统计表、年度 IC 柱状图（多周期合并）、月度 IC 热力图、每日 IC 折线（含移动平均）、累计 IC 曲线。
- 多空回测：绩效统计表、净值与回撤双 Y 轴图、月度收益热力图、多空操作汇总表。
- 多头/空头：分层收益柱状图（可指定组 1/5/10）、分组统计表、分层净值曲线、操作汇总表。

所有绘图方法返回 matplotlib Figure，由 report_generator 写入 PDF。字体与负号已配置为兼容显示。
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.gridspec import GridSpec

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


class FactorVisualizerOptimized:
    """
    因子可视化器 - 优化版
    """
    
    def __init__(self, figsize=(12, 6)):
        self.figsize = figsize
        self.colors = {
            'period1': '#2E86AB',
            'period5': '#F4A261',
            'period10': '#2A9D8F',
            'ic_line': '#2E86AB',
            'ma_line': '#E76F51',
            'nav_line': '#A23B72',
            'dd_fill': 'salmon'
        }
    
    # ===========================================
    # IC 分析图表
    # ===========================================
    
    def plot_ic_statistics_table(self, ic_stats_combined):
        """
        图1: IC统计表（所有调仓周期）
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('tight')
        ax.axis('off')
        
        fig.text(0.5, 0.95, 'IC Statistics', 
                ha='center', va='top', fontsize=20, fontweight='bold')
        
        # 准备表格数据
        table_data = []
        headers = ['Metric'] + list(ic_stats_combined.keys())
        table_data.append(headers)
        
        metrics = ['Mean', 'Std', 'IR', 'Skew', 'Kurtosis', 't_value', 'p_value', 'Win_Rate', 'IC>0.02']
        
        for metric in metrics:
            row = [metric]
            for period_key in ic_stats_combined.keys():
                ic_val = ic_stats_combined[period_key]['IC'].get(metric, np.nan)
                if metric == 'p_value':
                    row.append(f'{ic_val:.6f}')
                else:
                    row.append(f'{ic_val:.4f}')
            table_data.append(row)
        
        table = ax.table(cellText=table_data, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.tight_layout()
        return fig
    
    def plot_annual_ic_bar_combined(self, annual_ic_dict):
        """
        图2: 年度IC柱状图（所有调仓周期合并）
        
        Parameters:
        -----------
        annual_ic_dict: dict, {period: annual_ic_df}
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # 获取所有年份
        all_years = set()
        for period, annual_ic in annual_ic_dict.items():
            all_years.update(annual_ic.index)
        all_years = sorted(list(all_years))
        
        # 每个周期一个颜色
        periods = sorted(annual_ic_dict.keys())
        colors = ['#2E86AB', '#F4A261', '#2A9D8F', '#E76F51']
        
        x = np.arange(len(all_years))
        width = 0.8 / len(periods)
        
        for i, period in enumerate(periods):
            annual_ic = annual_ic_dict[period]
            ic_values = [annual_ic.loc[year, 'IC'] if year in annual_ic.index else 0 
                        for year in all_years]
            
            offset = (i - len(periods)/2 + 0.5) * width
            ax.bar(x + offset, ic_values, width, 
                  label=f'{period}d', color=colors[i % len(colors)], alpha=0.8)
        
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('IC (%)', fontsize=12)
        ax.set_title('Annual IC by Period', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(all_years)
        ax.legend(fontsize=10)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def plot_monthly_ic_heatmap(self, monthly_ic, title='Monthly IC Heatmap'):
        """
        图3: 月度IC热力图（单个调仓周期）
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        # 空矩阵或全 NaN 时 seaborn.heatmap 会报 zero-size array
        if monthly_ic is None or monthly_ic.size == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=14)
            ax.set_title(title, fontsize=14, fontweight='bold')
            plt.tight_layout()
            return fig
        clean = monthly_ic.dropna(how='all').dropna(axis=1, how='all')
        if clean.size == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=14)
            ax.set_title(title, fontsize=14, fontweight='bold')
            plt.tight_layout()
            return fig
        sns.heatmap(monthly_ic, annot=True, fmt='.2f', cmap='RdYlGn',
                   center=0, linewidths=0.5, cbar_kws={'label': 'IC (%)'},
                   ax=ax)
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Year', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_ic_with_ma(self, ic_df, ma_window=30, title='IC Time Series'):
        """
        图4: 每日IC折线图（蓝线=每日，红线=30天移动平均）
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 每日IC（蓝线）
        ax.plot(ic_df.index, ic_df['IC'], linewidth=1, color='#2E86AB', 
               label='Daily', alpha=0.6)
        
        # 30天移动平均（红线）
        ic_ma = ic_df['IC'].rolling(ma_window).mean()
        ax.plot(ic_df.index, ic_ma, linewidth=2, color='#E76F51', 
               label=f'{ma_window}d MA')
        
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('IC (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_cumulative_ic(self, ic_df, title='Cumulative IC'):
        """
        图5: 累计IC曲线
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        cumsum_ic = ic_df['IC'].cumsum()
        ax.plot(ic_df.index, cumsum_ic, linewidth=2, color='#2E86AB', label='Cumulative')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative IC', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_group_ic_time_series(self, group_ic_df, title='Group IC Time Series'):
        """
        Group IC 时间序列：调仓日 vs Group_IC / Group_Rank_IC
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(group_ic_df.index, group_ic_df['Group_IC'], label='Group IC', color='#2E86AB', linewidth=2)
        ax.plot(group_ic_df.index, group_ic_df['Group_Rank_IC'], label='Group Rank IC', color='#E76F51', linewidth=2, alpha=0.8)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Rebalance Date', fontsize=12)
        ax.set_ylabel('IC', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    
    def plot_group_ic_stats_table(self, group_ic_stats_combined, title='Group IC Statistics'):
        """
        Group IC 统计表（多周期合并）
        group_ic_stats_combined: {period_key: {'Group_IC': stats_dict, 'Group_Rank_IC': stats_dict}}
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.axis('tight')
        ax.axis('off')
        fig.text(0.5, 0.95, title, ha='center', va='top', fontsize=20, fontweight='bold')
        table_data = []
        headers = ['Metric'] + list(group_ic_stats_combined.keys())
        table_data.append(headers)
        metrics = ['Mean', 'Std', 'IR', 't_value', 'p_value', 'Win_Rate']
        for metric in metrics:
            row = [metric]
            for period_key in group_ic_stats_combined.keys():
                gic = group_ic_stats_combined[period_key]['Group_IC'].get(metric, np.nan)
                if pd.isna(gic):
                    row.append('—')
                elif metric == 'p_value':
                    row.append(f'{float(gic):.6f}')
                else:
                    row.append(f'{float(gic):.4f}')
            table_data.append(row)
        table = ax.table(cellText=table_data, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        plt.tight_layout()
        return fig
    
    def plot_rebalance_period_returns(self, period_returns, title='Rebalance Period Returns (%)'):
        """
        调仓日收益率折线图。period_returns: Series（单条线）或 DataFrame（多列则多条线），index=调仓日，值为期间收益率（小数或百分数均可，图中按百分数显示）。
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        if isinstance(period_returns, pd.Series):
            period_returns = period_returns.to_frame(name='Return')
        # 转为百分数
        pct = period_returns * 100 if period_returns.abs().max().max() <= 2 else period_returns
        for col in pct.columns:
            ax.plot(pct.index, pct[col].values, label=str(col), linewidth=2, alpha=0.8)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Rebalance Date', fontsize=12)
        ax.set_ylabel('Period Return (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    
    # ===========================================
    # 多空回测图表
    # ===========================================
    
    def plot_performance_table(self, stats_dict, title='Performance Statistics'):
        """
        图6: Performance统计表
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('tight')
        ax.axis('off')
        
        fig.text(0.5, 0.95, title, 
                ha='center', va='top', fontsize=20, fontweight='bold')
        
        table_data = []
        headers = ['Metric'] + list(stats_dict.keys())
        table_data.append(headers)
        
        metrics = [
            ('Annual_Return', 'AReturnRate', '.4f'),
            ('Volatility', 'AVolatility', '.4f'),
            ('Sharpe', 'SharpeRatio', '.4f'),
            ('Max_Drawdown', 'MaxDrawdown', '.4f'),
            ('Win_Rate', 'WinningRatio', '.4f'),
            ('Profit_Loss_Ratio', 'PnLRatio', '.4f')
        ]
        
        for metric_key, metric_name, fmt in metrics:
            row = [metric_name]
            for period_key in stats_dict.keys():
                val = stats_dict[period_key].get(metric_key, np.nan)
                row.append(f'{val:{fmt}}')
            table_data.append(row)
        
        table = ax.table(cellText=table_data, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.tight_layout()
        return fig
    
    def plot_nav_with_drawdown(self, nav, title='NAV & Drawdown'):
        """
        图7: 净值与回撤双Y轴图
        """
        fig = plt.figure(figsize=self.figsize)
        
        # 创建主轴（左Y轴 - 净值）
        ax1 = fig.add_subplot(111)
        
        # 净值曲线（蓝线）
        line1 = ax1.plot(nav.index, nav.values, linewidth=2, 
                        color='#2E86AB', label='Net Value')
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Net Value', fontsize=12, color='#2E86AB')
        ax1.tick_params(axis='y', labelcolor='#2E86AB')
        ax1.grid(True, alpha=0.3)
        
        # 创建次轴（右Y轴 - 回撤）
        ax2 = ax1.twinx()
        
        # 计算回撤
        cummax = nav.cummax()
        drawdown = (nav - cummax) / cummax
        
        # 回撤填充区域（红色）；扩大右轴范围（约 2 倍实际回撤范围）以缩小回撤在图中的比例
        ax2.fill_between(drawdown.index, 0, drawdown.values, 
                        color='salmon', alpha=0.5, label='Drawdown')
        ax2.set_ylabel('Drawdown', fontsize=12, color='salmon')
        ax2.tick_params(axis='y', labelcolor='salmon')
        dd_min = float(drawdown.min()) if len(drawdown) else 0
        if dd_min < 0:
            ax2.set_ylim(dd_min * 4, 0)  # 右轴范围约为实际回撤的 3 倍
        else:
            ax2.set_ylim(-0.1, 0)
        
        # 图例
        lines = line1
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', fontsize=10)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_monthly_returns_heatmap(self, monthly_returns, title='Monthly Returns Heatmap'):
        """
        图8: 月度收益率热力图
        """
        # 转换为矩阵格式
        monthly_returns_df = monthly_returns.to_frame(name='Returns')
        monthly_returns_df['Year'] = monthly_returns_df.index.year
        monthly_returns_df['Month'] = monthly_returns_df.index.month
        
        monthly_matrix = monthly_returns_df.pivot_table(
            values='Returns',
            index='Year',
            columns='Month',
            aggfunc='first'
        )
        
        # 转换为百分比
        monthly_matrix = monthly_matrix * 100
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        sns.heatmap(monthly_matrix, annot=True, fmt='.2f', cmap='RdYlGn', 
                   center=0, linewidths=0.5, cbar_kws={'label': 'Return (%)'},
                   ax=ax)
        
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Year', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    # ===========================================
    # 多头/空头回测图表
    # ===========================================
    
    def plot_group_returns_by_period(self, group_returns_by_period, 
                                     groups_to_show=[1, 5, 10],
                                     title='Group Return'):
        """
        图9: 分层收益柱状图（显示不同组在不同周期的表现）
        
        Parameters:
        -----------
        group_returns_by_period: dict, {period: {group_num: stats}}
        groups_to_show: list, 要显示的组号
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        periods = sorted(group_returns_by_period.keys())
        n_periods = len(periods)
        x = np.arange(n_periods)
        width = 0.25
        
        colors = ['#2E86AB', '#F4A261', '#2A9D8F']
        
        for i, group_num in enumerate(groups_to_show):
            returns_pct = []
            for period in periods:
                stats = group_returns_by_period[period].get(group_num, {})
                total_return = stats.get('Total_Return', 0)
                returns_pct.append(total_return * 100)  # 转换为百分比
            
            offset = (i - len(groups_to_show)/2 + 0.5) * width
            ax.bar(x + offset, returns_pct, width, 
                  label=f'Group {group_num}', color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Period', fontsize=12)
        ax.set_ylabel('Return (%)', fontsize=12)
        ax.set_title(title + '\n(Group 1=lowest factor, Group 10=highest; for momentum Group 10 is typically best)', 
                     fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{p}' for p in periods])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def plot_group_stats_table(self, group_stats_combined, title='Group Statistics'):
        """
        图10: 分组统计数据表格
        
        Parameters:
        -----------
        group_stats_combined: dict, {period_key: {group_num: stats}}
        """
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.axis('tight')
        ax.axis('off')
        
        fig.text(0.5, 0.95, title, 
                ha='center', va='top', fontsize=20, fontweight='bold')
        
        # 准备表格数据
        table_data = []
        
        # 获取所有周期和组
        periods = sorted(group_stats_combined.keys())
        first_period = periods[0]
        all_groups = sorted(group_stats_combined[first_period].keys())
        
        # 表头
        headers = ['Period', 'Group'] + ['AReturnRate', 'AVolatility', 'SharpeRatio', 
                                         'MaxDrawdown', 'WinningRatio', 'PnLRatio']
        table_data.append(headers)
        
        # 数据行
        for period_key in periods:
            for group_num in all_groups:
                stats = group_stats_combined[period_key].get(group_num, {})
                row = [
                    period_key,
                    f'G{group_num}',
                    f"{stats.get('Annual_Return', 0):.4f}",
                    f"{stats.get('Volatility', 0):.4f}",
                    f"{stats.get('Sharpe', 0):.4f}",
                    f"{stats.get('Max_Drawdown', 0):.4f}",
                    f"{stats.get('Win_Rate', 0):.4f}",
                    f"{stats.get('Profit_Loss_Ratio', 0):.4f}"
                ]
                table_data.append(row)
        
        table = ax.table(cellText=table_data, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.tight_layout()
        return fig
    
    def plot_operation_table(self, table_df, title='Operation Summary', max_stocks_display=8):
        """
        绘制操作汇总表（调仓日、买卖股票、期间收益等）
        table_df: DataFrame, 列至少包含表格需要展示的字段
        max_stocks_display: int, 股票列表最多显示几只，其余用 "...(共N只)" 表示
        """
        def _truncate_stocks(stocks, max_n):
            if not isinstance(stocks, (list, tuple)):
                return str(stocks)[:80]
            s = list(stocks)[:max_n]
            txt = ', '.join(str(x) for x in s)
            if len(stocks) > max_n:
                txt += f' ...(共{len(stocks)}只)'
            return txt[:120] + ('...' if len(txt) > 120 else '')

        fig, ax = plt.subplots(figsize=(14, max(6, len(table_df) * 0.35)))
        ax.axis('tight')
        ax.axis('off')
        fig.suptitle(title, fontsize=14, fontweight='bold')
        # 转为可显示字符串，截断过长股票列表
        display_df = table_df.copy()
        for col in display_df.columns:
            if 'stock' in col.lower() or '股票' in col or col in ('Long_Stocks', 'Short_Stocks', 'Stocks'):
                display_df[col] = display_df[col].apply(
                    lambda x: _truncate_stocks(x, max_stocks_display) if isinstance(x, (list, tuple)) else str(x)[:80]
                )
            elif 'date' in col.lower() or 'Date' in col:
                display_df[col] = display_df[col].astype(str).str[:10]
            elif 'return' in col.lower() or 'pct' in col.lower():
                def _fmt_pct(x):
                    if pd.isna(x) or x == '':
                        return ''
                    try:
                        return f'{float(x):.2f}%'
                    except (TypeError, ValueError):
                        return str(x)
                display_df[col] = display_df[col].apply(_fmt_pct)
        table = ax.table(
            cellText=display_df.values.tolist(),
            colLabels=display_df.columns.tolist(),
            cellLoc='left',
            loc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.8)
        for i in range(len(display_df.columns)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        plt.tight_layout()
        return fig

    def plot_group_nav_curves(self, group_results, title='Group NAV Curves'):
        """
        图11: 分层净值曲线（所有10组）
        
        Parameters:
        -----------
        group_results: dict, {group_num: {'nav': nav, 'returns': returns}}
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 使用渐变色
        colors = plt.cm.RdYlGn(np.linspace(0, 1, len(group_results)))
        
        for i, (group_num, result) in enumerate(sorted(group_results.items())):
            nav = result['nav']
            ax.plot(nav.index, nav.values, linewidth=2, 
                   label=f'{group_num}', color=colors[i])
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Net Value', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, 
                 ncol=1, title='Group')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        plt.tight_layout()
        return fig