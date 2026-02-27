"""
单因子测试主程序 (run_single_factor_test.py)
=====================================
本模块是单因子测试的入口：对一个因子、多个调仓周期运行完整流程，并生成一份合并的 PDF 报告。

流程概要：
1. 从 config 加载因子文件与价格文件，计算日频收益率。
2. 对每个调仓周期：用 RebalancePeriodManager 对齐因子与期间收益 → ICAnalyzerEnhanced 计算 IC → GrouperEnhanced 分组并计算组收益 → 多空（做多组 10+9、做空组 1+2）与分层多头/空头回测 → PerformanceAnalyzer 计算绩效 → 汇总操作表。
3. 用 FactorVisualizerOptimized 生成所有图表（IC、多空、多头、空头）。
4. 用 FactorReportGeneratorOptimized 将图表与统计表写入 PDF（文件名含因子名与时间戳）。

使用方式：SingleFactorTesterOptimized(config).run()，或直接运行本文件执行 config 中配置的单一因子。
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# 导入所有模块
from config import SingleFactorConfig as Config
from rebalance_manager import RebalancePeriodManager
from ic import ICAnalyzerEnhanced
from grouping import GrouperEnhanced
from backtest import (
    LongShortBacktestEnhanced,
    LongOnlyBacktest,
    ShortOnlyBacktest,
)
from performance import PerformanceAnalyzer
from visualization import FactorVisualizerOptimized
from report_generator import FactorReportGeneratorOptimized


class SingleFactorTesterOptimized:
    """
    单因子测试器 - 优化版
    一个因子生成一个完整的PDF报告
    """
    
    def __init__(self, config=Config):
        self.config = config
        self.visualizer = FactorVisualizerOptimized(
            figsize=config.FIGURE_SIZE
        )
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    def load_data(self):
        """加载数据"""
        print("="*60)
        print("Step 1: 加载数据")
        print("="*60)
        
        # 加载因子
        print(f"加载因子文件: {self.config.FACTOR_FILE}")
        factor = pd.read_excel(self.config.FACTOR_FILE, sheet_name=0, index_col=0)
        factor.index = pd.to_datetime(factor.index)
        factor = factor.apply(pd.to_numeric, errors='coerce')
        
        print(f"因子数据: {factor.shape}")
        print(f"日期范围: {factor.index.min()} to {factor.index.max()}")
        
        # 加载价格数据；收益率：有 Return 列就用，没有就用收盘价 pct_change 算
        print(f"\n加载价格文件: {self.config.PRICE_FILE}")
        price_data = pd.read_excel(self.config.PRICE_FILE, sheet_name=None)
        return_col = getattr(self.config, 'RETURN_COLUMN', 'Return')
        
        close_df = pd.DataFrame()
        ret = pd.DataFrame()
        for ticker, df in price_data.items():
            if 'Date' not in df.columns or 'Adj Close' not in df.columns:
                continue
            df = df.copy()
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            close_df[ticker] = df['Adj Close']
            if return_col in df.columns:
                ret[ticker] = df[return_col]
            else:
                ret[ticker] = df['Adj Close'].pct_change()
        
        print(f"价格数据: {close_df.shape}")
        ret = ret.replace([np.inf, -np.inf], np.nan)
        
        self.factor = factor
        self.ret = ret
        self.close_df = close_df
    
    def run_all_periods(self):
        """
        运行所有调仓周期的测试
        返回合并后的结果
        """
        all_results = {}
        
        for period in self.config.REBALANCE_PERIODS:
            print(f"\n{'='*60}")
            print(f"调仓周期: {period} 天")
            print(f"{'='*60}")
            
            result = self.run_single_period(period)
            all_results[period] = result
        
        return all_results
    
    def run_single_period(self, rebalance_period):
        """运行单个调仓周期的测试"""
        
        # 1. 准备数据
        print("\nStep 2: 准备调仓期数据")
        manager = RebalancePeriodManager(self.factor, self.ret, rebalance_period)
        factor_periods, ret_periods = manager.align_factor_return_by_period()
        
        print(f"调仓期数: {len(factor_periods)}")
        
        # 2. IC分析
        print("\nStep 3: IC分析")
        ic_analyzer = ICAnalyzerEnhanced(factor_periods, ret_periods)
        ic_df = ic_analyzer.calculate_ic()
        
        ic_stats = {
            'IC': ic_analyzer.calculate_statistics(ic_df['IC']),
            'Rank_IC': ic_analyzer.calculate_statistics(ic_df['Rank_IC'])
        }
        
        annual_ic = ic_analyzer.get_annual_ic(ic_df.copy())
        monthly_ic = ic_analyzer.get_monthly_ic(ic_df.copy())
        
        # 3. 分组
        print("\nStep 4: 因子分组")
        grouper = GrouperEnhanced(
            factor_periods, 
            self.config.GROUP_NUM,
            self.config.WEIGHT_METHOD
        )
        group_dict = grouper.split()
        weight_dict = grouper.get_group_weights(group_dict)
        group_returns = grouper.calculate_group_returns(group_dict, ret_periods, weight_dict)
        
        # Group IC：组号与组收益率的相关性
        group_ic_df = ic_analyzer.calculate_group_ic(group_dict, group_returns)
        group_ic_stats = {
            'Group_IC': ic_analyzer.calculate_statistics(group_ic_df['Group_IC']),
            'Group_Rank_IC': ic_analyzer.calculate_statistics(group_ic_df['Group_Rank_IC']),
        }
        
        # 4. 多空回测：买入最高两组，卖出最低两组（按列位置：前两列=低因子组，后两列=高因子组）
        # print("\nStep 5: 多空回测（做多组10+9，做空组1+2）")
        # cols = group_returns.columns.tolist()
        # if len(cols) < 4:
        #     # 分组不足（如 period=1 时因子与收益对齐后无有效组）：用占位数据避免崩溃，该周期表现为无收益
        #     index_gr = ret_periods.index if len(group_returns) == 0 else group_returns.index
        #     group_returns = pd.DataFrame(
        #         np.zeros((len(index_gr), 4)),
        #         index=index_gr,
        #         columns=[1, 2, 3, 4]
        #     )
        #     top2_cols = [4, 3]
        #     bottom2_cols = [1, 2]
        # else:
        #     top2_cols = cols[-2:]
        #     bottom2_cols = cols[:2]
        # long_combined_returns = group_returns[top2_cols].mean(axis=1)
        # short_combined_returns = group_returns[bottom2_cols].mean(axis=1)
        #
        # # 多空收益 = 多头组合收益 - 空头组合收益 - 双边交易成本（空头端 = 做空底组，收益为 -做多底组收益）
        # ls_returns = long_combined_returns - short_combined_returns - 2 * self.config.TRANSACTION_COST
        # ls_nav = (1 + ls_returns).cumprod()
        # # #region agent log
        # try:
        #     _d = group_returns.index[0]
        #     r1 = float(group_returns.loc[_d, 1]) if 1 in group_returns.columns else None
        #     r2 = float(group_returns.loc[_d, 2]) if 2 in group_returns.columns else None
        #     r9 = float(group_returns.loc[_d, 9]) if 9 in group_returns.columns else None
        #     r10 = float(group_returns.loc[_d, 10]) if 10 in group_returns.columns else None
        #     lcr = float(long_combined_returns.loc[_d]) if _d in long_combined_returns.index else None
        #     scr = float(short_combined_returns.loc[_d]) if _d in short_combined_returns.index else None
        #     lsr = float(ls_returns.loc[_d]) if _d in ls_returns.index else None
        #     cost = 2 * self.config.TRANSACTION_COST
        #     formula_ok = abs((lcr - scr - cost) - lsr) < 1e-6 if lcr is not None and scr is not None and lsr is not None else None
        #     with open(r"d:\新建文件夹\qqq\.cursor\debug.log", "a", encoding="utf-8") as _f:
        #         _f.write('{"id":"ls_check","location":"run_single_period","message":"Long-short composition","data":{"R1":r1,"R2":r2,"R9":r9,"R10":r10,"long_combined":lcr,"short_combined":scr,"ls_return":lsr,"cost":cost,"formula_ok":formula_ok},"hypothesisId":"H2"}\n')
        # except Exception:
        #     pass
        # # #endregion
        # ls_perf = PerformanceAnalyzer(ls_nav, ls_returns, self.config.RISK_FREE_RATE)
        # ls_stats = ls_perf.calculate_metrics()
        # ls_monthly_returns = ls_perf.get_monthly_returns()
        #
        # print(f"年化收益: {ls_stats['Annual_Return']:.2%}")
        # print(f"夏普比率: {ls_stats['Sharpe']:.2f}")
        
        # 5. 分层多头回测
        print("\nStep 6: 分层多头回测")
        long_backtester = LongOnlyBacktest(group_returns, self.config.TRANSACTION_COST)
        long_results = long_backtester.run_all_groups()
        
        long_stats = {}
        for group_num, result in long_results.items():
            perf = PerformanceAnalyzer(result['nav'], result['returns'], self.config.RISK_FREE_RATE)
            long_stats[group_num] = perf.calculate_metrics()
        
        # 6. 空头回测：做空最低两组（合并）。做空收益 = -组收益，单边成本
        # print("\nStep 7: 空头回测（做空组1和2）")
        # short_combined_returns = -(group_returns[bottom2_cols].mean(axis=1)) - self.config.TRANSACTION_COST
        # short_combined_nav = (1 + short_combined_returns).cumprod()
        # short_combined_perf = PerformanceAnalyzer(
        #     short_combined_nav, short_combined_returns, self.config.RISK_FREE_RATE
        # )
        # short_combined_stats = short_combined_perf.calculate_metrics()
        #
        # short_backtester = ShortOnlyBacktest(group_returns, self.config.TRANSACTION_COST)
        # short_results = short_backtester.run_all_groups()
        # short_stats = {}
        # for group_num, result in short_results.items():
        #     perf = PerformanceAnalyzer(result['nav'], result['returns'], self.config.RISK_FREE_RATE)
        #     short_stats[group_num] = perf.calculate_metrics()

        # 7. 操作汇总表数据（调仓日、买卖股票、期间收益）
        ops_long = []
        # ops_ls = []
        # ops_short = []
        # ops_short_combined = []
        for date in group_returns.index:
            if date not in group_dict:
                continue
            g = group_dict[date]
            # # 多空：做多组10+9，做空组1+2
            # long_stocks = g.get(10, []) + g.get(9, [])
            # short_stocks = g.get(1, []) + g.get(2, [])
            # ret_ls = ls_returns.loc[date] if date in ls_returns.index else np.nan
            # ops_ls.append({
            #     'Rebalance_Date': date,
            #     'Long_Stocks': long_stocks,
            #     'Short_Stocks': short_stocks,
            #     'Period_Return_LS_Pct': ret_ls * 100 if pd.notna(ret_ls) else np.nan,
            # })
            # # 空头做空组1+2 汇总
            # short_12_stocks = g.get(1, []) + g.get(2, [])
            # ret_short_12 = short_combined_returns.loc[date] if date in short_combined_returns.index else np.nan
            # ops_short_combined.append({
            #     'Rebalance_Date': date,
            #     'Short_Stocks_G1_G2': short_12_stocks,
            #     'Period_Return_Pct': ret_short_12 * 100 if pd.notna(ret_short_12) else np.nan,
            # })
            for grp_num in sorted(g.keys()):
                stocks = g[grp_num]
                ret_grp = group_returns.loc[date, grp_num] if grp_num in group_returns.columns else np.nan
                ops_long.append({
                    'Rebalance_Date': date,
                    'Group': grp_num,
                    'Action': 'Buy',
                    'Stocks': stocks,
                    'Period_Return_Pct': ret_grp * 100 if pd.notna(ret_grp) else np.nan,
                })
                # ret_short_grp = -ret_grp if pd.notna(ret_grp) else np.nan
                # ops_short.append({
                #     'Rebalance_Date': date,
                #     'Group': grp_num,
                #     'Action': 'Short',
                #     'Stocks': stocks,
                #     'Period_Return_Pct': ret_short_grp * 100 if pd.notna(ret_short_grp) else np.nan,
                # })
        ops_long_df = pd.DataFrame(ops_long)
        # ops_ls_df = pd.DataFrame(ops_ls)
        # ops_short_df = pd.DataFrame(ops_short)
        # ops_short_combined_df = pd.DataFrame(ops_short_combined)

        return {
            'ic_df': ic_df,
            'ic_stats': ic_stats,
            'group_ic_df': group_ic_df,
            'group_ic_stats': group_ic_stats,
            'annual_ic': annual_ic,
            'monthly_ic': monthly_ic,
            # 'ls_nav': ls_nav,
            # 'ls_returns': ls_returns,
            # 'ls_stats': ls_stats,
            # 'ls_monthly_returns': ls_monthly_returns,
            'long_results': long_results,
            'long_stats': long_stats,
            # 'short_results': short_results,
            # 'short_stats': short_stats,
            # 'short_combined_nav': short_combined_nav,
            # 'short_combined_returns': short_combined_returns,
            # 'short_combined_stats': short_combined_stats,
            'group_returns': group_returns,
            'group_dict': group_dict,
            # 'ops_ls': ops_ls_df,
            'ops_long': ops_long_df,
            # 'ops_short': ops_short_df,
            # 'ops_short_combined': ops_short_combined_df,
        }
    
    def generate_combined_figures(self, all_results):
        """
        生成合并的图表
        
        Parameters:
        -----------
        all_results: dict, {period: results}
        """
        print("\nStep 8: 生成图表")
        
        figures = {
            'ic': {},
            'long_short': {},
            'long_only': {},
            'short_only': {}
        }
        
        periods = sorted(all_results.keys())
        
        # ===========================================
        # IC 分析图表
        # ===========================================
        
        # 图1: IC统计表（所有周期合并）
        ic_stats_combined = {}
        for period in periods:
            ic_stats_combined[f'Period_{period}'] = all_results[period]['ic_stats']
        
        figures['ic']['stats_table'] = self.visualizer.plot_ic_statistics_table(ic_stats_combined)
        
        # 图2: 年度IC柱状图（所有周期合并在一张图）
        annual_ic_dict = {period: all_results[period]['annual_ic'] for period in periods}
        figures['ic']['annual_bar'] = self.visualizer.plot_annual_ic_bar_combined(annual_ic_dict)
        
        # 图3: 月度IC热力图（每个周期一张）
        for period in periods:
            monthly_ic = all_results[period]['monthly_ic']
            figures['ic'][f'monthly_heatmap_P{period}'] = self.visualizer.plot_monthly_ic_heatmap(
                monthly_ic, f'Monthly IC Heatmap (Period={period}d)'
            )
        
        # 图4: 每日IC折线图（每个周期一张，包含30天移动平均）
        for period in periods:
            ic_df = all_results[period]['ic_df']
            figures['ic'][f'daily_ic_P{period}'] = self.visualizer.plot_ic_with_ma(
                ic_df, ma_window=30, title=f'IC Time Series (Period={period}d)'
            )
        
        # 图5: 累计IC曲线（每个周期一张）
        for period in periods:
            ic_df = all_results[period]['ic_df']
            figures['ic'][f'cumulative_ic_P{period}'] = self.visualizer.plot_cumulative_ic(
                ic_df, title=f'Cumulative IC (Period={period}d)'
            )
        
        # Group IC 统计表与时间序列（每个周期一张）
        group_ic_stats_combined = {f'Period_{p}': all_results[p]['group_ic_stats'] for p in periods}
        figures['ic']['group_ic_stats_table'] = self.visualizer.plot_group_ic_stats_table(
            group_ic_stats_combined, title='Group IC Statistics'
        )
        for period in periods:
            group_ic_df = all_results[period]['group_ic_df']
            figures['ic'][f'group_ic_ts_P{period}'] = self.visualizer.plot_group_ic_time_series(
                group_ic_df, title=f'Group IC Time Series (Period={period}d)'
            )
        
        # ===========================================
        # 多空回测图表（已注释）
        # ===========================================

        # # 图6: Performance统计表
        # ls_stats_combined = {f'Period_{period}': all_results[period]['ls_stats'] for period in periods}
        # figures['long_short']['performance_table'] = self.visualizer.plot_performance_table(
        #     ls_stats_combined, 'Long-Short Performance'
        # )
        #
        # # 图7: 净值与回撤双Y轴图（每个周期一张）
        # for period in periods:
        #     nav = all_results[period]['ls_nav']
        #     figures['long_short'][f'nav_drawdown_P{period}'] = self.visualizer.plot_nav_with_drawdown(
        #         nav, title=f'Long-Short NAV & Drawdown (Period={period}d)'
        #     )
        #
        # # 图8: 月度收益率热力图（每个周期一张）
        # for period in periods:
        #     monthly_returns = all_results[period]['ls_monthly_returns']
        #     figures['long_short'][f'monthly_returns_P{period}'] = self.visualizer.plot_monthly_returns_heatmap(
        #         monthly_returns, title=f'Long-Short Monthly Returns (Period={period}d)'
        #     )
        # # 调仓日收益率折线图（每个周期一张）
        # for period in periods:
        #     ls_returns = all_results[period]['ls_returns']
        #     figures['long_short'][f'period_returns_P{period}'] = self.visualizer.plot_rebalance_period_returns(
        #         ls_returns, title=f'Long-Short Rebalance Period Returns (Period={period}d)'
        #     )
        # # 多空操作汇总表（每个周期一张）
        # # for period in periods:
        # #     ops_ls = all_results[period].get('ops_ls')
        # #     if ops_ls is not None and len(ops_ls) > 0:
        # #         figures['long_short'][f'ops_table_P{period}'] = self.visualizer.plot_operation_table(
        # #             ops_ls, title=f'Long-Short Operation Summary (Period={period}d)'
        # #         )
        
        # ===========================================
        # 多头回测图表
        # ===========================================
        
        # 图9: 分层收益柱状图（显示第1、5、10组在不同周期的表现）
        group_returns_by_period = {}
        for period in periods:
            group_returns_by_period[period] = all_results[period]['long_stats']
        
        figures['long_only']['group_returns_bar'] = self.visualizer.plot_group_returns_by_period(
            group_returns_by_period, groups_to_show=[1, 5, 10]
        )
        
        # 图10: 分组统计数据表格
        long_stats_combined = {f'Period_{period}': all_results[period]['long_stats'] for period in periods}
        figures['long_only']['stats_table'] = self.visualizer.plot_group_stats_table(
            long_stats_combined, 'Long-Only Performance by Group'
        )
        
        # 图11: 分层净值曲线（每个周期一张，包含所有10组）
        for period in periods:
            long_results = all_results[period]['long_results']
            figures['long_only'][f'group_nav_P{period}'] = self.visualizer.plot_group_nav_curves(
                long_results, title=f'Long-Only Group NAV (Period={period}d)'
            )
        # 多头调仓日收益率折线图（每组一条线）
        for period in periods:
            group_returns = all_results[period]['group_returns']
            figures['long_only'][f'period_returns_P{period}'] = self.visualizer.plot_rebalance_period_returns(
                group_returns, title=f'Long-Only Rebalance Period Returns (Period={period}d)'
            )
        # 多头操作汇总表（每个周期一张）- 已注释，不生成
        # for period in periods:
        #     ops_long = all_results[period].get('ops_long')
        #     if ops_long is not None and len(ops_long) > 0:
        #         ops_show = ops_long.head(50) if len(ops_long) > 50 else ops_long
        #         figures['long_only'][f'ops_table_P{period}'] = self.visualizer.plot_operation_table(
        #             ops_show, title=f'Long-Only Operation Summary (Period={period}d, first 50 rows)'
        #         )
        
        # ===========================================
        # 空头回测图表（已注释）
        # ===========================================

        # # 做空组1+2 绩效表
        # short_combined_stats_combined = {
        #     f'Period_{period}': all_results[period]['short_combined_stats'] for period in periods
        # }
        # figures['short_only']['performance_table_combined'] = self.visualizer.plot_performance_table(
        #     short_combined_stats_combined, 'Short G1+G2 Performance (做空组1和2)'
        # )
        # # 做空组1+2 净值与回撤（每个周期一张）
        # for period in periods:
        #     nav_sc = all_results[period]['short_combined_nav']
        #     figures['short_only'][f'nav_drawdown_combined_P{period}'] = self.visualizer.plot_nav_with_drawdown(
        #         nav_sc, title=f'Short G1+G2 NAV & Drawdown (Period={period}d)'
        #     )
        #
        # # 分层收益柱状图（各组做空）
        # short_returns_by_period = {}
        # for period in periods:
        #     short_returns_by_period[period] = all_results[period]['short_stats']
        # figures['short_only']['group_returns_bar'] = self.visualizer.plot_group_returns_by_period(
        #     short_returns_by_period, groups_to_show=[1, 5, 10], title='Short-Only Group Returns'
        # )
        #
        # # 分组统计数据表格（各组做空）
        # short_stats_combined = {f'Period_{period}': all_results[period]['short_stats'] for period in periods}
        # figures['short_only']['stats_table'] = self.visualizer.plot_group_stats_table(
        #     short_stats_combined, 'Short-Only Performance by Group'
        # )
        #
        # # 分层净值曲线（各组做空）
        # for period in periods:
        #     short_results = all_results[period]['short_results']
        #     figures['short_only'][f'group_nav_P{period}'] = self.visualizer.plot_group_nav_curves(
        #         short_results, title=f'Short-Only Group NAV (Period={period}d)'
        #     )
        # # 空头调仓日收益率折线图（每组一条线，做空收益 = -组收益）
        # for period in periods:
        #     group_returns = all_results[period]['group_returns']
        #     short_period_returns = -group_returns
        #     figures['short_only'][f'period_returns_P{period}'] = self.visualizer.plot_rebalance_period_returns(
        #         short_period_returns, title=f'Short-Only Rebalance Period Returns (Period={period}d)'
        #     )
        # # 空头操作汇总表（做空组1+2）
        # # for period in periods:
        # #     ops_sc = all_results[period].get('ops_short_combined')
        # #     if ops_sc is not None and len(ops_sc) > 0:
        # #         figures['short_only'][f'ops_table_P{period}'] = self.visualizer.plot_operation_table(
        # #             ops_sc, title=f'Short G1+G2 Operation Summary (Period={period}d)'
        # #         )
        
        print(f"生成图表完成")
        
        return figures
    
    def generate_report(self, all_results, figures):
        """生成PDF报告"""
        print("\nStep 9: 生成PDF报告")
        
        output_path = os.path.join(
            self.config.OUTPUT_DIR,
            f"{self.config.FACTOR_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )
        
        report_gen = FactorReportGeneratorOptimized(
            output_path,
            self.config.FACTOR_NAME,
            list(all_results.keys())
        )
        
        # 准备统计数据
        periods = sorted(all_results.keys())

        ic_stats_combined = {f'Period_{p}': all_results[p]['ic_stats'] for p in periods}
        long_stats_combined = {f'Period_{p}': all_results[p]['long_stats'] for p in periods}
        # ls_stats_combined = {f'Period_{p}': all_results[p]['ls_stats'] for p in periods}
        # short_stats_combined = {f'Period_{p}': all_results[p]['short_stats'] for p in periods}

        report_gen.generate(
            figures=figures,
            ic_stats=ic_stats_combined,
            ls_stats={},
            long_stats=long_stats_combined,
            short_stats={},
        )
        
        return output_path
    
    def run(self):
        """运行完整测试"""
        print("\n" + "="*60)
        print("单因子测试系统 - 优化版")
        print("="*60)
        
        try:
            # 加载数据
            self.load_data()
            
            # 运行所有调仓周期
            all_results = self.run_all_periods()
            
            # 生成合并的图表
            figures = self.generate_combined_figures(all_results)
            
            # 生成报告
            report_path = self.generate_report(all_results, figures)
            
            print(f"\n✓ 测试完成")
            print(f"  报告路径: {report_path}")
            
        except Exception as e:
            print(f"\n✗ 测试失败")
            print(f"  错误信息: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "="*60)
        print("测试完成！")
        print("="*60)


def main():
    tester = SingleFactorTesterOptimized()
    tester.run()


if __name__ == "__main__":
    main()