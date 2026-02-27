"""
IC / Rank_IC 分析模块 (ic.py)
=====================================
本模块在「每个调仓期截面」上计算因子与收益的相关性，用于评估因子预测能力。

主要类：ICAnalyzerEnhanced(factor, ret)，其中 factor/ret 为按调仓期对齐的 DataFrame（每行一个调仓日，列为标的）。

主要方法：
- calculate_ic()：逐期计算 Pearson IC 与 Spearman Rank IC，返回 DataFrame 含列 ['IC', 'Rank_IC']。
- calculate_statistics(ic_series)：对 IC 序列计算均值、标准差、IR、偏度、峰度、t 检验、p 值、胜率、IC>0.02 比例。
- get_annual_ic(ic_df)：按年聚合得到年度 IC。
- get_monthly_ic(ic_df)：透视为年月矩阵，用于月度 IC 热力图。
- calculate_group_ic(group_dict, group_returns)：计算组号与组收益率的 Group IC / Group Rank IC，检验分组单调性。
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, skew, kurtosis, ttest_1samp

class ICAnalyzerEnhanced:
    """
    IC分析器 - 增强版
    支持多个调仓周期的IC分析
    """

    def __init__(self, factor, ret):
        """
        Parameters:
        -----------
        factor: DataFrame, 因子数据（每个调仓期一行）
        ret: DataFrame, 收益率数据（每个调仓期一行）
        """
        self.factor = factor
        self.ret = ret

    def calculate_ic(self):
        """
        计算每期的 IC 和 Rank IC
        
        Returns:
        --------
        DataFrame: columns=['IC', 'Rank_IC']
        """
        ic_list = []
        rank_ic_list = []
        dates = []

        for date in self.factor.index:
            f = self.factor.loc[date].dropna()
            r = self.ret.loc[date].dropna()
            
            # 找到共同的股票
            common_stocks = f.index.intersection(r.index)
            if len(common_stocks) < 2:
                continue
            
            f = f.loc[common_stocks]
            r = r.loc[common_stocks]

            # 常量输入（std=0）时相关系数无定义，直接记 NaN 避免警告
            if f.std() == 0 or r.std() == 0:
                ic_list.append(np.nan)
                rank_ic_list.append(np.nan)
                dates.append(date)
                continue

            # IC (Pearson correlation)
            ic = f.corr(r)

            # Rank IC (Spearman correlation)
            rank_ic, _ = spearmanr(f, r)

            ic_list.append(ic)
            rank_ic_list.append(rank_ic)
            dates.append(date)

        ic_df = pd.DataFrame({
            "IC": ic_list,
            "Rank_IC": rank_ic_list
        }, index=dates)

        return ic_df

    def calculate_statistics(self, ic_series):
        """
        计算 IC 统计量
        
        Parameters:
        -----------
        ic_series: Series, IC序列
        
        Returns:
        --------
        dict: 统计指标
        """
        mean = ic_series.mean()
        std = ic_series.std()
        ir = mean / std if std != 0 else np.nan
        clean = ic_series.dropna()
        n_valid = len(clean)
        # 小样本不调用 scipy，避免 SmallSampleWarning 并返回 NaN
        if n_valid >= 3:
            skewness = skew(clean)
            kurt = kurtosis(clean)
        else:
            skewness = np.nan
            kurt = np.nan
        if n_valid >= 2:
            t_stat, p_val = ttest_1samp(clean, 0)
        else:
            t_stat, p_val = np.nan, np.nan
        
        # 胜率
        win_rate = (ic_series > 0).mean()
        
        # IC > 0.02 的比例
        ic_gt_002 = (ic_series > 0.02).mean()

        return {
            "Mean": mean,
            "Std": std,
            "IR": ir,
            "Skew": skewness,
            "Kurtosis": kurt,
            "t_value": t_stat,
            "p_value": p_val,
            "Win_Rate": win_rate,
            "IC>0.02": ic_gt_002
        }
    
    def get_annual_ic(self, ic_df):
        """
        获取年度IC
        
        Returns:
        --------
        DataFrame: 每年的IC均值
        """
        ic_df = ic_df.copy()
        ic_df.index = pd.to_datetime(ic_df.index)
        ic_df['Year'] = ic_df.index.year
        annual_ic = ic_df.groupby('Year')[['IC', 'Rank_IC']].mean()
        return annual_ic
    
    def get_monthly_ic(self, ic_df):
        """
        获取月度IC
        
        Returns:
        --------
        DataFrame: 月度IC矩阵（行=年，列=月）
        """
        ic_df = ic_df.copy()
        ic_df.index = pd.to_datetime(ic_df.index)
        ic_df['Year'] = ic_df.index.year
        ic_df['Month'] = ic_df.index.month
        
        monthly_ic = ic_df.pivot_table(
            values='IC',
            index='Year',
            columns='Month',
            aggfunc='mean'
        )
        
        return monthly_ic

    def calculate_group_ic(self, group_dict, group_returns):
        """
        计算分组 IC（Group IC）：组号与组收益率的截面相关性，检验分组单调性。

        Parameters
        ----------
        group_dict : dict
            {date: {group_num: [stocks]}}
        group_returns : DataFrame
            每组的收益率，index=调仓日，columns=组号

        Returns
        -------
        DataFrame
            columns=['Group_IC', 'Group_Rank_IC'], index=调仓日
        """
        group_ic_list = []
        group_rank_ic_list = []
        dates = []
        for date in group_returns.index:
            group_nums = group_returns.columns
            group_ret = group_returns.loc[date]
            valid = group_ret.dropna()
            if len(valid) < 2:
                continue
            nums = valid.index.astype(float).values
            rets = valid.values

            # 常量输入（std=0）时相关系数无定义，直接记 NaN 避免警告
            if np.std(nums) == 0 or np.std(rets) == 0:
                group_ic_list.append(np.nan)
                group_rank_ic_list.append(np.nan)
                dates.append(date)
                continue

            ic = np.corrcoef(nums, rets)[0, 1]
            rank_ic, _ = spearmanr(nums, rets)
            group_ic_list.append(ic)
            group_rank_ic_list.append(rank_ic)
            dates.append(date)
        return pd.DataFrame(
            {"Group_IC": group_ic_list, "Group_Rank_IC": group_rank_ic_list},
            index=dates,
        )