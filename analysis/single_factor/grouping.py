"""
因子分组模块 (grouping.py)
=====================================
本模块在每期横截面上按因子值排序并分成若干组，并计算组内加权收益。

主要类：GrouperEnhanced(factor, group_num, weight_method='equal')
- factor：按调仓期对齐的因子 DataFrame（行=日期，列=标的）。
- group_num：分组数（如 10）；Group 1 = 因子值最小，Group 10 = 因子值最大。
- weight_method：'equal' 等权，'factor_weight' 因子值加权（归一化后为权重）。

主要方法：
- split()：每期按因子值升序排序，均分（或最后一组含余数）得到 group_dict[date][group_num] = [stocks]。
- get_group_weights(group_dict)：按 weight_method 计算每组内各标的权重，返回 weight_dict[date][group_num][stock]。
- calculate_group_returns(group_dict, ret, weight_dict)：用当期的组内权重与标的收益率计算各组收益，返回 DataFrame（行=日期，列=组号）。供多空/多头/空头回测使用。
"""

import pandas as pd
import numpy as np

class GrouperEnhanced:
    """
    因子分组器 - 增强版
    支持等权重和因子值加权
    """

    def __init__(self, factor, group_num, weight_method='equal'):
        """
        Parameters:
        -----------
        factor: DataFrame, 因子数据
        group_num: int, 分组数量
        weight_method: str, 加权方式
            - 'equal': 等权重
            - 'factor_weight': 因子值加权（因子值归一化后作为权重）
        """
        self.factor = factor
        self.group_num = group_num
        self.weight_method = weight_method

    def split(self):
        """
        每期横截面排序分组（按因子值升序）
        Group 1 = 因子值最小，Group 10 = 因子值最大。
        对动量类因子，Group 10 通常表现最好；对反转类因子，Group 1 通常表现最好。
        
        Returns:
        --------
        dict: {date: {group_num: [stocks]}}
        """
        group_dict = {}

        for date in self.factor.index:
            f = self.factor.loc[date].dropna().sort_values()  # 升序：组1最小，组10最大
            n = len(f)
            
            # 如果有效股票数少于分组数，跳过
            if n < self.group_num:
                continue
            
            group_size = n // self.group_num

            groups = {}

            for i in range(self.group_num):
                start = i * group_size
                
                # 最后一组包含所有剩余股票
                if i == self.group_num - 1:
                    end = n
                else:
                    end = (i + 1) * group_size
                
                groups[i+1] = f.index[start:end].tolist()

            group_dict[date] = groups

            # #region agent log
            if len(group_dict) == 1:
                g1_vals = f.loc[groups[1]]
                g10_vals = f.loc[groups[10]]
                try:
                    with open(r"d:\qqq\.cursor\debug.log", "a", encoding="utf-8") as _f:
                        _f.write('{"id":"grp_check","timestamp":0,"location":"grouping.split","message":"Group1 vs Group10 factor","data":{"date":str(date),"g1_min":float(g1_vals.min()),"g1_max":float(g1_vals.max()),"g10_min":float(g10_vals.min()),"g10_max":float(g10_vals.max()),"g1_lt_g10":bool(g1_vals.max() < g10_vals.min())},"hypothesisId":"H1"}\n')
                except Exception:
                    pass
            # #endregion

        return group_dict
    
    def get_group_weights(self, group_dict):
        """
        计算每组内股票的权重
        
        Returns:
        --------
        dict: {date: {group_num: {stock: weight}}}
        """
        weight_dict = {}
        
        for date in group_dict.keys():
            weight_dict[date] = {}
            
            for group_num, stocks in group_dict[date].items():
                if self.weight_method == 'equal':
                    # 等权重
                    weights = {stock: 1.0 / len(stocks) for stock in stocks}
                
                elif self.weight_method == 'factor_weight':
                    # 因子值加权
                    factor_values = self.factor.loc[date, stocks]
                    
                    # 将因子值转换为正数（加上最小值的绝对值 + 一个小常数）
                    if factor_values.min() < 0:
                        factor_values = factor_values - factor_values.min() + 1e-8
                    
                    # 归一化
                    total = factor_values.sum()
                    if total > 0:
                        weights = (factor_values / total).to_dict()
                    else:
                        weights = {stock: 1.0 / len(stocks) for stock in stocks}
                
                else:
                    raise ValueError(f"Unknown weight method: {self.weight_method}")
                
                weight_dict[date][group_num] = weights
        
        return weight_dict
    
    def calculate_group_returns(self, group_dict, ret, weight_dict=None):
        """
        计算每组的收益率
        
        Parameters:
        -----------
        group_dict: dict, 分组结果
        ret: DataFrame, 收益率数据
        weight_dict: dict, 权重字典（如果为None，使用等权重）
        
        Returns:
        --------
        DataFrame: 每组的收益率 (index=date, columns=group_num)
        """
        group_returns = []
        dates = []
        
        for date in ret.index:
            if date not in group_dict:
                continue
            
            groups = group_dict[date]
            group_ret = {}
            
            for group_num, stocks in groups.items():
                # 获取该组股票的收益率
                stocks_valid = [s for s in stocks if s in ret.columns]
                if len(stocks_valid) == 0:
                    continue
                
                ret_values = ret.loc[date, stocks_valid]
                
                # 计算加权收益率
                if weight_dict is not None:
                    weights = weight_dict[date][group_num]
                    weights_valid = np.array([weights.get(s, 0) for s in stocks_valid])
                    
                    # 重新归一化权重（因为可能有些股票没有收益率数据）
                    weights_valid = weights_valid / weights_valid.sum()
                    
                    group_ret[group_num] = (ret_values * weights_valid).sum()
                else:
                    # 等权重
                    group_ret[group_num] = ret_values.mean()
            
            if len(group_ret) > 0:
                group_returns.append(group_ret)
                dates.append(date)
        
        group_returns_df = pd.DataFrame(group_returns, index=dates)
        
        return group_returns_df