"""
Walk-Forward Validation Engine

核心引擎，执行滚动窗口验证：
1. 生成训练/测试窗口
2. 在训练窗口计算复合因子权重
3. 在测试窗口应用固定权重并进行策略网格搜索
4. 返回所有walk的所有策略结果

关键防泄露机制：
- 因子处理只使用训练期数据
- 复合因子权重只使用训练期IC/beta
- 投资组合优化只使用历史收益（已在strategy_backtest中实现）
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Tuple, Dict

# 添加项目路径
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from analysis.walk_forward import walk_forward_config as config
from analysis.walk_forward.rolling_data_processor import (
    process_factors_rolling,
    get_common_dates
)
from analysis.strategy.strategy_backtest import StrategyBacktester


class WalkForwardEngine:
    """
    Walk-Forward验证引擎

    执行流程：
    1. 加载价格数据
    2. 生成滚动窗口（训练/测试分割）
    3. 对每个窗口：
       a. 处理因子（只用训练期数据）
       b. 计算复合因子权重（只用训练期IC/beta）
       c. 应用权重到测试期
       d. 网格搜索策略参数
    4. 返回所有结果
    """

    def __init__(self, verbose: bool = True):
        """
        初始化引擎

        Args:
            verbose: 是否打印详细信息
        """
        self.config = config
        self.verbose = verbose

        # 加载价格数据
        print("Loading price data...")
        self.price_df = self._load_price_data()
        self.ret_df = self._compute_returns()

        # 获取因子文件列表
        self.factor_files = config.get_selected_factor_files()

        if self.verbose:
            print(f"[OK] Loaded price data: {self.price_df.shape}")
            print(f"[OK] Computed returns: {self.ret_df.shape}")
            print(f"[OK] Selected {len(self.factor_files)} factors")

    def _load_price_data(self) -> pd.DataFrame:
        """加载价格数据"""
        xl_file = pd.ExcelFile(config.PRICE_FILE)

        # 读取所有sheet并合并
        all_dfs = []
        for sheet_name in xl_file.sheet_names:
            df = pd.read_excel(config.PRICE_FILE, sheet_name=sheet_name)
            df['Date'] = pd.to_datetime(df['Date'])

            # 使用Adj Close作为价格
            price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'

            # 提取Date和价格列，使用sheet名称作为列名
            df_ticker = df[['Date', price_col]].copy()
            df_ticker.columns = ['Date', sheet_name]
            df_ticker = df_ticker.set_index('Date')

            all_dfs.append(df_ticker)

        # 合并所有股票
        df_combined = pd.concat(all_dfs, axis=1)
        return df_combined

    def _compute_returns(self) -> pd.DataFrame:
        """
        计算日频收益率

        价格数据已经是 Date × Ticker 格式
        """
        # 直接计算收益率
        ret_df = self.price_df.pct_change()
        return ret_df

    def run(self) -> List[Dict]:
        """
        执行walk-forward验证

        Returns:
            List[Dict]: 所有walk的所有策略结果
                每个dict包含：walk_id, train_period, test_period, strategy_name, metrics
        """
        # 生成窗口
        walks = self._generate_walk_windows()

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Generated {len(walks)} walk-forward windows")
            print(f"{'='*80}\n")

        all_results = []

        for walk_id, (train_start, train_end, test_start, test_end) in enumerate(walks):
            if self.verbose:
                print(f"\n{'='*80}")
                print(f"Walk {walk_id+1}/{len(walks)}")
                print(f"Train: [{train_start.date()}, {train_end.date()}] ({(train_end - train_start).days} days)")
                print(f"Test:  [{test_start.date()}, {test_end.date()}] ({(test_end - test_start).days} days)")
                print(f"{'='*80}\n")

            try:
                # Step 1: 处理因子（只用训练期数据）
                if self.verbose:
                    print("Step 1/4: Processing factors (training data only)...")
                processed_factors = self._process_factors_rolling(train_end)

                # Step 2: 计算复合因子权重（只用训练期数据）
                if self.verbose:
                    print("Step 2/4: Computing composite factor weights...")
                composite_factor_train = self._compute_composite_factor(
                    processed_factors, train_start, train_end
                )

                # Step 3: 应用权重到测试期
                if self.verbose:
                    print("Step 3/4: Applying composite formula to test period...")
                composite_factor_test = self._apply_composite_formula(
                    composite_factor_train, test_start, test_end
                )

                # Step 4: 网格搜索策略参数
                if self.verbose:
                    print("Step 4/4: Running strategy grid search...")
                walk_results = self._run_strategy_grid_search(
                    composite_factor_test, test_start, test_end
                )

                # 添加walk信息
                for result in walk_results:
                    result['walk_id'] = walk_id
                    result['train_period'] = (train_start, train_end)
                    result['test_period'] = (test_start, test_end)
                    all_results.append(result)

                if self.verbose:
                    print(f"\n[OK] Walk {walk_id+1} complete: {len(walk_results)} strategies tested")

            except Exception as e:
                print(f"\n[ERROR] Walk {walk_id+1} failed: {e}")
                import traceback
                traceback.print_exc()
                continue

        return all_results

    def _generate_walk_windows(self) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """
        生成滚动窗口

        Returns:
            List of (train_start, train_end, test_start, test_end)
        """
        dates = sorted(self.price_df.index)
        walks = []

        current_idx = 0
        while current_idx + config.TRAINING_WINDOW + config.TESTING_WINDOW <= len(dates):
            train_start = dates[current_idx]
            train_end = dates[current_idx + config.TRAINING_WINDOW - 1]

            # 考虑train-test gap
            test_start_idx = current_idx + config.TRAINING_WINDOW + config.TRAIN_TEST_GAP
            if test_start_idx >= len(dates):
                break

            test_start = dates[test_start_idx]
            test_end_idx = min(
                test_start_idx + config.TESTING_WINDOW - 1,
                len(dates) - 1
            )
            test_end = dates[test_end_idx]

            walks.append((train_start, train_end, test_start, test_end))
            current_idx += config.ROLL_FORWARD_STEP

        return walks

    def _process_factors_rolling(self, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """
        处理因子，只使用 <= end_date 的数据

        Args:
            end_date: 截止日期

        Returns:
            Dict[str, pd.DataFrame]: {因子名: 处理后的DataFrame}
        """
        return process_factors_rolling(
            self.factor_files,
            end_date,
            sheet_name=None,  # 使用第一个sheet
            verbose=self.verbose
        )

    def _compute_composite_factor(
        self,
        processed_factors: Dict[str, pd.DataFrame],
        train_start: datetime,
        train_end: datetime
    ) -> Dict:
        """
        计算复合因子权重（只使用训练期数据）

        Args:
            processed_factors: 处理后的因子数据
            train_start: 训练期开始日期
            train_end: 训练期结束日期

        Returns:
            Dict: 包含weights和method信息
        """
        # 过滤到训练期
        train_factors = {
            name: df.loc[train_start:train_end]
            for name, df in processed_factors.items()
        }

        # 获取训练期收益率
        train_ret = self.ret_df.loc[train_start:train_end]

        # 计算调仓期收益（与composite_factor.py逻辑一致）
        ret_periods = self._compute_period_returns(
            train_ret,
            list(train_factors.values())[0].index,
            config.COMPOSITE_REBALANCE_PERIOD
        )

        # 计算IC/beta（使用composite_factor.py的逻辑）
        stats_dict = self._compute_betas_ics(train_factors, ret_periods)

        # 根据配置的方法计算权重
        if config.COMPOSITE_METHOD == "beta_m3":
            # Beta加权，方法3（滚动窗口）
            weights = self._compute_univariate_weights(
                stats_dict, 'beta', method=3, window=config.N_WINDOW
            )
        else:
            raise ValueError(f"Unsupported composite method: {config.COMPOSITE_METHOD}")

        return {
            'weights': weights,
            'method': config.COMPOSITE_METHOD,
            'stats': stats_dict
        }

    def _apply_composite_formula(
        self,
        composite_info: Dict,
        test_start: datetime,
        test_end: datetime
    ) -> pd.DataFrame:
        """
        应用复合因子公式到测试期（使用训练期学到的权重）

        Args:
            composite_info: 包含weights的字典
            test_start: 测试期开始日期
            test_end: 测试期结束日期

        Returns:
            pd.DataFrame: 测试期的复合因子值
        """
        # 处理测试期因子（使用test_end作为截止日期）
        test_factors = process_factors_rolling(
            self.factor_files,
            test_end,
            sheet_name=None,
            verbose=False
        )

        # 过滤到测试期
        test_factors = {
            name: df.loc[test_start:test_end]
            for name, df in test_factors.items()
        }

        # 获取权重DataFrame
        weights_df = composite_info['weights']

        # 应用权重合成复合因子
        composite_factor = self._composite_from_weights(
            test_factors,
            weights_df,
            test_start,
            test_end
        )

        return composite_factor

    def _run_strategy_grid_search(
        self,
        composite_factor: pd.DataFrame,
        test_start: datetime,
        test_end: datetime
    ) -> List[Dict]:
        """
        在测试期进行策略网格搜索

        Args:
            composite_factor: 复合因子DataFrame
            test_start: 测试期开始日期
            test_end: 测试期结束日期

        Returns:
            List[Dict]: 所有策略的结果
        """
        # 获取测试期收益率
        test_ret = self.ret_df.loc[test_start:test_end]

        # 创建一个临时config对象用于StrategyBacktester
        class TempConfig:
            GROUP_NUMS = config.GROUP_NUMS
            TARGET_GROUP_RANKS = config.TARGET_GROUP_RANKS
            REBALANCE_PERIODS = config.REBALANCE_PERIODS
            WEIGHT_METHODS = config.WEIGHT_METHODS
            OPTIMIZATION_LOOKBACK = config.OPTIMIZATION_LOOKBACK
            MAX_WEIGHT = config.MAX_WEIGHT
            TRANSACTION_COST = config.TRANSACTION_COST
            RISK_FREE_RATE = config.RISK_FREE_RATE

        # 使用StrategyBacktester进行网格搜索
        backtester = StrategyBacktester(composite_factor, test_ret, TempConfig())
        strategy_results = backtester.run_grid()

        # 转换结果格式
        results = []
        for strategy_name, result in strategy_results.items():
            # 计算性能指标
            metrics = self._compute_metrics(result)

            results.append({
                'strategy_name': strategy_name,
                'group_num': result['params']['group_num'],
                'target_rank': result['params']['target_rank'],
                'rebalance_period': result['params']['rebalance_period'],
                'weight_method': result['params']['weight_method'],
                **metrics
            })

        return results

    # ========== 辅助函数 ==========

    def _compute_period_returns(
        self,
        ret_df: pd.DataFrame,
        rebalance_dates: pd.DatetimeIndex,
        rebalance_period: int
    ) -> pd.DataFrame:
        """
        计算调仓期收益率

        Args:
            ret_df: 日频收益率
            rebalance_dates: 调仓日期
            rebalance_period: 调仓周期（天）

        Returns:
            pd.DataFrame: 期间收益率（index=调仓日, columns=股票）
        """
        # 选择调仓日
        selected_dates = self._select_rebalance_dates(rebalance_dates, rebalance_period)

        period_returns = pd.DataFrame(index=selected_dates, columns=ret_df.columns)

        for i in range(len(selected_dates) - 1):
            current_date = selected_dates[i]
            next_date = selected_dates[i + 1]

            # 持仓期：(current_date, next_date]
            mask = (ret_df.index > current_date) & (ret_df.index <= next_date)
            period_ret = ret_df.loc[mask]

            if len(period_ret) > 0:
                # 累计收益率
                cumulative_ret = (1 + period_ret).prod() - 1
                period_returns.loc[current_date] = cumulative_ret

        return period_returns.dropna(how='all')

    def _select_rebalance_dates(
        self,
        dates: pd.DatetimeIndex,
        rebalance_period: int
    ) -> List[datetime]:
        """
        选择调仓日期

        Args:
            dates: 可用日期
            rebalance_period: 调仓周期（天）

        Returns:
            List[datetime]: 选中的调仓日期
        """
        dates = sorted(dates)
        if not dates:
            return []

        selected = [dates[0]]
        for d in dates[1:]:
            if (d - selected[-1]).days >= rebalance_period:
                selected.append(d)

        return selected

    def _compute_betas_ics(
        self,
        factor_dict: Dict[str, pd.DataFrame],
        ret_periods: pd.DataFrame
    ) -> Dict:
        """
        计算每个因子的beta和IC

        Args:
            factor_dict: {因子名: DataFrame}
            ret_periods: 期间收益率

        Returns:
            Dict: {因子名: {'beta': Series, 'ic': Series, 'rank_ic': Series}}
        """
        from scipy.stats import spearmanr

        result = {}
        dates = ret_periods.index

        for name, fdf in factor_dict.items():
            betas, ics, rank_ics = [], [], []
            valid_dates = []

            for d in dates:
                if d not in fdf.index:
                    continue

                f = fdf.loc[d].dropna()
                r = ret_periods.loc[d].dropna()
                common = f.index.intersection(r.index)

                if len(common) < 3:
                    continue

                fv, rv = f[common].values, r[common].values

                if np.std(fv) == 0 or np.std(rv) == 0:
                    continue

                # 确保是1D数组
                fv = np.asarray(fv).flatten()
                rv = np.asarray(rv).flatten()

                # Beta (OLS) - 使用手动计算避免np.cov的问题
                try:
                    cov_fr = np.mean((fv - np.mean(fv)) * (rv - np.mean(rv)))
                    var_f = np.var(fv)
                    if var_f > 0:
                        beta = cov_fr / var_f
                    else:
                        continue
                except:
                    continue

                # IC (Pearson) - 使用手动计算（避免numpy 2.x的bug）
                try:
                    # 手动计算Pearson相关系数
                    fv_mean = np.mean(fv)
                    rv_mean = np.mean(rv)
                    numerator = np.sum((fv - fv_mean) * (rv - rv_mean))
                    denominator = np.sqrt(np.sum((fv - fv_mean)**2) * np.sum((rv - rv_mean)**2))
                    if denominator > 0:
                        ic = numerator / denominator
                    else:
                        continue
                    if not np.isfinite(ic):
                        continue
                except:
                    continue

                # Rank IC (Spearman)
                try:
                    ric, _ = spearmanr(fv, rv)
                    if not np.isfinite(ric):
                        continue
                except:
                    continue

                betas.append(beta)
                ics.append(ic)
                rank_ics.append(ric)
                valid_dates.append(d)

            result[name] = {
                'beta': pd.Series(betas, index=valid_dates),
                'ic': pd.Series(ics, index=valid_dates),
                'rank_ic': pd.Series(rank_ics, index=valid_dates)
            }

        return result

    def _compute_univariate_weights(
        self,
        stats_dict: Dict,
        key: str,
        method: int,
        window: int = None
    ) -> pd.DataFrame:
        """
        计算单变量权重（beta/IC/rank_IC）

        Args:
            stats_dict: {因子名: {'beta': Series, 'ic': Series, ...}}
            key: 'beta', 'ic', 或 'rank_ic'
            method: 1/2/3
            window: 方法3的窗口大小

        Returns:
            pd.DataFrame: 权重矩阵（index=日期, columns=因子名）
        """
        names = list(stats_dict.keys())
        series_map = {n: stats_dict[n][key] for n in names}

        # 获取所有日期
        all_dates = pd.DatetimeIndex([])
        for s in series_map.values():
            all_dates = all_dates.union(s.index)
        all_dates = all_dates.sort_values()

        if method == 1:
            # 全期均值（含前瞻，仅供研究）
            rows = []
            for d in all_dates:
                row = {}
                for n in names:
                    s = series_map[n]
                    past = s[s.index < d]
                    row[n] = past.mean() if len(past) > 0 else np.nan
                rows.append(row)
            weight_df = pd.DataFrame(rows, index=all_dates)

        elif method == 2:
            # 扩展窗口（无前瞻）
            rows = []
            for d in all_dates:
                row = {}
                for n in names:
                    s = series_map[n]
                    past = s[s.index < d]
                    row[n] = past.mean() if len(past) > 0 else np.nan
                rows.append(row)
            weight_df = pd.DataFrame(rows, index=all_dates)

        else:  # method == 3
            # 滚动窗口（无前瞻）
            assert window is not None
            rows = []
            for d in all_dates:
                row = {}
                for n in names:
                    s = series_map[n]
                    past = s[s.index < d].iloc[-window:] if len(s[s.index < d]) > 0 else pd.Series()
                    row[n] = past.mean() if len(past) > 0 else np.nan
                rows.append(row)
            weight_df = pd.DataFrame(rows, index=all_dates)

        return weight_df

    def _composite_from_weights(
        self,
        factor_dict: Dict[str, pd.DataFrame],
        weight_df: pd.DataFrame,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        根据权重合成复合因子

        Args:
            factor_dict: {因子名: DataFrame}
            weight_df: 权重DataFrame（index=日期, columns=因子名）
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            pd.DataFrame: 复合因子
        """
        # 获取日期范围
        dates = pd.date_range(start_date, end_date, freq='D')
        dates = dates.intersection(list(factor_dict.values())[0].index)

        # 获取股票列表
        stocks = list(factor_dict.values())[0].columns

        # 初始化结果
        result = pd.DataFrame(np.nan, index=dates, columns=stocks)

        names = list(factor_dict.keys())

        for d in dates:
            # 如果日期不在权重索引中，使用最近的历史权重（向前填充）
            if d not in weight_df.index:
                # 找到最近的历史权重
                past_weights = weight_df[weight_df.index <= d]
                if len(past_weights) == 0:
                    continue
                ws = past_weights.iloc[-1]
            else:
                ws = weight_df.loc[d]
            row = pd.Series(0.0, index=stocks)
            total_w = 0.0

            for name in names:
                w = ws.get(name, np.nan)
                if np.isnan(w):
                    continue

                if d in factor_dict[name].index:
                    frow = factor_dict[name].loc[d]
                    row = row.add(frow.reindex(stocks) * w, fill_value=0)
                    total_w += abs(w)

            if total_w > 0:
                result.loc[d] = row / total_w
            else:
                result.loc[d] = row

        return result

    def _compute_metrics(self, result: Dict) -> Dict:
        """
        计算策略性能指标

        Args:
            result: StrategyBacktester返回的结果

        Returns:
            Dict: 性能指标
        """
        daily_returns = result.get('daily_returns', pd.Series())

        if len(daily_returns) == 0:
            return {
                'sharpe': np.nan,
                'annual_return': np.nan,
                'annual_vol': np.nan,
                'max_drawdown': np.nan,
                'calmar': np.nan,
                'total_return': np.nan,
                'num_periods': 0
            }

        # 年化收益率
        total_return = (1 + daily_returns).prod() - 1
        num_days = len(daily_returns)
        annual_return = (1 + total_return) ** (252 / num_days) - 1

        # 年化波动率
        annual_vol = daily_returns.std() * np.sqrt(252)

        # Sharpe比率
        rf_daily = config.RISK_FREE_RATE / 252
        excess_return = daily_returns - rf_daily
        sharpe = excess_return.mean() / excess_return.std() * np.sqrt(252) if excess_return.std() > 0 else 0

        # 最大回撤
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Calmar比率
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else np.nan

        return {
            'sharpe': sharpe,
            'annual_return': annual_return,
            'annual_vol': annual_vol,
            'max_drawdown': max_drawdown,
            'calmar': calmar,
            'total_return': total_return,
            'num_periods': len(result.get('rebalance_dates', []))
        }


if __name__ == "__main__":
    print("=" * 80)
    print("Walk-Forward Validation Engine Test")
    print("=" * 80)

    engine = WalkForwardEngine(verbose=True)

    print("\nTesting window generation...")
    walks = engine._generate_walk_windows()
    print(f"Generated {len(walks)} walks:")
    for i, (train_start, train_end, test_start, test_end) in enumerate(walks[:3]):
        print(f"  Walk {i+1}: Train [{train_start.date()}, {train_end.date()}], Test [{test_start.date()}, {test_end.date()}]")
    if len(walks) > 3:
        print(f"  ... and {len(walks)-3} more walks")
