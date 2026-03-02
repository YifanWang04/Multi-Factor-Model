"""
策略回测引擎 (strategy_backtest.py)
=====================================
核心职责：网格遍历所有参数组合（分层数、目标组、调仓周期、资产配置方式），
对每个组合计算日频组合收益率序列，以及每持仓周期的期间收益率序列（供开仓统计）。

关键时序约定（与现有 rebalance_manager.py 保持一致）：
  - 调仓日 T：使用当日可用的复合因子信号（信号本身已在 composite_factor.py 中
    使用 < current_date 的数据构建，无前瞻偏误）
  - 持仓区间：(T, T_next]，即 T 当日不计入持仓收益，从 T+1 开始
  - 交易成本：在每个持仓周期首个交易日扣除一次单边成本

调仓日选取逻辑：
  从复合因子已有日期序列中，按日历天数间隔 ≥ rebalance_period_days 取样。
  若请求周期短于因子原生间隔，则使用全部因子日期（取样不再稀释）。

输出结构（每个策略名 → dict）：
  daily_returns     : pd.Series（日期 → 日收益率，覆盖全持仓区间）
  nav               : pd.Series（净值曲线）
  rebalance_dates   : list（实际使用的调仓日列表）
  rebalance_returns : pd.Series（调仓日 → 该期间总收益率，用于开仓统计）
  params            : dict（该策略的参数字典）
"""

import sys
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from portfolio_optimizer import compute_weights


# ---------------------------------------------------------------------------
# 分组工具（独立函数，与 GrouperEnhanced 逻辑一致）
# ---------------------------------------------------------------------------

def _build_groups(factor_signal: pd.Series, group_num: int) -> dict:
    """
    按因子值升序排序后均分为 group_num 组；最后一组包含余数。
    返回 {group_id(1-based): [stocks]}，group_num = 最高因子值组。
    """
    f = factor_signal.dropna().sort_values(ascending=True)
    n = len(f)
    if n < group_num:
        return {}

    group_size = n // group_num
    groups = {}
    for i in range(group_num):
        start = i * group_size
        end = n if i == group_num - 1 else (i + 1) * group_size
        groups[i + 1] = f.index[start:end].tolist()
    return groups


def _select_rebalance_dates(factor_index: pd.DatetimeIndex,
                             rebalance_period_days: int) -> list:
    """
    从因子日期序列中，选取日历间隔 ≥ rebalance_period_days 的节点。
    保证第一个因子日期始终被选入。
    """
    dates = sorted(factor_index)
    if not dates:
        return []

    selected = [dates[0]]
    for d in dates[1:]:
        if (d - selected[-1]).days >= rebalance_period_days:
            selected.append(d)
    return selected


# ---------------------------------------------------------------------------
# 策略回测引擎
# ---------------------------------------------------------------------------

class StrategyBacktester:
    """
    多参数组合网格回测引擎。

    Parameters
    ----------
    factor_df  : 复合因子 DataFrame（index=调仓日, columns=股票）
    ret_df     : 日频收益率 DataFrame（index=日期, columns=股票）
    config     : strategy_config 模块（提供 GROUP_NUMS / REBALANCE_PERIODS 等）
    """

    def __init__(self, factor_df: pd.DataFrame, ret_df: pd.DataFrame, config):
        self.factor_df = factor_df
        self.ret_df = ret_df
        self.config = config

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def run_grid(self) -> dict:
        """
        遍历所有参数组合，返回 {strategy_name: result_dict}。
        """
        combinations = self._all_combinations()
        total = len(combinations)
        results = {}

        for idx, (group_num, target_rank, rebalance_period, weight_method) in enumerate(
            combinations, start=1
        ):
            target_group = group_num - (target_rank - 1)
            strategy_name = (
                f"{weight_method}_{group_num}G_Top{target_rank}_P{rebalance_period}d"
            )
            print(f"  [{idx:>3}/{total}] {strategy_name}", flush=True)

            try:
                result = self._run_single(
                    group_num, target_group, rebalance_period, weight_method
                )
                result["params"] = {
                    "group_num": group_num,
                    "target_group": target_group,
                    "target_rank": target_rank,
                    "rebalance_period": rebalance_period,
                    "weight_method": weight_method,
                }
                results[strategy_name] = result
            except Exception as exc:
                print(f"    ⚠  跳过：{exc}")
                results[strategy_name] = {
                    **self._empty_result(),
                    "params": {
                        "group_num": group_num,
                        "target_group": target_group,
                        "target_rank": target_rank,
                        "rebalance_period": rebalance_period,
                        "weight_method": weight_method,
                    },
                }

        return results

    # ------------------------------------------------------------------
    # 内部：单策略回测
    # ------------------------------------------------------------------

    def _run_single(
        self,
        group_num: int,
        target_group: int,
        rebalance_period: int,
        weight_method: str,
    ) -> dict:
        """
        运行单一参数组合的策略，返回日频收益率序列和期间收益序列。
        """
        rebalance_dates = _select_rebalance_dates(
            self.factor_df.index, rebalance_period
        )
        if len(rebalance_dates) < 2:
            return self._empty_result()

        all_daily_rets: list[float] = []
        all_dates: list = []
        period_rets: list[float] = []
        period_dates: list = []

        cfg = self.config

        for i in range(len(rebalance_dates) - 1):
            rb_date = rebalance_dates[i]
            next_rb = rebalance_dates[i + 1]

            # ── 因子信号 ──────────────────────────────────────────────
            if rb_date in self.factor_df.index:
                signal_date = rb_date
            else:
                avail = self.factor_df.index[self.factor_df.index <= rb_date]
                if len(avail) == 0:
                    continue
                signal_date = avail[-1]

            factor_signal = self.factor_df.loc[signal_date]

            # ── 分组 ──────────────────────────────────────────────────
            groups = _build_groups(factor_signal, group_num)
            if target_group not in groups or len(groups[target_group]) == 0:
                continue
            group_stocks = groups[target_group]

            # ── 权重 ──────────────────────────────────────────────────
            hist_ret = self.ret_df.loc[
                self.ret_df.index < rb_date, :
            ]
            weights = compute_weights(
                method=weight_method,
                stocks=group_stocks,
                factor_values=factor_signal,
                hist_returns=hist_ret,
                lookback=getattr(cfg, "OPTIMIZATION_LOOKBACK", 252),
                rf=getattr(cfg, "RISK_FREE_RATE", 0.02),
                max_weight=getattr(cfg, "MAX_WEIGHT", 0.4),
            )

            # ── 持仓期收益 ─────────────────────────────────────────────
            # 持仓区间：(rb_date, next_rb]，跳过 rb_date 当日（T+1 时序）
            holding_mask = (
                (self.ret_df.index > rb_date) & (self.ret_df.index <= next_rb)
            )
            period_df = self.ret_df.loc[holding_mask, :]

            if len(period_df) == 0:
                continue

            period_daily: list[float] = []

            for j, (date, row) in enumerate(period_df.iterrows()):
                valid = weights.index[
                    weights.index.isin(row.dropna().index)
                ]
                if len(valid) == 0:
                    port_ret = 0.0
                else:
                    w = weights[valid]
                    w = w / w.sum()
                    port_ret = float((row[valid] * w).sum())

                # 交易成本：持仓期首日扣除一次单边成本
                if j == 0:
                    port_ret -= getattr(cfg, "TRANSACTION_COST", 0.001)

                period_daily.append(port_ret)
                all_daily_rets.append(port_ret)
                all_dates.append(date)

            # 期间总收益率（用于开仓统计）
            if period_daily:
                period_cum = float(
                    pd.Series(period_daily).add(1.0).prod() - 1.0
                )
                period_rets.append(period_cum)
                period_dates.append(rb_date)

        if not all_dates:
            return self._empty_result()

        daily_returns = pd.Series(all_daily_rets, index=all_dates, name="port_ret")
        nav = (1.0 + daily_returns).cumprod()
        rebalance_returns = pd.Series(period_rets, index=period_dates, name="period_ret")

        return {
            "daily_returns": daily_returns,
            "nav": nav,
            "rebalance_dates": period_dates,
            "rebalance_returns": rebalance_returns,
        }

    # ------------------------------------------------------------------
    # 内部：辅助
    # ------------------------------------------------------------------

    def _all_combinations(self) -> list:
        cfg = self.config
        combos = []
        for gn in cfg.GROUP_NUMS:
            for rp in cfg.REBALANCE_PERIODS:
                for tr in cfg.TARGET_GROUP_RANKS:
                    for wm in cfg.WEIGHT_METHODS:
                        combos.append((gn, tr, rp, wm))
        return combos

    @staticmethod
    def _empty_result() -> dict:
        return {
            "daily_returns": pd.Series(dtype=float),
            "nav": pd.Series(dtype=float),
            "rebalance_dates": [],
            "rebalance_returns": pd.Series(dtype=float),
        }
