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
  从复合因子已有日期序列中，按交易日间隔 ≥ rebalance_period_days 取样。
  即相邻调仓日之间至少相隔 rebalance_period_days 个交易日。

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

# 从共享工具模块导入（保持 _build_groups 独立定义以兼容已有调用）
from strategy_utils import _build_groups
from portfolio_optimizer import compute_weights
from rebalance_calendar import get_rebalance_calendar as _get_rebalance_calendar


# ---------------------------------------------------------------------------
# 分组工具（独立函数，与 GrouperEnhanced 逻辑一致）
# ---------------------------------------------------------------------------

def _select_rebalance_dates(
    factor_index: pd.DatetimeIndex,
    ret_index: pd.DatetimeIndex,
    rebalance_period_days: int,
) -> list:
    """
    从因子日期序列中，选取交易日间隔 ≥ rebalance_period_days 的节点。
    委托至 rebalance_calendar.get_rebalance_calendar 统一实现。
    """
    return _get_rebalance_calendar(factor_index, ret_index, rebalance_period_days)


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
            self.factor_df.index,
            self.ret_df.index,
            rebalance_period,
        )
        if len(rebalance_dates) < 2:
            return self._empty_result()

        # 将最后一期持仓延伸到收益率数据最后一个交易日，
        # 避免因 "最后一个调仓日" 造成净值曲线提前截断。
        end_date = self.ret_df.index.max()
        if rebalance_dates[-1] < end_date:
            rebalance_dates = list(rebalance_dates) + [end_date]

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
            # 历史收益率：取调仓日之前的最近 lookback 期数据（避免使用全部历史）
            hist_ret = self.ret_df.loc[
                self.ret_df.index < rb_date, :
            ].tail(getattr(cfg, "OPTIMIZATION_LOOKBACK", 252))

            weights = compute_weights(
                method=weight_method,
                stocks=group_stocks,
                factor_values=factor_signal,
                hist_returns=hist_ret,
                lookback=getattr(cfg, "OPTIMIZATION_LOOKBACK", 252),
                rf=getattr(cfg, "RISK_FREE_RATE", 0.02),
                max_weight=getattr(cfg, "MAX_WEIGHT", 0.4),
            )

            # ── 持仓期收益（向量化替代 iterrows）──────────────────────────
            # 持仓区间：(rb_date, next_rb]，跳过 rb_date 当日（T+1 时序）
            holding_mask = (
                (self.ret_df.index > rb_date) & (self.ret_df.index <= next_rb)
            )
            period_df = self.ret_df.loc[holding_mask, :]

            if len(period_df) == 0:
                continue

            # 有效标的：至少需要有历史收益率数据
            port_stocks = weights.index
            if len(port_stocks) == 0:
                continue

            # ── 持仓期收益（向量化）───────────────────────────────
            # 取本组合的权重索引（仅目标分组内的标的）
            # period_df 中只有 port_stocks 列有实际意义
            ret_port = period_df[port_stocks].copy()
            # 重索引权重（不在组合内的标的 → NaN，乘以 0 掩码后不影响结果）
            w_all = weights.reindex(ret_port.columns).fillna(0.0)

            # 有效掩码：权重非零 × 收益非空
            valid_mask = (w_all != 0) & ret_port.notna()
            # 按日期归一化权重（每日仅对当日有效的股票归一）
            row_sum = valid_mask.mul(w_all).sum(axis=1)  # Series: date → sum(w * valid)
            w_norm = valid_mask.mul(w_all).div(row_sum, axis=0)  # DataFrame: date × stock
            # 防御性列对齐：确保 w_norm 列顺序与 ret_port 完全一致
            w_norm = w_norm[ret_port.columns]

            port_ret_all = (w_norm * ret_port).sum(axis=1)  # Series: date → ret
            port_ret_all = port_ret_all.fillna(0.0)

            period_daily = port_ret_all.to_list()

            # 交易成本：持仓期首日扣除往返成本（买入+卖出各一次）
            period_daily[0] -= 2 * getattr(cfg, "TRANSACTION_COST", 0.001)

            all_daily_rets.extend(period_daily)
            all_dates.extend(period_daily.index.tolist())

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
