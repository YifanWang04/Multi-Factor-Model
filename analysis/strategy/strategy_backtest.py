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
from types import SimpleNamespace

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# 从共享工具模块导入（保持 _build_groups 独立定义以兼容已有调用）
from strategy_utils import _build_groups, _get_price_on_date
from portfolio_optimizer import compute_weights
from rebalance_calendar import (
    RebalanceAnchorError,
    RebalanceCalendarError,
    get_rebalance_calendar as _get_rebalance_calendar,
)
from tp_sl_exit import (
    EXIT_DYNAMIC_TP_SL,
    EXIT_FIXED_REBALANCE,
    build_exit_events,
    event_counts,
    normalize_exit_policy,
)
from qqq_core.parallel import get_max_workers, ordered_parallel_map


class CompositeCalendarError(ValueError):
    """Raised when a composite factor calendar cannot support a strategy period."""


def _fixed_week_value_for_period(config, rebalance_period: int, field: str):
    """Return a fixed-week field only for the profile's configured P value."""

    value = getattr(config, field, None)
    if value is None:
        return None
    fixed_period = getattr(config, "FIXED_WEEK_REBALANCE_PERIOD", None)
    if fixed_period is None:
        fixed_period = rebalance_period
    return value if int(fixed_period) == int(rebalance_period) else None


_OPTIMIZER_WEIGHT_METHODS = {"min_variance", "mvo", "max_return"}


def _as_float_grid(value, default: float = 0.4) -> list[float]:
    """Normalize a scalar or iterable config value into a non-empty float list."""
    if value is None:
        return [float(default)]
    if isinstance(value, (str, bytes)):
        return [float(value)]
    try:
        values = list(value)
    except TypeError:
        return [float(value)]
    if not values:
        return [float(default)]
    return [float(v) for v in values]


def _max_weight_tag(max_weight: float) -> str:
    text = f"{float(max_weight) * 100:g}".replace(".", "p")
    return f"MW{text}"


# ---------------------------------------------------------------------------
# 分组工具（独立函数，与 GrouperEnhanced 逻辑一致）
# ---------------------------------------------------------------------------

def _select_rebalance_dates(
    factor_index: pd.DatetimeIndex,
    ret_index: pd.DatetimeIndex,
    rebalance_period_days: int,
    rebalance_anchor_date: str | pd.Timestamp | None = None,
    rebalance_interval_weeks: int | None = None,
    rebalance_weekday: int | None = None,
    rebalance_week_anchor_date: str | pd.Timestamp | None = None,
) -> list:
    """
    从因子日期序列中，选取交易日间隔 ≥ rebalance_period_days 的节点。
    委托至 rebalance_calendar.get_rebalance_calendar 统一实现。
    """
    selected = _get_rebalance_calendar(
        factor_index,
        ret_index,
        rebalance_period_days,
        anchor_date=rebalance_anchor_date,
        interval_weeks=rebalance_interval_weeks,
        weekday=rebalance_weekday,
        week_anchor_date=rebalance_week_anchor_date,
    )
    _assert_calendar_supports_requested_period(
        selected,
        ret_index,
        rebalance_period_days,
    )
    return selected


def _assert_calendar_supports_requested_period(
    rebalance_dates: list,
    ret_index: pd.DatetimeIndex,
    rebalance_period_days: int,
) -> None:
    """
    Fail fast when a sparse composite-factor calendar is coarser than the
    requested strategy period.

    Example: a P10 composite factor file used by a P5 strategy used to produce
    P10 historical backtests but P5 future extrapolation in rebalance reports.
    """
    if len(rebalance_dates) < 2:
        return

    ret_sorted = pd.DatetimeIndex(ret_index).sort_values()
    too_long = []
    for prev_date, cur_date in zip(rebalance_dates[:-1], rebalance_dates[1:]):
        n_trading_days = int(((ret_sorted > prev_date) & (ret_sorted <= cur_date)).sum())
        if n_trading_days > int(rebalance_period_days):
            too_long.append((pd.Timestamp(prev_date), pd.Timestamp(cur_date), n_trading_days))

    if not too_long:
        return

    examples = ", ".join(
        f"{start.date()}->{end.date()}={days}d"
        for start, end, days in too_long[:3]
    )
    raise CompositeCalendarError(
        "Composite factor calendar is coarser than the requested strategy "
        f"rebalance period P{int(rebalance_period_days)}d. "
        f"Examples: {examples}. Regenerate composite factors with matching "
        "REBALANCE_PERIOD or switch to a strategy_param that matches the "
        "composite factor frequency."
    )


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

    def __init__(
        self,
        factor_df: pd.DataFrame,
        ret_df: pd.DataFrame,
        config,
        price_df: pd.DataFrame | None = None,
        factor_dfs_by_period: dict[int, pd.DataFrame] | None = None,
    ):
        self.factor_df = factor_df
        self.ret_df = ret_df
        self.config = config
        self.price_df = price_df
        self.factor_dfs_by_period = {
            int(k): v for k, v in (factor_dfs_by_period or {}).items()
        }

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def run_grid(self) -> dict:
        """
        遍历所有参数组合，返回 {strategy_name: result_dict}。
        """
        combinations = self._all_combinations()
        total = len(combinations)
        config_snapshot = _strategy_config_snapshot(self.config)
        worker_count = get_max_workers(total)
        combo_tasks = [
            (idx, combo, self._should_tag_max_weight(combo[3]))
            for idx, combo in enumerate(combinations, start=1)
        ]
        chunk_size = max(1, int(np.ceil(len(combo_tasks) / max(1, worker_count))))
        chunks = [
            combo_tasks[i:i + chunk_size]
            for i in range(0, len(combo_tasks), chunk_size)
        ]
        tasks = [
            (
                chunk,
                total,
                self.factor_df,
                self.ret_df,
                config_snapshot,
                self.price_df,
                self.factor_dfs_by_period,
            )
            for chunk in chunks
        ]
        chunk_results = ordered_parallel_map(
            _run_strategy_chunk_worker,
            tasks,
            label="strategy_grid",
        )
        pairs = [pair for chunk in chunk_results for pair in chunk]
        return {name: result for name, result in pairs}
        results = {}

        for idx, combo in enumerate(combinations, start=1):
            (
                group_num,
                target_rank,
                rebalance_period,
                weight_method,
                max_weight,
                exit_policy,
                tp_base,
                sl_base,
                probability,
            ) = combo
            target_group = group_num - (target_rank - 1)
            base_name = (
                f"{weight_method}_{group_num}G_Top{target_rank}_P{rebalance_period}d"
            )
            if self._should_tag_max_weight(weight_method):
                base_name = f"{base_name}_{_max_weight_tag(max_weight)}"
            if exit_policy == EXIT_DYNAMIC_TP_SL:
                strategy_name = (
                    f"{base_name}__{exit_policy}__"
                    f"tp{int(round(float(tp_base) * 100)):02d}_"
                    f"sl{int(round(float(sl_base) * 100)):02d}"
                )
            else:
                strategy_name = f"{base_name}__{exit_policy}"
            print(f"  [{idx:>3}/{total}] {strategy_name}", flush=True)

            try:
                result = self._run_single(
                    group_num,
                    target_group,
                    rebalance_period,
                    weight_method,
                    max_weight=max_weight,
                    exit_policy=exit_policy,
                    tp_base=tp_base,
                    sl_base=sl_base,
                    probability=probability,
                )
                result["params"] = {
                    "group_num": group_num,
                    "target_group": target_group,
                    "target_rank": target_rank,
                    "rebalance_period": rebalance_period,
                    "weight_method": weight_method,
                    "max_weight": max_weight,
                    "exit_policy": exit_policy,
                    "tp_base": tp_base,
                    "sl_base": sl_base,
                    "probability": probability,
                    "rebalance_interval_weeks": _fixed_week_value_for_period(
                        self.config, rebalance_period, "REBALANCE_INTERVAL_WEEKS"
                    ),
                    "rebalance_weekday": _fixed_week_value_for_period(
                        self.config, rebalance_period, "REBALANCE_WEEKDAY"
                    ),
                    "rebalance_week_anchor_date": _fixed_week_value_for_period(
                        self.config, rebalance_period, "REBALANCE_WEEK_ANCHOR_DATE"
                    ),
                    **result.get("exit_stats", {}),
                }
                results[strategy_name] = result
            except (
                CompositeCalendarError,
                RebalanceAnchorError,
                RebalanceCalendarError,
            ):
                raise
            except Exception as exc:
                print(f"    [!] 跳过：{exc}")
                results[strategy_name] = {
                    **self._empty_result(),
                    "params": {
                        "group_num": group_num,
                        "target_group": target_group,
                        "target_rank": target_rank,
                        "rebalance_period": rebalance_period,
                        "weight_method": weight_method,
                        "max_weight": max_weight,
                        "exit_policy": exit_policy,
                        "tp_base": tp_base,
                        "sl_base": sl_base,
                        "probability": probability,
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
        max_weight: float | None = None,
        exit_policy: str = EXIT_FIXED_REBALANCE,
        tp_base: float | None = None,
        sl_base: float | None = None,
        probability: float = 1.0,
    ) -> dict:
        """
        运行单一参数组合的策略，返回日频收益率序列和期间收益序列。
        """
        factor_df = self.factor_dfs_by_period.get(int(rebalance_period), self.factor_df)
        rebalance_dates = _select_rebalance_dates(
            factor_df.index,
            self.ret_df.index,
            rebalance_period,
            rebalance_anchor_date=getattr(
                self.config,
                "REBALANCE_ANCHOR_DATE",
                None,
            ),
            rebalance_interval_weeks=_fixed_week_value_for_period(
                self.config,
                rebalance_period,
                "REBALANCE_INTERVAL_WEEKS",
            ),
            rebalance_weekday=_fixed_week_value_for_period(
                self.config,
                rebalance_period,
                "REBALANCE_WEEKDAY",
            ),
            rebalance_week_anchor_date=_fixed_week_value_for_period(
                self.config,
                rebalance_period,
                "REBALANCE_WEEK_ANCHOR_DATE",
            ),
        )
        if len(rebalance_dates) < 2:
            return self._empty_result()

        # 将最后一期持仓延伸到收益率数据最后一个交易日，
        # 避免因 "最后一个调仓日" 造成净值曲线提前截断。
        end_date = self.ret_df.index.max()
        if rebalance_dates[-1] < end_date:
            rebalance_dates = list(rebalance_dates) + [end_date]

        exit_policy = normalize_exit_policy(exit_policy)
        all_daily_rets: list[float] = []
        all_dates: list = []
        period_rets: list[float] = []
        period_dates: list = []
        exit_stats = {"tp_count": 0, "sl_count": 0, "forced_close_count": 0}

        cfg = self.config
        if max_weight is None:
            max_weight = getattr(cfg, "MAX_WEIGHT", 0.4)
        max_weight = float(max_weight)

        for i in range(len(rebalance_dates) - 1):
            rb_date = rebalance_dates[i]
            next_rb = rebalance_dates[i + 1]

            # ── 因子信号 ──────────────────────────────────────────────
            if rb_date in factor_df.index:
                signal_date = rb_date
            else:
                avail = factor_df.index[factor_df.index <= rb_date]
                if len(avail) == 0:
                    continue
                signal_date = avail[-1]

            factor_signal = factor_df.loc[signal_date]

            # ── 分组 ──────────────────────────────────────────────────
            groups = _build_groups(factor_signal, group_num)
            if target_group not in groups or len(groups[target_group]) == 0:
                continue
            group_stocks = groups[target_group]

            # ── 权重 ──────────────────────────────────────────────────
            # 历史收益率：取调仓日之前的最近 lookback 期数据（避免使用全部历史）
            # T close is the signal/entry timestamp; future holding returns
            # still start after T, so <= rb_date is not look-ahead.
            hist_ret = self.ret_df.loc[
                self.ret_df.index <= rb_date, :
            ].tail(getattr(cfg, "OPTIMIZATION_LOOKBACK", 252))

            weights = compute_weights(
                method=weight_method,
                stocks=group_stocks,
                factor_values=factor_signal,
                hist_returns=hist_ret,
                lookback=getattr(cfg, "OPTIMIZATION_LOOKBACK", 252),
                rf=getattr(cfg, "RISK_FREE_RATE", 0.02),
                max_weight=max_weight,
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
            if exit_policy == EXIT_DYNAMIC_TP_SL:
                if self.price_df is None or self.price_df.empty:
                    raise ValueError("dynamic_tp_sl requires an Adj Close price_df")

                buy_prices = _get_price_on_date(self.price_df, rb_date, list(port_stocks))
                common = (
                    pd.Index(port_stocks)
                    .intersection(buy_prices.index)
                    .intersection(period_df.columns)
                    .intersection(self.price_df.columns)
                )
                if len(common) == 0:
                    continue

                w_common = weights.reindex(common).fillna(0.0)
                w_sum = float(w_common.sum())
                if not np.isfinite(w_sum) or abs(w_sum) < 1e-12:
                    continue
                w_common = w_common / w_sum

                buy_p = buy_prices.reindex(common).dropna()
                common = w_common.index.intersection(buy_p.index)
                if len(common) == 0:
                    continue
                w_common = w_common.reindex(common)
                buy_p = buy_p.reindex(common)

                events = build_exit_events(
                    price_df=self.price_df,
                    entry_prices=buy_p,
                    rb_date=rb_date,
                    exit_end_date=next_rb,
                    rebalance_period=rebalance_period,
                    tp_base=float(tp_base),
                    sl_base=float(sl_base),
                    probability=float(probability),
                )
                common = pd.Index([str(sym) for sym in common if str(sym) in events])
                if len(common) == 0:
                    continue

                ret_port = period_df[common].copy()
                for sym in common:
                    event = events[str(sym)]
                    ret_port.loc[ret_port.index > event.exit_date, sym] = 0.0

                port_ret_all = (
                    ret_port.fillna(0.0)
                    .mul(w_common.reindex(common), axis=1)
                    .sum(axis=1)
                    .dropna()
                )
                if port_ret_all.empty:
                    continue

                period_daily_vals = port_ret_all.values.copy()
                period_dates_list = port_ret_all.index.tolist()
                period_daily_vals[0] -= 2 * getattr(cfg, "TRANSACTION_COST", 0.001)

                all_daily_rets.extend(period_daily_vals.tolist())
                all_dates.extend(period_dates_list)

                period_cum = float(pd.Series(period_daily_vals).add(1.0).prod() - 1.0)
                period_rets.append(period_cum)
                period_dates.append(rb_date)

                counts = event_counts({str(sym): events[str(sym)] for sym in common})
                for key, value in counts.items():
                    exit_stats[key] += int(value)
                continue

            ret_port = period_df[port_stocks].copy()

            # 重索引权重（不在组合内的标的 → NaN，乘以 0 掩码后不影响结果）
            # 使用 loc 明确指定按列标签对齐，避免 pandas reindex 的维度混淆问题
            w_all = weights.reindex(ret_port.columns).fillna(0.0)

            # 有效掩码：权重非零 × 收益非空
            valid_mask = (w_all != 0) & ret_port.notna()
            # 按日期归一化权重（每日仅对当日有效的股票归一）
            row_sum = valid_mask.mul(w_all).sum(axis=1)  # Series: date → sum(w * valid)
            valid_days = row_sum > 1e-12
            if not valid_days.any():
                continue
            ret_port = ret_port.loc[valid_days]
            valid_mask = valid_mask.loc[valid_days]
            row_sum = row_sum.loc[valid_days]
            w_norm = valid_mask.mul(w_all).div(row_sum, axis=0)  # DataFrame: date × stock
            # 防御性列对齐：确保 w_norm 列顺序与 ret_port 完全一致
            w_norm = w_norm[ret_port.columns]

            port_ret_all = (w_norm * ret_port).sum(axis=1, min_count=1).dropna()  # Series: date → ret
            if port_ret_all.empty:
                continue

            # ── 持仓期收益计算：纯 numpy 位置运算，规避 pandas 索引对齐问题 ──
            # port_ret_all.values: 持仓期天数 × 1，扁平 numpy 数组
            period_daily_vals = port_ret_all.values.copy()   # 可写的 numpy array
            period_dates_list = port_ret_all.index.tolist()   # 日期列表

            # 交易成本：持仓期首日扣除往返成本（买入+卖出各一次）
            period_daily_vals[0] -= 2 * getattr(cfg, "TRANSACTION_COST", 0.001)

            all_daily_rets.extend(period_daily_vals.tolist())
            all_dates.extend(period_dates_list)

            # 期间总收益率（用于开仓统计）
            if len(period_daily_vals) > 0:
                period_cum = float(
                    pd.Series(period_daily_vals).add(1.0).prod() - 1.0
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
            "exit_stats": exit_stats,
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
                        for mw in self._max_weight_grid_for_method(wm):
                            for ep in getattr(cfg, "EXIT_POLICY_GRID", [EXIT_FIXED_REBALANCE]):
                                ep = normalize_exit_policy(ep)
                                if ep == EXIT_DYNAMIC_TP_SL:
                                    for tp in getattr(cfg, "TP_BASE_GRID", [getattr(cfg, "TP_BASE", 0.08)]):
                                        for sl in getattr(cfg, "SL_BASE_GRID", [getattr(cfg, "SL_BASE", 0.05)]):
                                            combos.append((
                                                gn,
                                                tr,
                                                rp,
                                                wm,
                                                mw,
                                                ep,
                                                float(tp),
                                                float(sl),
                                                float(getattr(cfg, "TP_SL_PROBABILITY", 1.0)),
                                            ))
                                else:
                                    combos.append((gn, tr, rp, wm, mw, ep, np.nan, np.nan, np.nan))
        return combos

    def _max_weight_grid_for_method(self, weight_method: str) -> list[float]:
        cfg = self.config
        default = float(getattr(cfg, "MAX_WEIGHT", 0.4))
        if weight_method not in _OPTIMIZER_WEIGHT_METHODS:
            return [default]
        return _as_float_grid(getattr(cfg, "MAX_WEIGHT_GRID", default), default)

    def _should_tag_max_weight(self, weight_method: str) -> bool:
        return (
            weight_method in _OPTIMIZER_WEIGHT_METHODS
            and len(self._max_weight_grid_for_method(weight_method)) > 1
        )

    @staticmethod
    def _empty_result() -> dict:
        return {
            "daily_returns": pd.Series(dtype=float),
            "nav": pd.Series(dtype=float),
            "rebalance_dates": [],
            "rebalance_returns": pd.Series(dtype=float),
            "exit_stats": {"tp_count": 0, "sl_count": 0, "forced_close_count": 0},
        }


def _strategy_config_snapshot(config) -> SimpleNamespace:
    """Make the config picklable for process-pool workers."""
    fields = [
        "OPTIMIZATION_LOOKBACK",
        "RISK_FREE_RATE",
        "TRANSACTION_COST",
        "MAX_WEIGHT",
        "DATA_DOWNLOAD_START_DATE",
        "REBALANCE_ANCHOR_DATE",
        "REBALANCE_INTERVAL_WEEKS",
        "REBALANCE_WEEKDAY",
        "REBALANCE_WEEK_ANCHOR_DATE",
        "FIXED_WEEK_REBALANCE_PERIOD",
    ]
    return SimpleNamespace(**{name: getattr(config, name, None) for name in fields})


def _strategy_name_for_combo(combo, tag_max_weight: bool) -> tuple[str, dict]:
    (
        group_num,
        target_rank,
        rebalance_period,
        weight_method,
        max_weight,
        exit_policy,
        tp_base,
        sl_base,
        probability,
    ) = combo
    target_group = group_num - (target_rank - 1)
    base_name = f"{weight_method}_{group_num}G_Top{target_rank}_P{rebalance_period}d"
    if tag_max_weight:
        base_name = f"{base_name}_{_max_weight_tag(max_weight)}"
    if exit_policy == EXIT_DYNAMIC_TP_SL:
        strategy_name = (
            f"{base_name}__{exit_policy}__"
            f"tp{int(round(float(tp_base) * 100)):02d}_"
            f"sl{int(round(float(sl_base) * 100)):02d}"
        )
    else:
        strategy_name = f"{base_name}__{exit_policy}"
    params = {
        "group_num": group_num,
        "target_group": target_group,
        "target_rank": target_rank,
        "rebalance_period": rebalance_period,
        "weight_method": weight_method,
        "max_weight": max_weight,
        "exit_policy": exit_policy,
        "tp_base": tp_base,
        "sl_base": sl_base,
        "probability": probability,
    }
    return strategy_name, params


def _run_strategy_combo_worker(task):
    (
        idx,
        total,
        combo,
        factor_df,
        ret_df,
        config,
        price_df,
        factor_dfs_by_period,
        tag_max_weight,
    ) = task
    strategy_name, params = _strategy_name_for_combo(combo, tag_max_weight)
    print(f"  [{idx:>3}/{total}] {strategy_name}", flush=True)
    try:
        backtester = StrategyBacktester(
            factor_df,
            ret_df,
            config,
            price_df=price_df,
            factor_dfs_by_period=factor_dfs_by_period,
        )
        result = backtester._run_single(
            params["group_num"],
            params["target_group"],
            params["rebalance_period"],
            params["weight_method"],
            max_weight=params["max_weight"],
            exit_policy=params["exit_policy"],
            tp_base=params["tp_base"],
            sl_base=params["sl_base"],
            probability=params["probability"],
        )
        result["params"] = {
            **params,
            "requested_data_download_start": getattr(
                config,
                "DATA_DOWNLOAD_START_DATE",
                None,
            ),
            "requested_rebalance_anchor": getattr(
                config,
                "REBALANCE_ANCHOR_DATE",
                None,
            ),
            "rebalance_interval_weeks": _fixed_week_value_for_period(
                config,
                params["rebalance_period"],
                "REBALANCE_INTERVAL_WEEKS",
            ),
            "rebalance_weekday": _fixed_week_value_for_period(
                config,
                params["rebalance_period"],
                "REBALANCE_WEEKDAY",
            ),
            "rebalance_week_anchor_date": _fixed_week_value_for_period(
                config,
                params["rebalance_period"],
                "REBALANCE_WEEK_ANCHOR_DATE",
            ),
            "effective_rebalance_anchor": (
                str(pd.Timestamp(result["rebalance_dates"][0]).date())
                if result.get("rebalance_dates")
                else None
            ),
            "effective_rebalance_start": (
                str(pd.Timestamp(result["rebalance_dates"][0]).date())
                if result.get("rebalance_dates")
                else None
            ),
            **result.get("exit_stats", {}),
        }
        return strategy_name, result
    except (CompositeCalendarError, RebalanceAnchorError, RebalanceCalendarError):
        raise
    except Exception as exc:
        print(f"    [!] skip {strategy_name}: {exc}", flush=True)
        return strategy_name, {
            **StrategyBacktester._empty_result(),
            "params": params,
        }


def _run_strategy_chunk_worker(task):
    (
        chunk,
        total,
        factor_df,
        ret_df,
        config,
        price_df,
        factor_dfs_by_period,
    ) = task
    out = []
    for idx, combo, tag_max_weight in chunk:
        out.append(
            _run_strategy_combo_worker(
                (
                    idx,
                    total,
                    combo,
                    factor_df,
                    ret_df,
                    config,
                    price_df,
                    factor_dfs_by_period,
                    tag_max_weight,
                )
            )
        )
    return out
