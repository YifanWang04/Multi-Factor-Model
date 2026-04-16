"""
Discord 通知模块 (discord_notifier.py)
=====================================
职责：
  - send_discord_notification: 发送 Discord Embed 通知（含绩效指标 + 持仓盈亏）
  - _get_holding_period_info: 获取当前持仓的盈亏情况
  - compute_extended_metrics: 从日收益率/Nav 计算完整绩效指标集

导出：
  send_discord_notification
  compute_extended_metrics
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import requests


# ---------------------------------------------------------------------------
# Discord 配置常量
# ---------------------------------------------------------------------------

DISCORD_WEBHOOK_URL = os.environ.get(
    "REBALANCE_DISCORD_WEBHOOK_URL",
    "https://discord.com/api/webhooks/1478641216659652709/TRe7zHYv0x5AbYJMngnJbi1TbjUwXiOhIct-rze0wHFFYgi-Yqt320iGOCY4J1NUbq68",
)

DISCORD_FIELD_MAX_CHARS: int = 1024
DISCORD_OPS_MAX_LINES: int = 20


# ---------------------------------------------------------------------------
# 绩效指标计算
# ---------------------------------------------------------------------------

def compute_extended_metrics(
    daily_returns: pd.Series,
    nav: pd.Series,
    rebalance_returns: pd.Series,
    rf_rate: float = 0.02,
) -> dict:
    """
    计算完整绩效指标集，供 Discord 通知使用。
    包含全局最大回撤 + 单周期最坏回撤。
    """
    if daily_returns.empty or nav.empty:
        return {}

    total_ret = float(nav.iloc[-1]) - 1.0 if len(nav) > 0 else float("nan")
    ann_ret = (1 + total_ret) ** (252 / max(1, len(daily_returns))) - 1 if len(daily_returns) > 0 else float("nan")
    vol = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else float("nan")
    sharpe = (ann_ret - rf_rate) / vol if vol and vol > 0 else float("nan")

    max_dd = float((nav / nav.cummax() - 1).min()) if len(nav) > 0 else float("nan")
    max_dd_pct = max_dd * 100
    calmar = ann_ret / abs(max_dd) if max_dd and max_dd != 0 else float("nan")

    # ── 单周期最坏回撤 ──────────────────────────────────────────────
    worst_dd = np.nan
    worst_dd_pct = np.nan
    if len(rebalance_returns) > 0:
        rb_dates = rebalance_returns.index.tolist()
        worst_val = 0.0
        for i, rb_start in enumerate(rb_dates):
            if i + 1 < len(rb_dates):
                rb_end = rb_dates[i + 1]
            else:
                if len(nav) == 0:
                    continue
                rb_end = nav.index[-1]
            period_ret = daily_returns[daily_returns.index > rb_start]
            if i + 1 < len(rb_dates):
                period_ret = period_ret[period_ret.index <= rb_end]
            if len(period_ret) == 0:
                continue
            if rb_start in nav.index:
                base_nav = nav.loc[rb_start]
            else:
                valid = nav.index[nav.index <= rb_start]
                if len(valid) == 0:
                    continue
                base_nav = nav.loc[valid[-1]]
            period_nav = (1.0 + period_ret).cumprod() * base_nav
            cummax = period_nav.cummax()
            dd = (period_nav - cummax) / cummax
            dd_min = dd.min()
            if dd_min < worst_val:
                worst_val = dd_min
        if worst_val < 0:
            worst_dd = float(worst_val)
            worst_dd_pct = worst_dd * 100
    # ── 单周期最坏回撤 end ──────────────��───────────────────────────

    win_days = int((daily_returns > 0).sum())
    total_days = len(daily_returns)
    win_rate = win_days / total_days if total_days > 0 else float("nan")

    avg_win = float(daily_returns[daily_returns > 0].mean()) if win_days > 0 else 0.0
    loss_days = int((daily_returns < 0).sum())
    avg_loss = float(daily_returns[daily_returns < 0].mean()) if loss_days > 0 else 0.0
    profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("nan")

    return {
        "total_return": total_ret,
        "annual_return": ann_ret,
        "volatility": vol,
        "volatility_pct": vol * 100 if not np.isnan(vol) else float("nan"),
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd_pct,
        "calmar": calmar,
        "win_rate": win_rate,
        "win_days": win_days,
        "total_days": total_days,
        "profit_loss_ratio": profit_loss_ratio,
        "worst_period_drawdown": worst_dd,
        "worst_period_drawdown_pct": worst_dd_pct,
    }


# ---------------------------------------------------------------------------
# 持仓盈亏信息
# ---------------------------------------------------------------------------

def _get_holding_period_info(
    operations_df: pd.DataFrame,
    current_rb: pd.Timestamp,
    price_df: pd.DataFrame,
    current_date: pd.Timestamp,
) -> Optional[pd.DataFrame]:
    """
    获取当前持仓的盈亏情况。
    传入 current_rb（当前调仓日，即这些仓位的买入日），
    返回含 Symbol / Buy_Price / Current_Price / Change_Pct 的 DataFrame。
    若无持仓或数据不足，返回 None。

    容错逻辑：当 current_rb 对应的记录不存在时，
    尝试使用 operations_df 中最新一期的持仓记录。
    """
    if operations_df.empty or "Rebalance_Date" not in operations_df.columns:
        return None

    ops = operations_df.copy()
    ops["Rebalance_Date"] = pd.to_datetime(ops["Rebalance_Date"], errors="coerce")
    ops["Next_Rebalance_Date"] = pd.to_datetime(ops["Next_Rebalance_Date"], errors="coerce")
    current_rb_ts = pd.Timestamp(current_rb)
    current_date_ts = pd.Timestamp(current_date)

    holding = ops[
        (ops["Rebalance_Date"] == current_rb_ts)
        & (
            ops["Next_Rebalance_Date"].isna()
            | (ops["Next_Rebalance_Date"] > current_date_ts)
        )
    ].copy()

    if holding.empty:
        latest_rb = ops["Rebalance_Date"].max()
        holding = ops[ops["Rebalance_Date"] == latest_rb].copy()

    if "Weight" in holding.columns:
        holding = holding.sort_values("Weight", ascending=False)
        holding = holding.drop_duplicates(subset=["Symbol"], keep="first")

    if holding.empty:
        return None

    if current_date_ts in price_df.index:
        current_prices = price_df.loc[current_date_ts]
    else:
        available = price_df.index[price_df.index <= current_date_ts]
        if len(available) == 0:
            return None
        current_prices = price_df.loc[available[-1]]

    holding["Current_Price"] = holding["Symbol"].map(
        lambda s: float(current_prices[s]) if s in current_prices.index else float("nan")
    )
    holding["Buy_Price"] = pd.to_numeric(holding["Buy_Price_Close"], errors="coerce")
    holding["Change_Pct"] = (holding["Current_Price"] - holding["Buy_Price"]) / holding["Buy_Price"]

    holding = holding.dropna(subset=["Buy_Price", "Current_Price", "Change_Pct"])
    if holding.empty:
        return None

    return holding[["Symbol", "Weight", "Buy_Price", "Current_Price", "Change_Pct", "Rebalance_Date"]].sort_values(
        by="Change_Pct", ascending=False
    )


# ---------------------------------------------------------------------------
# Discord 通知发送
# ---------------------------------------------------------------------------

def _fmt_metric(value: float, fmt: str) -> str:
    """安全格式化指标值，NaN 时返回 '-'。"""
    if isinstance(value, float) and np.isnan(value):
        return "-"
    return fmt.format(value)


def _trunc_text(text: str, max_chars: int) -> str:
    """截断文本并加省略号。"""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _filter_weight_lt(ops: pd.DataFrame, threshold: float = 0.0001) -> pd.DataFrame:
    """过滤 Weight 列 < threshold 的行。"""
    if "Weight" not in ops.columns:
        return ops
    before = len(ops)
    ops = ops[ops["Weight"] >= threshold].copy()
    removed = before - len(ops)
    if removed > 0:
        print(f"  过滤 Weight < {threshold}，移除 {removed} 行")
    return ops


def send_discord_notification(
    status: dict,
    current_ops: pd.DataFrame,
    result: dict,
    webhook_url: str = DISCORD_WEBHOOK_URL,
    used_live_prices: bool = False,
    # 外部传入的配置参数（避免直接引用 run_rebalance_day 中的模块级变量）
    selected_factor_names: Optional[list[str]] = None,
    composite_factor_sheet: str = "ic_m3_N20",
    strategy_param: str = "",
    strategy_params: Optional[dict] = None,
    data_start_offset_days: int = 0,
    rf_rate: float = 0.02,
) -> None:
    """
    发送 Discord 通知（含完整绩效指标 + 持仓盈亏，低权重操作已过滤）。

    Parameters
    ----------
    status : dict
        调仓日状态（来自 get_rebalance_day_status）
    current_ops : pd.DataFrame
        当前调仓日操作明细
    result : dict
        回测结果（含 daily_returns, nav, operations_df, _price_df 等）
    webhook_url : str
        Discord Webhook URL
    used_live_prices : bool
        是否使用了实时价格
    selected_factor_names : list[str]
        选定因子名称列表
    composite_factor_sheet : str
        复合因子方法名
    strategy_param : str
        策略参数字符串
    strategy_params : dict
        解析后的策略参数字典（weight_method, group_num, target_rank, rebalance_period）
    data_start_offset_days : int
        数据起始日偏移
    rf_rate : float
        无风险利率
    """
    if not webhook_url:
        print("未配置 Discord Webhook URL，跳过推送")
        return

    if strategy_params is None:
        strategy_params = {}
    if selected_factor_names is None:
        selected_factor_names = []

    try:
        as_of = pd.Timestamp(datetime.now().date())
        is_rebalance = status["is_rebalance_today"]
        current_rb = status["current_rebalance_date"]
        next_rb = status["next_rebalance_date"]

        factor_info = (
            f"**选定因子：** {', '.join(selected_factor_names)}\n"
            f"**复合因子：** {composite_factor_sheet}（IC加权 M3/N20）\n"
            f"**策略参数：** {strategy_param}\n"
            f"**权重方式：** {strategy_params.get('weight_method', '')}　"
            f"**分组数：** {strategy_params.get('group_num', '')}　"
            f"**目标组：** Top{strategy_params.get('target_rank', '')}　"
            f"**调仓周期：** {strategy_params.get('rebalance_period', '')} 交易日　"
            f"**数据起始日偏移：** {data_start_offset_days} 交易日\n"
        )

        dr = result.get("daily_returns", pd.Series(dtype=float))
        nv = result.get("nav", pd.Series(dtype=float))
        rb_rets = result.get("rebalance_returns", pd.Series(dtype=float))
        price_df = result.get("_price_df", pd.DataFrame())
        ops_df = result.get("operations_df", pd.DataFrame())
        metrics = compute_extended_metrics(dr, nv, rb_rets, rf_rate=rf_rate)

        # ── 当前持仓盈亏区块 ───────────────────────────────────────────
        holding_field: Optional[dict] = None
        if current_rb is not None and not price_df.empty and not current_ops.empty:
            ops_for_holding = _filter_weight_lt(current_ops, 0.0001)

            if "Action" in ops_for_holding.columns:
                buy_ops = ops_for_holding[ops_for_holding["Action"] == "Buy"].copy()
            else:
                buy_ops = ops_for_holding.copy()

            if not buy_ops.empty:
                buy_ops = buy_ops.copy()
                use_mtm = (
                    "Period_Return" in buy_ops.columns
                    and buy_ops["Period_Return"].notna().any()
                    and "Sell_Price_Close" in buy_ops.columns
                )
                if "Weight" in buy_ops.columns:
                    buy_ops = buy_ops.sort_values("Weight", ascending=False)
                    buy_ops = buy_ops.drop_duplicates(subset=["Symbol"], keep="first")

                if use_mtm:
                    holding_info = buy_ops[
                        buy_ops["Period_Return"].notna()
                        & buy_ops["Buy_Price_Close"].notna()
                        & buy_ops["Sell_Price_Close"].notna()
                    ].copy()
                    if not holding_info.empty:
                        w = pd.to_numeric(holding_info["Weight"], errors="coerce").fillna(0.0)
                        r = pd.to_numeric(holding_info["Period_Return"], errors="coerce").fillna(0.0)
                        total_change = float((w * r).sum())
                        lines = []
                        for _, row in holding_info.iterrows():
                            wt = float(row.get("Weight", 0)) * 100
                            pr = float(row["Period_Return"])
                            bp = float(row["Buy_Price_Close"])
                            sp = float(row["Sell_Price_Close"])
                            sv = pd.to_numeric(row.get("Sell_Value"), errors="coerce")
                            pos_s = f"${float(sv):.2f}" if pd.notna(sv) else "-"
                            lines.append(
                                f"• {row['Symbol']}: 权重 {wt:.1f}% | 区间 {pr*100:+.2f}% | "
                                f"买 ${bp:.2f} → 假设卖 ${sp:.2f} | 头寸 {pos_s}"
                            )
                        if len(lines) > DISCORD_OPS_MAX_LINES:
                            lines = lines[:DISCORD_OPS_MAX_LINES]
                            lines.append(f"  ...另有 {len(holding_info) - DISCORD_OPS_MAX_LINES} 只")
                        holding_text = _trunc_text(
                            "\n".join(lines), DISCORD_FIELD_MAX_CHARS
                        )
                        holding_field = {
                            "name": (
                                f"📊 当前持仓（买入日 {current_rb.date()}，"
                                f"加权区间 {total_change*100:+.2f}%）"
                            ),
                            "value": (
                                f"假设卖出价=As_Of 收盘或现价（与报表 MTM 一致）\n"
                                f"**详情（共 {len(holding_info)} 只）：**\n{holding_text}"
                            ),
                            "inline": False,
                        }
                else:
                    buy_ops["Buy_Price"] = pd.to_numeric(buy_ops["Buy_Price_Close"], errors="coerce")

                    def _get_latest_price(sym: str, _pdf: pd.DataFrame, _asof: pd.Timestamp) -> float:
                        if sym not in _pdf.columns:
                            return float("nan")
                        series = _pdf[sym].dropna()
                        if len(series) == 0:
                            return float("nan")
                        valid = series[series.index <= _asof]
                        if len(valid) > 0:
                            return float(valid.iloc[-1])
                        return float(series.iloc[-1])

                    buy_ops["Current_Price"] = buy_ops["Symbol"].map(
                        lambda s: _get_latest_price(s, price_df, as_of)
                    )
                    buy_ops["Change_Pct"] = (
                        buy_ops["Current_Price"] - buy_ops["Buy_Price"]
                    ) / buy_ops["Buy_Price"]

                    holding_info = buy_ops[
                        buy_ops["Buy_Price"].notna() & buy_ops["Current_Price"].notna()
                    ].copy()

                    if not holding_info.empty:
                        total_change = float(
                            (holding_info["Weight"] * holding_info["Change_Pct"]).sum()
                        )

                        lines = []
                        for _, row in holding_info.iterrows():
                            pct = float(row["Change_Pct"]) * 100
                            bp = float(row["Buy_Price"])
                            cp = float(row["Current_Price"])
                            wt = float(row.get("Weight", 0)) * 100
                            pos_value = wt * 100.0
                            lines.append(
                                f"• {row['Symbol']}: {pct:+.2f}% | 权重 {wt:.1f}% | "
                                f"买入 ${bp:.2f} → 当前 ${cp:.2f} | 约 ${pos_value:.0f}"
                            )
                        if len(lines) > DISCORD_OPS_MAX_LINES:
                            lines = lines[:DISCORD_OPS_MAX_LINES]
                            lines.append(f"  ...另有 {len(holding_info) - DISCORD_OPS_MAX_LINES} 只")

                        holding_text = _trunc_text(
                            "\n".join(lines), DISCORD_FIELD_MAX_CHARS
                        )
                        holding_field = {
                            "name": (
                                f"📊 当前持仓（买入日 {current_rb.date()}，"
                                f"区间 {total_change*100:+.2f}%）"
                            ),
                            "value": (
                                f"整体区间涨跌：{total_change*100:+.2f}%\n"
                                f"**详情（共 {len(holding_info)} 只）：**\n{holding_text}"
                            ),
                            "inline": False,
                        }

        # ── 构造 embed ─────────────────────────────────────────────────
        if is_rebalance:
            title = "🔔 调仓日提醒 - 今日需要操作"
            color = 0x00FF00

            total_ret = metrics.get("total_return", float("nan"))
            ann_ret = metrics.get("annual_return", float("nan"))
            sharpe = metrics.get("sharpe", float("nan"))
            max_dd_pct = metrics.get("max_drawdown_pct", float("nan"))
            wp_dd_pct = metrics.get("worst_period_drawdown_pct", float("nan"))
            calmar = metrics.get("calmar", float("nan"))
            win_rate = metrics.get("win_rate", float("nan"))
            pl_ratio = metrics.get("profit_loss_ratio", float("nan"))

            description = (
                factor_info
                + f"**调仓日期：** {current_rb.date()}\n"
                + f"**策略：** {strategy_param}\n"
                f"**执行时间建议：** 美东时间 15:45-16:00（收盘前15分钟）\n\n"
                f"**策略表现：**\n"
                f"• 总收益率：{_fmt_metric(total_ret, '{:.2%}')}\n"
                f"• 年化收益率：{_fmt_metric(ann_ret, '{:.2%}')}\n"
                f"• 夏普比率：{_fmt_metric(sharpe, '{:.2f}')}\n"
                f"• 最大回撤：{_fmt_metric(max_dd_pct, '{:.2f}%')}（全局）\n"
                f"• 单周期最坏：{_fmt_metric(wp_dd_pct, '{:.2f}%')}（最差持仓周期）\n"
                f"• Calmar 比率：{_fmt_metric(calmar, '{:.2f}')}\n"
                f"• 胜率：{_fmt_metric(win_rate, '{:.2%}')}\n"
                f"• 盈亏比：{_fmt_metric(pl_ratio, '{:.2f}')}\n"
            )

            fields: list = []

            filtered_ops = _filter_weight_lt(current_ops, 0.0001)

            if not filtered_ops.empty:
                sell_ops = filtered_ops[filtered_ops["Action"] == "Sell"]
                buy_ops = filtered_ops[filtered_ops["Action"] == "Buy"].sort_values(
                    by="Weight", ascending=False
                )

                if not sell_ops.empty:
                    lines = []
                    for _, row in sell_ops.iterrows():
                        sym = row["Symbol"]
                        weight = row.get("Weight", 0) * 100
                        sell_price = row.get("Sell_Price_Close", 0)
                        pr = row.get("Period_Return", np.nan)
                        sv = row.get("Sell_Value", np.nan)
                        pr_s = f" 区间 {float(pr)*100:+.2f}%" if pd.notna(pr) else ""
                        sv_s = f" | 卖出头寸 ${float(sv):.2f}" if pd.notna(sv) else ""
                        lines.append(
                            f"• {sym}: 权重 {weight:.1f}% | 卖价 ${float(sell_price):.2f}{pr_s}{sv_s}"
                        )
                    sell_text = _trunc_text(
                        "\n".join(lines), DISCORD_FIELD_MAX_CHARS
                    )
                    fields.append({
                        "name": f"🔴 卖出操作 ({len(sell_ops)} 只)",
                        "value": sell_text,
                        "inline": False,
                    })

                if not buy_ops.empty:
                    lines = []
                    for _, row in buy_ops.iterrows():
                        sym = row["Symbol"]
                        weight = row.get("Weight", 0) * 100
                        buy_price = row.get("Buy_Price_Close", 0)
                        sp = row.get("Sell_Price_Close", np.nan)
                        pr = row.get("Period_Return", np.nan)
                        sv = row.get("Sell_Value", np.nan)
                        mtm = ""
                        if pd.notna(sp) and pd.notna(pr):
                            mtm = f" | 假设卖 ${float(sp):.2f} | 区间 {float(pr)*100:+.2f}%"
                        sv_s = f" | 头寸 ${float(sv):.2f}" if pd.notna(sv) else ""
                        lines.append(
                            f"• {sym}: 权重 {weight:.1f}% | 买 ${float(buy_price):.2f}{mtm}{sv_s}"
                        )
                    buy_text = _trunc_text(
                        "\n".join(lines[:DISCORD_OPS_MAX_LINES]),
                        DISCORD_FIELD_MAX_CHARS,
                    )
                    fields.append({
                        "name": f"🟢 买入操作 ({len(buy_ops)} 只)",
                        "value": buy_text,
                        "inline": False,
                    })
            else:
                fields.append({
                    "name": "⚠️ 操作明细",
                    "value": "无操作数据（可能是数据不足或最后一个调仓日）",
                    "inline": False,
                })

            if next_rb:
                fields.append({
                    "name": "📅 下一调仓日",
                    "value": str(next_rb.date()),
                    "inline": False,
                })

            if holding_field:
                fields.append(holding_field)

        else:
            title = "ℹ️ 非调仓日 - 无需操作"
            color = 0x808080
            description = (
                factor_info
                + f"**当前日期：** {as_of.date()}\n"
                + f"**最近调仓日：** {current_rb.date() if current_rb else '无'}\n"
                + f"**下一调仓日：** {next_rb.date() if next_rb else '未知'}\n\n"
            )
            fields = []
            if holding_field:
                fields.append(holding_field)
            description += "\n今日无需操作，请等待下一调仓日。"

        embed = {
            "title": title,
            "description": description,
            "color": color,
            "fields": fields,
            "footer": {
                "text": (
                    f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                    f"因子：{', '.join(selected_factor_names)}"
                ),
            },
        }

        payload = {"embeds": [embed]}
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        print(f"✅ Discord 通知已发送（状态码：{response.status_code}）")

    except Exception as e:
        print(f"❌ Discord 通知发送失败：{e}")
