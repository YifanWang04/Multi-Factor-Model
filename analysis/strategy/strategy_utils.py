"""
策略模块共享工具 (analysis/strategy/strategy_utils.py)
=====================================================
集中管理 strategy 子模块中重复的函数，避免多处维护同一份代码。

包含：
  - load_price_data:  加载日频价格数据（宽表）
  - load_composite_factor: 加载复合因子 Excel（指定 sheet）
  - _get_price_on_date: 获取指定日期各标的收盘价
  - build_factor_suffix: 生成因子后缀字符串
  - parse_strategy_param: 解析策略参数字符串
  - _filter_weight_lt: 过滤低权重操作行
  - _format_metric / _truncate_text: Discord 消息格式化工具
  - MarkToMarket: 未到期持仓市值重估封装类
"""

from __future__ import annotations

import os
import re
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 价格数据加载
# ---------------------------------------------------------------------------

def load_price_data(price_file: str, price_column: str = "Adj Close") -> pd.DataFrame:
    """
    加载日频价格数据，返回宽表 DataFrame(index=日期, columns=股票代码)。
    使用 pd.concat 一次性构建，避免循环 insert 导致的 fragmentation 告警。
    """
    if not os.path.isfile(price_file):
        raise FileNotFoundError(f"价格文件不存在: {price_file}")

    price_data = pd.read_excel(price_file, sheet_name=None)
    columns_dict = {}
    for ticker, df in price_data.items():
        if "Date" not in df.columns or price_column not in df.columns:
            continue
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
        columns_dict[ticker] = df[price_column]

    if not columns_dict:
        return pd.DataFrame()

    price_df = pd.concat(columns_dict, axis=1)
    price_df = price_df.apply(pd.to_numeric, errors="coerce")
    price_df.sort_index(inplace=True)
    return price_df


def _get_price_on_date(
    price_df: pd.DataFrame,
    date: pd.Timestamp,
    stocks: list,
) -> pd.Series:
    """
    获取指定日期各标的收盘价，缺失则取不超过该日期的最近可交易日。
    """
    if date not in price_df.index:
        idx = price_df.index[price_df.index <= date]
        if len(idx) == 0:
            return pd.Series(dtype=float)
        date = idx[-1]
    row = price_df.loc[date]
    return row.reindex(stocks).dropna()


# ---------------------------------------------------------------------------
# 复合因子加载
# ---------------------------------------------------------------------------

def load_composite_factor(file_path: str, sheet_name: str) -> pd.DataFrame:
    """
    从 composite_factors.xlsx 加载指定 sheet 的复合因子数据。
    index = 调仓日（DatetimeIndex），columns = 股票代码。
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"复合因子文件不存在: {file_path}")

    xl = pd.ExcelFile(file_path)
    available = xl.sheet_names
    if sheet_name not in available:
        raise ValueError(
            f"Sheet '{sheet_name}' 不存在于 {os.path.basename(file_path)}。\n"
            f"可用 sheet: {available}"
        )

    df = pd.read_excel(file_path, sheet_name=sheet_name, index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.apply(pd.to_numeric, errors="coerce")
    df.sort_index(inplace=True)
    return df


def load_composite_factor_with_fallback(
    primary_path: str,
    sheet: str,
    std_file: str,
) -> pd.DataFrame:
    """
    加载复合因子，主路径失败时尝试标准路径回退。
    主路径和标准路径均失败时抛出 FileNotFoundError（含已尝试路径信息）。
    """
    tried: list[str] = []
    for path in (primary_path, std_file):
        if not path:
            tried.append(f"<空路径>")
            continue
        tried.append(path)
        if os.path.isfile(path):
            try:
                df = load_composite_factor(path, sheet)
                if not df.empty:
                    return df
            except Exception:
                pass

    raise FileNotFoundError(
        f"无法加载复合因子 sheet '{sheet}'（已尝试: {', '.join(tried)}）"
    )


# ---------------------------------------------------------------------------
# 分组工具（与 strategy_backtest._build_groups 保持一致）
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


# ---------------------------------------------------------------------------
# 因子后缀
# ---------------------------------------------------------------------------

def build_factor_suffix(factor_indices: list[int] | None = None, default_indices: list[int] | None = None) -> str:
    """
    基于因子编号列表生成简短后缀，如 f95-24-64-65-32。
    未提供时优先使用 default_indices；两者均无则返回空字符串。
    """
    if factor_indices is None:
        factor_indices = default_indices
    if factor_indices is None:
        return ""
    nums = [str(int(i)) for i in factor_indices]
    return "f" + "-".join(nums)


def composite_factors_path(base_dir: str, factor_indices: list[int]) -> str:
    """
    返回 composite_factor_reports 目录下带因子后缀的文件路径。
    base_dir：完整路径（可含 composite_factor_reports/ 子目录）
    当 base_dir 已含 composite_factor_reports 子目录时，直接在其下生成文件；
    否则自动追加该子目录。
    """
    suffix = build_factor_suffix(factor_indices)
    name = f"composite_factors_{suffix}.xlsx"
    # 若 base_dir 已为完整目录（已含 composite_factor_reports 子目录），直接使用
    if os.path.basename(base_dir) == "composite_factor_reports":
        return os.path.join(base_dir, name)
    return os.path.join(base_dir, "composite_factor_reports", name)


# ---------------------------------------------------------------------------
# 策略参数字符串解析
# ---------------------------------------------------------------------------

def parse_strategy_param(param: str) -> tuple:
    """
    解析策略参数字符串，格式：{weight_method}_{N}G_Top{R}_P{D}d
    例：max_return_10G_Top1_P20d -> (weight_method, group_num, target_rank, rebalance_days)
    """
    m = re.match(r"^(.+)_(\d+)G_Top(\d+)_P(\d+)d$", param.strip())
    if not m:
        raise ValueError(
            f"策略参数格式错误: '{param}'，应为 {{weight_method}}_{{N}}G_Top{{R}}_P{{D}}d，"
            "例：max_return_10G_Top1_P10d"
        )
    weight_method = m.group(1)
    group_num = int(m.group(2))
    target_rank = int(m.group(3))
    rebalance_days = int(m.group(4))
    return weight_method, group_num, target_rank, rebalance_days


def strategy_param_from_params(params: dict) -> str:
    """从 params 字典还原策略参数字符串。"""
    w = params.get("weight_method", "")
    g = params.get("group_num", "")
    r = params.get("target_rank", "")
    p = params.get("rebalance_period", "")
    if w != "" and g != "" and r != "" and p != "":
        return f"{w}_{g}G_Top{r}_P{p}d"
    return ""


# ---------------------------------------------------------------------------
# 操作数据过滤
# ---------------------------------------------------------------------------

def filter_weight_lt(
    ops: pd.DataFrame,
    threshold: float = 0.0001,
    logger=None,
) -> pd.DataFrame:
    """
    过滤 Weight 列 < threshold 的行。可选传入 logger 打印移除数量。
    """
    if "Weight" not in ops.columns:
        return ops
    before = len(ops)
    ops = ops[ops["Weight"] >= threshold].copy()
    removed = before - len(ops)
    if removed > 0 and logger is not None:
        logger(f"  过滤 Weight < {threshold}，移除 {removed} 行")
    return ops


# ---------------------------------------------------------------------------
# Discord 格式化工具
# ---------------------------------------------------------------------------

def format_metric(value: float, fmt: str) -> str:
    """安全格式化指标值，NaN 时返回 '-'。"""
    if isinstance(value, float) and np.isnan(value):
        return "-"
    return fmt.format(value)


def truncate_text(text: str, max_chars: int) -> str:
    """截断文本并加省略号。"""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def safe_tag(s: str) -> str:
    """将字符串转成适合文件名的 tag（尽量保持可读性）。"""
    s = str(s)
    s = s.strip().replace(" ", "")
    return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in s)


# ---------------------------------------------------------------------------
# MarkToMarket：未到期持仓市值重估封装
# ---------------------------------------------------------------------------

class MarkToMarket:
    """
    对未到期持仓进行市值重估（MTM）。

    使用方式：
        mtm = MarkToMarket(ops_df, price_df, as_of_date)
        mtm.apply(live_prices=...)  # live_prices: {sym: {"current": float, ...}}
        marked_ops = mtm.operations_df
        patch_period_summary(mtm, result)  # 同步更新 period_summary_df
    """

    # 卖出价来源标注
    SOURCE_OPEN_PERIOD = "假设市价(未到期)"   # 下一调仓日尚未到期
    SOURCE_FILL_MISSING = "假设市价(补全)"     # 卖出价历史缺失
    SOURCE_MATURED = "到期收盘"               # 已到期，有历史卖出价

    def __init__(
        self,
        ops_df: pd.DataFrame,
        price_df: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ):
        self._ops_df = ops_df.copy()
        self._price_df = price_df
        self._as_of = pd.Timestamp(as_of_date).normalize()
        self._live_prices: dict = {}
        self._source_col = "Sell_Price_Source"

        self._ops_df["Next_Rebalance_Date"] = pd.to_datetime(
            self._ops_df.get("Next_Rebalance_Date", pd.NaT), errors="coerce"
        )
        self._ops_df["Rebalance_Date"] = pd.to_datetime(
            self._ops_df.get("Rebalance_Date", pd.NaT), errors="coerce"
        )
        if self._source_col not in self._ops_df.columns:
            self._ops_df[self._source_col] = ""

    @property
    def operations_df(self) -> pd.DataFrame:
        return self._ops_df

    def _mark_price_for_symbol(self, symbol: str) -> float:
        """取 as_of 及之前最近可用的 Adj Close；若无则用 live_prices。"""
        if symbol not in self._price_df.columns:
            lp = self._live_prices.get(symbol, {})
            cur = lp.get("current")
            return float(cur) if cur is not None else float("nan")

        series = self._price_df[symbol].dropna()
        if len(series) == 0:
            lp = self._live_prices.get(symbol, {})
            cur = lp.get("current")
            return float(cur) if cur is not None else float("nan")

        valid = series[series.index <= self._as_of]
        if len(valid) > 0:
            return float(valid.iloc[-1])

        lp = self._live_prices.get(symbol, {})
        cur = lp.get("current")
        return float(cur) if cur is not None else float("nan")

    def apply(self, live_prices: dict | None = None) -> "MarkToMarket":
        """
        执行市值重估。传入 live_prices: {symbol: {"current": float}}，
        用于 price_df 在 as_of 无有效价时的实时回退。
        返回 self（支持链式调用）。

        向量化实现（替代逐行 iterrows）：
          Phase 1: 批量获取所有标的的 mark 价格（本地向量查 + yfinance 补缺）
          Phase 2: 向量条件过滤 + 批量赋值
        """
        if self._ops_df.empty:
            return self

        if live_prices:
            self._live_prices = live_prices

        # ── Phase 1: 批量获取所有标的的 mark 价格 ──────────────────────
        syms = self._ops_df["Symbol"].unique().tolist()
        mark_prices = {
            sym: self._mark_price_for_symbol(sym)
            for sym in syms
        }

        # ── Phase 2: 向量条件过滤 + 批量赋值 ───────────────────────────
        ops = self._ops_df
        next_rb_col = ops["Next_Rebalance_Date"]
        nr_ts = pd.to_datetime(next_rb_col, errors="coerce")
        sell_was = pd.to_numeric(ops["Sell_Price_Close"], errors="coerce")
        need_mtm = nr_ts.notna() & (
            (nr_ts > self._as_of) | sell_was.isna()
        )

        if not need_mtm.any():
            empty_src = ops[self._source_col].replace("", np.nan).isna()
            if empty_src.any():
                ops.loc[empty_src, self._source_col] = self.SOURCE_MATURED
            return self

        target = ops[need_mtm].copy()
        sym_col = target["Symbol"]
        marks = sym_col.map(mark_prices)
        valid_mark = marks.notna() & (marks > 0)

        bp = pd.to_numeric(target["Buy_Price_Close"], errors="coerce")
        valid_bp = bp.notna() & (bp > 0)

        wt = pd.to_numeric(target["Weight"], errors="coerce")
        buy_value_raw = pd.to_numeric(target["Buy_Value"], errors="coerce")
        buy_value = buy_value_raw.copy()
        use_weight = buy_value_raw.isna() & wt.notna()
        buy_value.loc[use_weight] = wt.loc[use_weight]

        valid_buy = buy_value.notna() & (buy_value > 0)
        final_mask = valid_mark & valid_bp & valid_buy

        idx_final = target.index[final_mask]

        if len(idx_final) > 0:
            bp_v = bp.loc[idx_final]
            mk_v = marks.loc[idx_final]
            bv_v = buy_value.loc[idx_final]
            nr_v = nr_ts.loc[idx_final]

            ops.loc[idx_final, "Sell_Price_Close"] = mk_v
            ops.loc[idx_final, "Period_Return"] = mk_v / bp_v - 1.0
            ops.loc[idx_final, "Sell_Value"] = bv_v * (1.0 + (mk_v / bp_v - 1.0))
            ops.loc[idx_final, "Shares"] = bv_v / bp_v
            ops.loc[idx_final, self._source_col] = np.where(
                nr_v > self._as_of,
                self.SOURCE_OPEN_PERIOD,
                self.SOURCE_FILL_MISSING,
            )

        return self

    def was_applied(self) -> bool:
        """检查是否实际执行了 MTM（而非仅标注到期收盘）。"""
        if self._ops_df.empty or self._source_col not in self._ops_df.columns:
            return False
        return self._ops_df[self._source_col].isin(
            [self.SOURCE_OPEN_PERIOD, self.SOURCE_FILL_MISSING]
        ).any()


def patch_period_summary_from_mtm(
    result: dict,
    mtm_ops: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> None:
    """
    用 MTM 后的 ops 数据同步更新 result["period_summary_df"]。
    对尚未到期的持仓期，用 MTM 后的个股收益加权更新 Period_Return 与 Holding_Days。

    向量化实现（替代逐行 iterrows）。
    """
    ps = result.get("period_summary_df")
    if ps is None or ps.empty:
        return

    ps = ps.copy()
    mtm_ops = mtm_ops.copy()
    as_of = pd.Timestamp(as_of_date).normalize()

    ps["Rebalance_Date"] = pd.to_datetime(ps["Rebalance_Date"], errors="coerce")
    ps["Next_Rebalance_Date"] = pd.to_datetime(ps["Next_Rebalance_Date"], errors="coerce")
    mtm_ops["Rebalance_Date"] = pd.to_datetime(mtm_ops["Rebalance_Date"], errors="coerce")
    mtm_ops["Next_Rebalance_Date"] = pd.to_datetime(mtm_ops["Next_Rebalance_Date"], errors="coerce")

    # 筛选未到期的持仓期
    nr_col = ps["Next_Rebalance_Date"]
    future = nr_col.notna() & (nr_col > as_of)
    if not future.any():
        result["period_summary_df"] = ps
        return

    future_idx = ps.index[future]
    rb_col = ps["Rebalance_Date"]

    w = pd.to_numeric(mtm_ops["Weight"], errors="coerce").fillna(0.0)
    r = pd.to_numeric(mtm_ops["Period_Return"], errors="coerce")

    new_rets = []
    new_days = []
    for i in future_idx:
        rb = pd.Timestamp(rb_col[i])
        nr = pd.Timestamp(nr_col[i])
        sub = mtm_ops[
            (mtm_ops["Rebalance_Date"] == rb)
            & (mtm_ops["Next_Rebalance_Date"] == nr)
        ]
        if sub.empty:
            new_rets.append(np.nan)
            new_days.append(ps.at[i, "Holding_Days"])
            continue
        sub_w = w.loc[sub.index]
        sub_r = r.loc[sub.index]
        ws = sub_w.sum()
        if ws > 0 and sub_r.notna().any():
            port_ret = float((sub_w * sub_r.fillna(0)).sum() / ws)
        else:
            port_ret = np.nan
        new_rets.append(port_ret)
        new_days.append(max(0, (as_of - rb).days))

    ps.loc[future_idx, "Period_Return"] = new_rets
    ps.loc[future_idx, "Holding_Days"] = new_days
    result["period_summary_df"] = ps
