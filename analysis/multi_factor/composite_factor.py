"""
因子复合方法 (composite_factor.py)
===================================
输入：factor_dict {name: DataFrame(date×stock)}, ret_periods DataFrame(date×stock)
输出：composite_dict {composite_name: DataFrame(date×stock)}

复合方法：
  一元回归加权 (beta)：方法1/2/3
  IC加权：方法1/2/3
  Rank_IC加权：方法1/2/3
  排序加权：排序相加、排序相乘
  多元回归加权 (OLS)：方法1/2/3
  PCA：主成分1/2/3
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def _zscore(df):
    """截面z-score标准化，逐行。"""
    mu = df.mean(axis=1)
    sd = df.std(axis=1)
    return df.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)


def _align_dates(factor_dict):
    """取所有因子的公共日期索引。"""
    dates = None
    for df in factor_dict.values():
        dates = df.index if dates is None else dates.intersection(df.index)
    return dates


def _factor_matrix(factor_dict, dates):
    """返回 {name: df.reindex(dates)} 的对齐字典（缺失日期为 NaN）。"""
    return {n: df.reindex(dates) for n, df in factor_dict.items()}


# ---------------------------------------------------------------------------
# 计算每期截面 beta（一元OLS斜率）和 IC
# ---------------------------------------------------------------------------

def _compute_betas_ics(factor_dict, ret_periods):
    """
    对每个因子、每个调仓期，计算截面 OLS beta 和 Pearson IC / Spearman Rank_IC。
    返回 {factor_name: {'beta': Series, 'ic': Series, 'rank_ic': Series}}，index=调仓日。
    """
    dates = ret_periods.index
    result = {}
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
            # beta via OLS
            beta = np.cov(fv, rv)[0, 1] / np.var(fv)
            ic = np.corrcoef(fv, rv)[0, 1]
            ric, _ = spearmanr(fv, rv)
            betas.append(beta)
            ics.append(ic)
            rank_ics.append(ric)
            valid_dates.append(d)
        result[name] = {
            "beta": pd.Series(betas, index=valid_dates),
            "ic": pd.Series(ics, index=valid_dates),
            "rank_ic": pd.Series(rank_ics, index=valid_dates),
        }
    return result


def _weighted_composite(factor_dict, weights_series_dict, dates):
    """
    按权重合成复合因子。
    weights_series_dict: {factor_name: scalar weight}（每期固定）或 {factor_name: Series(date)}
    返回 DataFrame(date×stock)。
    注意：本函数当前未被调用，实际路径均走 _composite_from_weight_df。
    """
    names = list(factor_dict.keys())
    all_dates = dates
    result = pd.DataFrame(index=all_dates, columns=next(iter(factor_dict.values())).columns, dtype=float)
    result[:] = 0.0

    for d in all_dates:
        row = pd.Series(0.0, index=result.columns)
        total_w = 0.0
        for name in names:
            w_val = weights_series_dict[name]
            w = w_val.get(d, np.nan) if isinstance(w_val, pd.Series) else w_val
            if np.isnan(w):
                continue
            frow = factor_dict[name].loc[d] if d in factor_dict[name].index else pd.Series(dtype=float)
            row = row.add(frow * w, fill_value=0)
            total_w += abs(w)
        result.loc[d] = row / total_w if total_w != 0 else row
    return result.apply(pd.to_numeric, errors="coerce")


# ---------------------------------------------------------------------------
# 公共：按权重序列合成（支持每期不同权重）
# ---------------------------------------------------------------------------

def _composite_from_weight_df(factor_dict, weight_df, dates):
    """
    weight_df: DataFrame(date × factor_name)，每行为该期各因子权重。
    返回 DataFrame(date × stock)。
    """
    stocks = next(iter(factor_dict.values())).columns
    result = pd.DataFrame(np.nan, index=dates, columns=stocks)
    names = list(factor_dict.keys())
    for d in dates:
        if d not in weight_df.index:
            continue
        ws = weight_df.loc[d]
        row = pd.Series(0.0, index=stocks)
        for name in names:
            w = ws.get(name, np.nan)
            if np.isnan(w):
                continue
            frow = factor_dict[name].loc[d] if d in factor_dict[name].index else pd.Series(dtype=float)
            row = row.add(frow.reindex(stocks) * w, fill_value=0)
        result.loc[d] = row
    return result.apply(pd.to_numeric, errors="coerce")


# ---------------------------------------------------------------------------
# 一元回归加权 / IC加权 / Rank_IC加权
# ---------------------------------------------------------------------------

def _univariate_weighted(factor_dict, stats_dict, key, dates, method, window=None):
    """
    通用：用 stats_dict[name][key] 序列按 method(1/2/3) 计算权重，合成复合因子。

    时序说明：
      - IC/beta 序列的索引是调仓日 d_k
      - IC(d_k) 使用 d_k 的因子值和 (d_k, d_{k+1}] 的收益计算
      - 在 d_{k+1} 调仓时，IC(d_k) 已知（因为 (d_k, d_{k+1}] 的收益已实现）
      - 因此在 d_{k+1} 调仓时，可用的最新 IC 是 IC(d_k)，即 s[s.index < d_{k+1}] 的最后一个

    method=1: 全期均值（固定权重）⚠️ 存在前瞻偏误，使用了全期（含未来期）的 IC/beta
              均值作为权重，属于"全知基准（oracle baseline）"，仅供研究对比，
              不可用于真实策略回测。
    method=2: 截至当期累计均值（无前瞻，用 index < d 的历史数据，包含最近一期）
    method=3: 滚动 window 期均值（无前瞻，用 index < d 的最近 N 期数据）
    """
    names = list(factor_dict.keys())
    # 收集各因子的 key 序列
    series_map = {n: stats_dict[n][key] for n in names}

    if method == 1:
        # 全期均值（含前瞻，仅供研究对比）
        rows = []
        for d in dates:
            row = {}
            for n in names:
                s = series_map[n]
                # 使用全部历史数据（含当前调仓日之前的所有 IC）
                past = s[s.index < d]
                row[n] = past.mean() if len(past) > 0 else np.nan
            rows.append(row)
        weight_df = pd.DataFrame(rows, index=dates)
    elif method == 2:
        # 截至当期累计均值（无前瞻）
        rows = []
        for d in dates:
            row = {}
            for n in names:
                s = series_map[n]
                # 使用 d 之前的所有 IC（不排除最近一期，因为它已经实现）
                past = s[s.index < d]
                row[n] = past.mean() if len(past) > 0 else np.nan
            rows.append(row)
        weight_df = pd.DataFrame(rows, index=dates)
    else:  # method == 3
        assert window is not None
        rows = []
        for d in dates:
            row = {}
            for n in names:
                s = series_map[n]
                # 使用 d 之前的最近 window 期 IC
                past = s[s.index < d].iloc[-window:]
                row[n] = past.mean() if len(past) > 0 else np.nan
            rows.append(row)
        weight_df = pd.DataFrame(rows, index=dates)

    return _composite_from_weight_df(factor_dict, weight_df, dates)


def beta_weighted(factor_dict, ret_periods, N_windows):
    """一元回归加权，返回 {name: composite_df}。"""
    dates = ret_periods.index
    aligned = _factor_matrix(factor_dict, dates)
    stats = _compute_betas_ics(aligned, ret_periods)
    out = {}
    out["beta_m1"] = _zscore(_univariate_weighted(aligned, stats, "beta", dates, 1))
    out["beta_m2"] = _zscore(_univariate_weighted(aligned, stats, "beta", dates, 2))
    for N in N_windows:
        out[f"beta_m3_N{N}"] = _zscore(_univariate_weighted(aligned, stats, "beta", dates, 3, N))
    return out


def ic_weighted(factor_dict, ret_periods, N_windows):
    """IC加权，返回 {name: composite_df}。"""
    dates = ret_periods.index
    aligned = _factor_matrix(factor_dict, dates)
    stats = _compute_betas_ics(aligned, ret_periods)
    out = {}
    out["ic_m1"] = _zscore(_univariate_weighted(aligned, stats, "ic", dates, 1))
    out["ic_m2"] = _zscore(_univariate_weighted(aligned, stats, "ic", dates, 2))
    for N in N_windows:
        out[f"ic_m3_N{N}"] = _zscore(_univariate_weighted(aligned, stats, "ic", dates, 3, N))
    return out


def rank_ic_weighted(factor_dict, ret_periods, N_windows):
    """Rank_IC加权，返回 {name: composite_df}。"""
    dates = ret_periods.index
    aligned = _factor_matrix(factor_dict, dates)
    stats = _compute_betas_ics(aligned, ret_periods)
    out = {}
    out["rank_ic_m1"] = _zscore(_univariate_weighted(aligned, stats, "rank_ic", dates, 1))
    out["rank_ic_m2"] = _zscore(_univariate_weighted(aligned, stats, "rank_ic", dates, 2))
    for N in N_windows:
        out[f"rank_ic_m3_N{N}"] = _zscore(_univariate_weighted(aligned, stats, "rank_ic", dates, 3, N))
    return out


# ---------------------------------------------------------------------------
# 排序加权
# ---------------------------------------------------------------------------

def rank_weighted(factor_dict, ret_periods):
    """排序相加 & 排序相乘，返回 {name: composite_df}。"""
    dates = ret_periods.index
    aligned = _factor_matrix(factor_dict, dates)
    names = list(aligned.keys())
    stocks = next(iter(aligned.values())).columns

    rank_add = pd.DataFrame(np.nan, index=dates, columns=stocks)
    rank_mul = pd.DataFrame(np.nan, index=dates, columns=stocks)

    for d in dates:
        rank_frames = []
        for n in names:
            if d not in aligned[n].index:
                continue
            row = aligned[n].loc[d]
            ranked = row.rank(method="average", na_option="keep")
            rank_frames.append(ranked)
        if not rank_frames:
            continue
        stacked = pd.concat(rank_frames, axis=1)
        rank_add.loc[d] = stacked.sum(axis=1, min_count=1)
        # 归一化后相乘
        norm_frames = [f / f.max() for f in rank_frames]
        mul = norm_frames[0].copy()
        for f in norm_frames[1:]:
            mul = mul * f
        rank_mul.loc[d] = mul

    return {
        "rank_add": _zscore(rank_add.apply(pd.to_numeric, errors="coerce")),
        "rank_mul": _zscore(rank_mul.apply(pd.to_numeric, errors="coerce")),
    }


# ---------------------------------------------------------------------------
# 多元回归加权 (OLS)
# ---------------------------------------------------------------------------

def _ols_betas_per_period(factor_dict, ret_periods):
    """
    每期截面多元OLS：r = alpha + beta1*F1 + ... + epsilon
    返回 DataFrame(date × factor_name)，每行为该期的 beta 向量。
    """
    names = list(factor_dict.keys())
    dates = ret_periods.index
    rows = []
    valid_dates = []
    for d in dates:
        cols_data = {}
        for n in names:
            if d in factor_dict[n].index:
                cols_data[n] = factor_dict[n].loc[d]
        if len(cols_data) < len(names):
            continue
        X = pd.DataFrame(cols_data).dropna()
        r = ret_periods.loc[d].reindex(X.index).dropna()
        X = X.loc[r.index]
        if len(X) < len(names) + 2:
            continue
        try:
            reg = LinearRegression(fit_intercept=True).fit(X.values, r.values)
            rows.append(dict(zip(names, reg.coef_)))
            valid_dates.append(d)
        except Exception:
            continue
    return pd.DataFrame(rows, index=valid_dates)


def multivariate_weighted(factor_dict, ret_periods, M_windows):
    """多元回归加权，返回 {name: composite_df}。"""
    dates = ret_periods.index
    aligned = _factor_matrix(factor_dict, dates)
    beta_df = _ols_betas_per_period(aligned, ret_periods)
    names = list(aligned.keys())
    out = {}

    def _build_weight_df(method, window=None):
        rows = []
        for d in dates:
            # 使用 d 之前的所有 beta（不排除最近一期，因为它已经实现）
            past = beta_df[beta_df.index < d]
            if method == 3:
                past = past.iloc[-window:]
            rows.append(past.mean().to_dict() if len(past) > 0 else {n: np.nan for n in names})
        return pd.DataFrame(rows, index=dates)

    out["ols_m1"] = _zscore(_composite_from_weight_df(aligned, _build_weight_df(1), dates))
    out["ols_m2"] = _zscore(_composite_from_weight_df(aligned, _build_weight_df(2), dates))
    for M in M_windows:
        out[f"ols_m3_M{M}"] = _zscore(_composite_from_weight_df(aligned, _build_weight_df(3, M), dates))
    return out


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def pca_composite(factor_dict, ret_periods, n_components=3):
    """
    PCA：对每期截面因子矩阵做PCA，取前n_components个主成分作为复合因子。
    返回 {f'pca_pc{i+1}': DataFrame(date×stock)}。
    """
    dates = ret_periods.index
    aligned = _factor_matrix(factor_dict, dates)
    names = list(aligned.keys())
    stocks = next(iter(aligned.values())).columns
    n_comp = min(n_components, len(names))

    # 收集每期截面数据：shape (n_stocks, n_factors)
    pc_results = {i: pd.DataFrame(np.nan, index=dates, columns=stocks) for i in range(n_comp)}

    for d in dates:
        rows = {}
        for n in names:
            if d in aligned[n].index:
                rows[n] = aligned[n].loc[d]
        if len(rows) < 2:
            continue
        X = pd.DataFrame(rows).dropna()  # stocks × factors
        if len(X) < n_comp + 1:
            continue
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X.values)
        pca = PCA(n_components=n_comp)
        try:
            components = pca.fit_transform(Xs)  # stocks × n_comp
        except Exception:
            continue
        for i in range(n_comp):
            pc_results[i].loc[d, X.index] = components[:, i]

    return {f"pca_pc{i+1}": _zscore(pc_results[i].apply(pd.to_numeric, errors="coerce"))
            for i in range(n_comp)}


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def compute_all_composites(factor_dict, ret_periods, N_windows, M_windows):
    """
    计算所有复合因子，返回 {composite_name: DataFrame(date×stock)}。
    factor_dict: {name: DataFrame(date×stock)} — 已按调仓期对齐的因子截面
    ret_periods: DataFrame(date×stock) — 调仓期累计收益
    """
    out = {}
    out.update(beta_weighted(factor_dict, ret_periods, N_windows))
    out.update(ic_weighted(factor_dict, ret_periods, N_windows))
    out.update(rank_ic_weighted(factor_dict, ret_periods, N_windows))
    out.update(rank_weighted(factor_dict, ret_periods))
    out.update(multivariate_weighted(factor_dict, ret_periods, M_windows))
    out.update(pca_composite(factor_dict, ret_periods))
    return out
