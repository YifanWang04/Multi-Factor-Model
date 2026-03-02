"""
组合优化器 (portfolio_optimizer.py)
=====================================
提供五种资产配置方式，统一通过 compute_weights() 调用。

方法说明：
  equal          等权配置：1/N
  factor_score   因子值打分：组内因子值归一化后作为权重
  min_variance   最小方差组合：min w'Σw，s.t. Σw=1, 0≤w≤MAX_WEIGHT
  mvo            马科维兹最优（最大化夏普比率）：max (w'μ-rf)/√(w'Σw)
  max_return     最大化预期收益：max w'μ，s.t. Σw=1, 0≤w≤MAX_WEIGHT

所有优化方法在数据不足或求解失败时自动降级为等权。
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _fallback_equal(n: int) -> np.ndarray:
    return np.ones(n) / n


def _clean_returns(ret_matrix: np.ndarray, min_rows: int = 10) -> np.ndarray | None:
    """
    清洗历史收益率矩阵：去除含 NaN 的行，限幅 [-0.5, 0.5]，并在行数不足时返回 None。
    """
    df = pd.DataFrame(ret_matrix).dropna()
    if len(df) < min_rows:
        return None
    arr = df.values.astype(float)
    arr = np.clip(arr, -0.5, 0.5)
    return arr


def _regularized_cov(arr: np.ndarray) -> np.ndarray:
    """带 Ledoit-Wolf 式对角正则化的协方差矩阵，防止奇异。"""
    cov = np.cov(arr.T)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    # 对角正则化：ε = 1e-6 * trace / N
    eps = max(1e-8, 1e-6 * np.trace(cov) / cov.shape[0])
    cov += np.eye(cov.shape[0]) * eps
    return cov


def _post_process(w: np.ndarray) -> np.ndarray:
    """截断负权重并重归一。"""
    w = np.maximum(w, 0.0)
    total = w.sum()
    return w / total if total > 1e-12 else _fallback_equal(len(w))


# ---------------------------------------------------------------------------
# 各配置方法
# ---------------------------------------------------------------------------

def equal_weight(n: int) -> np.ndarray:
    """等权配置。"""
    return np.ones(n) / n


def factor_score_weight(factor_values: np.ndarray) -> np.ndarray:
    """
    因子值打分加权：先平移到正数域，再归一。
    若所有因子值为 NaN 或全相等，降级为等权。
    """
    fv = np.array(factor_values, dtype=float)
    if np.all(np.isnan(fv)):
        return _fallback_equal(len(fv))
    fv = np.where(np.isnan(fv), 0.0, fv)
    fv = fv - fv.min() + 1e-8
    total = fv.sum()
    return fv / total if total > 1e-12 else _fallback_equal(len(fv))


def min_variance_weight(ret_matrix: np.ndarray, max_weight: float = 0.4) -> np.ndarray:
    """
    最小方差组合：min w'Σw，约束 Σw=1，0 ≤ w ≤ max_weight。
    数据不足或求解失败时返回等权。
    """
    arr = _clean_returns(ret_matrix)
    if arr is None:
        return _fallback_equal(ret_matrix.shape[1] if ret_matrix.ndim > 1 else 1)

    T, N = arr.shape
    if N == 1:
        return np.array([1.0])
    if T < N + 2:
        return _fallback_equal(N)

    cov = _regularized_cov(arr)

    def obj(w):
        return float(w @ cov @ w)

    def grad(w):
        return 2.0 * cov @ w

    w0 = _fallback_equal(N)
    bounds = [(0.0, max_weight)] * N
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

    try:
        res = minimize(
            obj, w0, jac=grad, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-9},
        )
        if res.success:
            return _post_process(res.x)
    except Exception:
        pass

    return _fallback_equal(N)


def mvo_weight(
    ret_matrix: np.ndarray,
    rf_daily: float = 0.02 / 252,
    max_weight: float = 0.4,
) -> np.ndarray:
    """
    马科维兹最优权重（最大化夏普比率）：max (w'μ - rf) / √(w'Σw)。
    约束 Σw=1，0 ≤ w ≤ max_weight。
    数据不足或求解失败时返回等权。
    """
    arr = _clean_returns(ret_matrix)
    if arr is None:
        return _fallback_equal(ret_matrix.shape[1] if ret_matrix.ndim > 1 else 1)

    T, N = arr.shape
    if N == 1:
        return np.array([1.0])
    if T < N + 2:
        return _fallback_equal(N)

    mu = arr.mean(axis=0)
    cov = _regularized_cov(arr)

    def neg_sharpe(w):
        port_ret = float(w @ mu) - rf_daily
        port_var = float(w @ cov @ w)
        port_vol = max(port_var ** 0.5, 1e-12)
        return -port_ret / port_vol

    w0 = _fallback_equal(N)
    bounds = [(0.0, max_weight)] * N
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

    try:
        res = minimize(
            neg_sharpe, w0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-9},
        )
        if res.success:
            return _post_process(res.x)
    except Exception:
        pass

    return _fallback_equal(N)


def max_return_weight(ret_matrix: np.ndarray, max_weight: float = 0.4) -> np.ndarray:
    """
    最大化预期收益：max w'μ，约束 Σw=1，0 ≤ w ≤ max_weight。
    由于目标为线性，最优解倾向集中；max_weight 约束保证适度分散。
    数据不足或求解失败时返回等权。
    """
    arr = _clean_returns(ret_matrix, min_rows=5)
    if arr is None:
        return _fallback_equal(ret_matrix.shape[1] if ret_matrix.ndim > 1 else 1)

    T, N = arr.shape
    if N == 1:
        return np.array([1.0])

    mu = arr.mean(axis=0)

    def neg_ret(w):
        return -float(w @ mu)

    def neg_ret_grad(w):
        return -mu

    w0 = _fallback_equal(N)
    bounds = [(0.0, max_weight)] * N
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

    try:
        res = minimize(
            neg_ret, w0, jac=neg_ret_grad, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-9},
        )
        if res.success:
            return _post_process(res.x)
    except Exception:
        pass

    return _fallback_equal(N)


# ---------------------------------------------------------------------------
# 统一调度入口
# ---------------------------------------------------------------------------

def compute_weights(
    method: str,
    stocks: list,
    factor_values: pd.Series,
    hist_returns: pd.DataFrame,
    lookback: int = 252,
    rf: float = 0.02,
    max_weight: float = 0.4,
) -> pd.Series:
    """
    统一权重计算入口，返回 pd.Series(index=stocks, values=权重)，权重之和为 1。

    Parameters
    ----------
    method        : 配置方式名称（equal/factor_score/min_variance/mvo/max_return）
    stocks        : 目标组股票列表
    factor_values : 当期截面因子值（index 为 stock）
    hist_returns  : 调仓日前的日频收益率 DataFrame（index=日期，columns=股票）
    lookback      : 优化方法使用的历史窗口（交易日数）
    rf            : 年化无风险利率
    max_weight    : 单只标的最大权重
    """
    n = len(stocks)
    if n == 0:
        return pd.Series(dtype=float)

    if method == "equal":
        w = equal_weight(n)
        return pd.Series(w, index=stocks)

    if method == "factor_score":
        fv = factor_values.reindex(stocks).fillna(0.0).values
        w = factor_score_weight(fv)
        return pd.Series(w, index=stocks)

    # 优化类方法需要历史收益率
    rf_daily = rf / 252
    valid_stocks = [s for s in stocks if s in hist_returns.columns]
    if len(valid_stocks) < 2:
        return pd.Series(equal_weight(n), index=stocks)

    hist = hist_returns[valid_stocks].tail(lookback)
    hist_clean = hist.dropna(how="all").ffill().fillna(0.0)

    if len(hist_clean) < 5:
        return pd.Series(equal_weight(n), index=stocks)

    ret_mat = hist_clean.values

    if method == "min_variance":
        w_arr = min_variance_weight(ret_mat, max_weight=max_weight)
    elif method == "mvo":
        w_arr = mvo_weight(ret_mat, rf_daily=rf_daily, max_weight=max_weight)
    elif method == "max_return":
        w_arr = max_return_weight(ret_mat, max_weight=max_weight)
    else:
        w_arr = equal_weight(len(valid_stocks))

    result = pd.Series(0.0, index=stocks)
    result[valid_stocks] = w_arr
    total = result.sum()
    if total > 1e-12:
        result = result / total
    else:
        result = pd.Series(equal_weight(n), index=stocks)
    return result
