"""
选定因子参考 (config/selected_factors_reference.py)
==================================================
仅供人工查阅的参考文件，包含因子 [95, 101, 62, 65, 32] 的完整代码与元数据。
本文件不应被任何代码 import 或调用。
"""

import numpy as np
import pandas as pd


# =============================================================================
# 因子编号与名称
# =============================================================================
REFERENCE_FACTOR_INDICES = [95, 101, 62, 65, 32]
REFERENCE_FACTOR_NAMES = ["alpha095", "alpha101", "alpha062", "alpha065", "alpha032"]


# =============================================================================
# 辅助函数（与 factor_library 一致，供下方 alpha 使用）
# =============================================================================

def rank(df):
    """横截面排名（每日按所有标的排序），返回百分位 [0, 1]"""
    return df.rank(axis=1, pct=True)


def delay(df, n):
    """n 期滞后：df.shift(n)"""
    return df.shift(n)


def sma(df, n):
    """n 期简单移动平均"""
    return df.rolling(n).mean()


def ts_min(df, n):
    """n 期滚动最小值"""
    return df.rolling(n).min()


def ts_rank(df, n):
    """n 期时间序列排名：当前值在过去 n 期中的排名（1-indexed）"""
    return df.rolling(n).apply(
        lambda x: pd.Series(x).rank().iloc[-1], raw=True
    )


def correlation(df1, df2, n):
    """两个 DataFrame 逐列（同名列之间）的 n 期滚动 Pearson 相关系数"""
    result = pd.DataFrame(index=df1.index, columns=df1.columns, dtype=float)
    for col in df1.columns:
        if col in df2.columns:
            result[col] = df1[col].rolling(n).corr(df2[col])
    return result


def scale(df, k=1):
    """横截面缩放：每日对所有标的缩放，使 sum(|x|) = k"""
    abs_sum = np.abs(df).sum(axis=1)
    return df.mul(k).div(abs_sum.where(abs_sum != 0), axis=0)


# =============================================================================
# Alpha#32
# =============================================================================
# 公式: scale((sma(close,7)/7 - close)) + 20*scale(correlation(vwap,delay(close,5),230))
# 逻辑: 收盘价偏离 7 日均线加 VWAP 与滞后收盘价长期相关系数，双重横截面标准化
# data_keys: ['close', 'vwap']

def alpha032(close, vwap):
    corr = correlation(vwap, delay(close, 5), 230).replace([-np.inf, np.inf], 0).fillna(0)
    return scale(sma(close, 7) / 7 - close) + 20 * scale(corr)


# =============================================================================
# Alpha#62
# =============================================================================
# 公式: (rank(correlation(vwap,sma(adv20,22),10)) < rank((rank(open)+rank(open) < rank((high+low)/2)+rank(high)))) * -1
# 逻辑: VWAP 与均量均值的相关系数排名，与开盘与高低价排名条件比较，取负
# data_keys: ['volume', 'open', 'high', 'low', 'vwap']

def alpha062(volume, open_, high, low, vwap):
    adv20 = sma(volume, 20)
    corr = correlation(vwap, sma(adv20, 22), 10).replace([-np.inf, np.inf], 0).fillna(0)
    cond = (rank(open_) + rank(open_)) < (rank((high + low) / 2) + rank(high))
    return (rank(corr) < rank(cond.astype(float))).astype(float) * -1


# =============================================================================
# Alpha#65
# =============================================================================
# 公式: (rank(correlation(open*0.008+vwap*0.992, sma(adv60,9), 6)) < rank(open - ts_min(open,14))) * -1
# 逻辑: 加权开盘 VWAP 与均量相关的排名，与开盘偏离 14 期低点的排名比较
# data_keys: ['volume', 'open', 'vwap']

def alpha065(volume, open_, vwap):
    adv60 = sma(volume, 60)
    mix = open_ * 0.00817205 + vwap * (1 - 0.00817205)
    corr = correlation(mix, sma(adv60, 9), 6).replace([-np.inf, np.inf], 0).fillna(0)
    return (rank(corr) < rank(open_ - ts_min(open_, 14))).astype(float) * -1


# =============================================================================
# Alpha#95
# =============================================================================
# 公式: rank(open - ts_min(open,12)) < ts_rank(rank(correlation(sma((high+low)/2,19), sma(adv40,19),13))^5, 12)
# 逻辑: 开盘偏离 12 期低点的排名，与中间价均量相关五次方排名的时序排名比较
# data_keys: ['volume', 'open', 'high', 'low']

def alpha095(volume, open_, high, low):
    adv40 = sma(volume, 40)
    corr = correlation(sma((high + low) / 2, 19), sma(adv40, 19), 13).replace([-np.inf, np.inf], 0).fillna(0)
    inner = rank(corr) ** 5
    return (rank(open_ - ts_min(open_, 12)) < ts_rank(inner, 12)).astype(float)


# =============================================================================
# Alpha#101
# =============================================================================
# 公式: (close - open) / ((high - low) + 0.001)
# 逻辑: 日内收益（收盘开盘差）除以振幅，衡量日内方向强度
# data_keys: ['open', 'high', 'low', 'close']

def alpha101(open_, high, low, close):
    return (close - open_) / ((high - low) + 0.001)


# =============================================================================
# 元数据（供查阅）
# =============================================================================
REFERENCE_FACTOR_DETAILS = {
    32: {
        "name": "alpha032",
        "formula": "scale((sma(close,7)/7 - close)) + 20*scale(correlation(vwap,delay(close,5),230))",
        "theory": "收盘价偏离 7 日均线加 VWAP 与滞后收盘价长期相关系数，双重横截面标准化",
        "direction": "混合",
        "holding_period": "长期",
        "category": "价格/VWAP",
        "data_keys": ["close", "vwap"],
    },
    62: {
        "name": "alpha062",
        "formula": "(rank(correlation(vwap,sma(adv20,22),10)) < rank((rank(open)+rank(open) < rank((high+low)/2)+rank(high)))) * -1",
        "theory": "VWAP与均量相关系数排名与开盘高低价排名条件比较",
        "direction": "负向",
        "holding_period": "短期",
        "category": "量价/VWAP",
        "data_keys": ["volume", "open", "high", "low", "vwap"],
    },
    65: {
        "name": "alpha065",
        "formula": "(rank(correlation(open*0.008+vwap*0.992, sma(adv60,9), 6)) < rank(open - ts_min(open,14))) * -1",
        "theory": "加权开盘VWAP与均量相关排名与开盘偏离低点排名比较",
        "direction": "负向",
        "holding_period": "短期",
        "category": "量价/VWAP",
        "data_keys": ["volume", "open", "vwap"],
    },
    95: {
        "name": "alpha095",
        "formula": "rank(open - ts_min(open,12)) < ts_rank(rank(correlation(sma((high+low)/2,19), sma(adv40,19),13))^5, 12)",
        "theory": "开盘偏离12期低点排名与中间价均量相关五次方时序排名比较",
        "direction": "混合",
        "holding_period": "短期",
        "category": "量价",
        "data_keys": ["volume", "open", "high", "low"],
    },
    101: {
        "name": "alpha101",
        "formula": "(close - open) / ((high - low) + 0.001)",
        "theory": "日内收益（收盘开盘差）除以振幅，衡量日内方向强度",
        "direction": "正向",
        "holding_period": "极短期",
        "category": "价格",
        "data_keys": ["open", "high", "low", "close"],
    },
}
