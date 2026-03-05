"""
因子库 (factors/factor_library.py)
=====================================
本模块实现 WorldQuant 101 Alpha 中的因子（Alpha#1–101，除需行业中性化的 #48、#56、#58–59、#63、#67、#69–70、#76、#79–80、#82、#87、#89–91、#93、#97、#100），
以及供 pipeline 使用的配置与元数据。

因子函数约定：
- 所有辅助函数与 alpha 函数均接受宽格式 DataFrame（index=日期，columns=标的代码）。
- rank() 为横截面排名（每日对所有标的排序，pct=True，输出 [0,1]）。
- ts_* 系列为时间序列（滚动窗口）操作，对每列独立计算。
- 所有 alpha 函数返回 DataFrame，index=日期，columns=标的代码。

数据说明：
- close   : 复权收盘价（Adj Close）
- open    : 开盘价（Open）
- high    : 最高价（High）
- low     : 最低价（Low）
- volume  : 成交量（Volume）
- returns : 日收益率（close.pct_change()）
- vwap    : 成交量加权均价（近似为 (high+low+close)/3，即典型价格）

FACTOR_CONFIGS：因子名 → {func, data_keys}，供 build_factors.py 使用。
VOLUME_REQUIRED_FACTORS：保留空字典以兼容原有 pipeline 接口。
FACTOR_DESCRIPTIONS：因子显示名、理论等元数据。
"""

import numpy as np
import pandas as pd


# ============================================================
# 辅助函数（宽格式 DataFrame 操作）
# ============================================================

def rank(df):
    """横截面排名（每日按所有标的排序），返回百分位 [0, 1]"""
    return df.rank(axis=1, pct=True)


def delta(df, n):
    """n 期差分：df - df.shift(n)"""
    return df.diff(n)


def delay(df, n):
    """n 期滞后：df.shift(n)"""
    return df.shift(n)


def log(df):
    """自然对数"""
    return np.log(df)


def stddev(df, n):
    """n 期滚动标准差"""
    return df.rolling(n).std()


def sma(df, n):
    """n 期简单移动平均"""
    return df.rolling(n).mean()


def ts_sum(df, n):
    """n 期滚动求和"""
    return df.rolling(n).sum()


def ts_min(df, n):
    """n 期滚动最小值"""
    return df.rolling(n).min()


def ts_max(df, n):
    """n 期滚动最大值"""
    return df.rolling(n).max()


def ts_rank(df, n):
    """
    n 期时间序列排名：当前值在过去 n 期中的排名（1-indexed）。
    与原公式中的 Ts_Rank 对应。
    """
    return df.rolling(n).apply(
        lambda x: pd.Series(x).rank().iloc[-1], raw=True
    )


def ts_argmax(df, n):
    """n 期滚动窗口内最大值的位置（0-indexed）"""
    return df.rolling(n).apply(np.argmax, raw=True)


def ts_argmin(df, n):
    """n 期滚动窗口内最小值的位置（0-indexed）"""
    return df.rolling(n).apply(np.argmin, raw=True)


def correlation(df1, df2, n):
    """
    两个 DataFrame 逐列（同名列之间）的 n 期滚动 Pearson 相关系数。
    结果中 ±inf 与极端值需由调用方处理。
    """
    result = pd.DataFrame(index=df1.index, columns=df1.columns, dtype=float)
    for col in df1.columns:
        if col in df2.columns:
            result[col] = df1[col].rolling(n).corr(df2[col])
    return result


def covariance(df1, df2, n):
    """两个 DataFrame 逐列（同名列之间）的 n 期滚动协方差"""
    result = pd.DataFrame(index=df1.index, columns=df1.columns, dtype=float)
    for col in df1.columns:
        if col in df2.columns:
            result[col] = df1[col].rolling(n).cov(df2[col])
    return result


def sign(df):
    """符号函数：返回 -1、0 或 1"""
    return np.sign(df)


def SignedPower(df, e):
    """有符号幂函数：sign(x) * |x|^e"""
    return np.sign(df) * (np.abs(df) ** e)


def product(df, n):
    """n 期滚动乘积"""
    return df.rolling(n).apply(lambda x: np.prod(x), raw=True)


def scale(df, k=1):
    """
    横截面缩放：每日对所有标的缩放，使 sum(|x|) = k。
    若某日所有绝对值之和为 0，则该日结果为 NaN。
    """
    abs_sum = np.abs(df).sum(axis=1)
    return df.mul(k).div(abs_sum.where(abs_sum != 0), axis=0)


def decay_linear(df, n):
    """
    线性加权移动平均（LWMA）。权重从旧到新线性增加，最近一期权重最大。
    weights = [1, 2, ..., n] / sum([1, ..., n])
    """
    weights = np.arange(1, n + 1, dtype=float)
    weights /= weights.sum()
    return df.rolling(n).apply(lambda x: np.dot(x, weights), raw=True)


# ============================================================
# WorldQuant 101 Alpha 因子（Alpha#1 ~ Alpha#10）
# ============================================================

def alpha001(close, returns):
    """
    Alpha#1: rank(Ts_ArgMax(SignedPower(((returns<0)?stddev(returns,20):close), 2.), 5)) - 0.5

    逻辑：当收益率为负时用收益率波动率替代价格，再取有符号平方，然后取 5 期 argmax 的横截面排名。
    """
    std_rets = stddev(returns, 20)
    inner = pd.DataFrame(
        np.where(returns.values < 0, std_rets.values, close.values),
        index=close.index, columns=close.columns,
    )
    return rank(ts_argmax(SignedPower(inner, 2), 5)) - 0.5


def alpha002(close, open_, volume):
    """
    Alpha#2: -1 * correlation(rank(delta(log(volume), 2)), rank((close-open)/open), 6)

    逻辑：成交量 2 日对数差分的横截面排名与价格日内涨幅排名的滚动相关系数。
    """
    df = -1 * correlation(
        rank(delta(log(volume), 2)),
        rank((close - open_) / open_),
        6,
    )
    return df.replace([-np.inf, np.inf], 0).fillna(0)


def alpha003(open_, volume):
    """
    Alpha#3: -1 * correlation(rank(open), rank(volume), 10)

    逻辑：开盘价排名与成交量排名的 10 期滚动相关系数的负值。
    """
    df = -1 * correlation(rank(open_), rank(volume), 10)
    return df.replace([-np.inf, np.inf], 0).fillna(0)


def alpha004(low):
    """
    Alpha#4: -1 * Ts_Rank(rank(low), 9)

    逻辑：最低价横截面排名的 9 期时间序列排名的负值。
    """
    return -1 * ts_rank(rank(low), 9)


def alpha005(open_, vwap, close):
    """
    Alpha#5: rank((open - sum(vwap,10)/10)) * (-1 * abs(rank((close - vwap))))

    逻辑：开盘价偏离 10 期均 VWAP 的横截面排名，乘以收盘偏离 VWAP 排名的负绝对值。
    """
    return rank(open_ - ts_sum(vwap, 10) / 10) * (-1 * np.abs(rank(close - vwap)))


def alpha006(open_, volume):
    """
    Alpha#6: -1 * correlation(open, volume, 10)

    逻辑：开盘价与成交量的 10 期滚动相关系数的负值。
    """
    df = -1 * correlation(open_, volume, 10)
    return df.replace([-np.inf, np.inf], 0).fillna(0)


def alpha007(close, volume):
    """
    Alpha#7: (adv20 < volume) ? (-1*ts_rank(abs(delta(close,7)),60)*sign(delta(close,7))) : -1

    逻辑：当近期成交量高于 20 日均量时，使用价格动量的反向时间排名；否则设为 -1。
    """
    adv20 = sma(volume, 20)
    momentum = -1 * ts_rank(np.abs(delta(close, 7)), 60) * sign(delta(close, 7))
    alpha = pd.DataFrame(
        np.where(adv20.values >= volume.values, -1.0, momentum.values),
        index=close.index, columns=close.columns,
    )
    return alpha


def alpha008(open_, returns):
    """
    Alpha#8: -1 * rank((sum(open,5)*sum(returns,5)) - delay((sum(open,5)*sum(returns,5)), 10))

    逻辑：开盘价 5 日之和与收益率 5 日之和的乘积，相对 10 日前的变化的横截面排名的负值。
    """
    val = ts_sum(open_, 5) * ts_sum(returns, 5)
    return -1 * rank(val - delay(val, 10))


def alpha009(close):
    """
    Alpha#9: (0 < ts_min(delta(close,1),5)) ? delta(close,1) :
             ((ts_max(delta(close,1),5) < 0) ? delta(close,1) : -1*delta(close,1))

    逻辑：若 5 期内最小日涨跌幅为正（持续上涨）或最大日涨跌幅为负（持续下跌），
    跟随趋势；否则反转。
    """
    delta_close = delta(close, 1)
    cond_1 = ts_min(delta_close, 5) > 0
    cond_2 = ts_max(delta_close, 5) < 0
    alpha = pd.DataFrame(
        np.where(
            cond_1.values | cond_2.values,
            delta_close.values,
            (-1 * delta_close).values,
        ),
        index=close.index, columns=close.columns,
    )
    return alpha


def alpha010(close):
    """
    Alpha#10: rank((0<ts_min(delta(close,1),4)) ? delta(close,1) :
                   ((ts_max(delta(close,1),4)<0) ? delta(close,1) : -1*delta(close,1)))

    逻辑：与 alpha009 类似，但窗口改为 4 期，并对最终结果做横截面排名。
    """
    delta_close = delta(close, 1)
    cond_1 = ts_min(delta_close, 4) > 0
    cond_2 = ts_max(delta_close, 4) < 0
    alpha = pd.DataFrame(
        np.where(
            cond_1.values | cond_2.values,
            delta_close.values,
            (-1 * delta_close).values,
        ),
        index=close.index, columns=close.columns,
    )
    return rank(alpha)


# ============================================================
# WorldQuant 101 Alpha 因子（Alpha#11 ~ Alpha#50）
# ============================================================

def alpha011(vwap, close, volume):
    """
    Alpha#11: (rank(ts_max(vwap-close,3)) + rank(ts_min(vwap-close,3))) * rank(delta(volume,3))

    逻辑：VWAP 与收盘价差值的极值排名之和，与成交量 3 期变化排名相乘。
    """
    return (
        (rank(ts_max(vwap - close, 3)) + rank(ts_min(vwap - close, 3)))
        * rank(delta(volume, 3))
    )


def alpha012(volume, close):
    """
    Alpha#12: sign(delta(volume,1)) * (-1 * delta(close,1))

    逻辑：成交量变化方向乘以价格变化的负值，量增价跌时为正。
    """
    return sign(delta(volume, 1)) * (-1 * delta(close, 1))


def alpha013(close, volume):
    """
    Alpha#13: -1 * rank(covariance(rank(close), rank(volume), 5))

    逻辑：收盘价排名与成交量排名的 5 期协方差的横截面排名的负值。
    """
    return -1 * rank(covariance(rank(close), rank(volume), 5))


def alpha014(returns, open_, volume):
    """
    Alpha#14: (-1 * rank(delta(returns,3))) * correlation(open,volume,10)

    逻辑：收益率 3 期差分排名（负）乘以开盘价与成交量的 10 期相关系数。
    """
    df = correlation(open_, volume, 10).replace([-np.inf, np.inf], 0).fillna(0)
    return -1 * rank(delta(returns, 3)) * df


def alpha015(high, volume):
    """
    Alpha#15: -1 * ts_sum(rank(correlation(rank(high),rank(volume),3)),3)

    逻辑：最高价排名与成交量排名的 3 期相关系数的横截面排名的 3 期和的负值。
    """
    df = correlation(rank(high), rank(volume), 3).replace([-np.inf, np.inf], 0).fillna(0)
    return -1 * ts_sum(rank(df), 3)


def alpha016(high, volume):
    """
    Alpha#16: -1 * rank(covariance(rank(high),rank(volume),5))

    逻辑：最高价排名与成交量排名的 5 期协方差的横截面排名的负值。
    """
    return -1 * rank(covariance(rank(high), rank(volume), 5))


def alpha017(close, volume):
    """
    Alpha#17: -1 * rank(ts_rank(close,10)) * rank(delta(delta(close,1),1)) * rank(ts_rank(volume/adv20,5))

    逻辑：价格时序排名、二阶差分排名与相对成交量时序排名的三因子乘积的负值。
    """
    adv20 = sma(volume, 20)
    return -1 * (
        rank(ts_rank(close, 10))
        * rank(delta(delta(close, 1), 1))
        * rank(ts_rank(volume / adv20, 5))
    )


def alpha018(close, open_):
    """
    Alpha#18: -1 * rank((stddev(|close-open|,5) + (close-open)) + correlation(close,open,10))

    逻辑：日内振幅波动性、日内偏差与价格开收相关系数之和的排名负值。
    """
    df = correlation(close, open_, 10).replace([-np.inf, np.inf], 0).fillna(0)
    return -1 * rank((stddev(np.abs(close - open_), 5) + (close - open_)) + df)


def alpha019(close, returns):
    """
    Alpha#19: (-1 * sign((close-delay(close,7)) + delta(close,7))) * (1+rank(1+ts_sum(returns,250)))

    逻辑：7 日价格方向的反向信号，乘以年化收益率累积排名的放大因子。
    """
    return (
        (-1 * sign((close - delay(close, 7)) + delta(close, 7)))
        * (1 + rank(1 + ts_sum(returns, 250)))
    )


def alpha020(open_, high, close, low):
    """
    Alpha#20: -1 * rank(open-delay(high,1)) * rank(open-delay(close,1)) * rank(open-delay(low,1))

    逻辑：今日开盘相对昨日高、低、收盘价差值排名的三因子乘积负值，捕捉隔夜跳空反转。
    """
    return -1 * (
        rank(open_ - delay(high, 1))
        * rank(open_ - delay(close, 1))
        * rank(open_ - delay(low, 1))
    )


def alpha021(close, volume):
    """
    Alpha#21: 条件因子，基于收盘价均值±波动与成交量相对均量的比较

    逻辑：当价格短期均值超过长期均值加波动，或成交量低于均量时，返回 -1；否则返回 1。
    """
    adv20 = sma(volume, 20)
    cond_1 = sma(close, 8) + stddev(close, 8) < sma(close, 2)
    cond_2 = adv20 / volume < 1
    return pd.DataFrame(
        np.where((cond_1 | cond_2).values, -1.0, 1.0),
        index=close.index, columns=close.columns,
    )


def alpha022(high, close, volume):
    """
    Alpha#22: -1 * delta(correlation(high,volume,5),5) * rank(stddev(close,20))

    逻辑：高价量相关系数的 5 期变化，乘以收盘价长期波动排名的负值。
    """
    df = correlation(high, volume, 5).replace([-np.inf, np.inf], 0).fillna(0)
    return -1 * delta(df, 5) * rank(stddev(close, 20))


def alpha023(high):
    """
    Alpha#23: (sma(high,20) < high) ? (-1 * delta(high,2)) : 0

    逻辑：当高价突破 20 日均线时，取高价 2 期差分的负值；否则为 0。
    """
    cond = sma(high, 20) < high
    return pd.DataFrame(
        np.where(cond.values, (-1 * delta(high, 2)).fillna(0).values, 0.0),
        index=high.index, columns=high.columns,
    )


def alpha024(close):
    """
    Alpha#24: (delta(sma(close,100),100)/delay(close,100)<=0.05) ?
              (-1*(close-ts_min(close,100))) : (-1*delta(close,3))

    逻辑：价格长期趋势平缓时使用偏离长期低点的负值，否则用短期动量反转。
    """
    cond = delta(sma(close, 100), 100) / delay(close, 100) <= 0.05
    return pd.DataFrame(
        np.where(
            cond.values,
            (-1 * (close - ts_min(close, 100))).values,
            (-1 * delta(close, 3)).values,
        ),
        index=close.index, columns=close.columns,
    )


def alpha025(returns, volume, vwap, high, close):
    """
    Alpha#25: rank((-1*returns) * adv20 * vwap * (high-close))

    逻辑：反向收益率、均量、VWAP 与日内振幅的四因子乘积的横截面排名。
    """
    adv20 = sma(volume, 20)
    return rank((-1 * returns) * adv20 * vwap * (high - close))


def alpha026(volume, high):
    """
    Alpha#26: -1 * ts_max(correlation(ts_rank(volume,5),ts_rank(high,5),5),3)

    逻辑：成交量时序排名与高价时序排名的 5 期相关系数的 3 期最大值的负值。
    """
    df = correlation(ts_rank(volume, 5), ts_rank(high, 5), 5).replace([-np.inf, np.inf], 0).fillna(0)
    return -1 * ts_max(df, 3)


def alpha027(volume, vwap):
    """
    Alpha#27: (rank(sma(correlation(rank(volume),rank(vwap),6),2)/2) > 0.5) ? -1 : 1

    逻辑：量价相关系数的 2 日均值排名，高于中位数时做空，否则做多。
    """
    alpha = rank(sma(correlation(rank(volume), rank(vwap), 6), 2) / 2.0)
    return pd.DataFrame(
        np.where(alpha.values > 0.5, -1.0, 1.0),
        index=alpha.index, columns=alpha.columns,
    )


def alpha028(volume, low, high, close):
    """
    Alpha#28: scale(correlation(adv20,low,5) + (high+low)/2 - close)

    逻辑：均量与低价的相关系数加上中间价偏离收盘价的横截面标准化。
    """
    adv20 = sma(volume, 20)
    df = correlation(adv20, low, 5).replace([-np.inf, np.inf], 0).fillna(0)
    return scale(df + (high + low) / 2 - close)


def alpha029(close, returns):
    """
    Alpha#29: ts_min(rank(rank(scale(log(ts_sum(rank(rank(-1*rank(delta(close-1,5)))),2))))),5)
              + ts_rank(delay(-1*returns,6),5)

    逻辑：价格差分多层嵌套排名的对数变换 5 期最小值，加上反向收益率的时序排名。
    """
    inner = ts_sum(rank(rank(-1 * rank(delta(close - 1, 5)))), 2).clip(lower=1e-10)
    return (
        ts_min(rank(rank(scale(log(inner)))), 5)
        + ts_rank(delay(-1 * returns, 6), 5)
    )


def alpha030(close, volume):
    """
    Alpha#30: (1-rank(sign(delta(close,1))+sign(delay(delta(close,1),1))+sign(delay(delta(close,1),2))))
              * ts_sum(volume,5) / ts_sum(volume,20)

    逻辑：过去 3 日价格方向一致性排名的反向，乘以成交量短期与长期比值。
    """
    delta_close = delta(close, 1)
    inner = sign(delta_close) + sign(delay(delta_close, 1)) + sign(delay(delta_close, 2))
    return ((1.0 - rank(inner)) * ts_sum(volume, 5)) / ts_sum(volume, 20)


def alpha031(close, volume, low):
    """
    Alpha#31: rank(rank(rank(decay_linear(-1*rank(rank(delta(close,10))),10))))
              + rank(-1*delta(close,3)) + sign(scale(correlation(adv20,low,12)))

    逻辑：价格 10 期动量的多层排名线性衰减，加上短期动量反转，加上量价关系方向。
    """
    adv20 = sma(volume, 20)
    df = correlation(adv20, low, 12).replace([-np.inf, np.inf], 0).fillna(0)
    p1 = rank(rank(rank(decay_linear(-1 * rank(rank(delta(close, 10))), 10))))
    p2 = rank(-1 * delta(close, 3))
    p3 = sign(scale(df))
    return p1 + p2 + p3


def alpha032(close, vwap):
    """
    Alpha#32: scale((sma(close,7)/7 - close)) + 20*scale(correlation(vwap,delay(close,5),230))

    逻辑：收盘价偏离 7 日均线，加上 VWAP 与 5 日前收盘价的长期相关系数，双重横截面标准化。
    """
    corr = correlation(vwap, delay(close, 5), 230).replace([-np.inf, np.inf], 0).fillna(0)
    return scale(sma(close, 7) / 7 - close) + 20 * scale(corr)


def alpha033(open_, close):
    """
    Alpha#33: rank(-1 + open/close)

    逻辑：开盘价相对收盘价偏差的横截面排名，捕捉隔夜效应。
    """
    return rank(-1 + open_ / close)


def alpha034(returns, close):
    """
    Alpha#34: rank(2 - rank(stddev(returns,2)/stddev(returns,5)) - rank(delta(close,1)))

    逻辑：短期波动率与长期波动率之比排名，加上价格动量排名的反向组合。
    """
    inner = (stddev(returns, 2) / stddev(returns, 5)).replace([-np.inf, np.inf], 1).fillna(1)
    return rank(2 - rank(inner) - rank(delta(close, 1)))


def alpha035(volume, close, high, low, returns):
    """
    Alpha#35: ts_rank(volume,32) * (1-ts_rank(close+high-low,16)) * (1-ts_rank(returns,32))

    逻辑：成交量时序排名乘以价格振幅时序排名的反向，再乘以收益率时序排名的反向。
    """
    return (
        ts_rank(volume, 32)
        * (1 - ts_rank(close + high - low, 16))
        * (1 - ts_rank(returns, 32))
    )


def alpha036(close, open_, returns, volume, vwap):
    """
    Alpha#36: 2.21*rank(corr(close-open,delay(volume,1),15)) + 0.7*rank(open-close)
              + 0.73*rank(ts_rank(delay(-1*returns,6),5)) + rank(|corr(vwap,adv20,6)|)
              + 0.6*rank((sma(close,200)/200-open)*(close-open))

    逻辑：收盘开盘差与滞后成交量相关 + 日内反转 + 滞后收益时序排名 + 量价相关 + 长期均价偏差。
    """
    adv20 = sma(volume, 20)
    corr_co_vol = correlation(close - open_, delay(volume, 1), 15).replace([-np.inf, np.inf], 0).fillna(0)
    corr_vwap_adv20 = correlation(vwap, adv20, 6).replace([-np.inf, np.inf], 0).fillna(0)
    return (
        2.21 * rank(corr_co_vol)
        + 0.7 * rank(open_ - close)
        + 0.73 * rank(ts_rank(delay(-1 * returns, 6), 5))
        + rank(np.abs(corr_vwap_adv20))
        + 0.6 * rank((sma(close, 200) / 200 - open_) * (close - open_))
    )


def alpha037(open_, close):
    """
    Alpha#37: rank(correlation(delay(open-close,1),close,200)) + rank(open-close)

    逻辑：昨日日内偏差与收盘价的长期相关系数排名，加上今日日内偏差排名。
    """
    corr = correlation(delay(open_ - close, 1), close, 200).replace([-np.inf, np.inf], 0).fillna(0)
    return rank(corr) + rank(open_ - close)


def alpha038(open_, close):
    """
    Alpha#38: -1 * rank(ts_rank(open,10)) * rank(close/open)

    逻辑：开盘价时序排名乘以收盘开盘比排名的负值，捕捉开盘强势后的反转。
    """
    inner = (close / open_).replace([-np.inf, np.inf], 1).fillna(1)
    return -1 * rank(ts_rank(open_, 10)) * rank(inner)


def alpha039(close, volume, returns):
    """
    Alpha#39: -1 * rank(delta(close,7)*(1-rank(decay_linear(volume/adv20,9))))
              * (1+rank(sma(returns,250)))

    逻辑：7 期价格动量乘以相对成交量线性衰减排名的反向，再乘以年化收益率趋势排名的放大。
    """
    adv20 = sma(volume, 20)
    return (
        (-1 * rank(delta(close, 7) * (1 - rank(decay_linear(volume / adv20, 9)))))
        * (1 + rank(sma(returns, 250)))
    )


def alpha040(high, volume):
    """
    Alpha#40: -1 * rank(stddev(high,10)) * correlation(high,volume,10)

    逻辑：高价波动率排名乘以高价量相关系数的负值，波动大且量价同向时做空。
    """
    return (
        -1 * rank(stddev(high, 10))
        * correlation(high, volume, 10).replace([-np.inf, np.inf], 0).fillna(0)
    )


def alpha041(high, low, vwap):
    """
    Alpha#41: (high*low)^0.5 - vwap

    逻辑：高低价几何均值与 VWAP 的差值，高低价均值高于 VWAP 时为正。
    """
    return np.power(high * low, 0.5) - vwap


def alpha042(vwap, close):
    """
    Alpha#42: rank(vwap-close) / rank(vwap+close)

    逻辑：VWAP 与收盘价差值排名除以二者之和排名，衡量价格相对 VWAP 的偏离方向。
    """
    return rank(vwap - close) / rank(vwap + close)


def alpha043(volume, close):
    """
    Alpha#43: ts_rank(volume/adv20,20) * ts_rank(-1*delta(close,7),8)

    逻辑：相对成交量时序排名乘以 7 期价格反转的时序排名。
    """
    adv20 = sma(volume, 20)
    return ts_rank(volume / adv20, 20) * ts_rank(-1 * delta(close, 7), 8)


def alpha044(high, volume):
    """
    Alpha#44: -1 * correlation(high, rank(volume), 5)

    逻辑：高价与成交量排名的 5 期相关系数的负值。
    """
    return -1 * correlation(high, rank(volume), 5).replace([-np.inf, np.inf], 0).fillna(0)


def alpha045(close, volume):
    """
    Alpha#45: -1 * rank(sma(delay(close,5),20)) * correlation(close,volume,2)
              * rank(correlation(ts_sum(close,5),ts_sum(close,20),2))

    逻辑：滞后收盘价均线排名、量价短期相关与多期收盘价相关的三因子乘积的负值。
    """
    df = correlation(close, volume, 2).replace([-np.inf, np.inf], 0).fillna(0)
    corr2 = correlation(ts_sum(close, 5), ts_sum(close, 20), 2).replace([-np.inf, np.inf], 0).fillna(0)
    return -1 * (rank(sma(delay(close, 5), 20)) * df * rank(corr2))


def alpha046(close):
    """
    Alpha#46: 基于收盘价加速度（二阶差分）的趋势/反转信号

    逻辑：价格加速趋势强（>0.25）时做空，加速趋势负（<0）时做多，
    近似中性时跟随日内变化反转。
    """
    inner = ((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)
    base = (-1 * delta(close, 1)).values.copy()
    base[inner.values < 0] = 1
    base[inner.values > 0.25] = -1
    return pd.DataFrame(base, index=close.index, columns=close.columns)


def alpha047(close, volume, high, vwap):
    """
    Alpha#47: rank(1/close)*volume/adv20 * high*rank(high-close)/(sma(high,5)/5)
              - rank(vwap-delay(vwap,5))

    逻辑：低价格倒数乘以相对成交量，加权以高价偏离收盘，减去 VWAP 5 期动量排名。
    """
    adv20 = sma(volume, 20)
    return (
        ((rank(1 / close) * volume) / adv20)
        * (high * rank(high - close)) / (sma(high, 5) / 5)
        - rank(vwap - delay(vwap, 5))
    )


def alpha049(close):
    """
    Alpha#49: (加速度 < -0.1) ? 1 : -1*delta(close,1)

    逻辑：价格加速度低于 -0.1 阈值时做多，否则按日内变化反转。
    """
    inner = ((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)
    base = (-1 * delta(close, 1)).values.copy()
    base[inner.values < -0.1] = 1
    return pd.DataFrame(base, index=close.index, columns=close.columns)


def alpha050(volume, vwap):
    """
    Alpha#50: -1 * ts_max(rank(correlation(rank(volume),rank(vwap),5)),5)

    逻辑：量价排名相关系数横截面排名的 5 期最大值的负值，量价高度相关时做空。
    """
    corr = correlation(rank(volume), rank(vwap), 5).replace([-np.inf, np.inf], 0).fillna(0)
    return -1 * ts_max(rank(corr), 5)


# ============================================================
# WorldQuant 101 Alpha 因子（Alpha#51 ~ Alpha#101）
# ============================================================

def alpha051(close):
    """
    Alpha#51: (加速度 < -0.05) ? 1 : -1*delta(close,1)

    逻辑：价格加速度低于 -0.05 时做多，否则按日内变化反转。
    """
    inner = ((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)
    base = (-1 * delta(close, 1)).values.copy()
    base[inner.values < -0.05] = 1
    return pd.DataFrame(base, index=close.index, columns=close.columns)


def alpha052(returns, low, volume):
    """
    Alpha#52: ((-1*delta(ts_min(low,5),5)) * rank((ts_sum(returns,240)-ts_sum(returns,20))/220))
              * ts_rank(volume,5)

    逻辑：低价 5 期最小值的 5 期差分负值，乘以长期/短期收益和之差的排名，乘以成交量时序排名。
    """
    return (
        (-1 * delta(ts_min(low, 5), 5))
        * rank((ts_sum(returns, 240) - ts_sum(returns, 20)) / 220)
        * ts_rank(volume, 5)
    )


def alpha053(close, high, low):
    """
    Alpha#53: -1 * delta(((close-low)-(high-close))/(close-low), 9)

    逻辑：日内位置（相对振幅）的 9 期差分负值。
    """
    inner = (close - low).replace(0, 1e-10)
    return -1 * delta(((close - low) - (high - close)) / inner, 9)


def alpha054(close, open_, high, low):
    """
    Alpha#54: -1 * (low-close) * (open^5) / ((low-high) * (close^5))

    逻辑：低价偏离收盘与开盘五次方，除以振幅与收盘五次方。
    """
    denom = (low - high).replace(0, -1e-10)
    return -1 * (low - close) * (open_ ** 5) / (denom * (close ** 5))


def alpha055(close, high, low, volume):
    """
    Alpha#55: -1 * correlation(rank((close-ts_min(low,12))/(ts_max(high,12)-ts_min(low,12))),
                              rank(volume), 6)

    逻辑：收盘在振幅内位置的排名与成交量排名的 6 期相关系数负值。
    """
    divisor = (ts_max(high, 12) - ts_min(low, 12)).replace(0, 1e-10)
    inner = (close - ts_min(low, 12)) / divisor
    df = correlation(rank(inner), rank(volume), 6)
    return -1 * df.replace([-np.inf, np.inf], 0).fillna(0)


def alpha057(close, vwap):
    """
    Alpha#57: -1 * (close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2)

    逻辑：收盘偏离 VWAP 除以收盘 30 期 argmax 排名的线性衰减。
    """
    denom = decay_linear(rank(ts_argmax(close, 30)), 2)
    denom = denom.replace(0, np.nan)
    return -1 * (close - vwap) / denom


def alpha060(close, high, low, volume):
    """
    Alpha#60: -1 * (2*scale(rank(((close-low)-(high-close))/(high-low)*volume))
                  - scale(rank(ts_argmax(close,10))))

    逻辑：日内位置乘成交量的排名缩放，减去收盘 argmax 排名的缩放。
    """
    divisor = (high - low).replace(0, 1e-10)
    inner = ((close - low) - (high - close)) * volume / divisor
    return -1 * (2 * scale(rank(inner)) - scale(rank(ts_argmax(close, 10))))


def alpha061(volume, vwap):
    """
    Alpha#61: rank(vwap - ts_min(vwap,16)) < rank(correlation(vwap, adv180, 18))

    逻辑：VWAP 偏离 16 期低点的排名，与 VWAP 和 180 日均量的相关系数排名比较。
    """
    adv180 = sma(volume, 180)
    corr = correlation(vwap, adv180, 18).replace([-np.inf, np.inf], 0).fillna(0)
    return (rank(vwap - ts_min(vwap, 16)) < rank(corr)).astype(float)


def alpha062(volume, open_, high, low, vwap):
    """
    Alpha#62: (rank(correlation(vwap,sma(adv20,22),10))
              < rank((rank(open)+rank(open) < rank((high+low)/2)+rank(high)))) * -1

    逻辑：VWAP 与均量均值的相关系数排名，与开盘与高低价排名条件比较，取负。
    """
    adv20 = sma(volume, 20)
    corr = correlation(vwap, sma(adv20, 22), 10).replace([-np.inf, np.inf], 0).fillna(0)
    cond = (rank(open_) + rank(open_)) < (rank((high + low) / 2) + rank(high))
    return (rank(corr) < rank(cond.astype(float))).astype(float) * -1


def alpha064(volume, open_, high, low, vwap):
    """
    Alpha#64: (rank(correlation(ts_sum(open*0.178+low*0.822,13), ts_sum(adv120,13), 17))
              < rank(delta(((high+low)/2*0.178+vwap*0.822), 4))) * -1

    逻辑：加权开盘低价与均量的相关系数排名，与加权中间价 VWAP 的 4 期差分排名比较。
    """
    adv120 = sma(volume, 120)
    mix1 = ts_sum(open_ * 0.178404 + low * (1 - 0.178404), 13)
    mix2 = ts_sum(adv120, 13)
    corr = correlation(mix1, mix2, 17).replace([-np.inf, np.inf], 0).fillna(0)
    hl_vwap = (high + low) / 2 * 0.178404 + vwap * (1 - 0.178404)
    return (rank(corr) < rank(delta(hl_vwap, 4))).astype(float) * -1


def alpha065(volume, open_, vwap):
    """
    Alpha#65: (rank(correlation(open*0.008+vwap*0.992, sma(adv60,9), 6))
              < rank(open - ts_min(open,14))) * -1

    逻辑：加权开盘 VWAP 与均量相关的排名，与开盘偏离 14 期低点的排名比较。
    """
    adv60 = sma(volume, 60)
    mix = open_ * 0.00817205 + vwap * (1 - 0.00817205)
    corr = correlation(mix, sma(adv60, 9), 6).replace([-np.inf, np.inf], 0).fillna(0)
    return (rank(corr) < rank(open_ - ts_min(open_, 14))).astype(float) * -1


def alpha066(open_, high, low, vwap):
    """
    Alpha#66: (rank(decay_linear(delta(vwap,4),7))
              + ts_rank(decay_linear((low-vwap)/(open-(high+low)/2), 11), 7)) * -1

    逻辑：VWAP 4 期差分的线性衰减排名，加上低价偏离 VWAP 相对开盘中间价差的衰减时序排名。
    """
    p1 = rank(decay_linear(delta(vwap, 4), 7))
    inner = (low - vwap) / (open_ - (high + low) / 2).replace(0, np.nan)
    p2 = ts_rank(decay_linear(inner, 11), 7)
    return (p1 + p2) * -1


# def alpha068(volume, close, high, low):
#     """
#     Alpha#68: (ts_rank(correlation(rank(high), rank(adv15), 9), 14)
#               < rank(delta(close*0.518+low*0.482, 1))) * -1
#
#     逻辑：高价与 15 日均量排名的相关系数时序排名，与加权收盘低价 1 期差分排名比较。
#     """
#     adv15 = sma(volume, 15)
#     corr = correlation(rank(high), rank(adv15), 9).replace([-np.inf, np.inf], 0).fillna(0)
#     mix = close * 0.518371 + low * (1 - 0.518371)
#     return (ts_rank(corr, 14) < rank(delta(mix, 1))).astype(float) * -1
# 注释：输出为0或空集


def alpha071(volume, close, open_, low, vwap):
    """
    Alpha#71: max(ts_rank(decay_linear(correlation(ts_rank(close,3),ts_rank(adv180,12),18),4),16),
                 ts_rank(decay_linear(rank((low+open)-(vwap+vwap))^2, 16), 4))

    逻辑：收盘与均量时序相关衰减的时序排名，与开盘低价偏离 VWAP 排名的衰减时序排名取大。
    """
    adv180 = sma(volume, 180)
    p1 = ts_rank(decay_linear(correlation(ts_rank(close, 3), ts_rank(adv180, 12), 18), 4), 16)
    inner = rank((low + open_) - (vwap + vwap)) ** 2
    p2 = ts_rank(decay_linear(inner, 16), 4)
    return pd.DataFrame(
        np.maximum(p1.values, p2.values),
        index=p1.index, columns=p1.columns,
    )


def alpha072(volume, high, low, vwap):
    """
    Alpha#72: rank(decay_linear(correlation((high+low)/2, adv40, 9), 10))
             / rank(decay_linear(correlation(ts_rank(vwap,4), ts_rank(volume,19), 7), 3))

    逻辑：中间价与 40 日均量相关的线性衰减排名，除以 VWAP 与成交量时序相关衰减排名。
    """
    adv40 = sma(volume, 40)
    p1 = rank(decay_linear(correlation((high + low) / 2, adv40, 9), 10))
    corr2 = correlation(ts_rank(vwap, 4), ts_rank(volume, 19), 7).replace([-np.inf, np.inf], 0).fillna(0)
    p2 = rank(decay_linear(corr2, 3))
    return p1 / p2.replace(0, np.nan)


def alpha073(open_, high, low, vwap):
    """
    Alpha#73: max(rank(decay_linear(delta(vwap,5),3)),
                 ts_rank(decay_linear(delta(open*0.147+low*0.853,2)/(open*0.147+low*0.853)*-1,3),17)) * -1

    逻辑：VWAP 5 期差分衰减排名与加权开盘低价变化衰减时序排名的最大值取负。
    """
    p1 = rank(decay_linear(delta(vwap, 5), 3))
    mix = open_ * 0.147155 + low * (1 - 0.147155)
    inner = (delta(mix, 2) / mix.replace(0, np.nan) * -1)
    p2 = ts_rank(decay_linear(inner, 3), 17)
    return pd.DataFrame(
        np.maximum(p1.values, p2.values) * -1,
        index=p1.index, columns=p1.columns,
    )


def alpha074(volume, close, high, vwap):
    """
    Alpha#74: (rank(correlation(close, sma(adv30,37), 15))
              < rank(correlation(rank(high*0.027+vwap*0.973), rank(volume), 11))) * -1

    逻辑：收盘与均量均值的相关系数排名，与加权高价 VWAP 排名和成交量排名的相关系数排名比较。
    """
    adv30 = sma(volume, 30)
    corr1 = correlation(close, sma(adv30, 37), 15).replace([-np.inf, np.inf], 0).fillna(0)
    mix = high * 0.0261661 + vwap * (1 - 0.0261661)
    corr2 = correlation(rank(mix), rank(volume), 11).replace([-np.inf, np.inf], 0).fillna(0)
    return (rank(corr1) < rank(corr2)).astype(float) * -1


def alpha075(volume, vwap, low):
    """
    Alpha#75: rank(correlation(vwap, volume, 4)) < rank(correlation(rank(low), rank(adv50), 12))

    逻辑：VWAP 与成交量的 4 期相关排名，与低价和 50 日均量排名的 12 期相关排名比较。
    """
    adv50 = sma(volume, 50)
    corr1 = correlation(vwap, volume, 4).replace([-np.inf, np.inf], 0).fillna(0)
    corr2 = correlation(rank(low), rank(adv50), 12).replace([-np.inf, np.inf], 0).fillna(0)
    return (rank(corr1) < rank(corr2)).astype(float)


def alpha077(volume, high, low, vwap):
    """
    Alpha#77: min(rank(decay_linear(((high+low)/2+high)-(vwap+high), 20)),
                 rank(decay_linear(correlation((high+low)/2, adv40, 3), 6)))

    逻辑：振幅与 VWAP 偏离的线性衰减排名，与中间价和 40 日均量相关衰减排名取小。
    """
    adv40 = sma(volume, 40)
    p1 = rank(decay_linear(((high + low) / 2 + high) - (vwap + high), 20))
    corr = correlation((high + low) / 2, adv40, 3).replace([-np.inf, np.inf], 0).fillna(0)
    p2 = rank(decay_linear(corr, 6))
    return pd.DataFrame(
        np.minimum(p1.values, p2.values),
        index=p1.index, columns=p1.columns,
    )


def alpha078(volume, low, vwap):
    """
    Alpha#78: rank(correlation(ts_sum(low*0.352+vwap*0.648, 20), ts_sum(adv40,20), 7))
              ** rank(correlation(rank(vwap), rank(volume), 6))

    逻辑：加权低价 VWAP 的 20 期和与均量和的相关系数排名，以量价相关排名为幂。
    """
    adv40 = sma(volume, 40)
    mix = low * 0.352233 + vwap * (1 - 0.352233)
    corr1 = correlation(ts_sum(mix, 20), ts_sum(adv40, 20), 7).replace([-np.inf, np.inf], 0).fillna(0)
    corr2 = correlation(rank(vwap), rank(volume), 6).replace([-np.inf, np.inf], 0).fillna(0)
    return rank(corr1).pow(rank(corr2))


def alpha081(volume, vwap):
    """
    Alpha#81: (rank(log(product(rank(rank(correlation(vwap,ts_sum(adv10,50),8))^4), 15)))
              < rank(correlation(rank(vwap), rank(volume), 5))) * -1

    逻辑：VWAP 与均量和的相关系数四次方排名的滚动乘积对数的排名，与量价排名相关比较。
    """
    adv10 = sma(volume, 10)
    corr_inner = correlation(vwap, ts_sum(adv10, 50), 8).replace([-np.inf, np.inf], 0).fillna(0)
    inner = rank(corr_inner) ** 4
    p1 = rank(np.log(product(inner.replace(0, 1e-10), 15).replace(0, 1e-10)))
    corr2 = correlation(rank(vwap), rank(volume), 5).replace([-np.inf, np.inf], 0).fillna(0)
    return (p1 < rank(corr2)).astype(float) * -1


def alpha083(close, high, low, volume, vwap):
    """
    Alpha#83: (rank(delay((high-low)/(ts_sum(close,5)/5), 2)) * rank(rank(volume)))
              / (((high-low)/(ts_sum(close,5)/5)) / (vwap-close))

    逻辑：滞后振幅均价比的排名乘成交量排名，除以振幅均价比与 VWAP 收盘差之比。
    """
    hl_ratio = (high - low) / (ts_sum(close, 5) / 5)
    denom = hl_ratio / (vwap - close).replace(0, np.nan)
    return (rank(delay(hl_ratio, 2)) * rank(rank(volume))) / denom


def alpha084(close, vwap):
    """
    Alpha#84: SignedPower(ts_rank(vwap - ts_max(vwap,15), 21), delta(close,5))

    逻辑：VWAP 偏离 15 期最大值的时序排名的有符号幂，指数为收盘 5 期差分。
    """
    base = ts_rank(vwap - ts_max(vwap, 15), 21)
    return SignedPower(base, delta(close, 5))


def alpha085(volume, high, low, close, vwap):
    """
    Alpha#85: rank(correlation(high*0.877+close*0.123, adv30, 10))
              ** rank(correlation(ts_rank((high+low)/2,4), ts_rank(volume,10), 7))

    逻辑：加权高价收盘与 30 日均量的相关系数排名，以中间价与成交量时序相关排名为幂。
    """
    adv30 = sma(volume, 30)
    mix = high * 0.876703 + close * (1 - 0.876703)
    corr1 = correlation(mix, adv30, 10).replace([-np.inf, np.inf], 0).fillna(0)
    corr2 = correlation(ts_rank((high + low) / 2, 4), ts_rank(volume, 10), 7).replace([-np.inf, np.inf], 0).fillna(0)
    return rank(corr1).pow(rank(corr2))


# def alpha086(volume, close, open_, vwap):
#     """
#     Alpha#86: (ts_rank(correlation(close, sma(adv20,15), 6), 20)
#               < rank((open+close)-(vwap+open))) * -1
#
#     逻辑：收盘与均量均值的相关系数时序排名，与开盘收盘偏离 VWAP 的排名比较取负。
#     """
#     adv20 = sma(volume, 20)
#     corr = correlation(close, sma(adv20, 15), 6).replace([-np.inf, np.inf], 0).fillna(0)
#     return (ts_rank(corr, 20) < rank((open_ + close) - (vwap + open_))).astype(float) * -1
# 注释：输出为0或空集


def alpha088(volume, open_, high, low, close):
    """
    Alpha#88: min(rank(decay_linear((rank(open)+rank(low))-(rank(high)+rank(close)), 8)),
                 ts_rank(decay_linear(correlation(ts_rank(close,8), ts_rank(adv60,21), 8), 7), 3))

    逻辑：开低高收排名差的线性衰减排名，与收盘均量时序相关衰减时序排名取小。
    """
    adv60 = sma(volume, 60)
    p1 = rank(decay_linear((rank(open_) + rank(low)) - (rank(high) + rank(close)), 8))
    corr = correlation(ts_rank(close, 8), ts_rank(adv60, 21), 8).replace([-np.inf, np.inf], 0).fillna(0)
    p2 = ts_rank(decay_linear(corr, 7), 3)
    return pd.DataFrame(
        np.minimum(p1.values, p2.values),
        index=p1.index, columns=p1.columns,
    )


def alpha092(volume, high, low, open_, close):
    """
    Alpha#92: min(ts_rank(decay_linear(((high+low)/2+close)<(low+open), 15), 19),
                 ts_rank(decay_linear(correlation(rank(low), rank(adv30), 8), 7), 7))

    逻辑：中间价加收盘低于低价加开盘的布尔衰减时序排名，与低价均量相关衰减时序排名取小。
    """
    adv30 = sma(volume, 30)
    cond = (((high + low) / 2 + close) < (low + open_)).astype(float)
    p1 = ts_rank(decay_linear(cond, 15), 19)
    corr = correlation(rank(low), rank(adv30), 8).replace([-np.inf, np.inf], 0).fillna(0)
    p2 = ts_rank(decay_linear(corr, 7), 7)
    return pd.DataFrame(
        np.minimum(p1.values, p2.values),
        index=p1.index, columns=p1.columns,
    )


def alpha094(volume, vwap):
    """
    Alpha#94: (rank(vwap - ts_min(vwap,12)) ** ts_rank(correlation(ts_rank(vwap,20), ts_rank(adv60,4),18), 3)) * -1

    逻辑：VWAP 偏离 12 期低点的排名，以 VWAP 与均量时序相关时序排名为幂，取负。
    """
    adv60 = sma(volume, 60)
    p1 = rank(vwap - ts_min(vwap, 12))
    corr = correlation(ts_rank(vwap, 20), ts_rank(adv60, 4), 18).replace([-np.inf, np.inf], 0).fillna(0)
    p2 = ts_rank(corr, 3)
    return (p1.pow(p2) * -1)


def alpha095(volume, open_, high, low):
    """
    Alpha#95: rank(open - ts_min(open,12))
              < ts_rank(rank(correlation(sma((high+low)/2,19), sma(adv40,19),13))^5, 12)

    逻辑：开盘偏离 12 期低点的排名，与中间价均量相关五次方排名的时序排名比较。
    """
    adv40 = sma(volume, 40)
    corr = correlation(sma((high + low) / 2, 19), sma(adv40, 19), 13).replace([-np.inf, np.inf], 0).fillna(0)
    inner = rank(corr) ** 5
    return (rank(open_ - ts_min(open_, 12)) < ts_rank(inner, 12)).astype(float)


def alpha096(volume, close, vwap):
    """
    Alpha#96: max(ts_rank(decay_linear(correlation(rank(vwap), rank(volume),4),4),8),
                 ts_rank(decay_linear(ts_argmax(correlation(ts_rank(close,7),ts_rank(adv60,4),4),13),14),13)) * -1

    逻辑：量价排名相关衰减时序排名，与收盘均量相关 argmax 衰减时序排名的最大值取负。
    """
    adv60 = sma(volume, 60)
    corr1 = correlation(rank(vwap), rank(volume), 4).replace([-np.inf, np.inf], 0).fillna(0)
    p1 = ts_rank(decay_linear(corr1, 4), 8)
    corr2 = correlation(ts_rank(close, 7), ts_rank(adv60, 4), 4).replace([-np.inf, np.inf], 0).fillna(0)
    argmax_ts = ts_argmax(corr2, 13)
    p2 = ts_rank(decay_linear(argmax_ts, 14), 13)
    return pd.DataFrame(
        np.maximum(p1.values, p2.values) * -1,
        index=p1.index, columns=p1.columns,
    )


def alpha098(volume, vwap, open_):
    """
    Alpha#98: rank(decay_linear(correlation(vwap, sma(adv5,26), 5), 7))
              - rank(decay_linear(ts_rank(ts_argmin(correlation(rank(open), rank(adv15),21), 9), 7), 8))

    逻辑：VWAP 与 5 日均量均值的相关系数衰减排名，减去开盘均量相关 argmin 衰减时序排名。
    """
    adv5 = sma(volume, 5)
    adv15 = sma(volume, 15)
    corr1 = correlation(vwap, sma(adv5, 26), 5).replace([-np.inf, np.inf], 0).fillna(0)
    p1 = rank(decay_linear(corr1, 7))
    corr2 = correlation(rank(open_), rank(adv15), 21).replace([-np.inf, np.inf], 0).fillna(0)
    argmin_ts = ts_argmin(corr2, 9)
    p2 = rank(decay_linear(ts_rank(argmin_ts, 7), 8))
    return p1 - p2


def alpha099(volume, high, low):
    """
    Alpha#99: (rank(correlation(ts_sum((high+low)/2,20), ts_sum(adv60,20), 9))
              < rank(correlation(low, volume, 6))) * -1

    逻辑：中间价 20 期和与均量 20 期和的相关系数排名，与低价成交量相关排名比较取负。
    """
    adv60 = sma(volume, 60)
    corr1 = correlation(ts_sum((high + low) / 2, 20), ts_sum(adv60, 20), 9).replace([-np.inf, np.inf], 0).fillna(0)
    corr2 = correlation(low, volume, 6).replace([-np.inf, np.inf], 0).fillna(0)
    return (rank(corr1) < rank(corr2)).astype(float) * -1


def alpha101(open_, high, low, close):
    """
    Alpha#101: (close - open) / ((high - low) + 0.001)

    逻辑：日内收益（收盘开盘差）除以振幅，衡量日内方向强度。
    """
    return (close - open_) / ((high - low) + 0.001)


# ============================================================
# 因子配置与元数据
# ============================================================

# data_keys 指定该因子所需的数据键，顺序与函数参数顺序一致。
# 可用键：'close', 'open', 'high', 'low', 'volume', 'returns', 'vwap'
FACTOR_CONFIGS = {
    'alpha001': {'func': alpha001, 'data_keys': ['close', 'returns']},
    'alpha002': {'func': alpha002, 'data_keys': ['close', 'open', 'volume']},
    'alpha003': {'func': alpha003, 'data_keys': ['open', 'volume']},
    'alpha004': {'func': alpha004, 'data_keys': ['low']},
    'alpha005': {'func': alpha005, 'data_keys': ['open', 'vwap', 'close']},
    'alpha006': {'func': alpha006, 'data_keys': ['open', 'volume']},
    'alpha007': {'func': alpha007, 'data_keys': ['close', 'volume']},
    'alpha008': {'func': alpha008, 'data_keys': ['open', 'returns']},
    'alpha009': {'func': alpha009, 'data_keys': ['close']},
    'alpha010': {'func': alpha010, 'data_keys': ['close']},
    'alpha011': {'func': alpha011, 'data_keys': ['vwap', 'close', 'volume']},
    'alpha012': {'func': alpha012, 'data_keys': ['volume', 'close']},
    'alpha013': {'func': alpha013, 'data_keys': ['close', 'volume']},
    'alpha014': {'func': alpha014, 'data_keys': ['returns', 'open', 'volume']},
    'alpha015': {'func': alpha015, 'data_keys': ['high', 'volume']},
    'alpha016': {'func': alpha016, 'data_keys': ['high', 'volume']},
    'alpha017': {'func': alpha017, 'data_keys': ['close', 'volume']},
    'alpha018': {'func': alpha018, 'data_keys': ['close', 'open']},
    'alpha019': {'func': alpha019, 'data_keys': ['close', 'returns']},
    'alpha020': {'func': alpha020, 'data_keys': ['open', 'high', 'close', 'low']},
    'alpha021': {'func': alpha021, 'data_keys': ['close', 'volume']},
    'alpha022': {'func': alpha022, 'data_keys': ['high', 'close', 'volume']},
    'alpha023': {'func': alpha023, 'data_keys': ['high']},
    'alpha024': {'func': alpha024, 'data_keys': ['close']},
    'alpha025': {'func': alpha025, 'data_keys': ['returns', 'volume', 'vwap', 'high', 'close']},
    'alpha026': {'func': alpha026, 'data_keys': ['volume', 'high']},
    'alpha027': {'func': alpha027, 'data_keys': ['volume', 'vwap']},
    'alpha028': {'func': alpha028, 'data_keys': ['volume', 'low', 'high', 'close']},
    'alpha029': {'func': alpha029, 'data_keys': ['close', 'returns']},
    'alpha030': {'func': alpha030, 'data_keys': ['close', 'volume']},
    'alpha031': {'func': alpha031, 'data_keys': ['close', 'volume', 'low']},
    'alpha032': {'func': alpha032, 'data_keys': ['close', 'vwap']},
    'alpha033': {'func': alpha033, 'data_keys': ['open', 'close']},
    'alpha034': {'func': alpha034, 'data_keys': ['returns', 'close']},
    'alpha035': {'func': alpha035, 'data_keys': ['volume', 'close', 'high', 'low', 'returns']},
    'alpha036': {'func': alpha036, 'data_keys': ['close', 'open', 'returns', 'volume', 'vwap']},
    'alpha037': {'func': alpha037, 'data_keys': ['open', 'close']},
    'alpha038': {'func': alpha038, 'data_keys': ['open', 'close']},
    'alpha039': {'func': alpha039, 'data_keys': ['close', 'volume', 'returns']},
    'alpha040': {'func': alpha040, 'data_keys': ['high', 'volume']},
    'alpha041': {'func': alpha041, 'data_keys': ['high', 'low', 'vwap']},
    'alpha042': {'func': alpha042, 'data_keys': ['vwap', 'close']},
    'alpha043': {'func': alpha043, 'data_keys': ['volume', 'close']},
    'alpha044': {'func': alpha044, 'data_keys': ['high', 'volume']},
    'alpha045': {'func': alpha045, 'data_keys': ['close', 'volume']},
    'alpha046': {'func': alpha046, 'data_keys': ['close']},
    'alpha047': {'func': alpha047, 'data_keys': ['close', 'volume', 'high', 'vwap']},
    'alpha049': {'func': alpha049, 'data_keys': ['close']},
    'alpha050': {'func': alpha050, 'data_keys': ['volume', 'vwap']},
    'alpha051': {'func': alpha051, 'data_keys': ['close']},
    'alpha052': {'func': alpha052, 'data_keys': ['returns', 'low', 'volume']},
    'alpha053': {'func': alpha053, 'data_keys': ['close', 'high', 'low']},
    'alpha054': {'func': alpha054, 'data_keys': ['close', 'open', 'high', 'low']},
    'alpha055': {'func': alpha055, 'data_keys': ['close', 'high', 'low', 'volume']},
    'alpha057': {'func': alpha057, 'data_keys': ['close', 'vwap']},
    'alpha060': {'func': alpha060, 'data_keys': ['close', 'high', 'low', 'volume']},
    'alpha061': {'func': alpha061, 'data_keys': ['volume', 'vwap']},
    'alpha062': {'func': alpha062, 'data_keys': ['volume', 'open', 'high', 'low', 'vwap']},
    'alpha064': {'func': alpha064, 'data_keys': ['volume', 'open', 'high', 'low', 'vwap']},
    'alpha065': {'func': alpha065, 'data_keys': ['volume', 'open', 'vwap']},
    'alpha066': {'func': alpha066, 'data_keys': ['open', 'high', 'low', 'vwap']},
    # 'alpha068': {'func': alpha068, 'data_keys': ['volume', 'close', 'high', 'low']},  # 输出为0或空集
    'alpha071': {'func': alpha071, 'data_keys': ['volume', 'close', 'open', 'low', 'vwap']},
    'alpha072': {'func': alpha072, 'data_keys': ['volume', 'high', 'low', 'vwap']},
    'alpha073': {'func': alpha073, 'data_keys': ['open', 'high', 'low', 'vwap']},
    'alpha074': {'func': alpha074, 'data_keys': ['volume', 'close', 'high', 'vwap']},
    'alpha075': {'func': alpha075, 'data_keys': ['volume', 'vwap', 'low']},
    'alpha077': {'func': alpha077, 'data_keys': ['volume', 'high', 'low', 'vwap']},
    'alpha078': {'func': alpha078, 'data_keys': ['volume', 'low', 'vwap']},
    'alpha081': {'func': alpha081, 'data_keys': ['volume', 'vwap']},
    'alpha083': {'func': alpha083, 'data_keys': ['close', 'high', 'low', 'volume', 'vwap']},
    'alpha084': {'func': alpha084, 'data_keys': ['close', 'vwap']},
    'alpha085': {'func': alpha085, 'data_keys': ['volume', 'high', 'low', 'close', 'vwap']},
    # 'alpha086': {'func': alpha086, 'data_keys': ['volume', 'close', 'open', 'vwap']},  # 输出为0或空集
    'alpha088': {'func': alpha088, 'data_keys': ['volume', 'open', 'high', 'low', 'close']},
    'alpha092': {'func': alpha092, 'data_keys': ['volume', 'high', 'low', 'open', 'close']},
    'alpha094': {'func': alpha094, 'data_keys': ['volume', 'vwap']},
    'alpha095': {'func': alpha095, 'data_keys': ['volume', 'open', 'high', 'low']},
    'alpha096': {'func': alpha096, 'data_keys': ['volume', 'close', 'vwap']},
    'alpha098': {'func': alpha098, 'data_keys': ['volume', 'vwap', 'open']},
    'alpha099': {'func': alpha099, 'data_keys': ['volume', 'high', 'low']},
    'alpha101': {'func': alpha101, 'data_keys': ['open', 'high', 'low', 'close']},
}

# 保留空字典以兼容原有 pipeline 接口（所有因子已统一至 FACTOR_CONFIGS）
VOLUME_REQUIRED_FACTORS = {}

FACTOR_DESCRIPTIONS = {
    'alpha001': {
        'name': 'Alpha#1 有符号幂 ArgMax 排名',
        'theory': '负收益时用波动率替代价格，取有符号平方后的 5 期 argmax 横截面排名',
        'direction': '正向',
        'holding_period': '短期',
        'category': '动量/波动',
        'evidence': 'WorldQuant 101 Alphas',
    },
    'alpha002': {
        'name': 'Alpha#2 量价相关',
        'theory': '成交量对数差分排名与日内涨幅排名的 6 期滚动相关系数负值',
        'direction': '负向（量价背离做多）',
        'holding_period': '短期',
        'category': '量价',
        'evidence': 'WorldQuant 101 Alphas',
    },
    'alpha003': {
        'name': 'Alpha#3 开盘量相关',
        'theory': '开盘价排名与成交量排名的 10 期滚动相关系数负值',
        'direction': '负向',
        'holding_period': '短期',
        'category': '量价',
        'evidence': 'WorldQuant 101 Alphas',
    },
    'alpha004': {
        'name': 'Alpha#4 低价时序排名',
        'theory': '最低价横截面排名的 9 期时间序列排名负值',
        'direction': '负向',
        'holding_period': '短期',
        'category': '价格',
        'evidence': 'WorldQuant 101 Alphas',
    },
    'alpha005': {
        'name': 'Alpha#5 开盘偏离 VWAP',
        'theory': '开盘价偏离均 VWAP 的排名乘以收盘偏离 VWAP 排名的负绝对值',
        'direction': '混合',
        'holding_period': '短期',
        'category': '价格/VWAP',
        'evidence': 'WorldQuant 101 Alphas',
    },
    'alpha006': {
        'name': 'Alpha#6 开盘量相关（原始）',
        'theory': '开盘价与成交量原始序列的 10 期滚动相关系数负值',
        'direction': '负向',
        'holding_period': '短期',
        'category': '量价',
        'evidence': 'WorldQuant 101 Alphas',
    },
    'alpha007': {
        'name': 'Alpha#7 放量动量',
        'theory': '放量时用价格 7 日变化的时序排名反向，缩量时固定 -1',
        'direction': '负向',
        'holding_period': '短期',
        'category': '动量/成交量',
        'evidence': 'WorldQuant 101 Alphas',
    },
    'alpha008': {
        'name': 'Alpha#8 开盘×收益变化',
        'theory': '开盘价与收益率 5 日和乘积的 10 期变化横截面排名负值',
        'direction': '负向',
        'holding_period': '短期',
        'category': '动量',
        'evidence': 'WorldQuant 101 Alphas',
    },
    'alpha009': {
        'name': 'Alpha#9 趋势跟随/反转',
        'theory': '5 期内持续上涨/下跌则顺势，否则反转',
        'direction': '趋势顺势或反转',
        'holding_period': '极短期',
        'category': '趋势/反转',
        'evidence': 'WorldQuant 101 Alphas',
    },
    'alpha010': {
        'name': 'Alpha#10 趋势跟随/反转（排名）',
        'theory': '4 期趋势判断同 alpha009，但对结果做横截面排名',
        'direction': '趋势顺势或反转',
        'holding_period': '极短期',
        'category': '趋势/反转',
        'evidence': 'WorldQuant 101 Alphas',
    },
    'alpha011': {
        'name': 'Alpha#11 VWAP 极值排名×量变化',
        'theory': 'VWAP 与收盘差值的高低排名之和乘以成交量 3 期差分排名',
        'direction': '混合',
        'holding_period': '短期',
        'category': '量价/VWAP',
        'evidence': 'WorldQuant 101 Alphas',
    },
    'alpha012': {
        'name': 'Alpha#12 量增价跌信号',
        'theory': '成交量变化方向乘以价格变化负值',
        'direction': '量增价跌做多',
        'holding_period': '极短期',
        'category': '量价',
        'evidence': 'WorldQuant 101 Alphas',
    },
    'alpha013': {
        'name': 'Alpha#13 价量协方差排名',
        'theory': '收盘价排名与成交量排名的 5 期协方差的横截面排名负值',
        'direction': '负向',
        'holding_period': '短期',
        'category': '量价',
        'evidence': 'WorldQuant 101 Alphas',
    },
    'alpha014': {
        'name': 'Alpha#14 收益动量×量价相关',
        'theory': '收益率 3 期差分排名负值乘以开盘量相关系数',
        'direction': '混合',
        'holding_period': '短期',
        'category': '动量/量价',
        'evidence': 'WorldQuant 101 Alphas',
    },
    'alpha015': {
        'name': 'Alpha#15 高价量相关排名累积',
        'theory': '最高价排名与成交量排名相关系数的排名的 3 期和负值',
        'direction': '负向',
        'holding_period': '短期',
        'category': '量价',
        'evidence': 'WorldQuant 101 Alphas',
    },
    'alpha016': {
        'name': 'Alpha#16 高价量协方差排名',
        'theory': '最高价排名与成交量排名的 5 期协方差的横截面排名负值',
        'direction': '负向',
        'holding_period': '短期',
        'category': '量价',
        'evidence': 'WorldQuant 101 Alphas',
    },
    'alpha017': {
        'name': 'Alpha#17 三因子动量排名乘积',
        'theory': '价格时序排名、二阶差分排名与相对成交量时序排名的乘积负值',
        'direction': '负向',
        'holding_period': '短期',
        'category': '动量/量价',
        'evidence': 'WorldQuant 101 Alphas',
    },
    'alpha018': {
        'name': 'Alpha#18 日内振幅综合排名',
        'theory': '日内振幅波动性加日内偏差加开收盘相关系数的排名负值',
        'direction': '负向',
        'holding_period': '短期',
        'category': '波动/价格',
        'evidence': 'WorldQuant 101 Alphas',
    },
    'alpha019': {
        'name': 'Alpha#19 中期价格方向反转',
        'theory': '7 日价格方向的反向信号乘以年化收益率累积排名放大因子',
        'direction': '反转',
        'holding_period': '中期',
        'category': '动量/反转',
        'evidence': 'WorldQuant 101 Alphas',
    },
    'alpha020': {
        'name': 'Alpha#20 隔夜跳空反转',
        'theory': '今日开盘相对昨日高低收盘价差值排名的三因子乘积负值',
        'direction': '反转',
        'holding_period': '极短期',
        'category': '价格/隔夜',
        'evidence': 'WorldQuant 101 Alphas',
    },
    'alpha021': {
        'name': 'Alpha#21 均值突破量能信号',
        'theory': '短期均值超过长期均值加波动或成交量低于均量时做空',
        'direction': '混合',
        'holding_period': '短期',
        'category': '价格/成交量',
        'evidence': 'WorldQuant 101 Alphas',
    },
    'alpha022': {
        'name': 'Alpha#22 量价相关变化×波动',
        'theory': '高价量相关系数的 5 期变化乘以收盘价长期波动排名的负值',
        'direction': '负向',
        'holding_period': '短期',
        'category': '量价/波动',
        'evidence': 'WorldQuant 101 Alphas',
    },
    'alpha023': {
        'name': 'Alpha#23 高价突破反转',
        'theory': '高价突破 20 日均线时取高价 2 期差分负值，否则为 0',
        'direction': '反转',
        'holding_period': '短期',
        'category': '价格/突破',
        'evidence': 'WorldQuant 101 Alphas',
    },
    'alpha024': {
        'name': 'Alpha#24 长期趋势条件因子',
        'theory': '长期趋势平缓时偏离低点负值，趋势明显时短期动量反转',
        'direction': '反转/趋势',
        'holding_period': '中长期',
        'category': '趋势/价格',
        'evidence': 'WorldQuant 101 Alphas',
    },
    'alpha025': {
        'name': 'Alpha#25 四因子综合排名',
        'theory': '反向收益率、均量、VWAP 与日内振幅的四因子乘积横截面排名',
        'direction': '正向',
        'holding_period': '短期',
        'category': '动量/量价/VWAP',
        'evidence': 'WorldQuant 101 Alphas',
    },
    'alpha026': {
        'name': 'Alpha#26 量价时序相关最大值',
        'theory': '成交量时序排名与高价时序排名的 5 期相关系数 3 期最大值的负值',
        'direction': '负向',
        'holding_period': '短期',
        'category': '量价',
        'evidence': 'WorldQuant 101 Alphas',
    },
    'alpha027': {
        'name': 'Alpha#27 量价相关分位信号',
        'theory': '量价排名相关系数 2 日均值排名高于中位数时做空，否则做多',
        'direction': '混合',
        'holding_period': '短期',
        'category': '量价/VWAP',
        'evidence': 'WorldQuant 101 Alphas',
    },
    'alpha028': {
        'name': 'Alpha#28 均量低价相关+中间价',
        'theory': '均量与低价相关系数加中间价偏离收盘价的横截面标准化',
        'direction': '混合',
        'holding_period': '短期',
        'category': '量价',
        'evidence': 'WorldQuant 101 Alphas',
    },
    'alpha029': {
        'name': 'Alpha#29 嵌套排名动量+反转时序',
        'theory': '价格差分多层嵌套排名对数变换的 5 期最小值加反向收益率时序排名',
        'direction': '混合',
        'holding_period': '短期',
        'category': '动量/反转',
        'evidence': 'WorldQuant 101 Alphas',
    },
    'alpha030': {
        'name': 'Alpha#30 方向一致性反转',
        'theory': '过去 3 日价格方向一致性排名反向乘以成交量短长期比',
        'direction': '反转',
        'holding_period': '极短期',
        'category': '趋势/成交量',
        'evidence': 'WorldQuant 101 Alphas',
    },
    'alpha031': {
        'name': 'Alpha#31 线性衰减动量+量价方向',
        'theory': '价格 10 期动量线性衰减多层排名 + 短期动量反转 + 量价相关方向',
        'direction': '混合',
        'holding_period': '短期',
        'category': '动量/量价',
        'evidence': 'WorldQuant 101 Alphas',
    },
    'alpha032': {
        'name': 'Alpha#32 均线偏离+VWAP 长期相关',
        'theory': '收盘价偏离 7 日均线加 VWAP 与滞后收盘价长期相关系数，双重横截面标准化',
        'direction': '混合',
        'holding_period': '长期',
        'category': '价格/VWAP',
        'evidence': 'WorldQuant 101 Alphas',
    },
    'alpha033': {
        'name': 'Alpha#33 开收比排名',
        'theory': '开盘价相对收盘价偏差的横截面排名，捕捉隔夜效应',
        'direction': '混合',
        'holding_period': '极短期',
        'category': '价格/隔夜',
        'evidence': 'WorldQuant 101 Alphas',
    },
    'alpha034': {
        'name': 'Alpha#34 波动率比+动量反向',
        'theory': '短期与长期波动率之比排名加价格动量排名的反向组合',
        'direction': '反转',
        'holding_period': '短期',
        'category': '波动/动量',
        'evidence': 'WorldQuant 101 Alphas',
    },
    'alpha035': {
        'name': 'Alpha#35 成交量×振幅×收益时序排名',
        'theory': '成交量时序排名乘以价格振幅时序排名反向，再乘以收益率时序排名反向',
        'direction': '正向',
        'holding_period': '中期',
        'category': '量价/波动',
        'evidence': 'WorldQuant 101 Alphas',
    },
    'alpha036': {
        'name': 'Alpha#36 五因子加权组合',
        'theory': '收盘开盘差与滞后成交量相关 + 日内反转 + 滞后收益时序 + 量价相关 + 长期均价',
        'direction': '混合',
        'holding_period': '中期',
        'category': '综合',
        'evidence': 'WorldQuant 101 Alphas',
    },
    'alpha037': {
        'name': 'Alpha#37 滞后日内偏差长期相关',
        'theory': '昨日日内偏差与收盘价的长期相关系数排名加今日日内偏差排名',
        'direction': '混合',
        'holding_period': '长期',
        'category': '价格',
        'evidence': 'WorldQuant 101 Alphas',
    },
    'alpha038': {
        'name': 'Alpha#38 开盘强势反转',
        'theory': '开盘价时序排名乘以收盘开盘比排名的负值，捕捉开盘强势后反转',
        'direction': '反转',
        'holding_period': '短期',
        'category': '价格/动量',
        'evidence': 'WorldQuant 101 Alphas',
    },
    'alpha039': {
        'name': 'Alpha#39 7 期动量×衰减量排名',
        'theory': '7 期价格动量乘以相对成交量线性衰减排名反向，乘以年化收益趋势放大',
        'direction': '混合',
        'holding_period': '中期',
        'category': '动量/成交量',
        'evidence': 'WorldQuant 101 Alphas',
    },
    'alpha040': {
        'name': 'Alpha#40 高价波动×量价相关',
        'theory': '高价波动率排名乘以高价量相关系数的负值',
        'direction': '负向',
        'holding_period': '短期',
        'category': '波动/量价',
        'evidence': 'WorldQuant 101 Alphas',
    },
    'alpha041': {
        'name': 'Alpha#41 几何均价偏离 VWAP',
        'theory': '高低价几何均值与 VWAP 的差值',
        'direction': '正向',
        'holding_period': '极短期',
        'category': '价格/VWAP',
        'evidence': 'WorldQuant 101 Alphas',
    },
    'alpha042': {
        'name': 'Alpha#42 VWAP-收盘偏离比',
        'theory': 'VWAP 与收盘价差值排名除以二者之和排名',
        'direction': '混合',
        'holding_period': '极短期',
        'category': 'VWAP/价格',
        'evidence': 'WorldQuant 101 Alphas',
    },
    'alpha043': {
        'name': 'Alpha#43 相对成交量×价格反转时序',
        'theory': '相对成交量时序排名乘以 7 期价格反转的时序排名',
        'direction': '反转',
        'holding_period': '短期',
        'category': '量价/动量',
        'evidence': 'WorldQuant 101 Alphas',
    },
    'alpha044': {
        'name': 'Alpha#44 高价量排名相关',
        'theory': '高价与成交量排名的 5 期相关系数的负值',
        'direction': '负向',
        'holding_period': '短期',
        'category': '量价',
        'evidence': 'WorldQuant 101 Alphas',
    },
    'alpha045': {
        'name': 'Alpha#45 三因子量价乘积',
        'theory': '滞后收盘价均线排名、量价短期相关与多期收盘价相关的三因子乘积负值',
        'direction': '负向',
        'holding_period': '短期',
        'category': '量价',
        'evidence': 'WorldQuant 101 Alphas',
    },
    'alpha046': {
        'name': 'Alpha#46 价格加速度信号',
        'theory': '价格加速趋势为正时做空，为负时做多，中性时按日内变化反转',
        'direction': '混合',
        'holding_period': '中期',
        'category': '趋势/反转',
        'evidence': 'WorldQuant 101 Alphas',
    },
    'alpha047': {
        'name': 'Alpha#47 低价成交综合因子',
        'theory': '低价格倒数乘以相对成交量加权高价偏离，减去 VWAP 5 期动量排名',
        'direction': '混合',
        'holding_period': '短期',
        'category': '量价/VWAP',
        'evidence': 'WorldQuant 101 Alphas',
    },
    'alpha049': {
        'name': 'Alpha#49 价格加速度阈值信号',
        'theory': '价格加速度低于 -0.1 时做多，否则按日内变化反转',
        'direction': '混合',
        'holding_period': '中期',
        'category': '趋势/反转',
        'evidence': 'WorldQuant 101 Alphas',
    },
    'alpha050': {
        'name': 'Alpha#50 量价排名相关最大值',
        'theory': '量价排名相关系数横截面排名的 5 期最大值的负值',
        'direction': '负向',
        'holding_period': '短期',
        'category': '量价/VWAP',
        'evidence': 'WorldQuant 101 Alphas',
    },
    'alpha051': {'name': 'Alpha#51 价格加速度阈值（-0.05）', 'theory': '加速度低于-0.05时做多，否则按日内变化反转', 'direction': '混合', 'holding_period': '中期', 'category': '趋势/反转', 'evidence': 'WorldQuant 101 Alphas'},
    'alpha052': {'name': 'Alpha#52 低价动量×收益差×成交量', 'theory': '低价5期最小值5期差分负值乘长短期收益和排名乘成交量时序排名', 'direction': '混合', 'holding_period': '中长期', 'category': '量价/动量', 'evidence': 'WorldQuant 101 Alphas'},
    'alpha053': {'name': 'Alpha#53 日内位置差分', 'theory': '收盘在振幅内相对位置的9期差分负值', 'direction': '负向', 'holding_period': '短期', 'category': '价格', 'evidence': 'WorldQuant 101 Alphas'},
    'alpha054': {'name': 'Alpha#54 低价偏离×开盘五次方', 'theory': '低价偏离收盘与开盘五次方除以振幅与收盘五次方', 'direction': '混合', 'holding_period': '短期', 'category': '价格', 'evidence': 'WorldQuant 101 Alphas'},
    'alpha055': {'name': 'Alpha#55 振幅位置与量相关', 'theory': '收盘在振幅内位置排名与成交量排名的6期相关系数负值', 'direction': '负向', 'holding_period': '短期', 'category': '量价', 'evidence': 'WorldQuant 101 Alphas'},
    'alpha057': {'name': 'Alpha#57 收盘偏离VWAP衰减', 'theory': '收盘偏离VWAP除以收盘argmax排名的线性衰减', 'direction': '负向', 'holding_period': '短期', 'category': 'VWAP/价格', 'evidence': 'WorldQuant 101 Alphas'},
    'alpha060': {'name': 'Alpha#60 日内位置缩放', 'theory': '日内位置乘成交量排名缩放减去收盘argmax排名缩放', 'direction': '负向', 'holding_period': '短期', 'category': '量价/VWAP', 'evidence': 'WorldQuant 101 Alphas'},
    'alpha061': {'name': 'Alpha#61 VWAP偏离与量相关比较', 'theory': 'VWAP偏离16期低点排名与VWAP均量相关系数排名比较', 'direction': '混合', 'holding_period': '中期', 'category': 'VWAP/量价', 'evidence': 'WorldQuant 101 Alphas'},
    'alpha062': {'name': 'Alpha#62 VWAP均量相关与开盘条件', 'theory': 'VWAP与均量相关系数排名与开盘高低价排名条件比较', 'direction': '负向', 'holding_period': '短期', 'category': '量价/VWAP', 'evidence': 'WorldQuant 101 Alphas'},
    'alpha064': {'name': 'Alpha#64 加权开盘低价与均量相关', 'theory': '加权开盘低价与均量相关系数排名与加权中间价VWAP差分比较', 'direction': '负向', 'holding_period': '短期', 'category': '量价/VWAP', 'evidence': 'WorldQuant 101 Alphas'},
    'alpha065': {'name': 'Alpha#65 加权开盘VWAP与均量相关', 'theory': '加权开盘VWAP与均量相关排名与开盘偏离低点排名比较', 'direction': '负向', 'holding_period': '短期', 'category': '量价/VWAP', 'evidence': 'WorldQuant 101 Alphas'},
    'alpha066': {'name': 'Alpha#66 VWAP差分与低价偏离衰减', 'theory': 'VWAP差分线性衰减排名加低价偏离VWAP相对开盘中间价衰减时序排名', 'direction': '负向', 'holding_period': '短期', 'category': 'VWAP/价格', 'evidence': 'WorldQuant 101 Alphas'},
    # 'alpha068': {'name': 'Alpha#68 高价均量相关与加权收盘', 'theory': '高价与15日均量排名的相关系数时序排名与加权收盘低价差分比较', 'direction': '负向', 'holding_period': '短期', 'category': '量价', 'evidence': 'WorldQuant 101 Alphas'},  # 输出为0或空集
    'alpha071': {'name': 'Alpha#71 收盘均量相关与开盘低价VWAP', 'theory': '收盘均量时序相关衰减与开盘低价偏离VWAP衰减时序排名的最大值', 'direction': '混合', 'holding_period': '中期', 'category': '量价/VWAP', 'evidence': 'WorldQuant 101 Alphas'},
    'alpha072': {'name': 'Alpha#72 中间价均量相关比', 'theory': '中间价与40日均量相关衰减排名除以VWAP成交量时序相关衰减排名', 'direction': '混合', 'holding_period': '短期', 'category': '量价/VWAP', 'evidence': 'WorldQuant 101 Alphas'},
    'alpha073': {'name': 'Alpha#73 VWAP差分与加权开盘低价衰减', 'theory': 'VWAP差分衰减排名与加权开盘低价变化衰减时序排名最大值取负', 'direction': '负向', 'holding_period': '短期', 'category': 'VWAP/价格', 'evidence': 'WorldQuant 101 Alphas'},
    'alpha074': {'name': 'Alpha#74 收盘均量与高价VWAP量相关比较', 'theory': '收盘均量相关系数排名与加权高价VWAP成交量相关排名比较', 'direction': '负向', 'holding_period': '短期', 'category': '量价', 'evidence': 'WorldQuant 101 Alphas'},
    'alpha075': {'name': 'Alpha#75 VWAP量与低价均量相关比较', 'theory': 'VWAP成交量相关排名与低价50日均量相关排名比较', 'direction': '混合', 'holding_period': '短期', 'category': '量价/VWAP', 'evidence': 'WorldQuant 101 Alphas'},
    'alpha077': {'name': 'Alpha#77 振幅VWAP与中间价均量衰减', 'theory': '振幅与VWAP偏离衰减排名与中间价40日均量相关衰减排名取小', 'direction': '混合', 'holding_period': '短期', 'category': '量价/VWAP', 'evidence': 'WorldQuant 101 Alphas'},
    'alpha078': {'name': 'Alpha#78 低价VWAP均量相关幂', 'theory': '加权低价VWAP与均量相关系数排名以量价相关排名为幂', 'direction': '混合', 'holding_period': '短期', 'category': '量价/VWAP', 'evidence': 'WorldQuant 101 Alphas'},
    'alpha081': {'name': 'Alpha#81 VWAP均量相关乘积与量价相关', 'theory': 'VWAP均量和相关系数四次方排名滚动乘积对数与量价排名相关比较', 'direction': '负向', 'holding_period': '中期', 'category': '量价/VWAP', 'evidence': 'WorldQuant 101 Alphas'},
    'alpha083': {'name': 'Alpha#83 振幅均价比×成交量除以VWAP偏离', 'theory': '滞后振幅均价比排名乘成交量排名除以振幅均价比与VWAP收盘差之比', 'direction': '混合', 'holding_period': '短期', 'category': '量价/VWAP', 'evidence': 'WorldQuant 101 Alphas'},
    'alpha084': {'name': 'Alpha#84 VWAP偏离有符号幂', 'theory': 'VWAP偏离15期最大值的时序排名以收盘5期差分为指数', 'direction': '混合', 'holding_period': '短期', 'category': 'VWAP/价格', 'evidence': 'WorldQuant 101 Alphas'},
    'alpha085': {'name': 'Alpha#85 高价收盘均量相关幂', 'theory': '加权高价收盘与30日均量相关排名以中间价成交量时序相关为幂', 'direction': '混合', 'holding_period': '短期', 'category': '量价', 'evidence': 'WorldQuant 101 Alphas'},
    # 'alpha086': {'name': 'Alpha#86 收盘均量相关与开盘收盘VWAP', 'theory': '收盘均量均值相关系数时序排名与开盘收盘偏离VWAP排名比较', 'direction': '负向', 'holding_period': '短期', 'category': '量价/VWAP', 'evidence': 'WorldQuant 101 Alphas'},  # 输出为0或空集
    'alpha088': {'name': 'Alpha#88 开低高收排名差衰减', 'theory': '开低高收排名差的线性衰减排名与收盘均量时序相关衰减取小', 'direction': '混合', 'holding_period': '短期', 'category': '价格/量价', 'evidence': 'WorldQuant 101 Alphas'},
    'alpha092': {'name': 'Alpha#92 中间价条件与低价均量衰减', 'theory': '中间价加收盘低于低价加开盘的布尔衰减与低价均量相关衰减取小', 'direction': '混合', 'holding_period': '短期', 'category': '价格/量价', 'evidence': 'WorldQuant 101 Alphas'},
    'alpha094': {'name': 'Alpha#94 VWAP偏离幂', 'theory': 'VWAP偏离12期低点排名以VWAP均量时序相关时序排名为幂', 'direction': '负向', 'holding_period': '短期', 'category': 'VWAP/量价', 'evidence': 'WorldQuant 101 Alphas'},
    'alpha095': {'name': 'Alpha#95 开盘偏离与中间价均量相关', 'theory': '开盘偏离12期低点排名与中间价均量相关五次方时序排名比较', 'direction': '混合', 'holding_period': '短期', 'category': '量价', 'evidence': 'WorldQuant 101 Alphas'},
    'alpha096': {'name': 'Alpha#96 量价相关与收盘均量argmax衰减', 'theory': '量价排名相关衰减与收盘均量相关argmax衰减时序排名最大值取负', 'direction': '负向', 'holding_period': '中期', 'category': '量价/VWAP', 'evidence': 'WorldQuant 101 Alphas'},
    'alpha098': {'name': 'Alpha#98 VWAP均量相关与开盘argmin衰减', 'theory': 'VWAP与5日均量均值相关衰减排名减去开盘均量相关argmin衰减排名', 'direction': '混合', 'holding_period': '短期', 'category': '量价/VWAP', 'evidence': 'WorldQuant 101 Alphas'},
    'alpha099': {'name': 'Alpha#99 中间价均量和相关与低价量相关', 'theory': '中间价20期和与均量20期和相关系数排名与低价成交量相关比较', 'direction': '负向', 'holding_period': '短期', 'category': '量价', 'evidence': 'WorldQuant 101 Alphas'},
    'alpha101': {'name': 'Alpha#101 日内收益除以振幅', 'theory': '日内收益（收盘开盘差）除以振幅，衡量日内方向强度', 'direction': '正向', 'holding_period': '极短期', 'category': '价格', 'evidence': 'WorldQuant 101 Alphas'},
}
