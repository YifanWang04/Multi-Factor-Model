"""
因子库 (factors/factor_library.py)
=====================================
本模块实现 WorldQuant 101 Alpha 中的前 10 个因子，以及供 pipeline 使用的配置与元数据。

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


def sign(df):
    """符号函数：返回 -1、0 或 1"""
    return np.sign(df)


def SignedPower(df, e):
    """有符号幂函数：sign(x) * |x|^e"""
    return np.sign(df) * (np.abs(df) ** e)


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
}
