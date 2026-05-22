import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'analysis', 'strategy'))
sys.path.insert(0, os.getcwd())
import pandas as pd
from rebalance.rebalance_operations import _nth_nyse_trading_day

# 验证 nth_nyse_trading_day 的 off-by-one
# P=10, start=4/24
# anchor=4/24 (上周五)
# 期望: 从 4/24 之后数 10 个交易日 = 5/8
# 实际: _nth_nyse_trading_day(4/24, 10) = ?

result = _nth_nyse_trading_day(pd.Timestamp('2026-04-24'), 10)
print("_nth_nyse_trading_day(4/24, 10) =", result.date())

# 手动数：从 4/24 之后第一个交易日是 4/27
# 第1: 4/27, 第2: 4/28, 第3: 4/29, 第4: 4/30, 第5: 5/1
# 第6: 5/4,  第7: 5/5,  第8: 5/6,  第9: 5/7,  第10: 5/8
# 期望: 5/8

# 追踪 rebalance_calendar 语义：
# 条件: n_trading_days >= P where n = count of trading days in (last_selected, d]
# 从 4/17 到 4/27: 交易日 = [4/20,4/21,4/22,4/23,4/24,4/27] = 6 个 < 10 → 4/27 不选
# 从 4/17 到 5/1:  [4/20,4/21,4/22,4/23,4/24,4/27,4/28,4/29,4/30,5/1] = 10 个 >= 10 → 5/1 选
# 从 4/17 到 5/4:  [4/20,4/21,4/22,4/23,4/24,4/27,4/28,4/29,4/30,5/1,5/4] = 11 个 >= 10 → 5/4 选
# 所以 4/27 不是调仓日（6<10），下一个调仓日是 5/1

# 现在验证 _nth_nyse_trading_day 实际含义
# 4/24 之后: 4/27(第1),4/28(第2),...,5/8(第11?)
r9 = _nth_nyse_trading_day(pd.Timestamp('2026-04-24'), 9)
r10 = _nth_nyse_trading_day(pd.Timestamp('2026-04-24'), 10)
r11 = _nth_nyse_trading_day(pd.Timestamp('2026-04-24'), 11)
print("n=9:", r9.date(), " n=10:", r10.date(), " n=11:", r11.date())
print("期望下一个调仓日(P=10)应该是: 5/4 或 5/1")
