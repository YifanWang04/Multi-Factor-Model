import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'analysis', 'strategy'))
sys.path.insert(0, os.getcwd())
import pandas as pd
from rebalance.rebalance_operations import get_rebalance_day_status

# 真实场景：P=10，上一调仓日=4/17
# 4/27 = today
rebalance_dates = [pd.Timestamp('2026-04-17')]
as_of_date = pd.Timestamp('2026-04-27')

scenarios = [
    ('场景A: pipeline上周五跑完, last_factor=4/24', pd.Timestamp('2026-04-24')),
    ('场景B: pipeline今日盘中跑完, last_factor=4/27', pd.Timestamp('2026-04-27')),
]

for label, lfd in scenarios:
    s = get_rebalance_day_status(
        rebalance_dates=rebalance_dates,
        rebalance_period=10,
        as_of_date=as_of_date,
        last_factor_date=lfd,
    )
    cur = s['current_rebalance_date']
    nxt = s['next_rebalance_date']
    today_flag = s['is_rebalance_today']
    print(label)
    print("  is_rebalance_today     = " + str(today_flag))
    print("  current_rb_date        = " + str(cur.date() if cur else None))
    print("  next_rb_date          = " + str(nxt.date() if nxt else None))
    print("")
