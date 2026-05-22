import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'analysis', 'strategy'))
sys.path.insert(0, os.getcwd())
import pandas as pd
from rebalance.rebalance_operations import get_rebalance_day_status

rebalance_dates = [pd.Timestamp('2026-04-17')]
as_of_date = pd.Timestamp('2026-04-27')

scenarios = [
    ('A: pipeline上周五(4/24)就跑完  last_factor=4/24', pd.Timestamp('2026-04-24'), '盘中（无今日数据）'),
    ('B: pipeline今日盘中(4/27)跑完  last_factor=4/27', pd.Timestamp('2026-04-27'), '盘中（有今日数据但未收盘）'),
    ('C: pipeline今日收盘(4/27)跑完  last_factor=4/27', pd.Timestamp('2026-04-27'), '收盘后'),
]

for label, lfd, note in scenarios:
    s = get_rebalance_day_status(
        rebalance_dates=rebalance_dates,
        rebalance_period=10,
        as_of_date=as_of_date,
        last_factor_date=lfd,
    )
    cur = s['current_rebalance_date']
    nxt = s['next_rebalance_date']
    print(f"{label}")
    print(f"  因子数据最后日期 = {lfd.date()} | 市场状态 = {note}")
    print(f"  is_rebalance_today      = {s['is_rebalance_today']}")
    print(f"  current_rebalance_date  = {cur.date() if cur else None}")
    print(f"  next_rebalance_date     = {nxt.date() if nxt else None}")
    print()
