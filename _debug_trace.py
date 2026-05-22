import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'analysis', 'strategy'))
sys.path.insert(0, os.getcwd())
import pandas as pd
from rebalance.rebalance_operations import get_rebalance_day_status, _nth_nyse_trading_day

REBALANCE_EXTRAPOLATE_MAX_PERIODS = 24
REBALANCE_EXTRAPOLATE_FUTURE_MIN = 12

rebalance_period = 10
rebalance_dates = [pd.Timestamp('2026-04-17')]
as_of_date = pd.Timestamp('2026-04-27')

for td_label, td_list in [
    ('trading_dates 到 4/24 (pipeline上周五跑)',
     ['2026-04-17','2026-04-20','2026-04-21','2026-04-22','2026-04-23','2026-04-24']),
    ('trading_dates 到 4/27 (pipeline今日盘中跑)',
     ['2026-04-17','2026-04-20','2026-04-21','2026-04-22','2026-04-23','2026-04-24','2026-04-27']),
]:
    print("=" * 60)
    print("CASE:", td_label)

    for lfd_label, lfd in [
        ('last_factor=4/24 (上周五)', pd.Timestamp('2026-04-24')),
        ('last_factor=4/27 (今日)',   pd.Timestamp('2026-04-27')),
    ]:
        rebalance_dates_sorted = sorted(rebalance_dates)
        sorted_td = sorted([pd.Timestamp(d) for d in td_list])
        anchor = max(rebalance_dates_sorted[-1], lfd)
        effective_as_of = as_of_date if as_of_date <= lfd else lfd

        print(f"\n  [{lfd_label}]")
        print(f"    anchor={anchor.date()}, effective_as_of={effective_as_of.date()}")

        extrapolated = []
        current_date = anchor
        for _ in range(REBALANCE_EXTRAPOLATE_MAX_PERIODS):
            if sorted_td:
                try:
                    idx = next(ii for ii, x in enumerate(sorted_td) if x > current_date)
                except StopIteration:
                    idx = len(sorted_td)
                next_idx = idx + rebalance_period - 1
                if next_idx < len(sorted_td):
                    current_date = sorted_td[next_idx]
                else:
                    current_date = _nth_nyse_trading_day(current_date, rebalance_period)
            else:
                current_date = _nth_nyse_trading_day(current_date, rebalance_period)
            extrapolated.append(current_date)
            future_so_far = [x for x in extrapolated if x > effective_as_of]
            if len(future_so_far) >= REBALANCE_EXTRAPOLATE_FUTURE_MIN:
                break

        all_dates = sorted(set(rebalance_dates_sorted) | set(extrapolated))
        past_all = [x for x in all_dates if x <= effective_as_of]
        future_all = [x for x in all_dates if x > effective_as_of]
        current_rebalance_date = past_all[-1] if past_all else None
        next_rebalance_date = future_all[0] if future_all else None

        is_rebalance_today = (
            current_rebalance_date is not None
            and current_rebalance_date.date() == as_of_date.date()
            and as_of_date <= lfd
        )

        print(f"    extrapolated (前6): {[str(d.date()) for d in extrapolated[:6]]}")
        print(f"    past_all:   {[str(d.date()) for d in past_all]}")
        print(f"    future_all: {[str(d.date()) for d in future_all[:3]]}")
        print(f"    current_rb_date = {current_rebalance_date.date() if current_rebalance_date else None}")
        print(f"    next_rb_date   = {next_rebalance_date.date() if next_rebalance_date else None}")
        print(f"    is_rebalance_today = {is_rebalance_today}")
