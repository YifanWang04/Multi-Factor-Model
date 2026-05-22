import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'analysis', 'strategy'))
sys.path.insert(0, os.getcwd())
import pandas as pd
import glob

# 使用最新跑的数据（修复后）
run_dir = os.path.join(os.getcwd(), 'output', 'rebalance_day_2026-04-27_014038')
cf_files = glob.glob(os.path.join(run_dir, 'composite_factor_reports', 'composite_factors*.xlsx'))
price_files = glob.glob(os.path.join(run_dir, 'data', '*.xlsx'))
cf_file = cf_files[0]
price_file = price_files[0]

factor_df = pd.read_excel(cf_file, sheet_name='ic_m3_N20', index_col=0, parse_dates=True)
factor_df.index = pd.to_datetime(factor_df.index)
print("因子数据日期范围:", factor_df.index[0].date(), "~", factor_df.index[-1].date())
print("最后因子日期 (last_factor_date):", factor_df.index[-1].date())

from strategy_backtest import _select_rebalance_dates
from rebalance.rebalance_operations import get_rebalance_day_status

ret_xl = pd.ExcelFile(price_file)
sheet = ret_xl.sheet_names[0]
ret_df = pd.read_excel(price_file, sheet_name=sheet, index_col=0, parse_dates=True)
ret_df.index = pd.to_datetime(ret_df.index)
ret_df.sort_index(inplace=True)
print("价格数据日期范围:", ret_df.index[0].date(), "~", ret_df.index[-1].date())

rb_dates = _select_rebalance_dates(factor_df.index, ret_df.index, 10)
print("调仓日数量:", len(rb_dates))
print("最后5个调仓日:", [d.date() for d in rb_dates[-5:]])
print("4/27 在调仓日历中:", pd.Timestamp('2026-04-27') in rb_dates)

# 测试两个场景
for label, lfd in [
    ('场景A: pipeline上周五跑完, last_factor=4/24', pd.Timestamp('2026-04-24')),
    ('场景B: pipeline今日盘中跑完, last_factor=今日', pd.Timestamp('2026-04-27')),
]:
    as_of_date = pd.Timestamp('2026-04-27')
    s = get_rebalance_day_status(
        rebalance_dates=rb_dates,
        rebalance_period=10,
        as_of_date=as_of_date,
        last_factor_date=lfd,
    )
    cur = s['current_rebalance_date']
    nxt = s['next_rebalance_date']
    print("")
    print(label)
    print("  is_rebalance_today = " + str(s['is_rebalance_today']))
    print("  has_factor_data    = " + str(s['has_factor_data']))
    print("  current_rb_date   = " + str(cur.date() if cur else None))
    print("  next_rb_date     = " + str(nxt.date() if nxt else None))
