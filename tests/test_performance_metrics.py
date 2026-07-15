import math
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from qqq_core.performance_metrics import (
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    max_drawdown,
    max_drawdown_info,
    loss_duration_stats,
    performance_summary,
    profit_loss_ratio,
    sharpe_ratio,
    total_return,
    win_rate,
    worst_period_drawdown,
)


class PerformanceMetricHelperTests(unittest.TestCase):
    def test_annual_return_includes_first_return_period(self):
        returns = pd.Series(
            [0.10, 0.20],
            index=pd.to_datetime(["2026-01-02", "2026-01-05"]),
        )

        self.assertAlmostEqual(total_return(returns), 0.32)
        self.assertAlmostEqual(annualized_return(returns, periods_per_year=2), 0.32)

    def test_max_drawdown_uses_initial_wealth_anchor(self):
        returns = pd.Series(
            [-0.10, 0.00],
            index=pd.to_datetime(["2026-01-02", "2026-01-05"]),
        )

        self.assertAlmostEqual(max_drawdown(returns), -0.10)
        dd, start, end = max_drawdown_info(returns)
        self.assertAlmostEqual(dd, -0.10)
        self.assertEqual(start, pd.Timestamp("2026-01-02"))
        self.assertEqual(end, pd.Timestamp("2026-01-02"))

    def test_sharpe_and_calmar_use_shared_annual_return_and_vol(self):
        returns = pd.Series(
            [0.10, -0.05],
            index=pd.to_datetime(["2026-01-02", "2026-01-05"]),
        )
        expected_ann = (1.10 * 0.95) ** (2 / 2) - 1.0
        expected_vol = returns.std(ddof=1) * math.sqrt(2)

        self.assertAlmostEqual(annualized_return(returns, 2), expected_ann)
        self.assertAlmostEqual(annualized_volatility(returns, 2), expected_vol)
        self.assertAlmostEqual(sharpe_ratio(returns, rf=0.01, periods_per_year=2), (expected_ann - 0.01) / expected_vol)
        self.assertAlmostEqual(calmar_ratio(returns, periods_per_year=2), expected_ann / 0.05)

    def test_empty_single_point_zero_vol_and_terminal_wealth_edges(self):
        self.assertTrue(np.isnan(annualized_return(pd.Series(dtype=float), 252)))
        self.assertTrue(np.isnan(max_drawdown(pd.Series(dtype=float))))

        one = pd.Series([0.10], index=[pd.Timestamp("2026-01-02")])
        self.assertAlmostEqual(annualized_return(one, 1), 0.10)
        self.assertTrue(np.isnan(annualized_volatility(one, 1)))
        self.assertTrue(np.isnan(sharpe_ratio(one, rf=0.0, periods_per_year=1)))

        flat = pd.Series([0.01, 0.01], index=pd.to_datetime(["2026-01-02", "2026-01-05"]))
        self.assertTrue(np.isnan(sharpe_ratio(flat, rf=0.0, periods_per_year=2)))

        wiped_out = pd.Series([-1.0], index=[pd.Timestamp("2026-01-02")])
        self.assertTrue(np.isnan(annualized_return(wiped_out, 252)))

    def test_win_rate_and_profit_loss_ratio_edges(self):
        all_positive = pd.Series([0.01, 0.02])
        all_negative = pd.Series([-0.01, -0.03])
        mixed = pd.Series([0.03, -0.01, 0.00])

        self.assertEqual(win_rate(all_positive), 1.0)
        self.assertTrue(np.isinf(profit_loss_ratio(all_positive)))
        self.assertEqual(win_rate(all_negative), 0.0)
        self.assertEqual(profit_loss_ratio(all_negative), 0.0)
        self.assertAlmostEqual(win_rate(mixed), 1 / 3)
        self.assertAlmostEqual(profit_loss_ratio(mixed), 3.0)

    def test_worst_period_drawdown_includes_first_holding_period(self):
        daily_returns = pd.Series(
            [-0.10, 0.02, -0.01],
            index=pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"]),
        )
        rebalance_returns = pd.Series(
            [-0.082, -0.01],
            index=pd.to_datetime(["2026-01-02", "2026-01-06"]),
        )

        dd, start, end = worst_period_drawdown(daily_returns, rebalance_returns)

        self.assertAlmostEqual(dd, -0.10)
        self.assertEqual(start, pd.Timestamp("2026-01-02"))
        self.assertEqual(end, pd.Timestamp("2026-01-05"))

    def test_loss_duration_uses_peak_to_recovery_periods(self):
        returns = pd.Series(
            [-0.10, 1 / 9, 0.10, -0.05, 0.06],
            index=pd.bdate_range("2026-01-02", periods=5),
        )

        max_duration, avg_duration = loss_duration_stats(returns)

        self.assertEqual(max_duration, 2.0)
        self.assertEqual(avg_duration, 2.0)

    def test_loss_duration_includes_unrecovered_episode(self):
        returns = pd.Series([0.10, -0.05, 0.00, 0.01])

        self.assertEqual(loss_duration_stats(returns), (3.0, 3.0))
        self.assertEqual(loss_duration_stats(pd.Series([0.01, 0.00])), (0.0, 0.0))
        empty_max, empty_avg = loss_duration_stats(pd.Series(dtype=float))
        self.assertTrue(np.isnan(empty_max))
        self.assertTrue(np.isnan(empty_avg))

    def test_performance_summary_matches_individual_helpers(self):
        returns = pd.Series(
            [0.04, -0.02, 0.01],
            index=pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]),
        )
        summary = performance_summary(returns, rf=0.01, periods_per_year=3)

        self.assertAlmostEqual(summary["total_return"], total_return(returns))
        self.assertAlmostEqual(summary["annual_return"], annualized_return(returns, 3))
        self.assertAlmostEqual(summary["annual_vol"], annualized_volatility(returns, 3))
        self.assertAlmostEqual(summary["sharpe"], sharpe_ratio(returns, 0.01, 3))
        self.assertAlmostEqual(summary["max_drawdown"], max_drawdown(returns))
        self.assertEqual(
            (summary["max_loss_duration"], summary["avg_loss_duration"]),
            loss_duration_stats(returns),
        )
        self.assertAlmostEqual(summary["calmar"], calmar_ratio(returns, 3))
        self.assertAlmostEqual(summary["win_rate"], win_rate(returns))
        self.assertAlmostEqual(summary["profit_loss_ratio"], profit_loss_ratio(returns))


if __name__ == "__main__":
    unittest.main()
