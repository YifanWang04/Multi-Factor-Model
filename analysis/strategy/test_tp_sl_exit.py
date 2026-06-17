import unittest

import pandas as pd

from tp_sl_exit import (
    EXIT_FORCED_REBALANCE,
    EXIT_STOP_LOSS,
    EXIT_TAKE_PROFIT,
    find_exit_event,
    thresholds_for_day,
)


class DynamicTpSlExitTests(unittest.TestCase):
    def test_thresholds_decay_for_ten_day_period(self):
        tp_1, sl_1 = thresholds_for_day(0.08, 0.05, 10, 1, 1.0)
        tp_5, sl_5 = thresholds_for_day(0.08, 0.05, 10, 5, 1.0)
        tp_9, sl_9 = thresholds_for_day(0.08, 0.05, 10, 9, 1.0)

        self.assertAlmostEqual(tp_1, 0.072)
        self.assertAlmostEqual(tp_5, 0.040)
        self.assertAlmostEqual(tp_9, 0.008)
        self.assertAlmostEqual(sl_1, 0.045)
        self.assertAlmostEqual(sl_5, 0.025)
        self.assertAlmostEqual(sl_9, 0.005)

    def test_take_profit_exit(self):
        prices = pd.Series(
            [102.0, 105.0, 107.0],
            index=pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]),
        )
        event = find_exit_event("AAA", 100.0, prices, 10, 0.08, 0.05, 1.0)

        self.assertIsNotNone(event)
        self.assertEqual(event.exit_reason, EXIT_TAKE_PROFIT)
        self.assertEqual(event.exit_date, pd.Timestamp("2026-01-06"))

    def test_stop_loss_exit(self):
        prices = pd.Series(
            [99.0, 96.0],
            index=pd.to_datetime(["2026-01-02", "2026-01-05"]),
        )
        event = find_exit_event("AAA", 100.0, prices, 10, 0.08, 0.05, 1.0)

        self.assertIsNotNone(event)
        self.assertEqual(event.exit_reason, EXIT_STOP_LOSS)
        self.assertEqual(event.exit_date, pd.Timestamp("2026-01-05"))

    def test_forced_exit_when_no_threshold_hit(self):
        prices = pd.Series(
            [100.2, 100.4, 100.6],
            index=pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]),
        )
        event = find_exit_event(
            "AAA",
            100.0,
            prices,
            10,
            0.08,
            0.05,
            1.0,
            force_exit_date=pd.Timestamp("2026-01-06"),
            force_exit_price=100.6,
        )

        self.assertIsNotNone(event)
        self.assertEqual(event.exit_reason, EXIT_FORCED_REBALANCE)
        self.assertEqual(event.exit_price, 100.6)


if __name__ == "__main__":
    unittest.main()
