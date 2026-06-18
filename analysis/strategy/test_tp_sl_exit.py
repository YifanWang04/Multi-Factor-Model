import unittest
import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd

from tp_sl_exit import (
    EXIT_FORCED_REBALANCE,
    EXIT_STOP_LOSS,
    EXIT_TAKE_PROFIT,
    EXIT_DYNAMIC_TP_SL,
    find_exit_event,
    thresholds_for_day,
)

sys.modules.setdefault("yfinance", types.ModuleType("yfinance"))

if "pandas_market_calendars" not in sys.modules:
    fake_pmc = types.ModuleType("pandas_market_calendars")

    class _FakeNYSECalendar:
        def valid_days(self, start_date, end_date):
            return pd.bdate_range(start_date, end_date, tz="UTC")

    fake_pmc.get_calendar = lambda name: _FakeNYSECalendar()
    sys.modules["pandas_market_calendars"] = fake_pmc

_REPORT_PATH = Path(__file__).resolve().parent / "rebalance" / "rebalance_report.py"
_REPORT_SPEC = importlib.util.spec_from_file_location("rebalance_report_for_test", _REPORT_PATH)
_rebalance_report = importlib.util.module_from_spec(_REPORT_SPEC)
assert _REPORT_SPEC is not None and _REPORT_SPEC.loader is not None
_REPORT_SPEC.loader.exec_module(_rebalance_report)

build_tp_sl_action_checklist = _rebalance_report.build_tp_sl_action_checklist
build_tp_sl_schedule = _rebalance_report.build_tp_sl_schedule


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

    def test_build_tp_sl_schedule_prices_and_excludes_rebalance_day(self):
        current_ops = pd.DataFrame(
            [
                {
                    "Action": "Buy",
                    "Symbol": "AAA",
                    "Rebalance_Date": pd.Timestamp("2026-01-02"),
                    "Next_Rebalance_Date": pd.Timestamp("2026-01-09"),
                    "Buy_Price_Close": 100.0,
                    "Weight": 0.25,
                    "Shares": 0.0025,
                }
            ]
        )

        schedule = build_tp_sl_schedule(
            current_ops=current_ops,
            as_of_date=pd.Timestamp("2026-01-02"),
            exit_policy=EXIT_DYNAMIC_TP_SL,
            rebalance_period=5,
            tp_base=0.10,
            sl_base=0.05,
            probability=1.0,
        )

        self.assertEqual(schedule["TD"].tolist(), [1, 2, 3, 4])
        self.assertEqual(schedule["Schedule_Date"].max(), pd.Timestamp("2026-01-08"))
        self.assertNotIn(5, schedule["TD"].tolist())

        first = schedule.iloc[0]
        self.assertAlmostEqual(first["TP_Return_Threshold"], 0.08)
        self.assertAlmostEqual(first["SL_Return_Threshold"], 0.04)
        self.assertAlmostEqual(first["TP_Price"], 108.0)
        self.assertAlmostEqual(first["SL_Price"], 96.0)

    def test_build_tp_sl_schedule_empty_for_fixed_rebalance(self):
        current_ops = pd.DataFrame(
            [
                {
                    "Action": "Buy",
                    "Symbol": "AAA",
                    "Rebalance_Date": pd.Timestamp("2026-01-02"),
                    "Next_Rebalance_Date": pd.Timestamp("2026-01-09"),
                    "Buy_Price_Close": 100.0,
                }
            ]
        )

        schedule = build_tp_sl_schedule(
            current_ops=current_ops,
            as_of_date=pd.Timestamp("2026-01-02"),
            exit_policy="fixed_rebalance",
            rebalance_period=5,
            tp_base=0.10,
            sl_base=0.05,
            probability=1.0,
        )

        self.assertTrue(schedule.empty)

    def test_action_checklist_keeps_nearest_future_dates(self):
        schedule = pd.DataFrame(
            {
                "Schedule_Date": pd.to_datetime(
                    ["2026-01-05", "2026-01-06", "2026-01-07"]
                ),
                "Symbol": ["AAA", "AAA", "AAA"],
                "TP_Price": [108.0, 106.0, 104.0],
                "SL_Price": [96.0, 97.0, 98.0],
                "Buy_Price_Close": [100.0, 100.0, 100.0],
                "Weight": [0.25, 0.25, 0.25],
            }
        )

        checklist = build_tp_sl_action_checklist(
            schedule,
            as_of_date=pd.Timestamp("2026-01-06"),
            lookahead_dates=1,
        )

        self.assertEqual(checklist["Date"].tolist(), [pd.Timestamp("2026-01-06")])
        self.assertEqual(checklist["Suggested_Check"].iloc[0], "Set price alert / Check near close")


if __name__ == "__main__":
    unittest.main()
