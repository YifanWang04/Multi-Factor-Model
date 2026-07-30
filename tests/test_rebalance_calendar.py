import unittest
from types import SimpleNamespace

import pandas as pd

from analysis.strategy.rebalance_calendar import (
    RebalanceAnchorError,
    RebalanceCalendarError,
    get_rebalance_calendar,
    get_future_rebalance_dates,
    periods_per_year_for_calendar,
    resolve_rebalance_anchor,
)
from qqq_config.strategy_profiles import StrategyProfile, get_strategy_profile
from analysis.strategy.strategy_report import StrategyReporter
from analysis.strategy.strategy_backtest import _fixed_week_value_for_period


class RebalanceAnchorTests(unittest.TestCase):
    def setUp(self):
        self.trading_dates = pd.DatetimeIndex(
            pd.to_datetime(
                [
                    "2023-01-03",
                    "2023-01-04",
                    "2023-01-05",
                    "2023-01-06",
                    "2023-01-09",
                    "2023-01-10",
                ]
            )
        )

    def test_weekend_anchor_resolves_to_next_nyse_session(self):
        effective = resolve_rebalance_anchor(
            self.trading_dates,
            self.trading_dates,
            "2023-01-01",
        )

        self.assertEqual(effective, pd.Timestamp("2023-01-03"))

    def test_profile_anchor_controls_rebalance_phase(self):
        from_jan_3 = get_rebalance_calendar(
            self.trading_dates,
            self.trading_dates,
            2,
            anchor_date="2023-01-01",
        )
        from_jan_4 = get_rebalance_calendar(
            self.trading_dates,
            self.trading_dates,
            2,
            anchor_date="2023-01-04",
        )

        self.assertEqual(
            from_jan_3,
            list(pd.to_datetime(["2023-01-03", "2023-01-05", "2023-01-09"])),
        )
        self.assertEqual(
            from_jan_4,
            list(pd.to_datetime(["2023-01-04", "2023-01-06", "2023-01-10"])),
        )

    def test_earlier_data_coverage_does_not_move_profile_phase(self):
        expanded_dates = pd.DatetimeIndex(
            pd.to_datetime(["2022-12-29", "2022-12-30"])
        ).append(self.trading_dates)

        selected = get_rebalance_calendar(
            expanded_dates,
            expanded_dates,
            2,
            anchor_date="2023-01-01",
        )

        self.assertEqual(
            selected,
            list(pd.to_datetime(["2023-01-03", "2023-01-05", "2023-01-09"])),
        )

    def test_missing_requested_session_fails_instead_of_shifting(self):
        with self.assertRaisesRegex(RebalanceAnchorError, "missing from factor calendar"):
            get_rebalance_calendar(
                self.trading_dates,
                self.trading_dates,
                2,
                anchor_date="2022-12-01",
            )

    def test_missing_anchor_in_composite_calendar_requires_regeneration(self):
        sparse_factor_dates = pd.DatetimeIndex(
            pd.to_datetime(["2023-01-03", "2023-01-05", "2023-01-09"])
        )

        with self.assertRaisesRegex(RebalanceAnchorError, "Regenerate"):
            get_rebalance_calendar(
                sparse_factor_dates,
                self.trading_dates,
                2,
                anchor_date="2023-01-04",
            )

    def test_none_anchor_preserves_first_factor_date_behavior(self):
        selected = get_rebalance_calendar(
            self.trading_dates,
            self.trading_dates,
            2,
        )

        self.assertEqual(selected[0], pd.Timestamp("2023-01-03"))


class StrategyProfileDataStartValidationTests(unittest.TestCase):
    @staticmethod
    def _profile(anchor, **kwargs):
        return StrategyProfile(
            name="test",
            factor_indices=(1,),
            composite_sheet="ic_m3_N20",
            strategy_param="equal_5G_Top1_P10d",
            ticker_universe="ORIGINAL_108",
            data_download_start_date=anchor,
            **kwargs,
        )

    def test_accepts_iso_data_download_start_date(self):
        self.assertEqual(
            self._profile("2023-01-01").data_download_start_date,
            "2023-01-01",
        )

    def test_rejects_invalid_data_download_start_date(self):
        with self.assertRaisesRegex(ValueError, "expected YYYY-MM-DD"):
            self._profile("2023/01/01")

    def test_strategy_report_records_download_start_and_natural_rebalance_start(self):
        reporter = StrategyReporter(
            results={
                "strategy": {
                    "params": {"effective_rebalance_anchor": "2023-01-03"}
                }
            },
            all_metrics={},
            config=SimpleNamespace(
                DATA_DOWNLOAD_START_DATE="2023-01-01",
                REBALANCE_ANCHOR_DATE=None,
            ),
        )

        metadata = reporter._build_metadata_df().set_index("Key")["Value"]

        self.assertEqual(metadata["Requested_Data_Download_Start"], "2023-01-01")
        self.assertEqual(metadata["Effective_Rebalance_Start"], "2023-01-03")
        self.assertEqual(metadata["Requested_Rebalance_Anchor"], "None")


class FixedWeekRebalanceTests(unittest.TestCase):
    @staticmethod
    def _nyse_dates(start, end):
        import pandas_market_calendars as mcal

        dates = mcal.get_calendar("NYSE").valid_days(start, end)
        return pd.DatetimeIndex(dates.tz_localize(None).normalize())

    def test_default_mode_keeps_strict_trading_day_periods(self):
        dates = self._nyse_dates("2026-01-02", "2026-04-30")
        for period in (5, 10, 20):
            selected = get_rebalance_calendar(dates, dates, period)
            for previous, current in zip(selected[:-1], selected[1:]):
                actual = int(((dates > previous) & (dates <= current)).sum())
                self.assertEqual(actual, period)

    def test_p10_fixed_wednesday_keeps_two_week_phase(self):
        dates = self._nyse_dates("2026-06-01", "2026-08-01")
        selected = get_rebalance_calendar(
            dates,
            dates,
            10,
            interval_weeks=2,
            weekday=3,
            week_anchor_date="2026-06-24",
        )

        self.assertEqual(
            selected,
            list(
                pd.to_datetime(
                    ["2026-06-10", "2026-06-24", "2026-07-08", "2026-07-22"]
                )
            ),
        )

    def test_data_start_change_does_not_change_fixed_week_phase(self):
        expanded = self._nyse_dates("2026-05-01", "2026-08-01")
        narrowed = expanded[expanded >= pd.Timestamp("2026-06-15")]
        kwargs = {
            "interval_weeks": 2,
            "weekday": 3,
            "week_anchor_date": "2026-06-24",
        }

        expanded_dates = get_rebalance_calendar(expanded, expanded, 10, **kwargs)
        narrowed_dates = get_rebalance_calendar(narrowed, narrowed, 10, **kwargs)

        self.assertEqual(
            narrowed_dates,
            [date for date in expanded_dates if date >= narrowed.min()],
        )

    def test_good_friday_moves_to_previous_nyse_session(self):
        dates = self._nyse_dates("2026-03-01", "2026-05-01")
        selected = get_rebalance_calendar(
            dates,
            dates,
            10,
            interval_weeks=2,
            weekday=5,
            week_anchor_date="2026-03-20",
        )

        self.assertIn(pd.Timestamp("2026-04-02"), selected)
        self.assertNotIn(pd.Timestamp("2026-04-03"), selected)
        previous = pd.Timestamp("2026-03-20")
        adjusted = pd.Timestamp("2026-04-02")
        actual_sessions = int(((dates > previous) & (dates <= adjusted)).sum())
        self.assertEqual(actual_sessions, 9)

    def test_future_dates_use_same_holiday_rule(self):
        future = get_future_rebalance_dates(
            "2026-03-20",
            10,
            2,
            interval_weeks=2,
            weekday=5,
            week_anchor_date="2026-03-20",
        )
        self.assertEqual(
            future,
            list(pd.to_datetime(["2026-04-02", "2026-04-17"])),
        )

    def test_fixed_week_annualization_uses_calendar_frequency(self):
        p10 = periods_per_year_for_calendar(10, 2, 3, "2026-06-24")
        p20 = periods_per_year_for_calendar(20, 4, 3, "2026-06-24")
        self.assertEqual(p10, 26.0)
        self.assertEqual(p20, 13.0)
        self.assertEqual(periods_per_year_for_calendar(10), 25.2)

    def test_missing_fixed_session_fails_instead_of_shifting(self):
        dates = self._nyse_dates("2026-06-01", "2026-07-31")
        factor_dates = dates[dates != pd.Timestamp("2026-07-08")]
        with self.assertRaisesRegex(
            RebalanceCalendarError,
            "missing from factor calendar",
        ):
            get_rebalance_calendar(
                factor_dates,
                dates,
                10,
                interval_weeks=2,
                weekday=3,
                week_anchor_date="2026-06-24",
            )

    def test_profile_accepts_complete_fixed_week_schedule(self):
        profile = StrategyProfileDataStartValidationTests._profile(
            "2023-01-01",
            rebalance_interval_weeks=2,
            rebalance_weekday=3,
            rebalance_week_anchor_date="2026-06-24",
        )
        self.assertTrue(profile.uses_fixed_week_rebalance)

    def test_strategy111_rebalances_every_second_friday(self):
        profile = get_strategy_profile("Strategy111")

        self.assertEqual(profile.rebalance_interval_weeks, 2)
        self.assertEqual(profile.rebalance_weekday, 5)
        self.assertEqual(profile.rebalance_week_anchor_date, "2026-06-26")
        self.assertTrue(profile.uses_fixed_week_rebalance)

    def test_profile_rejects_partial_or_multiple_weekdays(self):
        with self.assertRaisesRegex(ValueError, "must set"):
            StrategyProfileDataStartValidationTests._profile(
                "2023-01-01",
                rebalance_interval_weeks=2,
            )
        with self.assertRaisesRegex(ValueError, "one integer"):
            StrategyProfileDataStartValidationTests._profile(
                "2023-01-01",
                rebalance_interval_weeks=2,
                rebalance_weekday=(1, 3),
                rebalance_week_anchor_date="2026-06-24",
            )

    def test_profile_rejects_period_and_anchor_mismatch(self):
        with self.assertRaisesRegex(ValueError, "implies P15d"):
            StrategyProfileDataStartValidationTests._profile(
                "2023-01-01",
                rebalance_interval_weeks=3,
                rebalance_weekday=3,
                rebalance_week_anchor_date="2026-06-24",
            )
        with self.assertRaisesRegex(ValueError, "not configured weekday"):
            StrategyProfileDataStartValidationTests._profile(
                "2023-01-01",
                rebalance_interval_weeks=2,
                rebalance_weekday=2,
                rebalance_week_anchor_date="2026-06-24",
            )

    def test_strategy_grid_uses_fixed_rule_only_for_profile_period(self):
        config = SimpleNamespace(
            FIXED_WEEK_REBALANCE_PERIOD=10,
            REBALANCE_INTERVAL_WEEKS=2,
            REBALANCE_WEEKDAY=3,
            REBALANCE_WEEK_ANCHOR_DATE="2026-06-24",
        )
        self.assertIsNone(
            _fixed_week_value_for_period(
                config,
                5,
                "REBALANCE_INTERVAL_WEEKS",
            )
        )
        self.assertEqual(
            _fixed_week_value_for_period(
                config,
                10,
                "REBALANCE_INTERVAL_WEEKS",
            ),
            2,
        )


if __name__ == "__main__":
    unittest.main()
