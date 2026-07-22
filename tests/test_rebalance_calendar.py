import unittest
from types import SimpleNamespace

import pandas as pd

from analysis.strategy.rebalance_calendar import (
    RebalanceAnchorError,
    get_rebalance_calendar,
    resolve_rebalance_anchor,
)
from qqq_config.strategy_profiles import StrategyProfile
from analysis.strategy.strategy_report import StrategyReporter


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
    def _profile(anchor):
        return StrategyProfile(
            name="test",
            factor_indices=(1,),
            composite_sheet="ic_m3_N20",
            strategy_param="equal_5G_Top1_P10d",
            ticker_universe="ORIGINAL_108",
            data_download_start_date=anchor,
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


if __name__ == "__main__":
    unittest.main()
