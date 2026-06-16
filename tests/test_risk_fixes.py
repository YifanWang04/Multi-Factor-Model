import importlib
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class RiskFixTests(unittest.TestCase):
    def test_nyse_future_extrapolation_counts_good_friday(self):
        from analysis.strategy.rebalance.rebalance_operations import _nth_nyse_trading_day

        # 2026-04-03 is Good Friday; NYSE is closed. The 5th NYSE trading day
        # after 2026-03-27 is therefore 2026-04-06.
        got = _nth_nyse_trading_day(pd.Timestamp("2026-03-27"), 5)
        self.assertEqual(got.normalize(), pd.Timestamp("2026-04-06"))

    def test_rebalance_status_uses_same_in_data_counting(self):
        from analysis.strategy.rebalance.rebalance_operations import get_rebalance_day_status

        trading_dates = list(pd.bdate_range("2026-01-02", "2026-01-30"))
        anchor = pd.Timestamp("2026-01-02")
        status = get_rebalance_day_status(
            rebalance_dates=[anchor],
            rebalance_period=10,
            as_of_date=anchor,
            last_factor_date=anchor,
            trading_dates=trading_dates,
        )
        self.assertEqual(status["next_rebalance_date"].normalize(), pd.Timestamp("2026-01-16"))

    def test_m1_uses_full_sample_and_m2_uses_past_only(self):
        from analysis.multi_factor.composite_factor import _univariate_weighted

        dates = pd.to_datetime(["2026-01-02", "2026-01-16", "2026-01-30"])
        stocks = ["AAA", "BBB"]
        factor_dict = {
            "f1": pd.DataFrame([[1, 2], [2, 1], [1, 3]], index=dates, columns=stocks),
            "f2": pd.DataFrame([[3, 1], [1, 4], [2, 2]], index=dates, columns=stocks),
        }
        stats = {
            "f1": {"beta": pd.Series([1.0, 1.0, 1.0], index=dates)},
            "f2": {"beta": pd.Series([0.0, 10.0, 10.0], index=dates)},
        }

        m1 = _univariate_weighted(factor_dict, stats, "beta", dates, method=1)
        m2 = _univariate_weighted(factor_dict, stats, "beta", dates, method=2)

        self.assertTrue(m1.iloc[0].notna().any())
        self.assertTrue(m2.iloc[0].isna().all())
        self.assertFalse(m1.equals(m2))

    def test_offset_price_file_missing_does_not_fallback(self):
        old = os.environ.get("REBALANCE_OFFSET_DAYS")
        os.environ["REBALANCE_OFFSET_DAYS"] = "9999"
        try:
            import data.data_config as dc
            dc = importlib.reload(dc)
            self.assertIn("offset9999d", dc.PRICE_FILE)
            with self.assertRaises(FileNotFoundError):
                dc.require_price_file_exists()
        finally:
            if old is None:
                os.environ.pop("REBALANCE_OFFSET_DAYS", None)
            else:
                os.environ["REBALANCE_OFFSET_DAYS"] = old
            import data.data_config as dc
            importlib.reload(dc)

    def test_selected_factor_files_fail_when_missing(self):
        import analysis.multi_factor.composite_config as cc

        old_dir = cc.FACTOR_PROCESSED_DIR
        old_names = cc.SELECTED_FACTOR_NAMES
        with tempfile.TemporaryDirectory() as tmp:
            open(os.path.join(tmp, "factor_alpha001_processed.xlsx"), "w", encoding="utf-8").close()
            cc.FACTOR_PROCESSED_DIR = tmp
            cc.SELECTED_FACTOR_NAMES = ["alpha001", "alpha002"]
            try:
                with self.assertRaises(FileNotFoundError):
                    cc.get_selected_factor_files()
            finally:
                cc.FACTOR_PROCESSED_DIR = old_dir
                cc.SELECTED_FACTOR_NAMES = old_names

    def test_optimizer_missing_returns_falls_back_without_zero_fill(self):
        from analysis.strategy.portfolio_optimizer import compute_weights

        stocks = ["AAA", "BBB", "CCC"]
        dates = pd.bdate_range("2026-01-01", periods=20)
        hist = pd.DataFrame(
            {
                "AAA": np.random.default_rng(1).normal(0, 0.01, len(dates)),
                "BBB": [np.nan] * len(dates),
                "CCC": [np.nan] * 19 + [0.01],
            },
            index=dates,
        )
        weights = compute_weights(
            method="min_variance",
            stocks=stocks,
            factor_values=pd.Series([1, 2, 3], index=stocks),
            hist_returns=hist,
            max_weight=0.5,
        )
        expected = pd.Series([1 / 3, 1 / 3, 1 / 3], index=stocks)
        pd.testing.assert_series_equal(weights, expected)

    def test_factor_build_defaults_to_legacy_raw_ohlc(self):
        from pipeline.build_factors import load_ohlcv_data

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "prices.xlsx")
            df = pd.DataFrame(
                {
                    "Date": pd.to_datetime(["2026-01-02", "2026-01-05"]),
                    "Open": [10.0, 11.0],
                    "High": [12.0, 13.0],
                    "Low": [9.0, 10.0],
                    "Close": [11.0, 12.0],
                    "Adj Close": [5.5, 6.0],
                    "Adj Open": [5.0, 5.5],
                    "Adj High": [6.0, 6.5],
                    "Adj Low": [4.5, 5.0],
                    "Volume": [100, 200],
                }
            )
            with pd.ExcelWriter(path) as writer:
                df.to_excel(writer, sheet_name="AAPL", index=False)
                df.to_excel(writer, sheet_name="NOT_IN_UNIVERSE", index=False)

            frames = load_ohlcv_data(path)
            self.assertEqual(float(frames["open"].loc[pd.Timestamp("2026-01-02"), "AAPL"]), 10.0)
            self.assertEqual(float(frames["high"].loc[pd.Timestamp("2026-01-02"), "AAPL"]), 12.0)
            self.assertEqual(float(frames["low"].loc[pd.Timestamp("2026-01-02"), "AAPL"]), 9.0)
            self.assertEqual(float(frames["close"].loc[pd.Timestamp("2026-01-02"), "AAPL"]), 5.5)
            self.assertNotIn("NOT_IN_UNIVERSE", frames["close"].columns)

    def test_strategy_price_loader_filters_extra_sheets(self):
        from analysis.strategy.strategy_utils import load_price_data
        from analysis.single_factor.run_multi_factor_test import load_return_data

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "prices.xlsx")
            df = pd.DataFrame(
                {
                    "Date": pd.to_datetime(["2026-01-02", "2026-01-05"]),
                    "Adj Close": [100.0, 101.0],
                    "Return": [np.nan, 0.01],
                }
            )
            with pd.ExcelWriter(path) as writer:
                df.to_excel(writer, sheet_name="AAPL", index=False)
                df.to_excel(writer, sheet_name="NOT_IN_UNIVERSE", index=False)

            prices = load_price_data(path, "Adj Close")
            returns = load_return_data(path, "Return")
            self.assertEqual(list(prices.columns), ["AAPL"])
            self.assertEqual(list(returns.columns), ["AAPL"])

    def test_strategy_and_composite_configs_share_active_profile(self):
        from qqq_config.strategy_profiles import get_active_profile
        import analysis.strategy.strategy_config as sc
        import analysis.multi_factor.composite_config as cc

        profile = get_active_profile()
        self.assertEqual(sc.ACTIVE_STRATEGY_PROFILE, profile.name)
        self.assertEqual(sc.COMPOSITE_FACTOR_SHEET, profile.composite_sheet)
        self.assertEqual(sc.STRATEGY_PARAM, profile.strategy_param)
        self.assertEqual(sc.STRATEGY_SELECTED_FACTOR_INDICES, list(profile.factor_indices))
        self.assertEqual(cc.SELECTED_FACTOR_INDICES, list(profile.factor_indices))
        self.assertEqual(cc.SELECTED_FACTOR_NAMES, list(profile.factor_names))

    def test_strategy_entrypoints_use_active_profile_config(self):
        from qqq_config.strategy_profiles import get_active_profile
        import analysis.strategy.run_rebalance_day as rd
        import analysis.strategy.run_detailed_backtest_report as detailed

        profile = get_active_profile()
        self.assertEqual(rd.ACTIVE_STRATEGY_PROFILE, profile.name)
        self.assertEqual(rd.COMPOSITE_FACTOR_SHEET, profile.composite_sheet)
        self.assertEqual(rd.STRATEGY_PARAM, profile.strategy_param)
        self.assertEqual(rd.SELECTED_FACTOR_INDICES, list(profile.factor_indices))
        self.assertEqual(detailed.COMPOSITE_FACTOR_SHEET, profile.composite_sheet)
        self.assertEqual(detailed.STRATEGY_PARAM, profile.strategy_param)

    def test_legacy_config_strategy_profiles_reexports_authority(self):
        import config.strategy_profiles as legacy
        import qqq_config.strategy_profiles as authority

        self.assertIs(legacy.get_active_profile, authority.get_active_profile)
        self.assertEqual(legacy.ACTIVE_STRATEGY_PROFILE, authority.ACTIVE_STRATEGY_PROFILE)
        self.assertIn("Strategy4", authority.STRATEGY_PROFILES)

    def test_project_paths_offset_and_run_dir_layout(self):
        from qqq_core.paths import ProjectPaths, resolve_output_path

        paths = ProjectPaths.from_env(offset=0)
        self.assertEqual(paths.price_filename, "us_top100_daily_2023_present.xlsx")
        self.assertTrue(str(paths.research_composite_factor_dir).endswith(os.path.join("output", "research", "composite_factor")))
        self.assertTrue(str(paths.strategy_backtest_dir).endswith(os.path.join("output", "strategy", "backtest")))

        offset_paths = ProjectPaths.from_env(offset=7)
        self.assertEqual(offset_paths.price_filename, "us_top100_daily_2023_present_offset7d.xlsx")
        self.assertTrue(str(offset_paths.factor_raw_dir).endswith("factor_raw_offset7d"))

        run_dir = os.path.join(ROOT, "output", "rebalance_runs", "sample")
        self.assertTrue(str(resolve_output_path("price_file", offset=0, run_dir=run_dir)).endswith(
            os.path.join("sample", "data", "us_top100_daily_2023_present.xlsx")
        ))
        self.assertTrue(str(resolve_output_path("rebalance_report_file", offset=0, run_dir=run_dir)).endswith(
            os.path.join("sample", "reports", "rebalance_day_report.xlsx")
        ))

    def test_excel_io_filters_sheets_and_requires_sheet(self):
        from qqq_core.excel_io import read_price_workbook, require_sheet, atomic_excel_writer
        from data.data_config import should_use_price_sheet

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "prices.xlsx")
            df = pd.DataFrame(
                {
                    "Date": pd.to_datetime(["2026-01-02", "2026-01-05"]),
                    "Adj Close": [100.0, 101.0],
                }
            )
            with atomic_excel_writer(path) as writer:
                df.to_excel(writer, sheet_name="AAPL", index=False)
                df.to_excel(writer, sheet_name="NOT_IN_UNIVERSE", index=False)

            prices = read_price_workbook(path, "Adj Close", sheet_filter=should_use_price_sheet)
            self.assertEqual(list(prices.columns), ["AAPL"])
            require_sheet(path, "AAPL")
            with self.assertRaises(ValueError):
                require_sheet(path, "missing_sheet")
            leftovers = [name for name in os.listdir(tmp) if name != "prices.xlsx"]
            self.assertEqual(leftovers, [])

    def test_composite_fallback_does_not_hide_primary_sheet_error(self):
        from analysis.strategy.strategy_utils import load_composite_factor_with_fallback

        with tempfile.TemporaryDirectory() as tmp:
            primary = os.path.join(tmp, "primary.xlsx")
            fallback = os.path.join(tmp, "fallback.xlsx")
            df = pd.DataFrame({"AAPL": [1.0]}, index=[pd.Timestamp("2026-01-02")])
            with pd.ExcelWriter(primary) as writer:
                df.to_excel(writer, sheet_name="wrong_sheet")
            with pd.ExcelWriter(fallback) as writer:
                df.to_excel(writer, sheet_name="ic_m3_N20")

            with self.assertRaises(ValueError):
                load_composite_factor_with_fallback(primary, "ic_m3_N20", fallback)

    def test_offset_factor_processed_dir_does_not_fallback_to_base(self):
        import analysis.multi_factor.composite_config as cc

        old_dir = cc.FACTOR_PROCESSED_DIR
        old_names = cc.SELECTED_FACTOR_NAMES
        with tempfile.TemporaryDirectory() as tmp:
            offset_dir = os.path.join(tmp, "factor_processed_offset5d")
            base_dir = os.path.join(tmp, "factor_processed")
            os.makedirs(base_dir)
            pd.DataFrame({"AAPL": [1.0]}, index=[pd.Timestamp("2026-01-02")]).to_excel(
                os.path.join(base_dir, "factor_alpha001_processed.xlsx")
            )
            cc.FACTOR_PROCESSED_DIR = offset_dir
            cc.SELECTED_FACTOR_NAMES = ["alpha001"]
            try:
                with self.assertRaises(FileNotFoundError):
                    cc.get_selected_factor_files()
            finally:
                cc.FACTOR_PROCESSED_DIR = old_dir
                cc.SELECTED_FACTOR_NAMES = old_names

    def test_strategy_backtest_drops_days_with_no_valid_returns(self):
        from analysis.strategy.strategy_backtest import StrategyBacktester

        cfg = type(
            "Cfg",
            (),
            {
                "GROUP_NUMS": [1],
                "REBALANCE_PERIODS": [1],
                "TARGET_GROUP_RANKS": [1],
                "WEIGHT_METHODS": ["equal"],
                "OPTIMIZATION_LOOKBACK": 5,
                "RISK_FREE_RATE": 0.02,
                "MAX_WEIGHT": 1.0,
                "TRANSACTION_COST": 0.0,
            },
        )
        dates = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
        factor = pd.DataFrame({"AAPL": [1.0, 1.0, 1.0]}, index=dates)
        returns = pd.DataFrame({"AAPL": [np.nan, np.nan, 0.02]}, index=dates)

        result = StrategyBacktester(factor, returns, cfg)._run_single(
            group_num=1,
            target_group=1,
            rebalance_period=1,
            weight_method="equal",
        )
        self.assertNotIn(pd.Timestamp("2026-01-05"), result["daily_returns"].index)
        self.assertIn(pd.Timestamp("2026-01-06"), result["daily_returns"].index)


if __name__ == "__main__":
    unittest.main()
