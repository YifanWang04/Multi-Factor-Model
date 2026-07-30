import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

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

    def test_mad_winsorize_preserves_zero_mad_rows(self):
        if "pandas_market_calendars" not in sys.modules:
            import types

            fake_mcal = types.ModuleType("pandas_market_calendars")
            fake_mcal.get_calendar = lambda name: None
            sys.modules["pandas_market_calendars"] = fake_mcal

        from pipeline.data_process import mad_winsorize, process_factor_df

        dates = pd.to_datetime(["2026-01-02", "2026-01-05"])
        raw = pd.DataFrame(
            {
                "AAA": [1.0, 1.0],
                "BBB": [1.0, 1.0],
                "CCC": [1.0, 10.0],
                "DDD": [1.0, 1.0],
            },
            index=dates,
        )

        winsorized = mad_winsorize(raw)
        pd.testing.assert_frame_equal(winsorized, raw)

        processed = process_factor_df(raw)
        self.assertAlmostEqual(float(processed.loc[dates[1], "CCC"]), 1.5)
        self.assertAlmostEqual(float(processed.loc[dates[1], "AAA"]), -0.5)

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
        self.assertEqual(cc.REBALANCE_PERIOD, profile.rebalance_period)

    def test_strategy_rejects_composite_calendar_coarser_than_strategy_period(self):
        from analysis.strategy.strategy_backtest import (
            CompositeCalendarError,
            _select_rebalance_dates,
        )

        ret_index = pd.bdate_range("2026-01-01", periods=25)
        p10_factor_index = pd.DatetimeIndex([ret_index[0], ret_index[10], ret_index[20]])

        with self.assertRaisesRegex(CompositeCalendarError, "coarser than the requested strategy"):
            _select_rebalance_dates(p10_factor_index, ret_index, 5)

        selected = _select_rebalance_dates(p10_factor_index, ret_index, 10)
        self.assertEqual(selected, list(p10_factor_index))

    def test_strategy_grid_stops_on_coarse_composite_calendar(self):
        from analysis.strategy.strategy_backtest import (
            CompositeCalendarError,
            StrategyBacktester,
        )

        cfg = type(
            "Cfg",
            (),
            {
                "GROUP_NUMS": [1],
                "REBALANCE_PERIODS": [5],
                "TARGET_GROUP_RANKS": [1],
                "WEIGHT_METHODS": ["equal"],
                "EXIT_POLICY_GRID": ["fixed_rebalance"],
                "OPTIMIZATION_LOOKBACK": 5,
                "RISK_FREE_RATE": 0.02,
                "MAX_WEIGHT": 1.0,
                "TRANSACTION_COST": 0.0,
            },
        )
        ret_index = pd.bdate_range("2026-01-01", periods=25)
        factor_index = pd.DatetimeIndex([ret_index[0], ret_index[10], ret_index[20]])
        factor = pd.DataFrame({"AAPL": [1.0, 1.0, 1.0]}, index=factor_index)
        returns = pd.DataFrame({"AAPL": 0.01}, index=ret_index)

        with self.assertRaises(CompositeCalendarError):
            StrategyBacktester(factor, returns, cfg).run_grid()

    def test_strategy_grid_uses_matching_composite_for_each_period(self):
        from analysis.strategy.strategy_backtest import StrategyBacktester

        cfg = type(
            "Cfg",
            (),
            {
                "GROUP_NUMS": [1],
                "REBALANCE_PERIODS": [5, 10],
                "TARGET_GROUP_RANKS": [1],
                "WEIGHT_METHODS": ["equal"],
                "EXIT_POLICY_GRID": ["fixed_rebalance"],
                "OPTIMIZATION_LOOKBACK": 5,
                "RISK_FREE_RATE": 0.02,
                "MAX_WEIGHT": 1.0,
                "TRANSACTION_COST": 0.0,
            },
        )
        ret_index = pd.bdate_range("2026-01-01", periods=25)
        returns = pd.DataFrame({"AAPL": 0.01}, index=ret_index)
        p5_index = pd.DatetimeIndex([ret_index[0], ret_index[5], ret_index[10], ret_index[15], ret_index[20]])
        p10_index = pd.DatetimeIndex([ret_index[0], ret_index[10], ret_index[20]])
        p5_factor = pd.DataFrame({"AAPL": 1.0}, index=p5_index)
        p10_factor = pd.DataFrame({"AAPL": 1.0}, index=p10_index)

        results = StrategyBacktester(
            p5_factor,
            returns,
            cfg,
            factor_dfs_by_period={5: p5_factor, 10: p10_factor},
        ).run_grid()

        self.assertEqual(
            len(results["equal_1G_Top1_P5d__fixed_rebalance"]["rebalance_returns"]),
            5,
        )
        self.assertEqual(
            len(results["equal_1G_Top1_P10d__fixed_rebalance"]["rebalance_returns"]),
            3,
        )

    def test_run_strategy_loads_matching_composite_for_each_period(self):
        import types

        if "pandas_market_calendars" not in sys.modules:
            fake_mcal = types.ModuleType("pandas_market_calendars")
            fake_mcal.get_calendar = lambda name: None
            sys.modules["pandas_market_calendars"] = fake_mcal
        if "scipy" not in sys.modules:
            fake_scipy = types.ModuleType("scipy")
            fake_optimize = types.ModuleType("scipy.optimize")
            fake_stats = types.ModuleType("scipy.stats")
            fake_optimize.minimize = lambda *args, **kwargs: None
            fake_stats.spearmanr = lambda *args, **kwargs: (np.nan, np.nan)
            fake_stats.skew = lambda *args, **kwargs: np.nan
            fake_stats.kurtosis = lambda *args, **kwargs: np.nan
            fake_stats.ttest_1samp = lambda *args, **kwargs: (np.nan, np.nan)
            fake_scipy.optimize = fake_optimize
            fake_scipy.stats = fake_stats
            sys.modules["scipy"] = fake_scipy
            sys.modules["scipy.optimize"] = fake_optimize
            sys.modules["scipy.stats"] = fake_stats

        import analysis.strategy.run_strategy as run_strategy

        cfg = type(
            "Cfg",
            (),
            {
                "REBALANCE_PERIODS": [5, 10],
                "COMPOSITE_FACTOR_FILES_BY_PERIOD": {
                    5: "composite_factors_P5.xlsx",
                    10: "composite_factors_P10.xlsx",
                },
                "COMPOSITE_FACTOR_FILE": "composite_factors.xlsx",
                "COMPOSITE_FACTOR_SHEET": "ic_m3_N20",
            },
        )
        dates = pd.bdate_range("2026-01-01", periods=20)
        frames = {
            "composite_factors_P5.xlsx": pd.DataFrame({"AAPL": 1.0}, index=dates[::5]),
            "composite_factors_P10.xlsx": pd.DataFrame({"AAPL": 1.0}, index=dates[::10]),
        }
        calls = []
        old_loader = run_strategy.load_composite_factor
        run_strategy.load_composite_factor = lambda path, sheet: calls.append((path, sheet)) or frames[path]
        try:
            got = run_strategy._load_composite_factors_for_strategy_periods(cfg)
        finally:
            run_strategy.load_composite_factor = old_loader

        self.assertEqual(set(got), {5, 10})
        self.assertEqual(
            calls,
            [
                ("composite_factors_P5.xlsx", "ic_m3_N20"),
                ("composite_factors_P10.xlsx", "ic_m3_N20"),
            ],
        )

    def test_strategy_grid_expands_max_weight_grid_as_scalars(self):
        from analysis.strategy.strategy_backtest import StrategyBacktester

        cfg = type(
            "Cfg",
            (),
            {
                "GROUP_NUMS": [1],
                "REBALANCE_PERIODS": [5],
                "TARGET_GROUP_RANKS": [1],
                "WEIGHT_METHODS": ["max_return"],
                "EXIT_POLICY_GRID": ["fixed_rebalance"],
                "OPTIMIZATION_LOOKBACK": 5,
                "RISK_FREE_RATE": 0.02,
                "MAX_WEIGHT": 0.5,
                "MAX_WEIGHT_GRID": [0.5, 1.0],
                "TRANSACTION_COST": 0.0,
            },
        )
        ret_index = pd.bdate_range("2026-01-01", periods=15)
        factor = pd.DataFrame(
            {
                "AAA": 1.0,
                "BBB": 2.0,
                "CCC": 3.0,
            },
            index=ret_index,
        )
        returns = pd.DataFrame(
            {
                "AAA": 0.001,
                "BBB": 0.002,
                "CCC": 0.003,
            },
            index=ret_index,
        )

        backtester = StrategyBacktester(factor, returns, cfg)
        self.assertEqual(len(backtester._all_combinations()), 2)

        results = backtester.run_grid()

        self.assertIn("max_return_1G_Top1_P5d_MW50__fixed_rebalance", results)
        self.assertIn("max_return_1G_Top1_P5d_MW100__fixed_rebalance", results)
        for result in results.values():
            self.assertGreater(len(result["daily_returns"]), 0)
            self.assertIsInstance(result["params"]["max_weight"], float)

    def test_composite_factor_writer_replaces_existing_workbook(self):
        from analysis.multi_factor.run_composite_factor import write_composite_factors_excel

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "composite.xlsx")
            old = pd.DataFrame({"AAPL": [1.0]}, index=[pd.Timestamp("2026-01-02")])
            new = pd.DataFrame({"AAPL": [2.0]}, index=[pd.Timestamp("2026-01-05")])

            write_composite_factors_excel({"ic_m3_N20": old}, path)
            write_composite_factors_excel({"ic_m3_N20": new}, path)

            got = pd.read_excel(path, sheet_name="ic_m3_N20", index_col=0)
            self.assertEqual(float(got.iloc[0, 0]), 2.0)
            leftovers = [name for name in os.listdir(tmp) if name != "composite.xlsx"]
            self.assertEqual(leftovers, [])

    def test_rebalance_sync_replaces_selected_sheet_and_preserves_others(self):
        import types

        if "pandas_market_calendars" not in sys.modules:
            fake_mcal = types.ModuleType("pandas_market_calendars")
            fake_mcal.get_calendar = lambda name: None
            sys.modules["pandas_market_calendars"] = fake_mcal
        if "scipy" not in sys.modules:
            fake_scipy = types.ModuleType("scipy")
            fake_optimize = types.ModuleType("scipy.optimize")
            fake_stats = types.ModuleType("scipy.stats")
            fake_optimize.minimize = lambda *args, **kwargs: None
            fake_stats.spearmanr = lambda *args, **kwargs: (np.nan, np.nan)
            fake_stats.skew = lambda *args, **kwargs: np.nan
            fake_stats.kurtosis = lambda *args, **kwargs: np.nan
            fake_stats.ttest_1samp = lambda *args, **kwargs: (np.nan, np.nan)
            fake_scipy.optimize = fake_optimize
            fake_scipy.stats = fake_stats
            sys.modules["scipy"] = fake_scipy
            sys.modules["scipy.optimize"] = fake_optimize
            sys.modules["scipy.stats"] = fake_stats
        if "yfinance" not in sys.modules:
            sys.modules["yfinance"] = types.ModuleType("yfinance")
        if "requests" not in sys.modules:
            fake_requests = types.ModuleType("requests")
            fake_requests.post = lambda *args, **kwargs: None
            sys.modules["requests"] = fake_requests

        from analysis.strategy.rebalance import rebalance_app

        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src.xlsx")
            dst = os.path.join(tmp, "dst.xlsx")
            idx = [pd.Timestamp("2026-01-02")]
            with pd.ExcelWriter(src) as writer:
                pd.DataFrame({"AAPL": [2.0]}, index=idx).to_excel(writer, sheet_name="ic_m3_N20")
            with pd.ExcelWriter(dst) as writer:
                pd.DataFrame({"AAPL": [1.0]}, index=idx).to_excel(writer, sheet_name="ic_m3_N20")
                pd.DataFrame({"AAPL": [9.0]}, index=idx).to_excel(writer, sheet_name="other")

            rebalance_app._sync_selected_composite_sheet(src, dst, "ic_m3_N20")

            selected = pd.read_excel(dst, sheet_name="ic_m3_N20", index_col=0)
            other = pd.read_excel(dst, sheet_name="other", index_col=0)
            self.assertEqual(float(selected.iloc[0, 0]), 2.0)
            self.assertEqual(float(other.iloc[0, 0]), 9.0)
            leftovers = [name for name in os.listdir(tmp) if name not in {"src.xlsx", "dst.xlsx"}]
            self.assertEqual(leftovers, [])

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = os.path.join(tmp, "run")
            std_dir = os.path.join(tmp, "standard")
            run_reports = os.path.join(run_dir, "composite_factor_reports")
            os.makedirs(run_reports)
            os.makedirs(std_dir)
            idx = [pd.Timestamp("2026-01-02")]

            legacy_src = os.path.join(run_reports, "composite_factors_f95.xlsx")
            p10_src = os.path.join(run_reports, "composite_factors_P10_f95.xlsx")
            p10_dst = os.path.join(std_dir, "composite_factors_P10_f95.xlsx")
            with pd.ExcelWriter(legacy_src) as writer:
                pd.DataFrame({"AAPL": [3.0]}, index=idx).to_excel(writer, sheet_name="ic_m3_N20")
            with pd.ExcelWriter(p10_src) as writer:
                pd.DataFrame({"AAPL": [4.0]}, index=idx).to_excel(writer, sheet_name="ic_m3_N20")
            with pd.ExcelWriter(p10_dst) as writer:
                pd.DataFrame({"AAPL": [1.0]}, index=idx).to_excel(writer, sheet_name="ic_m3_N20")
                pd.DataFrame({"AAPL": [8.0]}, index=idx).to_excel(writer, sheet_name="other")

            old_output_dir = rebalance_app.COMPOSITE_FACTOR_OUTPUT_DIR
            old_indices = rebalance_app.SELECTED_FACTOR_INDICES
            rebalance_app.COMPOSITE_FACTOR_OUTPUT_DIR = std_dir
            rebalance_app.SELECTED_FACTOR_INDICES = [95]
            try:
                rebalance_app._sync_composite_factor_to_standard(run_dir, "ic_m3_N20")
            finally:
                rebalance_app.COMPOSITE_FACTOR_OUTPUT_DIR = old_output_dir
                rebalance_app.SELECTED_FACTOR_INDICES = old_indices

            legacy_dst = os.path.join(std_dir, "composite_factors_f95.xlsx")
            self.assertTrue(os.path.isfile(legacy_dst))
            self.assertFalse(os.path.exists(os.path.join(std_dir, "composite_factor_reports")))
            legacy = pd.read_excel(legacy_dst, sheet_name="ic_m3_N20", index_col=0)
            selected = pd.read_excel(p10_dst, sheet_name="ic_m3_N20", index_col=0)
            other = pd.read_excel(p10_dst, sheet_name="other", index_col=0)
            self.assertEqual(float(legacy.iloc[0, 0]), 3.0)
            self.assertEqual(float(selected.iloc[0, 0]), 4.0)
            self.assertEqual(float(other.iloc[0, 0]), 8.0)

    def test_strategy_entrypoints_use_active_profile_config(self):
        from qqq_config.strategy_profiles import get_active_profile
        import analysis.strategy.run_rebalance_day as rd
        import analysis.strategy.run_detailed_backtest_report as detailed
        import analysis.strategy.rebalance.rebalance_app as rebalance_app

        profile = get_active_profile()
        self.assertEqual(rd.ACTIVE_STRATEGY_PROFILE, profile.name)
        self.assertEqual(rd.COMPOSITE_FACTOR_SHEET, profile.composite_sheet)
        self.assertEqual(rd.STRATEGY_PARAM, profile.strategy_param)
        self.assertEqual(rd.SELECTED_FACTOR_INDICES, list(profile.factor_indices))
        self.assertEqual(
            rebalance_app.STRATEGY_PARAMS["data_download_start_date"],
            profile.data_download_start_date,
        )
        self.assertEqual(detailed.COMPOSITE_FACTOR_SHEET, profile.composite_sheet)
        self.assertEqual(detailed.STRATEGY_PARAM, profile.strategy_param)

    def test_legacy_config_strategy_profiles_reexports_authority(self):
        import config.strategy_profiles as legacy
        import qqq_config.strategy_profiles as authority

        self.assertIs(legacy.get_active_profile, authority.get_active_profile)
        self.assertEqual(legacy.ACTIVE_STRATEGY_PROFILE, authority.ACTIVE_STRATEGY_PROFILE)
        self.assertIn("Strategy4", authority.STRATEGY_PROFILES)

    def test_strategy_profiles_own_ticker_universe(self):
        import qqq_config.strategy_profiles as profiles

        self.assertEqual(len(profiles.TICKER_UNIVERSES["ORIGINAL_108"]), 108)
        self.assertEqual(len(profiles.TICKER_UNIVERSES["ORIGINAL_143"]), 143)
        self.assertEqual(len(profiles.TICKER_UNIVERSES["NASDAQ_100_LAST_6_YEARS"]), 162)
        self.assertEqual(
            len(profiles.TICKER_UNIVERSES["ORIGINAL_108_PLUS_NASDAQ_100"]),
            235,
        )
        for universe in profiles.TICKER_UNIVERSES.values():
            self.assertEqual(len(universe), len(set(universe)))
        nasdaq_history = profiles.TICKER_UNIVERSES["NASDAQ_100_LAST_6_YEARS"]
        for ticker in ("BMRN", "PTON", "MRNA", "MDB", "INSM", "VSNT"):
            self.assertIn(ticker, nasdaq_history)
        for ticker in ("PLTR", "ARM", "ALAB", "RKLB", "SPCX"):
            self.assertIn(ticker, nasdaq_history)
        self.assertEqual(
            profiles.STRATEGY_PROFILES["Strategy1"].ticker_universe,
            "ORIGINAL_108",
        )
        self.assertEqual(
            profiles.STRATEGY_PROFILES["Strategy2"].ticker_universe,
            "ORIGINAL_108",
        )
        for name in ("Strategy3", "Strategy4"):
            profile = profiles.STRATEGY_PROFILES[name]
            tickers = profile.ticker_symbols
            self.assertEqual(profile.ticker_universe, "ORIGINAL_143")
            self.assertEqual(tickers, profiles.TICKER_UNIVERSES["ORIGINAL_143"])
            self.assertIn("AMAT", tickers)
            self.assertIn("KLAC", tickers)
        self.assertEqual(
            profiles.STRATEGY_PROFILES["Strategy12"].ticker_universe,
            "ORIGINAL_108_PLUS_NASDAQ_100",
        )
        self.assertEqual(
            profiles.STRATEGY_PROFILES["Strategy12"].data_download_start_date,
            "2020-01-01",
        )

    def test_data_config_tickers_use_config_unless_explicitly_overridden(self):
        old_profile = os.environ.get("QQQ_STRATEGY_PROFILE")
        old_rebalance_universe = os.environ.get("REBALANCE_TICKER_UNIVERSE")
        old_yfinance_universe = os.environ.get("YFINANCE_TICKER_UNIVERSE")
        had_market_calendar = "pandas_market_calendars" in sys.modules
        old_market_calendar = sys.modules.get("pandas_market_calendars")
        if not had_market_calendar:
            class _MarketCalendarStub:
                pass
            sys.modules["pandas_market_calendars"] = _MarketCalendarStub()
        os.environ["QQQ_STRATEGY_PROFILE"] = "Strategy4"
        os.environ.pop("REBALANCE_TICKER_UNIVERSE", None)
        os.environ.pop("YFINANCE_TICKER_UNIVERSE", None)
        try:
            import data.data_config as dc

            dc = importlib.reload(dc)
            self.assertEqual(dc.YFINANCE_TICKER_UNIVERSE, dc.DATA_PULL_TICKER_UNIVERSE)
            self.assertEqual(
                dc.YFINANCE_TICKER_UNIVERSE,
                "ORIGINAL_108_PLUS_NASDAQ_100",
            )
            self.assertEqual(len(dc.YFINANCE_TICKERS), 235)
            self.assertFalse(dc.should_use_price_sheet("NOT_IN_UNIVERSE"))

            os.environ["REBALANCE_TICKER_UNIVERSE"] = "ORIGINAL_143"
            self.assertEqual(dc.resolve_ticker_universe_name(), "ORIGINAL_143")
            self.assertTrue(dc.should_use_price_sheet("AMAT"))
            self.assertTrue(dc.should_use_price_sheet("KLAC"))
        finally:
            if old_profile is None:
                os.environ.pop("QQQ_STRATEGY_PROFILE", None)
            else:
                os.environ["QQQ_STRATEGY_PROFILE"] = old_profile
            if old_rebalance_universe is None:
                os.environ.pop("REBALANCE_TICKER_UNIVERSE", None)
            else:
                os.environ["REBALANCE_TICKER_UNIVERSE"] = old_rebalance_universe
            if old_yfinance_universe is None:
                os.environ.pop("YFINANCE_TICKER_UNIVERSE", None)
            else:
                os.environ["YFINANCE_TICKER_UNIVERSE"] = old_yfinance_universe
            import data.data_config as dc
            importlib.reload(dc)
            if had_market_calendar:
                sys.modules["pandas_market_calendars"] = old_market_calendar
            else:
                sys.modules.pop("pandas_market_calendars", None)

    def test_project_paths_offset_and_run_dir_layout(self):
        from qqq_core.paths import ProjectPaths, resolve_output_path
        from qqq_core.run_context import RunContext

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

        ctx = RunContext(paths=paths, profile="Strategy1", run_dir=os.path.join(ROOT, "output", "rebalance_runs", "sample"))
        self.assertTrue(str(ctx.price_file).endswith(
            os.path.join("sample", "data", "us_top100_daily_2023_present.xlsx")
        ))
        self.assertTrue(str(ctx.factor_processed_dir).endswith(os.path.join("sample", "factor_processed")))
        self.assertTrue(str(ctx.rebalance_report_file).endswith(
            os.path.join("sample", "reports", "rebalance_day_report.xlsx")
        ))

    def test_strategy_param_helpers_are_core_interfaces(self):
        from qqq_core.strategy_params import (
            build_factor_suffix,
            composite_factors_path,
            parse_strategy_param,
            safe_tag,
            strategy_param_from_params,
        )

        self.assertEqual(build_factor_suffix([95, 101, 32]), "f95-101-32")
        parsed = parse_strategy_param("max_return_5G_Top2_P20d")
        self.assertEqual(parsed, ("max_return", 5, 2, 20))
        self.assertEqual(
            strategy_param_from_params({
                "weight_method": parsed[0],
                "group_num": parsed[1],
                "target_rank": parsed[2],
                "rebalance_period": parsed[3],
            }),
            "max_return_5G_Top2_P20d",
        )
        self.assertTrue(composite_factors_path("run_dir", [95]).endswith(
            os.path.join("run_dir", "composite_factor_reports", "composite_factors_f95.xlsx")
        ))
        self.assertEqual(safe_tag("ic/m3 N20"), "ic_m3N20")

    def test_excel_io_filters_sheets_and_requires_sheet(self):
        from qqq_core.excel_io import (
            atomic_excel_writer,
            read_factor_sheet,
            read_factor_workbook,
            read_price_workbook,
            require_sheet,
        )
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

            factor_path = os.path.join(tmp, "factor.xlsx")
            factor_df = pd.DataFrame(
                {"AAPL": [1.0, 2.0]},
                index=pd.to_datetime(["2026-01-05", "2026-01-02"]),
            )
            with atomic_excel_writer(factor_path) as writer:
                factor_df.to_excel(writer, sheet_name="N5")
                (factor_df * 2).to_excel(writer, sheet_name="N10")

            one_sheet = read_factor_sheet(factor_path, "N5")
            self.assertEqual(list(one_sheet.index), sorted(one_sheet.index))
            workbook = read_factor_workbook(factor_path)
            self.assertEqual(set(workbook), {"N5", "N10"})

    def test_analyze_report_finds_latest_standard_report(self):
        from qqq_core.paths import ProjectPaths
        from tools.analyze_report import find_latest_rebalance_report

        with tempfile.TemporaryDirectory() as tmp:
            paths = ProjectPaths(root=Path(tmp), offset=0)
            old_report = paths.rebalance_runs_dir / "old" / "reports" / "rebalance_day_report.xlsx"
            new_report = paths.rebalance_runs_dir / "new" / "reports" / "rebalance_day_report.xlsx"
            old_report.parent.mkdir(parents=True)
            new_report.parent.mkdir(parents=True)
            old_report.write_text("old", encoding="utf-8")
            new_report.write_text("new", encoding="utf-8")
            os.utime(old_report, (1, 1))
            os.utime(new_report, (2, 2))

            self.assertEqual(find_latest_rebalance_report(paths), new_report)

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

    def test_performance_analyzer_matches_shared_metrics(self):
        from analysis.single_factor.performance import PerformanceAnalyzer
        from qqq_core.performance_metrics import performance_summary

        returns = pd.Series(
            [0.10, -0.05, 0.02],
            index=pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]),
        )
        nav = (1.0 + returns).cumprod()
        expected = performance_summary(returns, rf=0.01, periods_per_year=3)
        actual = PerformanceAnalyzer(nav, returns, rf=0.01, periods_per_year=3).calculate_metrics()

        self.assertAlmostEqual(actual["Total_Return"], expected["total_return"])
        self.assertAlmostEqual(actual["Annual_Return"], expected["annual_return"])
        self.assertAlmostEqual(actual["Volatility"], expected["annual_vol"])
        self.assertAlmostEqual(actual["Sharpe"], expected["sharpe"])
        self.assertAlmostEqual(actual["Max_Drawdown"], expected["max_drawdown"])
        self.assertAlmostEqual(actual["Calmar"], expected["calmar"])
        self.assertAlmostEqual(actual["Win_Rate"], expected["win_rate"])
        self.assertAlmostEqual(actual["Profit_Loss_Ratio"], expected["profit_loss_ratio"])

    def test_strategy_and_rebalance_metrics_match_shared_metrics(self):
        import types

        if "yfinance" not in sys.modules:
            sys.modules["yfinance"] = types.ModuleType("yfinance")
        if "requests" not in sys.modules:
            fake_requests = types.ModuleType("requests")
            fake_requests.post = lambda *args, **kwargs: None
            sys.modules["requests"] = fake_requests
        if "pandas_market_calendars" not in sys.modules:
            fake_mcal = types.ModuleType("pandas_market_calendars")
            fake_mcal.get_calendar = lambda name: None
            sys.modules["pandas_market_calendars"] = fake_mcal
        if "scipy" not in sys.modules:
            fake_scipy = types.ModuleType("scipy")
            fake_optimize = types.ModuleType("scipy.optimize")
            fake_optimize.minimize = lambda *args, **kwargs: None
            fake_scipy.optimize = fake_optimize
            sys.modules["scipy"] = fake_scipy
            sys.modules["scipy.optimize"] = fake_optimize

        from analysis.strategy.strategy_metrics import StrategyMetrics
        from analysis.strategy.rebalance.discord_notifier import compute_extended_metrics
        from qqq_core.performance_metrics import performance_summary, worst_period_drawdown

        daily_returns = pd.Series(
            [-0.10, 0.02, 0.03],
            index=pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"]),
        )
        rebalance_returns = pd.Series(
            [-0.082, 0.03],
            index=pd.to_datetime(["2026-01-02", "2026-01-06"]),
        )
        nav = (1.0 + daily_returns).cumprod()
        expected = performance_summary(daily_returns, rf=0.01, periods_per_year=252)
        expected_wp_dd, _, _ = worst_period_drawdown(daily_returns, rebalance_returns)

        strategy = StrategyMetrics(daily_returns, rebalance_returns, rf=0.01).compute_all()
        rebalance = compute_extended_metrics(daily_returns, nav, rebalance_returns, rf_rate=0.01)

        self.assertAlmostEqual(strategy["annual_return"], expected["annual_return"])
        self.assertAlmostEqual(strategy["annual_vol"], expected["annual_vol"])
        self.assertAlmostEqual(strategy["sharpe"], expected["sharpe"])
        self.assertAlmostEqual(strategy["max_drawdown"], expected["max_drawdown"])
        self.assertEqual(strategy["max_loss_duration"], expected["max_loss_duration"])
        self.assertEqual(strategy["avg_loss_duration"], expected["avg_loss_duration"])
        self.assertAlmostEqual(strategy["calmar"], expected["calmar"])
        self.assertAlmostEqual(strategy["worst_period_drawdown"], expected_wp_dd)

        self.assertAlmostEqual(rebalance["annual_return"], expected["annual_return"])
        self.assertAlmostEqual(rebalance["volatility"], expected["annual_vol"])
        self.assertAlmostEqual(rebalance["sharpe"], expected["sharpe"])
        self.assertAlmostEqual(rebalance["max_drawdown"], expected["max_drawdown"])
        self.assertEqual(rebalance["max_loss_duration"], expected["max_loss_duration"])
        self.assertEqual(rebalance["avg_loss_duration"], expected["avg_loss_duration"])
        self.assertAlmostEqual(rebalance["calmar"], expected["calmar"])
        self.assertAlmostEqual(rebalance["win_rate"], expected["win_rate"])
        self.assertAlmostEqual(rebalance["profit_loss_ratio"], expected["profit_loss_ratio"])
        self.assertAlmostEqual(rebalance["worst_period_drawdown"], expected_wp_dd)

    def test_strategy_review_metrics_keep_fields_and_match_shared_metrics(self):
        import types

        if "pandas_market_calendars" not in sys.modules:
            fake_mcal = types.ModuleType("pandas_market_calendars")
            fake_mcal.get_calendar = lambda name: None
            sys.modules["pandas_market_calendars"] = fake_mcal
        if "yfinance" not in sys.modules:
            sys.modules["yfinance"] = types.ModuleType("yfinance")
        if "requests" not in sys.modules:
            fake_requests = types.ModuleType("requests")
            fake_requests.post = lambda *args, **kwargs: None
            sys.modules["requests"] = fake_requests
        fake_scipy = sys.modules.get("scipy") or types.ModuleType("scipy")
        if "scipy.stats" not in sys.modules:
            fake_stats = types.ModuleType("scipy.stats")
            fake_stats.spearmanr = lambda *args, **kwargs: (np.nan, np.nan)
            fake_stats.skew = lambda *args, **kwargs: np.nan
            fake_stats.kurtosis = lambda *args, **kwargs: np.nan
            fake_stats.ttest_1samp = lambda *args, **kwargs: (np.nan, np.nan)
            sys.modules["scipy.stats"] = fake_stats
            fake_scipy.stats = fake_stats
        if "scipy.optimize" not in sys.modules:
            fake_optimize = types.ModuleType("scipy.optimize")
            fake_optimize.minimize = lambda *args, **kwargs: None
            sys.modules["scipy.optimize"] = fake_optimize
            fake_scipy.optimize = fake_optimize
        sys.modules["scipy"] = fake_scipy
        if "sklearn" not in sys.modules:
            fake_sklearn = types.ModuleType("sklearn")
            fake_linear_model = types.ModuleType("sklearn.linear_model")
            fake_decomposition = types.ModuleType("sklearn.decomposition")
            fake_preprocessing = types.ModuleType("sklearn.preprocessing")

            class _LinearRegression:
                def fit(self, *args, **kwargs):
                    self.coef_ = np.array([])
                    return self

            class _PCA:
                def __init__(self, *args, **kwargs):
                    pass

                def fit_transform(self, values):
                    return np.asarray(values)

            class _StandardScaler:
                def fit_transform(self, values):
                    return np.asarray(values)

            fake_linear_model.LinearRegression = _LinearRegression
            fake_decomposition.PCA = _PCA
            fake_preprocessing.StandardScaler = _StandardScaler
            fake_sklearn.linear_model = fake_linear_model
            fake_sklearn.decomposition = fake_decomposition
            fake_sklearn.preprocessing = fake_preprocessing
            sys.modules["sklearn"] = fake_sklearn
            sys.modules["sklearn.linear_model"] = fake_linear_model
            sys.modules["sklearn.decomposition"] = fake_decomposition
            sys.modules["sklearn.preprocessing"] = fake_preprocessing

        from analysis.strategy.run_strategy_review import compute_metrics
        from qqq_core.performance_metrics import performance_summary

        daily_returns = pd.Series(
            [0.04, -0.02, 0.01],
            index=pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]),
        )
        expected = performance_summary(daily_returns, rf=0.01, periods_per_year=252)
        actual = compute_metrics(daily_returns, rf=0.01, label="strategy")

        self.assertEqual(actual["label"], "strategy")
        self.assertEqual(actual["Trading_Days"], 3)
        self.assertIn("Total_Return", actual)
        self.assertIn("Ann_Return", actual)
        self.assertIn("Ann_Vol", actual)
        self.assertIn("Sharpe", actual)
        self.assertIn("Max_Drawdown", actual)
        self.assertIn("Max_Loss_Duration_TradingDays", actual)
        self.assertIn("Avg_Loss_Duration_TradingDays", actual)
        self.assertIn("Win_Rate_Daily", actual)
        self.assertAlmostEqual(actual["Total_Return"], expected["total_return"])
        self.assertAlmostEqual(actual["Ann_Return"], expected["annual_return"])
        self.assertAlmostEqual(actual["Ann_Vol"], expected["annual_vol"])
        self.assertAlmostEqual(actual["Sharpe"], expected["sharpe"])
        self.assertAlmostEqual(actual["Max_Drawdown"], expected["max_drawdown"])
        self.assertEqual(actual["Max_Loss_Duration_TradingDays"], expected["max_loss_duration"])
        self.assertEqual(actual["Avg_Loss_Duration_TradingDays"], expected["avg_loss_duration"])
        self.assertAlmostEqual(actual["Win_Rate_Daily"], expected["win_rate"])


if __name__ == "__main__":
    unittest.main()
