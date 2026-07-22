import tempfile
import unittest
from pathlib import Path

import pandas as pd
import pandas_market_calendars as pmc

from analysis.strategy.rebalance.rebalance_report import (
    build_performance_by_year,
    build_period_summary_2,
    build_return_attribution,
    write_rebalance_day_report,
)
from qqq_core.performance_metrics import performance_summary


class PeriodSummary2Tests(unittest.TestCase):
    @staticmethod
    def _period_summary() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Rebalance_Date": pd.to_datetime(
                    ["2026-03-13", "2026-03-27", "2026-04-13"]
                ),
                "Next_Rebalance_Date": pd.to_datetime(
                    ["2026-03-27", "2026-04-13", "2026-04-27"]
                ),
                "Holding_Days": [14, 17, 14],
                "Position_Count": [22, 22, 21],
                "Period_Return": [-0.05, 0.10, -0.02],
                "Period_Cumulative_Return": [9.0, 10.0, 11.0],
                "Symbols": ["A", "B", "C"],
            }
        )

    def test_filters_from_march_27_and_resets_cumulative_return(self):
        original = self._period_summary()

        result = build_period_summary_2(original)

        self.assertEqual(
            result["Rebalance_Date"].dt.strftime("%Y-%m-%d").tolist(),
            ["2026-03-27", "2026-04-13"],
        )
        self.assertEqual(result.columns.tolist(), original.columns.tolist())
        self.assertAlmostEqual(result.loc[0, "Period_Cumulative_Return"], 0.10)
        self.assertAlmostEqual(result.loc[1, "Period_Cumulative_Return"], 0.078)
        self.assertEqual(original.loc[1, "Period_Cumulative_Return"], 10.0)

    def test_rebalance_workbook_contains_period_summary_2(self):
        dates = pd.to_datetime(["2026-03-27", "2026-03-30"])
        daily_returns = pd.Series([0.01, -0.005], index=dates)
        result = {
            "daily_returns": daily_returns,
            "nav": (1.0 + daily_returns).cumprod(),
            "rebalance_returns": pd.Series(
                [0.005], index=pd.to_datetime(["2026-03-27"])
            ),
            "rebalance_dates": pd.to_datetime(["2026-03-27"]),
            "operations_df": pd.DataFrame(columns=["Weight"]),
            "period_summary_df": self._period_summary(),
            "params": {
                "requested_data_download_start": "2020-01-01",
                "requested_rebalance_anchor": None,
                "effective_rebalance_anchor": "2026-03-27",
                "effective_rebalance_start": "2026-03-27",
            },
            "_ret_df": pd.DataFrame({"A": [0.01, -0.005]}, index=dates),
        }
        status = {
            "is_rebalance_today": False,
            "current_rebalance_date": None,
            "next_rebalance_date": None,
            "future_rebalance_dates": [],
        }

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            output_path = Path(temp_dir) / "rebalance_day_report.xlsx"
            write_rebalance_day_report(
                result,
                status,
                pd.DataFrame(columns=["Weight"]),
                str(output_path),
                strategy_params={
                    "data_download_start_date": "2020-01-01",
                    "rebalance_anchor_date": None,
                },
            )

            with pd.ExcelFile(output_path) as workbook:
                self.assertIn("Period_Summary_2", workbook.sheet_names)
                self.assertIn("Performance_By_Year", workbook.sheet_names)
                self.assertIn("Return_Attribution", workbook.sheet_names)
                written = pd.read_excel(workbook, sheet_name="Period_Summary_2")
                config_status = pd.read_excel(
                    workbook,
                    sheet_name="Rebalance_Config_Status",
                ).set_index("Parameter")["Value"]

        self.assertEqual(len(written), 2)
        self.assertAlmostEqual(written.loc[0, "Period_Cumulative_Return"], 0.10)
        self.assertEqual(str(config_status["Data_Coverage_Start"]), "2026-03-27")
        self.assertEqual(str(config_status["Requested_Data_Download_Start"]), "2020-01-01")
        self.assertEqual(str(config_status["Effective_Rebalance_Start"]), "2026-03-27")


class RebalanceAnalysisSheetTests(unittest.TestCase):
    def test_performance_by_year_marks_partial_year_and_keeps_flat_year_timeline(self):
        nyse = pmc.get_calendar("NYSE")
        full_2024 = pd.DatetimeIndex(nyse.valid_days("2024-01-01", "2024-12-31")).tz_convert(None)
        dates = full_2024.append(pd.DatetimeIndex([pd.Timestamp("2025-01-02")]))
        returns = pd.Series([0.001] * len(full_2024) + [0.02], index=dates)

        result = build_performance_by_year(returns, rf_rate=0.02).set_index("Year")

        self.assertFalse(bool(result.loc[2024, "Is_Partial_Year"]))
        self.assertTrue(bool(result.loc[2025, "Is_Partial_Year"]))
        self.assertAlmostEqual(
            result.loc[2024, "Total_Return"],
            (1.001 ** len(full_2024)) - 1.0,
        )
        expected_flat = returns.copy()
        expected_flat.loc[expected_flat.index.year == 2024] = 0.0
        expected_cagr = performance_summary(expected_flat)["annual_return"]
        self.assertAlmostEqual(result.loc[2024, "Full_CAGR_If_Year_Flat"], expected_cagr)

    @staticmethod
    def _attribution_fixture():
        dates = pd.to_datetime(["2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08"])
        stock_returns = pd.DataFrame(
            {
                "A": [0.10, 0.00, -0.02, 0.00],
                "B": [0.00, 0.05, 0.01, 0.01],
            },
            index=dates,
        )
        daily_returns = pd.Series(
            [0.06, 0.02, -0.008, 0.004],
            index=dates,
            name="Daily_Return",
        )
        period_summary = pd.DataFrame(
            {
                "Rebalance_Date": pd.to_datetime(["2025-01-02", "2025-01-06"]),
                "Next_Rebalance_Date": pd.to_datetime(["2025-01-06", "2025-01-08"]),
                "Holding_Days": [4, 2],
                "Period_Return": [(1.06 * 1.02) - 1.0, (0.992 * 1.004) - 1.0],
                "Symbols": ["A:60.0%, B:40.0%", "A:60.0%, B:40.0%"],
            }
        )
        operations = pd.DataFrame(
            {
                "Rebalance_Date": pd.to_datetime(
                    ["2025-01-02", "2025-01-02", "2025-01-06", "2025-01-06"]
                ),
                "Next_Rebalance_Date": pd.to_datetime(
                    ["2025-01-06", "2025-01-06", "2025-01-08", "2025-01-08"]
                ),
                "Symbol": ["A", "B", "A", "B"],
                "Weight": [0.6, 0.4, 0.6, 0.4],
                "Period_Return": [0.10, 0.05, -0.02, (1.01 ** 2) - 1.0],
            }
        )
        return daily_returns, period_summary, operations, stock_returns

    def test_return_attribution_ranks_periods_and_aggregates_tickers(self):
        daily, periods, operations, stock_returns = self._attribution_fixture()

        period_attr, ticker_attr, exclusion, methodology = build_return_attribution(
            daily_returns=daily,
            period_summary=periods,
            operations=operations,
            stock_returns=stock_returns,
            exit_policy="fixed_rebalance",
            recent_years=2,
        )

        self.assertEqual(period_attr.loc[0, "Return_Rank"], 1)
        self.assertEqual(period_attr.loc[0, "Rebalance_Date"], pd.Timestamp("2025-01-02"))
        ticker = ticker_attr.set_index("Symbol")
        self.assertEqual(ticker.loc["A", "Holding_Count"], 2)
        self.assertAlmostEqual(ticker.loc["A", "Average_Weight"], 0.6)
        self.assertAlmostEqual(ticker.loc["A", "Simple_Return_Contribution"], 0.048)
        self.assertAlmostEqual(ticker["Recent_Contribution_Share"].sum(), 1.0)
        self.assertIn("Ticker_Exclusion_Method", methodology["Methodology_Item"].tolist())

        stress = exclusion.set_index("Symbol")
        expected_a_excluded = pd.Series([0.0, 0.02, 0.004, 0.004], index=daily.index)
        expected_a_cagr = performance_summary(expected_a_excluded)["annual_return"]
        self.assertAlmostEqual(
            stress.loc["A", "CAGR_If_Excluded_To_Cash"], expected_a_cagr
        )
        self.assertEqual(set(stress.index), {"A", "B"})


if __name__ == "__main__":
    unittest.main()
