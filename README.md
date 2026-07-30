# Quantitative Factor Research and Strategy Backtesting

This project is a US equity quantitative research system for the full factor workflow:

`yfinance data -> factor construction -> factor processing -> single/multi-factor testing -> composite factor generation -> strategy backtesting -> rebalance-day reporting`

It is built around daily OHLCV data for roughly 100 US stocks, WorldQuant-style alpha factors, Excel/PDF reports, walk-forward validation, and optional Discord rebalance notifications.

## Quick Start

Run all commands from the project root:

```powershell
cd D:\qqq
.\.venv\Scripts\Activate.ps1
```

Typical research pipeline:

```powershell
python data/pull_yhfinance_Data.py
python pipeline/build_factors.py
python pipeline/data_process.py
python analysis/single_factor/run_multi_factor_test.py
python analysis/single_factor/run_collinearity_analysis.py
python analysis/multi_factor/run_composite_factor.py
python analysis/strategy/run_strategy.py
```

Performance mode:

```powershell
$env:QQQ_MAX_WORKERS = "8"        # set to "1" to force serial execution
python pipeline/build_factors.py
python pipeline/data_process.py
python analysis/strategy/run_strategy.py
python analysis/multi_factor/run_composite_factor.py
python analysis/single_factor/run_multi_factor_test.py
```

The heavy research loops use `QQQ_MAX_WORKERS` for process-level parallelism and
reuse workbook-derived DataFrames through a pickle cache under `output/cache/`.
Set `QQQ_DISABLE_CACHE=1` to bypass the cache, or `QQQ_CACHE_DIR=<path>` to
store it elsewhere.
`build_factors.py` writes each factor workbook atomically and retries transient
Windows Excel I/O errors up to three times. A persistent write failure stops the
run and preserves the previous valid workbook.
`data_process.py` applies the same atomic-write and retry policy to processed
factor workbooks. In parallel runs it loads the reference trading dates once
before dispatch, and any persistent factor failure makes the command fail
instead of silently continuing with a stale output workbook.
`run_strategy.py` keeps all parameter combinations in the statistics sheet, but
limits the daily/cumulative return time-series sheets to the top
`REPORT_TIMESERIES_TOP_N` strategies by Sharpe by default. Set that config value
to `None` or `0` to export every strategy time series.

Live/rebalance-day pipeline:

```powershell
python analysis/strategy/run_rebalance_day.py --inline
python analysis/strategy/run_rebalance_day.py --skip-pipeline
python analysis/strategy/run_rebalance_day.py --skip-pull
python analysis/strategy/run_rebalance_day.py --no-discord
python analysis/strategy/run_rebalance_day.py --run-dir <existing_run_dir>
```

When `run_composite_factor.py` is run directly, it generates the periods listed
in `analysis/multi_factor/composite_config.py::COMPOSITE_REBALANCE_PERIODS`.
When it is invoked by `run_rebalance_day.py`, the rebalance-day pipeline
overrides that list with the active strategy period parsed from `STRATEGY_PARAM`
so the live run only builds the matching P-period composite factor.

Other useful entries:

```powershell
python analysis/single_factor/run_single_factor_test.py
python analysis/single_factor/run_batch_single_factor_tests.py
python analysis/multi_factor/inspect_ols_weights.py
python analysis/strategy/run_detailed_backtest_report.py
python analysis/strategy/run_strategy_review.py
python analysis/strategy/test_discord_notification.py
python analysis/walk_forward/run_walk_forward.py
python analyze_report.py [rebalance_day_report.xlsx]
python backfill_close.py
```

## Environment

Main dependencies:

- Python 3
- pandas, numpy
- scipy, scikit-learn
- matplotlib, seaborn
- openpyxl
- yfinance
- requests
- pandas_market_calendars

The local virtual environment is expected at `.venv/`.

## Project Layout

```text
qqq/
├── qqq_core/                    # Shared path, run-context, and Excel I/O helpers
├── data/                         # Data config and yfinance puller
├── pipeline/                     # Raw factor build and processed factor pipeline
├── factors/                      # WorldQuant 101-style alpha library
├── config/                       # Human-readable selected factor reference
├── analysis/
│   ├── single_factor/            # IC, Rank_IC, grouping, single/multi-factor reports
│   ├── multi_factor/             # Composite factor methods and OLS weight inspection
│   ├── strategy/                 # Strategy backtest, reports, review, rebalance day
│   │   └── rebalance/            # Rebalance operations, Discord, report, market value
│   └── walk_forward/             # Leak-aware rolling train/test validation
├── factor_raw*/                  # Generated raw factor Excel files
├── factor_processed*/            # Generated processed factor Excel files
├── output/                       # Reports and timestamped run directories
│   ├── research/                 # Single/multi/composite/walk-forward research reports
│   ├── strategy/                 # Strategy backtest, detailed, and review reports
│   └── rebalance_runs/           # New rebalance-day run directories
├── tools/                        # Utility scripts; root wrappers are kept for compatibility
├── docs/                         # Design/code checklists
├── notes_markdown/               # Research notes exported from Notion
├── World Quant 101 Factors.pdf   # Factor reference document
├── analyze_report.py             # Compatibility wrapper for tools/analyze_report.py
├── backfill_close.py             # Compatibility wrapper for tools/backfill_close.py
└── README.md / README_zh.md
```

Temporary `_debug_*.py` and `_test_*.py` files are ad hoc diagnostics for recent rebalance-date and intraday scenarios.

## Workflow and Outputs

| Stage | Script | Main Input | Main Output |
| --- | --- | --- | --- |
| Data pull | `data/pull_yhfinance_Data.py` | yfinance | `data/us_top100_daily_2023_present*.xlsx` |
| Build factors | `pipeline/build_factors.py` | OHLCV Excel | `factor_raw*/factor_alphaXXX_raw.xlsx` |
| Process factors | `pipeline/data_process.py` | raw factors | `factor_processed*/factor_alphaXXX_processed.xlsx` |
| Single factor test | `analysis/single_factor/run_single_factor_test.py` | one processed factor | PDF report |
| Batch single-factor tests | `analysis/single_factor/run_batch_single_factor_tests.py` | processed factor directory | multiple PDF reports |
| Multi-factor test | `analysis/single_factor/run_multi_factor_test.py` | selected factors and returns | Excel report |
| Collinearity analysis | `analysis/single_factor/run_collinearity_analysis.py` | selected factors | Excel matrices and series |
| Composite factors | `analysis/multi_factor/run_composite_factor.py` | processed factors | `composite_factors_fXX-...xlsx` and report |
| Strategy backtest | `analysis/strategy/run_strategy.py` | composite factor | `strategy_backtest_report.xlsx` |
| Detailed strategy report | `analysis/strategy/run_detailed_backtest_report.py` | composite factor and strategy params | operation/return detail workbook |
| Strategy review | `analysis/strategy/run_strategy_review.py` | review config | timestamped `strategy_review.xlsx` |
| Rebalance day | `analysis/strategy/run_rebalance_day.py` | full pipeline or existing run dir | `rebalance_day_report.xlsx` and Discord message |
| Walk-forward | `analysis/walk_forward/run_walk_forward.py` | walk-forward config | report, charts, stability analysis |

Offset-aware paths use suffixes such as `factor_raw_offset7d/`, `factor_processed_offset7d/`, `output/research/composite_factor_offset7d/`, `output/strategy/backtest_offset7d/`, and `output/rebalance_runs/YYYY-MM-DD_HHMMSS_<profile>_offset7/`. Legacy `output/composite_factor_reports*`, `output/strategy_reports*`, and `output/rebalance_day_*` directories remain readable for existing workbooks and run directories.

## Core Concepts

### Factor Library

`factors/factor_library.py` implements Alpha 1-101 style factors, excluding formulas that require industry neutralization. Inputs are wide DataFrames with dates as rows and tickers as columns.

Common input keys:

| Key | Meaning |
| --- | --- |
| `open`, `high`, `low`, `close` | OHLC prices, with `close` using adjusted close |
| `volume` | trading volume |
| `returns` | `close.pct_change()` |
| `vwap` | approximate VWAP, `(high + low + close) / 3` |

When yfinance is pulled with `auto_adjust=False`, the raw workbook keeps original OHLC/Close/Adj Close and derives `Adj Open`, `Adj High`, and `Adj Low` from the `Adj Close / Close` adjustment ratio. Factor construction defaults to the legacy research-compatible OHLC convention (`Open/High/Low` raw, `close` adjusted). Set `FACTOR_USE_ADJUSTED_OHLC=True` in `data/data_config.py` to opt into adjusted OHLC for split-consistent factor construction; this will change historical backtest results. Factor, price, and return loaders filter Excel sheets to the current ticker universe resolved by `data/data_config.py`, so stale or experimental extra sheets in a price workbook do not silently change the cross-sectional universe.

Core helper operations include cross-sectional rank, lag/delay, delta, rolling sum/min/max/rank, rolling correlation/covariance, signed power, scaling, and linear decay.

### Factor Processing

`pipeline/data_process.py` processes factors cross-sectionally by date:

- winsorization, currently median MAD clipping by date
- z-score standardization by date
- output as one processed Excel file per factor

The strategy assumes processed factors are already aligned to the same trading calendar as returns.

### Single and Multi-Factor Testing

Single-factor testing computes grouping performance, IC, Rank_IC, long/short tests, long-only tests, short-only tests, and charts.

Multi-factor testing summarizes selected factors in Excel:

- IC, Rank_IC, group Rank_IC mean/IR/t-value/p-value
- long-short return and Sharpe
- long excess return and Sharpe
- cumulative IC, cumulative long return, and cumulative long excess return
- factor statistics for the full sample and trailing 3M, 6M, 1Y, and 2Y windows

### Composite Factor Methods

`analysis/multi_factor/composite_factor.py` supports:

| Family | Variants | Notes |
| --- | --- | --- |
| Beta weighted | `beta_m1`, `beta_m2`, `beta_m3_N{N}` | univariate OLS slope weighting |
| IC weighted | `ic_m1`, `ic_m2`, `ic_m3_N{N}` | Pearson IC weighting |
| Rank_IC weighted | `rank_ic_m1`, `rank_ic_m2`, `rank_ic_m3_N{N}` | Spearman Rank_IC weighting |
| Rank weighted | `rank_add`, `rank_mul` | cross-sectional rank sum/product |
| OLS weighted | `ols_m1`, `ols_m2`, `ols_m3_M{M}` | multivariate regression weighting |
| PCA | `pca_pc1`, `pca_pc2`, `pca_pc3` | principal components as composite factors |

Method meanings:

- `m1`: full-period mean. This is an oracle/research baseline and contains look-ahead bias.
- `m2`: expanding historical mean up to the current date.
- `m3`: rolling historical mean over N or M windows.

`m1` sheets intentionally use full-sample statistics and must be treated as look-ahead/oracle baselines only. They are useful for research comparison, not for live deployment decisions.

`inspect_ols_weights.py` is only useful for `ols_*` composite sheets.

### Strategy Backtesting

Strategies are generated from a composite factor by selecting groups and weight methods.

Supported allocation methods in `portfolio_optimizer.py`:

| Method | Meaning |
| --- | --- |
| `equal` | equal weight |
| `factor_score` | normalized factor scores as weights |
| `min_variance` | minimum variance portfolio |
| `mvo` | Markowitz max-Sharpe portfolio |
| `max_return` | maximize expected return under constraints |

Optimization uses SLSQP and falls back to equal weights when data is insufficient or the solver fails. The optimizer's historical return window includes returns known by the rebalance-date close T; realized holding returns still use `(T, T_next]`. The covariance matrix uses diagonal regularization to reduce singularity risk.

Performance reports include annualized return, annualized volatility, Sharpe, win rate, profit/loss ratio, max drawdown, maximum/average loss duration, Calmar ratio, and worst-period drawdown. Worst-period drawdown is the worst drawdown inside any single holding interval from one rebalance date to the next.

### Performance Metric Convention

All standard performance reports use the return series as the source of truth. Total return is compounded as `prod(1 + returns) - 1`; annualized return is `(1 + total_return) ** (periods_per_year / n) - 1`; annualized volatility is sample volatility with `ddof=1`; Sharpe uses annualized return minus annual risk-free rate over annualized volatility; max drawdown and Calmar include an implicit initial wealth anchor of `1.0`.

Single-factor and composite-factor long metrics are based on rebalance-period returns, using `252 / rebalance_period` periods per year. Strategy backtest, detailed report, rebalance-day report, strategy review, and walk-forward full-period metrics are based on daily portfolio returns and use 252 trading days per year. Rebalance report `Win_Rate` and `Profit_Loss_Ratio` are daily-return based; strategy `open_*` metrics remain rebalance-period/opening based.

Strategy loss duration is measured in trading days from the previous NAV high-water mark until NAV first recovers to or exceeds that level. Maximum and average loss duration include an unfinished underwater episode through the final backtest observation. A strategy that never falls below its high-water mark reports zero for both duration metrics.

This is a metric-convention correction. Historical Excel reports are not backfilled; rerun the corresponding scripts to generate reports with the corrected values.

### Dynamic TP/SL Exit Research

Strategy backtests can compare two exit policies in one `run_strategy.py` run:

- `fixed_rebalance`: the existing fixed rebalance-date exit. This remains the baseline and does not scan TP/SL parameters.
- `dynamic_tp_sl`: each open long position is checked with adjusted close during `(T, T_next]`. If a position hits the dynamic take-profit or stop-loss threshold before the next rebalance date, that position exits and its capital stays in cash until the next rebalance.

The active profile in `qqq_config/strategy_profiles.py` owns the default live/detailed-report parameters:

```python
exit_policy = "fixed_rebalance"  # or "dynamic_tp_sl"
max_weight = 0.4
data_download_start_date = "2023-01-01"
preserve_price_scale = True
price_scale_base_run_dir = r"D:\qqq\output\rebalance_runs\2026-06-24_155408_strategy11_offset0"
tp_base = 0.08
sl_base = 0.05
tp_sl_probability = 1.0
```

`analysis/strategy/strategy_config.py` owns the research grid used by `run_strategy.py`:

```python
EXIT_POLICY_GRID = ["fixed_rebalance", "dynamic_tp_sl"]
TP_BASE_GRID = [0.04, 0.06, 0.08, 0.10, 0.12]
SL_BASE_GRID = [0.02, 0.03, 0.05, 0.07]
TP_SL_PROBABILITY = ACTIVE_PROFILE.tp_sl_probability
MAX_WEIGHT = ACTIVE_PROFILE.max_weight
MAX_WEIGHT_GRID = [0.4, 0.6, 1.0]
```

`MAX_WEIGHT` is derived from the active profile and is the default scalar used by detailed backtest and rebalance-day
flows. Rebalance-day config/status reports and Discord summaries include this
profile weight cap. `run_strategy.py` expands `MAX_WEIGHT_GRID` only for optimizer-based
allocation methods (`min_variance`, `mvo`, `max_return`) and adds the weight cap
to strategy names and report parameters when multiple caps are scanned.

For a 10-trading-day holding period, the dynamic thresholds are:

```text
TP = tp_base * (10 - TD) / 10 * P
SL = sl_base * (10 - TD) / 10 * P
```

There is no machine-learning classifier in v1, so `P=1.0` is the neutral default. Do not interpret it as a learned signal probability until a real probability matrix is added. Detailed and rebalance-day reports use the active profile defaults and include exit fields such as `Exit_Date`, `Exit_Reason`, `TP_Threshold`, `SL_Threshold`, and `Signal_Probability`.

For live use, v1 does not require running a daily Python monitor and does not place broker orders. When the active profile uses `dynamic_tp_sl`, the rebalance-day workbook adds:

- `TP_SL_Schedule`: one row per current holding and future holding trading day before the next rebalance, with precomputed TP/SL return thresholds and TP/SL prices.
- `TP_SL_Action_Checklist`: the nearest schedule dates for manual price alerts or near-close checks.

The intended workflow is to generate the full schedule on rebalance day, use broker/TradingView/manual alerts between rebalance dates, and rerun the rebalance-day report only when a TP/SL line is near or triggered.

### Rebalance Calendar and Timing

`analysis/strategy/rebalance_calendar.py` is the single source of truth for historical rebalance-date selection.

Each strategy profile uses `data_download_start_date` as the exact yfinance
download start date. It does not fix the first trade or the later P5/P10/P20
calendar phase. Rebalance dates start from the first usable factor/return date
after the pipeline has finished. Leaving the field as `None` falls back to
`DATA_BASE_START_DATE` and the configured data offset.

Timing conventions:

- factor value date: rebalance day T, end of day
- trade execution: T close
- return interval: `(T, T_next]`
- trade price: adjusted close
- by default, period names such as `P10d` mean a strict 10-trading-day interval

Profiles may optionally enable a fixed-week schedule with all three fields:

```python
rebalance_interval_weeks = 2
rebalance_weekday = 3              # ISO weekday: Monday=1, Friday=5
rebalance_week_anchor_date = "2026-06-24"
```

The fields must be set together and must satisfy
`P{N}d == rebalance_interval_weeks * 5`; one profile accepts only one weekday.
In the example, P10 rebalances every second Wednesday. If that Wednesday is an
NYSE holiday, the rebalance moves to the preceding NYSE session, so a nominal
P10 interval may contain 8 or 9 actual sessions. Period-return reports use
`52 / rebalance_interval_weeks` for annualization in this mode; daily strategy
metrics continue to use 252. `Strategy111` enables this mode with a two-week
interval, Friday (`rebalance_weekday=5`), and a `2026-06-26` phase anchor.
Other profiles retain strict trading-day behavior.

Future rebalance-date extrapolation uses the NYSE calendar through `pandas_market_calendars`, so holidays such as Good Friday are handled correctly. Historical and future date selection use the same trading-day counting semantics.

## Configuration

Important config files:

| File | Purpose |
| --- | --- |
| `qqq_core/paths.py` | single source of truth for project root, offset-aware paths, and output layout |
| `qqq_core/run_context.py` | resolved profile/offset/run-dir context for one run |
| `qqq_core/excel_io.py` | shared Excel sheet validation, price workbook loading, and atomic Excel writing |
| `qqq_core/strategy_params.py` | shared factor suffix, composite workbook path, strategy-param parsing, and safe filename tags |
| `qqq_config/strategy_profiles.py` | single source of truth for active strategy profile, selected factors, composite sheet, profile download start, and live strategy parameter |
| `qqq_config/ticker_universes.py` | named ticker universes imported by strategy profiles and data configuration |
| `data/data_config.py` | data coverage start date, direct-pull ticker universe, offset-aware paths |
| `analysis/single_factor/config.py` | single-factor test settings |
| `analysis/single_factor/multi_factor_config.py` | multi-factor test settings |
| `analysis/multi_factor/composite_config.py` | selected factors and composite settings |
| `analysis/strategy/strategy_config.py` | composite sheet, factor suffix, strategy grid |
| `analysis/strategy/strategy_review_config.py` | self-contained strategy review settings |
| `analysis/walk_forward/walk_forward_config.py` | walk-forward windows and grid |

Config modules derive `PROJECT_ROOT` from `qqq_core.paths.ProjectPaths`; set `QQQ_PROJECT_ROOT` only when intentionally running from a relocated checkout. The current project root is `D:\qqq`.

### Data Offset

`DATA_START_OFFSET_DAYS` in `data/data_config.py` shifts the default yfinance data-coverage start earlier by N trading days and isolates offset artifact paths. During a rebalance-day run, a non-empty profile `data_download_start_date` takes precedence as the exact download start; the offset still controls artifact naming but does not move that profile date.

Rules:

- `offset=0` uses default folders and files.
- `offset=N` uses `_offset{N}d` folders and filenames.
- Offset price files no longer fall back to the baseline price file. If `offset=N` is requested and `data/us_top100_daily_2023_present_offset{N}d.xlsx` is missing, data consumers fail fast so backtests cannot silently mix calendars.
- Offset factor and composite directories no longer fall back to baseline directories. Generate matching offset factor/composite files before running offset research.
- Offset start-date calculation uses the NYSE trading calendar, not generic weekdays.
- A profile download start takes precedence over `DATA_BASE_START_DATE`; it is not passed into the rebalance calendar.
- `run_rebalance_day.py` propagates the offset to subprocesses through `REBALANCE_OFFSET_DAYS`.
- After changing the offset, rerun pull, factor build, factor processing, and composite factor generation.

### Factor Selection

Core strategy selection is centralized in `qqq_config/strategy_profiles.py`. `analysis/strategy/strategy_config.py` and `analysis/multi_factor/composite_config.py` derive their default selected factors, composite sheet, strategy parameter, and data download start from the active profile. Use `QQQ_STRATEGY_PROFILE=<profile_name>` for a temporary profile override; `REBALANCE_SELECTED_FACTOR_INDICES` remains a runtime override used by the rebalance pipeline subprocesses. For direct composite-method research before a profile is finalized, set `COMPOSITE_RESEARCH_FACTOR_INDICES` in `analysis/multi_factor/composite_config.py`.

Each strategy profile selects one complete ticker universe through `ticker_universe`. `ORIGINAL_108` and `ORIGINAL_143` preserve the two original pools. `NASDAQ_100_LAST_6_YEARS` is the 162-ticker union of all Nasdaq-100 securities present from 2020-07-15 through the 2026-07-15 snapshot, including constituents that exited during the window. `ORIGINAL_108_PLUS_NASDAQ_100` is the duplicate-free 235-ticker union used by `Strategy12`; `Strategy1`, `Strategy11`, and `Strategy2` use `ORIGINAL_108`, while `Strategy3` and `Strategy4` use `ORIGINAL_143`.

The six-year Nasdaq universe is a static research universe, not a point-in-time membership series. Using it unchanged across historical dates includes securities before their actual Nasdaq-100 entry date. It is suitable for broad data collection and candidate research, but membership-accurate backtests need date-effective constituent masks.

Direct runs of `data/pull_yhfinance_Data.py` use `DATA_PULL_TICKER_UNIVERSE` in `data/data_config.py`, which defaults to the largest available universe, `ORIGINAL_108_PLUS_NASDAQ_100` (235 tickers), independent of `QQQ_STRATEGY_PROFILE`. Rebalance-day and other callers should pass the intended universe explicitly, either through `pull_yhfinance_Data.main(ticker_universe=...)` or the `REBALANCE_TICKER_UNIVERSE` / `YFINANCE_TICKER_UNIVERSE` environment variables. `run_rebalance_day.py` passes the active strategy profile's `ticker_universe` into the pipeline automatically. The pull script prints the resolved ticker universe, source, and ticker count at startup.

Each strategy profile can set `preserve_price_scale=True` to protect live rebalance runs from yfinance corporate-action rewrites. Use `price_scale_base_run_dir` in the same profile to pin the canonical base run; leave it as `None` to auto-select the newest previous official run for the same profile and offset. `run_rebalance_day.py` passes these values into the data puller automatically.

Composite factor selection is resolved in this order:

1. `REBALANCE_SELECTED_FACTOR_INDICES`, set by the rebalance pipeline for a single run
2. `COMPOSITE_RESEARCH_FACTOR_INDICES` in `analysis/multi_factor/composite_config.py`, for manual direct runs of `run_composite_factor.py`
3. the active profile in `qqq_config/strategy_profiles.py`

`run_rebalance_day.py` derives the factor indices and composite sheet directly
from the active strategy profile and passes them to the pipeline through
`REBALANCE_SELECTED_FACTOR_INDICES` / `REBALANCE_SELECTED_COMPOSITE`, so live
rebalance runs do not inherit `COMPOSITE_RESEARCH_FACTOR_INDICES` or
`STRATEGY_RESEARCH_FACTOR_INDICES`.

Before a strategy profile is finalized, `analysis/strategy/strategy_config.py` also supports direct research overrides for `run_strategy.py`: set only `STRATEGY_RESEARCH_FACTOR_INDICES` and `STRATEGY_RESEARCH_COMPOSITE_SHEET`. The strategy parameter grid remains the normal fixed `GROUP_NUMS`, `REBALANCE_PERIODS`, `TARGET_GROUP_RANKS`, and `WEIGHT_METHODS` section. Leave research overrides as `None` for the active profile defaults, and reset them to `None` before live/rebalance-day runs.

`config/selected_factors_reference.py` is a human reference for selected factor metadata and is not imported by the pipeline.

`config/strategy_profiles.py` is now only a compatibility wrapper that re-exports `qqq_config/strategy_profiles.py`; do not edit it as a second strategy-profile source.

Single-factor tests read `SingleFactorConfig.FACTOR_SHEET`; use this when testing a multi-sheet factor file instead of relying on the first sheet.

Composite factor loading only falls back to the standard path when the primary file is absent. If the primary file exists but is missing the requested sheet or is unreadable, the run fails fast.

`analysis/multi_factor/composite_config.py` derives the primary `REBALANCE_PERIOD` from the active strategy profile's `strategy_param` (`P{N}d`). For multi-period research, set `COMPOSITE_REBALANCE_PERIODS`, for example `[5, 10]`; `run_composite_factor.py` writes period-specific workbooks such as `composite_factors_P5_fXX-...xlsx` and `composite_factors_P10_fXX-...xlsx`. `run_strategy.py` reads the matching workbook for each strategy period, so P5 and P10 strategies do not silently share one composite-factor calendar. The legacy `composite_factors_fXX-...xlsx` file is still written for the primary active-profile period used by detailed/rebalance-day flows.

## Rebalance-Day Reporting

`analysis/strategy/run_rebalance_day.py` can run the full pipeline or reuse an existing run directory.

Main steps:

1. pull data, build factors, process factors, and generate composite factors
2. run detailed strategy backtest
3. determine current, previous, and next rebalance dates
4. generate `rebalance_day_report.xlsx`
5. optionally send a Discord notification

New rebalance-day runs are created under `output/rebalance_runs/YYYY-MM-DD_HHMMSS_<profile>_offsetN/`. Intermediate data remains in `data/`, `factor_raw/`, `factor_processed/`, and `composite_factor_reports/` inside the run directory; the final workbook is written to `reports/rebalance_day_report.xlsx`. Existing `output/rebalance_day_*` run directories can still be passed with `--run-dir`.

For profiles with `preserve_price_scale=True`, the data pull step keeps the configured base run's price workbook as the canonical scale: historical rows through the base run are frozen, fresh rows after that cutoff are appended, and any split-like stable price ratio detected in the overlap is applied to the appended rows only when its absolute difference from the previous scale (`1.0`) is at least `0.01`. This is meant to keep HON/SPGI-style yfinance restatements from changing past rebalance reports while still allowing new dates to be added. The pull step writes `data/price_snapshot_manifest.json` in the run directory with the base workbook and any ticker-level scale factors. The final Excel report includes `Price_Scale_Adjustments` plus `Price_Scale_Config_Base_Run_Dir`, `Price_Scale_Base_Run`, and `Price_Scale_Base_File` summary rows in `Rebalance_Config_Status`, so the configured base and the actual workbook used are visible without checking environment variables.

The report includes configuration, current operations, historical operations, return series, cumulative returns, period summaries, current holdings, mark-to-market fields, and next-rebalance information. `Rebalance_Config_Status` records `Requested_Data_Download_Start`, actual `Data_Coverage_Start`, and `Effective_Rebalance_Start`. Legacy anchor rows remain for workbook compatibility but are no longer populated from the profile download date. `Period_Summary_2` mirrors `Period_Summary` from `2026-03-27` onward and resets `Period_Cumulative_Return` at that date. `Performance_By_Year` reports calendar-year return, volatility, Sharpe, max drawdown, partial-year status, and the full-period CAGR if each year's returns were set to zero while preserving the original timeline. `Return_Attribution` contains ranked rebalance-period log-return contributions, ticker holding/weight/simple-contribution summaries with a trailing two-year contribution share, and an all-held-ticker exclusion stress table. The exclusion stress replaces one ticker's reconstructed daily contribution with cash, leaves other weights unchanged, and retains the original portfolio transaction cost. For `dynamic_tp_sl` profiles the report also includes `TP_SL_Schedule` and `TP_SL_Action_Checklist` for precomputed manual TP/SL monitoring. `Current_Operations_All` is no longer emitted; `All_Operations_All` filters out rows with `Weight < 0.01`.

Discord messages include:

- selected factors, composite method, and strategy parameters
- performance metrics
- current holding PnL aligned with the Excel mark-to-market logic
- today's buy/sell operations with tiny weights filtered out
- next rebalance date

Discord delivery is disabled unless `REBALANCE_DISCORD_WEBHOOK_URL` is set in the environment. Webhook URLs should not be committed to the repository.

Live price fallback uses local prices first. If a completed historical bar is missing close data, yfinance is queried for the completed daily bar with retry/backoff. `fast_info.last_price` is not written into historical close data.

The yfinance pull isolates failures by ticker. A symbol that returns no data or raises a download exception is skipped with an encoding-safe `[SKIP]` notice and included in the final skipped-symbol summary; the remaining symbols continue. The pull still fails if every requested symbol is skipped, preventing an empty price workbook from entering the factor pipeline.

## Strategy Review

`analysis/strategy/run_strategy_review.py` is a self-contained review tool. It does not require a prebuilt composite factor workbook.

It loads configured factors from `factor_processed`, computes the selected composite factor, runs the configured strategy, optionally compares broker records, optionally runs parameter sensitivity, and writes a timestamped `strategy_review.xlsx`.

Configure it in `analysis/strategy/strategy_review_config.py`.

## Walk-Forward Validation

`analysis/walk_forward/` validates strategies with rolling train/test windows.

Pipeline:

1. create rolling windows
2. process factors using training data only
3. compute composite weights using training-period beta/IC statistics only
4. apply fixed weights to the test window
5. grid-search strategy parameters on test data
6. summarize parameter stability, sensitivity, daily returns, cumulative returns, and per-walk results

This is the preferred tool for checking overfitting and parameter stability after exploratory full-period research.

## Operational Notes

- Use the same price data file for comparable runs. Corporate actions can change adjusted historical prices across yfinance pulls; BKNG's 1:25 split is a known example that can materially affect cross-sectional z-scores and holdings.
- `run_rebalance_day.py --inline` is usually faster and easier to monitor than subprocess mode.
- Path and output layout rules are centralized in `qqq_core.paths.ProjectPaths`; avoid hard-coding new `output/...` paths in business scripts.
- Per-run profile/offset/run-dir paths are centralized in `qqq_core.run_context.RunContext`; new orchestration code should cross this Interface instead of re-parsing environment variables.
- Factor suffixes, composite workbook names, and strategy parameter parsing are centralized in `qqq_core.strategy_params`; `analysis/strategy/strategy_utils.py` keeps compatibility exports for older imports.
- The rebalance-day report filters current operations with `Weight < 0.0001`, and filters the historical `All_Operations_All` sheet with `Weight < 0.01` to reduce noise.
- Shared utilities in `analysis/strategy/strategy_utils.py` centralize price loading, composite loading, strategy parameter parsing, factor suffix construction, small-weight filtering, and mark-to-market patching.
- `build_factor_suffix` is centralized through `strategy_utils` and reused by composite/strategy scripts.
- `MarkToMarket` uses vectorized masks, skips invalid rows where both buy value and weight are missing, and recomputes period return, sell value, and shares for open positions.
- If a period has no valid composite weights, the composite row remains missing instead of being silently set to zero.
- Rebalance-day status handles intraday runs: a future date is not treated as a valid rebalance day unless the factor data and calendar state support it; when today is a valid rebalance day, operations are computed from the latest available factor and live/local prices.
- Single-factor and composite-factor period returns are annualized with `252 / rebalance_period`, while strategy backtests based on daily returns still use 252 trading days per year. Reports generated before this convention are not directly comparable.
- Factor processing no longer deletes raw all-empty/all-zero factors; skipped files are recorded in `factor_processing_skip_manifest.csv`.
- `python analyze_report.py` now inspects the newest standard rebalance report by default; pass a workbook path to inspect a specific historical report.

## Reference Docs

- `docs/NOTES_VS_CODE_CHECKLIST.md`: checklist comparing design notes with code behavior
- `AGENTS.md`: coding-agent operating instructions for this repository
- `analysis/walk_forward/README.md`: detailed walk-forward validation notes
- `notes_markdown/notion_notes_EN.md`: translated research/design notes

This README is the concise project map. Module-level behavior should be checked in the corresponding config and script files before running a production rebalance.
