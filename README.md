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

Live/rebalance-day pipeline:

```powershell
python analysis/strategy/run_rebalance_day.py --inline
python analysis/strategy/run_rebalance_day.py --skip-pipeline
python analysis/strategy/run_rebalance_day.py --skip-pull
python analysis/strategy/run_rebalance_day.py --no-discord
python analysis/strategy/run_rebalance_day.py --run-dir <existing_run_dir>
```

Other useful entries:

```powershell
python analysis/single_factor/run_single_factor_test.py
python analysis/single_factor/run_all_factors_backtest.py
python analysis/multi_factor/inspect_ols_weights.py
python analysis/strategy/run_detailed_backtest_report.py
python analysis/strategy/run_strategy_review.py
python analysis/strategy/test_discord_notification.py
python analysis/walk_forward/run_walk_forward.py
python analyze_report.py
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
| Batch single factor | `analysis/single_factor/run_all_factors_backtest.py` | processed factor directory | multiple PDF reports |
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

Optimization uses SLSQP and falls back to equal weights when data is insufficient or the solver fails. The covariance matrix uses diagonal regularization to reduce singularity risk.

Performance reports include annualized return, annualized volatility, Sharpe, win rate, profit/loss ratio, max drawdown, Calmar ratio, and worst-period drawdown. Worst-period drawdown is the worst drawdown inside any single holding interval from one rebalance date to the next.

### Rebalance Calendar and Timing

`analysis/strategy/rebalance_calendar.py` is the single source of truth for historical rebalance-date selection.

Timing conventions:

- factor value date: rebalance day T, end of day
- trade execution: T close
- return interval: `(T, T_next]`
- trade price: adjusted close
- period names such as `P10d` mean 10 trading days, not 10 calendar days

Future rebalance-date extrapolation uses the NYSE calendar through `pandas_market_calendars`, so holidays such as Good Friday are handled correctly. Historical and future date selection use the same trading-day counting semantics.

## Configuration

Important config files:

| File | Purpose |
| --- | --- |
| `qqq_core/paths.py` | single source of truth for project root, offset-aware paths, and output layout |
| `qqq_core/run_context.py` | resolved profile/offset/run-dir context for one run |
| `qqq_core/excel_io.py` | shared Excel sheet validation, price workbook loading, and atomic Excel writing |
| `qqq_config/strategy_profiles.py` | single source of truth for active strategy profile, ticker universe, selected factors, composite sheet, and live strategy parameter |
| `data/data_config.py` | data start date, direct-pull ticker universe, offset-aware paths |
| `analysis/single_factor/config.py` | single-factor test settings |
| `analysis/single_factor/multi_factor_config.py` | multi-factor test settings |
| `analysis/multi_factor/composite_config.py` | selected factors and composite settings |
| `analysis/strategy/strategy_config.py` | composite sheet, factor suffix, strategy grid |
| `analysis/strategy/strategy_review_config.py` | self-contained strategy review settings |
| `analysis/walk_forward/walk_forward_config.py` | walk-forward windows and grid |

Keep `PROJECT_ROOT` consistent across configs. The current project root is `D:\qqq`.

### Data Offset

`DATA_START_OFFSET_DAYS` in `data/data_config.py` shifts the yfinance start date earlier by N trading days. This is used to test sensitivity to data-start alignment without overwriting default outputs.

Rules:

- `offset=0` uses default folders and files.
- `offset=N` uses `_offset{N}d` folders and filenames.
- Offset price files no longer fall back to the baseline price file. If `offset=N` is requested and `data/us_top100_daily_2023_present_offset{N}d.xlsx` is missing, data consumers fail fast so backtests cannot silently mix calendars.
- Offset factor and composite directories no longer fall back to baseline directories. Generate matching offset factor/composite files before running offset research.
- Offset start-date calculation uses the NYSE trading calendar, not generic weekdays.
- `run_rebalance_day.py` propagates the offset to subprocesses through `REBALANCE_OFFSET_DAYS`.
- After changing the offset, rerun pull, factor build, factor processing, and composite factor generation.

### Factor Selection

Core strategy selection is centralized in `qqq_config/strategy_profiles.py`. `analysis/strategy/strategy_config.py` and `analysis/multi_factor/composite_config.py` derive their default selected factors, composite sheet, and strategy parameter from the active profile. Use `QQQ_STRATEGY_PROFILE=<profile_name>` for a temporary profile override; `REBALANCE_SELECTED_FACTOR_INDICES` remains a runtime override used by the rebalance pipeline subprocesses. For direct composite-method research before a profile is finalized, set `COMPOSITE_RESEARCH_FACTOR_INDICES` in `analysis/multi_factor/composite_config.py`.

Each strategy profile selects one complete ticker universe through `ticker_universe`: `US_108` for the original 108-stock pool, or `US_143` for the full 143-stock pool used by the June 2026 profiles. `Strategy1` and `Strategy2` use `US_108`; `Strategy3`, `Strategy4`, and `Strategy5` use `US_143`, which includes names such as `AMAT`, `LRCX`, `CRDO`, `ARM`, `MRVL`, `ASML`, `DDOG`, `PANW`, `CRWD`, and `KLAC`.

Direct runs of `data/pull_yhfinance_Data.py` use `DATA_PULL_TICKER_UNIVERSE` in `data/data_config.py`, independent of `QQQ_STRATEGY_PROFILE`. Rebalance-day and other callers should pass the intended universe explicitly, either through `pull_yhfinance_Data.main(ticker_universe=...)` or the `REBALANCE_TICKER_UNIVERSE` / `YFINANCE_TICKER_UNIVERSE` environment variables. `run_rebalance_day.py` passes the active strategy profile's `ticker_universe` into the pipeline automatically. The pull script prints the resolved ticker universe, source, and ticker count at startup.

Composite factor selection is resolved in this order:

1. `REBALANCE_SELECTED_FACTOR_INDICES`, set by the rebalance pipeline for a single run
2. `COMPOSITE_RESEARCH_FACTOR_INDICES` in `analysis/multi_factor/composite_config.py`, for manual direct runs of `run_composite_factor.py`
3. the active profile in `qqq_config/strategy_profiles.py`

Before a strategy profile is finalized, `analysis/strategy/strategy_config.py` also supports direct research overrides for `run_strategy.py`: set only `STRATEGY_RESEARCH_FACTOR_INDICES` and `STRATEGY_RESEARCH_COMPOSITE_SHEET`. The strategy parameter grid remains the normal fixed `GROUP_NUMS`, `REBALANCE_PERIODS`, `TARGET_GROUP_RANKS`, and `WEIGHT_METHODS` section. Leave research overrides as `None` for the active profile defaults, and reset them to `None` before live/rebalance-day runs.

`config/selected_factors_reference.py` is a human reference for selected factor metadata and is not imported by the pipeline.

`config/strategy_profiles.py` is now only a compatibility wrapper that re-exports `qqq_config/strategy_profiles.py`; do not edit it as a second strategy-profile source.

Single-factor tests read `SingleFactorConfig.FACTOR_SHEET`; use this when testing a multi-sheet factor file instead of relying on the first sheet.

Composite factor loading only falls back to the standard path when the primary file is absent. If the primary file exists but is missing the requested sheet or is unreadable, the run fails fast.

## Rebalance-Day Reporting

`analysis/strategy/run_rebalance_day.py` can run the full pipeline or reuse an existing run directory.

Main steps:

1. pull data, build factors, process factors, and generate composite factors
2. run detailed strategy backtest
3. determine current, previous, and next rebalance dates
4. generate `rebalance_day_report.xlsx`
5. optionally send a Discord notification

New rebalance-day runs are created under `output/rebalance_runs/YYYY-MM-DD_HHMMSS_<profile>_offsetN/`. Intermediate data remains in `data/`, `factor_raw/`, `factor_processed/`, and `composite_factor_reports/` inside the run directory; the final workbook is written to `reports/rebalance_day_report.xlsx`. Existing `output/rebalance_day_*` run directories can still be passed with `--run-dir`.

The report includes configuration, operations, return series, cumulative returns, period summaries, current holdings, mark-to-market fields, and next-rebalance information.

Discord messages include:

- selected factors, composite method, and strategy parameters
- performance metrics
- current holding PnL aligned with the Excel mark-to-market logic
- today's buy/sell operations with tiny weights filtered out
- next rebalance date

Discord delivery is disabled unless `REBALANCE_DISCORD_WEBHOOK_URL` is set in the environment. Webhook URLs should not be committed to the repository.

Live price fallback uses local prices first. If a completed historical bar is missing close data, yfinance is queried for the completed daily bar with retry/backoff. `fast_info.last_price` is not written into historical close data.

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
- The pipeline filters operations with `Weight <= 0.0001` to reduce noise and speed reports.
- Shared utilities in `analysis/strategy/strategy_utils.py` centralize price loading, composite loading, strategy parameter parsing, factor suffix construction, small-weight filtering, and mark-to-market patching.
- `build_factor_suffix` is centralized through `strategy_utils` and reused by composite/strategy scripts.
- `MarkToMarket` uses vectorized masks, skips invalid rows where both buy value and weight are missing, and recomputes period return, sell value, and shares for open positions.
- If a period has no valid composite weights, the composite row remains missing instead of being silently set to zero.
- Rebalance-day status handles intraday runs: a future date is not treated as a valid rebalance day unless the factor data and calendar state support it; when today is a valid rebalance day, operations are computed from the latest available factor and live/local prices.
- Single-factor and composite-factor period returns are annualized with `252 / rebalance_period`, while strategy backtests based on daily returns still use 252 trading days per year. Reports generated before this convention are not directly comparable.
- Factor processing no longer deletes raw all-empty/all-zero factors; skipped files are recorded in `factor_processing_skip_manifest.csv`.

## Reference Docs

- `docs/NOTES_VS_CODE_CHECKLIST.md`: checklist comparing design notes with code behavior
- `AGENTS.md`: coding-agent operating instructions for this repository
- `analysis/walk_forward/README.md`: detailed walk-forward validation notes
- `notes_markdown/notion_notes_EN.md`: translated research/design notes

This README is the concise project map. Module-level behavior should be checked in the corresponding config and script files before running a production rebalance.
