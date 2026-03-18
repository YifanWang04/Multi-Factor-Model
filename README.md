# Quantitative Factor Research & Strategy Backtesting

> Project context document for AI assistants and developers to understand code structure and development conventions.

---

## Project Overview

This project is a **US equity quantitative factor research and multi-factor strategy backtesting** system, covering the full pipeline: data acquisition → factor construction → data processing → single/multi/composite factor testing → strategy construction and backtesting.

**Tech Stack:**
- Python 3 + pandas, numpy
- scipy, sklearn (regression, PCA, optimization)
- matplotlib, seaborn (visualization)
- openpyxl (Excel read/write)
- yfinance (data acquisition)

**Data:** ~100 US equity tickers, daily price/volume (2023–present); outputs include Excel reports and PDF reports.

---

## Directory Structure

```
qqq/
├── data/                    # Raw data
│   ├── data_config.py      # Data paths and DATA_START_OFFSET_DAYS config
│   ├── us_top100_daily_2023_present.xlsx   # Main daily price/volume (offset=0)
│   ├── us_top100_daily_2023_present_offset{N}d.xlsx  # When offset!=0, avoid overwrite
│   └── pull_yhfinance_Data.py             # Pull data from yfinance
├── pipeline/                # Data and factor construction pipeline
│   ├── build_factors.py     # Build raw factors from price/volume → factor_raw[_offset{N}d]/
│   └── data_process.py     # Winsorization, standardization → factor_processed[_offset{N}d]/
├── analysis/
│   ├── single_factor/      # Single factor testing
│   ├── multi_factor/       # Multi-factor and composite factors
│   ├── strategy/           # Strategy construction and grid backtesting
│   │   ├── run_strategy.py              # Main strategy backtest entry
│   │   ├── run_detailed_backtest_report.py  # Single strategy detailed report
│   │   ├── run_rebalance_day.py         # Rebalance day full pipeline (pull→factors→composite→backtest→report)
│   │   ├── strategy_config.py           # Strategy config
│   │   ├── strategy_backtest.py         # Backtest engine
│   │   ├── strategy_report.py           # Report generation
│   │   ├── strategy_metrics.py          # Metrics calculation
│   │   └── portfolio_optimizer.py       # Portfolio optimization
│   └── walk_forward/       # Walk-Forward validation (anti-overfitting)
│       ├── walk_forward_config.py   # Time windows, strategy grid, composite factor config
│       ├── rolling_data_processor.py # Leak-free data processing (train/test split)
│       ├── walk_forward_engine.py   # Core validation engine (multi-walk backtest)
│       ├── walk_forward_analyzer.py # Result analysis (parameter stability, sensitivity)
│       ├── run_walk_forward.py      # Main entry
│       ├── test_engine.py           # Quick tests
│       └── README.md
├── factor_raw/             # Built raw factors (offset=0)
├── factor_raw_offset{N}d/  # When offset!=0, subdirs by offset
├── factor_processed/       # Processed factors (offset=0)
├── factor_processed_offset{N}d/  # When offset!=0
├── output/                 # Output directory (subdirs by offset)
│   ├── single_factor_reports/
│   ├── multi_factor_reports/
│   ├── composite_factor_reports/
│   ├── strategy_reports/
│   ├── walk_forward_reports/
│   ├── *_offset{N}d/       # Subdirs when offset!=0
│   └── rebalance_day_YYYY-MM-DD_HHMMSS/  # run_rebalance_day output
├── docs/                   # Documentation (notes checklist, etc.)
└── README.md               # This file
```

---

## Core Workflow & Entry Points

| Stage | Entry Script | Input | Output |
|-------|--------------|-------|--------|
| Data acquisition | `data/pull_yhfinance_Data.py` | yfinance API | `data/us_top100_daily_*.xlsx` |
| Factor construction | `pipeline/build_factors.py` | Price/volume Excel | `factor_raw[_offset{N}d]/*.xlsx` |
| Data processing | `pipeline/data_process.py` | factor_raw | `factor_processed[_offset{N}d]/*.xlsx` |
| Single factor test | `analysis/single_factor/run_single_factor_test.py` | config-specified factor + price | PDF report |
| Batch single factor | `analysis/single_factor/run_all_factors_backtest.py` | factor_processed full set | Multiple PDFs |
| Multi-factor test | `analysis/single_factor/run_multi_factor_test.py` | multi_factor_config | Excel report |
| Factor collinearity | `analysis/single_factor/run_collinearity_analysis.py` | config + multi-factor Excel | Collinearity analysis Excel |
| Factor composition | `analysis/multi_factor/run_composite_factor.py` | composite_config | composite_factors.xlsx + backtest report |
| OLS weights inspection | `analysis/multi_factor/inspect_ols_weights.py` | composite_config | ols_m3_M5_weights.xlsx |
| Strategy backtest | `analysis/strategy/run_strategy.py` | strategy_config | strategy_backtest_report.xlsx |
| Detailed strategy report | `analysis/strategy/run_detailed_backtest_report.py` | Composite factor + strategy params (in-script config) | strategy_detailed_backtest_report.xlsx |
| Rebalance day pipeline | `analysis/strategy/run_rebalance_day.py` | pull_data→build_factors→data_process→run_composite_factor + fixed strategy params | output/rebalance_day_YYYY-MM-DD_HHMMSS/ |
| Walk-Forward validation | `analysis/walk_forward/run_walk_forward.py` | walk_forward_config | walk_forward_report.xlsx + visualizations |

**Run convention:** Execute from project root, e.g.:
```bash
python analysis/strategy/run_strategy.py
python analysis/strategy/run_detailed_backtest_report.py
python analysis/strategy/run_rebalance_day.py                    # Full pipeline
python analysis/strategy/run_rebalance_day.py --skip-pipeline    # Use existing data to generate report
python analysis/strategy/run_rebalance_day.py --skip-pull        # Pipeline skips data pull
python analysis/walk_forward/run_walk_forward.py
python pipeline/build_factors.py
```

---

## Configuration Conventions

- Each module has its own config: `config.py` (single factor), `multi_factor_config.py`, `composite_config.py`, `strategy_config.py`, `walk_forward_config.py`
- Key variable `PROJECT_ROOT`: **must be consistent**; project root is `D:\qqq`
- Common path variables: `PRICE_FILE`, `RETURN_COLUMN`, `FACTOR_FILE`, `OUTPUT_DIR`
- **DATA_START_OFFSET_DAYS**: Number of trading days to shift data start date earlier
  - **Config location:** `data/data_config.py`; can be overridden by env var `DATA_START_OFFSET_DAYS`
  - **Implementation:** In `pull_yhfinance_Data.py`, start_date is shifted back N trading days so factors and rebalance calendar stay aligned
  - **Subdirs by offset (no overwrite):** offset=0 uses default paths; offset!=0 uses `_offset{N}d` suffix, e.g. `factor_raw_offset7d/`, `factor_processed_offset7d/`, `output/composite_factor_reports_offset7d/`, `output/strategy_reports_offset7d/`, etc.

---

## Timing & Alignment Conventions

- **Factor values:** Cross-section on rebalance day T (EOD, no look-ahead)
- **Returns:** Interval (T, T_next], i.e. T+1 through next rebalance day (inclusive)
- **Trading:** Executed at T close; T-day return not included in current holding period
- **Trade prices:** Use **Adj Close**; T-close execution trades at T close price
- `RebalancePeriodManager` and `strategy_backtest._select_rebalance_dates` select rebalance dates by **trading-day** interval (e.g. P10 = rebalance every 10 trading days)

---

## Naming & Coding Conventions

- Config classes: `SingleFactorConfig`; config modules: `*_config.py`
- Factor Excel: `factor_alpha001_processed.xlsx` etc.; sheet names may be `N5`, `N10` for observation period N
- Grouping: Ascending groups: group 1 = smallest, group 10 = largest; long-short: long 10+9, short 1+2
- Asset allocation methods: `equal`, `factor_weight`, `min_variance`, `mvo`, `max_return`, `factor_score`
- Chinese comments and docs; variable names and APIs in English

---

## Common Pitfalls & Notes

1. **PROJECT_ROOT:** Unified to `D:\qqq`; update all config files when changing root path
2. **Sheet name & composite factor:** `strategy_config.COMPOSITE_FACTOR_SHEET` must match actual sheet in `composite_factors.xlsx`
3. **Rebalance period:** System selects rebalance days by **trading days** (e.g. P10 = every 10 trading days)
4. **Single factor multi-sheet:** Default reads first sheet; can set `FACTOR_SHEET` to specify sheet
5. **Return column:** Use `Return` column from Excel if present, otherwise compute via `pct_change()`
6. **Dependencies:** Requires `.venv` or equivalent with pandas, numpy, scipy, sklearn, matplotlib, openpyxl, yfinance, etc.
7. **run_detailed_backtest_report:** Run `run_composite_factor.py` first so `composite_factors.xlsx` exists with target sheet
8. **Walk-Forward:** Train/test strictly separated; factor processing and composite weights use training data only
9. **run_rebalance_day strategy name:** Generated from `TARGET_WEIGHT_METHOD`, `TARGET_GROUP_NUM`, `TARGET_RANK`, `TARGET_REBALANCE_DAYS`
10. **Data loading performance:** `load_price_data`, `load_return_data` use `pd.concat` once to avoid fragmentation warnings
11. **After changing DATA_START_OFFSET_DAYS, re-run pipeline:** Must re-run pull → build_factors → data_process → run_composite_factor

---

## Reference Docs

- `docs/NOTES_VS_CODE_CHECKLIST.md`: Checklist vs design notes
- `analysis/walk_forward/README.md`: Walk-Forward validation system (timing, leak prevention, result interpretation)

---

*This document is updated as the project evolves.*
