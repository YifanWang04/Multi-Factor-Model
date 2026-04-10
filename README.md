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

**Activate environment:** `.\.venv\Scripts\Activate.ps1` (PowerShell) or `.\.venv\Scripts\activate` (CMD/bash)

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
├── factors/
│   └── factor_library.py   # WorldQuant 101 Alpha implementation (Alpha #1–101)
├── config/
│   └── selected_factors_reference.py  # Human reference doc for selected factors (not imported by code)
├── analysis/
│   ├── single_factor/      # Single factor testing
│   │   ├── config.py               # Single factor test config
│   │   ├── backtest.py             # Single factor backtest engine
│   │   ├── grouping.py             # Grouping logic
│   │   ├── ic.py                   # IC/Rank_IC calculation and statistical analysis
│   │   ├── performance.py          # Performance metrics (annualized return, Sharpe, max drawdown, etc.)
│   │   ├── rebalance_manager.py   # Rebalance date management and alignment
│   │   ├── visualization.py        # Single factor visualization
│   │   ├── report_generator.py    # PDF report generator
│   │   ├── run_single_factor_test.py      # Single factor entry (specified factor)
│   │   ├── run_all_factors_backtest.py    # Batch single factor entry (full directory)
│   │   ├── run_multi_factor_test.py       # Multi-factor test entry
│   │   └── run_collinearity_analysis.py    # Factor collinearity analysis
│   ├── multi_factor/       # Multi-factor and composite factors
│   │   ├── composite_config.py      # Composite factor config
│   │   ├── composite_factor.py       # Composite factor engine (beta/IC/Rank_IC/OLS/PCA, etc.)
│   │   ├── run_composite_factor.py   # Composite factor entry
│   │   └── inspect_ols_weights.py    # OLS weight inspection tool
│   ├── strategy/           # Strategy construction and grid backtesting
│   │   ├── strategy_config.py              # Strategy config
│   │   ├── strategy_backtest.py            # Strategy backtest engine
│   │   ├── strategy_utils.py               # 共享工具（价格/因子加载、MTM、Discord 格式化等）
│   │   ├── strategy_report.py              # Strategy Excel report generation
│   │   ├── strategy_metrics.py              # Strategy performance metrics
│   │   ├── strategy_review_config.py        # Strategy review config
│   │   ├── portfolio_optimizer.py           # Portfolio optimizer (MVO, min-variance, equal-weight, etc.)
│   │   ├── rebalance_calendar.py           # 统一调仓日历生成（唯一权威实现）
│   │   ├── run_strategy.py                  # Strategy backtest entry
│   │   ├── run_detailed_backtest_report.py  # Detailed strategy report entry
│   │   ├── run_strategy_review.py           # Strategy review report entry (multi-sheet Excel)
│   │   ├── run_rebalance_day.py            # Rebalance day full pipeline (pull→factors→composite→backtest→report)
│   │   ├── test_discord_notification.py    # Discord Webhook test tool
│   │   ├── debug_offset_impact.py          # offset parameter impact debug
│   │   └── debug_daily_return_blanks.py    # Daily return blanks debug
│   └── walk_forward/       # Walk-Forward validation (anti-overfitting)
│       ├── walk_forward_config.py    # Time windows, strategy grid, composite factor config
│       ├── rolling_data_processor.py # Leak-free data processing (train/test split)
│       ├── walk_forward_engine.py    # Core validation engine (multi-walk backtest)
│       ├── walk_forward_analyzer.py  # Result analysis (parameter stability, sensitivity)
│       ├── run_walk_forward.py       # Main entry
│       ├── test_engine.py            # Quick test script
│       ├── __init__.py
│       └── README.md                 # Walk-Forward detailed documentation
├── factor_raw/             # Built raw factors (offset=0)
├── factor_raw_offset{N}d/ # When offset!=0, subdirs by offset
├── factor_processed/       # Processed factors (offset=0)
├── factor_processed_offset{N}d/  # When offset!=0
├── output/                 # Output directory
│   ├── single_factor_reports/
│   ├── multi_factor_reports/
│   ├── composite_factor_reports/
│   ├── composite_factor_reports_offset{N}d/   # offset!=0 subdirs
│   ├── strategy_reports/
│   ├── strategy_reports_offset{N}d/          # offset!=0 subdirs
│   ├── walk_forward_reports/
│   ├── strategy_review_YYYY-MM-DD_HHMMSS/     # run_strategy_review output (contains strategy_review.xlsx)
│   └── rebalance_day_YYYY-MM-DD_HHMMSS/      # run_rebalance_day output (contains rebalance_day_report.xlsx, strategy_detailed_backtest_report*.xlsx)
├── docs/
│   └── NOTES_VS_CODE_CHECKLIST.md  # Notes vs code implementation checklist
├── analyze_report.py               # Quick viewer tool for rebalance day report content
└── README.md                       # This file
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
| Factor composition | `analysis/multi_factor/run_composite_factor.py` | composite_config | composite_factors_{fXX-XX-...}.xlsx + backtest report |
| OLS weights inspection | `analysis/multi_factor/inspect_ols_weights.py` | composite_config | ols_m3_M5_weights.xlsx |
| Strategy backtest | `analysis/strategy/run_strategy.py` | strategy_config (auto-derives factor suffix from composite_config) | strategy_backtest_report.xlsx |
| Detailed strategy report | `analysis/strategy/run_detailed_backtest_report.py` | Composite factor + strategy params (auto-derives factor suffix) | strategy_detailed_backtest_report.xlsx |
| Strategy review report | `analysis/strategy/run_strategy_review.py` | strategy_review_config + composite_factors + price data | output/strategy_review_YYYY-MM-DD_HHMMSS/strategy_review.xlsx |
| Rebalance day pipeline | `analysis/strategy/run_rebalance_day.py` | pull_data→build_factors→data_process→run_composite_factor + fixed strategy params | `output/rebalance_day_YYYY-MM-DD_HHMMSS/` (contains `rebalance_day_report.xlsx`) |
| Walk-Forward validation | `analysis/walk_forward/run_walk_forward.py` | walk_forward_config | walk_forward_report.xlsx + visualizations |
| Quick report viewer | `analyze_report.py` | rebalance_day_report.xlsx | Console output (sheet names + data preview) |

**Run convention:** Execute from project root (activate `.venv` first), e.g. (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
python analysis/single_factor/run_single_factor_test.py
python analysis/single_factor/run_all_factors_backtest.py
python analysis/multi_factor/run_composite_factor.py
python analysis/multi_factor/inspect_ols_weights.py
python analysis/strategy/run_strategy.py
python analysis/strategy/run_detailed_backtest_report.py
python analysis/strategy/run_strategy_review.py                  # Strategy review (edit strategy_review_config first)
python analysis/strategy/run_rebalance_day.py                    # Full pipeline (subprocess mode)
python analysis/strategy/run_rebalance_day.py --inline           # Full pipeline (inline mode, faster)
python analysis/strategy/run_rebalance_day.py --skip-pipeline    # Generate report from existing data
python analysis/strategy/run_rebalance_day.py --skip-pull       # Skip pull_data in pipeline
python analysis/strategy/run_rebalance_day.py --no-discord      # Skip Discord notification
python analysis/strategy/run_rebalance_day.py --run-dir <path>  # Specify run directory to reuse data
python analysis/strategy/test_discord_notification.py           # Test Discord Webhook
python analysis/walk_forward/run_walk_forward.py
python pipeline/build_factors.py
python pipeline/data_process.py
```

---

## Composite Factor Methods (composite_factor.py)

| Method Family | Variants | Description |
|---|---|---|
| Beta weighted | beta_m1/m2, beta_m3_N{5/10/20} | Univariate OLS slope weighting |
| IC weighted | ic_m1/m2, ic_m3_N{5/10/20} | Pearson IC weighting |
| Rank_IC weighted | rank_ic_m1/m2, rank_ic_m3_N{5/10/20} | Spearman Rank_IC weighting |
| Rank weighted | rank_add, rank_mul | Cross-sectional rank sum / product |
| OLS weighted | ols_m1/m2, ols_m3_M{5/10/20} | Multivariate regression weighting |
| PCA | pca_pc1/2/3 | Principal component analysis |

Weighting method meaning:
- **m1**: Full-period mean (oracle baseline, contains look-ahead bias — research only)
- **m2**: Cumulative mean up to current date (no look-ahead)
- **m3**: Rolling window mean (no look-ahead)

---

## Rebalance Day Pipeline (run_rebalance_day)

**Purpose:** Full rebalance day pipeline — pull data → build factors → data processing → composite factor → strategy backtest → generate Excel rebalance report + Discord notification.

**Pipeline:**
1. Pipeline stage (subprocess or inline mode)
2. Load composite factor and returns
3. Run strategy backtest (`run_detailed_backtest`)
4. Determine rebalance dates (historical + extrapolate future)
5. Generate Excel report (`rebalance_day_report.xlsx`)
6. Send Discord notification (performance metrics + current holding P&L)

**Output:** `output/rebalance_day_YYYY-MM-DD_HHMMSS/`, containing:
- `data/` — raw price/volume data
- `factor_raw/` — raw factor values
- `factor_processed/` — processed factor data
- `composite_factor_reports/` — composite factor results
- `rebalance_day_report.xlsx` — multi-sheet rebalance report (Config, Opers, Returns, etc.)

**Discord notification content:**
- Factor selection, composite method, strategy parameters
- Performance metrics: total return, annualized return, Sharpe ratio, max drawdown, Calmar ratio, win rate, profit/loss ratio
- Current holding P&L: aligned with MTM in Excel — weight, assumed sell (As_Of close or live), `Period_Return`, `Sell_Value` (position notionals)
- Today's operations (buy/sell list, Weight ≥ 0.0001): shows weight, prices, optional MTM interval return and position
- Next rebalance date

**Pipeline execution modes:**
- `--inline`: All steps in the same Python process (recommended, faster, no subprocess overhead)
- subprocess (default): Each step as a separate child process (stdout/stderr streamed in real time for progress monitoring)

**Run directory convention:** Each run creates a timestamped directory `output/rebalance_day_YYYY-MM-DD_HHMMSS/`; `--run-dir` can specify an existing directory to reuse data.

---

## Configuration Conventions

- Each module has its own config: `config.py` (single factor), `multi_factor_config.py`, `composite_config.py`, `strategy_config.py`, `strategy_review_config.py`, `walk_forward_config.py`
- Key variable `PROJECT_ROOT`: **must be consistent**; project root is `D:\qqq`
- Common path variables: `PRICE_FILE`, `RETURN_COLUMN`, `FACTOR_FILE`, `OUTPUT_DIR`
- **DATA_START_OFFSET_DAYS**: Number of trading days to shift data start date earlier
  - **Config location:** `data/data_config.py` (set directly in code; no env var override)
  - **Implementation:** In `pull_yhfinance_Data.py`, start_date is shifted back N trading days so factors and rebalance calendar stay aligned
  - **Subdirs by offset (no overwrite):** offset=0 uses default paths; offset!=0 uses `_offset{N}d` suffix, e.g. `factor_raw_offset7d/`, `factor_processed_offset7d/`, `output/composite_factor_reports_offset7d/`, `output/strategy_reports_offset7d/`, etc.
- **Factor selection mechanism (composite_config.py):**
  - **Priority 1:** Env var `REBALANCE_SELECTED_FACTOR_INDICES` (set by `run_rebalance_day.py`, ensures consistent factors across the full pipeline)
  - **Priority 2:** `MANUALLY_SELECTED_FACTOR_INDICES` (manual config in this file, for temporary testing)
  - **Recommended practice:** For long-term factor changes, update `MANUALLY_SELECTED_FACTOR_INDICES` + sync `strategy_config.STRATEGY_SELECTED_FACTOR_INDICES`; for temporary tests, just update `MANUALLY_SELECTED_FACTOR_INDICES`
- **Selected factor reference:** `config/selected_factors_reference.py` contains complete code and metadata for the currently selected factors, for human reference only; not imported by any code

---

## Strategy Review (run_strategy_review)

**Purpose:** Fully self-contained strategy review, no need to run `run_composite_factor` first. Based on configured five factors, composite method, and strategy parameters, it automatically reads from `factor_processed` → computes composite factor → runs strategy backtest → generates review report.

**Pipeline:**
1. Load the five configured factors from `factor_processed` (`SELECTED_FACTOR_INDICES`)
2. Compute composite factor by selected method (`COMPOSITE_FACTOR_SHEET`)
3. Load daily returns and price data
4. Fetch benchmark data (e.g., QQQ)
5. Run strategy backtest (`STRATEGY_PARAM`)
6. Load single factor files (for factor attribution)
7. Parameter sensitivity analysis (optional)
8. Broker record comparison (optional)
9. Write Excel report

**Output:** `output/strategy_review_YYYY-MM-DD_HHMMSS/strategy_review.xlsx`, containing 6 sheets.

**Usage:** Edit `analysis/strategy/strategy_review_config.py` then run:

```powershell
python analysis/strategy/run_strategy_review.py
```

**Configuration items:**

| Config Item | Description | Example |
|------------|-------------|---------|
| `SELECTED_FACTOR_INDICES` | Five factor indices (resolved by composite_config, priority: 1=env var REBALANCE_SELECTED_FACTOR_INDICES, 2=MANUALLY_SELECTED_FACTOR_INDICES) | `[32, 62, 65, 95, 101]` |
| `COMPOSITE_FACTOR_SHEET` | Composite method | `"ic_m3_N20"`, `"ols_m3_M5"`, etc. |
| `STRATEGY_PARAM` | Strategy parameters | `"max_return_5G_Top1_P10d"` |
| `LIVE_START_DATE` | Live trading start date | `"2025-06-01"` or `None` |
| `BROKER_RECORDS_FILE` | Broker trade records | `"path/to/trades.csv"` or `None` |
| `RUN_PARAM_SENSITIVITY` | Run parameter sensitivity | `True` / `False` |
| `FACTOR_DIR` | Single factor directory override | `None`=default |
| `OUTPUT_DIR` | Output directory override | `None`=timestamped subdirectory |

**Prerequisites:** Requires `pipeline/build_factors.py` and `pipeline/data_process.py` to be run first to generate factor files under `factor_processed`.

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
6. **Dependencies:** Requires `.venv` with pandas, numpy, scipy, sklearn, matplotlib, openpyxl, yfinance, requests, etc.
7. **Factor selection:** The factor source for `run_composite_factor.py` / `run_strategy.py` / `run_detailed_backtest_report.py` is determined by `composite_config._resolve_selected_factor_indices()` (priority: 1=env var `REBALANCE_SELECTED_FACTOR_INDICES`, 2=`MANUALLY_SELECTED_FACTOR_INDICES`); the factor suffix in filenames is auto-derived from `composite_config.SELECTED_FACTOR_INDICES`; switching factors requires modifying `MANUALLY_SELECTED_FACTOR_INDICES` (temporary) or syncing with `strategy_config` (long-term)
8. **Walk-Forward:** Train/test strictly separated; factor processing and composite weights use training data only
9. **run_rebalance_day strategy name:** Generated from `TARGET_WEIGHT_METHOD`, `TARGET_GROUP_NUM`, `TARGET_RANK`, `TARGET_REBALANCE_DAYS`
10. **Data loading performance:** `load_price_data`, `load_return_data` use `pd.concat` once to avoid fragmentation warnings
11. **After changing DATA_START_OFFSET_DAYS, re-run pipeline:** Must re-run pull → build_factors → data_process → run_composite_factor
12. **run_strategy_review:** Fully self-contained, no need to run `run_composite_factor` first; factors are resolved by composite_config (see "Factor selection mechanism" above), configure `MANUALLY_SELECTED_FACTOR_INDICES` and run directly
13. **BKNG split causes historical price inconsistency:** For stocks that have undergone splits, yfinance may return different historical adjusted prices on different pull dates (e.g., BKNG had a 1:25 split on 2026-04-06; pulling on Apr 1 vs Apr 8 may yield prices differing by exactly 25x). This causes the Z-score cross-section in `factor_raw` → `factor_processed` to be dominated by BKNG, leading to different IC composite factors, holdings, and backtest results between independent runs. Solution: always use the same price data file, or set a fixed `DATA_START_OFFSET_DAYS` in `data_config.py` so both runs reference the same offset subdirectory.
14. **run_rebalance_day Discord holding P&L block not showing:** `operations_df` only contains records with completed holding periods (Next_Rebalance_Date confirmed); current holdings have an extrapolated Next_Rebalance_Date and are written normally during backtesting; `_get_holding_period_info` forces `pd.to_datetime` conversion on date columns to avoid type mismatches.
15. **Pipeline subprocess output buffering:** Default subprocess mode buffers child process stdout/stderr; use `--inline` to observe real-time progress in-line.
16. **inspect_ols_weights.py:** Only effective when the composite factor sheet name is in the `ols_*` series; if beta/IC/Rank_IC composite methods are selected, this script produces no output.
17. **run_rebalance_day mark-to-market (MTM):** For any holding whose `Next_Rebalance_Date` is still in the future (or `Sell_Price_Close` is missing), the report fills **assumed** `Sell_Price_Close` using the As_Of date's **Adj Close** (last available ≤ As_Of) or **yfinance live price** when the bar is missing. `Period_Return`, `Sell_Value`, and `Shares` are recomputed; see column **`Sell_Price_Source`** (`假设市价(未到期)` vs `到期收盘`). `Period_Summary` for open periods is updated from MTM line items. Completed periods (next rebalance ≤ As_Of with a backtest sell price) keep historical exit prices. MTM runs in two rounds — round 2 (for current_ops after live price update) also patches `period_summary_df` via `patch_period_summary_from_mtm` (now correctly passes the full MTM operations DataFrame, not just current_ops).
18. **Shared utility module (strategy_utils.py):** Centralized reusable functions — `load_price_data`, `load_composite_factor`, `parse_strategy_param`, `build_factor_suffix` (with `default_indices` parameter for configurable fallback), `filter_weight_lt`, `MarkToMarket` class, `patch_period_summary_from_mtm`. Avoids code duplication across `run_rebalance_day.py`, `run_strategy.py`, `run_detailed_backtest_report.py`, etc.
19. **build_factor_suffix unification:** Three locations previously had duplicate implementations (`strategy_utils.py`, `composite_config.py`, `run_composite_factor.py`). Now unified — `strategy_utils.build_factor_suffix` is the canonical implementation with `default_indices` parameter; `composite_config` and `run_composite_factor` import from there via `sys.path` registration.
20. **MarkToMarket Buy_Value guard:** `MarkToMarket.apply()` now skips rows where both `Buy_Value` and `Weight` are NaN, preventing invalid zero-valued MTM entries.
21. **_composite_from_weight_df boundary:** Added `total_w` tracking — when no valid weights exist for a period, the row is left as NaN instead of being silently set to zero.
22. **collect_live_prices_for_mtm vectorized:** No longer uses iterrows; uses vectorized masking to filter open-period rows, then `_get_price_for_symbols_vectorized` for bulk local price lookup. Only symbols with missing local prices trigger yfinance API calls, reducing redundant network requests. The decision is driven by `_is_target_date_session_closed` — it uses a "target_date + 1 day 00:00 UTC" cutoff to confirm the historical bar is closed, instead of checking the current wall-clock time. **Pre-fix bug:** original code used `_is_market_closed_now()` (UTC 21:00 threshold), causing backfill to be skipped during intraday runs (e.g., UTC 06:35) even for yesterday's already-closed bars with missing Close. **Post-fix behavior:**
    - Yesterday closed + yfinance returned OHL but missing Close → triggers backfill
    - Yesterday closed + yfinance already has complete Close → skips
    - Yesterday still in-session (UTC 04:00-21:00 intraday) → skips (safety guard)
    - Yesterday is a non-trading day → automatically rolls back to prior trading day, same logic applies
    - Today intraday → skips (`target_date >= today` returns False immediately; today's bar is never polluted)
23. **strategy_backtest.py unused import (Bug 16):** Removed unused `from datetime import timedelta`.
24. **SimpleNamespace → dataclass (Bug 17):** `run_composite_factor.py` now uses `@dataclass CompositeBacktestConfig` instead of `types.SimpleNamespace`, providing type hints, IDE autocomplete, and clearer intent.
25. **DataFrame slice assignment (Bug 18):** `composite_factor.py` uses `result.iloc[:, :] = 0.0` instead of `result[:] = 0.0` for explicit positional assignment.
26. **load_price_data pd.concat (Bug 19):** No action needed — `strategy_utils.load_price_data` already uses `pd.concat` once for efficient concatenation.
27. **run_strategy_review fpath scope (Bug 20):** No action needed — the fpath variable in the f-string is correctly scoped within the loop iteration where it's defined.
28. **rebalance_manager available[-1] (Bug 21):** No action needed — the call site already has an explicit `if len(available) == 0: continue` guard before `available[-1]`.
29. **Loop variable naming (Bug 22):** `composite_factor.py` `_weighted_composite` and `_composite_from_weight_df` renamed inner date loop variable from `d` to `_date` to avoid shadowing outer-scope references and improve readability.
30. **P1 Bug 23 — `_run_single` 空组合保护：** `strategy_backtest._run_single` 新增 `if len(port_stocks) == 0: continue` 检查，避免目标组为空时产生 NaN 日收益。
31. **P6 Bug 24 — 防御性列对齐：** `strategy_backtest._run_single` 在向量化持仓期收益计算中新增 `w_norm = w_norm[ret_port.columns]`，确保权重 DataFrame 列顺序与收益 DataFrame 完全一致，防止广播顺序风险。
32. **P3 Bug 25 — MarkToMarket 死代码：** `strategy_utils.MarkToMarket.apply` 删除 `ops.loc[need_mtm & ~need_mtm, ...]` 恒假条件行。
33. **P2 Arc-26 — 调仓日历统一实现：** 新建 `analysis/strategy/rebalance_calendar.py` 作为调仓日历唯一权威实现；`strategy_backtest._select_rebalance_dates` 委托至该模块；`rebalance_manager.RebalancePeriodManager.get_rebalance_dates` 导入使用，消除两处重复实现。
34. **P5 Arc-27 — `iterrows` 向量化：** `run_detailed_backtest_report.run_detailed_backtest` 将 `for j, (date, row) in enumerate(period_df.iterrows())` 循环替换为向量化 pandas 批量操作，消除逐行迭代的性能瓶颈。

## Reference Docs

- `docs/NOTES_VS_CODE_CHECKLIST.md`: Checklist vs design notes
- `analysis/walk_forward/README.md`: Walk-Forward validation system (timing, leak prevention, result interpretation)

---

*This document is updated as the project evolves.*
