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
│   │   ├── run_strategy_review.py       # Strategy review report (multi-sheet Excel)
│   │   ├── run_rebalance_day.py         # Rebalance day full pipeline (pull→factors→composite→backtest→report)
│   │   ├── strategy_config.py           # Strategy config
│   │   ├── strategy_review_config.py    # Strategy review config
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
│   ├── strategy_review_YYYY-MM-DD_HHMMSS/  # run_strategy_review output（含 strategy_review.xlsx）
│   └── rebalance_day_YYYY-MM-DD_HHMMSS/  # run_rebalance_day output（含 rebalance_day_report.xlsx、strategy_detailed_backtest_report*.xlsx）
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
| Factor composition | `analysis/multi_factor/run_composite_factor.py` | composite_config | composite_factors_{fXX-XX-...}.xlsx + backtest report |
| OLS weights inspection | `analysis/multi_factor/inspect_ols_weights.py` | composite_config | ols_m3_M5_weights.xlsx |
| Strategy backtest | `analysis/strategy/run_strategy.py` | strategy_config（自动从 composite_config 推导因子后缀） | strategy_backtest_report.xlsx |
| Detailed strategy report | `analysis/strategy/run_detailed_backtest_report.py` | Composite factor + strategy params（自动从 composite_config 推导因子后缀） | strategy_detailed_backtest_report.xlsx |
| Strategy review report | `analysis/strategy/run_strategy_review.py` | strategy_review_config + composite_factors + price data | output/strategy_review_YYYY-MM-DD_HHMMSS/strategy_review.xlsx |
| Rebalance day pipeline | `analysis/strategy/run_rebalance_day.py` | pull_data→build_factors→data_process→run_composite_factor + fixed strategy params | output/rebalance_day_YYYY-MM-DD_HHMMSS/（含 rebalance_day_report.xlsx、strategy_detailed_backtest_report*.xlsx） |
| Walk-Forward validation | `analysis/walk_forward/run_walk_forward.py` | walk_forward_config | walk_forward_report.xlsx + visualizations |

**Run convention:** Execute from project root, e.g.:
```bash
python analysis/strategy/run_strategy.py
python analysis/strategy/run_detailed_backtest_report.py
python analysis/strategy/run_strategy_review.py                  # Strategy review (edit strategy_review_config first)
python analysis/strategy/run_rebalance_day.py                    # Full pipeline
python analysis/strategy/run_rebalance_day.py --skip-pipeline    # Use existing data to generate report
python analysis/strategy/run_rebalance_day.py --skip-pull        # Pipeline skips data pull
python analysis/walk_forward/run_walk_forward.py
python pipeline/build_factors.py
```

---

## Configuration Conventions

- Each module has its own config: `config.py` (single factor), `multi_factor_config.py`, `composite_config.py`, `strategy_config.py`, `strategy_review_config.py`, `walk_forward_config.py`
- Key variable `PROJECT_ROOT`: **must be consistent**; project root is `D:\qqq`
- Common path variables: `PRICE_FILE`, `RETURN_COLUMN`, `FACTOR_FILE`, `OUTPUT_DIR`
- **DATA_START_OFFSET_DAYS**: Number of trading days to shift data start date earlier
  - **Config location:** `data/data_config.py` (set directly in code; no env var override)
  - **Implementation:** In `pull_yhfinance_Data.py`, start_date is shifted back N trading days so factors and rebalance calendar stay aligned
  - **Subdirs by offset (no overwrite):** offset=0 uses default paths; offset!=0 uses `_offset{N}d` suffix, e.g. `factor_raw_offset7d/`, `factor_processed_offset7d/`, `output/composite_factor_reports_offset7d/`, `output/strategy_reports_offset7d/`, etc.
- **因子选择机制（composite_config.py）：**
  - **优先级 1：** 环境变量 `REBALANCE_SELECTED_FACTOR_INDICES`（由 `run_rebalance_day.py` 设置，确保 pipeline 全流程因子一致）
  - **优先级 2：** `MANUALLY_SELECTED_FACTOR_INDICES`（本文件手动配置，适合临时测试）
  - **推荐做法：** 长期换因子 → 修改 `MANUALLY_SELECTED_FACTOR_INDICES` + 同步更新 `strategy_config.STRATEGY_SELECTED_FACTOR_INDICES`；临时测试只需改 `MANUALLY_SELECTED_FACTOR_INDICES`

---

## Strategy Review (run_strategy_review)

**用途：** 完全自包含的策略复盘，无需前置运行 `run_composite_factor`。根据配置的五个因子、复合方式、策略参数，自动从 `factor_processed` 读取 → 计算复合因子 → 运行策略回测 → 生成复盘报表。

**流程：**
1. 从 `factor_processed` 加载配置的五个因子（`SELECTED_FACTOR_INDICES`）
2. 按选定复合方式（`COMPOSITE_FACTOR_SHEET`）计算复合因子
3. 加载日频收益率与价格数据
4. 拉取基准数据（如 QQQ）
5. 运行策略回测（`STRATEGY_PARAM`）
6. 加载单因子文件（用于因子归因）
7. 参数敏感度分析（可选）
8. 券商记录对比（可选）
9. 写入 Excel 报表

**输出：** `output/strategy_review_YYYY-MM-DD_HHMMSS/strategy_review.xlsx`，含 6 个 Sheet。

**用法：** 修改 `analysis/strategy/strategy_review_config.py` 后运行：

```bash
python analysis/strategy/run_strategy_review.py
```

**配置项：**

| 配置项 | 说明 | 示例 |
|--------|------|------|
| `SELECTED_FACTOR_INDICES` | 五个因子编号（由 composite_config 自动解析，优先级：1=环境变量 REBALANCE_SELECTED_FACTOR_INDICES，2=MANUALLY_SELECTED_FACTOR_INDICES） | `[32, 62, 65, 95, 101]` |
| `COMPOSITE_FACTOR_SHEET` | 复合方式 | `"ic_m3_N20"`, `"ols_m3_M5"` 等 |
| `STRATEGY_PARAM` | 策略参数 | `"max_return_5G_Top1_P10d"` |
| `LIVE_START_DATE` | 实盘开始日期 | `"2025-06-01"` 或 `None` |
| `BROKER_RECORDS_FILE` | 券商成交记录 | `"path/to/trades.csv"` 或 `None` |
| `RUN_PARAM_SENSITIVITY` | 是否运行参数敏感度 | `True` / `False` |
| `FACTOR_DIR` | 单因子目录覆盖 | `None`=默认 |
| `OUTPUT_DIR` | 输出目录覆盖 | `None`=时间戳子目录 |

**前置条件：** 需先运行 `pipeline/build_factors.py` 和 `pipeline/data_process.py` 生成 `factor_processed` 下的因子文件。

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
7. **因子选择：** `run_composite_factor.py` / `run_strategy.py` / `run_detailed_backtest_report.py` 的因子来源由 `composite_config._resolve_selected_factor_indices()` 决定（优先级：1=环境变量 REBALANCE_SELECTED_FACTOR_INDICES，2=MANUALLY_SELECTED_FACTOR_INDICES）；文件名中的因子后缀由 `composite_config.SELECTED_FACTOR_INDICES` 自动推导；切换因子需修改 `MANUALLY_SELECTED_FACTOR_INDICES`（临时）或同步改 `strategy_config`（长期）
8. **Walk-Forward:** Train/test strictly separated; factor processing and composite weights use training data only
9. **run_rebalance_day strategy name:** Generated from `TARGET_WEIGHT_METHOD`, `TARGET_GROUP_NUM`, `TARGET_RANK`, `TARGET_REBALANCE_DAYS`
10. **Data loading performance:** `load_price_data`, `load_return_data` use `pd.concat` once to avoid fragmentation warnings
11. **After changing DATA_START_OFFSET_DAYS, re-run pipeline:** Must re-run pull → build_factors → data_process → run_composite_factor
12. **run_strategy_review：** 完全自包含，无需先运行 `run_composite_factor`；因子由 composite_config 解析（见上文"因子选择机制"），配置 `MANUALLY_SELECTED_FACTOR_INDICES` 后直接运行

---

## Reference Docs

- `docs/NOTES_VS_CODE_CHECKLIST.md`: Checklist vs design notes
- `analysis/walk_forward/README.md`: Walk-Forward validation system (timing, leak prevention, result interpretation)

---

*This document is updated as the project evolves.*
