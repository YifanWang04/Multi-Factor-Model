# 量化因子研究与策略回测

> 本文档旨在帮助 AI 助手和开发者理解代码结构和开发规范。

---

## 项目概述

本项目是一套**美股量化因子研究与多因子策略回测**系统，覆盖完整流程：数据获取 → 因子构建 → 数据处理 → 单因子/多因子/复合因子测试 → 策略构建与回测。

**技术栈：**
- Python 3 + pandas, numpy
- scipy, sklearn（回归、PCA、优化）
- matplotlib, seaborn（可视化）
- openpyxl（Excel 读写）
- yfinance（数据获取）

**数据：** ~100 只美股 ticker，日频价格/成交量（2023 至今）；输出包括 Excel 报表和 PDF 报告。

**激活环境：** `.\.venv\Scripts\Activate.ps1`（PowerShell）或 `.\.venv\Scripts\activate`（CMD/bash）

---

## 目录结构

```
qqq/
├── data/                    # 原始数据
│   ├── data_config.py      # 数据路径和 DATA_START_OFFSET_DAYS 配置
│   ├── us_top100_daily_2023_present.xlsx   # 主数据文件（offset=0）
│   ├── us_top100_daily_2023_present_offset{N}d.xlsx  # offset!=0 时避免覆盖
│   └── pull_yhfinance_Data.py             # 从 yfinance 拉取数据
├── pipeline/                # 数据和因子构建流程
│   ├── build_factors.py     # 从价格/成交量构建原始因子 → factor_raw[_offset{N}d]/
│   └── data_process.py     # 去极值、标准化 → factor_processed[_offset{N}d]/
├── factors/
│   └── factor_library.py   # WorldQuant 101 Alpha 函数实现（Alpha #1–101）
├── config/
│   └── selected_factors_reference.py  # 选定因子的人工查阅参考文档（不被代码引用）
├── analysis/
│   ├── single_factor/      # 单因子测试
│   │   ├── config.py               # 单因子测试配置文件
│   │   ├── backtest.py             # 单因子回测核心引擎
│   │   ├── grouping.py             # 分层分组逻辑
│   │   ├── ic.py                   # IC/Rank_IC 计算与统计分析
│   │   ├── performance.py          # 绩效指标计算（年化收益、夏普、最大回撤等）
│   │   ├── rebalance_manager.py   # 调仓日管理与日期对齐
│   │   ├── visualization.py        # 单因子可视化
│   │   ├── report_generator.py    # PDF 报告生成器
│   │   ├── run_single_factor_test.py      # 单因子入口（指定因子）
│   │   ├── run_all_factors_backtest.py    # 批量单因子入口（全目录）
│   │   ├── run_multi_factor_test.py       # 多因子测试入口
│   │   └── run_collinearity_analysis.py    # 因子共线性分析
│   ├── multi_factor/       # 多因子与复合因子
│   │   ├── composite_config.py      # 复合因子配置文件
│   │   ├── composite_factor.py       # 复合因子计算引擎（beta/IC/Rank_IC/OLS/PCA 等）
│   │   ├── run_composite_factor.py   # 复合因子入口
│   │   └── inspect_ols_weights.py    # OLS 权重查看工具
│   ├── strategy/           # 策略构建与网格回测
│   │   ├── strategy_config.py              # 策略配置文件
│   │   ├── strategy_backtest.py            # 策略回测核心引擎
│   │   ├── strategy_report.py              # 策略 Excel 报告生成
│   │   ├── strategy_metrics.py              # 策略绩效指标计算
│   │   ├── strategy_review_config.py        # 策略复盘配置文件
│   │   ├── portfolio_optimizer.py           # 组合优化器（MVO、最小方差、等权等）
│   │   ├── run_strategy.py                  # 策略回测入口
│   │   ├── run_detailed_backtest_report.py # 单策略详细报告入口
│   │   ├── run_strategy_review.py           # 策略复盘报告入口（多 sheet Excel）
│   │   ├── run_rebalance_day.py            # 调仓日全流程（pull→factors→composite→backtest→report）
│   │   ├── test_discord_notification.py    # Discord Webhook 测试工具
│   │   ├── debug_offset_impact.py          # offset 参数影响调试
│   │   └── debug_daily_return_blanks.py    # 日收益空白调试
│   └── walk_forward/       # Walk-Forward 验证（防过拟合）
│       ├── walk_forward_config.py    # 时间窗口、策略网格、复合因子配置
│       ├── rolling_data_processor.py # 防泄露数据处理（train/test split）
│       ├── walk_forward_engine.py    # 核心验证引擎（多-walk 回测）
│       ├── walk_forward_analyzer.py  # 结果分析（参数稳定性、敏感性）
│       ├── run_walk_forward.py       # 主程序入口
│       ├── test_engine.py            # 快速测试脚本
│       ├── __init__.py
│       └── README.md                 # Walk-Forward 详细文档
├── factor_raw/             # 构建好的原始因子（offset=0）
├── factor_raw_offset{N}d/ # offset!=0 时按 offset 分子目录
├── factor_processed/       # 处理后因子（offset=0）
├── factor_processed_offset{N}d/  # offset!=0 时按 offset 分子目录
├── output/                 # 输出目录
│   ├── single_factor_reports/
│   ├── multi_factor_reports/
│   ├── composite_factor_reports/
│   ├── composite_factor_reports_offset{N}d/   # offset!=0 子目录
│   ├── strategy_reports/
│   ├── strategy_reports_offset{N}d/          # offset!=0 子目录
│   ├── walk_forward_reports/
│   ├── strategy_review_YYYY-MM-DD_HHMMSS/     # run_strategy_review 输出（含 strategy_review.xlsx）
│   └── rebalance_day_YYYY-MM-DD_HHMMSS/      # run_rebalance_day 输出（含 rebalance_day_report.xlsx、strategy_detailed_backtest_report*.xlsx）
├── docs/
│   └── NOTES_VS_CODE_CHECKLIST.md  # Notes 与代码实现对照检查清单
├── analyze_report.py               # 快速查看调仓日报表内容的工具脚本
└── README.md                       # 本文件
```

---

## 核心流程与入口点

| 阶段 | 入口脚本 | 输入 | 输出 |
|------|----------|------|------|
| 数据获取 | `data/pull_yhfinance_Data.py` | yfinance API | `data/us_top100_daily_*.xlsx` |
| 因子构建 | `pipeline/build_factors.py` | 价格/成交量 Excel | `factor_raw[_offset{N}d]/*.xlsx` |
| 数据处理 | `pipeline/data_process.py` | factor_raw | `factor_processed[_offset{N}d]/*.xlsx` |
| 单因子测试 | `analysis/single_factor/run_single_factor_test.py` | 配置指定的因子 + 价格 | PDF 报告 |
| 批量单因子 | `analysis/single_factor/run_all_factors_backtest.py` | factor_processed 全部 | 多份 PDF |
| 多因子测试 | `analysis/single_factor/run_multi_factor_test.py` | multi_factor_config | Excel 报告 |
| 因子共线性 | `analysis/single_factor/run_collinearity_analysis.py` | config + 多因子 Excel | 共线性分析 Excel |
| 因子复合 | `analysis/multi_factor/run_composite_factor.py` | composite_config | composite_factors_{fXX-XX-...}.xlsx + 回测报告 |
| OLS 权重查看 | `analysis/multi_factor/inspect_ols_weights.py` | composite_config | ols_m3_M5_weights.xlsx |
| 策略回测 | `analysis/strategy/run_strategy.py` | strategy_config（自动从 composite_config 推导因子后缀） | strategy_backtest_report.xlsx |
| 策略详细报告 | `analysis/strategy/run_detailed_backtest_report.py` | 复合因子 + 策略参数（自动从 composite_config 推导因子后缀） | strategy_detailed_backtest_report.xlsx |
| 策略复盘报告 | `analysis/strategy/run_strategy_review.py` | strategy_review_config + 复合因子 + 价格数据 | output/strategy_review_YYYY-MM-DD_HHMMSS/strategy_review.xlsx |
| 调仓日全流程 | `analysis/strategy/run_rebalance_day.py` | pull_data→build_factors→data_process→run_composite_factor + 固定策略参数 | `output/rebalance_day_YYYY-MM-DD_HHMMSS/`（含 `rebalance_day_report.xlsx`） |
| Walk-Forward 验证 | `analysis/walk_forward/run_walk_forward.py` | walk_forward_config | walk_forward_report.xlsx + 可视化 |
| 快速报表查看 | `analyze_report.py` | rebalance_day_report.xlsx | 控制台输出（sheet 名称 + 数据预览） |

**运行约定：** 从项目根目录执行（先激活 `.venv`），例如（PowerShell）：

```powershell
.\.venv\Scripts\Activate.ps1
python analysis/single_factor/run_single_factor_test.py
python analysis/single_factor/run_all_factors_backtest.py
python analysis/multi_factor/run_composite_factor.py
python analysis/multi_factor/inspect_ols_weights.py
python analysis/strategy/run_strategy.py
python analysis/strategy/run_detailed_backtest_report.py
python analysis/strategy/run_strategy_review.py                  # 策略复盘（先修改 strategy_review_config）
python analysis/strategy/run_rebalance_day.py                    # 全流程（subprocess 模式）
python analysis/strategy/run_rebalance_day.py --inline           # 全流程（内联模式，更快）
python analysis/strategy/run_rebalance_day.py --skip-pipeline    # 使用已有数据生成报表
python analysis/strategy/run_rebalance_day.py --skip-pull       # Pipeline 中跳过 pull_data
python analysis/strategy/run_rebalance_day.py --no-discord      # 不发送 Discord 通知
python analysis/strategy/run_rebalance_day.py --run-dir <path>  # 指定运行目录复用数据
python analysis/strategy/test_discord_notification.py           # 测试 Discord Webhook
python analysis/walk_forward/run_walk_forward.py
python pipeline/build_factors.py
python pipeline/data_process.py
```

---

## 复合因子方法 (composite_factor.py)

| 方法族 | 变体 | 说明 |
|--------|------|------|
| Beta 加权 | beta_m1/m2, beta_m3_N{5/10/20} | 一元 OLS 斜率加权 |
| IC 加权 | ic_m1/m2, ic_m3_N{5/10/20} | Pearson IC 加权 |
| Rank_IC 加权 | rank_ic_m1/m2, rank_ic_m3_N{5/10/20} | Spearman Rank_IC 加权 |
| 排序加权 | rank_add, rank_mul | 横截面排名相加 / 相乘 |
| OLS 加权 | ols_m1/m2, ols_m3_M{5/10/20} | 多元回归加权 |
| PCA | pca_pc1/2/3 | 主成分分析 |

加权方法含义：
- **m1**：全期均值（Oracle 基准线，存在前瞻偏误——仅供研究对比）
- **m2**：截至当期累计均值（无前瞻）
- **m3**：滚动窗口均值（无前瞻）

---

## 调仓日全流程 (run_rebalance_day)

**用途：** 调仓日全流程——拉取数据 → 构建因子 → 数据处理 → 复合因子 → 策略回测 → 生成 Excel 调仓日报表 + Discord 通知。

**流程：**
1. Pipeline 阶段（subprocess 或 inline 内联模式）
2. 加载复合因子与收益率
3. 运行策略回测（`run_detailed_backtest`）
4. 调仓日判定（历史调仓日 + 外推未来调仓日）
5. 生成 Excel 报表（`rebalance_day_report.xlsx`）
6. 发送 Discord 通知（含绩效指标 + 当前持仓盈亏）

**输出：** `output/rebalance_day_YYYY-MM-DD_HHMMSS/`，含以下子目录和文件：
- `data/` — 原始行情数据
- `factor_raw/` — 因子原始值
- `factor_processed/` — 因子处理后数据
- `composite_factor_reports/` — 复合因子结果
- `rebalance_day_report.xlsx` — 调仓日报表（多 Sheet 合一：Config、Opers、Returns 等）

**Discord 通知内容：**
- 因子选择、复合方式、策略参数
- 绩效指标：总收益率、年化收益率、夏普比率、最大回撤、Calmar 比率、胜率、盈亏比
- 当前持仓盈亏：上次调仓日买入至今的各标的买入价→当前价、区间涨跌幅、涨跌胜率统计
- 今日操作明细（买卖列表，过滤 Weight < 0.0001 的低权重操作）
- 下一调仓日

**Pipeline 执行模式：**
- `--inline`：各步骤在同一 Python 进程中执行（推荐，速度快，无进程启动开销）
- subprocess（默认）：各步骤独立子进程（stdout/stderr 实时流式打印，便于观察进度）

**运行目录约定：** 每次运行自动创建带时间戳的独立目录 `output/rebalance_day_YYYY-MM-DD_HHMMSS/`，各步骤输出保存其中；`--run-dir` 可指定已有目录用于复用数据。

---

## 配置约定

- 各模块均有独立配置：`config.py`（单因子）、`multi_factor_config.py`、`composite_config.py`、`strategy_config.py`、`strategy_review_config.py`、`walk_forward_config.py`
- 关键变量 `PROJECT_ROOT`：**必须一致**；项目根目录为 `D:\qqq`
- 通用路径变量：`PRICE_FILE`、`RETURN_COLUMN`、`FACTOR_FILE`、`OUTPUT_DIR`
- **DATA_START_OFFSET_DAYS**：将数据起始日期提前的交易日数
  - **配置位置：** `data/data_config.py`（直接在代码中设置，无环境变量覆盖）
  - **实现方式：** `pull_yhfinance_Data.py` 将 start_date 回退 N 个交易日，使因子与调仓日历对齐
  - **按 offset 分子目录（不覆盖）：** offset=0 使用默认路径；offset!=0 使用 `_offset{N}d` 后缀，例如 `factor_raw_offset7d/`、`factor_processed_offset7d/`、`output/composite_factor_reports_offset7d/`、`output/strategy_reports_offset7d/` 等
- **因子选择机制（composite_config.py）：**
  - **优先级 1：** 环境变量 `REBALANCE_SELECTED_FACTOR_INDICES`（由 `run_rebalance_day.py` 设置，确保 pipeline 全流程因子一致）
  - **优先级 2：** `MANUALLY_SELECTED_FACTOR_INDICES`（本文件手动配置，适合临时测试）
  - **推荐做法：** 长期换因子 → 修改 `MANUALLY_SELECTED_FACTOR_INDICES` + 同步更新 `strategy_config.STRATEGY_SELECTED_FACTOR_INDICES`；临时测试只需改 `MANUALLY_SELECTED_FACTOR_INDICES`
- **选定因子参考：** `config/selected_factors_reference.py` 包含当前选定因子的完整代码与元数据，仅供人工查阅，不被任何代码 import

---

## 策略复盘 (run_strategy_review)

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

```powershell
python analysis/strategy/run_strategy_review.py
```

**配置项：**

| 配置项 | 说明 | 示例 |
|--------|------|------|
| `SELECTED_FACTOR_INDICES` | 五个因子编号（由 composite_config 自动解析，优先级：1=环境变量 REBALANCE_SELECTED_FACTOR_INDICES，2=MANUALLY_SELECTED_FACTOR_INDICES） | `[32, 62, 65, 95, 101]` |
| `COMPOSITE_FACTOR_SHEET` | 复合方式 | `"ic_m3_N20"`、`"ols_m3_M5"` 等 |
| `STRATEGY_PARAM` | 策略参数 | `"max_return_5G_Top1_P10d"` |
| `LIVE_START_DATE` | 实盘开始日期 | `"2025-06-01"` 或 `None` |
| `BROKER_RECORDS_FILE` | 券商成交记录 | `"path/to/trades.csv"` 或 `None` |
| `RUN_PARAM_SENSITIVITY` | 是否运行参数敏感度 | `True` / `False` |
| `FACTOR_DIR` | 单因子目录覆盖 | `None`=默认 |
| `OUTPUT_DIR` | 输出目录覆盖 | `None`=时间戳子目录 |

**前置条件：** 需先运行 `pipeline/build_factors.py` 和 `pipeline/data_process.py` 生成 `factor_processed` 下的因子文件。

---

## 时序对齐约定

- **因子值：** 调仓日 T 的横截面（收盘价，无前瞻）
- **收益率：** 区间 (T, T_next]，即 T+1 到下一调仓日（含）
- **交易：** T 日收盘执行；T 日收益不计入当期持仓
- **交易价格：** 使用 **Adj Close**；T 日收盘执行以 T 收盘价成交
- `RebalancePeriodManager` 和 `strategy_backtest._select_rebalance_dates` 按**交易日**间隔选择调仓日（如 P10 = 每 10 个交易日调仓）

---

## 命名与编码约定

- 配置类：`SingleFactorConfig`；配置模块：`*_config.py`
- 因子 Excel：`factor_alpha001_processed.xlsx` 等；sheet 名可为 `N5`、`N10`（对应观察期 N）
- 分组：升序分组，组 1 最小、组 10 最大；多空组合：买 10+9，卖 1+2
- 资产配置方式：`equal`、`factor_weight`、`min_variance`、`mvo`、`max_return`、`factor_score`
- 中文注释与文档；英文变量名和 API

---

## 常见陷阱与注意事项

1. **PROJECT_ROOT：** 统一为 `D:\qqq`；修改根目录时需同步更新所有配置文件
2. **Sheet 名与复合因子：** `strategy_config.COMPOSITE_FACTOR_SHEET` 必须与 `composite_factors.xlsx` 中的实际 sheet 名一致
3. **调仓周期：** 系统按**交易日**间隔选择调仓日（如 P10 = 每 10 个交易日调仓）
4. **单因子多 sheet：** 默认读第一个 sheet；可设置 `FACTOR_SHEET` 指定 sheet
5. **收益率列：** 优先使用 Excel 中的 `Return` 列，否则通过 `pct_change()` 计算
6. **依赖项：** 需要 `.venv` 包含 pandas、numpy、scipy、sklearn、matplotlib、openpyxl、yfinance、requests 等
7. **因子选择：** `run_composite_factor.py` / `run_strategy.py` / `run_detailed_backtest_report.py` 的因子来源由 `composite_config._resolve_selected_factor_indices()` 决定（优先级：1=环境变量 `REBALANCE_SELECTED_FACTOR_INDICES`，2=`MANUALLY_SELECTED_FACTOR_INDICES`）；文件名中的因子后缀由 `composite_config.SELECTED_FACTOR_INDICES` 自动推导；切换因子需修改 `MANUALLY_SELECTED_FACTOR_INDICES`（临时）或同步改 `strategy_config`（长期）
8. **Walk-Forward：** 训练/测试严格分离；因子处理和复合因子权重仅使用训练期数据
9. **run_rebalance_day 策略名称：** 由 `TARGET_WEIGHT_METHOD`、`TARGET_GROUP_NUM`、`TARGET_RANK`、`TARGET_REBALANCE_DAYS` 生成
10. **数据加载性能：** `load_price_data`、`load_return_data` 使用一次性 `pd.concat` 避免碎片化警告
11. **修改 DATA_START_OFFSET_DAYS 后需重跑 pipeline：** 必须重新运行 pull → build_factors → data_process → run_composite_factor
12. **run_strategy_review：** 完全自包含，无需先运行 `run_composite_factor`；因子由 composite_config 解析（见上文"因子选择机制"），配置 `MANUALLY_SELECTED_FACTOR_INDICES` 后直接运行
13. **BKNG 拆股导致历史价格不一致：** yfinance 对已发生拆股的股票，每次拉取时的历史复权基准可能不同（如 BKNG 2026-04-06 发生1:25拆股，4月1日和4月8日两次拉取的历史价格可能相差精确的25倍）。这会导致 `factor_raw` → `factor_processed` 阶段的 Z-score 标准化截面被 BKNG 主导拉偏，进而使 IC 复合因子、持仓组合和回测结果在两次独立运行间产生差异。解决方法：固定使用同一版价格数据文件，或在 `data_config.py` 中将 `DATA_START_OFFSET_DAYS` 设为固定值确保两次运行引用同一 offset 子目录。
14. **run_rebalance_day Discord 持仓盈亏区块不显示：** `operations_df` 中只有完整持仓期结束的记录（Next_Rebalance_Date 已确定）；当前持仓因尚未卖出、Next_Rebalance_Date 为外推日期，在回测阶段会被正常写入；`_get_holding_period_info` 对日期列强制 `pd.to_datetime` 转换以避免类型不匹配。
15. **Pipeline subprocess 输出延迟：** 默认 subprocess 模式会缓冲子进程输出；使用 `--inline` 可在内联进程中实时观察各步骤进度。
16. **inspect_ols_weights.py：** 仅在复合因子 sheet 名为 `ols_*` 系列时生效；若选择 beta/IC/Rank_IC 等非 OLS 复合方式，该脚本无输出。

---

## 参考文档

- `docs/NOTES_VS_CODE_CHECKLIST.md`：Notes 与代码实现对照检查清单
- `analysis/walk_forward/README.md`：Walk-Forward 验证系统（时序、防泄露、结果解读）

---

*本文档随项目演进持续更新。*
