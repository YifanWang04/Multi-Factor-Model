# 量化因子研究与策略回测系统

> 本文档为 AI 助手和开发者提供项目背景、代码结构和开发规范的参考。

---

## 项目概述

本项目是一套**美股量化因子研究及多因子策略回测**系统，完整覆盖：数据获取 → 因子构建 → 数据处理 → 单因子/多因子/复合因子测试 → 策略构建与回测。

**技术栈：**
- Python 3 + pandas, numpy
- scipy, sklearn（回归、PCA、优化）
- matplotlib, seaborn（可视化）
- openpyxl（Excel 读写）
- yfinance（数据获取）

**数据：** ~100 只美股，日频价格/成交量（2023 年至今）；输出包括 Excel 报告和 PDF 报告。

**激活环境：** `.\.venv\Scripts\Activate.ps1`（PowerShell）或 `.\.venv\Scripts\activate`（CMD/bash）

---

## 目录结构

```
qqq/
├── data/                    # 原始数据
│   ├── data_config.py      # 数据路径和 DATA_START_OFFSET_DAYS 配置
│   ├── us_top100_daily_2023_present.xlsx   # 主力日频价格/成交量（offset=0）
│   ├── us_top100_daily_2023_present_offset{N}d.xlsx  # offset!=0 时避免覆盖
│   └── pull_yhfinance_Data.py             # 从 yfinance 拉取数据
├── pipeline/                # 数据与因子构建流水线
│   ├── build_factors.py     # 从价格/成交量 → 原始因子 → factor_raw[_offset{N}d]/
│   └── data_process.py     # 去极值、标准化 → factor_processed[_offset{N}d]/
├── factors/
│   └── factor_library.py   # WorldQuant 101 Alpha 实现（Alpha #1–101）
├── config/
│   └── selected_factors_reference.py  # 已选因子的参考文档（仅供人工参考，代码不导入）
├── analysis/
│   ├── single_factor/      # 单因子测试
│   │   ├── config.py               # 单因子测试配置
│   │   ├── backtest.py             # 单因子回测引擎（多空/纯多/纯空）
│   │   ├── grouping.py             # 分组逻辑
│   │   ├── ic.py                   # IC/Rank_IC 计算与统计分析
│   │   ├── performance.py          # 绩效指标（年化收益、夏普、最大回撤等）
│   │   ├── rebalance_manager.py   # 调仓日管理与对齐
│   │   ├── visualization.py        # 单因子可视化
│   │   ├── report_generator.py    # PDF 报告生成器
│   │   ├── run_single_factor_test.py      # 单因子测试入口（指定因子）
│   │   ├── run_all_factors_backtest.py    # 批量单因子回测入口（全目录）
│   │   ├── run_multi_factor_test.py       # 多因子测试入口
│   │   └── run_collinearity_analysis.py    # 因子共线性分析
│   ├── multi_factor/       # 多因子与复合因子
│   │   ├── composite_config.py      # 复合因子配置
│   │   ├── composite_factor.py       # 复合因子引擎（beta/IC/Rank_IC/OLS/PCA 等）
│   │   ├── run_composite_factor.py   # 复合因子入口
│   │   └── inspect_ols_weights.py    # OLS 权重检查工具
│   ├── strategy/           # 策略构建与网格回测
│   │   ├── strategy_config.py              # 策略配置
│   │   ├── strategy_backtest.py            # 策略回测引擎
│   │   ├── strategy_utils.py               # 共享工具（价格/因子加载、MTM、Discord 格式化等）
│   │   ├── strategy_report.py              # 策略 Excel 报告生成
│   │   ├── strategy_metrics.py              # 策略绩效指标
│   │   ├── strategy_review_config.py        # 策略回顾配置
│   │   ├── portfolio_optimizer.py           # 组合优化器（MVO、最小方差、等权等）
│   │   ├── rebalance_calendar.py           # 统一调仓日历生成（唯一权威实现）
│   │   ├── run_strategy.py                  # 策略回测入口
│   │   ├── run_detailed_backtest_report.py  # 详细策略报告入口
│   │   ├── run_strategy_review.py           # 策略回顾报告入口（多 Sheet Excel）
│   │   ├── run_rebalance_day.py            # 调仓日全流水线（拉数据→因子→复合→回测→报告）
│   │   ├── test_discord_notification.py    # Discord Webhook 测试工具
│   │   ├── debug_offset_impact.py          # offset 参数影响调试
│   │   └── debug_daily_return_blanks.py    # 日收益空白调试
│   │   ├── rebalance/                    # 调仓日操作模块
│   │   │   ├── rebalance_operations.py   # 调仓日判定、操作明细、实时价格、MTM
│   │   │   ├── discord_notifier.py       # Discord 通知发送
│   │   │   ├── rebalance_report.py       # 调仓日 Excel 报告生成
│   │   │   └── market_value.py           # 市值计算
│   └── walk_forward/       # Walk-Forward 验证（防过拟合）
│       ├── walk_forward_config.py    # 时间窗口、策略网格、复合因子配置
│       ├── rolling_data_processor.py # 无泄露数据处理（训练/测试分割）
│       ├── walk_forward_engine.py    # 核心验证引擎（多 walk 回测）
│       ├── walk_forward_analyzer.py  # 结果分析（参数稳定性、敏感性）
│       ├── run_walk_forward.py       # 主入口
│       ├── test_engine.py            # 快速测试脚本
│       ├── __init__.py
│       └── README.md                 # Walk-Forward 详细文档
├── factor_raw/             # 构建的原始因子（offset=0）
├── factor_raw_offset{N}d/ # offset!=0 时的子目录
├── factor_processed/       # 处理后的因子（offset=0）
├── factor_processed_offset{N}d/  # offset!=0 时的子目录
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
│   └── NOTES_VS_CODE_CHECKLIST.md  # 设计笔记 vs 代码实现检查清单
├── analyze_report.py               # 调仓日报告内容快速查看工具
└── README.md                       # 本文件（英文版）
```

---

## 核心工作流与入口脚本

| 阶段 | 入口脚本 | 输入 | 输出 |
|------|---------|------|------|
| 数据获取 | `data/pull_yhfinance_Data.py` | yfinance API | `data/us_top100_daily_*.xlsx` |
| 因子构建 | `pipeline/build_factors.py` | 价格/成交量 Excel | `factor_raw[_offset{N}d]/*.xlsx` |
| 数据处理 | `pipeline/data_process.py` | factor_raw | `factor_processed[_offset{N}d]/*.xlsx` |
| 单因子测试 | `analysis/single_factor/run_single_factor_test.py` | 配置指定因子 + 价格 | PDF 报告 |
| 批量单因子 | `analysis/single_factor/run_all_factors_backtest.py` | factor_processed 全量 | 多份 PDF |
| 多因子测试 | `analysis/single_factor/run_multi_factor_test.py` | multi_factor_config | Excel 报告 |
| 因子共线性 | `analysis/single_factor/run_collinearity_analysis.py` | config + 多因子 Excel | 共线性分析 Excel |
| 因子复合 | `analysis/multi_factor/run_composite_factor.py` | composite_config | composite_factors_{fXX-XX-...}.xlsx + 回测报告 |
| OLS 权重检查 | `analysis/multi_factor/inspect_ols_weights.py` | composite_config | ols_m3_M5_weights.xlsx |
| 策略回测 | `analysis/strategy/run_strategy.py` | strategy_config（自动从 composite_config 推导因子后缀） | strategy_backtest_report.xlsx |
| 详细策略报告 | `analysis/strategy/run_detailed_backtest_report.py` | 复合因子 + 策略参数（自动推导因子后缀） | strategy_detailed_backtest_report.xlsx |
| 策略回顾报告 | `analysis/strategy/run_strategy_review.py` | strategy_review_config + composite_factors + 价格数据 | output/strategy_review_YYYY-MM-DD_HHMMSS/strategy_review.xlsx |
| 调仓日流水线 | `analysis/strategy/run_rebalance_day.py` | pull_data→build_factors→data_process→run_composite_factor + 固定策略参数 | `output/rebalance_day_YYYY-MM-DD_HHMMSS/`（含 `rebalance_day_report.xlsx`） |
| Walk-Forward 验证 | `analysis/walk_forward/run_walk_forward.py` | walk_forward_config | walk_forward_report.xlsx + 可视化 |
| 快速报告查看 | `analyze_report.py` | rebalance_day_report.xlsx | 控制台输出（Sheet 名 + 数据预览） |

**运行约定：** 在项目根目录下执行（先激活 `.venv`），例如（PowerShell）：

```powershell
.\.venv\Scripts\Activate.ps1
python analysis/single_factor/run_single_factor_test.py
python analysis/single_factor/run_all_factors_backtest.py
python analysis/multi_factor/run_composite_factor.py
python analysis/multi_factor/inspect_ols_weights.py
python analysis/strategy/run_strategy.py
python analysis/strategy/run_detailed_backtest_report.py
python analysis/strategy/run_strategy_review.py                  # 策略回顾（先编辑 strategy_review_config）
python analysis/strategy/run_rebalance_day.py                    # 全流水线（子进程模式）
python analysis/strategy/run_rebalance_day.py --inline           # 全流水线（内联模式，更快）
python analysis/strategy/run_rebalance_day.py --skip-pipeline    # 从已有数据生成报告
python analysis/strategy/run_rebalance_day.py --skip-pull       # 跳过 pull_data
python analysis/strategy/run_rebalance_day.py --no-discord      # 跳过 Discord 通知
python analysis/strategy/run_rebalance_day.py --run-dir <path>  # 指定运行目录复用数据
python analysis/strategy/test_discord_notification.py           # 测试 Discord Webhook
python analysis/walk_forward/run_walk_forward.py
python pipeline/build_factors.py
python pipeline/data_process.py
```

---

## 复合因子方法（composite_factor.py）

| 方法族 | 变体 | 说明 |
|--------|------|------|
| Beta 加权 | beta_m1/m2, beta_m3_N{5/10/20} | 单变量 OLS 斜率加权 |
| IC 加权 | ic_m1/m2, ic_m3_N{5/10/20} | Pearson IC 加权 |
| Rank_IC 加权 | rank_ic_m1/m2, rank_ic_m3_N{5/10/20} | Spearman Rank_IC 加权 |
| 排序加权 | rank_add, rank_mul | 截面排名求和 / 求积 |
| OLS 加权 | ols_m1/m2, ols_m3_M{5/10/20} | 多元回归加权 |
| PCA | pca_pc1/2/3 | 主成分分析 |

权重方法含义：
- **m1**：全期均值（oracle 基线，含前瞻偏误——仅供研究）
- **m2**：截至当期累计均值（无前瞻）
- **m3**：滚动窗口均值（无前瞻）

---

## 调仓日流水线（run_rebalance_day）

**目的：** 完整调仓日流水线——拉数据 → 构建因子 → 数据处理 → 复合因子 → 策略回测 → 生成 Excel 调仓报告 + Discord 通知。

**流水线：**
1. 流水线阶段（子进程或内联模式）
2. 加载复合因子和收益
3. 运行策略回测（`run_detailed_backtest`）
4. 确定调仓日期（历史 + 外推未来）
5. 生成 Excel 报告（`rebalance_day_report.xlsx`）
6. 发送 Discord 通知（绩效指标 + 当前持仓盈亏）

**输出：** `output/rebalance_day_YYYY-MM-DD_HHMMSS/`，包含：
- `data/` — 原始价格/成交量数据
- `factor_raw/` — 原始因子值
- `factor_processed/` — 处理后因子数据
- `composite_factor_reports/` — 复合因子结果
- `rebalance_day_report.xlsx` — 多 Sheet 调仓报告（Config、Opers、Returns 等）

**Discord 通知内容：**
- 因子选择、复合方法、策略参数
- 绩效指标：总收益、年化收益、夏普比率、最大回撤、Calmar 比率、胜率、盈亏比
- 当前持仓盈亏：与 Excel 中 MTM 对齐——权重、假设卖出价（As_Of 收盘或实时）、区间收益、卖出头寸
- 今日操作（买/卖清单，Weight ≥ 0.0001）：权重、价格、可选区间收益和持仓
- 下一调仓日

**流水线执行模式：**
- `--inline`：所有步骤在同一 Python 进程（推荐，更快，无子进程开销）
- 子进程（默认）：每步作为独立子进程（stdout/stderr 实时流式输出便于进度监控）

**运行目录约定：** 每次运行创建带时间戳的目录 `output/rebalance_day_YYYY-MM-DD_HHMMSS/`；`--run-dir` 可指定已有目录以复用数据。

---

## 配置约定

- 各模块独立配置：`config.py`（单因子）、`multi_factor_config.py`、`composite_config.py`、`strategy_config.py`、`strategy_review_config.py`、`walk_forward_config.py`
- 关键变量 `PROJECT_ROOT`：**必须保持一致**；项目根目录为 `D:\qqq`
- 通用路径变量：`PRICE_FILE`、`RETURN_COLUMN`、`FACTOR_FILE`、`OUTPUT_DIR`
- **DATA_START_OFFSET_DAYS**：数据起始日提前的交易日数
  - **配置位置：** `data/data_config.py`（直接设置，不支持环境变量覆盖）
  - **实现：** 在 `pull_yhfinance_Data.py` 中将 start_date 向回偏移 N 个交易日，以对齐因子和调仓日历
  - **按 offset 分子目录（避免覆盖）：** offset=0 使用默认路径；offset!=0 使用 `_offset{N}d` 后缀，如 `factor_raw_offset7d/`、`factor_processed_offset7d/`、`output/composite_factor_reports_offset7d/`、`output/strategy_reports_offset7d/` 等
- **因子选择机制（composite_config.py）：**
  - **优先级 1：** 环境变量 `REBALANCE_SELECTED_FACTOR_INDICES`（由 `run_rebalance_day.py` 设置，确保全流水线因子一致）
  - **优先级 2：** `MANUALLY_SELECTED_FACTOR_INDICES`（本文件人工配置，用于临时测试）
  - **推荐做法：** 长期因子变更——更新 `MANUALLY_SELECTED_FACTOR_INDICES` + 同步 `strategy_config.STRATEGY_SELECTED_FACTOR_INDICES`；临时测试——仅更新 `MANUALLY_SELECTED_FACTOR_INDICES`
- **已选因子参考：** `config/selected_factors_reference.py` 包含当前选中因子的完整代码和元数据，仅供人工参考；不导入任何代码

---

## 策略回顾（run_strategy_review）

**目的：** 完全自包含的策略回顾，无需先运行 `run_composite_factor`。基于配置的五个因子、复合方法和策略参数，自动从 `factor_processed` 读取 → 计算复合因子 → 运行策略回测 → 生成回顾报告。

**流水线：**
1. 从 `factor_processed` 加载五个配置因子（`SELECTED_FACTOR_INDICES`）
2. 按选定方法计算复合因子（`COMPOSITE_FACTOR_SHEET`）
3. 加载日频收益和价格数据
4. 获取基准数据（如 QQQ）
5. 运行策略回测（`STRATEGY_PARAM`）
6. 加载单因子文件（用于因子归因）
7. 参数敏感性分析（可选）
8. 券商记录对比（可选）
9. 写入 Excel 报告

**输出：** `output/strategy_review_YYYY-MM-DD_HHMMSS/strategy_review.xlsx`，包含 6 个 Sheet。

**用法：** 编辑 `analysis/strategy/strategy_review_config.py` 后运行：

```powershell
python analysis/strategy/run_strategy_review.py
```

**配置项：**

| 配置项 | 说明 | 示例 |
|--------|------|------|
| `SELECTED_FACTOR_INDICES` | 五个因子索引（由 composite_config 解析，优先级：1=环境变量 REBALANCE_SELECTED_FACTOR_INDICES，2=MANUALLY_SELECTED_FACTOR_INDICES） | `[32, 62, 65, 95, 101]` |
| `COMPOSITE_FACTOR_SHEET` | 复合方法 | `"ic_m3_N20"`, `"ols_m3_M5"` 等 |
| `STRATEGY_PARAM` | 策略参数 | `"max_return_5G_Top1_P10d"` |
| `LIVE_START_DATE` | 实盘开始日期 | `"2025-06-01"` 或 `None` |
| `BROKER_RECORDS_FILE` | 券商交易记录 | `"path/to/trades.csv"` 或 `None` |
| `RUN_PARAM_SENSITIVITY` | 运行参数敏感性 | `True` / `False` |
| `FACTOR_DIR` | 单因子目录覆盖 | `None`=默认 |
| `OUTPUT_DIR` | 输出目录覆盖 | `None`=带时间戳子目录 |

**前置条件：** 需要先运行 `pipeline/build_factors.py` 和 `pipeline/data_process.py` 生成 `factor_processed` 下的因子文件。

---

## 时间与对齐约定

- **因子值：** 调仓日 T 的截面（EOD，无前瞻）
- **收益：** 区间（T, T_next]，即 T+1 至下一调仓日（含）
- **交易：** T 收盘执行；T 日收益不计入当前持仓期
- **交易价格：** 使用 **Adj Close**；T 收盘价执行按 T 收盘价
- `RebalancePeriodManager` 和 `strategy_backtest._select_rebalance_dates` 按**交易日**间隔选择调仓日（如 P10 = 每 10 个交易日调仓一次）

---

## 命名与代码约定

- 配置类：`SingleFactorConfig`；配置模块：`*_config.py`
- 因子 Excel：`factor_alpha001_processed.xlsx` 等；Sheet 名称可为 `N5`、`N10`（观察期 N）
- 分组：升序分组，第 1 组 = 最小，第 10 组 = 最大；多空：做多 10+9，做空 1+2
- 资产配置方法：`equal`、`factor_weight`、`min_variance`、`mvo`、`max_return`、`factor_score`
- 中文注释和文档；英文变量名和 API

---

## 因子库（factors/factor_library.py）

**WorldQuant 101 Alpha 实现** — 实现了 Alpha #1 至 #101（去除了需要行业中性化的：#48、56、58–59、63、67、69–70、76、79–80、82、87、89–91、93、97、100）。所有函数均操作宽格式 DataFrame（index = 日期，columns = 标的代码）。

### 数据键（输入约定）

| 键 | 说明 |
|----|------|
| `close` | 复权收盘价（Adj Close） |
| `open` | 开盘价（Open） |
| `high` | 最高价（High） |
| `low` | 最低价（Low） |
| `volume` | 成交量（Volume） |
| `returns` | 日收益率 = `close.pct_change()` |
| `vwap` | 成交量加权均价 ≈ `(high+low+close)/3` |

### 辅助函数（宽格式 DataFrame 操作）

| 函数 | 说明 |
|------|------|
| `rank(df)` | 截面排名（每日所有标的排序，输出 0–1 百分位） |
| `delta(df, n)` | n 期差分：`df - df.shift(n)` |
| `delay(df, n)` | n 期滞后：`df.shift(n)` |
| `log(df)` | 自然对数 |
| `stddev(df, n)` | n 期滚动标准差 |
| `sma(df, n)` | n 期简单移动平均 |
| `ts_sum(df, n)` | n 期滚动求和 |
| `ts_min / ts_max` | n 期滚动最小值 / 最大值 |
| `ts_rank(df, n)` | 时间序列排名（过去 n 期中当前值的排名） |
| `ts_argmax / ts_argmin` | 滚动窗口内最大值 / 最小值的位置 |
| `correlation(df1, df2, n)` | 滚动 Pearson 相关系数（按列匹配） |
| `covariance(df1, df2, n)` | 滚动协方差（按列匹配） |
| `sign(df)` | 符号函数（−1, 0, 或 1） |
| `SignedPower(df, e)` | `sign(x) * |x|^e` |
| `scale(df, k)` | 截面缩放，使 `sum(|x|) = k` |
| `decay_linear(df, n)` | 线性加权移动平均（近期权重最大） |

### 可用 Alpha

| 范围 | Alpha |
|------|-------|
| #1–10 | 有符号幂 ArgMax、成交量-价格相关、开盘量相关、低价排名、VWAP 偏离、动量/反转信号 |
| #11–50 | VWAP 极值、成交量-价格协方差、价格加速度、线性衰减组合、多因子复合 |
| #51–101 | 高级动量/成交量/VWAP 组合、类 PCA 排名乘积、中间价信号 |

每个 Alpha 在 `FACTOR_DESCRIPTIONS` 中有文档，包含：名称、理论、方向（多空偏误）、典型持仓周期和分类。

---

## 数据处理流水线（pipeline/）

### 第一步 — 因子构建（`pipeline/build_factors.py`）
- 从 `data/us_top100_daily_2023_present.xlsx` 读取 OHLCV 数据
- 衍生 `returns = close.pct_change()` 和 `vwap = (high+low+close)/3`
- 应用每个 `FACTOR_CONFIGS` 函数生成原始因子
- 输出：`factor_raw/factor_alpha{XXX}_raw.xlsx`（每个因子一个文件，单个 Sheet 名为 "factor"）

### 第二步 — 数据处理（`pipeline/data_process.py`）
- 去极值：按日期列截断异常值（默认：1%–99% 分位）
- 标准化：按日期列 Z-score（均值=0，标准差=1）
- 输出：`factor_processed/factor_alpha{XXX}_processed.xlsx`

---

## 组合优化器（analysis/strategy/portfolio_optimizer.py）

五种资产配置方法，均通过 `compute_weights()` 统一调用：

| 方法 | 说明 |
|------|------|
| `equal` | 等权配置：1/N |
| `factor_score` | 因子值打分：组内因子值归一化后作为权重 |
| `min_variance` | 最小方差组合：min w'Σw，约束 Σw=1，`0≤w≤max_weight` |
| `mvo` | 马科维兹最优（最大化夏普比率）：max (w'μ−rf)/√(w'Σw) |
| `max_return` | 最大化预期收益：max w'μ，约束 Σw=1，`0≤w≤max_weight` |

所有优化方法在数据不足或求解失败时自动降级为等权。协方差矩阵使用 Ledoit-Wolf 式对角正则化防止奇异。

---

## 调仓日历（analysis/strategy/rebalance_calendar.py）

调仓日选择的**唯一权威实现**。`get_rebalance_calendar()` 函数从 `factor_index` 中选取日期，使得相邻调仓日之间至少相隔 `rebalance_period_days` 个交易日（按 `ret_index` 计数）。

被以下模块使用：
- `strategy_backtest._select_rebalance_dates`
- `rebalance_manager.RebalancePeriodManager.get_rebalance_dates`
- `run_rebalance_day`（经由 `strategy_backtest`）

---

## 市值重估（MarkToMarket，analysis/strategy/strategy_utils.py → `MarkToMarket` 类）

对于未到期持仓（`Next_Rebalance_Date > As_Of` 或 `Sell_Price_Close` 缺失），报告执行市值重估：

| Sell_Price_Source | 含义 |
|-------------------|------|
| `假设市价(未到期)` | 下一调仓日尚未到期；价格 = As_Of 日期的 Adj Close |
| `假设市价(补全)` | 历史卖出价缺失；以最近可用 Adj Close 填充 |
| `到期收盘` | 下一调仓日 ≤ As_Of，有确认的历史出场价 |

MTM 分两轮执行——第二轮还通过 `patch_period_summary_from_mtm` 同步更新 `period_summary_df`。`MarkToMarket.apply()` 方法使用向量化掩码（无 `iterrows`）以保证性能。

---

## Discord 集成（analysis/strategy/rebalance/discord_notifier.py）

每次 `run_rebalance_day` 执行后发送 Discord 通知，包含：

**绩效指标：** 总收益、年化收益、夏普比率、最大回撤、Calmar 比率、胜率、盈亏比

**当前持仓盈亏：** 权重、假设卖出价（As_Of Adj Close 或实时价）、区间收益、卖出头寸——与 Excel 报告中的 MTM 对齐

**今日操作：** 买/卖清单，含权重、价格、区间收益和持仓名义值（仅显示 `Weight ≥ 0.0001`）

**下一调仓日：** 从调仓日历外推

实时价格获取：当本地价格 DataFrame 在 As_Of 日期缺少某标的时，通过 yfinance 获取（3 次重试，指数退避）。`_is_target_date_session_closed` 保护使用"target_date + 1 天 00:00 UTC"判断 bar 是否已结算后再回填。

---

## Walk-Forward 验证（analysis/walk_forward/）

带严格训练/测试分离的防过拟合验证系统：

**流水线：**
1. 生成滚动窗口（训练 + 测试，含间隔）
2. 仅使用训练期数据处理因子
3. 仅使用训练期 IC/beta 计算复合因子权重
4. 将固定权重应用到测试期
5. 在测试集上网格搜索策略参数

**防泄露关键机制：**
- 因子去极值/标准化仅使用训练期数据
- 复合因子权重仅使用训练期 IC/beta 统计量
- 组合优化仅使用历史收益（已在 `strategy_backtest` 中实现）

**配置（`walk_forward_config.py`）：**

| 参数 | 说明 |
|------|------|
| `TRAINING_WINDOW` | 训练窗口长度（交易日） |
| `TESTING_WINDOW` | 测试窗口长度 |
| `ROLL_FORWARD_STEP` | 窗口间滚动步长 |
| `TRAIN_TEST_GAP` | 训练与测试之间的间隔（避免前瞻） |
| `COMPOSITE_METHOD` | 复合加权方法（如 `beta_m3`） |
| `N_WINDOW` | 方法 3 的滚动窗口大小 |

---

## 常见陷阱与注意事项

1. **PROJECT_ROOT：** 统一为 `D:\qqq`；更改根路径时需更新所有配置文件
2. **Sheet 名称与复合因子：** `strategy_config.COMPOSITE_FACTOR_SHEET` 必须与 `composite_factors.xlsx` 中实际的 Sheet 名称一致
3. **调仓周期：** 系统按**交易日**选择调仓日（如 P10 = 每 10 个交易日）
4. **单因子多 Sheet：** 默认读取第一个 Sheet；可设置 `FACTOR_SHEET` 指定 Sheet
5. **收益列：** 优先使用 Excel 中的 `Return` 列，否则通过 `pct_change()` 计算
6. **依赖库：** 需要 `.venv` 中装有 pandas、numpy、scipy、sklearn、matplotlib、openpyxl、yfinance、requests 等
7. **因子选择：** `run_composite_factor.py` / `run_strategy.py` / `run_detailed_backtest_report.py` 的因子来源由 `composite_config._resolve_selected_factor_indices()` 决定（优先级：1=环境变量 `REBALANCE_SELECTED_FACTOR_INDICES`，2=`MANUALLY_SELECTED_FACTOR_INDICES`）；文件名中的因子后缀由 `composite_config.SELECTED_FACTOR_INDICES` 自动推导；切换因子需修改 `MANUALLY_SELECTED_FACTOR_INDICES`（临时）或同步 `strategy_config`（长期）
8. **Walk-Forward：** 训练/测试严格分离；因子处理和复合权重仅使用训练数据
9. **run_rebalance_day 策略名称：** 由 `TARGET_WEIGHT_METHOD`、`TARGET_GROUP_NUM`、`TARGET_RANK`、`TARGET_REBALANCE_DAYS` 生成
10. **数据加载性能：** `load_price_data`、`load_return_data` 使用一次性 `pd.concat` 避免碎片化告警
11. **修改 DATA_START_OFFSET_DAYS 后需重新运行流水线：** 必须重新执行 pull → build_factors → data_process → run_composite_factor
12. **run_strategy_review：** 完全自包含，无需先运行 `run_composite_factor`；因子由 composite_config 解析（见上文"因子选择机制"），直接配置 `MANUALLY_SELECTED_FACTOR_INDICES` 后运行即可
13. **BKNG 拆股导致历史价格不一致：** 对于经历过拆股的股票，yfinance 在不同拉取日期可能返回不同的历史复权价格（如 BKNG 于 2026-04-06 发生 1:25 拆股；4 月 1 日和 4 月 8 日拉取的价格可能恰好相差 25 倍）。这导致 `factor_raw` → `factor_processed` 中的 Z-score 截面被 BKNG 主导，造成不同独立运行之间复合因子、持仓和回测结果不同。解决方案：始终使用同一价格数据文件，或在 `data_config.py` 中设置固定的 `DATA_START_OFFSET_DAYS`，使两次运行引用相同的 offset 子目录
14. **run_rebalance_day Discord 持仓盈亏区块不显示：** `operations_df` 仅包含已完成持仓期的记录（Next_Rebalance_Date 已确认）；当前持仓的 Next_Rebalance_Date 为外推值，在回测中正常写入；`_get_holding_period_info` 强制对日期列执行 `pd.to_datetime` 转换以避免类型不匹配
15. **流水线子进程输出缓冲：** 默认子进程模式缓冲子进程 stdout/stderr；使用 `--inline` 以内联方式实时观察进度
16. **inspect_ols_weights.py：** 仅在复合因子 Sheet 名称为 `ols_*` 系列时有效；如果选择了 beta/IC/Rank_IC 复合方法，此脚本不产生输出
17. **run_rebalance_day 市值重估（MTM）：** 对于 `Next_Rebalance_Date` 仍在未来（或 `Sell_Price_Close` 缺失）的持仓，报告使用 As_Of 日期的 **Adj Close**（最近可用 ≤ As_Of）或 **yfinance 实时价**填充**假设的** `Sell_Price_Close`。`Period_Return`、`Sell_Value` 和 `Shares` 会重新计算；见列 **`Sell_Price_Source`**（`假设市价(未到期)` vs `到期收盘`）。未到期持仓的 `Period_Summary` 从 MTM 项目更新。已完成持仓期（next rebalance ≤ As_Of 且有历史出场价）保留历史出场价。MTM 分两轮执行——第二轮（current_ops 实时价更新后）还通过 `patch_period_summary_from_mtm` 同步 `period_summary_df`（现已正确传入完整 MTM 操作 DataFrame，而非仅传 current_ops）
18. **共享工具模块（strategy_utils.py）：** 集中管理的可复用函数——`load_price_data`、`load_composite_factor`、`parse_strategy_param`、`build_factor_suffix`（含 `default_indices` 参数用于可配置回退）、`filter_weight_lt`、`MarkToMarket` 类、`patch_period_summary_from_mtm`。避免 `run_rebalance_day.py`、`run_strategy.py`、`run_detailed_backtest_report.py` 等模块间的代码重复
19. **build_factor_suffix 统一：** 此前三处有重复实现（`strategy_utils.py`、`composite_config.py`、`run_composite_factor.py`）。现已统一——`strategy_utils.build_factor_suffix` 是规范实现，带 `default_indices` 参数；`composite_config` 和 `run_composite_factor` 通过 `sys.path` 注册后从该模块导入
20. **MarkToMarket Buy_Value 保护：** `MarkToMarket.apply()` 现在跳过 `Buy_Value` 和 `Weight` 均为 NaN 的行，防止生成无效的零值 MTM 条目
21. **_composite_from_weight_df 边界：** 新增 `total_w` 跟踪——某期无有效权重时，该行保留 NaN 而非静默设为零
22. **collect_live_prices_for_mtm 向量化：** 不再使用 iterrows；使用向量化掩码过滤未到期行，然后通过 `_get_price_for_symbols_vectorized` 批量本地价格查询。仅对本地价格缺失的标的触发 yfinance API 调用，减少冗余网络请求。判断依据为 `_is_target_date_session_closed`——使用"target_date + 1 天 00:00 UTC"判断 bar 是否已结算，而非检查当前挂钟时间。**修复前 bug：** 原代码使用 `_is_market_closed_now()`（UTC 21:00 阈值），导致盘中运行时（如 UTC 06:35）即使昨日 bar 已结算但缺少 Close 也不会回填。**修复后行为：**
    - 昨日已结算 + yfinance 返回 OHL 但缺少 Close → 触发回填
    - 昨日已结算 + yfinance 已有完整 Close → 跳过
    - 昨日仍在交易中（UTC 04:00–21:00 盘中）→ 跳过（安全保护）
    - 昨日为非交易日 → 自动回溯至前一交易日，逻辑相同
    - 今日盘中 → 跳过（`target_date >= today` 立即返回 False；今日 bar 不被污染）
23. **strategy_backtest.py 未使用导入（Bug 16）：** 移除了未使用的 `from datetime import timedelta`
24. **SimpleNamespace → dataclass（Bug 17）：** `run_composite_factor.py` 现在使用 `@dataclass CompositeBacktestConfig` 替代 `types.SimpleNamespace`，提供类型提示、IDE 自动补全和更清晰的意图
25. **DataFrame 切片赋值（Bug 18）：** `composite_factor.py` 使用 `result.iloc[:, :] = 0.0` 而非 `result[:] = 0.0` 进行显式位置赋值
26. **load_price_data pd.concat（Bug 19）：** 无需操作——`strategy_utils.load_price_data` 已使用一次性 `pd.concat` 进行高效拼接
27. **run_strategy_review fpath 作用域（Bug 20）：** 无需操作——f-string 中的 `fpath` 变量在其定义的循环迭代中正确作用域
28. **rebalance_manager available[-1]（Bug 21）：** 无需操作——调用点已有显式 `if len(available) == 0: continue` 保护
29. **循环变量命名（Bug 22）：** `composite_factor.py` 的 `_weighted_composite` 和 `_composite_from_weight_df` 将内部日期循环变量从 `d` 重命名为 `_date`，避免遮蔽外层引用并提高可读性
30. **P1 Bug 23 — `_run_single` 空组合保护：** `strategy_backtest._run_single` 新增 `if len(port_stocks) == 0: continue` 检查，避免目标组为空时产生 NaN 日收益
31. **P6 Bug 24 — 防御性列对齐：** `strategy_backtest._run_single` 在向量化持仓期收益计算中新增 `w_norm = w_norm[ret_port.columns]`，确保权重 DataFrame 列顺序与收益 DataFrame 完全一致，防止广播顺序风险
32. **P3 Bug 25 — MarkToMarket 死代码：** `strategy_utils.MarkToMarket.apply` 删除 `ops.loc[need_mtm & ~need_mtm, ...]` 恒假条件行
33. **P2 Arc-26 — 调仓日历统一实现：** 新建 `analysis/strategy/rebalance_calendar.py` 作为调仓日历唯一权威实现；`strategy_backtest._select_rebalance_dates` 委托至该模块；`rebalance_manager.RebalancePeriodManager.get_rebalance_dates` 导入使用，消除两处重复实现
34. **P5 Arc-27 — `iterrows` 向量化：** `run_detailed_backtest_report.run_detailed_backtest` 将 `for j, (date, row) in enumerate(period_df.iterrows())` 循环替换为向量化 pandas 批量操作，消除逐行迭代的性能瓶颈

---

## 参考文档

- `docs/NOTES_VS_CODE_CHECKLIST.md`：设计笔记 vs 代码实现检查清单
- `analysis/walk_forward/README.md`：Walk-Forward 验证系统（时间机制、防泄露、结果解读）

---

*本文档随项目演进持续更新。*
