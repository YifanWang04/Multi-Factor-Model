# 量化因子研究与策略回测系统

本项目是一套美股量化研究系统，覆盖完整因子流程：

`yfinance 数据 -> 因子构建 -> 因子处理 -> 单因子/多因子测试 -> 复合因子生成 -> 策略回测 -> 调仓日报告`

系统基于约 100 只美股的日频 OHLCV 数据、WorldQuant 风格 Alpha 因子、Excel/PDF 报告、Walk-Forward 验证，以及可选的 Discord 调仓通知。

## 快速开始

所有命令都在项目根目录运行：

```powershell
cd D:\qqq
.\.venv\Scripts\Activate.ps1
```

常规研究流程：

```powershell
python data/pull_yfinance_data.py
python factor_pipeline/build_factors.py
python factor_pipeline/process_factors.py
python analysis/single_factor/run_multi_factor_test.py
python analysis/single_factor/run_collinearity_analysis.py
python analysis/composite_factor/run_composite_factor.py
python analysis/strategy/run_strategy.py
```

性能模式：

```powershell
$env:QQQ_MAX_WORKERS = "8"        # 设为 "1" 可强制串行
python factor_pipeline/build_factors.py
python factor_pipeline/process_factors.py
python analysis/strategy/run_strategy.py
python analysis/composite_factor/run_composite_factor.py
python analysis/single_factor/run_multi_factor_test.py
```

耗时较重的研究循环会读取 `QQQ_MAX_WORKERS` 做进程级并行，并将 Excel 派生的
DataFrame 缓存在 `output/cache/`。如需禁用缓存，设置 `QQQ_DISABLE_CACHE=1`；
如需改缓存目录，设置 `QQQ_CACHE_DIR=<path>`。
`factor_pipeline/build_factors.py` 会原子写入每个因子工作簿，并对 Windows 下的瞬态 Excel I/O
错误最多重试三次；持续写入失败会终止本次运行，同时保留上一次有效工作簿。
`factor_pipeline/process_factors.py` 对处理后因子工作簿采用相同的原子写入与重试策略；并行运行时
只在派发前读取一次参考交易日期。若某个因子持续失败，命令会明确失败退出，
不会静默沿用旧的处理结果。
`run_strategy.py` 的统计汇总 sheet 会保留全部参数组合，但日收益和累计收益
时间序列 sheet 默认只导出 Sharpe 排名前 `REPORT_TIMESERIES_TOP_N` 的策略。
如需导出全部策略时间序列，可把该配置设为 `None` 或 `0`。

实盘/调仓日流程：

```powershell
python analysis/strategy/run_rebalance_day.py --inline
python analysis/strategy/run_rebalance_day.py --skip-pipeline
python analysis/strategy/run_rebalance_day.py --skip-pull
python analysis/strategy/run_rebalance_day.py --no-discord
python analysis/strategy/run_rebalance_day.py --run-dir <existing_run_dir>
```

直接运行 `run_composite_factor.py` 时，会按
`analysis/composite_factor/composite_config.py::COMPOSITE_REBALANCE_PERIODS`
生成研究用的多个调仓周期。由 `run_rebalance_day.py` 调用时，调仓日流程会用
`STRATEGY_PARAM` 解析出的 active strategy 周期覆盖该列表，因此实盘流程只生成与
策略匹配的 P 周期复合因子。

其他常用入口：

```powershell
python analysis/single_factor/run_single_factor_test.py
python analysis/single_factor/run_batch_single_factor_tests.py
python analysis/composite_factor/inspect_ols_weights.py
python analysis/strategy/run_detailed_backtest_report.py
python analysis/strategy/run_strategy_review.py
python analysis/strategy/test_discord_notification.py
python analysis/walk_forward/run_walk_forward.py
python analyze_report.py [rebalance_day_report.xlsx]
python backfill_close.py
```

## 环境

主要依赖：

- Python 3
- pandas, numpy
- scipy, scikit-learn
- matplotlib, seaborn
- openpyxl
- yfinance
- requests
- pandas_market_calendars

本地虚拟环境默认位于 `.venv/`。

## 项目结构

```text
qqq/
├── qqq_core/                    # 共享路径、运行上下文、Excel I/O、命名与参数解析
├── qqq_config/                  # 策略 profile、股票池和实盘配置权威来源
├── data/                         # 行情配置、快照逻辑和 yfinance 拉取脚本
├── factor_pipeline/              # WorldQuant 因子库、原始因子构建和处理
├── config/                       # 已选因子的人类可读参考
├── analysis/
│   ├── single_factor/            # IC、Rank_IC、分组、单/多因子报告
│   ├── composite_factor/         # 复合因子方法和 OLS 权重检查
│   ├── strategy/                 # 策略回测、报告、复盘、调仓日
│   │   └── rebalance/            # 调仓操作、Discord、报告、市值计算
│   └── walk_forward/             # 防泄露滚动训练/测试验证
├── factor_raw*/                  # 生成的原始因子 Excel
├── factor_processed*/            # 生成的处理后因子 Excel
├── output/                       # 报告和带时间戳的运行目录
│   ├── research/                 # 单因子、多因子、复合因子、Walk-Forward 报告
│   ├── strategy/                 # 策略回测、详细报告、复盘报告
│   └── rebalance_runs/           # 新调仓日运行目录
├── tools/                        # 长期工具脚本；根目录保留兼容 wrapper
├── docs/                         # 设计/代码检查清单
├── notes_markdown/               # 从 Notion 导出的研究笔记
├── World Quant 101 Factors.pdf   # 因子参考文档
├── analyze_report.py             # tools/analyze_report.py 的兼容 wrapper
├── backfill_close.py             # tools/backfill_close.py 的兼容 wrapper
└── README.md / README_zh.md
```

临时 `_debug_*.py` 和 `_test_*.py` 文件用于近期调仓日、盘中场景等诊断。

## 流程与输出

| 阶段 | 脚本 | 主要输入 | 主要输出 |
| --- | --- | --- | --- |
| 数据拉取 | `data/pull_yfinance_data.py` | yfinance | `data/us_top100_daily_2023_present*.xlsx` |
| 因子构建 | `factor_pipeline/build_factors.py` | OHLCV Excel | `factor_raw*/factor_alphaXXX.xlsx` |
| 因子处理 | `factor_pipeline/process_factors.py` | 原始因子 | `factor_processed*/factor_alphaXXX_processed.xlsx` |
| 单因子测试 | `analysis/single_factor/run_single_factor_test.py` | 单个处理后因子 | PDF 报告 |
| 批量单因子测试 | `analysis/single_factor/run_batch_single_factor_tests.py` | 处理后因子目录 | 多个 PDF 报告 |
| 多因子测试 | `analysis/single_factor/run_multi_factor_test.py` | 已选因子和收益 | Excel 报告 |
| 共线性分析 | `analysis/single_factor/run_collinearity_analysis.py` | 已选因子 | Excel 矩阵和序列 |
| 复合因子 | `analysis/composite_factor/run_composite_factor.py` | 处理后因子 | `composite_factors_fXX-...xlsx` 和报告 |
| 策略回测 | `analysis/strategy/run_strategy.py` | 复合因子 | `strategy_backtest_report.xlsx` |
| 详细策略报告 | `analysis/strategy/run_detailed_backtest_report.py` | 复合因子和策略参数 | 操作/收益明细工作簿 |
| 策略复盘 | `analysis/strategy/run_strategy_review.py` | 复盘配置 | 带时间戳的 `strategy_review.xlsx` |
| 调仓日 | `analysis/strategy/run_rebalance_day.py` | 全流水线或已有运行目录 | `rebalance_day_report.xlsx` 和 Discord 消息 |
| Walk-Forward | `analysis/walk_forward/run_walk_forward.py` | Walk-Forward 配置 | 报告、图表、稳定性分析 |

带 offset 的路径会使用后缀，例如 `factor_raw_offset7d/`、`factor_processed_offset7d/`、`output/research/composite_factor_offset7d/`、`output/strategy/backtest_offset7d/`、`output/rebalance_runs/YYYY-MM-DD_HHMMSS_<profile>_offset7/`。旧的 `output/composite_factor_reports*`、`output/strategy_reports*` 和 `output/rebalance_day_*` 目录仍可读取，用于兼容历史工作簿和运行目录。

## 核心概念

### 因子库

`factor_pipeline/factor_library.py` 实现 Alpha 1-101 风格因子，排除了需要行业中性化的公式。输入为宽表 DataFrame，行是日期，列是股票代码。

常用输入键：

| 键 | 含义 |
| --- | --- |
| `open`, `high`, `low`, `close` | OHLC 价格，其中 `close` 使用复权收盘价 |
| `volume` | 成交量 |
| `returns` | `close.pct_change()` |
| `vwap` | 近似 VWAP，`(high + low + close) / 3` |

当 yfinance 使用 `auto_adjust=False` 拉取数据时，原始工作簿会保留未复权 OHLC/Close/Adj Close，并通过 `Adj Close / Close` 比例派生 `Adj Open`、`Adj High`、`Adj Low`。因子构建默认使用历史兼容口径（`Open/High/Low` 为原始列，`close` 为复权收盘价）；如需启用拆股一致的复权 OHLC，可在 `data/data_config.py` 将 `FACTOR_USE_ADJUSTED_OHLC=True`，但这会改变历史回测结果。因子、价格和收益加载都会把价格工作簿 sheet 过滤到 `data/data_config.py` 当前解析出的股票池，避免旧文件里残留或实验性新增股票静默改变横截面股票池。

核心辅助操作包括截面排名、滞后、差分、滚动求和/最小/最大/排名、滚动相关/协方差、有符号幂、缩放和线性衰减。

### 因子处理

`factor_pipeline/process_factors.py` 按日期进行截面处理：

- 去极值，目前按日期做中位数 MAD 截断
- 按日期做 z-score 标准化
- 每个因子输出一个处理后 Excel 文件

策略侧假设处理后因子已经与收益使用同一交易日历对齐。

### 单因子与多因子测试

单因子测试计算分组表现、IC、Rank_IC、多空测试、纯多测试、纯空测试和图表。

多因子测试在 Excel 中汇总已选因子：

- IC、Rank_IC、group Rank_IC 的均值/IR/t 值/p 值
- 多空收益和夏普
- 纯多超额收益和夏普
- 累计 IC、纯多累计收益、纯多累计超额收益
- 全样本及近 3 个月、6 个月、1 年、2 年窗口的因子统计

### 复合因子方法

`analysis/composite_factor/composite_factor.py` 支持：

| 方法族 | 变体 | 说明 |
| --- | --- | --- |
| Beta 加权 | `beta_m1`, `beta_m2`, `beta_m3_N{N}` | 单因子 OLS 斜率加权 |
| IC 加权 | `ic_m1`, `ic_m2`, `ic_m3_N{N}` | Pearson IC 加权 |
| Rank_IC 加权 | `rank_ic_m1`, `rank_ic_m2`, `rank_ic_m3_N{N}` | Spearman Rank_IC 加权 |
| 排名加权 | `rank_add`, `rank_mul` | 截面排名求和/求积 |
| OLS 加权 | `ols_m1`, `ols_m2`, `ols_m3_M{M}` | 多元回归加权 |
| PCA | `pca_pc1`, `pca_pc2`, `pca_pc3` | 主成分作为复合因子 |

方法含义：

- `m1`：全样本均值。这是 oracle/research 基线，存在前瞻偏误。
- `m2`：截至当前日期的历史扩展均值。
- `m3`：最近 N 或 M 个窗口的滚动历史均值。

`m1` sheet 会刻意使用全样本统计，只能作为含前瞻偏误的 research/oracle baseline 做研究对比，不应用于实盘策略选择。

`inspect_ols_weights.py` 只适用于 `ols_*` 复合因子 sheet。

### 策略回测

策略由复合因子生成，核心是选择目标分组和权重方法。

`portfolio_optimizer.py` 支持的资产配置方法：

| 方法 | 含义 |
| --- | --- |
| `equal` | 等权 |
| `factor_score` | 归一化因子分数作为权重 |
| `min_variance` | 最小方差组合 |
| `mvo` | Markowitz 最大夏普组合 |
| `max_return` | 在约束下最大化预期收益 |

优化使用 SLSQP；当数据不足或求解失败时自动降级为等权。权重优化的历史收益窗口包含调仓日 T 收盘时已经可知的数据；实际持仓收益仍使用 `(T, T_next]`。协方差矩阵使用对角正则化降低奇异风险。

绩效报告包括年化收益、年化波动、夏普、胜率、盈亏比、最大回撤、最大/平均亏损持续期、Calmar 比率和单周期最坏回撤。单周期最坏回撤是所有调仓持仓区间内最差的一次区间内回撤。

### 绩效指标口径

所有标准绩效报告都以收益率序列为权威输入。总收益按 `prod(1 + returns) - 1` 复利计算；年化收益按 `(1 + total_return) ** (periods_per_year / n) - 1` 计算；年化波动使用 `ddof=1` 的样本标准差；夏普使用年化收益减年化无风险利率再除以年化波动；最大回撤和 Calmar 都包含隐含初始净值 `1.0`。

单因子和复合因子的 long 指标基于调仓期收益，年化频率为 `252 / rebalance_period`。策略回测、详细报告、调仓日报告、策略复盘和 Walk-Forward 的全局绩效指标基于日频组合收益，年化频率为 252。调仓日报告中的 `Win_Rate` 和 `Profit_Loss_Ratio` 保持日收益口径；策略中的 `open_*` 指标保持调仓期/开仓期口径。

策略的亏损持续期按交易日计算：从净值的前一个历史高水位开始，到净值首次恢复或超过该水位为止。最大和平均亏损持续期都会将回测结束时尚未修复的水下区间统计到最后一个观测日；若净值从未跌破高水位，两项指标均为 0。

本次属于指标口径修正，不批量回填历史 Excel；重新运行对应脚本后，新报告会使用修正后的数值。

### 动态止盈止损研究

`run_strategy.py` 可以在同一次策略网格回测中比较两种退出方式：

- `fixed_rebalance`：现有固定调仓日卖出逻辑，作为 baseline，不参与 TP/SL 参数网格。
- `dynamic_tp_sl`：持仓期内每日用 adjusted close 检查单只股票是否触发动态止盈或止损；提前退出后该资金留现金，到下一次 rebalance 再重新分配。

`qqq_config/strategy_profiles.py` 中的 active profile 保存详细回测和调仓日流程默认使用的权重与退出参数：

```python
exit_policy = "fixed_rebalance"  # 或 "dynamic_tp_sl"
max_weight = 0.4
data_download_start_date = "2023-01-01"
preserve_price_scale = True
price_scale_base_run_dir = r"D:\qqq\output\rebalance_runs\2026-06-24_155408_strategy11_offset0"
tp_base = 0.08
sl_base = 0.05
tp_sl_probability = 1.0
```

`analysis/strategy/strategy_config.py` 中的研究网格控制 `run_strategy.py` 批量搜索：

```python
EXIT_POLICY_GRID = ["fixed_rebalance", "dynamic_tp_sl"]
TP_BASE_GRID = [0.04, 0.06, 0.08, 0.10, 0.12]
SL_BASE_GRID = [0.02, 0.03, 0.05, 0.07]
TP_SL_PROBABILITY = ACTIVE_PROFILE.tp_sl_probability
MAX_WEIGHT = ACTIVE_PROFILE.max_weight
MAX_WEIGHT_GRID = [0.4, 0.6, 1.0]
```

`MAX_WEIGHT` 从 active profile 派生，是详细回测和调仓日流程使用的默认标量；
调仓日配置/状态表和 Discord 摘要会显示该 profile 的权重上限。`run_strategy.py`
只会对优化型权重方法（`min_variance`、`mvo`、`max_return`）展开 `MAX_WEIGHT_GRID`，
多权重上限扫描时会把权重上限写入策略名和报表参数。

以 10 个交易日持仓周期为例，阈值为：

```text
TP = tp_base * (10 - TD) / 10 * P
SL = sl_base * (10 - TD) / 10 * P
```

v1 尚未训练机器学习分类器，因此 `P=1.0` 是中性默认值，不应解释为模型概率。详细报告和调仓日报告会输出 `Exit_Date`、`Exit_Reason`、`TP_Threshold`、`SL_Threshold`、`Signal_Probability` 等字段。

V1 实盘流程不要求每天运行 Python，也不会自动下单。active profile 使用 `dynamic_tp_sl` 时，调仓日报告会新增：

- `TP_SL_Schedule`：按当前持仓和下一次调仓日前的每个未来持仓交易日，预先列出 TP/SL 收益阈值和 TP/SL 价格。
- `TP_SL_Action_Checklist`：筛出最近几个需要人工设置价格提醒或临近收盘检查的日期和价格。

推荐流程是在调仓日一次性生成完整 schedule，持仓期间用券商、TradingView 或人工表格提醒；只有价格接近或触发 TP/SL 线时，再重新运行调仓日报告做确认和记录。

### 调仓日历与时间对齐

`analysis/strategy/rebalance_calendar.py` 是历史调仓日选择的唯一权威实现。

strategy profile 使用 `data_download_start_date` 表示 yfinance 的精确下载起始日。
它不固定首个交易日或后续 P5/P10/P20 日历相位；pipeline 完成后，调仓日从首个
可用因子/收益日期自然生成。字段设为 `None` 时回退到 `DATA_BASE_START_DATE`
与数据 offset 逻辑。

时间约定：

- 因子日期：调仓日 T，收盘后截面
- 交易执行：T 收盘
- 收益区间：`(T, T_next]`
- 交易价格：复权收盘价
- 默认情况下，`P10d` 等周期表示严格间隔 10 个交易日，不是 10 个自然日

profile 可以同时设置以下三个字段，显式启用固定星期调仓：

```python
rebalance_interval_weeks = 2
rebalance_weekday = 3              # ISO 星期：周一=1，周五=5
rebalance_week_anchor_date = "2026-06-24"
```

三个字段必须同时设置，并满足
`P{N}d == rebalance_interval_weeks * 5`；一个 profile 只允许一个 weekday。
以上示例表示 P10 每两周周三调仓。如果目标周三是 NYSE 休市日，则提前到最近一个
NYSE 交易日，因此名义 P10 区间可能只有 8 或 9 个实际交易日。固定模式的调仓期
收益按 `52 / rebalance_interval_weeks` 年化，策略日收益指标仍按 252 年化。
`Strategy111` 已启用该模式：每两周、周五（`rebalance_weekday=5`）调仓，
并以 `2026-06-26` 确定隔周相位。其他 profile 继续使用严格交易日模式。

未来调仓日外推通过 `pandas_market_calendars` 使用 NYSE 日历，因此 Good Friday 等非联邦但美股休市日会被正确处理。历史和未来日期选择使用一致的交易日计数语义。

## 配置

重要配置文件：

| 文件 | 用途 |
| --- | --- |
| `qqq_core/paths.py` | 项目根目录、offset 路径和输出目录布局的统一入口 |
| `qqq_core/run_context.py` | 一次运行的 profile、offset、run-dir 上下文 |
| `qqq_core/excel_io.py` | Excel sheet 校验、价格 workbook 读取、因子 sheet 读取和原子写入 |
| `qqq_core/strategy_params.py` | 因子后缀、复合因子文件名、策略参数解析和安全文件名标签 |
| `qqq_config/strategy_profiles.py` | active strategy profile、选定因子、复合 sheet、profile 下载起点、实盘策略参数的唯一权威配置源 |
| `qqq_config/ticker_universes.py` | strategy profile 与数据配置共用的命名股票池 |
| `data/data_config.py` | 数据覆盖起始日、直接拉取默认股票池、offset 路径 |
| `analysis/single_factor/config.py` | 单因子测试配置 |
| `analysis/single_factor/multi_factor_config.py` | 多因子测试配置 |
| `analysis/composite_factor/composite_config.py` | 已选因子和复合设置 |
| `analysis/strategy/strategy_config.py` | 复合 sheet、因子后缀、策略网格 |
| `analysis/strategy/strategy_review_config.py` | 自包含策略复盘配置 |
| `analysis/walk_forward/walk_forward_config.py` | Walk-Forward 窗口和网格 |

各配置中的 `PROJECT_ROOT` 由 `qqq_core.paths.ProjectPaths` 派生；只有迁移项目根目录时才需要显式设置 `QQQ_PROJECT_ROOT`。当前项目根目录是 `D:\qqq`。

### 数据 Offset

`data/data_config.py` 中的 `DATA_START_OFFSET_DAYS` 会将默认 yfinance 数据覆盖起点提前 N 个交易日，并隔离 offset 产物。调仓日运行时，只要 profile 的 `data_download_start_date` 非空，就以它作为精确下载起点；offset 仍控制产物命名，但不再移动这个 profile 日期。

规则：

- `offset=0` 使用默认文件和目录。
- `offset=N` 使用 `_offset{N}d` 文件和目录。
- offset 价格文件不再静默回退到基线价格文件。若请求 `offset=N` 但 `data/us_top100_daily_2023_present_offset{N}d.xlsx` 不存在，读取方会直接报错，避免不同交易日历被混用。
- offset 因子目录和复合因子目录也不再回退到基线目录。运行 offset 研究前必须生成匹配的 offset 因子和复合因子文件。
- offset 起始日回推使用 NYSE 交易日历，不再使用普通工作日。
- profile 下载起点优先于 `DATA_BASE_START_DATE`，且不会再传入调仓日历。
- `run_rebalance_day.py` 通过 `REBALANCE_OFFSET_DAYS` 将 offset 传给子进程。
- 修改 offset 后，需要重新运行 pull、因子构建、因子处理和复合因子生成。

### 因子选择

核心策略选择集中在 `qqq_config/strategy_profiles.py`。`analysis/strategy/strategy_config.py` 和 `analysis/composite_factor/composite_config.py` 默认从 active profile 派生选定因子、复合 sheet、策略参数和数据下载起点。临时切换 profile 可设置 `QQQ_STRATEGY_PROFILE=<profile_name>`；`REBALANCE_SELECTED_FACTOR_INDICES` 仍保留为调仓日 pipeline 子进程的运行时覆盖。在新 profile 尚未定稿、需要直接探索复合方法时，可在 `analysis/composite_factor/composite_config.py` 中设置 `COMPOSITE_RESEARCH_FACTOR_INDICES`。

每个 strategy profile 通过 `ticker_universe` 选择一套完整股票池。`ORIGINAL_108` 和 `ORIGINAL_143` 分别保留原始的 108、143 只股票池；`NASDAQ_100_LAST_6_YEARS` 是 2020-07-15 至 2026-07-15 快照期间所有纳斯达克 100 成分证券的 162 个 ticker 去重合集，包含期间已退出的成分；`ORIGINAL_108_PLUS_NASDAQ_100` 是与原始 108 股票池合并后的 235 个 ticker 去重合集，由 `Strategy12` 使用。`Strategy1`、`Strategy11`、`Strategy2` 使用 `ORIGINAL_108`，`Strategy3`、`Strategy4` 使用 `ORIGINAL_143`。

这份六年纳指股票池是静态研究全集，不是按日期生效的 point-in-time 成分表。若直接把它用于整个历史区间，会在部分股票正式加入纳斯达克 100 之前就将其纳入截面；它适合批量拉数和候选池研究，但严格复现历史成分的回测仍需按生效日期生成成分掩码。

直接运行 `data/pull_yfinance_data.py` 时使用 `data/data_config.py` 中的 `DATA_PULL_TICKER_UNIVERSE`，默认选择现有最大股票池 `ORIGINAL_108_PLUS_NASDAQ_100`（235 个 ticker），不受 `QQQ_STRATEGY_PROFILE` 影响。调仓日或其他调用方需要显式传入股票池，可调用 `pull_yfinance_data.main(ticker_universe=...)`，或设置 `REBALANCE_TICKER_UNIVERSE` / `YFINANCE_TICKER_UNIVERSE` 环境变量。`run_rebalance_day.py` 会自动把 active strategy profile 的 `ticker_universe` 传给 pipeline。拉取脚本启动时会打印解析后的 ticker universe、来源和 ticker 数量。

每个 strategy profile 可以单独设置 `preserve_price_scale=True`。同一个 profile 里的 `price_scale_base_run_dir` 用来固定价格口径基准 run；设为 `None` 时，会自动选择同 profile、同 offset 的上一份正式 run。`run_rebalance_day.py` 会自动把这些配置传给拉数脚本，日常不需要记环境变量。

复合因子选择按以下优先级解析：

1. `REBALANCE_SELECTED_FACTOR_INDICES`，由调仓日 pipeline 为单次运行设置
2. `analysis/composite_factor/composite_config.py` 中的 `COMPOSITE_RESEARCH_FACTOR_INDICES`，用于手动直接运行 `run_composite_factor.py`
3. `qqq_config/strategy_profiles.py` 中的 active profile

`run_rebalance_day.py` 会直接从 active strategy profile 派生因子编号和复合因子
sheet，并通过 `REBALANCE_SELECTED_FACTOR_INDICES` / `REBALANCE_SELECTED_COMPOSITE`
传给 pipeline，因此实盘调仓流程不会继承 `COMPOSITE_RESEARCH_FACTOR_INDICES`
或 `STRATEGY_RESEARCH_FACTOR_INDICES`。

新策略尚未定稿前，`analysis/strategy/strategy_config.py` 也支持 `run_strategy.py` 的研究覆盖：只设置 `STRATEGY_RESEARCH_FACTOR_INDICES` 和 `STRATEGY_RESEARCH_COMPOSITE_SHEET`。策略参数网格仍使用普通固定配置区的 `GROUP_NUMS`、`REBALANCE_PERIODS`、`TARGET_GROUP_RANKS`、`WEIGHT_METHODS`。研究覆盖保持为 `None` 时继续使用 active profile 默认值；实盘/调仓日前应恢复为 `None`。

`config/selected_factors_reference.py` 只是已选因子的人工参考，不会被流水线导入。

单因子测试通过 `SingleFactorConfig.FACTOR_SHEET` 指定读取的因子 sheet；测试多 sheet 因子文件时不要依赖默认第一个 sheet。

复合因子加载只有在主文件不存在时才回退到标准路径。若主文件存在但缺少目标 sheet 或无法读取，流程会直接失败，避免静默使用旧结果。

`analysis/composite_factor/composite_config.py` 的主周期 `REBALANCE_PERIOD` 会从 active strategy profile 的 `strategy_param`（`P{N}d`）派生。多周期研究请设置 `COMPOSITE_REBALANCE_PERIODS`，例如 `[5, 10]`；`run_composite_factor.py` 会写出 `composite_factors_P5_fXX-...xlsx`、`composite_factors_P10_fXX-...xlsx` 等 period-specific 工作簿。`run_strategy.py` 会按策略周期读取匹配的复合因子文件，避免 P5/P10 策略静默共用同一个复合因子日历。兼容用的 `composite_factors_fXX-...xlsx` 仍会为 active profile 主周期写出，供详细回测和调仓日流程使用。

## 调仓日报告

`analysis/strategy/run_rebalance_day.py` 可以运行完整流水线，也可以复用已有运行目录。

主要步骤：

1. 拉取数据、构建因子、处理因子、生成复合因子
2. 运行详细策略回测
3. 确定当前、上一期和下一期调仓日
4. 生成 `rebalance_day_report.xlsx`
5. 可选发送 Discord 通知

如果 active profile 的 `preserve_price_scale=True`，拉数阶段会把 profile 配置的基准 run 价格 workbook 作为口径：基准 run 截止日及以前的历史行保持不变，只追加 cutoff 之后的新行；若重叠区间检测到稳定的拆股式价格比例，且该比例与旧口径 `1.0` 的绝对差大于等于 `0.01`，才把新追加行乘回旧价格口径，并按反向比例调整 Volume。每次运行会在 run 目录写出 `data/price_snapshot_manifest.json`，记录实际使用的基准 workbook 和每个 ticker 的比例；最终 Excel 会输出 `Price_Scale_Adjustments` sheet，并在 `Rebalance_Config_Status` 显示 `Price_Scale_Config_Base_Run_Dir`、`Price_Scale_Base_Run`、`Price_Scale_Base_File`，所以不看环境变量也能知道配置基准和实际使用文件。

报告包含配置、当前操作、历史操作、收益序列、累计收益、周期汇总、当前持仓、市值重估字段和下一调仓日信息。`Rebalance_Config_Status` 会记录 `Requested_Data_Download_Start`、实际 `Data_Coverage_Start` 和 `Effective_Rebalance_Start`。为兼容历史 workbook，旧锚点行仍保留，但 profile 下载日期不再写入调仓锚点。`Period_Summary_2` 与 `Period_Summary` 列结构一致，只保留 `2026-03-27` 起的数据，并从该日重新累计 `Period_Cumulative_Return`。`Performance_By_Year` 输出自然年收益、年化波动率、Sharpe、最大回撤、是否为不完整年份，以及把该年收益设为零但保留原时间长度后的全期 CAGR。`Return_Attribution` 分别输出按调仓周期排名的对数收益贡献、按股票汇总的持有次数/平均权重/简单收益贡献/最近两年贡献占比，以及所有历史持仓股票的单股剔除压力测试。单股剔除采用“原权重转为现金、其余股票权重不放大、保留原组合交易成本”的口径。`dynamic_tp_sl` profile 还会输出 `TP_SL_Schedule` 和 `TP_SL_Action_Checklist`，用于提前设置人工/外部价格提醒。`Current_Operations_All` 不再输出；`All_Operations_All` 会过滤 `Weight < 0.01` 的行。

Discord 消息包含：

- 已选因子、复合方法、策略参数
- 绩效指标
- 与 Excel 市值重估逻辑一致的当前持仓盈亏
- 过滤微小权重后的今日买卖操作
- 下一调仓日

Discord 发送默认关闭，只有设置环境变量 `REBALANCE_DISCORD_WEBHOOK_URL` 后才会推送。Webhook URL 不应提交到仓库。

实时价格回退逻辑优先使用本地价格。若已完成的历史 bar 缺少 close，会通过 yfinance 重新拉取已完成日线；`fast_info.last_price` 不再写入历史 close 数据。

yfinance 拉取按股票代码隔离失败。单只股票返回空数据或抛出下载异常时，会输出兼容当前终端编码的 `[SKIP]` 提示并计入最终跳过汇总，其余股票继续执行；只有全部股票都被跳过时才终止，避免空价格工作簿进入因子流水线。

## 策略复盘

`analysis/strategy/run_strategy_review.py` 是自包含复盘工具，不需要先生成复合因子工作簿。

它会从 `factor_processed` 读取配置因子，计算选定复合因子，运行配置策略，可选对比券商记录，可选运行参数敏感性分析，并写入带时间戳的 `strategy_review.xlsx`。

配置文件是 `analysis/strategy/strategy_review_config.py`。

## Walk-Forward 验证

`analysis/walk_forward/` 使用滚动训练/测试窗口验证策略。

流程：

1. 生成滚动窗口
2. 仅用训练期数据处理因子
3. 仅用训练期 beta/IC 统计量计算复合权重
4. 将固定权重应用到测试窗口
5. 在测试数据上网格搜索策略参数
6. 汇总参数稳定性、敏感性、日收益、累计收益和每个 walk 的结果

这是在全样本探索研究之后，检查过拟合和参数稳定性的首选工具。

## 运行注意事项

- 对比不同运行结果时应使用同一个价格数据文件。公司行动可能导致 yfinance 不同日期拉取的历史复权价格不同；BKNG 1:25 拆股就是已知例子，会明显影响截面 z-score 和持仓。
- `run_rebalance_day.py --inline` 通常比子进程模式更快，也更容易观察进度。
- 路径和输出布局规则集中在 `qqq_core.paths.ProjectPaths`；新增脚本不要再手写新的 `output/...` 路径。
- 单次运行的 profile、offset、run-dir 路径集中在 `qqq_core.run_context.RunContext`；新的流程编排代码应使用这个 Interface，而不是重复解析环境变量。
- 因子后缀、复合因子工作簿名和策略参数解析集中在 `qqq_core.strategy_params`；`analysis/strategy/strategy_utils.py` 保留旧导入兼容。
- 调仓日报告会在当前操作中筛掉 `Weight < 0.0001` 的行，并在历史 `All_Operations_All` sheet 中筛掉 `Weight < 0.01` 的行，以减少噪音。
- `analysis/strategy/strategy_utils.py` 集中管理价格加载、复合因子加载、策略参数解析、因子后缀构建、小权重过滤和市值重估修补。
- `build_factor_suffix` 已统一由 `strategy_utils` 提供，并被复合/策略脚本复用。
- `MarkToMarket` 使用向量化掩码，跳过买入金额和权重都缺失的无效行，并为未到期持仓重算区间收益、卖出价值和股数。
- 如果某一期没有有效复合权重，复合因子该行保留缺失值，不会静默设为 0。
- 调仓日状态支持盘中运行：除非因子数据和日历状态支持，否则不会把未来日期误判为有效调仓日；若今天确实是调仓日，会用最近可用因子和实时/本地价格计算操作。
- 单因子和复合因子中基于调仓期收益的指标使用 `252 / rebalance_period` 年化；基于日收益的策略回测仍使用 252 个交易日年化。旧报告与新口径报告不可直接比较。
- 因子处理不再删除全空/全零原始因子；跳过记录会写入 `factor_processing_skip_manifest.csv`。
- `python analyze_report.py` 默认检查最新的标准调仓日报告；也可以传入某个 `rebalance_day_report.xlsx` 路径检查指定历史报告。

## 参考文档

- `docs/NOTES_VS_CODE_CHECKLIST.md`：设计笔记与代码行为检查清单
- `AGENTS.md`：本仓库的 coding agent 协作说明
- `analysis/walk_forward/README.md`：Walk-Forward 验证详细说明
- `notes_markdown/notion_notes_ZH.md`：中文研究/设计笔记

本 README 是项目的简明地图。生产调仓前，请以对应配置文件和脚本中的实际参数为准。

## 2026-06 路径与入口重构说明

- `qqq_core/paths.py` 统一管理项目根目录、offset 路径和输出目录；新增脚本不要再手写新的 `output/...` 路径。
- `qqq_core/run_context.py` 表示一次运行的 profile、offset 和 run-dir 上下文。
- `qqq_core/excel_io.py` 统一提供 Excel sheet 校验、价格 workbook 读取、因子 sheet 读取和原子写入。
- `qqq_core/strategy_params.py` 统一提供因子后缀、复合因子文件名、策略参数解析和安全文件名标签。
- `qqq_config/strategy_profiles.py` 是唯一策略 profile 配置源；`config/strategy_profiles.py` 只做兼容转发，不再维护第二份配置。
- 新研究输出写入 `output/research/`，策略输出写入 `output/strategy/`，新的调仓日运行目录写入 `output/rebalance_runs/YYYY-MM-DD_HHMMSS_<profile>_offsetN/`。
- 调仓日最终报告现在位于 run 目录的 `reports/rebalance_day_report.xlsx`；旧的 `output/rebalance_day_*` 目录仍可通过 `--run-dir` 复用。
- 根目录 `analyze_report.py`、`backfill_close.py` 已变为兼容 wrapper，长期实现位于 `tools/`。
