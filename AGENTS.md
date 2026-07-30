# AGENTS.md

本文件是给 Codex、Claude、Cursor 等 coding agent 的项目协作说明。开始任何代码修改前，请先阅读 `README.md`，再按本文约定工作。

## 角色与沟通

- 默认以量化工程师视角理解问题，重点关注金融数据、因子构造、回测时间对齐、前瞻偏误、风险指标和实盘调仓可执行性。
- 使用简体中文回复。涉及代码改动时，结论要清楚说明改了什么、如何验证、还有哪些风险。
- 如果修改了功能、配置、运行流程、报告输出或重要目录结构，同步更新 `README.md` / `README_zh.md` 中对应说明。
- 不要把研究结论写成投资建议；保持为工程与回测分析结论。

## 项目概览

这是一个美股量化因子研究与策略回测系统，主流程为：

```text
yfinance data -> factor construction -> factor processing -> single/multi-factor testing -> composite factor generation -> strategy backtesting -> rebalance-day reporting
```

核心目录：

- `data/`: yfinance 行情拉取、快照逻辑与数据路径配置。
- `factor_pipeline/`: WorldQuant 101 风格 alpha 因子库、原始因子构建与因子处理流水线。
- `analysis/single_factor/`: IC、Rank_IC、分组、多空、多头、空头测试。
- `analysis/composite_factor/`: 复合因子构建、不同复合方法报告、OLS 权重检查。
- `analysis/strategy/`: 策略回测、详细报告、调仓日流程、策略复盘。
- `analysis/strategy/rebalance/`: 调仓操作、Discord 通知、报告、市值重估。
- `analysis/walk_forward/`: 防泄露 walk-forward 样本外验证。
- `notes_markdown/`: Notion 导出的研究/设计笔记。
- `docs/`: notes 与代码行为对照、设计检查清单。

## 常用命令

所有命令从项目根目录 `D:\qqq` 运行：

```powershell
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

实盘/调仓日流程：

```powershell
python analysis/strategy/run_rebalance_day.py --inline
python analysis/strategy/run_rebalance_day.py --skip-pipeline
python analysis/strategy/run_rebalance_day.py --skip-pull
python analysis/strategy/run_rebalance_day.py --no-discord
python analysis/strategy/run_rebalance_day.py --run-dir <existing_run_dir>
```

Walk-forward 验证：

```powershell
python analysis/walk_forward/run_walk_forward.py
```

## 配置与路径约定

- `data/data_config.py` 是价格文件、offset 路径、因子目录和输出目录的核心入口。
- `DATA_START_OFFSET_DAYS=0` 使用默认目录；非 0 时使用 `_offset{N}d` 后缀目录和文件。
- `run_rebalance_day.py` 会通过 `REBALANCE_OFFSET_DAYS` 向子进程传播 offset。
- `analysis/composite_factor/composite_config.py` 中因子选择优先级为：
  1. `REBALANCE_SELECTED_FACTOR_INDICES`
  2. `MANUALLY_SELECTED_FACTOR_INDICES`
- `analysis/strategy/strategy_config.py` 中的 `STRATEGY_SELECTED_FACTOR_INDICES` 是策略侧配置。长期换因子时，要与 `composite_config.py` 同步。
- `config/selected_factors_reference.py` 只做人类可读参考，不应被当作 pipeline 的权威输入。
- 当前项目根路径在多个配置中写为 `D:\qqq`；如果迁移项目，优先检查所有 `PROJECT_ROOT`。

## 时间对齐与防泄露

这是本项目最重要的工程约束：

- 因子值日期为调仓日 T 的收盘截面。
- 交易执行按 T 收盘价。
- 收益区间为 `(T, T_next]`。
- `P10d`、`P20d` 等周期表示交易日，不是自然日。
- 历史和未来调仓日期必须使用统一交易日语义；未来日期使用 NYSE 日历，不能只按周末/联邦假日推断。
- `m1` 复合方法使用全样本统计，是 research/oracle baseline，默认视为有前瞻偏误。
- Walk-forward 中训练期、测试期必须严格隔离：因子处理、IC/beta/OLS 权重、组合优化都只能使用当时可获得的数据。

## 代码修改原则

- 优先复用已有模块和工具函数，尤其是 `analysis/strategy/strategy_utils.py` 中的路径、suffix、加载、过滤和市值重估工具。
- 不要让同一逻辑在 composite、strategy、rebalance 三处各写一份；已经集中到工具函数的逻辑继续沿用。
- 处理 Excel 数据时尽量保留现有 sheet 名、列名、日期索引语义，避免破坏历史输出兼容性。
- 生成目录包括 `factor_raw*`, `factor_processed*`, `output/*`，通常不应手动提交或大规模改动。
- 临时 `_debug_*.py`、`_test_*.py` 仅用于诊断；新增长期入口脚本应写入 README。
- 如果涉及 yfinance 数据，注意公司行动会改变历史复权价格。BKNG 1:25 拆股曾导致 z-score、分组、持仓和回测结果明显变化。
- 对比不同运行结果时，优先确认是否使用同一个价格文件、同一 offset、同一因子集合、同一复合 sheet、同一调仓周期。

## 验证清单

改动后根据影响范围选择验证：

- 数据/因子路径：运行 `python data/pull_yfinance_data.py`、`python factor_pipeline/build_factors.py`、`python factor_pipeline/process_factors.py`。
- 单因子或多因子指标：运行 `python analysis/single_factor/run_multi_factor_test.py` 和相关单因子脚本。
- 共线性：运行 `python analysis/single_factor/run_collinearity_analysis.py`。
- 复合因子：运行 `python analysis/composite_factor/run_composite_factor.py`。
- 策略回测：运行 `python analysis/strategy/run_strategy.py` 或 `python analysis/strategy/run_detailed_backtest_report.py`。
- 调仓日：优先运行 `python analysis/strategy/run_rebalance_day.py --inline --no-discord` 做本地验证。
- 过拟合/稳健性：运行 `python analysis/walk_forward/run_walk_forward.py`。

如果不能完整运行，至少说明未运行的原因、已检查的文件、以及剩余风险。

## 文档来源

开始分析时优先读：

- `README.md` / `README_zh.md`: 项目地图、流程、命令和核心约定。
- `notes_markdown/notion_notes_ZH.md`: 从 Notion 导出的研究设计、实盘策略与待办记录。
- `docs/NOTES_VS_CODE_CHECKLIST.md`: notes 与当前代码行为的对照。
- `analysis/walk_forward/README.md`: walk-forward 防泄露验证细节。
