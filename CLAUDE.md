# 项目上下文：量化因子研究与策略回测

> 本文档为 AI 助手（Claude）提供项目级上下文，便于更精准地理解代码结构与开发约定。

---

## 项目概述

本项目是一个**美股量化因子研究与多因子策略回测**系统，涵盖完整的数据获取 → 因子构建 → 数据处理 → 单因子/多因子/复合因子测试 → 策略构建与回测 全流程。

**技术栈：**
- Python 3 + pandas、numpy
- scipy、sklearn（回归、PCA、优化）
- matplotlib、seaborn（可视化）
- openpyxl（Excel 读写）
- yfinance（数据获取）

**数据：** 美股约 100 只标的，日频量价（2023 至今）；输出为 Excel 报表、PDF 报告。

---

## 目录结构

```
qqq/
├── data/                    # 原始数据
│   ├── us_top100_daily_2023_present.xlsx   # 日频量价主数据
│   └── pull_yhfinance_Data.py             # 从 yfinance 拉取数据
├── pipeline/                # 数据与因子构建流水线
│   ├── build_factors.py     # 从量价构建原始因子 → factor_raw/*.xlsx
│   └── data_process.py     # 去极值、标准化 → factor_processed/*.xlsx
├── analysis/
│   ├── single_factor/      # 单因子测试
│   ├── multi_factor/       # 多因子与复合因子
│   └── strategy/           # 策略构建与网格回测
├── factor_raw/             # 构建后的原始因子 Excel
├── factor_processed/       # 处理后的因子 Excel（去极值、标准化）
├── output/                 # 输出目录
│   ├── single_factor_reports/
│   ├── multi_factor_reports/
│   ├── composite_factor_reports/
│   └── strategy_reports/
├── docs/                   # 文档（notes 对照清单等）
└── claude.md               # 本文件
```

---

## 核心流程与入口

| 阶段 | 入口脚本 | 输入 | 输出 |
|------|----------|------|------|
| 数据获取 | `data/pull_yhfinance_Data.py` | yfinance API | `data/us_top100_daily_*.xlsx` |
| 因子构建 | `pipeline/build_factors.py` | 量价 Excel | `factor_raw/*.xlsx` |
| 数据处理 | `pipeline/data_process.py` | factor_raw | `factor_processed/*.xlsx` |
| 单因子测试 | `analysis/single_factor/run_single_factor_test.py` | config 指定因子+价格 | PDF 报告 |
| 批量单因子 | `analysis/single_factor/run_all_factors_backtest.py` | factor_processed 全量 | 多份 PDF |
| 多因子测试 | `analysis/single_factor/run_multi_factor_test.py` | multi_factor_config | Excel 报表 |
| 因子复合 | `analysis/multi_factor/run_composite_factor.py` | composite_config | composite_factors.xlsx + 回测报表 |
| 策略回测 | `analysis/strategy/run_strategy.py` | strategy_config | strategy_backtest_report.xlsx |

**运行约定：** 从项目根目录执行，例如：
```bash
python analysis/strategy/run_strategy.py
python pipeline/build_factors.py
```

---

## 配置约定

- 各模块有独立 config：`config.py`（单因子）、`multi_factor_config.py`、`composite_config.py`、`strategy_config.py`
- 关键变量 `PROJECT_ROOT`：**务必统一**，项目根路径为 `D:\qqq`，所有 config 与 debug 路径均已统一
- 常用路径变量：`PRICE_FILE`、`RETURN_COLUMN`、`FACTOR_FILE`、`OUTPUT_DIR`

---

## 时序与对齐约定

- **因子值：** 调仓日 T 当日截面（EOD，无前瞻）
- **收益：** (T, T_next] 区间，即 T+1 到下一调仓日（含）
- **交易：** T 日收盘执行，T 日收益不计入当期持仓
- `RebalancePeriodManager` 与 `strategy_backtest._select_rebalance_dates` 均按**日历天数**间隔选取调仓日

---

## 命名与编码规范

- 配置类：`SingleFactorConfig`；配置模块：`*_config.py`
- 因子 Excel：`factor_alpha001_processed.xlsx` 等，sheet 名可为 `N5`、`N10` 等对应观察期 N
- 分组：升序分组时组 1 最小、组 10 最大；多空：做多 10+9、做空 1+2
- 资产配置方式：`equal`、`factor_weight`、`min_variance`、`mvo`、`max_return`、`factor_score`
- 中文注释与文档为主，变量名与 API 保持英文

---

## 常见陷阱与注意事项

1. **PROJECT_ROOT：** 已统一为 `D:\qqq`，修改根路径时需同步更新各 config 文件及 debug.log 引用
2. **Sheet 名与复合因子：** `strategy_config.COMPOSITE_FACTOR_SHEET` 需与 `composite_factors.xlsx` 中实际 sheet 一致（如 `rank_mul`）
3. **调仓周期：** 系统按日历天数选调仓日；因子若为 10 交易日周期，则 <14 天等价于「每期必换」
4. **单因子多 sheet：** 默认读第一个 sheet，可配置 `FACTOR_SHEET` 指定 sheet
5. **收益率列：** 若 Excel 有 `Return` 列则直接使用，否则用 `pct_change()` 计算
6. **依赖：** 需 `.venv` 或相应虚拟环境，包含 pandas、numpy、scipy、sklearn、matplotlib、openpyxl、yfinance 等

---

## 参考文档

- `docs/NOTES_VS_CODE_CHECKLIST.md`：与设计 notes 的对照检查清单，标注实现符合/部分符合/不符合项

---

*本文档随项目演化更新，旨在帮助 AI 助手快速理解项目结构与约定。*
