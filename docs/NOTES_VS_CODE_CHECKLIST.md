# Notes 与代码对照检查清单

本文档对照当前 notes 与代码实现，标注符合 / 部分符合 / 不符合项。

---

## 一、数据获取

| Notes 要求 | 代码实现 | 状态 |
|------------|----------|------|
| 标的池：美股 | `yhfinance_Data.py` 下载美股 ticker（约 100 只） | ✅ 符合 |
| 字段：开盘价、收盘价、最高价、最低价、成交量、**成交额(amt)、涨跌幅、换手率** | 仅从 yfinance 取 **Adj Close、Volume**；未写入 Open/High/Low；**未保存成交额、涨跌幅、换手率**（涨跌幅可由价格事后计算） | ⚠️ 部分符合 |
| 时间：**20年至今** 日频 | `start_date = "2023-01-01"`，仅为 **2023 至今** | ❌ 不符合 |
| 来源：yahoofinance / alpaca | 仅使用 yfinance，未接 alpaca | ⚠️ 部分符合 |

**建议**：若严格按 notes，需在 `yhfinance_Data.py` 中：① 将 `start_date` 改为约 2005 年；② 写入 Open/High/Low 并落盘；③ 若有数据源则增加成交额、换手率（yfinance 无换手率需另算或标注缺失）。

---

## 二、构造因子

| Notes 要求 | 代码实现 | 状态 |
|------------|----------|------|
| 输入：原始量价 Excel，观察期 N（如 2–60 天） | `build_factors.py` 读 `close_df`/`volume_df`，各因子有固定 N（如 N=20/60） | ✅ 符合 |
| 输出：**每个因子一个 Excel，不同 sheet 对应不同观察期 N** | 已实现：FACTOR_CONFIGS/VOLUME_REQUIRED_FACTORS 支持 `n_param`+`n_list`，每个 N 写入 sheet 名 `N5`、`N10` 等；无 n_list 的因子仍为单 sheet | ✅ 符合 |
| 量价因子：基于价格、涨跌幅、成交量、换手率 | 因子基于 close、pct_change、volume；**未使用换手率**（因数据层未提供） | ⚠️ 部分符合 |

**建议**：若需「每因子多 N 多 sheet」，需在 `build_factors.py` / `factor_library` 中对每个因子循环多个 N，并写入同一 Excel 的多个 sheet（如 sheet 名 `N5`、`N20`、`N60`）。

---

## 三、数据处理

| Notes 要求 | 代码实现 | 状态 |
|------------|----------|------|
| 输入：构造好的因子 Excel | `data_process.py` 读 `factor_raw/*.xlsx` | ✅ 符合 |
| 处理：**截面** 去极值、标准化（中性化已划掉） | 按行（每行=一日截面）处理 | ✅ 符合 |
| 去极值：**中位数 MAD**（median ± 3×1.4826×MAD） | `mad_winsorize(df, n=3)` 实现 | ✅ 符合 |
| 标准化：z-score | `zscore_standardize` 实现 | ✅ 符合 |
| 输出：处理后因子 Excel | 写入 `factor_processed/*_processed.xlsx` | ✅ 符合 |

均值方差去极值、分位数去极值在 notes 中为可选，当前仅实现 MAD，与「使用最多」一致。

---

## 四、单因子测试

### 4.1 输入与前置

| Notes 要求 | 代码实现 | 状态 |
|------------|----------|------|
| 输入：处理好的因子 Excel、**return 当日收益率 Excel**、调仓周期、分层数、层内配置方式 | 已支持：config 中可配置 `RETURN_FILE` 与 `RETURN_COLUMN`；若配置且文件存在则从该 Excel 读收益率，否则仍由价格 `pct_change` 计算 | ✅ 符合 |
| 因子下移一行 / 调仓日前一天因子值 | `rebalance_manager.align_factor_return_by_period()` 用 **调仓日前一交易日** 因子值 | ✅ 符合 |
| 周期内累计收益率 | 调仓日到下一调仓日的 `(1+ret).prod()-1` | ✅ 符合 |
| 分组：十分层，因子值最高为组 10、最低为组 1 | `grouping.GrouperEnhanced`，`split()` 升序分组，组 1 最小、组 10 最大 | ✅ 符合 |
| 组内收益：等权或因子加权 | `WEIGHT_METHOD`: `equal` / `factor_weight` | ✅ 符合 |

### 4.2 Group IC

| Notes 要求 | 代码实现 | 状态 |
|------------|----------|------|
| 对 1–10 组与组内收益率做相关性分析（组号 vs 组收益） | 已恢复 `ic.calculate_group_ic()`，报告中含 Group IC 统计表与各周期 Group IC 时间序列图 | ✅ 符合 |

### 4.3 IC / Rank_IC 分析

| Notes 要求 | 代码实现 | 状态 |
|------------|----------|------|
| 遍历不同调仓周期，分别取 IC、Rank_IC | 按 `REBALANCE_PERIODS` 逐周期计算 | ✅ 符合 |
| 统计：均值、IR、偏度、峰度、t、p 值 | `ic.calculate_statistics()` | ✅ 符合 |
| 图：年度 IC 柱状图、月度 IC 色阶图、调仓日 IC 折线图、调仓日累计 IC | 年度柱状、月度热力图、IC 折线（含 30 日 MA）、累计 IC 曲线 | ✅ 符合 |

### 4.4 多空测试

| Notes 要求 | 代码实现 | 状态 |
|------------|----------|------|
| 买 10+9，卖 1+2 | `(group_returns[10]+group_returns[9])/2 - (group_returns[1]+group_returns[2])/2` | ✅ 符合 |
| 统计：年化收益、波动率、夏普、最大回撤、胜率、PnL | `PerformanceAnalyzer.calculate_metrics()` | ✅ 符合 |
| 图：净值折线图、月度收益率色阶图、**调仓日收益率折线图** | 已实现：多空/多头/空头均有调仓日收益率折线图；多空/多头/空头操作汇总表已重新写入 PDF | ✅ 符合 |

### 4.5 分层多头 / 空头

| Notes 要求 | 代码实现 | 状态 |
|------------|----------|------|
| 每组统计：年化、波动率、夏普、最大回撤、胜率、PnL | 各组 `PerformanceAnalyzer` 统计 | ✅ 符合 |
| 图：分组净值折线图、分组累计收益柱状图、调仓日收益率折线图 | 分组净值曲线、分层收益柱状图有；**调仓日收益率折线图** 未单独实现（有操作表但未入 PDF） | ⚠️ 部分符合 |

### 4.6 输出

| Notes 要求 | 代码实现 | 状态 |
|------------|----------|------|
| 单因子测试 PDF：IC 分析、Rank_IC、多空、多头、空头 | `report_generator` 按节输出上述内容 | ✅ 符合 |
| 写入本地 | 输出到 `output/single_factor_reports/` | ✅ 符合 |

---

## 五、汇总：已实现的项（与 notes 对齐）

- **构造因子**：每因子多 N 多 sheet 已实现（`factor_library` 的 `n_param`/`n_list`，`build_factors` 写入 `N5`、`N10` 等 sheet）。
- **Group IC**：已恢复并加入报告（统计表 + 各周期 Group IC 时间序列图）。
- **调仓日收益率折线图**：多空/多头/空头均已增加；操作汇总表已重新写入 PDF。
- **Return 输入**：config 支持 `RETURN_FILE`、`RETURN_COLUMN`，可选从 Excel 读收益率。

尚未实现（可选）：
- 数据获取：时间范围 20 年至今；字段成交额/换手率等。
- 单因子测试若使用多 sheet 因子文件，当前仍读第一个 sheet（`sheet_name=0`），可选增加 `FACTOR_SHEET` 配置指定 sheet。

---

*检查基于当前仓库代码与上述 notes 文本，如有 notes 更新可再对表更新。*
