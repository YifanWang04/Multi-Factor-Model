# Walk-Forward Validation System

## 概述

Walk-forward验证系统用于评估量化策略的稳健性，通过滚动窗口的方式在多个时间段上测试策略表现，防止过拟合。

## 核心特性

1. **防止信息泄露**
   - 因子处理只使用训练期数据
   - 复合因子权重只使用训练期IC/beta
   - 投资组合优化只使用历史收益

2. **全面的参数评估**
   - 测试120种策略组合（2组数 × 3目标排名 × 4调仓周期 × 5权重方法）
   - 识别稳健参数 vs. 时期特定参数
   - 验证选定策略是否真正最优

3. **保守的时间窗口**
   - 训练窗口：400天（~1.6年）
   - 测试窗口：60天（~3个月）
   - 滚动步长：30天（~1.5个月）
   - 约10个walk，覆盖~600天的样本外数据

4. **丰富的分析报告**
   - 参数稳定性指标（平均Sharpe、标准差、胜率）
   - 稳健策略识别（top 10最一致的策略）
   - 参数敏感性分析（哪些参数最重要）
   - Walk对比（时期难度检测）
   - 全面可视化（热力图、散点图、箱线图）

## 文件结构

```
analysis/walk_forward/
├── __init__.py                    # 包初始化
├── walk_forward_config.py         # 配置参数
├── rolling_data_processor.py      # 防泄露数据处理
├── walk_forward_engine.py         # 核心验证引擎
├── walk_forward_analyzer.py       # 结果分析器
├── run_walk_forward.py            # 主程序入口
├── test_engine.py                 # 快速测试脚本
└── README.md                      # 本文件
```

## 使用方法

### 1. 配置参数

编辑 `walk_forward_config.py`：

```python
# 时间窗口参数
TRAINING_WINDOW = 400  # 训练窗口（天）
TESTING_WINDOW = 60    # 测试窗口（天）
ROLL_FORWARD_STEP = 30 # 滚动步长（天）

# 复合因子配置
SELECTED_FACTOR_INDICES = [20, 16, 43, 17, 34]  # 选择的因子索引
COMPOSITE_METHOD = "beta_m3"  # 复合方法
N_WINDOW = 10  # 滚动窗口大小

# 策略网格搜索参数
GROUP_NUMS = [5, 10]
TARGET_GROUP_RANKS = [1, 2, 3]
REBALANCE_PERIODS = [10, 20, 30, 60]
WEIGHT_METHODS = ["equal", "min_variance", "mvo", "max_return", "factor_score"]
```

### 2. 运行验证

```bash
# 从项目根目录运行
python analysis/walk_forward/run_walk_forward.py
```

### 3. 查看结果

输出文件位于 `output/walk_forward_reports/`：

- `parameter_stability.xlsx` - 所有120种参数组合的稳定性指标
- `robust_strategies.xlsx` - Top 10最稳健的策略
- `parameter_sensitivity.xlsx` - 参数敏感性分析
- `walk_comparison.xlsx` - 各个walk的表现对比
- `visualizations/` - 可视化图表
  - `sharpe_heatmap.png` - Sharpe热力图
  - `stability_scatter.png` - 稳定性散点图
  - `parameter_boxplots.png` - 参数箱线图
  - `walk_performance.png` - Walk表现对比

## 关键概念

### Walk-Forward vs. 传统回测

**传统回测：**
- 使用全部历史数据
- 一次性测试
- 容易过拟合

**Walk-Forward：**
- 滚动窗口测试
- 多个独立测试期
- 评估时间稳定性

### 时间对齐约定

```
训练期: [T0, T_train]
  ↓ 计算复合因子权重（只用训练期数据）
  ↓ 最后调仓日: T_train或更早
  ↓ IC/beta使用截至T_train的收益

测试期: (T_train, T_test]
  ↓ 第一个调仓日: T_train之后的第一个因子日期
  ↓ 使用训练期学到的固定权重
  ↓ 投资组合权重每次调仓重新计算（只用历史收益）
  ↓ 收益从调仓日+1开始计算
```

### 防泄露机制

**三层防护：**

1. **因子处理层**
   ```python
   # 错误：处理全部数据
   all_data = pd.read_excel("factor_raw/alpha001.xlsx")
   processed = mad_winsorize(all_data)  # 使用全局分布

   # 正确：只处理训练期数据
   train_data = all_data.loc[:train_end]
   processed = mad_winsorize(train_data)  # 只用历史分布
   ```

2. **复合因子权重层**
   ```python
   # 错误：使用全部数据计算IC
   ic_series = compute_ic(factor, returns)  # 包含未来收益
   weights = ic_series.mean()

   # 正确：只使用训练期数据
   train_ic = compute_ic(factor.loc[:train_end], returns.loc[:train_end])
   weights = train_ic.mean()
   ```

3. **投资组合优化层**
   ```python
   # 错误：使用测试期收益优化
   hist_returns = returns.loc[:test_end]  # 包含未来数据

   # 正确：只使用调仓日之前的收益
   hist_returns = returns.loc[returns.index < rebalance_date]
   ```

## 结果解读

### 参数稳定性指标

- **平均Sharpe > 1.0** - 优秀
- **Sharpe标准差 < 0.5** - 稳定
- **胜率 > 70%** - 一致盈利
- **一致性得分** - 综合指标（越高越好）

### 策略评估场景

**场景1：策略稳健**
- 选定策略排名top 10
- 平均Sharpe > 1.0，胜率 > 70%
- → 可以自信部署

**场景2：策略中等**
- 选定策略排名中间50%
- 平均Sharpe 0.5-1.0，胜率50-70%
- → 考虑切换到更稳健的参数

**场景3：策略过拟合**
- 选定策略排名后25%
- 平均Sharpe < 0.5，胜率 < 50%
- → 调查原因：MVO不稳定？5组太集中？P10d太频繁？

**场景4：所有策略不稳定**
- 没有参数组合胜率 > 60%
- 所有策略方差都很高
- → 复合因子(beta_m3_N10)或底层因子有问题
- → 考虑测试其他复合方法(ic_m3, rank_mul等)

## 计算成本

- 总回测数：~1200次（10 walks × 120 strategies）
- 预计运行时间：10-30分钟（取决于硬件）

## 注意事项

1. **数据要求**
   - 至少需要 TRAINING_WINDOW + TESTING_WINDOW 天的数据
   - 当前配置需要至少460天数据

2. **内存使用**
   - 每个walk会加载和处理因子数据
   - 建议至少8GB RAM

3. **并行化**
   - 当前版本串行执行
   - 未来可以并行化各个walk

4. **复合因子周期**
   - 复合因子固定为10天周期
   - 策略可以使用[10, 20, 30, 60]天周期（10的倍数）

## 扩展功能

### 测试其他复合因子

修改 `walk_forward_config.py`：

```python
COMPOSITE_METHOD = "ic_m3"  # 或 "rank_mul", "ols_m3", "pca_pc1"
```

### 调整时间窗口

```python
TRAINING_WINDOW = 252  # 更短的训练期（1年）
TESTING_WINDOW = 40    # 更短的测试期（2个月）
ROLL_FORWARD_STEP = 20 # 更频繁的滚动
```

### 自定义策略网格

```python
GROUP_NUMS = [10]  # 只测试10组
WEIGHT_METHODS = ["mvo", "min_variance"]  # 只测试优化方法
```

## 验证清单

实施后验证：

- [ ] 无前瞻偏误：训练数据不包含测试日期
- [ ] 时间对齐：因子在T → 收益从(T, T_next]
- [ ] 权重固定：复合因子权重在测试期不变
- [ ] 投资组合优化：只使用历史收益
- [ ] 可重复性：相同walk产生相同结果
- [ ] 一致性：walk-forward平均 ≈ 全期回测（如无过拟合）
- [ ] 稳定性：Sharpe标准差合理（不太高）
- [ ] 覆盖率：测试期覆盖足够的样本外数据（~600天）

## 技术支持

如有问题，请检查：

1. 配置文件是否正确
2. 数据文件是否存在
3. 因子文件是否完整
4. Python环境是否安装所需包

## 更新日志

### v1.0.0 (2026-03-03)
- 初始版本
- 支持beta_m3复合因子
- 120种策略组合网格搜索
- 完整的分析报告和可视化
