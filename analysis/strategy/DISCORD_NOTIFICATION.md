# Discord 通知集成说明

## 功能概述

`run_rebalance_day.py` 已集成 Discord webhook 通知功能，每次运行后自动推送调仓日状态和操作建议。

## 通知类型

### 1. 调仓日通知（绿色）
当今日是调仓日时，推送包含：
- 调仓日期
- 策略参数（mvo_5G_Top2_P10d）
- 执行时间建议（美东时间 15:45-16:00）
- 策略表现（总收益率、年化收益率、夏普比率）
- 卖出操作清单（股票代码、权重、价格）
- 买入操作清单（股票代码、权重、价格）
- 下一调仓日

### 2. 非调仓日通知（灰色）
当今日不是调仓日时，推送包含：
- 当前日期
- 最近调仓日
- 下一调仓日
- 提示无需操作

## 使用方法

### 基本用法（发送通知）
```bash
# 完整 pipeline + Discord 通知
python analysis/strategy/run_rebalance_day.py

# 跳过数据拉取 + Discord 通知
python analysis/strategy/run_rebalance_day.py --skip-pull

# 跳过 pipeline，使用已有数据 + Discord 通知
python analysis/strategy/run_rebalance_day.py --skip-pipeline
```

### 禁用通知
```bash
# 不发送 Discord 通知
python analysis/strategy/run_rebalance_day.py --no-discord
```

## 测试通知

运行测试脚本验证 Discord webhook 是否正常工作：

```bash
python analysis/strategy/test_discord_notification.py
```

测试脚本会发送 3 条消息：
1. 简单文本消息
2. 调仓日通知（模拟）
3. 非调仓日通知（模拟）

## 配置

Discord webhook URL 配置在 `run_rebalance_day.py` 第 75 行：

```python
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/YOUR_WEBHOOK_URL"
```

如需更换 webhook，修改此变量即可。

## 建议的自动化流程

### 方案 1：每日定时运行（推荐）
```bash
# 每天美股收盘后（北京时间 05:00）运行
# Windows 任务计划程序 / Linux cron
python analysis/strategy/run_rebalance_day.py --skip-pull
```

优点：
- 每天自动检查是否调仓日
- 非调仓日也会收到提醒（下一调仓日）
- 数据更新可手动控制（避免 yfinance 限流）

### 方案 2：仅调仓日运行
手动查看 `Future_Rebalance_Dates` sheet，在调仓日当天运行：
```bash
python analysis/strategy/run_rebalance_day.py
```

优点：
- 减少不必要的运行
- 节省计算资源

## 执行建议

收到调仓日通知后：
1. **14:00 ET** - 收到 Discord 通知
2. **15:45 ET** - 开始执行交易（使用 Limit Order，价格设为当前市价 ±0.5%）
3. **16:00 ET** - 确保所有订单提交完成
4. **16:30 ET** - 检查成交情况，未成交订单改为 Market Order 或次日处理

## 故障排查

### 通知未发送
1. 检查网络连接
2. 验证 webhook URL 是否正确
3. 检查 Discord 服务器是否正常
4. 查看控制台错误信息

### 通知格式异常
1. 确保 pandas、numpy 版本兼容
2. 检查数据文件是否完整
3. 查看 `rebalance_day_report.xlsx` 是否正常生成

### 中文乱码（Windows）
脚本已自动处理 UTF-8 编码，如仍有问题：
```bash
# 设置环境变量
set PYTHONIOENCODING=utf-8
python analysis/strategy/run_rebalance_day.py
```

## 安全提示

⚠️ **Webhook URL 包含敏感信息，请勿公开分享！**

如需分享代码：
1. 将 webhook URL 移至环境变量或配置文件
2. 添加 `.env` 到 `.gitignore`
3. 使用示例配置文件（不含真实 URL）

## 更新日志

- **2026-03-04**: 初始版本，支持调仓日/非调仓日通知
