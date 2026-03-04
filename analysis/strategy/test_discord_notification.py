"""
测试 Discord 通知功能
======================
快速测试 Discord webhook 是否正常工作。
"""

import sys
import io
import requests
from datetime import datetime

# 设置 UTF-8 输出（Windows 兼容）
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1478641216659652709/TRe7zHYv0x5AbYJMngnJbi1TbjUwXiOhIct-rze0wHFFYgi-Yqt320iGOCY4J1NUbq68"


def test_simple_message():
    """测试简单文本消息"""
    payload = {
        "content": f"✅ Discord 通知测试成功！\n时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    }

    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        response.raise_for_status()
        print(f"✅ 简单消息发送成功（状态码：{response.status_code}）")
        return True
    except Exception as e:
        print(f"❌ 简单消息发送失败：{e}")
        return False


def test_rebalance_day_notification():
    """测试调仓日通知格式"""
    embed = {
        "title": "🔔 调仓日提醒 - 今日需要操作",
        "description": (
            "**调仓日期：** 2026-03-04\n"
            "**策略：** mvo_5G_Top2_P10d\n"
            "**执行时间建议：** 美东时间 15:45-16:00（收盘前15分钟）\n\n"
            "**策略表现：**\n"
            "• 总收益率：15.23%\n"
            "• 年化收益率：18.45%\n"
            "• 夏普比率：1.85\n"
        ),
        "color": 0x00FF00,  # 绿色
        "fields": [
            {
                "name": "🔴 卖出操作 (3 只)",
                "value": "• AAPL: 10.5% @ $175.23\n• MSFT: 8.3% @ $420.15\n• GOOGL: 7.2% @ $142.80\n",
                "inline": False
            },
            {
                "name": "🟢 买入操作 (3 只)",
                "value": "• NVDA: 12.0% @ $880.50\n• META: 9.5% @ $485.20\n• TSLA: 8.0% @ $195.75\n",
                "inline": False
            },
            {
                "name": "📅 下一调仓日",
                "value": "2026-03-14",
                "inline": False
            }
        ],
        "footer": {
            "text": f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 策略：beta_m3_N10"
        }
    }

    payload = {"embeds": [embed]}

    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        response.raise_for_status()
        print(f"✅ 调仓日通知发送成功（状态码：{response.status_code}）")
        return True
    except Exception as e:
        print(f"❌ 调仓日通知发送失败：{e}")
        return False


def test_non_rebalance_day_notification():
    """测试非调仓日通知格式"""
    embed = {
        "title": "ℹ️ 非调仓日 - 无需操作",
        "description": (
            "**当前日期：** 2026-03-04\n"
            "**最近调仓日：** 2026-03-01\n"
            "**下一调仓日：** 2026-03-11\n\n"
            "今日无需操作，请等待下一调仓日。"
        ),
        "color": 0x808080,  # 灰色
        "footer": {
            "text": f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 策略：beta_m3_N10"
        }
    }

    payload = {"embeds": [embed]}

    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        response.raise_for_status()
        print(f"✅ 非调仓日通知发送成功（状态码：{response.status_code}）")
        return True
    except Exception as e:
        print(f"❌ 非调仓日通知发送失败：{e}")
        return False


if __name__ == "__main__":
    print("=" * 64)
    print("  Discord 通知测试")
    print("=" * 64)

    print("\n[测试 1] 简单文本消息...")
    test_simple_message()

    print("\n[测试 2] 调仓日通知（模拟）...")
    test_rebalance_day_notification()

    print("\n[测试 3] 非调仓日通知（模拟）...")
    test_non_rebalance_day_notification()

    print("\n" + "=" * 64)
    print("测试完成！请检查 Discord 群组是否收到 3 条消息。")
    print("=" * 64)
