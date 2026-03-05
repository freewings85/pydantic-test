"""
触发方：恢复一个等待中的 interrupt

用法：
  审核通过: uv run python test_interrupt_resume.py
  审核拒绝: uv run python test_interrupt_resume.py reject
"""

import asyncio
import sys

from temporalio.client import Client

from lib.interrupt import resume


async def main():
    client = await Client.connect("localhost:7233")

    if len(sys.argv) > 1 and sys.argv[1] == "reject":
        data = {"approved": False, "comment": "方案需要补充性能测试"}
        print("发送: 审核拒绝")
    else:
        data = {"approved": True, "comment": "方案可行，同意执行"}
        print("发送: 审核通过")

    await resume(client, "conversation_id1", data)
    print("信号已发送，发起方将继续执行")


if __name__ == "__main__":
    asyncio.run(main())
