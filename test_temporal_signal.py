"""
向暂停中的 Workflow 发送审批信号（模拟人工审批）

用法：
  审批通过: uv run python test_temporal_signal.py
  审批拒绝: uv run python test_temporal_signal.py reject
"""

import asyncio
import sys
from dataclasses import dataclass

from temporalio.client import Client


@dataclass
class ApprovalResult:
    approved: bool
    comment: str


WORKFLOW_ID = "approval-TASK-001"


async def main():
    client = await Client.connect("localhost:7233")
    handle = client.get_workflow_handle(WORKFLOW_ID)

    # 先查看当前方案
    try:
        plan = await handle.query("get_plan")
        print(f"当前方案: {plan}\n")
    except Exception as e:
        print(f"查询失败（Workflow 可能还没启动）: {e}")
        return

    # 根据命令行参数决定审批结果
    if len(sys.argv) > 1 and sys.argv[1] == "reject":
        result = ApprovalResult(approved=False, comment="方案不够详细，请重新设计")
        print("发送信号: 审批拒绝")
    else:
        result = ApprovalResult(approved=True, comment="方案可行，同意执行")
        print("发送信号: 审批通过")

    await handle.signal("approve", result)
    print("信号已发送，Workflow 将继续执行")


if __name__ == "__main__":
    asyncio.run(main())
