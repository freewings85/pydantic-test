"""
interrupt/resume 抽象层演示

步骤 1: 运行本脚本（发起方 + Worker）
    uv run python test_interrupt_demo.py

步骤 2: 另开终端，运行触发方
    uv run python test_interrupt_resume.py
    uv run python test_interrupt_resume.py reject
"""

import asyncio
from datetime import timedelta

from temporalio.client import Client

from lib.interrupt import interrupt, create_interrupt_worker


# --- callback：普通 async 函数，在本进程直接执行 ---

async def notify_reviewer(data: dict, interrupt_id: str) -> None:
    """模拟写出 stream event"""
    print(f"  [stream] AI event: 生成方案完成")
    print(f"  [stream] AI event: 方案内容 = {data.get('plan')}")
    print(f"  [stream] AI event: type=interrupt, interrupt_id={interrupt_id}")
    print(f"  [stream] AI event: 等待人工审核...")


async def main():
    client = await Client.connect("localhost:7233")

    # Worker 只处理 Workflow（不需要注册 activity）
    worker = create_interrupt_worker(client)

    async with worker:
        print("=" * 50)
        print("模拟 AI stream 输出...")
        print("  [stream] AI event: 开始分析任务")
        print("  [stream] AI event: 正在生成方案")

        # interrupt：先在本进程写出 event，再等待 resume
        response = await interrupt(
            client,
            key="conversation_id1",
            callback=notify_reviewer,
            data={"plan": "重构用户模块", "author": "AI Agent"},
            task_timeout=timedelta(minutes=10),
        )

        # resume 之后继续 stream
        print(f"  [stream] AI event: 收到审核结果")
        print("=" * 50)
        print(f"审核结果: {response}")

        if response.get("approved"):
            print(f"  [stream] AI event: 继续执行方案")
        else:
            print(f"  [stream] AI event: 方案被拒绝，终止")


if __name__ == "__main__":
    asyncio.run(main())
