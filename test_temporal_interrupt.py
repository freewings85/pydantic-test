"""
Temporal 实现 human-in-the-loop（类似 LangGraph interrupt）
场景：AI 生成方案 → 等待人工审批 → 审批通过则执行，拒绝则终止

使用方式：
  1. 先启动本脚本（Worker + 发起 Workflow），流程会暂停在审批环节
  2. 再运行 test_temporal_signal.py 发送审批信号，流程继续执行
"""

import asyncio
import sys
from datetime import timedelta
from dataclasses import dataclass

from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.worker import Worker, UnsandboxedWorkflowRunner


# --- 数据模型 ---

@dataclass
class Task:
    task_id: str
    description: str


@dataclass
class ApprovalResult:
    approved: bool
    comment: str


# --- Activities ---

@activity.defn
async def ai_generate_plan(task: Task) -> str:
    """AI 生成执行方案"""
    print(f"  [AI] 正在为任务 '{task.description}' 生成方案...")
    await asyncio.sleep(1)
    plan = f"方案：针对'{task.description}'，建议分3步执行：1.调研 2.实施 3.验证"
    print(f"  [AI] 方案已生成: {plan}")
    return plan


@activity.defn
async def execute_plan(plan: str) -> str:
    """执行已审批的方案"""
    print(f"  [执行] 开始执行方案...")
    await asyncio.sleep(1)
    print(f"  [执行] 方案执行完成")
    return f"执行完成: {plan}"


@activity.defn
async def notify_rejection(task: Task, comment: str) -> str:
    """通知方案被拒绝"""
    print(f"  [通知] 任务 {task.task_id} 被拒绝，原因: {comment}")
    return "已通知"


# --- Workflow ---

@workflow.defn
class ApprovalWorkflow:
    def __init__(self):
        self._approval: ApprovalResult | None = None
        self._plan: str = ""

    @workflow.signal
    async def approve(self, result: ApprovalResult):
        self._approval = result

    @workflow.query
    def get_plan(self) -> str:
        return self._plan

    @workflow.run
    async def run(self, task: Task) -> str:
        # 第 1 步：AI 生成方案
        self._plan = await workflow.execute_activity(
            ai_generate_plan,
            task,
            start_to_close_timeout=timedelta(seconds=30),
        )

        print(f"\n  ⏸ 流程暂停，等待人工审批...")
        print(f"  方案内容: {self._plan}")
        print(f"  请运行 test_temporal_signal.py 发送审批信号\n")

        # 第 2 步：interrupt — 等待人工审批
        await workflow.wait_condition(lambda: self._approval is not None)

        # 第 3 步：根据审批结果继续
        if self._approval.approved:
            print(f"  ✓ 审批通过: {self._approval.comment}")
            result = await workflow.execute_activity(
                execute_plan,
                self._plan,
                start_to_close_timeout=timedelta(seconds=30),
            )
            return f"已完成 - {result}"
        else:
            print(f"  ✗ 审批拒绝: {self._approval.comment}")
            await workflow.execute_activity(
                notify_rejection,
                task,
                self._approval.comment,
                start_to_close_timeout=timedelta(seconds=10),
            )
            return f"已拒绝 - {self._approval.comment}"


# --- 固定的 Workflow ID，方便 signal 脚本找到它 ---
WORKFLOW_ID = "approval-TASK-001"


async def main():
    client = await Client.connect("localhost:7233")

    # 调试模式下禁用 sandbox，避免与 debugpy 冲突
    is_debugging = "debugpy" in sys.modules
    worker = Worker(
        client,
        task_queue="approval-queue",
        workflows=[ApprovalWorkflow],
        activities=[ai_generate_plan, execute_plan, notify_rejection],
        **({"workflow_runner": UnsandboxedWorkflowRunner()} if is_debugging else {}),
    )

    async with worker:
        task = Task(task_id="TASK-001", description="优化数据库查询性能")

        print("=" * 60)
        print(f"发起任务: {task.description}")
        print(f"Workflow ID: {WORKFLOW_ID}")
        print("=" * 60)

        # 发起 Workflow（不等它完成）
        handle = await client.start_workflow(
            ApprovalWorkflow.run,
            task,
            id=WORKFLOW_ID,
            task_queue="approval-queue",
            task_timeout=timedelta(minutes=10),  # 调试用，默认 10 秒太短
        )

        # 等待 Workflow 完成（会一直等到收到 Signal）
        print("等待审批中... (运行 test_temporal_signal.py 来审批)")
        result = await handle.result()

        print("=" * 60)
        print(f"最终结果: {result}")


if __name__ == "__main__":
    asyncio.run(main())
