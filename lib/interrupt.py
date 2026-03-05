"""
通用 interrupt/resume 抽象层

发起方:
    worker = create_interrupt_worker(client)
    async with worker:
        response = await interrupt(client, "chat-abc-123", notify_reviewer, {"plan": "..."})

触发方（可以在任意进程）:
    await resume(client, "chat-abc-123", {"approved": True, "comment": "OK"})

崩溃恢复：前端用同一个 key 重连时，自动判断 callback 是否已执行。
"""

import sys
from datetime import timedelta
from typing import Callable

from temporalio import workflow
from temporalio.client import Client, WorkflowExecutionStatus
from temporalio.worker import Worker, UnsandboxedWorkflowRunner


@workflow.defn
class InterruptWorkflow:
    """纯等待 Workflow —— 不执行任何 Activity，只等 resume 信号"""

    def __init__(self):
        self._response: dict | None = None
        self._callback_done: bool = False

    @workflow.signal
    async def on_resume(self, data: dict):
        self._response = data

    @workflow.signal
    async def on_callback_done(self):
        """interrupt() 在 callback 执行完后发送此信号，标记 callback 已完成"""
        self._callback_done = True

    @workflow.query
    def status(self) -> str:
        if self._response is not None:
            return "resumed"
        if self._callback_done:
            return "waiting"
        return "pending"  # Workflow 已创建，但 callback 还没执行完

    @workflow.run
    async def run(self, data: dict) -> dict:
        # 必须先等 callback 完成，才能被 resume
        await workflow.wait_condition(lambda: self._callback_done)
        await workflow.wait_condition(lambda: self._response is not None)
        return self._response


async def interrupt(
    client: Client,
    key: str,
    callback: Callable,
    data: dict,
    *,
    task_queue: str = "interrupt-queue",
    task_timeout: timedelta = timedelta(seconds=10),
) -> dict:
    """
    发起一个 interrupt 调用。

    同一个会话可能多次 interrupt，key 建议使用 session_id。
    框架会自动生成全局唯一的 interrupt_id（Workflow Run ID）传给 callback，
    callback 和下游可用此 ID 做去重。

    重连时使用同一个 key，会根据状态自动恢复：
    - pending:  Workflow 已建，callback 未完成 → 重新执行 callback
    - waiting:  callback 已完成，等待 resume → 跳过 callback，直接等
    - resumed:  已收到 resume → 直接返回结果

    callback 签名要求：
        async def my_callback(data: dict, interrupt_id: str) -> None

    - interrupt_id: 框架生成的全局唯一 ID（Temporal Run ID），
      callback 应将此 ID 传给下游，用于去重和 resume 定位
    - callback 必须是幂等的（崩溃恢复时可能被重复调用，
      下游根据 interrupt_id 去重）

    Args:
        client: Temporal client
        key: 会话标识（Workflow ID）
        callback: 幂等的 async 函数，签名 (data, interrupt_id) -> None
        data: 传给 callback 的业务数据
        task_queue: Temporal task queue 名称
        task_timeout: Workflow task 超时（调试时需加大）
    """
    handle = None

    # 尝试获取已存在的 Workflow（重连场景）
    try:
        handle = client.get_workflow_handle(key)
        desc = await handle.describe()

        if desc.status == WorkflowExecutionStatus.RUNNING:
            status = await handle.query(InterruptWorkflow.status)
            interrupt_id = desc.run_id

            if status == "waiting":
                # callback 已执行过，直接等 resume
                return await handle.result()

            if status == "pending":
                # Workflow 建了但 callback 没执行完，需要重新执行
                await callback(data, interrupt_id)
                await handle.signal(InterruptWorkflow.on_callback_done)
                return await handle.result()

        if desc.status == WorkflowExecutionStatus.COMPLETED:
            return await handle.result()

        # 其他状态（FAILED / TERMINATED / CANCELED），当作新流程
        handle = None
    except Exception:
        handle = None

    # 首次调用：注册 → callback → 标记 → 等待
    handle = await client.start_workflow(
        InterruptWorkflow.run,
        data,
        id=key,
        task_queue=task_queue,
        task_timeout=task_timeout,
    )

    # 获取 Temporal 自动生成的 Run ID 作为 interrupt_id
    interrupt_id = handle.result_run_id

    await callback(data, interrupt_id)
    await handle.signal(InterruptWorkflow.on_callback_done)

    return await handle.result()


async def resume(client: Client, key: str, data: dict) -> None:
    """
    恢复一个等待中的 interrupt。

    如果 callback 尚未完成（status=pending），抛出异常拒绝 resume。

    Args:
        client: Temporal client
        key: interrupt 时使用的同一个 key
        data: 传回给发起方的数据
    """
    handle = client.get_workflow_handle(key)

    status = await handle.query(InterruptWorkflow.status)
    if status == "pending":
        raise RuntimeError(f"Interrupt '{key}' 尚未就绪（callback 未完成），不能 resume")

    await handle.signal(InterruptWorkflow.on_resume, data)


def create_interrupt_worker(
    client: Client,
    *,
    task_queue: str = "interrupt-queue",
) -> Worker:
    """
    创建 interrupt Worker（只处理 Workflow，不需要注册 Activity）。
    Worker 完全无状态，可以部署任意多个实例。

    Args:
        client: Temporal client
        task_queue: 与 interrupt() 使用相同的 task queue
    """
    is_debugging = "debugpy" in sys.modules
    return Worker(
        client,
        task_queue=task_queue,
        workflows=[InterruptWorkflow],
        **({"workflow_runner": UnsandboxedWorkflowRunner()} if is_debugging else {}),
    )
