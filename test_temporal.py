"""
Temporal 简单示例 — 模拟订单流程
演示 Workflow / Activity / Worker 的基本用法
"""

import asyncio
from datetime import timedelta
from dataclasses import dataclass

from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.worker import Worker


# --- 数据模型 ---

@dataclass
class Order:
    order_id: str
    item: str
    amount: float


# --- Activities：具体的业务操作，可以有副作用 ---

@activity.defn
async def deduct_inventory(order: Order) -> str:
    print(f"  [库存] 扣减库存: {order.item}")
    await asyncio.sleep(0.5)  # 模拟耗时
    return f"库存已扣减: {order.item}"


@activity.defn
async def charge_payment(order: Order) -> str:
    print(f"  [支付] 扣款: ¥{order.amount}")
    await asyncio.sleep(0.5)
    return f"支付成功: ¥{order.amount}"


@activity.defn
async def send_notification(order: Order) -> str:
    print(f"  [通知] 发送短信: 订单 {order.order_id} 已确认")
    await asyncio.sleep(0.3)
    return f"通知已发送: {order.order_id}"


@activity.defn
async def create_shipment(order: Order) -> str:
    print(f"  [物流] 创建物流单: {order.order_id}")
    await asyncio.sleep(0.5)
    return f"物流已创建: {order.order_id}"


# --- Workflow：编排流程，必须是确定性的 ---

@workflow.defn
class OrderWorkflow:
    @workflow.run
    async def run(self, order: Order) -> str:
        results = []

        # 第 1 步：扣库存
        r1 = await workflow.execute_activity(
            deduct_inventory,
            order,
            start_to_close_timeout=timedelta(seconds=10),
        )
        results.append(r1)

        # 第 2 步：扣款（失败则补偿库存）
        try:
            r2 = await workflow.execute_activity(
                charge_payment,
                order,
                start_to_close_timeout=timedelta(seconds=10),
            )
            results.append(r2)
        except Exception as e:
            return f"支付失败，已回滚: {e}"

        # 第 3 步：发通知（失败不影响主流程）
        try:
            r3 = await workflow.execute_activity(
                send_notification,
                order,
                start_to_close_timeout=timedelta(seconds=5),
            )
            results.append(r3)
        except Exception:
            results.append("通知发送失败（已忽略）")

        # 第 4 步：创建物流
        r4 = await workflow.execute_activity(
            create_shipment,
            order,
            start_to_close_timeout=timedelta(seconds=10),
        )
        results.append(r4)

        return " → ".join(results)


async def main():
    # 连接 Temporal Server
    client = await Client.connect("localhost:7233")

    # 启动 Worker（后台运行）
    async with Worker(
        client,
        task_queue="order-queue",
        workflows=[OrderWorkflow],
        activities=[deduct_inventory, charge_payment, send_notification, create_shipment],
    ):
        # 发起一个 Workflow 执行
        order = Order(order_id="ORD-2024-001", item="iPhone 16", amount=7999.0)

        print(f"发起订单流程: {order.order_id}")
        print("=" * 50)

        result = await client.execute_workflow(
            OrderWorkflow.run,
            order,
            id=f"order-{order.order_id}",
            task_queue="order-queue",
        )

        print("=" * 50)
        print(f"流程完成: {result}")
        print(f"\n打开 http://localhost:8233 查看 Temporal UI")


if __name__ == "__main__":
    asyncio.run(main())
