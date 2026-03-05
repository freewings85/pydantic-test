# Temporal 工作流引擎

## 概述

Temporal 是一个持久执行引擎，核心能力是让长时间运行的流程具备容错性。本项目包含两个示例：

- `test_temporal.py` — 基础订单流程（Workflow + Activity + Worker）
- `test_temporal_interrupt.py` + `test_temporal_signal.py` — human-in-the-loop 审批流程

## 前置条件

### 启动 Temporal Server

```bash
temporal server start-dev
```

默认地址：
- gRPC: `localhost:7233`
- Web UI: `http://localhost:8233`

## 基础示例：订单流程

```bash
uv run python test_temporal.py
```

流程：扣库存 → 扣款 → 发通知 → 创建物流，一次性执行完毕。

## Interrupt 示例：人工审批流程

### 实现原理

类似 LangGraph 的 interrupt，使用 Temporal 的 Signal + Wait 机制：

```
AI 生成方案 → workflow.wait_condition() 暂停 → 收到 Signal → 继续执行/拒绝
```

核心代码：

```python
# Workflow 中定义信号处理
@workflow.signal
async def approve(self, result: ApprovalResult):
    self._approval = result

# 暂停等待（interrupt 点）
await workflow.wait_condition(lambda: self._approval is not None)

# 外部发送信号恢复流程
await handle.signal("approve", result)
```

### 测试步骤

**步骤 1：启动 Worker + 发起 Workflow**

```bash
uv run python test_temporal_interrupt.py
```

输出示例：

```
============================================================
发起任务: 优化数据库查询性能
Workflow ID: approval-TASK-001
============================================================
  [AI] 正在为任务 '优化数据库查询性能' 生成方案...
  [AI] 方案已生成: 方案：针对'优化数据库查询性能'，建议分3步执行：1.调研 2.实施 3.验证

  ⏸ 流程暂停，等待人工审批...
等待审批中... (运行 test_temporal_signal.py 来审批)
```

此时流程暂停在 `wait_condition`，等待外部信号。

**步骤 2：发送审批信号（另开一个终端）**

```bash
# 审批通过
uv run python test_temporal_signal.py

# 审批拒绝
uv run python test_temporal_signal.py reject
```

发送信号后，步骤 1 的进程会继续执行并输出最终结果。

### 在 Cursor/VSCode 中调试

`launch.json` 已配置好三个调试项：

| 配置名称 | 作用 |
|----------|------|
| Temporal: Worker + Workflow (等待审批) | 启动 Worker 并发起 Workflow |
| Temporal: 发送审批 (通过) | 发送审批通过信号 |
| Temporal: 发送审批 (拒绝) | 发送审批拒绝信号 |

操作方式：
1. 先启动 "Worker + Workflow (等待审批)"，打好断点
2. 等控制台输出 "等待审批中..."
3. 再启动 "发送审批 (通过)" 或 "(拒绝)"

## 踩坑记录

### 1. Workflow Sandbox 与 debugpy 冲突（死锁）

**现象**：在 Cursor/VSCode 中调试时，报错：

```
ModuleNotFoundError: No module named '_pydevd_bundle'
Potential deadlock detected, workflow didn't yield within 2 second(s)
```

**原因**：Temporal 默认将 Workflow 代码运行在沙箱中，沙箱会拦截所有 Python `import`，阻止非确定性模块加载。而 debugpy 需要动态导入 `_pydevd_bundle` 模块来注入断点，被沙箱拦截后导致死锁。

**解决方案**：调试时自动禁用沙箱，生产环境保留：

```python
import sys
from temporalio.worker import Worker, UnsandboxedWorkflowRunner

is_debugging = "debugpy" in sys.modules
worker = Worker(
    client,
    task_queue="approval-queue",
    workflows=[ApprovalWorkflow],
    activities=[...],
    **({"workflow_runner": UnsandboxedWorkflowRunner()} if is_debugging else {}),
)
```

`"debugpy" in sys.modules` 判断当前是否通过调试器启动：
- Cursor/VSCode 调试 → debugpy 已加载 → 禁用沙箱
- 命令行直接运行 → debugpy 未加载 → 保留默认沙箱

> **注意**：沙箱的作用是在开发阶段检测 Workflow 中的非确定性代码（如直接调用 `datetime.now()`、`random` 等）。生产环境建议保留。

### 2. Workflow Task Timeout（调试时超时）

**现象**：调试时在断点停留超过 10 秒，Temporal 认为 Workflow Task 超时，不断重试。

**解决方案**：`start_workflow` 时增大 `task_timeout`：

```python
handle = await client.start_workflow(
    ApprovalWorkflow.run,
    task,
    id=WORKFLOW_ID,
    task_queue="approval-queue",
    task_timeout=timedelta(minutes=10),  # 默认 10 秒，调试时需要加大
)
```

### 3. Workflow ID 与 Run ID

- **Workflow ID**：业务标识，由你定义（如 `approval-TASK-001`），同一时刻只能有一个在运行
- **Run ID**：每次执行的唯一标识，Temporal 自动生成

日常操作（Signal、Query、Cancel）只需要 Workflow ID。同一个 Workflow ID 结束后可以重复使用，Temporal UI 中会保留所有历史执行记录。

### 4. Activity 结果持久化

Temporal 将每个 Activity 的输入和返回值都序列化存储。Worker 崩溃重启后，重放 Workflow 时不会重新执行已完成的 Activity，而是直接从历史记录中取出缓存的结果继续执行。这就是 Temporal "持久执行" 的核心机制。
