# Interrupt/Resume 抽象层

基于 Temporal 封装的 human-in-the-loop 通用方案，适用于 AI Agent 流程中需要人工确认的场景。

## 核心 API

### interrupt — 发起中断等待

```python
from lib.interrupt import interrupt, create_interrupt_worker

response = await interrupt(
    client,                    # Temporal client
    key="session-abc-123",     # 会话 ID（Workflow ID）
    callback=notify_reviewer,  # 中断时执行的回调
    data={"plan": "..."},      # 传给 callback 的业务数据
)
# response 是 resume 传入的数据，如 {"approved": True, "comment": "OK"}
```

### resume — 恢复中断

```python
from lib.interrupt import resume

await resume(
    client,                                          # Temporal client
    key="session-abc-123",                           # 同一个会话 ID
    data={"approved": True, "comment": "方案可行"},   # 传回给发起方的数据
)
```

### create_interrupt_worker — 创建 Worker

```python
worker = create_interrupt_worker(client)
async with worker:
    # 在此范围内可以使用 interrupt()
    ...
```

Worker 完全无状态，可以部署任意多个实例。

## callback 要求

callback 签名必须是：

```python
async def my_callback(data: dict, interrupt_id: str) -> None:
```

| 参数 | 说明 |
|------|------|
| `data` | 调用 `interrupt()` 时传入的业务数据 |
| `interrupt_id` | 框架自动生成的全局唯一 ID（Temporal Run ID），用于去重 |

**callback 必须是幂等的**。进程崩溃恢复时可能被重复调用，下游应根据 `interrupt_id` 去重。

示例：

```python
async def notify_reviewer(data: dict, interrupt_id: str) -> None:
    await stream.write({
        "type": "interrupt",
        "interrupt_id": interrupt_id,  # 下游用此 ID 去重
        "plan": data["plan"],
    })
```

## 执行流程

```
发起方进程                           Temporal Server              触发方进程

interrupt(key, callback, data)
  │
  ├─ start_workflow(id=key)  ──────→  Workflow 创建（status=pending）
  │                                   可以接收 resume 了
  │
  ├─ callback(data, interrupt_id)     在本进程执行（写 stream event）
  │
  ├─ signal(on_callback_done)  ────→  status 变为 waiting
  │                                   此刻起 resume 才被允许
  │
  ├─ await handle.result()            挂起等待（不卡线程）
  │                                                               resume(key, data)
  │                                   ←──────────────────────────  signal(on_resume)
  │
  ├─ 收到结果  ←───────────────────   Workflow 完成
  │
  └─ return response
```

## 状态机

```
pending  ──callback完成──→  waiting  ──收到resume──→  resumed（Workflow完成）
   │                          │
   │  resume() 会报错拒绝      │  resume() 正常执行
```

- **pending**: Workflow 已创建，callback 未完成。此时 `resume()` 会抛出异常
- **waiting**: callback 已完成，等待 resume
- **resumed**: 已收到 resume，Workflow 完成

## 崩溃恢复

前端重连时使用同一个 `key`，框架自动判断状态并恢复：

| 崩溃时机 | 重连后行为 |
|----------|-----------|
| `start_workflow` 之前 | Workflow 不存在，全新流程 |
| `start_workflow` 之后，callback 之前 | status=pending，重新执行 callback |
| callback 之后，`on_callback_done` 之前 | status=pending，重新执行 callback（幂等） |
| `on_callback_done` 之后 | status=waiting，跳过 callback，直接等 resume |
| resume 之后 | status=resumed / COMPLETED，直接返回结果 |

## 同一会话多次 interrupt

同一个会话内可能有多次 interrupt（如先审批方案，再确认部署）。`key` 需要区分每次 interrupt：

```python
response1 = await interrupt(
    client,
    key=f"{session_id}-approve-plan",
    callback=notify_reviewer,
    data={"plan": "..."},
)

response2 = await interrupt(
    client,
    key=f"{session_id}-confirm-deploy",
    callback=notify_deployer,
    data={"target": "production"},
)
```

每次 interrupt 的 `interrupt_id`（Temporal Run ID）都不同，下游据此去重。

## 在 FastAPI 中使用

```python
from fastapi import FastAPI
from lib.interrupt import resume
from temporalio.client import Client

app = FastAPI()
client: Client = None

@app.on_event("startup")
async def startup():
    global client
    client = await Client.connect("localhost:7233")

@app.post("/approve/{session_id}")
async def approve(session_id: str, body: ApproveRequest):
    await resume(client, session_id, {"approved": True, "comment": body.comment})
    return {"status": "ok"}
```

发起方（AI Agent）在后台长时间运行，FastAPI 只暴露 resume 接口。

## 测试步骤

### 命令行

```bash
# 终端 1：启动发起方
uv run python test_interrupt_demo.py

# 终端 2：发送审核通过
uv run python test_interrupt_resume.py

# 或发送审核拒绝
uv run python test_interrupt_resume.py reject
```

### Cursor/VSCode 调试

`launch.json` 中已配置：

| 配置名称 | 作用 |
|----------|------|
| Interrupt: 发起方 (等待审核) | 启动 Worker + 发起 interrupt |
| Interrupt: 审核通过 | 发送 resume（通过） |
| Interrupt: 审核拒绝 | 发送 resume（拒绝） |

操作：先启动"发起方"，等输出"等待人工审核"，再启动通过或拒绝。

## 注意事项

- **callback 在本进程执行**，不走 Temporal Activity，保证与 stream 在同一个连接中
- **Temporal 只负责等待/恢复**，不存储 callback 的执行状态，通过 `on_callback_done` 信号标记
- **Nondeterminism 错误**：如果修改了 `InterruptWorkflow` 的结构（如增删 Activity），旧的 Workflow 实例会报此错误。需要先终止旧 Workflow：`temporal workflow terminate --workflow-id <key>`
- **调试时** `task_timeout` 需要加大（默认 10 秒，调试断点时会超时），代码中已自动检测 debugpy 并禁用 sandbox
