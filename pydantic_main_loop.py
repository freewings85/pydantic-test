"""
验证最小化 pydantic-ai agent 核心龙骨：
1. iter 中修改发送给大模型的内容（history_processors）
2. 动态工具集（DynamicToolset）
3. 依赖注入在 tool 函数中可修改
4. 依赖注入在 eval 时可 mock
"""

import asyncio
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.agent import ModelRequestNode, CallToolsNode
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.toolsets._dynamic import DynamicToolset
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_graph import End
from openai import AsyncAzureOpenAI

load_dotenv()


# ---- 1. Deps：依赖对象，tool 可修改，eval 可 mock ----

# 默认 tool 实现
DEFAULT_TOOL_MAP: dict[str, callable] = {}  # 在 tool 函数定义后填充


@dataclass
class AgentDeps:
    session_id: str = "default"
    user_id: str = "anonymous"
    available_tools: list[str] = field(default_factory=lambda: ["get_weather"])
    # eval 时可替换 tool 实现
    tool_map: dict[str, callable] = field(default_factory=dict)
    # tool 执行过程中可修改的状态
    tool_call_count: int = 0
    last_tool_result: str = ""

    def __post_init__(self):
        if not self.tool_map:
            self.tool_map = dict(DEFAULT_TOOL_MAP)


# ---- 2. Tool 函数：通过 RunContext 访问和修改 deps ----

async def get_weather(ctx: RunContext[AgentDeps], city: str) -> str:
    """获取指定城市的天气信息"""
    ctx.deps.tool_call_count += 1
    result = f"{city}: 晴天 25°C"
    ctx.deps.last_tool_result = result
    return result


async def get_time(ctx: RunContext[AgentDeps], timezone: str) -> str:
    """获取指定时区的当前时间"""
    ctx.deps.tool_call_count += 1
    result = f"{timezone}: 2026-03-06 14:30:00"
    ctx.deps.last_tool_result = result
    return result


# 填充默认 tool map
DEFAULT_TOOL_MAP.update({
    "get_weather": get_weather,
    "get_time": get_time,
})


# ---- 3. DynamicToolset：每步根据 deps 动态决定工具集 ----

def get_tools(ctx: RunContext[AgentDeps]) -> FunctionToolset:
    toolset = FunctionToolset()
    for name in ctx.deps.available_tools:
        if name in ctx.deps.tool_map:
            toolset.add_tool(Tool(ctx.deps.tool_map[name], name=name))
    return toolset


# ---- 4. history_processor：每次调 LLM 前修改消息 ----

def inject_context_processor(messages):
    """在发给 LLM 之前，可以修改消息列表"""
    # 示例：在最后一个 ModelRequest 中注入额外上下文
    # 这里只做记录，证明 processor 被调用了
    print(f"[history_processor] 消息数量: {len(messages)}")
    return messages


# ---- 5. 创建 Agent ----

def create_agent(model=None) -> Agent:
    """工厂函数，eval 时可传入 mock model"""
    if model is None:
        client = AsyncAzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ["AZURE_API_VERSION"],
        )
        provider = OpenAIProvider(openai_client=client)
        model = OpenAIChatModel(os.environ["AZURE_DEPLOYMENT_NAME"], provider=provider)

    return Agent(
        model,
        deps_type=AgentDeps,
        system_prompt="你是一个助手。用户问天气就用 get_weather，问时间就用 get_time。",
        toolsets=[DynamicToolset(get_tools, per_run_step=True)],
        history_processors=[inject_context_processor],
    )


# ---- 6. 核心循环：手动 iter/next ----

async def run_agent(agent: Agent, user_input: str, deps: AgentDeps) -> dict:
    """手动驱动 agent loop，返回执行摘要"""
    nodes_log = []

    async with agent.iter(user_input, deps=deps) as run:
        node = run.next_node

        #这种写法，是node被确定，但是node执行之前
        while not isinstance(node, End):
            node_name = type(node).__name__
            print(f"[node] {node_name}")

            if isinstance(node, ModelRequestNode):
                # 验证点1：在调 LLM 前可以修改 deps（动态加工具）
                if deps.tool_call_count > 0 and "get_time" not in deps.available_tools:
                    deps.available_tools.append("get_time")
                    print("[iter] 动态添加了 get_time 工具")

            elif isinstance(node, CallToolsNode):
                # 验证点2：可以观察模型响应
                for part in node.model_response.parts:
                    if hasattr(part, "tool_name"):
                        print(f"[iter] 模型调用工具: {part.tool_name}")

            nodes_log.append(node_name)
            node = await run.next(node)

        nodes_log.append("End")

    return {
        "output": run.result.output if run.result else "",
        "nodes": nodes_log,
        "tool_call_count": deps.tool_call_count,
        "last_tool_result": deps.last_tool_result,
        "available_tools": deps.available_tools,
    }


# ---- 7. 直接运行 ----

async def main():
    agent = create_agent()
    deps = AgentDeps(session_id="test-001", user_id="user-1")

    print("=== 运行 Agent ===")
    result = await run_agent(agent, "上海今天天气怎么样？", deps)

    print(f"\n=== 结果 ===")
    print(f"输出: {result['output']}")
    print(f"节点: {result['nodes']}")
    print(f"工具调用次数: {result['tool_call_count']}")
    print(f"最后工具结果: {result['last_tool_result']}")
    print(f"可用工具: {result['available_tools']}")


if __name__ == "__main__":
    asyncio.run(main())
