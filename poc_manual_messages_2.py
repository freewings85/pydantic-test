"""
P0-1 验证：通过 run._graph_run.state.message_history 手动操作消息

已知路径：run._graph_run.state.message_history (list)
"""

import asyncio
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.agent import ModelRequestNode, CallToolsNode
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.toolsets._dynamic import DynamicToolset
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_graph import End
from openai import AsyncAzureOpenAI

load_dotenv()


@dataclass
class TestDeps:
    call_count: int = 0
    available_tools: list[str] = field(default_factory=lambda: ["lookup"])


async def lookup(ctx: RunContext[TestDeps], query: str) -> str:
    """查询信息"""
    ctx.deps.call_count += 1
    return f"查询'{query}'的结果：数据#{ctx.deps.call_count}"


def get_tools(ctx: RunContext[TestDeps]) -> FunctionToolset:
    toolset = FunctionToolset()
    toolset.add_tool(Tool(lookup, name="lookup"))
    return toolset


def create_model():
    client = AsyncAzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_API_VERSION"],
    )
    provider = OpenAIProvider(openai_client=client)
    return OpenAIChatModel(os.environ["AZURE_DEPLOYMENT_NAME"], provider=provider)


def get_message_history(run):
    """获取 run 内部的可变消息列表"""
    return run._graph_run.state.message_history


async def test_manual_inject():
    """测试 1：手动注入消息"""
    model = create_model()
    agent = Agent(
        model,
        deps_type=TestDeps,
        system_prompt="你是助手。用户提问时用 lookup 工具查一下再回答。",
        toolsets=[DynamicToolset(get_tools, per_run_step=True)],
    )
    deps = TestDeps()

    print("=" * 60)
    print("测试 1：手动注入消息（模拟 attachment）")
    print("=" * 60)

    async with agent.iter("请查询主题A的信息", deps=deps) as run:
        node = run.next_node
        iteration = 0

        while not isinstance(node, End):
            iteration += 1
            node_name = type(node).__name__

            if isinstance(node, ModelRequestNode):
                history = get_message_history(run)
                print(f"\n--- 迭代 {iteration}: {node_name} ---")
                print(f"  message_history 长度: {len(history)}")
                print(f"  all_messages() 长度: {len(run.all_messages())}")
                print(f"  是否同一引用: {history is run.all_messages()}")

                # 手动注入 attachment
                inject = ModelRequest(parts=[
                    UserPromptPart(content=f"<system-reminder>手动注入 #{iteration}</system-reminder>")
                ])
                history.append(inject)
                print(f"  注入后 message_history: {len(history)}")
                print(f"  注入后 all_messages(): {len(run.all_messages())}")

            node = await run.next(node)
            if iteration >= 15:
                break

    final = run.all_messages()
    print(f"\n最终消息数: {len(final)}")
    for i, msg in enumerate(final):
        if isinstance(msg, ModelRequest):
            parts = []
            for p in msg.parts:
                if isinstance(p, UserPromptPart):
                    parts.append(f"User: {p.content[:80]}")
                else:
                    parts.append(type(p).__name__)
            print(f"  [{i}] ModelRequest: {', '.join(parts)}")
        else:
            print(f"  [{i}] {type(msg).__name__}")


async def test_manual_compact():
    """测试 2：手动删减消息（模拟 compact）"""
    model = create_model()
    agent = Agent(
        model,
        deps_type=TestDeps,
        system_prompt="你是助手。用户提问时用 lookup 工具查一下再回答。请查 A 和 B 两个主题。",
        toolsets=[DynamicToolset(get_tools, per_run_step=True)],
    )
    deps = TestDeps()

    print("\n" + "=" * 60)
    print("测试 2：手动删减消息（模拟 compact）")
    print("=" * 60)

    compacted = False

    async with agent.iter("请分别查询A和B两个主题", deps=deps) as run:
        node = run.next_node
        iteration = 0

        while not isinstance(node, End):
            iteration += 1
            node_name = type(node).__name__

            if isinstance(node, ModelRequestNode):
                history = get_message_history(run)
                print(f"\n--- 迭代 {iteration}: {node_name} ---")
                print(f"  message_history 长度: {len(history)}")

                # 当消息数 > 2 时，手动 compact
                if len(history) > 2 and not compacted:
                    print(f"  触发手动 compact!")
                    # 保留最后 2 条，前面替换为摘要
                    summary = ModelRequest(parts=[
                        UserPromptPart(content="[COMPACT] 之前的对话摘要：用户查询了主题A。")
                    ])
                    # 清空并重建
                    keep = list(history[-2:])
                    history.clear()
                    history.append(summary)
                    history.extend(keep)
                    compacted = True
                    print(f"  compact 后 message_history: {len(history)}")

            node = await run.next(node)
            if iteration >= 15:
                break

    final = run.all_messages()
    print(f"\n最终消息数: {len(final)}")
    has_compact = any(
        isinstance(msg, ModelRequest) and
        any(isinstance(p, UserPromptPart) and "[COMPACT]" in p.content for p in msg.parts)
        for msg in final
    )
    print(f"包含 [COMPACT] 摘要: {has_compact}")

    for i, msg in enumerate(final):
        if isinstance(msg, ModelRequest):
            parts = []
            for p in msg.parts:
                if isinstance(p, UserPromptPart):
                    parts.append(f"User: {p.content[:80]}")
                else:
                    parts.append(type(p).__name__)
            print(f"  [{i}] ModelRequest: {', '.join(parts)}")
        else:
            print(f"  [{i}] {type(msg).__name__}")


async def main():
    await test_manual_inject()
    await test_manual_compact()


if __name__ == "__main__":
    asyncio.run(main())
