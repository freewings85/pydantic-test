"""
P0-1 验证：手动 compact 测试（使用 message_history 的第二轮请求）

发现：message_history 在首次 ModelRequestNode 时为空，
说明 Pydantic AI 在 next(node) 执行时才把消息加进去。
所以 compact 应该在第 2 次及以后的 ModelRequestNode 前操作。

方案：使用两次 agent.iter() 调用，第一次积累消息，第二次验证 compact。
通过 message_history 参数传递历史。
"""

import asyncio
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.agent import ModelRequestNode, CallToolsNode
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart, TextPart
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
    ctx.deps.call_count += 1
    return f"关于{query}：数据#{ctx.deps.call_count}"


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
    return run._graph_run.state.message_history


async def main():
    model = create_model()
    agent = Agent(
        model,
        deps_type=TestDeps,
        system_prompt="你是助手。用 lookup 工具查信息。简短回答。",
        toolsets=[DynamicToolset(get_tools, per_run_step=True)],
    )

    print("=" * 60)
    print("第一轮对话：积累消息历史")
    print("=" * 60)

    deps = TestDeps()

    # 第一轮
    async with agent.iter("查一下Python", deps=deps) as run:
        node = run.next_node
        while not isinstance(node, End):
            node = await run.next(node)

    first_messages = run.all_messages()
    print(f"第一轮结束，消息数: {len(first_messages)}")

    print("\n" + "=" * 60)
    print("第二轮对话：带历史消息 + 手动 compact")
    print("=" * 60)

    # 第二轮，带上历史消息
    async with agent.iter(
        "再查一下Rust",
        deps=deps,
        message_history=first_messages,
    ) as run:
        node = run.next_node
        iteration = 0
        compacted = False

        while not isinstance(node, End):
            iteration += 1
            node_name = type(node).__name__

            if isinstance(node, ModelRequestNode):
                history = get_message_history(run)
                print(f"\n--- 迭代 {iteration}: {node_name} ---")
                print(f"  message_history 长度: {len(history)}")

                # compact：如果历史消息多于 2 条，替换前半段
                if len(history) > 2 and not compacted:
                    print(f"  >> 触发手动 compact!")
                    summary = ModelRequest(parts=[
                        UserPromptPart(content="[COMPACT] 摘要：用户之前查询了Python的信息。")
                    ])
                    keep = list(history[-2:])
                    history.clear()
                    history.append(summary)
                    history.extend(keep)
                    compacted = True
                    print(f"  >> compact 后: {len(history)} 条")

            node = await run.next(node)
            if iteration >= 15:
                break

    final = run.all_messages()
    print(f"\n{'='*60}")
    print(f"最终结果")
    print(f"{'='*60}")
    print(f"最终消息数: {len(final)}")

    has_compact = any(
        isinstance(msg, ModelRequest) and
        any(isinstance(p, UserPromptPart) and "[COMPACT]" in p.content for p in msg.parts)
        for msg in final
    )
    print(f"包含 [COMPACT]: {has_compact}")
    print(f"compact 是否触发: {compacted}")

    for i, msg in enumerate(final):
        if isinstance(msg, ModelRequest):
            parts = []
            for p in msg.parts:
                if isinstance(p, UserPromptPart):
                    parts.append(f"User: {p.content[:100]}")
                else:
                    parts.append(type(p).__name__)
            print(f"  [{i}] ModelRequest: {', '.join(parts)}")
        else:
            print(f"  [{i}] {type(msg).__name__}")


if __name__ == "__main__":
    asyncio.run(main())
