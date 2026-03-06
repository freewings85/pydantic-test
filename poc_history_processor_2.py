"""
P0-1 验证（续）：验证 compact（删减消息）的行为

关键问题：
1. processor 删减消息后，run 内部状态是否也变少？
2. 如果 run 内部保留完整历史但 processor 只发送裁剪后的，是否有 tool_use/tool_result 配对问题？
3. attachment 累积问题：processor 添加的消息会永久保留，需要清理机制
"""

import asyncio
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.agent import ModelRequestNode, CallToolsNode
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models import Model
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


# ---- 激进的 compact：消息数 > 2 就压缩 ----

compact_calls: list[dict] = []


def aggressive_compact_processor(messages: list[ModelMessage]) -> list[ModelMessage]:
    """激进压缩：消息数 > 2 时只保留最后 2 条"""
    call_info = {"input_count": len(messages)}

    if len(messages) > 2:
        # 只保留最后 2 条
        summary = ModelRequest(parts=[
            UserPromptPart(content=f"[SUMMARY] 前面的对话摘要：用户进行了多轮查询。")
        ])
        result = [summary] + list(messages[-2:])
        call_info["output_count"] = len(result)
        call_info["action"] = "compacted"
        compact_calls.append(call_info)
        print(f"  [compact] {len(messages)} → {len(result)} 条")
        return result

    call_info["output_count"] = len(messages)
    call_info["action"] = "pass-through"
    compact_calls.append(call_info)
    print(f"  [compact] {len(messages)} 条，不压缩")
    return list(messages)


def create_model() -> Model:
    client = AsyncAzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_API_VERSION"],
    )
    provider = OpenAIProvider(openai_client=client)
    return OpenAIChatModel(os.environ["AZURE_DEPLOYMENT_NAME"], provider=provider)


async def test_compact():
    model = create_model()

    agent = Agent(
        model,
        deps_type=TestDeps,
        system_prompt="你是助手。每次用户提问，都用 lookup 工具查一下再回答。请分别查 A、B、C 三个主题。",
        toolsets=[DynamicToolset(get_tools, per_run_step=True)],
        history_processors=[aggressive_compact_processor],
    )

    deps = TestDeps()

    print("=" * 60)
    print("P0-1 验证（续）：compact 删减消息行为")
    print("=" * 60)

    async with agent.iter(
        "请帮我分别查询 A、B、C 三个主题的信息，每个主题单独查询",
        deps=deps,
    ) as run:
        node = run.next_node
        iteration = 0

        while not isinstance(node, End):
            iteration += 1
            node_name = type(node).__name__
            print(f"\n--- 迭代 {iteration}: {node_name} ---")

            if isinstance(node, ModelRequestNode):
                internal_count = len(run.all_messages())
                print(f"  [run 内部消息数] {internal_count}")

            node = await run.next(node)

            if iteration >= 20:
                print("  [安全退出]")
                break

    # ---- 结果 ----
    print("\n" + "=" * 60)
    print("验证结果")
    print("=" * 60)

    final_messages = run.all_messages()
    print(f"\n最终 run.all_messages() 数量: {len(final_messages)}")

    has_summary = False
    for msg in final_messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart) and "[SUMMARY]" in part.content:
                    has_summary = True

    print(f"最终消息中包含 [SUMMARY]: {has_summary}")
    print(f"工具调用次数: {deps.call_count}")

    print(f"\ncompact_processor 调用记录:")
    for i, call in enumerate(compact_calls):
        print(f"  第{i+1}次: {call}")

    print(f"\n最终消息列表:")
    for i, msg in enumerate(final_messages):
        if isinstance(msg, ModelRequest):
            parts_summary = []
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    preview = part.content[:100].replace('\n', ' ')
                    parts_summary.append(f"UserPrompt: {preview}")
                else:
                    parts_summary.append(f"{type(part).__name__}")
            print(f"  [{i}] ModelRequest: {', '.join(parts_summary)}")
        else:
            print(f"  [{i}] {type(msg).__name__}")


if __name__ == "__main__":
    asyncio.run(test_compact())
