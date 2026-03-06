"""
P0-1 验证：history_processors 能否实现 compact 和 attachment 注入

验证目标：
1. history_processors 能否删减消息（compact：删除前半段，替换为摘要）
2. history_processors 能否添加消息（attachment：注入 system-reminder）
3. 修改是只影响本次 API 调用的输入，还是永久改变 run 内部状态
4. 在 agent.iter() 的多轮迭代中，history_processors 是否每轮都被调用
"""

import asyncio
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.agent import ModelRequestNode, CallToolsNode
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart, SystemPromptPart
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.toolsets._dynamic import DynamicToolset
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_graph import End
from openai import AsyncAzureOpenAI

load_dotenv()


# ---- Deps ----

@dataclass
class TestDeps:
    call_count: int = 0
    processor_call_log: list[str] = field(default_factory=list)
    available_tools: list[str] = field(default_factory=lambda: ["get_info"])


# ---- Tool ----

async def get_info(ctx: RunContext[TestDeps], topic: str) -> str:
    """获取信息，用于触发多轮迭代"""
    ctx.deps.call_count += 1
    return f"关于{topic}的信息：这是第{ctx.deps.call_count}次查询的结果。"


def get_tools(ctx: RunContext[TestDeps]) -> FunctionToolset:
    toolset = FunctionToolset()
    toolset.add_tool(Tool(get_info, name="get_info"))
    return toolset


# ---- 验证 1：删减消息（模拟 compact） ----

compact_triggered: bool = False


def compact_processor(messages: list[ModelMessage]) -> list[ModelMessage]:
    """模拟 compact：当消息数 > 4 时，删除前半段，替换为摘要"""
    global compact_triggered
    msg_count = len(messages)
    print(f"  [compact_processor] 收到 {msg_count} 条消息")

    if msg_count > 4:
        compact_triggered = True
        # 保留最后 2 条，前面替换为摘要
        summary = ModelRequest(parts=[
            UserPromptPart(content=f"[COMPACT SUMMARY] 前 {msg_count - 2} 条消息的摘要：用户询问了一些问题，助手使用工具查询并回答了。")
        ])
        result = [summary] + list(messages[-2:])
        print(f"  [compact_processor] 压缩: {msg_count} → {len(result)} 条")
        return result

    return list(messages)


# ---- 验证 2：添加消息（模拟 attachment 注入） ----

attachment_inject_count: int = 0


def attachment_processor(messages: list[ModelMessage]) -> list[ModelMessage]:
    """模拟 attachment：每次调 LLM 前注入一条 system-reminder"""
    global attachment_inject_count
    attachment_inject_count += 1

    # 在消息列表末尾添加一条 attachment（模拟 changed_files 等动态注入）
    attachment = ModelRequest(parts=[
        UserPromptPart(content=f"<system-reminder>动态注入 #{attachment_inject_count}: token_usage=1234</system-reminder>")
    ])

    result = list(messages) + [attachment]
    print(f"  [attachment_processor] 注入第 {attachment_inject_count} 次 attachment，消息数: {len(messages)} → {len(result)}")
    return result


# ---- 验证 3：检查 run 内部状态是否被永久修改 ----

def create_model() -> Model:
    client = AsyncAzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_API_VERSION"],
    )
    provider = OpenAIProvider(openai_client=client)
    return OpenAIChatModel(os.environ["AZURE_DEPLOYMENT_NAME"], provider=provider)


async def test_history_processors():
    """主测试函数"""
    model = create_model()

    agent = Agent(
        model,
        deps_type=TestDeps,
        system_prompt="你是一个助手。用户问任何问题都先用 get_info 查一下再回答。请查两次不同的主题。",
        toolsets=[DynamicToolset(get_tools, per_run_step=True)],
        history_processors=[compact_processor, attachment_processor],
    )

    deps = TestDeps()

    print("=" * 60)
    print("P0-1 验证：history_processors 能力测试")
    print("=" * 60)

    async with agent.iter(
        "请帮我查一下Python和Rust这两个主题的信息",
        deps=deps,
    ) as run:
        node = run.next_node
        iteration = 0

        while not isinstance(node, End):
            iteration += 1
            node_name = type(node).__name__
            print(f"\n--- 迭代 {iteration}: {node_name} ---")

            if isinstance(node, ModelRequestNode):
                # 在 next(node) 之前，打印当前 run 内部的消息数
                internal_msg_count = len(run.all_messages())
                print(f"  [run 内部消息数（processor 前）] {internal_msg_count}")

            elif isinstance(node, CallToolsNode):
                for part in node.model_response.parts:
                    if hasattr(part, "tool_name"):
                        print(f"  [工具调用] {part.tool_name}")

            node = await run.next(node)

            if iteration >= 15:
                print("  [安全退出] 达到最大迭代数")
                break

    # ---- 结果分析 ----
    print("\n" + "=" * 60)
    print("验证结果")
    print("=" * 60)

    final_messages = run.all_messages()
    print(f"\n1. run.all_messages() 最终消息数: {len(final_messages)}")
    print(f"   （如果 compact/attachment 永久修改了 run 内部状态，这里的数量会变）")

    print(f"\n2. compact_processor 是否触发: {compact_triggered}")
    print(f"   （消息数 > 4 时应该触发）")

    print(f"\n3. attachment_processor 注入次数: {attachment_inject_count}")
    print(f"   （应该等于 ModelRequestNode 的执行次数）")

    print(f"\n4. 工具调用次数: {deps.call_count}")

    # 检查最终消息中是否包含 compact summary 或 attachment
    has_compact_summary = False
    has_attachment = False
    for msg in final_messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    if "[COMPACT SUMMARY]" in part.content:
                        has_compact_summary = True
                    if "<system-reminder>" in part.content:
                        has_attachment = True

    print(f"\n5. 最终消息中包含 COMPACT SUMMARY: {has_compact_summary}")
    print(f"   最终消息中包含 system-reminder: {has_attachment}")
    print(f"   （如果为 False，说明 processor 的修改不影响 run 内部状态——只影响发给 API 的）")

    # 打印所有消息的简要内容
    print(f"\n6. 最终消息列表:")
    for i, msg in enumerate(final_messages):
        if isinstance(msg, ModelRequest):
            parts_summary = []
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    content_preview = part.content[:80].replace('\n', ' ')
                    parts_summary.append(f"UserPrompt: {content_preview}...")
                elif isinstance(part, SystemPromptPart):
                    parts_summary.append(f"SystemPrompt: {part.content[:50]}...")
                else:
                    parts_summary.append(f"{type(part).__name__}")
            print(f"   [{i}] ModelRequest: {', '.join(parts_summary)}")
        else:
            print(f"   [{i}] {type(msg).__name__}: (部分省略)")


if __name__ == "__main__":
    asyncio.run(test_history_processors())
