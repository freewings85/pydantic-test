"""
P0-1 验证（续）：能否在 loop 中手动操作 run 的消息列表

验证目标：
1. run 对象上是否有可写的消息列表属性
2. 直接修改该列表后，下一轮 LLM 调用是否使用修改后的消息
3. 找到正确的属性名和操作方式
"""

import asyncio
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.agent import ModelRequestNode, CallToolsNode, AgentRun
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


def create_model() -> Model:
    client = AsyncAzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_API_VERSION"],
    )
    provider = OpenAIProvider(openai_client=client)
    return OpenAIChatModel(os.environ["AZURE_DEPLOYMENT_NAME"], provider=provider)


async def test_manual_messages():
    model = create_model()

    agent = Agent(
        model,
        deps_type=TestDeps,
        system_prompt="你是助手。用户提问时用 lookup 工具查一下再回答。",
        toolsets=[DynamicToolset(get_tools, per_run_step=True)],
    )

    deps = TestDeps()

    print("=" * 60)
    print("验证：手动操作 run 的消息列表")
    print("=" * 60)

    async with agent.iter(
        "请查询主题A的信息",
        deps=deps,
    ) as run:
        # ---- 第一步：探索 run 对象的属性 ----
        print("\n--- 探索 run 对象 ---")
        print(f"run 类型: {type(run).__name__}")

        # 查看所有公开属性
        public_attrs = [a for a in dir(run) if not a.startswith('__')]
        print(f"公开属性: {public_attrs}")

        # 查看可能的消息相关属性
        for attr_name in ['_messages', 'messages', '_result', 'result',
                          'all_messages', '_all_messages', '_graph_run',
                          '_state', 'ctx', '_ctx']:
            if hasattr(run, attr_name):
                val = getattr(run, attr_name)
                val_type = type(val).__name__
                if callable(val):
                    print(f"  {attr_name}: {val_type} (callable)")
                elif isinstance(val, list):
                    print(f"  {attr_name}: list[{len(val)}]")
                else:
                    print(f"  {attr_name}: {val_type}")

        # 尝试找 graph_run 或 state
        if hasattr(run, '_graph_run'):
            gr = run._graph_run
            print(f"\n--- _graph_run 对象 ---")
            print(f"类型: {type(gr).__name__}")
            gr_attrs = [a for a in dir(gr) if not a.startswith('__')]
            print(f"属性: {gr_attrs}")

            if hasattr(gr, 'state'):
                state = gr.state
                print(f"\n--- state 对象 ---")
                print(f"类型: {type(state).__name__}")
                state_attrs = [a for a in dir(state) if not a.startswith('__')]
                print(f"属性: {state_attrs}")

                # 查找消息列表
                for attr_name in ['messages', 'message_history', '_messages',
                                  'all_messages', 'history']:
                    if hasattr(state, attr_name):
                        val = getattr(state, attr_name)
                        if isinstance(val, list):
                            print(f"  state.{attr_name}: list[{len(val)}]")
                        else:
                            print(f"  state.{attr_name}: {type(val).__name__}")

        # ---- 第二步：尝试修改消息 ----
        node = run.next_node
        iteration = 0
        injected = False

        while not isinstance(node, End):
            iteration += 1
            node_name = type(node).__name__

            if isinstance(node, ModelRequestNode) and not injected:
                print(f"\n--- 迭代 {iteration}: {node_name} ---")
                msgs_before = run.all_messages()
                print(f"  all_messages() 数量: {len(msgs_before)}")

                # 尝试直接操作
                # 方案 A：如果 all_messages() 返回的是内部列表的引用
                msgs = run.all_messages()
                is_same_ref = False

                # 方案 B：查找内部可变列表
                if hasattr(run, '_graph_run') and hasattr(run._graph_run, 'state'):
                    state = run._graph_run.state
                    for attr_name in dir(state):
                        if attr_name.startswith('_'):
                            continue
                        val = getattr(state, attr_name)
                        if isinstance(val, list) and len(val) > 0:
                            # 检查是否是消息列表
                            from pydantic_ai.messages import ModelMessage
                            if all(isinstance(v, ModelMessage) for v in val):
                                print(f"\n  找到消息列表: state.{attr_name}")
                                print(f"  长度: {len(val)}")
                                print(f"  是否与 all_messages() 相同引用: {val is msgs}")

                                # 尝试注入一条消息
                                inject_msg = ModelRequest(parts=[
                                    UserPromptPart(content="<system-reminder>手动注入的 attachment</system-reminder>")
                                ])
                                val.append(inject_msg)
                                injected = True
                                print(f"  注入后长度: {len(val)}")
                                print(f"  all_messages() 长度: {len(run.all_messages())}")
                                break

            elif isinstance(node, ModelRequestNode) and injected:
                print(f"\n--- 迭代 {iteration}: {node_name}（注入后） ---")
                print(f"  all_messages() 数量: {len(run.all_messages())}")

            node = await run.next(node)

            if iteration >= 15:
                break

    # ---- 结果 ----
    print("\n" + "=" * 60)
    print("最终结果")
    print("=" * 60)

    final_messages = run.all_messages()
    print(f"最终消息数: {len(final_messages)}")

    has_injection = False
    for msg in final_messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart) and "手动注入" in part.content:
                    has_injection = True

    print(f"最终消息中包含手动注入: {has_injection}")

    print(f"\n消息列表:")
    for i, msg in enumerate(final_messages):
        if isinstance(msg, ModelRequest):
            parts_summary = []
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    preview = part.content[:80].replace('\n', ' ')
                    parts_summary.append(f"UserPrompt: {preview}")
                else:
                    parts_summary.append(f"{type(part).__name__}")
            print(f"  [{i}] ModelRequest: {', '.join(parts_summary)}")
        else:
            print(f"  [{i}] {type(msg).__name__}")


if __name__ == "__main__":
    asyncio.run(test_manual_messages())
