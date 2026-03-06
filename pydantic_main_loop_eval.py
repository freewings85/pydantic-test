"""
eval 验证：用 FunctionModel mock 依赖，验证核心龙骨的 4 个能力
"""

import asyncio
from pydantic_ai.models.function import FunctionModel, AgentInfo
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
)

from pydantic_ai import RunContext

from pydantic_main_loop import (
    AgentDeps,
    create_agent,
    run_agent,
)


# ---- Mock Model：确定性行为，不调真实 LLM ----

def mock_weather_then_answer(
    messages: list[ModelMessage], info: AgentInfo
) -> ModelResponse:
    """
    第1次调用: 返回 get_weather tool call
    第2次调用: 返回文本回复
    """
    # 检查是否已有 tool 结果（说明是第2次调用）
    for msg in messages:
        for part in msg.parts:
            if hasattr(part, "part_kind") and part.part_kind == "tool-return":
                return ModelResponse(
                    parts=[TextPart(content=f"根据查询结果，天气情况是：{part.content}")]
                )

    # 第1次：调用 get_weather
    return ModelResponse(
        parts=[ToolCallPart(tool_name="get_weather", args={"city": "上海"})]
    )


def mock_two_tools(
    messages: list[ModelMessage], info: AgentInfo
) -> ModelResponse:
    """
    第1次: 调 get_weather
    第2次: 调 get_time（验证动态添加工具）
    第3次: 返回文本
    """
    tool_return_count = sum(
        1
        for msg in messages
        for part in msg.parts
        if hasattr(part, "part_kind") and part.part_kind == "tool-return"
    )

    if tool_return_count == 0:
        return ModelResponse(
            parts=[ToolCallPart(tool_name="get_weather", args={"city": "北京"})]
        )
    elif tool_return_count == 1:
        return ModelResponse(
            parts=[ToolCallPart(tool_name="get_time", args={"timezone": "Asia/Shanghai"})]
        )
    else:
        return ModelResponse(
            parts=[TextPart(content="北京晴天，当前时间14:30")]
        )


# ---- 测试用例 ----

async def test_1_basic_tool_call():
    """验证：基本 tool call 流程 + deps 被 tool 修改"""
    print("\n=== Test 1: 基本 tool call + deps 修改 ===")

    model = FunctionModel(mock_weather_then_answer)
    agent = create_agent(model=model)
    deps = AgentDeps(session_id="eval-001", user_id="test-user")

    result = await run_agent(agent, "上海天气怎么样？", deps)

    # 验证 tool 被调用
    assert result["tool_call_count"] == 1, f"期望1次, 实际{result['tool_call_count']}"
    # 验证 deps 被 tool 修改
    assert "上海" in result["last_tool_result"], f"结果应包含上海: {result['last_tool_result']}"
    # 验证最终输出包含天气信息
    assert result["output"], "应有输出"
    # 验证节点流转
    assert "UserPromptNode" in result["nodes"]
    assert "ModelRequestNode" in result["nodes"]
    assert "CallToolsNode" in result["nodes"]
    assert "End" in result["nodes"]

    print(f"  输出: {result['output']}")
    print(f"  节点: {result['nodes']}")
    print("  PASS")


async def test_2_dynamic_tool_addition():
    """验证：iter 循环中动态添加工具"""
    print("\n=== Test 2: 动态添加工具 ===")

    model = FunctionModel(mock_two_tools)
    agent = create_agent(model=model)
    # 初始只有 get_weather
    deps = AgentDeps(
        session_id="eval-002",
        available_tools=["get_weather"],
    )

    result = await run_agent(agent, "北京天气和时间", deps)

    # 验证两个工具都被调用
    assert result["tool_call_count"] == 2, f"期望2次, 实际{result['tool_call_count']}"
    # 验证 get_time 被动态添加
    assert "get_time" in result["available_tools"], "get_time 应被动态添加"

    print(f"  输出: {result['output']}")
    print(f"  工具调用次数: {result['tool_call_count']}")
    print(f"  可用工具: {result['available_tools']}")
    print("  PASS")


async def test_3_deps_mock():
    """验证：eval 时 deps 完全可控"""
    print("\n=== Test 3: deps mock ===")

    model = FunctionModel(mock_weather_then_answer)
    agent = create_agent(model=model)

    # mock deps：自定义初始状态
    deps = AgentDeps(
        session_id="eval-mock",
        user_id="mock-user-999",
        available_tools=["get_weather"],
        tool_call_count=10,  # 初始值不为0
    )

    result = await run_agent(agent, "深圳天气", deps)

    # 验证 deps 在 tool 中被正确修改（基于 mock 初始值）
    assert result["tool_call_count"] == 11, f"期望11, 实际{result['tool_call_count']}"
    assert deps.user_id == "mock-user-999", "user_id 不应被修改"

    print(f"  tool_call_count: {result['tool_call_count']}（初始10 + 1）")
    print(f"  user_id: {deps.user_id}")
    print("  PASS")


async def test_4_history_processor_called():
    """验证：history_processor 在每次 LLM 调用前被执行"""
    print("\n=== Test 4: history_processor 被调用 ===")

    call_log = []

    def tracking_processor(messages):
        call_log.append(len(messages))
        return messages

    model = FunctionModel(mock_weather_then_answer)
    agent = create_agent(model=model)
    # 替换 history_processors
    agent.history_processors = [tracking_processor]

    deps = AgentDeps()
    result = await run_agent(agent, "天气", deps)

    # history_processor 应被调用2次（第1次调 LLM 返回 tool call，第2次调 LLM 返回文本）
    assert len(call_log) == 2, f"期望2次, 实际{len(call_log)}次"
    # 第2次消息数应比第1次多（多了 tool call + tool result）
    assert call_log[1] > call_log[0], f"消息应递增: {call_log}"

    print(f"  processor 调用次数: {len(call_log)}")
    print(f"  每次消息数: {call_log}")
    print("  PASS")


async def test_5_tool_mock():
    """验证：eval 时可以 mock tool 的实现（替换返回值）"""
    print("\n=== Test 5: tool 实现 mock ===")

    model = FunctionModel(mock_weather_then_answer)
    agent = create_agent(model=model)

    # mock tool：替换 get_weather 的实现，类型注解必须正确
    async def fake_get_weather(ctx: RunContext[AgentDeps], city: str) -> str:
        """获取指定城市的天气信息"""
        ctx.deps.tool_call_count += 1
        result = f"{city}: 暴雨 -10°C（mock）"
        ctx.deps.last_tool_result = result
        return result

    deps = AgentDeps(
        session_id="eval-tool-mock",
        available_tools=["get_weather"],
        tool_map={"get_weather": fake_get_weather},
    )

    result = await run_agent(agent, "上海天气", deps)

    # 验证用的是 mock 的 tool 实现
    assert "暴雨" in result["last_tool_result"], f"应该用 mock 实现: {result['last_tool_result']}"
    assert "mock" in result["last_tool_result"], f"应包含 mock 标记: {result['last_tool_result']}"
    # 验证 LLM 拿到的是 mock 结果
    assert "暴雨" in result["output"], f"LLM 输出应包含 mock 天气: {result['output']}"

    print(f"  tool 返回: {result['last_tool_result']}")
    print(f"  LLM 输出: {result['output']}")
    print("  PASS")


# ---- 运行所有测试 ----

async def main():
    print("=" * 50)
    print("Pydantic AI 核心龙骨验证")
    print("=" * 50)

    await test_1_basic_tool_call()
    await test_2_dynamic_tool_addition()
    await test_3_deps_mock()
    await test_4_history_processor_called()
    await test_5_tool_mock()

    print("\n" + "=" * 50)
    print("全部通过")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
