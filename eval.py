"""
Pydantic Evals 评估示例 - 独立脚本，不依赖 pytest
评估天气 agent 的回答质量
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import (
    Evaluator,
    EvaluatorContext,
    HasMatchingSpan,
    LLMJudge,
)

from main import agent, model


# --- 自定义代码评估器 ---


@dataclass
class ContainsTemperature(Evaluator[str, str]):
    """检查回答中是否包含温度信息"""

    async def evaluate(self, ctx: EvaluatorContext[str, str]) -> dict:
        output = ctx.output.lower()
        has_temp = "°c" in output or "°f" in output or "celsius" in output or "fahrenheit" in output
        return {"contains_temperature": has_temp}


@dataclass
class MentionsCity(Evaluator[str, str]):
    """检查回答中是否提到了被问到的城市"""

    async def evaluate(self, ctx: EvaluatorContext[str, str]) -> dict:
        city = ctx.expected_output.lower() if ctx.expected_output else ""
        return {"mentions_city": city in ctx.output.lower()}


# --- 构建测试数据集 ---

dataset = Dataset(
    cases=[
        Case(
            name="beijing_weather",
            inputs="What's the weather like in Beijing?",
            expected_output="Beijing",
            metadata={"language": "en", "city": "Beijing"},
        ),
        Case(
            name="tokyo_weather",
            inputs="Tell me the current weather in Tokyo",
            expected_output="Tokyo",
            metadata={"language": "en", "city": "Tokyo"},
        ),
        Case(
            name="shanghai_weather",
            inputs="What's the weather in Shanghai?",
            expected_output="Shanghai",
            metadata={"language": "en", "city": "Shanghai"},
        ),
    ],
    evaluators=[
        # 1. 代码评估器：检查格式
        ContainsTemperature(),
        MentionsCity(),

        # 2. LLM Judge：评估回答质量
        LLMJudge(
            rubric=(
                "The response should: "
                "1) Mention the specific city asked about. "
                "2) Include temperature information. "
                "3) Be concise and natural. "
                "4) Not contain hallucinated or made-up weather data."
            ),
            include_input=True,
            model=model,
        ),

        # 3. Span-Based 评估器：检查执行过程
        # 验证 agent 确实调用了 get_weather tool
        # span 名是 "running tool"，工具名在 attribute "gen_ai.tool.name" 里
        HasMatchingSpan(
            query={"has_attributes": {"gen_ai.tool.name": "get_weather"}},
            evaluation_name="called_weather_tool",
        ),
        # 验证有 HTTP 请求发出（说明 tool 真的调了外部 API）
        # span 名是 "GET"，URL 在 attribute "http.method" 里
        HasMatchingSpan(
            query={"name_equals": "GET"},
            evaluation_name="made_http_request",
        ),
        # 验证没有调用危险操作（安全性检查示例）
        HasMatchingSpan(
            query={"not_": {"name_contains": "DELETE"}},
            evaluation_name="no_delete_operations",
        ),
    ],
)


# --- Task 函数 ---


async def weather_task(inputs: str) -> str:
    result = await agent.run(inputs)
    return result.output


# --- 运行评估 ---

if __name__ == "__main__":
    report = dataset.evaluate_sync(weather_task)

    # 1. 终端打印
    report.print(include_input=True, include_output=True)

    # 2. 输出到文件
    output_dir = Path("eval-reports")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 文本报告
    txt_path = output_dir / f"report_{timestamp}.txt"
    txt_path.write_text(report.render(include_input=True, include_output=True))
    print(f"\nReport saved to: {txt_path}")

    # JSON 报告（完整结构化数据）
    import json

    json_path = output_dir / f"report_{timestamp}.json"
    json_data = {
        "name": report.name,
        "timestamp": timestamp,
        "cases": [
            {
                "name": c.name,
                "inputs": c.inputs,
                "output": c.output,
                "scores": {k: v for k, v in (c.scores or {}).items()},
                "assertions": {k: v.value if hasattr(v, "value") else str(v) for k, v in (c.assertions or {}).items()},
                "metrics": {k: v for k, v in (c.metrics or {}).items()},
                "duration": float(c.task_duration) if c.task_duration else None,
            }
            for c in report.cases
        ],
    }
    averages = report.averages()
    if averages:
        json_data["averages"] = {
            "scores": {k: v for k, v in (averages.scores or {}).items()},
            "metrics": {k: v for k, v in (averages.metrics or {}).items()},
        }
    json_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False, default=str))
    print(f"JSON saved to: {json_path}")
