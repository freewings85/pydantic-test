#验证pydantic的agent iter

import asyncio
import os
import httpx
import logfire
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv

from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

load_dotenv()

logfire.configure(
    send_to_logfire=False,
    additional_span_processors=[
        SimpleSpanProcessor(
            OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces")
        )
    ],
)
logfire.instrument_pydantic_ai()
logfire.instrument_httpx()

client = AsyncAzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["AZURE_API_VERSION"],
)

provider = OpenAIProvider(openai_client=client)
model = OpenAIChatModel(os.environ["AZURE_DEPLOYMENT_NAME"], provider=provider)
agent = Agent(
    model,
    system_prompt="You are a helpful assistant. Use the get_weather tool when asked about weather.",
)

async def main():
    nodes = []
    # Begin an AgentRun, which is an async-iterable over the nodes of the agent's graph
    async with agent.iter('What is the capital of France?') as agent_run:
        async for node in agent_run:
            # Each node represents a step in the agent's execution
            nodes.append(node)
    print(nodes)

if __name__ == "__main__":
    asyncio.run(main())