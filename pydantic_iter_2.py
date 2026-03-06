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
from pydantic_graph import End

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
    async with agent.iter('What is the capital of France?') as agent_run:
        node = agent_run.next_node

        all_nodes = [node]

        # Drive the iteration manually:
        while not isinstance(node, End):  
            node = await agent_run.next(node)  
            all_nodes.append(node)
        
        print(all_nodes)

if __name__ == "__main__":
    asyncio.run(main())