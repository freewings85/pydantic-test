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


@agent.tool
async def get_weather(ctx: RunContext[None], city: str) -> str:
    """Get the current weather for a given city."""
    async with httpx.AsyncClient(timeout=30) as http_client:
        response = await http_client.get(
            f"https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
        )
        geo = response.json()["results"][0]
        lat, lon = geo["latitude"], geo["longitude"]

        weather_resp = await http_client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,weather_code",
            },
        )
        current = weather_resp.json()["current"]
        return (
            f"{city}: {current['temperature_2m']}°C, "
            f"weather code {current['weather_code']}, "
            f"humidity {current['relative_humidity_2m']}%"
        )


async def main():
    result = await agent.run("What's the weather like in Beijing?")
    print(f"output: {result.output}")


if __name__ == "__main__":
    asyncio.run(main())
