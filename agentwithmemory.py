import asyncio
import os
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_core.memory import ListMemory, MemoryContent
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.ui import Console
from autogen_core.models import AssistantMessage, LLMMessage, ModelFamily

# model_client = OllamaChatCompletionClient(model="llama3.2")
load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')

model_client = OllamaChatCompletionClient(
    model="deepseek-r1:8b",
    model_info={
        "vision": False,
        "function_calling": False,
        "json_output": False,
        "family": ModelFamily.R1,
        "structured_output": True,
    },
)

async def main() -> None:
    # model_client = OpenAIChatCompletionClient(model='gemini-1.5-flash-8b', api_key=api_key)
    memory = ListMemory()
    await memory.add(MemoryContent(content="User likes pizza", mime_type="text/plain"))
    await memory.add(MemoryContent(content="User dislikes cheese",mime_type="text/plain"))

    agent = AssistantAgent(
        name ="assistant",
        model_client=model_client,
        memory=[memory],
        system_message="You are a helpful assistant"

    )

    result = await agent.run(task="What is a good dinner idea?")
    print(result.messages[-1].content)

asyncio.run(main())