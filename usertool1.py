import os
import asyncio
import json
from typing import Annotated
from autogen_agentchat.base import Handoff
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, HandoffTermination
from autogen_core.models import UserMessage
from autogen_core.tools import FunctionTool
from autogen_core import CancellationToken
from autogen_agentchat.ui import Console
from dotenv import load_dotenv
import random

load_dotenv()

# api_key = os.getenv('GEMINI_API_KEY')

model_client = OllamaChatCompletionClient(model='llama3.2')

async def get_stock_price(ticker:str, date: Annotated[str, "Date in DD-MMM-YYYY"]) -> float:
    return random.uniform(20, 200)

stock_price_tool = FunctionTool(get_stock_price, description='Get the stock price')

canc_token = CancellationToken()

user_message = UserMessage(content="What is the stock price f AAPL on 10-Jan-2025", source="user")


async def main():

    create_result = await model_client.create(
    messages=[user_message], cancellation_token=canc_token)

    print(create_result.content)
    
    assert isinstance(create_result.content, list)
    arguments = json.loads(create_result.content[0].arguments)
    tool_result = await stock_price_tool.run_json(arguments, canc_token)
    tool_result_str = stock_price_tool.return_value_as_string(tool_result)
    print(tool_result_str)

asyncio.run(main())    