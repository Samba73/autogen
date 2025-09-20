import os
import asyncio
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

assistant = AssistantAgent(
    name='assistant',
    model_client=model_client,
    tools = [stock_price_tool],
    reflect_on_tool_use=True
)

async def main():
    # result = await stock_price_tool.run_json({"ticker": "AAPL", "date": "10-Jul-1973"}, cancellation_token=canc_token)
    task = "Get th stock price for 'AAPL' dated '10-JUL-2025'"


    result = await assistant.run(task=task)
    
    print(next((r.content for m in result.messages if m.type=='ToolCallExecutionEvent' for r in m.content if r.name=='get_stock_price'),None))
asyncio.run(main())    