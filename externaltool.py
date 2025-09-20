import os
import asyncio
from typing import Annotated
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.tools.http import HttpTool
from autogen_core.models import UserMessage
from autogen_core.tools import FunctionTool
from autogen_core import CancellationToken
from langchain_community.utilities import GoogleSerperAPIWrapper
from autogen_agentchat.ui import Console
from dotenv import load_dotenv
import random

load_dotenv()

# api_key = os.getenv('GEMINI_API_KEY')
os.environ['SERPER_API_KEY'] = '55dc7c83d352b90f08208ec5465bbf600afbe27c'



search_tool_wrapper = GoogleSerperAPIWrapper(type='news')

def search_web(query:str)->str:
    try:
        return search_tool_wrapper.run(query)
    except Exception as e:
        return f'Search failed: {str(e)}'

model_client = OllamaChatCompletionClient(model='llama3.2')

search_agent = AssistantAgent(
    name='search_agent',
    model_client=model_client,
    system_message="""
    you are a helpful assistant that can search the web to find current information.
    when asked a question, use the search_web tool to find relevant current information and provide a comprehensive answer based on the search results
""",
    description='Searches the internet and provides a detailed answers based on search results',
    tools=[search_web],
    reflect_on_tool_use=True
)

async def demo_search():
    test_queries = [
        "Who won the ICC Men's world cup in 2023?",
        "Who won the ICC last Men's champions trophy?",
        "Who won the last ICC Men's T20 world cup?"
    ]

    for query in test_queries:
        try:
            result = await search_agent.run(task=query)
            print(f'Search Result: {result.messages[-1].content}')
        except Exception as e:
            print(f'Error:{e}')    
async def main():
    await demo_search()

if __name__=="__main__":
    asyncio.run(main())                