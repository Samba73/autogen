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

async def reverse_string(text:str)->str:
    """
    Reverses a given string

    Input: 
        Type: String

    Output:
        Type: String [Reverse of input]

    """
    return text[::-1]

reverse_tool = FunctionTool(reverse_string, description='Reverse given string')

canc_token = CancellationToken()

assistant = AssistantAgent(
    model_client=model_client,
    name="reverse_assitant",
    tools=[reverse_tool],
    reflect_on_tool_use=True,
    system_message='You are a helpful assistant that can reverse a given text using the reverse_string tool. Give result with summary'
)

async def main():
    task = "HelloBaby"
    result = await assistant.run(task=task)
    print(f'Agent reponse: {result.messages[-1].content}')
    print('\n')
    reversed = await reverse_string('HelloBaby')
    print(f'Function response: {reversed}')

asyncio.run(main())