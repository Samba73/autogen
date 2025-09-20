from ast import Assign
import os
import asyncio
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_core.models import UserMessage
from autogen_agentchat.ui import Console
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')

model_client = OpenAIChatCompletionClient(model='gemini-1.5-flash-8b', api_key=api_key)


assistant = AssistantAgent(
    name='assistant',
    model_client=model_client,
    system_message='You are a helpful assistant'
)

user_proxy_agent = UserProxyAgent(
    name='human_agent',
    description='a human interation agent',
    input_func=input
)

termination = TextMentionTermination('APPROVE')

team = RoundRobinGroupChat(
    participants=[assistant, user_proxy_agent],
    termination_condition=termination
)

stream = team.run_stream(task = 'Write a 4 line poem on long distance love')

async def main():
    await Console(stream)

if __name__ == "__main__":
    asyncio.run(main())