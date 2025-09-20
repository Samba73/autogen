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

narrator = AssistantAgent(
    name='narrator',
    model_client=model_client
)

hero = AssistantAgent(
    name='hero',
    model_client=model_client
)

guide = AssistantAgent(
    name='guide',
    model_client=model_client
)

team = RoundRobinGroupChat(
    participants=[narrator, hero, guide],
    max_turns=1
)

async def main():
    task = 'Write a 3 part story about a mysterious forest'

    while True:
        await Console(team.run_stream(task=task))

        feedback = input('Please provide your feedback (type "exit" to stop conversation):')
        if feedback.lower().strip()=='exit':
            break
        task = feedback
        
if __name__ == '__main__':
    asyncio.run(main())        