import os
import asyncio
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_core.models import UserMessage
from autogen_agentchat.ui import Console
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')

# model_client = OpenAIChatCompletionClient(model='gemini-1.5-flash-8b', api_key=api_key)
model_client = OllamaChatCompletionClient(model='llama3.2')

assistant = AssistantAgent(
    name='assistant',
    model_client=model_client,
    system_message="Give the answer to the question asked with the information or provide response based on information provided. If you cannot complete the task, Just say 'TERMINATE'"
)

team = RoundRobinGroupChat(
    participants=[assistant],
    termination_condition=TextMentionTermination('TERMINATE'),
    max_turns=2
)

async def main():
    task = 'Give me the current weather of Mumbai'

    while True:
        await Console(team.run_stream(task=task))

        feedback = input('Please provide your feedback (type "exit" to stop conversation):')
        if feedback.lower().strip()=='exit':
            break
        task = feedback
        
if __name__ == '__main__':
    asyncio.run(main())        