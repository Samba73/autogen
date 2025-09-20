import os
import asyncio
from autogen_agentchat.base import Handoff
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, HandoffTermination
from autogen_core.models import UserMessage
from autogen_agentchat.ui import Console
from dotenv import load_dotenv

load_dotenv()

# api_key = os.getenv('GEMINI_API_KEY')

api_key = os.getenv("OPENROUTER_API_KEY")

# model_client = OpenAIChatCompletionClient(
#     base_url="https://openrouter.ai/api/v1",
#     model="deepseek/deepseek-r1-0528:free",
#     api_key= api_key,
#     model_info={
#         "family":'deepseek',
#         "vision":True,
#         "function_calling":True,
#         "json_output":False,
#         "structured_output":True
#     }
# )

# model_client = OpenAIChatCompletionClient(model='gemini-1.5-flash-8b', api_key=api_key)
model_client = OllamaChatCompletionClient(model='llama3.2')

assistant = AssistantAgent(
    name='assistant',
    model_client=model_client,
    handoffs=[Handoff(target='user', message='transferring to user')],
    system_message="If you cannot complete the task, just transfer it to user."
)

handoff_termination = HandoffTermination("user")
text_termination = TextMentionTermination("TERMINATE")

termination_condition = handoff_termination | text_termination

team = RoundRobinGroupChat(
    participants=[assistant],
    termination_condition=termination_condition
)

async def main():
    task = 'Give me the current weather of Mumbai'

    await Console(team.run_stream(task=task), output_stats=True)


        
if __name__ == '__main__':
    asyncio.run(main())        