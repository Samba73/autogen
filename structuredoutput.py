import asyncio
from typing import Literal

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_core.tools import FunctionTool
from autogen_ext.models.ollama import OllamaChatCompletionClient
from pydantic import BaseModel
from sqlalchemy import desc

class AgentResponse(BaseModel):
    thougths: str
    response: Literal['happy', 'sad', 'neutral']

def sentiment_analysis(txt: str) -> str:
    """Given a text, return the sentiment"""
    return 'happy' if 'happy' in txt else 'sad' if 'sad' in txt else 'neutral'    

tool = FunctionTool(sentiment_analysis,description="Sentiment Analysis")


model_client = OllamaChatCompletionClient(model="llama3.2", response_format=AgentResponse)

agent = AssistantAgent(name="assistant", model_client=model_client, tools=[tool],
                       system_message="Use the tool to analyse sentiment"
                    #    response_format=AgentResponse,
                    #    ,output_content_type=AgentResponse)
)
async def main():
    stream = agent.run_stream(task="I'm happy today!")
    await Console(stream)

asyncio.run(main())