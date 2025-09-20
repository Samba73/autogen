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

schema = {
    "type": "object",
    "properties":{
        "fact":{"type":"string","description":"A random cat fact"},
        "length":{"type":"integer","description":"The character length of the cat fact"}
    },
    "required": ["fact"]
}

alt_schema ={
    "type": "object",
    "properties": {
        "fact": {"type": "string"},
        "length": {"type": "integer"}
    },
    "additionalProperties": True
}

fact_tool = HttpTool(
    name="catfact",
    description="Fetch random cat fact from the cat fact API",
    scheme="https",
    host="catfact.ninja",
    path='/fact',
    port=443,
    method="GET",
    return_type="json",
    json_schema=schema
)

async def main():
    assistant = AssistantAgent(
        name="catfact",
        model_client=model_client,
        tools=[fact_tool, reverse_tool],
        reflect_on_tool_use=True,
        system_message='you are a helpful assistant that can fetch random cat facts(call the tool) and reverse the text using tool'
    )

    response = await assistant.on_messages(
        [TextMessage(content='Can you fetch a cat fact using the tool and use the tool to reverse the fetched fact text?', source='user')],
        CancellationToken()
    )
    # print(response)
    # print('\n')

    print(response.chat_message)
asyncio.run(main())    