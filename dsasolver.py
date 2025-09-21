import asyncio
import os
from dotenv import load_dotenv

from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.agents import AssistantAgent,CodeExecutorAgent, ApprovalRequest, ApprovalResponse
from autogen_agentchat.messages import TextMessage
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_core.models import AssistantMessage, ModelFamily
from autogen_core import CancellationToken


load_dotenv()


model_client = OllamaChatCompletionClient(model="llama3.2")

problem_solver = AssistantAgent(
    name="problem_solver",
    model_client=model_client,
    system_message="""
    you are a expert problem solving agent.
    you provide an efficient python code to solve the given problem statement.
    you provide complete code that can run to provide the solution to the given problem.
    Once completed with solution end the conversation with SOLUTION_COMPLETE
"""

)


team = RoundRobinGroupChat(
        participants=[problem_solver],
        termination_condition=MaxMessageTermination(max_messages=3) | TextMentionTermination("SOLUTION_COMPLETE")
    )
async def test_team():
    try:
        task = TextMessage(content='Write an efficient python code to sort list using merge sort', source='user')

        result = await team.run(task=task)
        print("=== CONVERSATION RESULTS ===")
        for each_message in result.messages:
            print(f'{each_message.source}: {each_message.content}')
    except asyncio.TimeoutError:
        print('The operation timed out after 60 seconds...')
    except Exception as e:
        print(f'An error occured: {e}')    


async def main():
    print("Starting Autogen Team....")
    await test_team()

if __name__ == "__main__":
    asyncio.run(main())


