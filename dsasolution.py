import asyncio
from math import e
from pydoc import doc
from urllib import robotparser
from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent, ApprovalRequest, ApprovalResponse
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_core import CancellationToken
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.base import TaskResult

model_client = OllamaChatCompletionClient(model="llama3.2")





def simple_approve(request: ApprovalRequest) -> ApprovalResponse:
    """simple approval func that requests user input for code execution approval"""
    print("Code execution approval requested:")
    print("=" * 50)
    print(request.code)
    print("=" * 50)

    while True:
        user_input = input("Do you want to execute this code (y/n)").strip().lower()
        if user_input in ['y', 'yes']:
            return ApprovalResponse(approved=True, reason="Approved by user")
        elif user_input in ['n', 'no']:
            return ApprovalResponse(approved=False, reason="Denied by user")
        else:
            print("Please enter 'y', 'yes', 'n', 'no'")

async def main() -> None:
    docker = DockerCommandLineCodeExecutor(
        work_dir="temp", timeout=120)
    
    code_executor_agent = CodeExecutorAgent(
        name="CodeExecutorAgent",
        code_executor=docker,
        approval_func=simple_approve)
    
    problem_solver_agent = AssistantAgent(
        name="problem_solver_agent",
        model_client=model_client,
        description="An agent that solves DSA problems",
        system_message="""
            you are a expert problem solving agent that can solve DSA problems.
            You will be working with code executor agent to execute the code.
             IMPORTANT: Always wrap your Python code in markdown code blocks like this:
            ```python
              code = '''
            def greet():
                print("Hello from new file!")

            greet()
            ''' 
            with open("new_script.py", "w") as f:
                f.write(code)
            print("New script saved as new_script.py")   
            ```
            At the beginning of your response you have to specify your plan to solve the given DSA task.
            Then you provide an efficient python code in a markdown code block (```python...```) that solves the given problem statement.
            you provide complete, runnable code that can be executed directly.
            you should write the code in one code block at a time and then pass the code to code executor agent for code execution.
            Make sure that we have atleast 3 test cases for the code you write.
            once the code is executed and if the same is done successfully, you have the results.
            You should explain the code execution results.
            After the explanation, make sure to call the code executor agent again execute the code.
            you save the code in
            a file named `solution.py` by providing python code block.
            you should also make sure to call the code executor agent to run it and save the same, like in the below format:
             
 
            In the end once the code is executed successfully, you have to say "SOLUTION COMPLETE"
            and stop the conversation
            Remember: ALWAYS use ```python code blocks for any code you want executed!
""")
    
    termination_condition = TextMentionTermination("SOLUTION COMPLETE")    

    team = RoundRobinGroupChat(
    participants=[problem_solver_agent, code_executor_agent],
    termination_condition=termination_condition, max_turns=10)

    try:
        await docker.start()
        task ="Write a python code to add two numbers"

        async for message in team.run_stream(task=task):
            if isinstance(message, TextMessage):
                print("==" * 50)
                print(message.source, ":", message.content)
                print("==" * 50)
            elif isinstance(message, TaskResult):
                print('Stop Reason:', message.stop_reason)
    except Exception as e:
        print('Error occured: {e}')
    finally:
        print('Stopping docker executor')
        await docker.stop()                

if __name__ == "__main__":
    asyncio.run(main())
