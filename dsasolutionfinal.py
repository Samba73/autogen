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
            You are a problem solver agent that is an expert solving DSA problems.
            You will be working with code executor agent to execute code.
            You will be given a task and you should.
            At the beginning of your response you have to specify your plan to solve the task.
            Then you should give the code in a code block. (python)
            You should write code in a one code block at a time and then pass it to code executor agent to execute it.
            Make sure that we have atleast 3 test cases for the code you write.
            Once the code is executed and if the same has been done successfully, you have the results.
            You should explain the code execution result.
            If your solution involves user interaction (user input) assume or provide default user input and provide solution for code executor accordingly.
            Once the code and explanation is done, you should ask the code executor agent to save the code in a file.
            in the format as below (use below code only as format to save the code):
            ```python
            def add_numbers(a, b):
                return a + b

            # Test cases with print statements
            print("Test 1:", add_numbers(2, 3))
            print("Test 2:", add_numbers(-1, 5)) 
            print("Test 3:", add_numbers(0, 0))
            ```

            To save the file, use this format:
            ```python
            # Save the working solution
            solution_code = '''def add_numbers(a, b):
                return a + b

            # Test cases
            print("Test 1:", add_numbers(2, 3))
            print("Test 2:", add_numbers(-1, 5)) 
            print("Test 3:", add_numbers(0, 0))
            print("Solution works correctly!")
                          
            '''

            with open('solution.py', 'w') as f:
                f.write(solution_code)
            print('Code saved successfully in solution.py')
            ```

            You should send the above code block to the code executor agent so that it can save the code in a file. Make sure to provide the code in a code block.
            In the end once code is executed successfully and executed code saved as file, you have to say "SOLUTION COMPLETE" to stop the conversation.        
""")
    
    termination_condition = TextMentionTermination("SOLUTION COMPLETE")

    team = RoundRobinGroupChat(
    participants=[problem_solver_agent, code_executor_agent],
    termination_condition=termination_condition, max_turns=20)

    try:
        await docker.start()
        task ="Write a python code to sort a given list using merge sort"

        async for message in team.run_stream(task=task):
            if isinstance(message, TextMessage):
                print("==" * 50)
                print(message.source, ":", message.content)
                print("==" * 50)
            elif isinstance(message, TaskResult):
                print('Stop Reason:', message.stop_reason)
    except Exception as e:
        print(f'Error occured: {e}')
    finally:
        print('Stopping docker executor')
        await docker.stop()                

if __name__ == "__main__":
    asyncio.run(main())
