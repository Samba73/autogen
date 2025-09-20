import asyncio
from autogen_agentchat.agents import CodeExecutorAgent, ApprovalRequest, ApprovalResponse
from autogen_agentchat.messages import TextMessage
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_core import CancellationToken

def simple_approval_func(request: ApprovalRequest)-> ApprovalResponse:
    """simple approval func that requests user input for code execution approval"""
    print("Code execution approval requested:")
    print("=" * 50)
    print(request.code)
    print("=" * 50)

    while True:
        user_input = input("Do you want to execute this code (y/n)?").strip().lower()
        if user_input in ['y', 'yes']:
            return ApprovalResponse(approved=True, reason="Approved by user")
        elif user_input in ['n', 'no']:
            return ApprovalResponse(approved=False, reason="Denied by user")
        else:
            print("Please enter 'y', 'yes, 'n' or 'no'")

async def run_code_executor_agent() -> None:
    code_executor = DockerCommandLineCodeExecutor(work_dir="temp",timeout=120)
    await code_executor.start()
    code_executor_agent = CodeExecutorAgent(
        name="code_executor",
        code_executor=code_executor,
        approval_func=simple_approval_func
    )
    task = TextMessage(
        content='''Here is some code
```python
print('Hello world')
```
''',
        source="user",
    )

    response = await code_executor_agent.on_messages([task], CancellationToken())
    print(response.chat_message.content)

    await code_executor.stop()
asyncio.run(run_code_executor_agent())    