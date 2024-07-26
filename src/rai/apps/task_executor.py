from typing import List

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool

TASK_EXECUTOR_PROMPT = """
I am a robot's execution task system. I am responsible for executing the current task using reasoning and available tooling. I am focused on the task's completion.
When the simplest solution is not enough, I will reason about the task and use available tools to complete it.
My job is complete whenever the task is finished or the task is deemed impossible to complete.
"""

TASK_SUMMARIZER_PROMPT = """
I am a robot's task summarizer. I am responsible for summarizing the task's completion status.
"""


class TaskState(BaseModel):
    done: bool = Field(
        ..., description="True if the robot has reached the final state of the task."
    )
    summary: str = Field(
        ..., description="Message describing the robot's mission fulfilness."
    )


class Task(BaseModel):
    task: str = Field(..., description="The task to be executed by the robot.")


def task_executor(
    llm: BaseChatModel,
    task: Task,
    tools: List[BaseTool],
    task_executor_prompt: str = TASK_EXECUTOR_PROMPT,
    task_summarizer_prompt: str = TASK_SUMMARIZER_PROMPT,
) -> TaskState:
    agent_template = ChatPromptTemplate.from_messages(
        [
            ("system", task_executor_prompt),
            ("human", "The task: {task}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=agent_template)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True, return_intermediate_steps=True
    )

    agent_output = agent_executor.invoke({"task": task.task})

    agent_reasoning = f"intermediate_steps: {str(agent_output['intermediate_steps'])} final_output: {agent_output['output']}"
    history = [
        SystemMessage(content=task_summarizer_prompt),
        HumanMessage(content=agent_reasoning, name="agent"),
    ]
    task_state = llm.with_structured_output(schema=TaskState).invoke(history)

    return task_state
