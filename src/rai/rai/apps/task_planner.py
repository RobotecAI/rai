from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

PLANNER_PROMPT = """
I am a part of a robot's planning system. I am responsible for planning the robot's actions by splitting tasks in to atomic steps.
"""


class StepsList(BaseModel):
    steps: list[str] = Field(
        ...,
        description="List of steps to be performed by the robot to complete the task.",
    )


def robot_agnostic_planner(
    llm: BaseChatModel,
    robot_constitution: str,
    task: str,
    planner_prompt: str = PLANNER_PROMPT,
) -> StepsList:
    planner_template = ChatPromptTemplate.from_messages(
        [
            ("system", robot_constitution),
            ("system", planner_prompt),
            ("human", "The task: {input}"),
        ]
    )

    planner = llm.with_structured_output(schema=StepsList)

    chain = planner_template | planner
    return chain.invoke({"input": task})


def robot_specific_planner(
    llm: BaseChatModel,
    robot_constitution: str,
    platfrom_specific_informations: str,
    task: str,
    planner_prompt: str = PLANNER_PROMPT,
) -> StepsList:
    planner_template = ChatPromptTemplate.from_messages(
        [
            ("system", robot_constitution),
            ("system", platfrom_specific_informations),
            ("system", planner_prompt),
            ("human", "The task: {input}"),
        ]
    )

    planner = llm.with_structured_output(schema=StepsList)

    chain = planner_template | planner
    return chain.invoke({"input": task})
