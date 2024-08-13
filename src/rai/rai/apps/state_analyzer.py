from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

STATE_ANALYZER_PROMPT = """
I am a part of a robot's supervising system. I am responsible for analyzing current robot's state with respect to the planned mission.
"""


class State(BaseModel):
    anomaly: bool = Field(
        ..., description="True if the robot's state is not as expected."
    )
    should_continue: bool = Field(
        ...,
        description="True if the robot should continue the task. False if the task is unrecoverable.",
    )
    message: str = Field(..., description="Message describing the robot's state.")


def robot_state_analyzer(
    llm: BaseChatModel,
    task: str,
    state: str,
    state_analyzer_prompt: str = STATE_ANALYZER_PROMPT,
) -> State:

    template = ChatPromptTemplate.from_messages(
        [
            ("system", state_analyzer_prompt),
            ("human", "Current task: {task}\nThe robot's state: {state}"),
        ]
    )

    status_analyzer = template | llm.with_structured_output(schema=State)
    return status_analyzer.invoke({"task": task, "state": state})
