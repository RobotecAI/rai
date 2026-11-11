# Copyright (C) 2025 Robotec.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, List, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel


class State(TypedDict):
    messages: List[Any]


def create_structured_output_runnable(
    llm: BaseChatModel,
    structured_output: type[BaseModel],
    system_prompt: str | SystemMessage,
) -> CompiledStateGraph:
    """One-node graph that appends ``{"raw", "parsed", "parsing_error"}`` from
    ``llm.with_structured_output(structured_output, include_raw=True)``."""
    model: Runnable[List[BaseMessage], Any] = llm.with_structured_output(
        schema=structured_output, include_raw=True
    )
    system = (
        SystemMessage(content=system_prompt)
        if isinstance(system_prompt, str)
        else system_prompt
    )

    def thinker(state: State, config: RunnableConfig) -> State:
        if state["messages"] and not isinstance(state["messages"][0], SystemMessage):
            state["messages"].insert(0, system)
        state["messages"].append(model.invoke(state["messages"], config))
        return state

    workflow = StateGraph(State)
    workflow.add_node("thinker", thinker)
    workflow.add_edge(START, "thinker")
    return workflow.compile()
