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


import logging
from functools import partial
from typing import Optional

from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import (
    SystemMessage,
)
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from rai.agents.langchain.core.conversational_agent import State, agent


def create_structured_output_runnable(
    llm: BaseChatModel,
    structured_output: type[BaseModel],
    system_prompt: str | SystemMessage,
    logger: Optional[logging.Logger] = None,
    debug: bool = False,
) -> CompiledStateGraph:
    _logger = None
    if logger:
        _logger = logger
    else:
        _logger = logging.getLogger(__name__)

    _logger.info("Creating structured output runnable")

    llm_with_structured_output = llm.with_structured_output(
        schema=structured_output, include_raw=True
    )

    workflow = StateGraph(State)

    workflow.add_node(
        "thinker", partial(agent, llm_with_structured_output, _logger, system_prompt)
    )

    workflow.add_edge(START, "thinker")

    app = workflow.compile(debug=debug)
    _logger.info("State based agent created")
    return app
