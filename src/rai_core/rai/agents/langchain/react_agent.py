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

from typing import Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from rai.agents.langchain.agent import LangChainAgent
from rai.agents.langchain.core import (
    ReActAgentState,
    create_react_runnable,
)
from rai.communication.hri_connector import HRIConnector, HRIMessage
from rai.messages.multimodal import SystemMultimodalMessage


class ReActAgent(LangChainAgent):
    def __init__(
        self,
        target_connectors: Dict[str, HRIConnector[HRIMessage]],
        llm: Optional[BaseChatModel] = None,
        tools: Optional[List[BaseTool]] = None,
        state: Optional[ReActAgentState] = None,
        system_prompt: Optional[str | SystemMultimodalMessage] = None,
        stream_response: bool = True,
    ):
        runnable = create_react_runnable(
            llm=llm, tools=tools, system_prompt=system_prompt
        )
        super().__init__(
            target_connectors=target_connectors,
            runnable=runnable,
            state=state,
            stream_response=stream_response,
        )
