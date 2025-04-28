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

from typing import Optional, cast

from langchain_core.language_models import BaseChatModel
from rai.initialization import get_llm_model
from rai.messages import HumanMessage, SystemMessage

from rai_whoami.models import EmbodimentInfo, EmbodimentSource

from .base import DataPreProcessor


class DocsPreProcessor(DataPreProcessor):
    SYSTEM_PROMPT = "You are an expert at deriving robot Embodiment Information from documentation. You are given documentation of a robot and you need to derive the robot Embodiment Information from it."

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        system_prompt: Optional[str] = None,
    ):
        self.llm = llm or get_llm_model(model_type="complex_model")
        self.system_prompt = system_prompt or self.SYSTEM_PROMPT

    def process(self, input: EmbodimentSource) -> EmbodimentInfo:
        if len(input.documentation) == 0:
            return EmbodimentInfo()
        context = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(
                content="\n".join([doc.page_content for doc in input.documentation])
            ),
        ]
        llm_with_structured_output = self.llm.with_structured_output(EmbodimentInfo)
        response = cast(EmbodimentInfo, llm_with_structured_output.invoke(context))
        return response
