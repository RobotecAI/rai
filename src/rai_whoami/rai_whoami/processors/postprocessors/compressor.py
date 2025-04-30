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

from rai_whoami.models import EmbodimentInfo

from .base import DataPostProcessor


class CompressorPostProcessor(DataPostProcessor):
    SYSTEM_PROMPT = "You are an expert at compressing text. You are given a text and you need to compress it to reduce redundancy."

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        system_prompt: Optional[str] = None,
    ):
        self.llm = llm or get_llm_model(model_type="complex_model")
        self.system_prompt = system_prompt or self.SYSTEM_PROMPT

    def process(self, input: EmbodimentInfo) -> EmbodimentInfo:
        prompt = (
            f"<description>\n{input.description}\n</description>"
            f"<rules>\n{input.rules}\n</rules>"
            f"<capabilities>\n{input.capabilities}\n</capabilities>"
            f"<behaviors>\n{input.behaviors}\n</behaviors>"
        )
        context = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt),
        ]
        llm_with_structured_output = self.llm.with_structured_output(EmbodimentInfo)
        response = cast(EmbodimentInfo, llm_with_structured_output.invoke(context))
        return response
