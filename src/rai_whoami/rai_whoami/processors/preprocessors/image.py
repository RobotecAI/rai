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

from typing import Optional

from langchain_core.language_models import BaseChatModel
from rai.initialization import get_llm_model
from rai.messages import HumanMultimodalMessage, SystemMessage

from rai_whoami.models import EmbodimentInfo, EmbodimentSource

from .base import DataPreProcessor


class ImagePreProcessor(DataPreProcessor):
    SYSTEM_PROMPT = "You are an expert at describing robots. You are given images of a robot and you need to describe it in a way that is easy to understand."

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        system_prompt: Optional[str] = None,
    ):
        self.llm = llm or get_llm_model(model_type="complex_model")
        self.system_prompt = system_prompt or self.SYSTEM_PROMPT

    def process(self, input: EmbodimentSource) -> EmbodimentInfo:
        context = [
            SystemMessage(content=self.system_prompt),
            HumanMultimodalMessage(
                content="Describe the image",
                images=input.images,
            ),
        ]
        response = self.llm.invoke(context)
        return EmbodimentInfo(
            description=str(response.content),
        )
