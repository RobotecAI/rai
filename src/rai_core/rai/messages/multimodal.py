# Copyright (C) 2024 Robotec.AI
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


from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.base import BaseMessage, get_msg_title_repr


class MultimodalMessage(BaseMessage):
    """
    Base class for multimodal messages.

    Attributes
    ----------
    images : Optional[List[str]]
        List of base64 encoded images.
    audios : Optional[Any]
        List of base64 encoded audios.
    """

    images: Optional[List[str]] = None
    audios: Optional[Any] = None

    def __init__(
        self,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)  # type: ignore

        if self.audios not in [None, []]:
            raise ValueError("Audio is not yet supported")

        _content: List[Union[str, Dict[str, Union[Dict[str, str], str]]]] = []

        if isinstance(self.content, str):
            _content.append({"type": "text", "text": self.content})
        else:
            raise ValueError("Content must be a string")  # for now, to guarantee compat

        if isinstance(self.images, list):
            _image_content = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image}",
                    },
                }
                for image in self.images
            ]
            _content.extend(_image_content)
        self.content = _content


class HumanMultimodalMessage(HumanMessage, MultimodalMessage):
    def __repr_args__(self) -> Any:
        args = super().__repr_args__()
        new_args = []
        for k, v in args:
            if k == "content":
                v = [c for c in v if c["type"] != "image_url"]
            elif k == "images":
                imgs_summary = [image[0:10] + "..." for image in v]
                v = f"{len(v)} base64 encoded images: [{', '.join(imgs_summary)}]"
            new_args.append((k, v))
        return new_args

    def _no_img_content(self):
        return [c for c in self.content if c["type"] != "image_url"]

    def pretty_repr(self, html: bool = False) -> str:
        title = get_msg_title_repr(self.type.title() + " Message", bold=html)
        # TODO: handle non-string content.
        if self.name is not None:
            title += f"\nName: {self.name}"
        return f"{title}\n\n{self._no_img_content()}"


class SystemMultimodalMessage(SystemMessage, MultimodalMessage):
    pass


class AIMultimodalMessage(AIMessage, MultimodalMessage):
    pass
