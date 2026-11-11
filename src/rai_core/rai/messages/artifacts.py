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

from typing import TypedDict, Union

from rai.content import Image


class MultimodalArtifact(TypedDict):
    """A tool artifact whose images the agent shows to the model
    (see :func:`rai.messages.with_tool_images`)."""

    images: list[Union[Image, str]]  # str: base64 PNG or JPEG
