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

"""The LLM boundary: RAI content to LangChain content blocks and back."""

import base64
from collections.abc import Iterable, Sequence
from typing import Any, Union

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage

from rai.content import Audio, Content, Image, Text


def _image_block(image: Union[Image, str]) -> dict[str, Any]:
    if isinstance(image, Image):
        return {"type": "image", "base64": image.encode(), "mime_type": "image/jpeg"}
    mime = "image/png" if image.startswith("iVBOR") else "image/jpeg"
    return {"type": "image", "base64": image, "mime_type": mime}


def to_blocks(content: Iterable[Union[Content, str]]) -> list[dict[str, Any]]:
    """LangChain content blocks of RAI content; ``str`` is text. Images are
    JPEG-encoded here, the only place they are encoded for the model."""
    blocks: list[dict[str, Any]] = []
    for item in content:
        if isinstance(item, (str, Text)):
            blocks.append({"type": "text", "text": getattr(item, "text", item)})
        elif isinstance(item, Image):
            blocks.append(_image_block(item))
        elif isinstance(item, Audio):
            wav = base64.b64encode(item.to_wav()).decode("ascii")
            blocks.append({"type": "audio", "base64": wav, "mime_type": "audio/wav"})
        else:
            raise TypeError(f"Not RAI content: {type(item).__name__}")
    return blocks


def from_blocks(
    content: Union[str, Sequence[Union[str, dict[str, Any]]]],
) -> list[Content]:
    """RAI content of a LangChain message's content. Base64 images and WAV
    audio are decoded; blocks of other kinds (e.g. URLs, tool calls) are skipped."""
    if isinstance(content, str):
        return [Text(content)] if content else []
    out: list[Content] = []
    for block in content:
        if isinstance(block, str):
            out.append(Text(block))
            continue
        kind = block.get("type")
        if kind == "text":
            out.append(Text(block["text"]))
        elif kind == "image" and "base64" in block:
            out.append(Image.decode(block["base64"]))
        elif kind == "image_url":
            url = block["image_url"]
            url = url["url"] if isinstance(url, dict) else url
            if url.startswith("data:"):
                out.append(Image.decode(url.split(",", 1)[1]))
        elif kind == "audio" and "base64" in block:
            out.append(Audio.from_wav(base64.b64decode(block["base64"])))
        elif kind == "input_audio" and block["input_audio"].get("format") == "wav":
            out.append(Audio.from_wav(base64.b64decode(block["input_audio"]["data"])))
    return out


def human(
    text: str = "",
    images: Iterable[Union[Image, str]] = (),
    audios: Iterable[Audio] = (),
) -> HumanMessage:
    """A HumanMessage of text, images (an Image or base64 PNG/JPEG) and audio."""
    images, audios = list(images), list(audios)
    if not images and not audios:
        return HumanMessage(text)
    blocks = to_blocks([text] if text else [])
    return HumanMessage([*blocks, *map(_image_block, images), *to_blocks(audios)])


def with_tool_images(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """Show tool images to the model: after each run of tool messages, add one
    human message with the images of their ``{"images": [...]}`` artifacts.

    Most providers reject images inside tool results, so this is done per model
    request; the history keeps the tool messages unchanged.
    """
    out: list[BaseMessage] = []
    images: list[Union[Image, str]] = []
    for message in messages:
        if isinstance(message, ToolMessage):
            if isinstance(message.artifact, dict):
                images += message.artifact.get("images") or []
        elif images:
            out.append(human("Images returned by tools:", images))
            images = []
        out.append(message)
    if images:
        out.append(human("Images returned by tools:", images))
    return out
