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


import pytest
from langchain_core.messages import BaseMessage as LangchainBaseMessage
from langchain_core.messages import HumanMessage

from rai.communication import HRIMessage, HRIPayload
from rai.messages.multimodal import MultimodalMessage as RAIMultimodalMessage


def test_initialization():
    payload = HRIPayload(text="Hello", images=["image1"], audios=["audio1"])
    message = HRIMessage(payload=payload, message_author="human")

    assert message.text == "Hello"
    assert message.images == ["image1"]
    assert message.audios == ["audio1"]
    assert message.message_author == "human"


def test_repr():
    payload = HRIPayload(text="Hello", images=None, audios=None)
    message = HRIMessage(payload=payload, message_author="ai")

    assert repr(message) == "HRIMessage(type=ai, text=Hello, images=None, audios=None)"


def test_to_langchain_human():
    payload = HRIPayload(text="Hi there", images=None, audios=None)
    message = HRIMessage(payload=payload, message_author="human")
    langchain_message = message.to_langchain()

    assert isinstance(langchain_message, HumanMessage)
    assert langchain_message.content == "Hi there"


def test_to_langchain_ai_multimodal():
    payload = HRIPayload(text="Response", images=["img"], audios=["audio"])
    message = HRIMessage(payload=payload, message_author="ai")

    with pytest.raises(
        ValueError
    ):  # NOTE: update when https://github.com/RobotecAI/rai/issues/370 is resolved
        _ = message.to_langchain()

    # assert isinstance(langchain_message, AiMultimodalMessage)
    # assert langchain_message.content == "Response"
    # assert langchain_message.images == ["img"]
    # assert langchain_message.audios == ["audio"]


def test_from_langchain_human():
    langchain_message = HumanMessage(content="Hello")
    hri_message = HRIMessage.from_langchain(langchain_message)

    assert hri_message.text == "Hello"
    assert hri_message.images is None
    assert hri_message.audios is None
    assert hri_message.message_author == "human"


def test_from_langchain_invalid_type():
    langchain_message = LangchainBaseMessage(content="Hi", type="bot")
    with pytest.raises(ValueError):
        HRIMessage.from_langchain(langchain_message)


def test_to_langchain_invalid_author():
    payload = HRIPayload(text="Test")
    message = HRIMessage(payload=payload, message_author="invalid")
    with pytest.raises(ValueError):
        message.to_langchain()


def test_from_langchain_missing_type():
    rai_message = RAIMultimodalMessage(content="No type", type="")
    with pytest.raises(ValueError):
        HRIMessage.from_langchain(rai_message)
