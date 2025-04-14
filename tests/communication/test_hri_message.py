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
from PIL import Image
from pydub import AudioSegment
from rai.communication import HRIMessage
from rai.messages import MultimodalMessage as RAIMultimodalMessage


@pytest.fixture
def image():
    img = Image.new("RGB", (100, 100), color="red")
    return img


@pytest.fixture
def audio():
    audio = AudioSegment.silent(duration=1000)
    return audio


def test_initialization(image, audio):
    message = HRIMessage(
        text="Hello",
        images=[image],
        audios=[audio],
        message_author="human",
        communication_id=HRIMessage.generate_communication_id(),
        seq_no=0,
        seq_end=True,
    )

    assert message.text == "Hello"
    assert message.images == [image]
    assert message.audios == [audio]
    assert message.message_author == "human"


def test_repr():
    comm_id = HRIMessage.generate_communication_id()
    message = HRIMessage(
        text="Hello",
        message_author="ai",
        communication_id=comm_id,
        seq_no=0,
        seq_end=True,
    )

    assert (
        repr(message)
        == f"HRIMessage(type=ai, text=Hello, images=[], audios=[], communication_id={comm_id}, seq_no=0, seq_end=True)"
    )


def test_to_langchain_human():
    message = HRIMessage(
        text="Hi there",
        message_author="human",
        communication_id=HRIMessage.generate_communication_id(),
        seq_no=0,
        seq_end=True,
    )
    langchain_message = message.to_langchain()

    assert isinstance(langchain_message, HumanMessage)
    assert langchain_message.content == "Hi there"


def test_to_langchain_ai_multimodal(image, audio):
    message = HRIMessage(
        text="Response",
        images=[image],
        audios=[audio],
        message_author="ai",
        communication_id=HRIMessage.generate_communication_id(),
        seq_no=0,
        seq_end=True,
    )

    with pytest.raises(
        ValueError
    ):  # NOTE: update when https://github.com/RobotecAI/rai/issues/370 is resolved
        _ = message.to_langchain()

    # assert isinstance(langchain_message, AIMultimodalMessage)
    # assert langchain_message.content == "Response"
    # assert langchain_message.images == ["img"]
    # assert langchain_message.audios == ["audio"]


def test_from_langchain_human():
    langchain_message = HumanMessage(content="Hello")
    hri_message = HRIMessage.from_langchain(
        langchain_message,
        communication_id=HRIMessage.generate_communication_id(),
    )

    assert hri_message.text == "Hello"
    assert hri_message.images == []
    assert hri_message.audios == []
    assert hri_message.message_author == "human"


def test_from_langchain_invalid_type():
    langchain_message = LangchainBaseMessage(content="Hi", type="bot")
    with pytest.raises(ValueError):
        HRIMessage.from_langchain(langchain_message)


def test_to_langchain_unspecified_author():
    message = HRIMessage(text="Test", message_author="unspecified")
    with pytest.raises(ValueError):
        message.to_langchain()


def test_from_langchain_missing_type():
    rai_message = RAIMultimodalMessage(content="No type", type="")
    with pytest.raises(ValueError):
        HRIMessage.from_langchain(rai_message)
