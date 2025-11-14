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


from langchain_core.messages import AIMessage
from rai.communication import HRIMessage
from rai.communication.hri_connector import HRIConnector


def test_build_message():
    class DummyHRIMessage(HRIMessage):
        pass

    class DummyHRIConnector(HRIConnector[DummyHRIMessage]):
        pass

    connector = DummyHRIConnector()
    message = connector.build_message(
        AIMessage(content="Hello"), "test_conversation_id", 0, False
    )
    assert message.text == "Hello"
    assert message.message_author == "ai"
    assert message.images == []
    assert message.audios == []
    assert isinstance(message, DummyHRIMessage)
