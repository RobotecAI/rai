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

import threading
from typing import Any

import pytest
from rai.communication.base_connector import (
    BaseConnector,
    BaseMessage,
    ConnectorException,
)


class DummyMessage(BaseMessage):
    pass


class DummyConnector(BaseConnector[DummyMessage]):
    def send_message(
        self, message: DummyMessage, target: str, **kwargs: Any
    ) -> None:  # pragma: no cover - not needed for tests
        raise NotImplementedError

    def receive_message(
        self, source: str, timeout_sec: float, **kwargs: Any
    ) -> DummyMessage:  # pragma: no cover - not needed for tests
        raise NotImplementedError

    def general_callback_preprocessor(self, message: Any) -> DummyMessage:
        if isinstance(message, DummyMessage):
            return message
        if isinstance(message, dict):
            payload = message.get("payload")
            return DummyMessage(payload=payload)
        raise TypeError("Unsupported message payload")


def test_base_connector_callback_registration_and_dispatch():
    connector = DummyConnector()

    received_messages: list[DummyMessage] = []
    callback_event = threading.Event()

    def callback(message: DummyMessage) -> None:
        received_messages.append(message)
        callback_event.set()

    callback_id = connector.register_callback("demo", callback)

    connector.general_callback("demo", {"payload": "hello"})

    assert callback_event.wait(timeout=1)
    assert len(received_messages) == 1
    assert isinstance(received_messages[0], DummyMessage)
    assert received_messages[0].payload == "hello"

    connector.unregister_callback(callback_id)

    with pytest.raises(ConnectorException):
        connector.unregister_callback(callback_id)

    callback_event.clear()
    connector.general_callback("demo", {"payload": "ignored"})
    assert not callback_event.wait(timeout=0.1)

    connector.shutdown()
