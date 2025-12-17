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

from collections import deque

import pytest
from rai.communication.ros2 import waiters


class DummyConnector:
    def __init__(self, services_seq=None, topics_seq=None, actions_seq=None):
        self._services_seq = deque(services_seq or [[]])
        self._topics_seq = deque(topics_seq or [[]])
        self._actions_seq = deque(actions_seq or [[]])

    def get_services_names_and_types(self):
        current = self._services_seq[0]
        if len(self._services_seq) > 1:
            self._services_seq.popleft()
        return current

    def get_topics_names_and_types(self):
        current = self._topics_seq[0]
        if len(self._topics_seq) > 1:
            self._topics_seq.popleft()
        return current

    def get_actions_names_and_types(self):
        current = self._actions_seq[0]
        if len(self._actions_seq) > 1:
            self._actions_seq.popleft()
        return current


def test_wait_for_ros2_services_adds_prefix_and_stops(monkeypatch):
    connector = DummyConnector(
        services_seq=[
            [],
            [("/other", ["srv/Type"])],
            [("/target_service", ["srv/Type"])],
        ]
    )
    monkeypatch.setattr(waiters.time, "sleep", lambda *_: None)

    waiters.wait_for_ros2_services(connector, ["target_service"], time_interval=0)


def test_wait_for_ros2_topics(monkeypatch):
    connector = DummyConnector(
        topics_seq=[
            [],
            [("/topic_a", ["msg/A"])],
        ]
    )
    monkeypatch.setattr(waiters.time, "sleep", lambda *_: None)

    waiters.wait_for_ros2_topics(connector, ["topic_a"], time_interval=0)


def test_wait_for_ros2_actions(monkeypatch):
    connector = DummyConnector(
        actions_seq=[
            [],
            [("/action_a", ["action/A"])],
        ]
    )
    monkeypatch.setattr(waiters.time, "sleep", lambda *_: None)

    waiters.wait_for_ros2_actions(connector, ["action_a"], time_interval=0)


def test_wait_for_ros2_services_timeout(monkeypatch):
    connector = DummyConnector(services_seq=[[]])
    monkeypatch.setattr(waiters.time, "sleep", lambda *_: None)

    with pytest.raises(TimeoutError):
        waiters.wait_for_ros2_services(
            connector, ["target_service"], time_interval=0.001, timeout=0.01
        )


def test_wait_for_ros2_topics_timeout(monkeypatch):
    connector = DummyConnector(topics_seq=[[]])
    monkeypatch.setattr(waiters.time, "sleep", lambda *_: None)

    with pytest.raises(TimeoutError):
        waiters.wait_for_ros2_topics(
            connector, ["target_topic"], time_interval=0.001, timeout=0.01
        )


def test_wait_for_ros2_actions_timeout(monkeypatch):
    connector = DummyConnector(actions_seq=[[]])
    monkeypatch.setattr(waiters.time, "sleep", lambda *_: None)

    with pytest.raises(TimeoutError):
        waiters.wait_for_ros2_actions(
            connector, ["target_action"], time_interval=0.001, timeout=0.01
        )


@pytest.mark.parametrize(
    "seq_type, seq_arg, wait_func, name",
    [
        ("topics_seq", [[]], waiters.wait_for_ros2_topics, "target_topic"),
        ("actions_seq", [[]], waiters.wait_for_ros2_actions, "target_action"),
        ("services_seq", [[]], waiters.wait_for_ros2_services, "target_service"),
    ],
)
def test_wait_for_ros2_negative_timeout(
    monkeypatch, seq_type, seq_arg, wait_func, name
):
    connector = DummyConnector(**{seq_type: seq_arg})
    monkeypatch.setattr(waiters.time, "sleep", lambda *_: None)

    with pytest.raises(ValueError):
        wait_func(connector, [name], time_interval=0.001, timeout=-0.01)
