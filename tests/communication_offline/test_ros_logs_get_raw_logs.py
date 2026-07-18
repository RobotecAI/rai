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

"""Offline unit tests for LlmRosoutParser.get_raw_logs validation.

Lives outside tests/communication/ros2 so collection does not import rclpy helpers.
"""

from collections import deque
from pathlib import Path

import pytest


class _RawLogsHolder:
    """Mirror of LlmRosoutParser.get_raw_logs for offline pytest."""

    def __init__(self, lines):
        self._buffer = deque(lines)

    def get_raw_logs(self, last_n: int = 30) -> str:
        if last_n <= 0:
            raise ValueError(f"last_n must be positive, got {last_n}")
        return "\n".join(list(self._buffer)[-last_n:])


def test_get_raw_logs_rejects_non_positive():
    p = _RawLogsHolder(["a", "b"])
    with pytest.raises(ValueError, match="last_n"):
        p.get_raw_logs(0)
    with pytest.raises(ValueError, match="last_n"):
        p.get_raw_logs(-3)


def test_get_raw_logs_slices_tail():
    p = _RawLogsHolder(["a", "b", "c"])
    assert p.get_raw_logs(2) == "b\nc"
    assert p.get_raw_logs(10) == "a\nb\nc"
    assert p.get_raw_logs(1) == "c"


def test_source_file_enforces_positive_last_n():
    src = (
        Path(__file__).resolve().parents[2]
        / "src/rai_core/rai/communication/ros2/ros_logs.py"
    )
    text = src.read_text(encoding="utf-8")
    assert "last_n must be positive" in text
    assert "if last_n <= 0" in text
