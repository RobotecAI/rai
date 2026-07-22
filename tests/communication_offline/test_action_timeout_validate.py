# Copyright (C) 2026 Robotec.AI
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

"""Offline unit tests for send_goal timeout validation helper text in action.py.

These tests re-bind the same predicate as ``_require_positive_timeout`` in
``rai.communication.ros2.api.action`` without importing rclpy.
"""

from pathlib import Path

import pytest


def _load_helper():
    path = (
        Path(__file__).resolve().parents[2]
        / "src/rai_core/rai/communication/ros2/api/action.py"
    )
    text = path.read_text()
    start = text.index("def _require_positive_timeout")
    # find end at next top-level def or class
    rest = text[start:]
    lines = rest.splitlines(True)
    body = [lines[0]]
    for line in lines[1:]:
        if line.startswith("def ") or line.startswith("class ") or line.startswith("@"):
            break
        body.append(line)
    ns: dict = {}
    exec("".join(body), ns, ns)
    return ns["_require_positive_timeout"]


@pytest.fixture(scope="module")
def require_positive_timeout():
    return _load_helper()


@pytest.mark.parametrize("bad", [0, -1, -0.5, 0.0])
def test_action_timeout_rejects_non_positive(require_positive_timeout, bad):
    with pytest.raises(ValueError, match="timeout_sec must be positive"):
        require_positive_timeout(bad)


def test_action_timeout_rejects_bool(require_positive_timeout):
    with pytest.raises(TypeError, match="timeout_sec must be a positive number"):
        require_positive_timeout(False)


def test_action_timeout_accepts_positive(require_positive_timeout):
    assert require_positive_timeout(1.0) == 1.0
