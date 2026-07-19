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

"""Offline tests for ros_logs bufsize validation (no rclpy import)."""

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
ROS_LOGS = REPO / "src/rai_core/rai/communication/ros2/ros_logs.py"


def _load_validate_bufsize():
    """Load validate_bufsize via AST exec of the pure function only."""
    source = ROS_LOGS.read_text()
    module = ast.parse(source)
    fn = None
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "validate_bufsize":
            fn = node
            break
    assert fn is not None, "validate_bufsize must exist in ros_logs.py"
    code = compile(ast.Module(body=[fn], type_ignores=[]), str(ROS_LOGS), "exec")
    ns: dict = {}
    exec(code, ns)
    return ns["validate_bufsize"]


validate_bufsize = _load_validate_bufsize()


@pytest.mark.parametrize("bad", [0, -1, -100])
def test_validate_bufsize_rejects_nonpositive(bad):
    with pytest.raises(ValueError, match="bufsize must be positive"):
        validate_bufsize(bad)


def test_validate_bufsize_accepts_positive():
    assert validate_bufsize(1) == 1
    assert validate_bufsize(100) == 100
