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

"""Offline tests for create_action_server result_timeout guards (no ROS_DISTRO required)."""

import ast
from pathlib import Path

import pytest

_ACTION_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "rai_core"
    / "rai"
    / "communication"
    / "ros2"
    / "api"
    / "action.py"
)


def _extract_result_timeout_guard_source() -> str:
    """Pull the result_timeout validation block from create_action_server as pure Python."""
    source = _ACTION_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "ROS2ActionAPI":
            for item in node.body:
                if (
                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name == "create_action_server"
                ):
                    guard_stmts = []
                    for stmt in item.body:
                        if isinstance(stmt, ast.Assign):
                            targets = [
                                t.id
                                for t in stmt.targets
                                if isinstance(t, ast.Name)
                            ]
                            if targets == ["handle"]:
                                break
                        if isinstance(stmt, ast.Expr) and isinstance(
                            stmt.value, ast.Constant
                        ):
                            continue
                        guard_stmts.append(stmt)
                    assert guard_stmts, (
                        "expected result_timeout guard in create_action_server"
                    )
                    module = ast.Module(body=guard_stmts, type_ignores=[])
                    return ast.unparse(module)
    raise AssertionError("ROS2ActionAPI.create_action_server not found")


def _run_guard(result_timeout):
    code = _extract_result_timeout_guard_source()
    ns: dict = {"result_timeout": result_timeout}
    exec(compile(code, str(_ACTION_PATH), "exec"), ns)  # noqa: S102 — test harness


def test_result_timeout_rejects_zero():
    with pytest.raises(ValueError, match="result_timeout must be positive"):
        _run_guard(0)


def test_result_timeout_rejects_negative():
    with pytest.raises(ValueError, match="result_timeout must be positive"):
        _run_guard(-1)


def test_result_timeout_rejects_bool():
    with pytest.raises(TypeError, match="result_timeout must be a positive number"):
        _run_guard(False)


def test_result_timeout_rejects_str():
    with pytest.raises(TypeError, match="result_timeout must be a positive number"):
        _run_guard("900")


def test_result_timeout_accepts_positive():
    _run_guard(900)
    _run_guard(0.5)
