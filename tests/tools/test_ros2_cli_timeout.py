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

"""Offline tests for ROS2 CLI timeout guards (no ROS_DISTRO required)."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_CLI_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "rai_core"
    / "rai"
    / "tools"
    / "ros2"
    / "cli.py"
)


def _load_cli_module():
    """Load cli.py without package __init__ (which requires ROS2 sourced)."""
    name = "_rai_cli_timeout_ut"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _CLI_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except ModuleNotFoundError:
        del sys.modules[name]
        # langchain_core missing: still load pure helpers with lightweight stubs
        from subprocess import PIPE, Popen
        from threading import Timer
        from typing import List, Literal, Optional

        src = _CLI_PATH.read_text(encoding="utf-8")
        ns = {
            "__name__": name,
            "PIPE": PIPE,
            "Popen": Popen,
            "Timer": Timer,
            "List": List,
            "Literal": Literal,
            "Optional": Optional,
            "BaseTool": object,
            "BaseToolkit": type("BaseToolkit", (), {"get_tools": lambda self: []}),
            "tool": lambda f=None, **_k: (f if f is not None else (lambda x: x)),
        }
        exec(compile(src, str(_CLI_PATH), "exec"), ns)
        mod = type(sys)(name)
        mod.__dict__.update(ns)
        sys.modules[name] = mod
    return mod


@pytest.fixture(scope="module")
def cli():
    return _load_cli_module()


def test_run_with_timeout_rejects_zero(cli):
    with pytest.raises(ValueError, match="positive"):
        cli.run_with_timeout(["echo", "x"], 0)


def test_run_with_timeout_rejects_negative(cli):
    with pytest.raises(ValueError, match="positive"):
        cli.run_with_timeout(["echo", "x"], -1)


def test_run_with_timeout_rejects_bool(cli):
    with pytest.raises(TypeError, match="positive number"):
        cli.run_with_timeout(["echo", "x"], False)  # type: ignore[arg-type]


def test_run_command_rejects_nonpositive(cli):
    with pytest.raises(ValueError, match="positive"):
        cli.run_command(["echo", "ok"], timeout=0)


def test_run_with_timeout_happy_path(cli, monkeypatch):
    proc = MagicMock()
    proc.communicate.return_value = (b"out\n", b"")
    monkeypatch.setattr(cli, "Popen", lambda *a, **k: proc)
    monkeypatch.setattr(cli, "Timer", lambda *a, **k: MagicMock())
    stdout, stderr = cli.run_with_timeout(["echo", "ok"], 1.0)
    assert stdout == b"out\n"
    assert stderr == b""
