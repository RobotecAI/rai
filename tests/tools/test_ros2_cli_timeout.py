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
import types
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
    """Load cli.py directly, bypassing the package __init__ (which requires
    ROS2 sourced) and stubbing langchain_core.tools if it is not installed."""
    name = "_rai_cli_timeout_ut"
    if name in sys.modules:
        return sys.modules[name]

    # Provide a minimal langchain_core.tools stub only when the real package is
    # unavailable, so the offline import of cli.py succeeds without ROS2/langchain.
    if importlib.util.find_spec("langchain_core") is None:
        lc = types.ModuleType("langchain_core")
        lc_tools = types.ModuleType("langchain_core.tools")
        lc_tools.BaseTool = object
        lc_tools.BaseToolkit = type("BaseToolkit", (), {"get_tools": lambda self: []})
        lc_tools.tool = lambda f=None, **_k: (f if f is not None else (lambda x: x))
        lc.tools = lc_tools
        sys.modules.setdefault("langchain_core", lc)
        sys.modules.setdefault("langchain_core.tools", lc_tools)

    spec = importlib.util.spec_from_file_location(name, _CLI_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
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
