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

import asyncio
import importlib.util
import sys
import time
from pathlib import Path

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
    """Load cli.py directly, bypassing the package __init__ which requires ROS2
    sourced. langchain_core is a core rai dependency and is always installed."""
    name = "_rai_cli_timeout_ut"
    if name in sys.modules:
        return sys.modules[name]

    spec = importlib.util.spec_from_file_location(name, _CLI_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cli():
    return _load_cli_module()


def run(cli, cmd, timeout):
    return asyncio.run(cli.run_command(cmd, timeout=timeout))


@pytest.mark.parametrize("timeout", [0, -1])
def test_run_command_rejects_nonpositive(cli, timeout):
    with pytest.raises(ValueError, match="positive"):
        run(cli, ["echo", "x"], timeout)


def test_run_command_rejects_bool(cli):
    with pytest.raises(TypeError, match="positive number"):
        run(cli, ["echo", "x"], False)


def test_run_command_rejects_shell_operators(cli):
    with pytest.raises(ValueError, match="forbidden characters"):
        run(cli, ["echo", "x;", "rm"], 1)


def test_run_command_happy_path(cli):
    assert "'stdout': 'ok\\n'" in run(cli, ["echo", "ok"], 1.0)


def test_run_command_kills_on_timeout(cli):
    start = time.monotonic()
    output = run(cli, ["sleep", "5"], 0.2)
    assert time.monotonic() - start < 2
    assert "no stdout output" in output


def test_cli_offers_no_write_commands(cli):
    """topic pub, service call, action send_goal and param set/delete/load
    would bypass the access lists and the egress hooks."""
    writes = {"pub", "call", "send_goal", "set", "delete", "load", "dump"}
    for tool in cli.ROS2CLIToolkit().get_tools():
        commands = set(
            tool.args_schema.model_json_schema()["properties"]["command"]["enum"]
        )
        assert not commands & writes, tool.name
        assert tool.effect == "read"
