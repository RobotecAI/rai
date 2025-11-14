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

import sys

from rai.initialization import config_initialization


def run_main(monkeypatch, args):
    monkeypatch.setattr(sys, "argv", ["config_init"] + args)
    config_initialization.main()


def test_config_initialization_main_creates_and_respects_force(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)

    run_main(monkeypatch, [])
    created = tmp_path / "config.toml"
    assert created.exists()
    default_content = created.read_bytes()

    created.write_text("custom-config", encoding="utf-8")
    run_main(monkeypatch, [])

    output = capsys.readouterr().out
    assert "config.toml already exists" in output
    assert created.read_text(encoding="utf-8") == "custom-config"

    run_main(monkeypatch, ["--force"])
    output = capsys.readouterr().out
    assert "config.toml created successfully." in output
    assert created.read_bytes() == default_content
