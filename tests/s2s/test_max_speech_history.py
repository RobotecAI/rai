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

import importlib.util
from pathlib import Path

import pytest


def _load():
    path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "rai_s2s"
        / "rai_s2s"
        / "positive_params.py"
    )
    spec = importlib.util.spec_from_file_location("pos_params_offline", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return_mod = mod
    return return_mod


@pytest.mark.parametrize("bad", [0, -1, -10])
def test_max_speech_history_rejects_non_positive(bad):
    m = _load()
    with pytest.raises(ValueError, match="max_speech_history must be positive"):
        m.require_positive_int(bad, name="max_speech_history")


@pytest.mark.parametrize("bad", [True, False, 1.5, "8", None])
def test_max_speech_history_rejects_non_int(bad):
    m = _load()
    with pytest.raises(TypeError, match="max_speech_history must be a positive int"):
        m.require_positive_int(bad, name="max_speech_history")


def test_max_speech_history_accepts_positive():
    m = _load()
    assert m.require_positive_int(64, name="max_speech_history") == 64
    assert m.require_positive_int(1, name="max_speech_history") == 1
