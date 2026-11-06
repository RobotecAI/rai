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

"""The openWakeWord pre-trained models are CC BY-NC-SA 4.0: one is fetched only when
it is named exactly, and an own model fetches none of them."""

import pytest

pytest.importorskip("openwakeword")

from rai_s2s.asr.models import open_wake_word  # noqa: E402


@pytest.fixture
def fetched(monkeypatch):
    names = []
    monkeypatch.setattr(
        open_wake_word, "download_models", lambda model_names: names.extend(model_names)
    )
    monkeypatch.setattr(
        open_wake_word,
        "OWWModel",
        lambda wakeword_models, inference_framework: wakeword_models,
    )
    return names


def test_a_named_pretrained_model_fetches_that_model_only(fetched):
    oww = open_wake_word.OpenWakeWord("hey jarvis")
    assert fetched == ["hey_jarvis_v0.1"]
    assert oww.model == ["hey_jarvis"]


def test_an_own_model_fetches_no_pretrained_model(fetched, tmp_path):
    own = tmp_path / "timer"
    own.write_bytes(b"")
    oww = open_wake_word.OpenWakeWord(str(own))
    assert fetched == [str(own)]
    assert oww.model == [str(own)]


@pytest.mark.parametrize("name", ["", " ", "_", "hey", "jarvis"])
def test_anything_else_is_refused(fetched, name):
    with pytest.raises(ValueError, match="CC BY-NC-SA"):
        open_wake_word.OpenWakeWord(name)
    assert fetched == []
