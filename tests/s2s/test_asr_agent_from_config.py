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

"""from_config dispatches on config strings, so the values TRANSCRIBE_MODELS
advertises and the configurator writes have to be the ones it matches on."""

from pathlib import Path

import pytest

from rai_s2s.asr import models
from rai_s2s.asr.agents.asr_agent import SpeechRecognitionAgent
from rai_s2s.asr.agents.initialization import TRANSCRIBE_MODELS

CONFIG = """
[asr]
recording_device_name = "default"
transcription_model = "{transcription_model}"
transcription_model_name = "tiny"
language = "en"
vad_model = "{vad_model}"
vad_threshold = 0.3
silence_grace_period = 0.3
use_wake_word = false
wake_word_model = ""
wake_word_model_name = ""
wake_word_threshold = 0.5
"""


@pytest.fixture
def config(tmp_path: Path):
    def write(transcription_model: str, vad_model: str = "SileroVAD") -> str:
        path = tmp_path / "config.toml"
        path.write_text(
            CONFIG.format(transcription_model=transcription_model, vad_model=vad_model)
        )
        return str(path)

    return write


@pytest.fixture(autouse=True)
def stub_models(monkeypatch):
    """The dispatch is under test, not the models it reaches for."""
    for name in ("LocalWhisper", "FasterWhisper", "OpenAIWhisper", "SileroVAD"):
        monkeypatch.setattr(models, name, lambda *args, **kwargs: object())
    monkeypatch.setattr(
        SpeechRecognitionAgent, "__init__", lambda self, *args, **kwargs: None
    )


@pytest.mark.parametrize("transcription_model", TRANSCRIBE_MODELS)
def test_every_advertised_model_is_dispatchable(config, transcription_model):
    assert isinstance(
        SpeechRecognitionAgent.from_config(config(transcription_model)),
        SpeechRecognitionAgent,
    )


def test_unknown_transcription_model_lists_the_valid_ones(config):
    # the old suffixed spelling is now rejected at config load, not silently
    # accepted and then dropped by the dispatch below
    with pytest.raises(ValueError, match="unknown model_type"):
        SpeechRecognitionAgent.from_config(config("LocalWhisper (Free)"))


def test_unknown_vad_model_is_reported(config):
    with pytest.raises(ValueError, match="Unknown VAD model"):
        SpeechRecognitionAgent.from_config(
            config(TRANSCRIBE_MODELS[0], vad_model="NotAVAD")
        )
