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

from dataclasses import dataclass
from typing import Literal, Optional

from pydantic import BaseModel
from rai.settings import section


@dataclass
class VADConfig:
    model_name: Literal["SileroVAD"] = "SileroVAD"
    threshold: float = 0.5
    silence_grace_period: float = 0.3

    def __post_init__(self):
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {self.threshold}")
        if self.silence_grace_period < 0:
            raise ValueError(
                f"silence_grace_period must not be negative, got {self.silence_grace_period}"
            )


@dataclass
class WWConfig:
    """Wake word detection. Off unless `is_used` is set, and there is no default model.

    `model_name` is a path to your own openWakeWord model, or the name of an openWakeWord
    pre-trained model (e.g. "hey jarvis"). The pre-trained models are licensed CC BY-NC-SA 4.0
    (non-commercial, https://github.com/dscripka/openWakeWord#license); naming one opts in to
    downloading it and to that licence.
    """

    model_name: str = ""
    model_type: Literal["OpenWakeWord"] = "OpenWakeWord"
    threshold: float = 0.01
    is_used: bool = False

    def __post_init__(self):
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {self.threshold}")
        if self.is_used and not self.model_name.strip():
            raise ValueError(
                "use_wake_word is set but wake_word_model_name is empty: name a path to your own "
                "openWakeWord model, or an openWakeWord pre-trained model (CC BY-NC-SA 4.0, "
                "non-commercial)"
            )


TRANSCRIBE_MODELS = ["LocalWhisper", "FasterWhisper", "OpenAI"]


@dataclass
class TranscribeConfig:
    model_type: str = TRANSCRIBE_MODELS[0]
    model_name: str = "tiny"
    language: str = "en"

    def __post_init__(self):
        if self.model_type not in TRANSCRIBE_MODELS:
            raise ValueError(
                f"unknown model_type: {self.model_type}. Must be one of {TRANSCRIBE_MODELS}"
            )


@dataclass
class MicrophoneConfig:
    device_name: str


@dataclass
class ASRAgentConfig:
    voice_activity_detection: VADConfig
    wakeword: WWConfig
    transcribe: TranscribeConfig
    microphone: MicrophoneConfig


class ASRSettings(BaseModel):
    """The `[asr]` table of rai.toml (or `RAI_ASR__<FIELD>` env vars)."""

    recording_device_name: str = "default"
    transcription_model: str = TRANSCRIBE_MODELS[0]
    transcription_model_name: str = "tiny"
    language: str = "en"
    vad_model: str = "SileroVAD"
    vad_threshold: float = 0.5
    silence_grace_period: float = 0.3
    use_wake_word: bool = False
    wake_word_model: str = "OpenWakeWord"
    wake_word_model_name: str = ""
    wake_word_threshold: float = 0.01


def load_config(config_path: Optional[str] = None) -> ASRAgentConfig:
    asr = section("asr", ASRSettings, config_path)
    return ASRAgentConfig(
        voice_activity_detection=VADConfig(
            model_name=asr.vad_model,  # type: ignore[arg-type]
            threshold=asr.vad_threshold,
            silence_grace_period=asr.silence_grace_period,
        ),
        wakeword=WWConfig(
            model_type=asr.wake_word_model,  # type: ignore[arg-type]
            model_name=asr.wake_word_model_name,
            threshold=asr.wake_word_threshold,
            is_used=asr.use_wake_word,
        ),
        transcribe=TranscribeConfig(
            model_type=asr.transcription_model,
            model_name=asr.transcription_model_name,
            language=asr.language,
        ),
        microphone=MicrophoneConfig(device_name=asr.recording_device_name),
    )
