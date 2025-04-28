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

import tomli


@dataclass
class VADConfig:
    model_name: Literal["SileroVAD"] = "SileroVAD"
    threshold: float = 0.5
    silence_grace_period: float = 0.3


@dataclass
class WWConfig:
    model_name: Literal["OpenWakeWord"] = "OpenWakeWord"
    threshold: float = 0.01
    is_used: bool = False


TRANSCRIBE_MODELS = ["LocalWhisper (Free)", "FasterWhisper (Free)", "OpenAI (Cloud)"]


@dataclass
class TranscribeConfig:
    model_name: str = TRANSCRIBE_MODELS[0]
    language: str = "en"

    def __post_init__(self):
        if self.model_name not in TRANSCRIBE_MODELS:
            raise ValueError(f"model_name must be one of {TRANSCRIBE_MODELS}")


@dataclass
class MicrophoneConfig:
    device_name: str


@dataclass
class ASRAgentConfig:
    voice_activity_detection: VADConfig
    wakeword: WWConfig
    transcribe: TranscribeConfig
    microphone: MicrophoneConfig


def load_config(config_path: Optional[str] = None) -> ASRAgentConfig:
    if config_path is None:
        with open("config.toml", "rb") as f:
            config_dict = tomli.load(f)
    else:
        with open(config_path, "rb") as f:
            config_dict = tomli.load(f)
    return ASRAgentConfig(
        voice_activity_detection=VADConfig(
            model_name=config_dict["asr"]["vad_model"],
            threshold=config_dict["asr"]["vad_threshold"],
            silence_grace_period=config_dict["asr"]["silence_grace_period"],
        ),
        wakeword=WWConfig(
            model_name=config_dict["asr"]["wake_word_model"],
            threshold=config_dict["asr"]["wake_word_threshold"],
            is_used=config_dict["asr"]["use_wake_word"],
        ),
        transcribe=TranscribeConfig(
            model_name=config_dict["asr"]["transcription_model"],
            language=config_dict["asr"]["language"],
        ),
        microphone=MicrophoneConfig(
            device_name=config_dict["asr"]["recording_device_name"],
        ),
    )
