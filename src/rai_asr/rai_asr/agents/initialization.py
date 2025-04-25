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
    sampling_rate: int = 1600


@dataclass
class WWConfig:
    model_name: Literal["OpenWakeWord"] = "OpenWakeWord"
    threshold: float = 0.01


@dataclass
class TranscribeConfig:
    model_name: Literal["LocalWhisper", "FasterWhisper", "OpenAIWhisper"] = (
        "LocalWhisper"
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


def load_config(config_path: Optional[str] = None) -> ASRAgentConfig:
    if config_path is None:
        with open("config.toml", "rb") as f:
            config_dict = tomli.load(f)
    else:
        with open(config_path, "rb") as f:
            config_dict = tomli.load(f)
    return ASRAgentConfig(
        voice_activity_detection=VADConfig(
            model_name=config_dict["voice_activity_detection"]["model_name"],
            threshold=config_dict["voice_activity_detection"]["threshold"],
            sampling_rate=config_dict["voice_activity_detection"]["sampling_rate"],
        ),
        wakeword=WWConfig(
            model_name=config_dict["wakeword"]["model_name"],
            threshold=config_dict["wakeword"]["threshold"],
        ),
        transcribe=TranscribeConfig(
            model_name=config_dict["transcribe"]["model_name"],
        ),
        microphone=MicrophoneConfig(
            device_name=config_dict["microphone"]["device_name"],
        ),
    )
