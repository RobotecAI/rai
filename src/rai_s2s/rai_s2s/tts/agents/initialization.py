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
from typing import Optional

from pydantic import BaseModel
from rai.settings import section


@dataclass
class SpeakerConfig:
    device_name: str


TTS_MODELS = ["KokoroTTS", "OpenTTS", "ElevenLabs"]


@dataclass
class TTSConfig:
    model_type: str = TTS_MODELS[0]
    voice: str = ""


@dataclass
class TTSAgentConfig:
    text_to_speech: TTSConfig
    speaker: SpeakerConfig


class TTSSettings(BaseModel):
    """The `[tts]` table of rai.toml (or `RAI_TTS__<FIELD>` env vars)."""

    vendor: str = TTS_MODELS[0]
    voice: str = ""
    speaker_device_name: str = "default"


def load_config(config_path: Optional[str] = None) -> TTSAgentConfig:
    tts = section("tts", TTSSettings, config_path)
    return TTSAgentConfig(
        text_to_speech=TTSConfig(model_type=tts.vendor, voice=tts.voice),
        speaker=SpeakerConfig(device_name=tts.speaker_device_name),
    )
