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


@dataclass
class VADConfig:
    model_name: str = "SileroVAD"
    threshold: float = 0.5
    sampling_rate: int = 1600


@dataclass
class WWConfig:
    model_name: str = "OpenWakeWord"
    threshold: float = 0.01


@dataclass
class TranscribeConfig:
    model_name: str = "LocalWhisper"


@dataclass
class MicrophoneConfig:
    device_name: str


@dataclass
class ASRAgentConfig:
    voice_activity_detection: VADConfig
    wakeword: WWConfig
    transcribe: TranscribeConfig
    microphone: MicrophoneConfig
