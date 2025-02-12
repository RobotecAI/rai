# Copyright (C) 2024 Robotec.AI
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

from rai.agents import TextToSpeechAgent, VoiceRecognitionAgent
from rai.agents.base import BaseAgent
from rai.communication.sound_device import SoundDeviceConfig
from rai.runners import BaseRunner
from rai_asr.models import LocalWhisper, OpenWakeWord, SileroVAD
from rai_tts.models import OpenTTS


@dataclass
class TTSConfig:
    device_name: str = "default"
    url: str = "http://localhost:5500/api/tts"
    voice: str = "larynx:blizzard_lessac-glow_tts"


@dataclass
class ASRConfig:
    device_name: str = "default"
    vad_threshold: float = 0.5
    oww_threshold: float = 0.1
    whisper_model: str = "tiny"
    oww_model: str = "hey jarvis"


class Speech2SpeechRunner(BaseRunner):
    def __init__(
        self,
        agents: Optional[list[BaseAgent]] = None,
        tts_cfg: TTSConfig = TTSConfig(),
        asr_config: ASRConfig = ASRConfig(),
    ):
        if agents is None:
            agents = []
        super().__init__(agents)
        tts = self._setup_tts_agent(tts_cfg)
        asr = self._setup_asr_agent(asr_config)
        self.agents.extend([tts, asr])

    def _setup_tts_agent(self, cfg: TTSConfig):
        speaker_config = SoundDeviceConfig(
            stream=True,
            is_output=True,
            device_name=cfg.device_name,
        )
        tts = OpenTTS(cfg.url, cfg.voice)
        return TextToSpeechAgent(speaker_config, "text_to_speech", tts)

    def _setup_asr_agent(self, cfg: ASRConfig):
        vad = SileroVAD(threshold=cfg.vad_threshold)
        oww = OpenWakeWord(cfg.oww_model, cfg.oww_threshold)
        whisper = LocalWhisper(
            cfg.whisper_model, vad.sampling_rate
        )  # models should have compatible sampling rate
        microphone_config = SoundDeviceConfig(
            stream=True,
            device_name=cfg.device_name,
            consumer_sampling_rate=vad.sampling_rate,
            is_input=True,
        )
        asr_agent = VoiceRecognitionAgent(
            microphone_config, "automatic_speech_recognition", whisper, vad
        )
        asr_agent.add_detection_model(oww, pipeline="record")
        return asr_agent
