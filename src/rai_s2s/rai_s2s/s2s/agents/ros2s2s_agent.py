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

import logging
from typing import Optional

from rai.communication.ros2 import ROS2HRIConnector, ROS2HRIMessage

from rai_s2s.asr.models import BaseTranscriptionModel, BaseVoiceDetectionModel
from rai_s2s.s2s.agents.s2s_agent import SpeechToSpeechAgent
from rai_s2s.sound_device import SoundDeviceConfig
from rai_s2s.tts.models.base import TTSModel


class ROS2S2SAgent(SpeechToSpeechAgent):
    def __init__(
        self,
        from_human_topic: str,
        to_human_topic: str,
        *,
        microphone_config: SoundDeviceConfig,
        speaker_config: SoundDeviceConfig,
        transcription_model: BaseTranscriptionModel,
        vad: BaseVoiceDetectionModel,
        tts: TTSModel,
        grace_period: float = 1,
        logger: Optional[logging.Logger] = None,
        **kwargs,
    ):
        super().__init__(
            from_human_topic,
            to_human_topic,
            microphone_config=microphone_config,
            speaker_config=speaker_config,
            transcription_model=transcription_model,
            vad=vad,
            tts=tts,
            grace_period=grace_period,
            logger=logger,
            **kwargs,
        )

    def _setup_hri_connector(self):
        hri_connector = ROS2HRIConnector()
        hri_connector.register_callback(self.to_human_topic, self._on_to_human_message)
        return hri_connector

    def _send_from_human_message(self, data: str):
        print(f"Sending message to {self.from_human_topic}")
        self.hri_connector.send_message(
            ROS2HRIMessage(text=data), self.from_human_topic
        )

    def _send_to_human_message(self, data: str):
        print(f"Sending message to {self.to_human_topic}")
        self.hri_connector.send_message(ROS2HRIMessage(text=data), self.to_human_topic)
