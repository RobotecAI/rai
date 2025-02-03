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


import logging
from typing import TYPE_CHECKING, Optional

from rai.agents.base import BaseAgent
from rai.communication import ROS2ARIConnector, SoundDeviceConnector

if TYPE_CHECKING:
    from rai.communication.sound_device.api import SoundDeviceConfig


class TextToSpeechAgent(BaseAgent):
    def __init__(
        self,
        speaker_config: SoundDeviceConfig,
        ros2_name: str,
        logger: Optional[logging.Logger] = None,
    ):
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

        speaker = SoundDeviceConnector(
            targets=[("speaker", speaker_config)], sources=[]
        )
        ros2_connector = ROS2ARIConnector(ros2_name)
        super().__init__(connectors={"ros2": ros2_connector, "speaker": speaker})
        self.running = False

    def __call__(self):
        self.run()

    def run(self):
        self.running = True
        self.logger.info("TextToSpeechAgent started")
