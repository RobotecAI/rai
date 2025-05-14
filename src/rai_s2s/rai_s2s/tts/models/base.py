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

from abc import ABC, abstractmethod
from typing import Tuple

from pydub import AudioSegment


class TTSModelError(Exception):
    pass


class TTSModel(ABC):
    sample_rate: int = -1
    channels: int = 1

    @abstractmethod
    def get_speech(self, text: str) -> AudioSegment:
        pass

    @abstractmethod
    def get_tts_params(self) -> Tuple[int, int]:
        pass

    def set_tts_params(self, target_sample_rate: int, channels: int):
        self.sample_rate = target_sample_rate
        self.channels = channels

    def _resample(self, audio: AudioSegment) -> AudioSegment:
        """
        Resample an AudioSegment to a specified sample rate and number of channels.

        :param audio: The input AudioSegment.
        :param target_sample_rate: The desired sample rate in Hz.
        :param channels: The desired number of audio channels.
        :return: A new AudioSegment with the specified sample rate and channels.
        """
        return audio.set_frame_rate(self.sample_rate)
