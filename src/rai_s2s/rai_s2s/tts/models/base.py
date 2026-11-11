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

import math
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from rai.content import Audio
from scipy.signal import resample_poly


class TTSModelError(Exception):
    pass


class TTSModel(ABC):
    sample_rate: int = -1
    channels: int = 1

    @abstractmethod
    def get_speech(self, text: str) -> Audio:
        pass

    @abstractmethod
    def get_tts_params(self) -> Tuple[int, int]:
        pass

    def set_tts_params(self, target_sample_rate: int, channels: int):
        self.sample_rate = target_sample_rate
        self.channels = channels

    def _resample(self, audio: Audio) -> Audio:
        """The audio at self.sample_rate, in its own sample type."""
        if self.sample_rate in (-1, audio.sample_rate):
            return audio
        gcd = math.gcd(self.sample_rate, audio.sample_rate)
        samples = resample_poly(
            audio.as_float32().samples,
            self.sample_rate // gcd,
            audio.sample_rate // gcd,
            axis=0,
        )
        out = Audio(samples.astype(np.float32), self.sample_rate)
        return out.as_int16() if audio.samples.dtype == np.int16 else out
