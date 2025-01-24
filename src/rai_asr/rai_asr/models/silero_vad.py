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

from typing import Any, Literal, Tuple

import numpy as np
import torch
from numpy.typing import NDArray

from rai_asr.models import BaseVoiceDetectionModel


class SileroVAD(BaseVoiceDetectionModel):
    def __init__(self, sampling_rate: Literal[8000, 16000] = 16000, threshold=0.5):
        super(SileroVAD, self).__init__()
        self.model_name = "silero_vad"
        self.model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model=self.model_name,
        )  # type: ignore
        # NOTE: See silero vad implementation: https://github.com/snakers4/silero-vad/blob/9060f664f20eabb66328e4002a41479ff288f14c/src/silero_vad/utils_vad.py#L61
        if sampling_rate == 16000:
            self.sampling_rate = 16000
            self.window_size = 512
        elif sampling_rate == 8000:
            self.sampling_rate = 8000
            self.window_size = 256
        else:
            raise ValueError(
                "Only 8000 and 16000 sampling rates are supported"
            )  # TODO: consider if this should be a ValueError or something else
        self.threshold = threshold

    def int2float(self, sound: NDArray[np.int16]):
        converted_sound = sound.astype("float32")
        converted_sound *= 1 / 32768
        converted_sound = converted_sound.squeeze()
        return converted_sound

    def detect(
        self, audio_data: NDArray, input_parameters: dict[str, Any]
    ) -> Tuple[bool, dict[str, Any]]:
        vad_confidence = self.model(
            torch.tensor(self.int2float(audio_data[-self.window_size :])),
            self.sampling_rate,
        ).item()
        ret = input_parameters.copy()
        ret.update({self.model_name: {"vad_confidence": vad_confidence}})

        return vad_confidence > self.threshold, ret
