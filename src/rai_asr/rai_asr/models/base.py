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
from typing import Any, Tuple

import numpy as np
from numpy._typing import NDArray


class BaseVoiceDetectionModel(ABC):

    @abstractmethod
    def detected(
        self, audio_data: NDArray, input_parameters: dict[str, Any]
    ) -> Tuple[bool, dict[str, Any]]:
        pass


class BaseTranscriptionModel(ABC):
    def __init__(self, model_name: str, sample_rate: int, language: str = "en"):
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.language = language

        self.latest_transcription = ""

    def consume_transcription(self) -> str:
        ret = self.latest_transcription
        self.latest_transcription = ""
        return ret

    @abstractmethod
    def transcribe(self, data: NDArray[np.int16]):
        pass
