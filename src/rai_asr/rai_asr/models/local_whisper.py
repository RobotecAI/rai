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

from typing import cast

import numpy as np
import whisper
from numpy._typing import NDArray

from rai_asr.models.base import BaseTranscriptionModel


class LocalWhisper(BaseTranscriptionModel):
    def __init__(self, model_name: str, sample_rate: int, language: str = "en"):
        super().__init__(model_name, sample_rate, language)
        self.whisper = whisper.load_model(self.model_name)

        self.samples = None

    def add_samples(self, data: NDArray[np.int16]):
        normalized_data = data.astype(np.float32) / 32768.0
        self.samples = (
            np.concatenate([self.samples, normalized_data])
            if self.samples is not None
            else data
        )

    def transcribe(self) -> str:
        if self.samples is None:
            raise ValueError("No samples to transcribe")
        result = whisper.transcribe(
            self.whisper, self.samples
        )  # TODO: handling of additional transcribe arguments (perhaps in model init)
        transcription = result["text"]
        transcription = cast(str, transcription)
        return transcription
