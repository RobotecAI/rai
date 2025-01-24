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
from typing import cast

import numpy as np
import torch
import whisper
from faster_whisper import WhisperModel
from numpy._typing import NDArray

from rai_asr.models.base import BaseTranscriptionModel


class LocalWhisper(BaseTranscriptionModel):
    def __init__(self, model_name: str, sample_rate: int, language: str = "en"):
        super().__init__(model_name, sample_rate, language)
        if torch.cuda.is_available():
            self.whisper = whisper.load_model(self.model_name, device="cuda")
        else:
            self.whisper = whisper.load_model(self.model_name)

        self.logger = logging.getLogger(__name__)

    def transcribe(self, data: NDArray[np.int16]) -> str:
        normalized_data = data.astype(np.float32) / 32768.0
        result = whisper.transcribe(
            self.whisper, normalized_data
        )  # TODO: handling of additional transcribe arguments (perhaps in model init)
        transcription = result["text"]
        self.logger.info("transcription: %s", transcription)
        transcription = cast(str, transcription)
        self.latest_transcription = transcription
        return transcription


class FasterWhisper(BaseTranscriptionModel):
    def __init__(
        self, model_name: str, sample_rate: int, language: str = "en", **kwargs
    ):
        super().__init__(model_name, sample_rate, language)
        self.model = WhisperModel(model_name, **kwargs)
        self.logger = logging.getLogger(__name__)

    def transcribe(self, data: NDArray[np.int16]) -> str:
        normalized_data = data.astype(np.float32) / 32768.0
        segments, _ = self.model.transcribe(normalized_data)
        transcription = " ".join(segment.text for segment in segments)
        self.logger.info("transcription: %s", transcription)
        return transcription
