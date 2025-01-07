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

import io
import os
from functools import partial

import numpy as np
from numpy.typing import NDArray
from openai import OpenAI
from scipy.io import wavfile

from rai_asr.models.base import BaseTranscriptionModel


class OpenAIWhisper(BaseTranscriptionModel):
    def __init__(self, model_name: str, sample_rate: int, language: str = "en"):
        super().__init__(model_name, sample_rate, language)
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        self.api_key = api_key
        self.openai_client = OpenAI()
        self.model = partial(
            self.openai_client.audio.transcriptions.create,
            model=self.model_name,
        )

    def transcribe(self, data: NDArray[np.int16]) -> str:
        with io.BytesIO() as temp_wav_buffer:
            wavfile.write(temp_wav_buffer, self.sample_rate, data)
            temp_wav_buffer.seek(0)
            temp_wav_buffer.name = "temp.wav"
            response = self.model(file=temp_wav_buffer, language=self.language)
        transcription = response.text
        return transcription
