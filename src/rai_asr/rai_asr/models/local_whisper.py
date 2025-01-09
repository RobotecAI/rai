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
import torch
import whisper
from numpy._typing import NDArray

from rai_asr.models.base import BaseTranscriptionModel


class LocalWhisper(BaseTranscriptionModel):
    def __init__(self, model_name: str, sample_rate: int, language: str = "en"):
        super().__init__(model_name, sample_rate, language)
        if torch.cuda.is_available():
            self.whisper = whisper.load_model(self.model_name, device="cuda")
        else:
            self.whisper = whisper.load_model(self.model_name)

        # TODO: remove sample storage before PR is merged, this is just to enable saving wav files for debugging
        # self.samples = None

    def consume_transcription(self) -> str:
        ret = super().consume_transcription()
        # self.samples = None
        return ret

    # def save_wav(self, output_filename: str):
    #     assert self.samples is not None, "No samples to save"
    #     combined_samples = self.samples
    #     if combined_samples.dtype.kind == "f":
    #         combined_samples = np.clip(combined_samples, -1.0, 1.0)
    #         combined_samples = (combined_samples * 32767).astype(np.int16)
    #     elif combined_samples.dtype != np.int16:
    #         combined_samples = combined_samples.astype(np.int16)

    #     with wave.open(output_filename, "wb") as wav_file:
    #         n_channels = 1
    #         sampwidth = 2
    #         wav_file.setnchannels(n_channels)
    #         wav_file.setsampwidth(sampwidth)
    #         wav_file.setframerate(self.sample_rate)
    #         wav_file.writeframes(combined_samples.tobytes())

    def transcribe(self, data: NDArray[np.int16]):
        # self.samples = (
        #     np.concatenate((self.samples, data)) if self.samples is not None else data
        # )
        normalized_data = data.astype(np.float32) / 32768.0
        result = whisper.transcribe(
            self.whisper, normalized_data
        )  # TODO: handling of additional transcribe arguments (perhaps in model init)
        transcription = result["text"]
        transcription = cast(str, transcription)
        self.latest_transcription += transcription
