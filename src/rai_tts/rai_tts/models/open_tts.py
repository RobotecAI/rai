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

from io import BytesIO
from typing import Tuple

import numpy as np
import requests
from pydub import AudioSegment
from scipy.io.wavfile import read

from rai_tts.models import TTSModel, TTSModelError


class OpenTTS(TTSModel):
    def __init__(
        self,
        url: str = "http://localhost:5500/api/tts",
        voice: str = "larynx:blizzard_lessac-glow_tts",
    ):
        self.url = url
        self.voice = voice

    def get_speech(self, text: str) -> AudioSegment:
        params = {
            "voice": self.voice,
            "text": text,
        }
        try:
            response = requests.get(self.url, params=params)
        except requests.exceptions.RequestException as e:
            raise TTSModelError(
                f"Error occurred while fetching audio: {e}, check if OpenTTS server is running correctly."
            ) from e

        content_type = response.headers.get("Content-Type", "")

        if "audio" not in content_type:
            raise ValueError("Response does not contain audio data")

        # Load audio into memory
        audio_bytes = BytesIO(response.content)
        sample_rate, data = read(audio_bytes)
        if data.dtype == np.int32:
            data = (data / 2**16).astype(np.int16)  # Scale down from int32
        elif data.dtype == np.uint8:
            data = (data - 128).astype(np.int16) * 256  # Convert uint8 to int16
        elif data.dtype == np.float32:
            data = (
                (data * 32768).clip(-32768, 32767).astype(np.int16)
            )  # Convert float32 to int16

        return AudioSegment(data, frame_rate=sample_rate, sample_width=2, channels=1)

    def get_tts_params(self) -> Tuple[int, int]:
        data = self.get_speech("A")
        print(data.frame_rate)
        return data.frame_rate, 1
