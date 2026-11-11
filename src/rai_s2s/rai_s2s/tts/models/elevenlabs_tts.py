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

import os
from typing import Tuple

import numpy as np
from elevenlabs.client import ElevenLabs
from elevenlabs.types.voice_settings import VoiceSettings
from rai.content import Audio

from rai_s2s.tts.models import TTSModel, TTSModelError


class ElevenLabsTTS(TTSModel):
    """
    A text-to-speech (TTS) model interface for ElevenLabs.

    Parameters
    ----------
    voice : str, optional
        The voice model to use.
    base_url : str, optional
        The API endpoint for the ElevenLabs API, by default None.
    """

    PCM_RATE = 24000  # raw 16-bit mono PCM, no decoder needed

    def __init__(
        self,
        voice: str,
        base_url: str | None = None,
    ):
        api_key = os.getenv(key="ELEVENLABS_API_KEY")
        if api_key is None:
            raise TTSModelError("ELEVENLABS_API_KEY environment variable is not set.")

        self.client = ElevenLabs(base_url=base_url, api_key=api_key)
        self.voice_settings = VoiceSettings(
            stability=0.7,
            similarity_boost=0.5,
        )

        voices = self.client.voices.get_all().voices
        voice_id = next((v.voice_id for v in voices if v.name == voice), None)
        if voice_id is None:
            raise TTSModelError(f"Voice {voice} not found")
        self.voice_id = voice_id

    def get_speech(self, text: str) -> Audio:
        """
        Converts text into speech using the ElevenLabs API.

        Parameters
        ----------
        text : str
            The input text to be converted into speech.

        Returns
        -------
        Audio
            The generated speech, int16.

        Raises
        ------
        TTSModelError
            If there is an issue with the request or the ElevenLabs API is unreachable.
            If the response does not contain valid audio data.
        """
        try:
            response = self.client.text_to_speech.convert(
                self.voice_id,
                text=text,
                voice_settings=self.voice_settings,
                optimize_streaming_latency=4,
                output_format=f"pcm_{self.PCM_RATE}",
            )
            audio_data = b"".join(response)
        except Exception as e:
            raise TTSModelError(f"Error occurred while fetching audio: {e}") from e

        samples = np.frombuffer(audio_data, "<i2").astype(np.int16)
        return self._resample(Audio(samples, self.PCM_RATE))

    def get_tts_params(self) -> Tuple[int, int]:
        """
        Returns TTS sampling rate and channels.

        The information is retrieved by running a sample transcription request, to ensure that the information will be accurate for generation.

        Returns
        -------
        Tuple[int, int]
            sample rate, channels

        Raises
        ------
        TTSModelError
            If there is an issue with the request or the ElevenLabs API is unreachable.
            If the response does not contain valid audio data.
        """
        data = self.get_speech("A")
        return data.sample_rate, 1
