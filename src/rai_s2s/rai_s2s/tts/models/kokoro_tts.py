# Copyright (C) 2025 Robotec.AI
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


from typing import Tuple

import numpy as np
from kokoro_onnx import Kokoro
from pydub import AudioSegment

from rai_s2s.tts.models import TTSModel, TTSModelError


class KokoroTTS(TTSModel):
    """
    A text-to-speech (TTS) model interface for Kokoro TTS.

    Parameters
    ----------
    model_path : str, optional
        Path to the ONNX model file for Kokoro TTS, by default "kokoro-v0_19.onnx".
    voices_path : str, optional
        Path to the JSON file containing voice configurations, by default "voices.json".
    voice : str, optional
        The voice model to use, by default "af_sarah".
    language : str, optional
        The language code for the TTS model, by default "en-us".
    speed : float, optional
        The speed of the speech generation, by default 1.0.
    Raises
    ------
    TTSModelError
        If there is an issue with initializing the Kokoro TTS model.

    """

    def __init__(
        self,
        model_path: str = "kokoro-v0_19.onnx",
        voices_path: str = "voices.json",
        voice: str = "af_sarah",
        language: str = "en-us",
        speed: float = 1.0,
    ):
        self.voice = voice
        self.speed = speed
        self.language = language

        try:
            self.kokoro = Kokoro(
                model_path=model_path, voices_path=voices_path
            )  # TODO (mkotynia) add method to download the model ?
        except Exception as e:
            raise TTSModelError(f"Failed to initialize Kokoro TTS model: {e}") from e

    def get_speech(self, text: str) -> AudioSegment:
        """
        Converts text into speech using the Kokoro TTS model.

        Parameters
        ----------
        text : str
            The input text to be converted into speech.

        Returns
        -------
        AudioSegment
            The generated speech as an `AudioSegment` object.

        Raises
        ------
        TTSModelError
            If there is an issue with processing TTS conversion by Kokoro TTS model.
        """
        try:
            samples, sample_rate = self.kokoro.create(
                text, voice=self.voice, speed=self.speed, lang=self.language
            )

            if samples.dtype == np.float32:
                samples = (samples * 32768).clip(-32768, 32767).astype(np.int16)
            else:
                raise TTSModelError(
                    f"Unsupported sample format: {samples.dtype}. Expected float32."
                )

            audio_segment = AudioSegment(
                data=samples.tobytes(),
                sample_width=2,
                frame_rate=sample_rate,
                channels=1,
            )

            return audio_segment
        except Exception as e:
            raise TTSModelError(f"Failed to process text with Kokoro TTS model: {e}")

    def get_tts_params(self) -> Tuple[int, int]:
        """
        Returns TTS samling rate and channels.

        The information is retrieved by running a sample transcription request, to ensure that the information will be accurate for generation.

        Returns
        -------
        Tuple[int, int]
            sample rate, channels

        Raises
        ------
        TTSModelError
            If there is an issue with processing TTS conversion by Kokoro TTS model.
        """

        data = self.get_speech("A")
        return data.frame_rate, 1

    def get_supported_languages(self) -> list[str]:
        """
        Returns a list of supported languages for the Kokoro TTS model.

        Returns
        -------
        list[str]
            List of supported languages.
        Raises
        ------
        TTSModelError
            If there is an issue with retrieving supported languages from the Kokoro TTS model.
        """
        try:
            return self.kokoro.get_languages()
        except Exception as e:
            raise TTSModelError(f"Failed to get supported languages: {e}")

    def get_available_voices(self) -> list[str]:
        """
        Returns a list of available voice names.

        Returns
        -------
        list[str]
            List of voice names available in the Kokoro TTS model.
        Raises
        ------
        TTSModelError
            If there is an issue with retrieving voice names from the Kokoro TTS model.
        """
        try:
            return list(self.kokoro.get_voices())
        except Exception as e:
            raise TTSModelError(f"Failed to retrieve voice names: {e}")
