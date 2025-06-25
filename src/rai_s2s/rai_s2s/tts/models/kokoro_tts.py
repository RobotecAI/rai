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

    """

    def __init__(
        self,
        model_path: str = "kokoro-v0_19.onnx",
        voices_path: str = "voices.json",
        voice: str = "af_sarah",
    ):
        self.model_path = model_path
        self.voices_path = voices_path
        self.voice = voice

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
            If there is an issue with Kokoro TTS model or processing audio.
        """
        try:
            kokoro = Kokoro(
                model_path=self.model_path, voices_path=self.voices_path
            )  # TODO (mkotynia) add method to download the model ?
        except Exception as e:
            raise TTSModelError(f"Failed to initialize Kokoro TTS model: {e}")
        try:
            samples, sample_rate = kokoro.create(
                text, voice=self.voice, speed=1.0, lang="en-us"
            )  # TODO (mkotynia) parametrize this

            if samples.dtype == np.float32:
                samples = (
                    (samples * 32768).clip(-32768, 32767).astype(np.int16)
                )  # TODO (mkotynia) consider writing tests for format in case kokoro_onnx version is changed
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
            If there is an issue with Kokoro TTS model or processing audio.
        """

        data = self.get_speech("A")
        return data.frame_rate, 1
