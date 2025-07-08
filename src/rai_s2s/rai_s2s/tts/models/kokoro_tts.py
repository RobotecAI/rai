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


import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
from kokoro_onnx import Kokoro
from pydub import AudioSegment

from rai_s2s.tts.models import TTSModel, TTSModelError


class _PhonemizerLogsFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "words count mismatch" not in record.getMessage()


class KokoroTTS(TTSModel):
    """
    A text-to-speech (TTS) model interface for Kokoro TTS.

    Parameters
    ----------
    voice : str, optional
        The voice model to use, by default "af_sarah".
    language : str, optional
        The language code for the TTS model, by default "en-us".
    speed : float, optional
        The speed of the speech generation, by default 1.0.
    cache_dir : str | Path, optional
        Directory to cache downloaded models, by default "~/.cache/rai/kokoro/".

    Raises
    ------
    TTSModelError
        If there is an issue with initializing the Kokoro TTS model or downloading
        required files.

    """

    BASE_MODEL_URL = (
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/"
    )
    VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.json"

    VOICES_FILENAME = "voices.json"

    def __init__(
        self,
        voice: str = "af_sarah",
        language: str = "en-us",
        speed: float = 1.0,
        model_size: Literal["small", "medium", "large"] = "large",
        cache_dir: str | Path = Path.home() / ".cache/rai/kokoro/",
    ):
        phonemizer_logger = logging.getLogger("phonemizer")
        phonemizer_logger.addFilter(_PhonemizerLogsFilter())
        self.voice = voice
        self.speed = speed
        self.language = language
        self.cache_dir = Path(cache_dir)

        os.makedirs(self.cache_dir, exist_ok=True)

        self.model_size = model_size
        self.model_path = self._ensure_model_exists()
        self.voices_path = self._ensure_voices_exists()

        try:
            self.kokoro = Kokoro(
                model_path=str(self.model_path), voices_path=str(self.voices_path)
            )
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
            text = self._preprocess_text(text)

            samples, sample_rate = self.kokoro.create(
                text, voice=self.voice, speed=self.speed, lang=self.language, trim=False
            )

            if samples.dtype == np.float32:
                samples = (samples * 32768).clip(-32768, 32767).astype(np.int16)
            else:
                raise TTSModelError(
                    f"Unsupported sample format: {samples.dtype}. Expected float32."
                )

            audio = AudioSegment(
                data=samples.tobytes(),
                sample_width=2,
                frame_rate=sample_rate,
                channels=1,
            )
            if self.sample_rate == -1:
                return audio
            else:
                return self._resample(audio)
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

    def _get_model_filename(self) -> str:
        """
        Gets the model filename based on the model size.

        Returns
        -------
        str
            The model filename for the specified model size.
        """
        if self.model_size == "large":
            return "kokoro-v0_19.onnx"
        elif self.model_size == "medium":
            return "kokoro-v0_19.fp16.onnx"
        elif self.model_size == "small":
            return "kokoro-v0_19.int8.onnx"
        else:
            raise TTSModelError(f"Unsupported model size: {self.model_size}")

    def _get_model_url(self) -> str:
        """
        Gets the full model URL based on the model size.

        Returns
        -------
        str
            The full URL for downloading the model.
        """
        return self.BASE_MODEL_URL + self._get_model_filename()

    def _ensure_model_exists(self) -> Path:
        """
        Checks if the model file exists and downloads it if necessary.

        Returns
        -------
        Path
            The path to the model file.

        Raises
        ------
        TTSModelError
            If the model cannot be downloaded or accessed.
        """
        model_filename = self._get_model_filename()
        model_path = self.cache_dir / model_filename
        if model_path.exists() and model_path.is_file():
            return model_path

        model_url = self._get_model_url()
        self._download_file(model_url, model_path)
        return model_path

    def _ensure_voices_exists(self) -> Path:
        """
        Checks if the voices file exists and downloads it if necessary.

        Returns
        -------
        Path
            The path to the voices file.

        Raises
        ------
        TTSModelError
            If the voices file cannot be downloaded or accessed.
        """
        voices_path = self.cache_dir / self.VOICES_FILENAME
        if voices_path.exists() and voices_path.is_file():
            return voices_path

        self._download_file(self.VOICES_URL, voices_path)
        return voices_path

    def _download_file(self, url: str, destination: Path) -> None:
        """
        Downloads a file from a URL to a destination path.

        Parameters
        ----------
        url : str
            The URL to download from.
        destination : Path
            The destination path to save the file.

        Raises
        ------
        Exception
            If the download fails.
        """
        try:
            subprocess.run(
                [
                    "wget",
                    url,
                    "-O",
                    str(destination),
                    "--progress=dot:giga",
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise Exception(f"Download failed with exit code {e.returncode}") from e
        except Exception as e:
            raise Exception(f"Download failed: {e}") from e

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocesses text by removing formatting characters that would be
        read aloud as words (like 'asterisk' for '*').

        Parameters
        ----------
        text : str
            The input text that may contain formatting characters.

        Returns
        -------
        str
            The cleaned text with formatting characters removed.
        """

        # Remove markdown headers (# symbols at start of line)
        text = re.sub(r"^#+\s*", "", text)

        # Remove bold markdown (**)
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)

        # Remove italic markdown (*)
        text = re.sub(r"\*(.*?)\*", r"\1", text)

        return text
