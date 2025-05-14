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

from rai_s2s.asr.models import BaseTranscriptionModel


class LocalWhisper(BaseTranscriptionModel):
    """
    A transcription model using OpenAI's Whisper, running locally.

    This class loads a Whisper model and performs speech-to-text transcription
    on audio data. It supports GPU acceleration if available.

    Parameters
    ----------
    model_name : str
        The name of the Whisper model to load.
    sample_rate : int
        The sample rate of the input audio, in Hz.
    language : str, optional
        The language of the transcription output. Default is "en" (English).
    **kwargs : dict, optional
        Additional keyword arguments for loading the Whisper model.

    Attributes
    ----------
    whisper : whisper.Whisper
        The loaded Whisper model for transcription.
    logger : logging.Logger
        Logger instance for logging transcription results.
    """

    def __init__(
        self, model_name: str, sample_rate: int, language: str = "en", **kwargs
    ):
        super().__init__(model_name, sample_rate, language)
        self.decode_options = {
            "language": language,  # Set language to English
            "task": "transcribe",  # Set task to transcribe (not translate)
            "fp16": False,  # Use FP32 instead of FP16 for better precision
            "without_timestamps": True,  # Don't include timestamps in output
            "suppress_tokens": [-1],  # Default tokens to suppress
            "suppress_blank": True,  # Suppress blank outputs
            "beam_size": 5,  # Beam size for beam search
        }
        if torch.cuda.is_available():
            self.whisper = whisper.load_model(self.model_name, device="cuda", **kwargs)
        else:
            self.whisper = whisper.load_model(self.model_name, **kwargs)

        self.logger = logging.getLogger(__name__)

    def transcribe(self, data: NDArray[np.int16]) -> str:
        """
        Transcribes speech from the given audio data using Whisper.

        This method normalizes the input audio, processes it using the Whisper model,
        and returns the transcribed text.

        Parameters
        ----------
        data : NDArray[np.int16]
            A NumPy array containing the raw audio waveform data.

        Returns
        -------
        str
            The transcribed text from the audio input.
        """
        normalized_data = data.astype(np.float32) / 32768.0

        result = whisper.transcribe(
            self.whisper, normalized_data, **self.decode_options
        )
        transcription = result["text"]
        self.logger.info("transcription: %s", transcription)
        transcription = cast(str, transcription)
        self.latest_transcription = transcription
        return transcription


class FasterWhisper(BaseTranscriptionModel):
    """
    A transcription model using Faster Whisper for efficient speech-to-text conversion.

    This class loads a Faster Whisper model, optimized for speed and efficiency.

    Parameters
    ----------
    model_name : str
        The name of the Faster Whisper model to load.
    sample_rate : int
        The sample rate of the input audio, in Hz.
    language : str, optional
        The language of the transcription output. Default is "en" (English).
    **kwargs : dict, optional
        Additional keyword arguments for loading the Faster Whisper model.

    Attributes
    ----------
    model : WhisperModel
        The loaded Faster Whisper model instance.
    logger : logging.Logger
        Logger instance for logging transcription results.
    """

    def __init__(
        self, model_name: str, sample_rate: int, language: str = "en", **kwargs
    ):
        super().__init__(model_name, sample_rate, language)
        self.model = WhisperModel(model_name, **kwargs)
        self.logger = logging.getLogger(__name__)

    def transcribe(self, data: NDArray[np.int16]) -> str:
        """
        Transcribes speech from the given audio data using Faster Whisper.

        This method normalizes the input audio, processes it using the Faster Whisper model,
        and returns the transcribed text.

        Parameters
        ----------
        data : NDArray[np.int16]
            A NumPy array containing the raw audio waveform data.

        Returns
        -------
        str
            The transcribed text from the audio input.
        """
        normalized_data = data.astype(np.float32) / 32768.0
        segments, _ = self.model.transcribe(normalized_data)
        transcription = " ".join(segment.text for segment in segments)
        self.logger.info("transcription: %s", transcription)
        return transcription
