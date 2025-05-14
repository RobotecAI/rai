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


from abc import ABC, abstractmethod
from typing import Any, Tuple

import numpy as np
from numpy._typing import NDArray


class BaseVoiceDetectionModel(ABC):
    """
    Abstract base class for voice detection models.

    This class provides a standard interface for voice detection models, where
    subclasses must implement the `detect` method to process audio data and determine
    whether a specific event (e.g., a wake word or other voice activity) has occurred.
    """

    def __call__(
        self, audio_data: NDArray, input_parameters: dict[str, Any]
    ) -> Tuple[bool, dict[str, Any]]:
        """
        Invokes the model to detect a voice event.

        This method calls the `detect` function, allowing the model instance to be used
        as a callable.

        Parameters
        ----------
        audio_data : NDArray
            A NumPy array containing audio input data.
        input_parameters : dict of str to Any
            Additional parameters for detection.

        Returns
        -------
        Tuple[bool, dict]
            A tuple where the first value is a boolean indicating whether a voice event
            was detected (`True` if detected, `False` otherwise). The second value is
            a dictionary containing additional detection information.
        """
        return self.detect(audio_data, input_parameters)

    @abstractmethod
    def detect(
        self, audio_data: NDArray, input_parameters: dict[str, Any]
    ) -> Tuple[bool, dict[str, Any]]:
        """
        Abstract method for detecting a voice event.

        Subclasses must implement this method to analyze audio data and determine
        whether a specific voice-related event has occurred.

        Parameters
        ----------
        audio_data : NDArray
            A NumPy array containing audio input data.
        input_parameters : dict of str to Any
            Additional parameters for detection.

        Returns
        -------
        Tuple[bool, dict]
            A tuple where the first value is a boolean indicating whether a voice event
            was detected (`True` if detected, `False` otherwise). The second value is
            a dictionary containing additional detection information.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Abstract method for resetting the voice detection model.

        Subclasses must implement this method to reset the internal state of the model.
        """
        pass


class BaseTranscriptionModel(ABC):
    """
    Abstract base class for speech transcription models.

    This class provides a standardized interface for speech-to-text models, where
    subclasses must implement the `transcribe` method to convert audio input into text.

    Parameters
    ----------
    model_name : str
        The name of the transcription model.
    sample_rate : int
        The sample rate of the input audio, in Hz.
    language : str, optional
        The language of the transcription output. Default is "en" (English).

    Attributes
    ----------
    model_name : str
        The name of the transcription model.
    sample_rate : int
        The sample rate of the input audio, in Hz.
    language : str
        The language of the transcription output.
    latest_transcription : str
        Stores the latest transcribed text.
    """

    def __init__(self, model_name: str, sample_rate: int, language: str = "en"):
        """
        Initializes the BaseTranscriptionModel with the model name, sample rate, and language.

        Parameters
        ----------
        model_name : str
            The name of the transcription model.
        sample_rate : int
            The sample rate of the input audio, in Hz.
        language : str, optional
            The language of the transcription output. Default is "en" (English).
        """
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.language = language

        self.latest_transcription = ""

    @abstractmethod
    def transcribe(self, data: NDArray[np.int16]) -> str:
        """
        Abstract method for transcribing speech from audio data.

        Subclasses must implement this method to convert the provided audio input into text.

        Parameters
        ----------
        data : NDArray[np.int16]
            A NumPy array containing the raw audio waveform data.

        Returns
        -------
        str
            The transcribed text from the audio input.
        """
        pass
