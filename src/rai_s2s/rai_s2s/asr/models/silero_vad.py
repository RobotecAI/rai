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

from typing import Any, Literal, Tuple

import numpy as np
import torch
from numpy.typing import NDArray

from rai_s2s.asr.models import BaseVoiceDetectionModel


class SileroVAD(BaseVoiceDetectionModel):
    """
    Voice Activity Detection (VAD) model using SileroVAD.

    This class loads the SileroVAD model from Torch Hub and detects speech presence in an audio signal.
    It supports two sampling rates: 8000 Hz and 16000 Hz.

    Parameters
    ----------
    sampling_rate : Literal[8000, 16000], optional
        The sampling rate of the input audio. Must be either 8000 or 16000. Default is 16000.
    threshold : float, optional
        Confidence threshold for voice detection. If the VAD confidence exceeds this threshold,
        the method returns `True` (indicating voice presence). Default is 0.5.

    Attributes
    ----------
    model_name : str
        Name of the VAD model, set to `"silero_vad"`.
    model : torch.nn.Module
        The loaded SileroVAD model.
    sampling_rate : int
        The sampling rate of the input audio (either 8000 or 16000).
    window_size : int
        The size of the processing window, determined by the sampling rate.
        - 512 samples for 16000 Hz
        - 256 samples for 8000 Hz
    threshold : float
        Confidence threshold for determining voice activity.

    Raises
    ------
    ValueError
        If an unsupported sampling rate is provided.
    """

    def __init__(self, sampling_rate: Literal[8000, 16000] = 16000, threshold=0.5):
        super(SileroVAD, self).__init__()
        self.model_name = "silero_vad"
        self.model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model=self.model_name,
        )  # type: ignore
        # NOTE: See silero vad implementation: https://github.com/snakers4/silero-vad/blob/9060f664f20eabb66328e4002a41479ff288f14c/src/silero_vad/utils_vad.py#L61
        if sampling_rate == 16000:
            self.sampling_rate = 16000
            self.window_size = 512
        elif sampling_rate == 8000:
            self.sampling_rate = 8000
            self.window_size = 256
        else:
            raise ValueError(
                "Only 8000 and 16000 sampling rates are supported"
            )  # TODO: consider if this should be a ValueError or something else
        self.threshold = threshold

    def _int2float(self, sound: NDArray[np.int16]):
        converted_sound = sound.astype("float32")
        converted_sound *= 1 / 32768
        converted_sound = converted_sound.squeeze()
        return converted_sound

    def detect(
        self, audio_data: NDArray, input_parameters: dict[str, Any]
    ) -> Tuple[bool, dict[str, Any]]:
        """
        Detects voice activity in the given audio data.

        This method processes a window of the most recent audio samples, computes a confidence score
        using the SileroVAD model, and determines if the confidence exceeds the specified threshold.

        Parameters
        ----------
        audio_data : NDArray
            A NumPy array containing audio input data.
        input_parameters : dict of str to Any
            Additional parameters for detection.

        Returns
        -------
        Tuple[bool, dict]
            - A boolean indicating whether voice activity was detected (`True` if detected, `False` otherwise).
            - A dictionary containing the computed VAD confidence score.
        """
        vad_confidence = self.model(
            torch.tensor(self._int2float(audio_data[-self.window_size :])),
            self.sampling_rate,
        ).item()
        ret = input_parameters.copy()
        ret.update({self.model_name: {"vad_confidence": vad_confidence}})

        return vad_confidence > self.threshold, ret

    def reset(self):
        """
        Resets the voice activity detection model.
        """
        self.model.reset()
