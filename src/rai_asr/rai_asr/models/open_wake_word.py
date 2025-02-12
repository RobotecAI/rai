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

from typing import Any, Tuple

from numpy.typing import NDArray
from openwakeword.model import Model as OWWModel
from openwakeword.utils import download_models

from rai_asr.models import BaseVoiceDetectionModel


class OpenWakeWord(BaseVoiceDetectionModel):
    """
    A wake word detection model using the Open Wake Word framework.

    This class loads a specified wake word model and detects whether a wake word is present
    in the provided audio input.

    Parameters
    ----------
    wake_word_model_path : str
        Path to the wake word model file or name of a standard one.
    threshold : float, optional
        The confidence threshold for wake word detection. If a prediction surpasses this
        value, the model will trigger a wake word detection. Default is 0.1.

    Attributes
    ----------
    model_name : str
        The name of the model, set to `"open_wake_word"`.
    model : OWWModel
        The Open Wake Word model instance used for inference.
    threshold : float
        The confidence threshold for determining wake word detection.
    """

    def __init__(self, wake_word_model_path: str, threshold: float = 0.1):
        """
        Initializes the OpenWakeWord detection model.

        Parameters
        ----------
        wake_word_model_path : str
            Path to the wake word model file.
        threshold : float, optional
            Confidence threshold for wake word detection. Default is 0.1.
        """
        super(OpenWakeWord, self).__init__()
        self.model_name = "open_wake_word"
        download_models()
        self.model = OWWModel(
            wakeword_models=[
                wake_word_model_path,
            ],
            inference_framework="onnx",
        )
        self.threshold = threshold

    def detect(
        self, audio_data: NDArray, input_parameters: dict[str, Any]
    ) -> Tuple[bool, dict[str, Any]]:
        """
        Detects whether a wake word is present in the given audio data.

        This method runs inference on the provided audio data and determines whether
        the detected confidence surpasses the threshold. If so, it resets the model
        and returns `True`, indicating a wake word detection.

        Parameters
        ----------
        audio_data : NDArray
            A NumPy array representing the input audio data.
        input_parameters : dict of str to Any
            Additional input parameters to be included in the output.

        Returns
        -------
        Tuple[bool, dict]
            A tuple where the first value is a boolean indicating whether the wake word
            was detected (`True` if detected, `False` otherwise). The second value is
            a dictionary containing predictions and confidence values for them.

        Raises
        ------
        Exception
            If the predictions returned by the model are not in the expected dictionary format.
        """
        predictions = self.model.predict(audio_data)
        ret = input_parameters.copy()
        ret.update({self.model_name: {"predictions": predictions}})
        if not isinstance(predictions, dict):
            raise Exception(
                f"Unexpected format from model predict {type(predictions)}:{predictions}"
            )
        for _, value in predictions.items():  # type ignore
            if value > self.threshold:
                self.model.reset()
                return True, ret
        return False, ret
