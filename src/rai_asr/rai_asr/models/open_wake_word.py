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
    def __init__(self, wake_word_model_path: str, threshold: float = 0.5):
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

    def detected(
        self, audio_data: NDArray, input_parameters: dict[str, Any]
    ) -> Tuple[bool, dict[str, Any]]:
        print(len(audio_data))
        predictions = self.model.predict(audio_data)
        ret = input_parameters.copy()
        ret.update({self.model_name: {"predictions": predictions}})
        for key, value in predictions.items():
            if value > self.threshold:
                self.model.reset()
                return True, ret
        return False, ret
