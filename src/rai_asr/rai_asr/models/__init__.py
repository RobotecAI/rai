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

from rai_asr.models.base import BaseTranscriptionModel, BaseVoiceDetectionModel
from rai_asr.models.local_whisper import LocalWhisper
from rai_asr.models.open_ai_whisper import OpenAIWhisper
from rai_asr.models.open_wake_word import OpenWakeWord
from rai_asr.models.silero_vad import SileroVAD

__all__ = [
    "BaseVoiceDetectionModel",
    "SileroVAD",
    "OpenWakeWord",
    "BaseTranscriptionModel",
    "LocalWhisper",
    "OpenAIWhisper",
]
