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

from .base import BaseTranscriptionModel, BaseVoiceDetectionModel
from .local_whisper import FasterWhisper, LocalWhisper
from .open_ai_whisper import OpenAIWhisper
from .open_wake_word import OpenWakeWord
from .silero_vad import SileroVAD

__all__ = [
    "BaseTranscriptionModel",
    "BaseVoiceDetectionModel",
    "FasterWhisper",
    "LocalWhisper",
    "OpenAIWhisper",
    "OpenWakeWord",
    "SileroVAD",
]
