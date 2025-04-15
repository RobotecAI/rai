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

import importlib.util

if importlib.util.find_spec("sounddevice") is None:
    raise ImportError(
        "This feature is based on sounddevice. Make sure sounddevice is installed."
    )

from .api import SoundDeviceAPI, SoundDeviceConfig, SoundDeviceError
from .connector import SoundDeviceConnector, SoundDeviceMessage

__all__ = [
    "SoundDeviceAPI",
    "SoundDeviceConfig",
    "SoundDeviceConnector",
    "SoundDeviceError",
    "SoundDeviceMessage",
]
