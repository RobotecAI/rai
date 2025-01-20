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


from typing import Sequence, Tuple, Union

import sounddevice as sd

from rai.communication import HRIConnector, HRIMessage

from .api import InputSoundDeviceConfig, OutputSoundDeviceConfig


class SoundDeviceMessage(HRIMessage):
    pass


class SoundDeviceConnector(HRIConnector[SoundDeviceMessage]):
    """SoundDevice connector implementing the Human-Robot Interface.

    This class provides audio streaming capabilities while conforming to the
    HRIConnector interface. It supports starting and stopping audio streams
    but does not implement message passing or service calls.
    """

    def __init__(
        self,
        targets: Sequence[Tuple[str, OutputSoundDeviceConfig]],
        sources: Sequence[Tuple[str, InputSoundDeviceConfig]],
    ):
        configured_targets = [target[0] for target in targets]
        configured_sources = [source[0] for source in sources]
        for dev_target, dev_config in [*targets, *sources]:
            self.configure_device(dev_target, dev_config)

        super().__init__(configured_targets, configured_sources)
        self.streams = {}
        sd.default.latency = ("low", "low")

    def configure_device(
        self,
        target: str,
        config: Union[InputSoundDeviceConfig, OutputSoundDeviceConfig],
    ):
        pass
