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

from dataclasses import dataclass
from typing import Optional, Union


class SoundDeviceError(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)


@dataclass
class InputSoundDeviceConfig:
    block_size: int
    consumer_sampling_rate: int
    target_sampling_rate: int
    dtype: str
    stream: bool
    device_number: Optional[int]


# TODO: add fields
@dataclass
class OutputSoundDeviceConfig:
    stream: bool
    block_size: int


@dataclass
class IOSoundDeviceConfig(InputSoundDeviceConfig, OutputSoundDeviceConfig):
    pass


class SoundDeviceAPI:

    def __init__(
        self,
        config: Union[
            InputSoundDeviceConfig, OutputSoundDeviceConfig, IOSoundDeviceConfig
        ],
    ):
        self.device_name = ""
        self.read_flag = False
        self.write_flag = False
        if isinstance(config, InputSoundDeviceConfig):
            self.read_flag = True
        if isinstance(config, OutputSoundDeviceConfig):
            self.write_flag = True
        self.stream_flag = config.stream

    def write(self, **kwargs):
        if not self.write_flag:
            raise SoundDeviceError(f"{self.device_name} does not support writing!")

    def write_stream(self, **kwargs):
        if not self.write_flag or not self.stream_flag:
            raise SoundDeviceError(
                f"{self.device_name} does not support streaming writing!"
            )

    def read(self, **kwargs):
        if not self.read_flag:
            raise SoundDeviceError(f"{self.device_name} does not support reading!")

    def read_stream(self, **kwargs):
        if not self.write_flag or not self.stream_flag:
            raise SoundDeviceError(
                f"{self.device_name} does not support streaming reading!"
            )
