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
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

from scipy.signal import resample

try:
    import sounddevice as sd
except ImportError:
    logging.warning("Install sound_device module to use sound device features!")
    sd = None


import numpy as np


class SoundDeviceError(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)


@dataclass
class SoundDeviceConfig:
    stream: bool
    block_size: int
    dtype: str
    channels: int
    device_number: Optional[int]
    device_name: Optional[str]

    def __post_init__(self):
        if self.device_number is None and self.device_name is None:
            raise ValueError("Either 'device_number' or 'device_name' must be set.")


@dataclass
class InputSoundDeviceConfig(SoundDeviceConfig):
    consumer_sampling_rate: int


# TODO: add fields
@dataclass
class OutputSoundDeviceConfig(SoundDeviceConfig):
    pass


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

        if not sd:
            raise SoundDeviceError("SoundDeviceAPI requires sound_device module!")
        if config.device_name:
            self.device_name = config.device_name
            devices = sd.query_devices()
            devices = list(devices) if isinstance(devices, sd.DeviceList) else [devices]
            for device in devices:
                if device["name"] == config.device_name:  # type: ignore
                    self.device_number = device["index"]  # type: ignore
                    break
        else:
            self.device_number = config.device_number
        self.sample_rate = sd.query_devices(device=self.device_number, kind="input")[
            "default_samplerate"
        ]  # type: ignore

        self.read_flag = False
        self.write_flag = False
        if isinstance(config, InputSoundDeviceConfig):
            self.read_flag = True
        if isinstance(config, OutputSoundDeviceConfig):
            self.write_flag = True
        self.stream_flag = config.stream
        self.config = config
        self.in_stream = None
        self.out_stream = None

    def write(self, **kwargs):
        if not self.write_flag:
            raise SoundDeviceError(f"{self.device_name} does not support writing!")

    def open_write_stream(
        self,
        feed_data: Callable[[np.ndarray, int, Any, Any], None],
        on_done: Callable = lambda _: None,
    ):
        if not self.write_flag or not self.stream_flag:
            raise SoundDeviceError(
                f"{self.device_name} does not support streaming writing!"
            )

        assert isinstance(self.config, OutputSoundDeviceConfig)

        # def callback(indata: np.ndarray, frames: int, _, status: CallbackFlags):
        #     _ = frames
        #     flag_dict = {
        #         "input_overflow": status.input_overflow,
        #         "input_underflow": status.input_underflow,
        #         "output_overflow": status.output_overflow,
        #         "output_underflow": status.output_underflow,
        #         "priming_output": status.priming_output,
        #     }
        #     feed_data(indata, frames, flag_dict)

        try:
            assert sd is not None
            print(self.sample_rate)
            print(self.config.channels)
            print(self.config.dtype)
            print(self.device_number)
            self.out_stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.config.channels,
                device=self.device_number,
                dtype=self.config.dtype,
                callback=feed_data,
                finished_callback=on_done,
                # TODO: add callbacks
            )
        except AttributeError:
            raise SoundDeviceError(
                f"Device {self.device_name} has not been correctly configured"
            )
        self.out_stream.start()

    def read(self, **kwargs):
        if not self.read_flag:
            raise SoundDeviceError(f"{self.device_name} does not support reading!")

    def open_read_stream(
        self,
        on_feedback: Callable[[np.ndarray, dict[str, Any]], None],
        on_done: Callable = lambda _: None,
    ):
        if not self.write_flag or not self.stream_flag:
            raise SoundDeviceError(
                f"{self.device_name} does not support streaming reading!"
            )
        if self.in_stream is not None:
            raise SoundDeviceError(
                f"Stream for {self.device_name} is already open, close it first!"
            )
        from sounddevice import CallbackFlags

        assert isinstance(self.config, InputSoundDeviceConfig)

        def callback(indata: np.ndarray, frames: int, _, status: CallbackFlags):
            assert isinstance(
                self.config, InputSoundDeviceConfig
            )  # NOTE: need to do this twice, pyright doesn't understand the higher scope assert
            _ = frames
            indata = indata.flatten()
            sample_time_length = len(indata) / self.sample_rate
            if self.sample_rate != self.config.consumer_sampling_rate:
                indata = resample(
                    indata, int(sample_time_length * self.config.consumer_sampling_rate)
                )  # type: ignore
            flag_dict = {
                "input_overflow": status.input_overflow,
                "input_underflow": status.input_underflow,
                "output_overflow": status.output_overflow,
                "output_underflow": status.output_underflow,
                "priming_output": status.priming_output,
            }
            on_feedback(indata, flag_dict)

        try:
            assert sd is not None
            window_size_samples = int(
                self.config.block_size
                * self.sample_rate
                / self.config.consumer_sampling_rate
            )

            self.in_stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.config.channels,
                device=self.device_number,
                dtype=self.config.dtype,
                blocksize=window_size_samples,
                callback=callback,
                finished_callback=on_done,
            )
        except AttributeError:
            raise SoundDeviceError(
                f"Device {self.device_name} has not been correctly configured"
            )
        self.in_stream.start()

    def close_read_stream(self):
        if self.in_stream is not None:
            self.in_stream.stop()
            self.in_stream.close()
            self.in_stream = None
