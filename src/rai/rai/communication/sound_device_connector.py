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

from typing import Any, Callable, Optional, TypedDict

import numpy as np
import sounddevice as sd
from scipy.signal import resample
from sounddevice import CallbackFlags

from rai.communication.base_connector import BaseConnector, BaseMessage


class SoundDeviceError(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)


class DeviceConfig(TypedDict):
    kind: str
    block_size: int
    consumer_sampling_rate: int
    target_smpling_rate: int
    dtype: str
    device_number: Optional[int]


class ConfiguredDevice:
    """
    A class to store the configuration of an audio device

    Attributes
    ----------
    sample_rate (int): Device sample rate
    consumer_sampling_rate (int): The sampling rate of the consumer
    window_size_samples (int): The size of the window in samples
    target_sampling_rate (int): The target sampling rate
    dtype (str): The data type of the audio samples
    """

    def __init__(self, config: DeviceConfig):
        self.sample_rate = sd.query_devices(
            device=config["device_number"], kind=config["kind"]
        )[
            "default_samplerate"
        ]  # type: ignore
        self.consumer_sampling_rate = config["consumer_sampling_rate"]
        self.window_size_samples = int(
            config["block_size"] * self.sample_rate / config["consumer_sampling_rate"]
        )
        self.target_sampling_rate = int(config["target_smpling_rate"])
        self.dtype = config["dtype"]


class StreamingAudioInputDevice(BaseConnector):
    def __init__(self):
        self.streams = {}
        sd.default.latency = ("low", "low")
        self.configred_devices: dict[str, ConfiguredDevice] = {}

    def configure_device(self, target: str, config: DeviceConfig):
        if target.isdigit():
            if config.get("device_number") is None:
                config["device_number"] = int(target)
            elif config["device_number"] != int(target):
                raise SoundDeviceError(
                    "device_number in config must be the same as target"
                )
            self.configred_devices[target] = ConfiguredDevice(config)
        else:
            raise SoundDeviceError("target must be a device number!")

    def send_message(self, msg: BaseMessage, target: str) -> None:
        raise SoundDeviceError(
            "StreamingAudioInputDevice does not suport sending messages"
        )

    def receive_message(self, source: str) -> BaseMessage:
        raise SoundDeviceError(
            "StreamingAudioInputDevice does not suport receiving messages messages"
        )

    def send_and_wait(self, target: str) -> BaseMessage:
        raise SoundDeviceError(
            "StreamingAudioInputDevice does not suport sending messages"
        )

    def start_action(
        self,
        target: str,
        on_feedback: Callable[[np.ndarray, dict[str, Any]], None],
        on_finish: Callable = lambda _: None,
    ) -> str:

        target_device = self.configred_devices.get(target)
        if target_device is None:
            raise SoundDeviceError(f"Device {target} has not been configured")

        def callback(indata: np.ndarray, frames: int, _, status: CallbackFlags):
            indata = indata.flatten()
            sample_time_length = len(indata) / target_device.target_sampling_rate
            if target_device.sample_rate != target_device.target_sampling_rate:
                indata = resample(indata, int(sample_time_length * target_device.target_samping_rate))  # type: ignore
            flag_dict = {
                "input_overflow": status.input_overflow,
                "input_underflow": status.input_underflow,
                "output_overflow": status.output_overflow,
                "output_underflow": status.output_underflow,
                "priming_output": status.priming_output,
            }
            on_feedback(indata, flag_dict)

        handle = self._generate_handle()
        try:
            stream = sd.InputStream(
                samplerate=target_device.sample_rate,
                channels=1,
                device=int(target),
                dtype=target_device.dtype,
                blocksize=target_device.window_size_samples,
                callback=callback,
                finished_callback=on_finish,
            )
        except AttributeError:
            raise SoundDeviceError(f"Device {target} has not been correctly configured")
        stream.start()
        self.streams[handle] = stream
        return handle

    def terminate_action(self, action_handle: str):
        self.streams[action_handle].stop()
