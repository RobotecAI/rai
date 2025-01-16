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

from rai.communication import HRIConnector, HRIMessage


class SoundDeviceError(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)


class AudioInputDeviceConfig(TypedDict):
    block_size: int
    consumer_sampling_rate: int
    target_sampling_rate: int
    dtype: str
    device_number: Optional[int]


class ConfiguredAudioInputDevice:
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

    def __init__(self, config: AudioInputDeviceConfig):
        self.sample_rate = sd.query_devices(
            device=config["device_number"], kind="input"
        )[
            "default_samplerate"
        ]  # type: ignore
        self.consumer_sampling_rate = config["consumer_sampling_rate"]
        self.window_size_samples = int(
            config["block_size"] * self.sample_rate / config["consumer_sampling_rate"]
        )
        self.target_sampling_rate = int(config["target_sampling_rate"])
        self.dtype = config["dtype"]


class StreamingAudioInputDevice(HRIConnector):
    """Audio input device connector implementing the Human-Robot Interface.

    This class provides audio streaming capabilities while conforming to the
    HRIConnector interface. It supports starting and stopping audio streams
    but does not implement message passing or service calls.
    """

    def __init__(self):
        self.streams = {}
        sd.default.latency = ("low", "low")
        self.configred_devices: dict[str, ConfiguredAudioInputDevice] = {}

    def configure_device(self, target: str, config: AudioInputDeviceConfig):
        if target.isdigit():
            if config.get("device_number") is None:
                config["device_number"] = int(target)
            elif config["device_number"] != int(target):
                raise SoundDeviceError(
                    "device_number in config must be the same as target"
                )
            self.configred_devices[target] = ConfiguredAudioInputDevice(config)
        else:
            raise SoundDeviceError("target must be a device number!")

    def send_message(self, message: HRIMessage, target: str) -> None:
        raise SoundDeviceError(
            "StreamingAudioInputDevice does not support sending messages"
        )

    def receive_message(self, source: str, timeout_sec: float = 1.0) -> HRIMessage:
        raise SoundDeviceError(
            "StreamingAudioInputDevice does not support receiving messages"
        )

    def service_call(
        self, message: HRIMessage, target: str, timeout_sec: float = 1.0
    ) -> HRIMessage:
        raise SoundDeviceError("StreamingAudioInputDevice does not support services")

    def start_action(
        self,
        action_data: Optional[HRIMessage],
        target: str,
        on_feedback: Callable[[np.ndarray, dict[str, Any]], None],
        on_done: Callable = lambda _: None,
        timeout_sec: float = 1.0,
    ) -> str:
        """Start streaming audio from the specified device.

        Args:
            action_data: Optional message containing action parameters
            target: Device ID to stream from
            on_feedback: Callback for processing audio data
            on_done: Callback invoked when streaming ends
            timeout_sec: Timeout in seconds for starting the stream

        Returns:
            str: Handle for managing the stream

        Raises:
            SoundDeviceError: If device is not configured or initialization fails
        """

        target_device = self.configred_devices.get(target)
        if target_device is None:
            raise SoundDeviceError(f"Device {target} has not been configured")

        def callback(indata: np.ndarray, frames: int, _, status: CallbackFlags):
            indata = indata.flatten()
            sample_time_length = len(indata) / target_device.target_sampling_rate
            if target_device.sample_rate != target_device.target_sampling_rate:
                indata = resample(indata, int(sample_time_length * target_device.target_sampling_rate))  # type: ignore
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
                finished_callback=on_done,
            )
        except AttributeError:
            raise SoundDeviceError(f"Device {target} has not been correctly configured")
        stream.start()
        self.streams[handle] = stream
        return handle

    def terminate_action(self, action_handle: str):
        self.streams[action_handle].stop()
