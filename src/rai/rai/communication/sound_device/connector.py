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


import base64
import io
from typing import Callable, Literal, Optional, Tuple

import numpy as np
from scipy.io import wavfile

try:
    import sounddevice as sd
except ImportError as e:
    raise ImportError(
        "The sounddevice package is required to use the SoundDeviceConnector."
    ) from e

from rai.communication import HRIConnector, HRIMessage, HRIPayload
from rai.communication.sound_device import (
    SoundDeviceAPI,
    SoundDeviceConfig,
    SoundDeviceError,
)


class SoundDeviceMessage(HRIMessage):
    read: bool = False
    stop: bool = False
    duration: Optional[float] = None

    def __init__(
        self,
        payload: Optional[HRIPayload] = None,
        message_author: Literal["ai", "human"] = "human",
        read: bool = False,
        stop: bool = False,
        duration: Optional[float] = None,
    ):
        if payload is None:
            payload = HRIPayload(text="")
        super().__init__(payload, message_author)
        self.read = read
        self.stop = stop
        self.duration = duration


class SoundDeviceConnector(HRIConnector[SoundDeviceMessage]):
    """SoundDevice connector implementing the Human-Robot Interface.

    This class provides audio streaming capabilities while conforming to the
    HRIConnector interface. It supports starting and stopping audio streams
    but does not implement message passing or service calls.
    """

    def __init__(
        self,
        targets: list[Tuple[str, SoundDeviceConfig]],
        sources: list[Tuple[str, SoundDeviceConfig]],
    ):
        configured_targets = [target[0] for target in targets]
        configured_sources = [source[0] for source in sources]
        self.devices: dict[str, SoundDeviceAPI] = {}
        self.action_handles: dict[str, Tuple[str, bool]] = {}

        tmp_devs = targets + sources
        all_names = [dev[0] for dev in tmp_devs]
        all_configs = [dev[1] for dev in tmp_devs]

        for dev_target, dev_config in zip(all_names, all_configs):
            self.configure_device(dev_target, dev_config)

        super().__init__(configured_targets, configured_sources)
        sd.default.latency = ("low", "low")  # type: ignore

    def configure_device(
        self,
        target: str,
        config: SoundDeviceConfig,
    ):
        self.devices[target] = SoundDeviceAPI(config)

    def send_message(self, message: SoundDeviceMessage, target: str, **kwargs) -> None:
        if message.stop:
            self.devices[target].stop()
        elif message.read:
            raise SoundDeviceError(
                "For recording use start_action or service_call with read=True."
            )
        else:
            if message.audios is not None:
                wav_bytes = base64.b64decode(message.audios[0])
                wav_buffer = io.BytesIO(wav_bytes)
                _, audio_data = wavfile.read(wav_buffer)
                self.devices[target].write(audio_data)
            else:
                raise SoundDeviceError("Failed to provice audios in message to play")

    def receive_message(
        self, source: str, timeout_sec: float = 1.0, **kwargs
    ) -> SoundDeviceMessage:
        raise SoundDeviceError(
            "SoundDeviceConnector does not support receiving messages. For recording use start_action or service_call with read=True."
        )

    def service_call(
        self,
        message: SoundDeviceMessage,
        target: str,
        timeout_sec: float = 1.0,
        *,
        duration: float = 1.0,
        **kwargs,
    ) -> SoundDeviceMessage:
        if message.stop:
            raise SoundDeviceError("For stopping use send_message with stop=True.")
        elif message.read:
            recording = self.devices[target].read(duration, blocking=True)
            payload = HRIPayload(
                text="",
                audios=[
                    base64.b64encode(recording).decode("utf-8")
                ],  # TODO: refactor once utility functions for encoding/decoding are available
            )
            ret = SoundDeviceMessage(payload)
        else:
            if message.audios is not None:
                wav_bytes = base64.b64decode(
                    message.audios[0]
                )  # TODO: refactor once utility functions for encoding/decoding are available
                wav_buffer = io.BytesIO(wav_bytes)
                _, audio_data = wavfile.read(wav_buffer)
                audio_data = np.array(audio_data)
                self.devices[target].write(audio_data, blocking=True)
            else:
                raise SoundDeviceError("Failed to provice audios in message to play")
            ret = SoundDeviceMessage()
        return ret

    def start_action(
        self,
        action_data: Optional[SoundDeviceMessage],
        target: str,
        on_feedback: Callable,
        on_done: Callable,
        timeout_sec: float = 1.0,
        **kwargs,
    ) -> str:
        handle = self._generate_handle()
        if action_data is None:
            raise SoundDeviceError("action_data must be provided")
        elif action_data.read:
            self.devices[target].open_read_stream(on_feedback, on_done)
            self.action_handles[handle] = (target, True)
        else:
            self.devices[target].open_write_stream(on_feedback, on_done)
            self.action_handles[handle] = (target, False)

        return handle

    def terminate_action(self, action_handle: str, **kwargs):
        target, read = self.action_handles[action_handle]
        if read:
            self.devices[target].close_read_stream()
        else:
            self.devices[target].close_write_stream()
        del self.action_handles[action_handle]
