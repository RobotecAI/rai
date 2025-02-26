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


from typing import Callable, Literal, NamedTuple, Optional, Tuple

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


class AudioParams(NamedTuple):
    sample_rate: int
    in_channels: int
    out_channels: int


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

    def get_audio_params(self, target: str) -> AudioParams:
        return AudioParams(
            self.devices[target].sample_rate,
            self.devices[target].in_channels,
            self.devices[target].out_channels,
        )

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
                self.devices[target].write(message.audios[0])
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
                audios=[recording],
            )
            ret = SoundDeviceMessage(payload)
        else:
            if message.audios is not None:
                self.devices[target].write(message.audios[0], blocking=True)
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
            sample_rate = kwargs.get("sample_rate", None)
            channels = kwargs.get("channels", None)

            self.devices[target].open_write_stream(
                on_feedback, on_done, sample_rate, channels
            )
            self.action_handles[handle] = (target, False)

        return handle

    def terminate_action(self, action_handle: str, **kwargs):
        target, read = self.action_handles[action_handle]
        if read:
            self.devices[target].close_read_stream()
        else:
            self.devices[target].close_write_stream()
        del self.action_handles[action_handle]
