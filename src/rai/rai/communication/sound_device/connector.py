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


from typing import Callable, Literal, Optional, Sequence, Tuple, Union

try:
    import sounddevice as sd
except ImportError as e:
    raise ImportError(
        "The sounddevice package is required to use the SoundDeviceConnector."
    ) from e

from rai.communication import HRIConnector, HRIMessage, HRIPayload

from .api import InputSoundDeviceConfig, OutputSoundDeviceConfig, SoundDeviceAPI


class SoundDeviceMessage(HRIMessage):
    read: bool = False

    def __init__(
        self,
        payload: Optional[HRIPayload] = None,
        message_author: Literal["ai", "human"] = "human",
        read: bool = False,
    ):
        if payload is None:
            payload = HRIPayload(text="")
        super().__init__(payload, message_author)
        self.read = read


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
        self.devices = {}
        self.action_handles = {}
        for dev_target, dev_config in [*targets, *sources]:
            self.configure_device(dev_target, dev_config)

        super().__init__(configured_targets, configured_sources)
        sd.default.latency = ("low", "low")  # type: ignore

    def configure_device(
        self,
        target: str,
        config: Union[InputSoundDeviceConfig, OutputSoundDeviceConfig],
    ):
        self.devices[target] = SoundDeviceAPI(config)

    def send_message(self, message: SoundDeviceMessage, target: str):
        pass

    def receive_message(
        self, source: str, timeout_sec: float = 1.0
    ) -> SoundDeviceMessage:
        pass

    def service_call(
        self, message: SoundDeviceMessage, target: str, timeout_sec: float = 1.0
    ) -> SoundDeviceMessage:
        pass

    def start_action(
        self,
        action_data: Optional[SoundDeviceMessage],
        target: str,
        on_feedback: Callable,
        on_done: Callable,
        timeout_sec: float = 1.0,
    ) -> str:
        if action_data is None:
            raise ValueError("action_data must be provided")
        elif action_data.read:
            self.devices[target].open_read_stream(on_feedback, on_done)
        else:
            raise ValueError(
                "SoundDeviceConnector does not support writing"
            )  # TODO: Implement writing

        handle = self._generate_handle()
        self.action_handles[handle] = (target, True)
        return handle

    def terminate_action(self, action_handle: str):
        target, read = self.action_handles[action_handle]
        if read:
            self.devices[target].in_stream.stop()
        else:
            raise ValueError(
                "SoundDeviceConnector does not support writing"
            )  # TODO: Implement writing
        del self.action_handles[action_handle]
