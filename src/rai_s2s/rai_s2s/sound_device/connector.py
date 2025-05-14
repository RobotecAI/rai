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


from typing import Callable, NamedTuple, Optional, Tuple

from rai.communication import HRIConnector, HRIMessage

from rai_s2s.sound_device.api import (
    SoundDeviceAPI,
    SoundDeviceConfig,
    SoundDeviceError,
)


class SoundDeviceMessage(HRIMessage):
    read: bool = False
    stop: bool = False
    duration: Optional[float] = None


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
        self.devices: dict[str, SoundDeviceAPI] = {}
        self.action_handles: dict[str, Tuple[str, bool]] = {}

        tmp_devs = targets + sources
        all_names = [dev[0] for dev in tmp_devs]
        all_configs = [dev[1] for dev in tmp_devs]

        for dev_target, dev_config in zip(all_names, all_configs):
            self.configure_device(dev_target, dev_config)

        super().__init__()

    def get_audio_params(self, target: str) -> AudioParams:
        """
        Retrieve audio parameters for a specified device.

        Parameters
        ----------
        target : str
            The name of the device for which to retrieve audio parameters.

        Returns
        -------
        AudioParams
            An `AudioParams` object containing the sample rate, input channels,
            and output channels of the specified device.
        """
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
        """
        Configure and register a new audio device.

        Parameters
        ----------
        target : str
            The name identifier for the device to configure.
        config : SoundDeviceConfig
            The configuration settings to initialize the device with.

        Notes
        -----
        The configured device is stored in `self.devices` under the given target name.
        """
        self.devices[target] = SoundDeviceAPI(config)

    def send_message(self, message: SoundDeviceMessage, target: str, **kwargs) -> None:
        """
        Send an audio message to a specified device.

        Parameters
        ----------
        message : SoundDeviceMessage
            The message containing audio data or control commands (e.g., stop or read request).
        target : str
            The name of the target device to which the message will be sent.
        **kwargs
            Additional keyword arguments (currently unused).

        Raises
        ------
        SoundDeviceError
            If the message requests reading (unsupported) or if no audio data is provided
            when attempting to play audio.

        Notes
        -----
        - If `message.stop` is `True`, the device will be stopped.
        - If `message.read` is `True`, an error is raised (reading must use actions or services).
        - Otherwise, attempts to write audio data to the device.
        """
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
        """
        SoundDeviceConnector does not support receiving messages. For recording use start_action or service_call with read=True.

        """
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
        """
        Perform a blocking service call to a sound device for playback or recording.

        Depending on the message content, either records audio from the device or plays
        provided audio data in a blocking manner.

        Parameters
        ----------
        message : SoundDeviceMessage
            The message specifying the operation: playback (with audio data) or recording (read flag).
        target : str
            The name of the target device on which to perform the action.
        timeout_sec : float, optional
            Timeout for the operation in seconds. Currently unused. Defaults to `1.0`.
        duration : float, optional
            Duration for recording audio in seconds. Only relevant when reading. Defaults to `1.0`.
        **kwargs
            Additional keyword arguments (currently unused).

        Returns
        -------
        SoundDeviceMessage
            A new message containing recorded audio if reading, or an empty message after successful playback.

        Raises
        ------
        SoundDeviceError
            If stopping is requested (unsupported in this method) or if playback is attempted
            without providing audio data.

        Notes
        -----
        - To stop a device, use `send_message` with `stop=True` instead.
        - Audio recording is done synchronously with `blocking=True`.
        """
        if message.stop:
            raise SoundDeviceError("For stopping use send_message with stop=True.")
        elif message.read:
            recording = self.devices[target].read(duration, blocking=True)

            ret = SoundDeviceMessage(
                text="", audios=[recording], message_author="human"
            )
        else:
            if message.audios is not None:
                self.devices[target].write(message.audios[0], blocking=True)
            else:
                raise SoundDeviceError("Failed to provice audios in message to play")
            ret = SoundDeviceMessage(message_author="human")
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
        """
        Start an asynchronous streaming action on a sound device.

        Depending on the action data, either opens a read (recording) or write (playback)
        audio stream. Handles for ongoing actions are tracked internally.

        Parameters
        ----------
        action_data : SoundDeviceMessage, optional
            The action request containing operation details. Must not be `None`.
            If `read` is `True`, a read (recording) stream is opened; otherwise, a write (playback) stream.
        target : str
            The name of the target device to interact with.
        on_feedback : Callable
            Callback function to handle feedback during the stream.
        on_done : Callable
            Callback function called upon stream completion.
        timeout_sec : float, optional
            Timeout for starting the stream, in seconds. Defaults to `1.0`. (Currently unused.)
        **kwargs
            Additional parameters:
            - `sample_rate` (int, optional): Desired sample rate for playback streams.
            - `channels` (int, optional): Number of channels for playback streams.

        Returns
        -------
        str
            A unique handle identifying the started action.

        Raises
        ------
        SoundDeviceError
            If `action_data` is not provided (`None`).

        Notes
        -----
        - For recording streams, `open_read_stream` is used.
        - For playback streams, `open_write_stream` is used, optionally customized by `sample_rate` and `channels`.
        - Started actions are stored in `self.action_handles` with the handle as the key.
        """
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

    def shutdown(self):
        for target in self.devices:
            self.devices[target].close_read_stream()
            self.devices[target].close_write_stream()
