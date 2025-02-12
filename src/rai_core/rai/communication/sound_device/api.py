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
from typing import Any, Callable, Optional

import numpy as np
from numpy._typing import NDArray
from pydub import AudioSegment
from scipy.signal import resample

try:
    import sounddevice as sd
except ImportError:
    logging.warning("Install sound_device module to use sound device features!")
    sd = None


class SoundDeviceError(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)


@dataclass
class SoundDeviceConfig:
    """
    Configuration settings for a sound device.

    This dataclass holds configuration parameters for audio input/output devices.
    It ensures that at least one identifier (`device_number` or `device_name`) is
    provided when initialized.

    Parameters
    ----------
    stream : bool, optional
        Whether the device should operate in streaming mode. Default is False.
    block_size : int, optional
        The block size for audio processing. Default is 1024.
    dtype : str, optional
        The data type of the audio stream (e.g., "int16", "float32"). Default is "int16".
    channels : int, optional
        The number of audio channels (e.g., 1 for mono, 2 for stereo). Default is 1.
    consumer_sampling_rate : Optional[int], optional
        The desired sampling rate for the audio consumer. Default is None.
    device_number : Optional[int], optional
        The device number for the sound device. If None, `device_name` must be set. Default is None.
    device_name : Optional[str], optional
        The name of the sound device. If None, `device_number` must be set. Default is None.
    is_input : bool, optional
        Indicates whether the device is used for input (recording). Default is False.
    is_output : bool, optional
        Indicates whether the device is used for output (playback). Default is False.

    Raises
    ------
    ValueError
        If neither `device_number` nor `device_name` is provided.
    """

    stream: bool = False
    block_size: int = 1024
    dtype: str = "int16"
    channels: int = 1
    consumer_sampling_rate: Optional[int] = None
    device_number: Optional[int] = None
    device_name: Optional[str] = None
    is_input: bool = False
    is_output: bool = False

    def __post_init__(self):
        if self.device_number is None and self.device_name is None:
            raise ValueError("Either 'device_number' or 'device_name' must be set.")


class SoundDeviceAPI:
    def __init__(self, config: SoundDeviceConfig):
        self.device_name = ""

        if not sd:
            raise SoundDeviceError("SoundDeviceAPI requires sound_device module!")
        if config.device_name:
            self.device_name = config.device_name
            devices = sd.query_devices()
            devices = list(devices) if isinstance(devices, sd.DeviceList) else [devices]
            for device in devices:
                if device["name"] == config.device_name:  # type: ignore
                    self.device_number = int(device["index"])  # type: ignore
                    break
        else:
            self.device_number = config.device_number
        self.sample_rate = int(
            sd.query_devices(device=self.device_number, kind="input")[
                "default_samplerate"
            ]  # type: ignore
        )

        self.read_flag = config.is_input
        self.write_flag = config.is_output
        self.stream_flag = config.stream
        self.config = config
        self.in_stream = None
        self.out_stream = None

    def write(self, data: AudioSegment, blocking: bool = False, loop: bool = False):
        """
        Write data to the sound device.

        Parameters
        ----------
        data : NDArray
            Data to be written.
        blocking : bool, optional
            If True, the function will block until the sound is played. Defaults to False.
        loop : bool, optional
            If True, the data will loop continuously. Defaults to False.

        Notes
        -----
        - If `blocking` is True, the function will block until the sound is played.
        - If both `blocking` and `loop` are True, the function will block indefinitely.
        - Calling this function will stop any sound that is currently playing and any
          recording currently happening.
        - Call `stop()` or `read()` to stop the sound.
        """
        if not self.write_flag:
            raise SoundDeviceError(f"{self.device_name} does not support writing!")
        assert sd is not None
        audio = np.array(data.get_array_of_samples())
        sd.play(
            audio,
            samplerate=data.frame_rate,
            blocking=blocking,
            loop=loop,
            device=self.device_number,
        )

    def read(self, time: float, blocking: bool = False) -> AudioSegment:
        """
        Read data from the sound device.

        Parameters
        ----------
        time : float
            Time in seconds to read.
        blocking : bool, optional
            If True, the function will block until the sound is read. If False,
            the function will return immediately with an unpopulated ndarray.
            Defaults to False.

        Returns
        -------
        NDArray
            Data read from the sound device.

        Notes
        -----
        - If `blocking` is True, the function will block until the sound is read.
        - If `blocking` is False, the function will return immediately with an
          unpopulated ndarray. The array will be populated with the sound data
          after the read is complete.
        - Calling this function will stop any sound that is currently playing
          and any recording currently happening.
        - Call `stop()` or `write()` to stop the recording.
        """

        if not self.read_flag:
            raise SoundDeviceError(f"{self.device_name} does not support reading!")
        assert sd is not None
        frames = int(time * self.sample_rate)
        recording = sd.rec(
            frames=frames,
            samplerate=self.sample_rate,
            channels=self.config.channels,
            device=self.device_number,
            blocking=blocking,
            dtype=self.config.dtype,
        )

        return AudioSegment(
            data=recording.flatten(),
            sample_width=recording.dtype.itemsize,
            frame_rate=self.sample_rate,
            channels=self.config.channels,
        )

    def stop(self):
        """
        Stop the sound device from playing or recording.

        Notes
        -----
        - This is a convenience function to stop the sound device from playing or recording.
        - It will stop any sound that is currently playing and any recording currently happening.
        """
        assert sd is not None
        sd.stop()

    def wait(self):
        """
        Wait for the sound device to finish playing or recording.

        Notes
        -----
        - This is a convenience function to wait for the sound device to finish playing or recording.
        - It will block until the sound is played or recorded.
        """
        assert sd is not None
        sd.wait()

    def open_write_stream(
        self,
        feed_data: Callable[[NDArray, int, Any, Any], None],
        on_done: Callable = lambda _: None,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
    ):
        if not self.write_flag or not self.stream_flag:
            raise SoundDeviceError(
                f"{self.device_name} does not support streaming writing!"
            )

        assert sd is not None
        from sounddevice import CallbackFlags

        def callback(indata: NDArray, frames: int, time: Any, status: CallbackFlags):
            _ = frames
            flag_dict = {
                "input_overflow": status.input_overflow,
                "input_underflow": status.input_underflow,
                "output_overflow": status.output_overflow,
                "output_underflow": status.output_underflow,
                "priming_output": status.priming_output,
            }
            feed_data(indata, frames, time, flag_dict)

        try:
            assert sd is not None
            sample_rate = self.sample_rate if sample_rate is None else sample_rate
            print(sample_rate)
            channels = self.config.channels if channels is None else channels
            self.out_stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=channels,
                device=self.device_number,
                dtype=self.config.dtype,
                callback=callback,
                finished_callback=on_done,
            )
        except AttributeError:
            raise SoundDeviceError(
                f"Device {self.device_name} has not been correctly configured"
            )
        self.out_stream.start()

    def open_read_stream(
        self,
        on_feedback: Callable[[NDArray, dict[str, Any]], None],
        on_done: Callable = lambda _: None,
    ):
        if not self.read_flag or not self.stream_flag:
            raise SoundDeviceError(
                f"{self.device_name} does not support streaming reading!"
            )
        if self.in_stream is not None:
            raise SoundDeviceError(
                f"Stream for {self.device_name} is already open, close it first!"
            )
        from sounddevice import CallbackFlags

        def callback(indata: NDArray, frames: int, _, status: CallbackFlags):
            _ = frames
            indata = indata.flatten()
            sample_time_length = len(indata) / self.sample_rate
            if (
                self.sample_rate != self.config.consumer_sampling_rate
                and self.config.consumer_sampling_rate is not None
            ):
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
            if self.config.consumer_sampling_rate is None:
                window_size_samples = self.config.block_size * self.sample_rate
            else:
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

    def close_write_stream(self):
        if self.out_stream is not None:
            self.out_stream.stop()
            self.out_stream.close()
            self.out_stream = None
