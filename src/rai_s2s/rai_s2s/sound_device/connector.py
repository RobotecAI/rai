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


from typing import Any, Callable, NamedTuple, Optional, Tuple

from numpy.typing import NDArray
from rai.content import Audio

from rai_s2s.sound_device.api import SoundDeviceAPI, SoundDeviceConfig


class AudioParams(NamedTuple):
    sample_rate: int
    in_channels: int
    out_channels: int


class SoundDeviceConnector:
    """Plays and records audio on named sound devices, as single clips or
    streams. Audio is :class:`rai.content.Audio`."""

    def __init__(
        self,
        targets: list[Tuple[str, SoundDeviceConfig]],
        sources: list[Tuple[str, SoundDeviceConfig]],
    ):
        self.devices: dict[str, SoundDeviceAPI] = {
            name: SoundDeviceAPI(config) for name, config in targets + sources
        }

    def get_audio_params(self, device: str) -> AudioParams:
        """Sample rate, input channels and output channels of a device."""
        api = self.devices[device]
        return AudioParams(api.sample_rate, api.in_channels, api.out_channels)

    def configure_device(self, device: str, config: SoundDeviceConfig) -> None:
        """Register (or replace) a device under a name."""
        self.devices[device] = SoundDeviceAPI(config)

    def play(self, target: str, audio: Audio, *, blocking: bool = False) -> None:
        """Play a clip on target; with blocking=True, return when it ends."""
        self.devices[target].write(audio, blocking=blocking)

    def record(self, source: str, duration: float) -> Audio:
        """Record duration seconds from source (blocks)."""
        return self.devices[source].read(duration, blocking=True)

    def stop(self, device: str) -> None:
        """Stop the clip playing or recording on device."""
        self.devices[device].stop()

    def open_input_stream(
        self,
        source: str,
        on_samples: Callable[[NDArray, dict[str, Any]], None],
        on_done: Callable[[], None] = lambda: None,
    ) -> None:
        """Stream from source: on_samples(samples, status_flags) per block."""
        self.devices[source].open_read_stream(on_samples, on_done)

    def open_output_stream(
        self,
        target: str,
        fill: Callable[[NDArray, int, Any, dict[str, Any]], None],
        on_done: Callable[[], None] = lambda: None,
        *,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
    ) -> None:
        """Stream to target: fill(outdata, frames, time, status_flags) writes
        each block into outdata."""
        self.devices[target].open_write_stream(fill, on_done, sample_rate, channels)

    def close_streams(self, device: Optional[str] = None) -> None:
        """Close the streams of device, or of every device."""
        for name in [device] if device is not None else list(self.devices):
            self.devices[name].close_read_stream()
            self.devices[name].close_write_stream()

    def close(self) -> None:
        self.close_streams()
