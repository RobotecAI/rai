# Copyright (C) 2025 Robotec.AI
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

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from rai.content import Audio

from rai_s2s.tts.models import OpenTTS, TTSModel


class Tone(TTSModel):
    def __init__(self, audio: Audio) -> None:
        self.audio = audio

    def get_speech(self, text: str) -> Audio:
        return self._resample(self.audio)

    def get_tts_params(self) -> tuple[int, int]:
        return self.audio.sample_rate, self.audio.channels


@pytest.mark.parametrize("dtype", [np.int16, np.float32])
@pytest.mark.parametrize("channels", [1, 2])
def test_resample_keeps_type_channels_and_pitch(dtype, channels):
    t = np.arange(24000) / 24000
    wave = 0.5 * np.sin(2 * np.pi * 440 * t)[:, None].repeat(channels, axis=1)
    samples = (wave * 32767).astype(np.int16) if dtype == np.int16 else wave
    tts = Tone(Audio(samples.astype(dtype), 24000))
    assert tts.get_speech("a") is tts.audio  # no target rate set
    tts.set_tts_params(48000, channels)
    out = tts.get_speech("a")
    assert (out.sample_rate, out.channels, out.samples.dtype) == (
        48000,
        channels,
        dtype,
    )
    assert len(out.samples) == 48000
    spectrum = np.abs(np.fft.rfft(out.as_float32().samples[:, 0]))
    assert np.argmax(spectrum) == 440  # 1 s of audio: bin == Hz


def test_open_tts_returns_int16_audio():
    wav = Audio(np.array([0, 1000, -1000, 0], np.int16), 22050).to_wav()
    response = MagicMock(content=wav, headers={"Content-Type": "audio/wav"})
    with patch("rai_s2s.tts.models.open_tts.requests.get", return_value=response):
        audio = OpenTTS().get_speech("hi")
    assert audio.sample_rate == 22050
    assert audio.samples.ravel().tolist() == [0, 1000, -1000, 0]
