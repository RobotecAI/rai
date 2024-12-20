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

from unittest import mock

import pytest

from rai.communication import SoundDeviceError, StreamingAudioInputDevice


@pytest.fixture
def setup_mock_input_stream():
    with mock.patch("sounddevice.InputStream") as mock_input_stream:
        yield mock_input_stream


@pytest.fixture
def device_config():
    return {
        "kind": "input",
        "block_size": 1024,
        "sampling_rate": 44100,
        "target_smpling_rate": 16000,
        "dtype": "float32",
    }


def test_configure(
    setup_mock_input_stream,
    device_config,
):
    mock_input_stream = setup_mock_input_stream
    mock_instance = mock.MagicMock()
    mock_input_stream.return_value = mock_instance
    audio_input_device = StreamingAudioInputDevice()
    audio_input_device.configure_device("0", device_config)
    assert audio_input_device.configred_devices["0"].sample_rate == 44100
    assert audio_input_device.configred_devices["0"].window_size_samples == 1024
    assert audio_input_device.configred_devices["0"].target_samping_rate == 16000
    assert audio_input_device.configred_devices["0"].dtype == "float32"


def test_start_action_failed_init(
    setup_mock_input_stream,
):
    mock_input_stream = setup_mock_input_stream
    mock_instance = mock.MagicMock()
    mock_input_stream.return_value = mock_instance
    audio_input_device = StreamingAudioInputDevice()

    feedback_callback = mock.MagicMock()
    finish_callback = mock.MagicMock()

    recording_device = 0
    with pytest.raises(SoundDeviceError, match="Device 0 has not been configured"):
        stream_handle = audio_input_device.start_action(
            str(recording_device), feedback_callback, finish_callback
        )


def test_start_action(
    setup_mock_input_stream,
    device_config,
):
    mock_input_stream = setup_mock_input_stream
    mock_instance = mock.MagicMock()
    mock_input_stream.return_value = mock_instance
    audio_input_device = StreamingAudioInputDevice()

    feedback_callback = mock.MagicMock()
    finish_callback = mock.MagicMock()

    recording_device = "0"
    audio_input_device.configure_device(recording_device, device_config)

    stream_handle = audio_input_device.start_action(
        str(recording_device), feedback_callback, finish_callback
    )

    assert mock_input_stream.call_count == 1
    init_args = mock_input_stream.call_args.kwargs
    assert init_args["samplerate"] == 44100.0
    assert init_args["channels"] == 1
    assert init_args["device"] == int(recording_device)
    assert init_args["dtype"] == "float32"
    assert init_args["blocksize"] == 1024
    assert init_args["finished_callback"] == finish_callback

    assert audio_input_device.streams.get(stream_handle) is not None
