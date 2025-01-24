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
import sounddevice as sd
from rai.communication import SoundDeviceError, StreamingAudioInputDevice


@pytest.fixture
def setup_mock_input_stream():
    with mock.patch("sounddevice.InputStream") as mock_input_stream:
        yield mock_input_stream


@pytest.fixture
def device_config():
    return {
        "block_size": 1024,
        "consumer_sampling_rate": 44100,
        "dtype": "float32",
    }


@pytest.mark.ci_only
def test_configure(
    setup_mock_input_stream,
    device_config,
):
    mock_input_stream = setup_mock_input_stream
    mock_instance = mock.MagicMock()
    mock_input_stream.return_value = mock_instance
    audio_input_device = StreamingAudioInputDevice()
    device = sd.query_devices(kind="input")
    if type(device) is dict:
        device_id = str(device["index"])
    elif isinstance(device, list):
        device_id = str(device[0]["index"])  # type: ignore
    else:
        raise AssertionError("No input device found")
    audio_input_device.configure_device(device_id, device_config)
    assert (
        audio_input_device.configred_devices[device_id].consumer_sampling_rate == 44100
    )
    assert audio_input_device.configred_devices[device_id].window_size_samples == 1024
    assert audio_input_device.configred_devices[device_id].dtype == "float32"


@pytest.mark.ci_only
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
        _ = audio_input_device.start_action(
            None, str(recording_device), feedback_callback, finish_callback
        )


@pytest.mark.ci_only
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

    device = sd.query_devices(kind="input")
    if type(device) is dict:
        device_id = str(device["index"])
    elif isinstance(device, list):
        device_id = str(device[0]["index"])  # type: ignore
    else:
        raise AssertionError("No input device found")
    audio_input_device.configure_device(device_id, device_config)

    stream_handle = audio_input_device.start_action(
        None, device_id, feedback_callback, finish_callback
    )

    assert mock_input_stream.call_count == 1
    init_args = mock_input_stream.call_args.kwargs
    assert init_args["device"] == int(device_id)
    assert init_args["finished_callback"] == finish_callback

    assert audio_input_device.streams.get(stream_handle) is not None
