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

import argparse
import signal
import time

import rclpy

from rai_s2s.asr import LocalWhisper, OpenWakeWord, SileroVAD, SpeechRecognitionAgent
from rai_s2s.sound_device import SoundDeviceConfig

VAD_THRESHOLD = 0.8  # Note that this might be different depending on your device
OWW_THRESHOLD = 0.1  # Note that this might be different depending on your device

VAD_SAMPLING_RATE = 16000  # Or 8000
DEFAULT_BLOCKSIZE = 1280


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Voice Activity Detection and Wake Word Detection Configuration",
        allow_abbrev=True,
    )

    # Predefined arguments
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=VAD_THRESHOLD,
        help="Voice Activity Detection threshold (default: 0.5)",
    )
    parser.add_argument(
        "--oww-threshold",
        type=float,
        default=OWW_THRESHOLD,
        help="OpenWakeWord threshold (default: 0.1)",
    )
    parser.add_argument(
        "--vad-sampling-rate",
        type=int,
        choices=[8000, 16000],
        default=VAD_SAMPLING_RATE,
        help="VAD sampling rate (default: 16000)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCKSIZE,
        help="Audio block size (default: 1280)",
    )
    parser.add_argument(
        "--device-name",
        type=str,
        default="default",
        help="Microphone device name (default: 'default')",
    )

    # Use parse_known_args to ignore unknown arguments
    args, unknown = parser.parse_known_args()

    if unknown:
        print(f"Ignoring unknown arguments: {unknown}")

    return args


if __name__ == "__main__":
    args = parse_arguments()

    microphone_configuration = SoundDeviceConfig(
        stream=True,
        channels=1,
        device_name=args.device_name,
        block_size=args.block_size,
        consumer_sampling_rate=args.vad_sampling_rate,
        dtype="int16",
        device_number=None,
        is_input=True,
        is_output=False,
    )
    vad = SileroVAD(args.vad_sampling_rate, args.vad_threshold)
    oww = OpenWakeWord("hey jarvis", args.oww_threshold)
    whisper = LocalWhisper("tiny", args.vad_sampling_rate)
    # you can easily switch the the provider by changing the whisper object
    # whisper = OpenAIWhisper("whisper-1", args.vad_sampling_rate, "en")

    rclpy.init()
    ros2_name = "rai_asr_agent"

    agent = SpeechRecognitionAgent(microphone_configuration, ros2_name, whisper, vad)
    # optionally add additional models to decide when to record data for transcription
    # agent.add_detection_model(oww, pipeline="record")

    agent.run()

    def cleanup(signum, frame):
        agent.stop()
        rclpy.shutdown()
        exit(0)

    signal.signal(signal.SIGINT, cleanup)

    while True:
        time.sleep(1)
