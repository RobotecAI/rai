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

from rai_s2s import KokoroTTS, OpenTTS, TextToSpeechAgent
from rai_s2s.sound_device import SoundDeviceConfig


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Text To Speech Configuration",
        allow_abbrev=True,
    )

    parser.add_argument(
        "--device-name",
        type=str,
        default="default",
        help="Speaker device name (default: 'default')",
    )

    parser.add_argument(
        "--tts-model",
        type=str,
        choices=["opentts", "kokoro"],
        default="kokoro",
        help="TTS model to use: 'opentts' or 'kokoro' (default: 'kokoro')",
    )

    # Use parse_known_args to ignore unknown arguments
    args, unknown = parser.parse_known_args()

    if unknown:
        print(f"Ignoring unknown arguments: {unknown}")

    return args


if __name__ == "__main__":
    rclpy.init()
    args = parse_arguments()

    config = SoundDeviceConfig(
        stream=True,
        is_output=True,
        # device_name="Sennheiser USB headset: Audio (hw:2,0)",
        # device_name="Jabra Speak2 40 MS: USB Audio (hw:2,0)",
        device_name=args.device_name,
    )

    tts = KokoroTTS()
    print("Using KokoroTTS model")
    if args.tts_model == "opentts":
        tts = OpenTTS()
        print("Using OpenTTS model")

    agent = TextToSpeechAgent(config, "text_to_speech", tts)
    agent.run()

    def cleanup(signum, frame):
        print("\nCustom handler: Caught SIGINT (Ctrl+C).")
        print("Performing cleanup")
        # Optionally exit the program
        agent.stop()
        rclpy.shutdown()
        exit(0)

    signal.signal(signal.SIGINT, cleanup)

    while True:
        time.sleep(1)
