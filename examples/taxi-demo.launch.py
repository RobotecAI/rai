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

from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription(
        [
            # Launch the taxi demo Python script
            ExecuteProcess(cmd=["python3", "examples/taxi-demo.py"], output="screen"),
            # Include the voice launch from rai_bringup
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    [
                        PathJoinSubstitution(
                            [
                                FindPackageShare("rai_bringup"),
                                "launch",
                                "voice.launch.py",
                            ]
                        )
                    ]
                ),
                launch_arguments={
                    "tts_vendor": "opentts",  # elevenlabs (paid), opentts (free, local model)
                    "asr_vendor": "whisper",  # whisper (free, local model), openai (paid)
                    "recording_device": "0",  # find your recording device using python -c 'import sounddevice as sd; print(sd.query_devices())'
                    "keep_speaker_busy": "false",
                }.items(),
            ),
        ]
    )
