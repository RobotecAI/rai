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
#

import io
import os
import threading
import time
from datetime import datetime, timedelta
from functools import partial
from typing import Literal, Optional, cast

import numpy as np
import rclpy
import sounddevice as sd
import torch
from numpy.typing import NDArray
from openai import OpenAI
from openwakeword.model import Model as OWWModel
from openwakeword.utils import download_models
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from scipy.io import wavfile
from std_msgs.msg import String

SAMPLING_RATE = 16000


class ASRNode(Node):
    def __init__(self):
        super().__init__("rai_asr")  # type: ignore
        self._declare_parameters()
        self._initialize_parameters()
        self._setup_node_components()
        self._setup_publishers_and_subscribers()

        self.asr_model = self._load_whisper_model()
        self.vad_model = self._initialize_vad_model()
        self.oww_model = self._initialize_open_wake_word()

        self.is_recording = False
        self.audio_buffer = []
        self.silence_start_time: Optional[datetime] = None
        self.last_transcription_time = 0
        self.hmi_lock = False
        self.tts_lock = False

        self.grace_period = timedelta(seconds=self.silence_grace_period)
        self.transcription_recording_timeout = 5
        self.get_logger().info("ASR Node has been initialized")  # type: ignore

    def _declare_parameters(self):
        self.declare_parameter(
            "use_wake_word",
            False,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description=("Whether to use wake word for starting conversation"),
            ),
        )
        self.declare_parameter(
            "wake_word_model",
            "",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description=("Wake word model onnx file"),
            ),
        )
        self.declare_parameter(
            "wake_word_threshold",
            0.1,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description=("Wake word threshold"),
            ),
        )
        self.declare_parameter(
            "vad_threshold",
            0.5,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description=("VAD threshold"),
            ),
        )
        self.declare_parameter(
            "recording_device",
            0,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description=(
                    "Recording device number. See available by running"
                    "python -c 'import sounddevice as sd; print(sd.query_devices())'"
                ),
            ),
        )
        self.declare_parameter(
            "language",
            "en",
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Language code for the ASR model",
            ),
        )
        self.declare_parameter(
            "model",
            "whisper-1",
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Model type for the ASR model",
            ),
        )
        self.declare_parameter(
            "silence_grace_period",
            1.0,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description="Grace period in seconds after silence to stop recording",
            ),
        )

    def _initialize_open_wake_word(self) -> Optional[OWWModel]:
        if self.use_wake_word:
            download_models()
            oww_model = OWWModel(
                wakeword_models=[
                    self.wake_word_model,
                ],
                inference_framework="onnx",
            )
            self.get_logger().info("Wake word model has been initialized")  # type: ignore
            return oww_model
        return None

    def _initialize_vad_model(self):
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
        )
        return model

    def _setup_node_components(self):
        self.callback_group = ReentrantCallbackGroup()
        self.sample_rate = SAMPLING_RATE

    def _initialize_parameters(self):
        self.silence_grace_period = cast(
            float,
            self.get_parameter("silence_grace_period")
            .get_parameter_value()
            .double_value,
        )
        self.whisper_model = cast(
            str,
            self.get_parameter("model").get_parameter_value().string_value,
        )
        self.language = cast(
            str,
            self.get_parameter("language").get_parameter_value().string_value,
        )
        self.vad_threshold = cast(
            float,
            self.get_parameter("vad_threshold").get_parameter_value().double_value,
        )

        self.use_wake_word = cast(
            bool,
            self.get_parameter("use_wake_word").get_parameter_value().bool_value,
        )
        self.wake_word_model = cast(
            str,
            self.get_parameter("wake_word_model").get_parameter_value().string_value,
        )
        self.wake_word_threshold = cast(
            float,
            self.get_parameter("wake_word_threshold")
            .get_parameter_value()
            .double_value,
        )
        self.recording_device_number = cast(
            int,
            self.get_parameter("recording_device").get_parameter_value().integer_value,
        )

        if self.use_wake_word:
            if not os.path.exists(self.wake_word_model):
                raise FileNotFoundError(f"Model file {self.wake_word_model} not found")

        self.get_logger().info("Parameters have been initialized")  # type: ignore

    def _setup_publishers_and_subscribers(self):
        self.transcription_publisher = self.create_publisher(String, "/from_human", 10)
        self.status_publisher = self.create_publisher(String, "/asr_status", 10)
        self.tts_status_subscriber = self.create_subscription(
            String,
            "/tts_status",
            self.tts_status_callback,
            10,
            callback_group=self.callback_group,
        )
        self.hmi_status_subscriber = self.create_subscription(
            String,
            "/hmi_status",
            self.hmi_status_callback,
            10,
            callback_group=self.callback_group,
        )

    def _load_whisper_model(self):
        self.openai_client = OpenAI()
        model = partial(
            self.openai_client.audio.transcriptions.create, model=self.whisper_model
        )
        return model

    def tts_status_callback(self, msg: String):
        if msg.data == "processing":
            self.tts_lock = True
        elif msg.data == "waiting":
            self.tts_lock = False

    def hmi_status_callback(self, msg: String):
        if msg.data == "processing":
            self.hmi_lock = True
        elif msg.data == "waiting":
            self.hmi_lock = False

    def should_listen(self, audio_data: NDArray[np.int16]) -> bool:
        def int2float(sound: NDArray[np.int16]):
            abs_max = np.abs(sound).max()
            sound = sound.astype("float32")
            if abs_max > 0:
                sound *= 1 / 32768
            sound = sound.squeeze()
            return sound

        vad_confidence = self.vad_model(
            torch.tensor(int2float(audio_data[-512:])), self.sample_rate
        ).item()

        if self.oww_model:
            if self.is_recording:
                self.get_logger().info(f"VAD confidence: {vad_confidence}")  # type: ignore
                return vad_confidence > self.vad_threshold
            else:
                predictions = self.oww_model.predict(audio_data)
                for key, value in predictions.items():
                    if value > self.wake_word_threshold:
                        self.get_logger().debug(f"Detected wake word: {key}")  # type: ignore
                        self.oww_model.reset()
                        return True
        else:
            return vad_confidence > self.vad_threshold

        return False

    def capture_sound(self):
        window_size_samples = 1280
        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            device=self.recording_device_number,
            dtype="int16",
            blocksize=window_size_samples,
        )
        stream.start()
        self.get_logger().info("Stream started. Waiting for speech...")  # type: ignore
        while True:
            audio_data, _ = stream.read(window_size_samples)
            audio_data = audio_data.flatten()

            asr_lock = (
                time.time()
                < self.last_transcription_time + self.transcription_recording_timeout
            )
            if asr_lock or self.hmi_lock or self.tts_lock:
                continue
                

            if self.should_listen(audio_data):
                self.silence_start_time = datetime.now()
                if not self.is_recording:
                    self.start_recording()
                    self.reset_buffer()
                    self.audio_buffer.append(audio_data)
            elif self.is_recording:
                self.audio_buffer.append(audio_data)
                if datetime.now() - self.silence_start_time > timedelta(seconds=self.silence_grace_period):
                    self.stop_recording_and_transcribe()
                    self.reset_buffer()

    def reset_buffer(self):
        self.audio_buffer = []

    def start_recording(self):
        self.get_logger().info("Recording...")  # type: ignore
        self.publish_status("recording")
        self.is_recording = True

    def stop_recording_and_transcribe(self):
        self.get_logger().info("Stopped recording. Transcribing...")  # type: ignore
        self.is_recording = False
        self.publish_status("transcribing")
        self.transcribe_audio()
        self.publish_status("waiting")
        self.get_logger().info("Done transcribing.")  # type: ignore

    def transcribe_audio(self):
        combined_audio = np.concatenate(self.audio_buffer)

        with io.BytesIO() as temp_wav_buffer:
            wavfile.write(temp_wav_buffer, self.sample_rate, combined_audio)
            temp_wav_buffer.seek(0)
            temp_wav_buffer.name = "temp.wav"

            response = self.asr_model(file=temp_wav_buffer, language=self.language)
            transcription = response.text
            self.get_logger().debug(f"Transcription: {transcription}")  # type: ignore
            self.publish_transcription(transcription)

        self.last_transcription_time = time.time()

        self.audio_buffer = []

    def publish_transcription(self, transcription: str):
        msg = String()
        msg.data = transcription
        self.transcription_publisher.publish(msg)

    def publish_status(
        self, status: Literal["recording", "transcribing", "dropping", "waiting"]
    ):
        msg = String()
        msg.data = status
        self.status_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ASRNode()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    thread = threading.Thread(target=node.capture_sound)
    thread.start()

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
