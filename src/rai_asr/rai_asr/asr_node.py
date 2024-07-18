import tempfile
import threading
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
import rclpy
import sounddevice as sd
import torch
import whisper
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.node import Node
from scipy.io import wavfile
from std_msgs.msg import String

SAMPLING_RATE = 16000


class ASRNode(Node):
    def __init__(self):
        super().__init__("rai_asr")
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
            "base",
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
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=True
        )

        (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = (
            utils
        )

        self.sample_rate = SAMPLING_RATE
        self.is_recording = False
        self.audio_buffer = []
        self.vad_iterator = VADIterator(model, sampling_rate=self.sample_rate)
        self.silence_start_time = None
        self.transcription_publisher = self.create_publisher(  # type: ignore
            String, "/from_human", 10
        )
        self.status_publisher = self.create_publisher(  # type: ignore
            String, "~/status", 10
        )

        self.get_logger().info(  # type: ignore
            "Voice Activity Detection enabled. Waiting for speech..."
        )

        self.language = (
            self.get_parameter("language").get_parameter_value().string_value
        )  # type: ignore

        self.model_type = self.get_parameter("model").get_parameter_value().string_value  # type: ignore

        silence_grace_period = (
            self.get_parameter("silence_grace_period")
            .get_parameter_value()
            .double_value
        )  # type: ignore
        self.grace_period = timedelta(seconds=silence_grace_period)

        self.get_logger().info(
            f"Using model: {self.model_type}, language: {self.language}"
        )

    def capture_sound(self):
        window_size_samples = 512 if self.sample_rate == 16000 else 256
        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=window_size_samples,
        )
        stream.start()

        while True:
            audio_data, _ = stream.read(window_size_samples)
            audio_data = audio_data.flatten()

            speech_dict = self.vad_iterator(audio_data, return_seconds=True)
            if speech_dict:
                if not self.is_recording:
                    self.start_recording()
                self.audio_buffer.append(audio_data)
                self.silence_start_time = None
                if "end" in speech_dict.keys():
                    self.silence_start_time = datetime.now()
            elif self.is_recording:
                self.audio_buffer.append(audio_data)
                if self.silence_start_time is not None:
                    if datetime.now() - self.silence_start_time > self.grace_period:
                        self.stop_recording_and_transcribe()

    def start_recording(self):
        if not self.is_recording:
            self.get_logger().info("Started recording")
            self.publish_status("recording")
            self.is_recording = True
            self.audio_buffer = []
            self.silence_start_time = None

    def stop_recording_and_transcribe(self):
        if self.is_recording:
            self.get_logger().info("Stopped recording")
            self.is_recording = False
            self.publish_status("transcribing")
            self.transcribe_audio()

    def transcribe_audio(self):
        self.get_logger().info("Calling ASR model")
        combined_audio = np.concatenate(self.audio_buffer)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav_file:
            wavfile.write(temp_wav_file.name, self.sample_rate, combined_audio)
            self.get_logger().debug(f"Saved audio to {temp_wav_file.name}")

            model = whisper.load_model(self.model_type)
            response = model.transcribe(temp_wav_file.name, language=self.language)
            transcription = response["text"]
            self.get_logger().info(f"Transcription: {transcription}")
            self.publish_transcription(transcription)
        self.audio_buffer = []

    def publish_transcription(self, transcription):
        msg = String()
        msg.data = transcription
        self.transcription_publisher.publish(msg)

    def publish_status(self, status: Literal["recording", "transcribing"]):
        msg = String()
        msg.data = status
        self.status_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ASRNode()

    thread = threading.Thread(target=node.capture_sound)
    thread.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
