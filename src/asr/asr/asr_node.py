import tempfile
import threading
from datetime import datetime, timedelta

import numpy as np
import rclpy
import sounddevice as sd
import torch
import whisper
from rclpy.node import Node
from scipy.io import wavfile

SAMPLING_RATE = 16000

# Download and load the model
model, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=True
)

(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils


class ASRNode(Node):
    def __init__(self):
        super().__init__("sound_capture_node")
        self.sample_rate = SAMPLING_RATE
        self.is_recording = False
        self.audio_buffer = []
        self.thread = None
        self.vad_iterator = VADIterator(model, sampling_rate=self.sample_rate)
        self.silence_start_time = None
        self.grace_period = timedelta(seconds=2)  # 1 second grace period for silence
        self.get_logger().info(
            "Voice Activity Detection enabled. Waiting for speech..."
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
            # print("Silence start time: ", self.silence_start_time)
            audio_data, _ = stream.read(window_size_samples)
            audio_data = audio_data.flatten()

            speech_dict = self.vad_iterator(audio_data, return_seconds=True)
            # print("Speech dict: ", speech_dict)
            if speech_dict:
                if not self.is_recording:
                    self.start_recording()
                self.audio_buffer.append(audio_data)
                self.silence_start_time = None
                if "end" in speech_dict.keys():
                    self.silence_start_time = datetime.now()
            elif self.is_recording:
                self.audio_buffer.append(audio_data)
                # if speech_dict is not None:
                #     if 'end' in speech_dict.keys():
                #         print("END Speech dict: ", speech_dict)
                #         self.silence_start_time = datetime.now()
                if self.silence_start_time is not None:
                    if datetime.now() - self.silence_start_time > self.grace_period:
                        self.stop_recording_and_transcribe()

    def start_recording(self):
        if not self.is_recording:
            self.get_logger().info("Started recording")
            self.is_recording = True
            self.audio_buffer = []
            self.silence_start_time = None

    def stop_recording_and_transcribe(self):
        if self.is_recording:
            self.get_logger().info("Stopped recording")
            self.is_recording = False
            self.transcribe_audio()

    def transcribe_audio(self):
        combined_audio = np.concatenate(self.audio_buffer)
        # print("Combined audio shape:", combined_audio.shape)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav_file:
            wavfile.write(temp_wav_file.name, self.sample_rate, combined_audio)
            self.get_logger().info(f"Saved audio to {temp_wav_file.name}")

            model = whisper.load_model("base")
            response = model.transcribe(temp_wav_file.name, language="en")
            self.get_logger().info(f'Transcription: {response["text"]}')
        self.audio_buffer = []


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
