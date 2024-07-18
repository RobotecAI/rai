import re
import subprocess
import threading
import time
from queue import PriorityQueue
from typing import NamedTuple, cast

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from .tts_clients import ElevenLabsClient, OpenTTSClient, TTSClient


class TTSJob(NamedTuple):
    id: int
    file_path: str


class TTSNode(Node):
    def __init__(self):
        super().__init__("rai_tts_node")

        self.declare_parameter("tts_client", "opentts")
        self.declare_parameter("voice", "larynx:blizzard_lessac-glow_tts")
        self.declare_parameter("base_url", "http://localhost:5500/api/tts")
        self.declare_parameter("topic", "to_human")

        topic_param = self.get_parameter("topic").get_parameter_value().string_value  # type: ignore

        self.subscription = self.create_subscription(  # type: ignore
            String, topic_param, self.listener_callback, 10  # type: ignore
        )
        self.queue: PriorityQueue[TTSJob] = PriorityQueue()
        self.it: int = 0
        self.job_id: int = 0
        self.tts_client = self._initialize_client()
        threading.Thread(target=self._process_queue).start()
        self.get_logger().info("TTS Node has been started")  # type: ignore

    def listener_callback(self, msg: String):
        self.get_logger().info(  # type: ignore
            f"Registering new TTS job: {self.job_id} length: {len(msg.data)} chars."  # type: ignore
        )
        threading.Thread(
            target=self.synthesize_speech, args=(self.job_id, msg.data)  # type: ignore
        ).start()
        self.job_id += 1

    def synthesize_speech(
        self,
        id: int,
        text: str,
    ) -> str:
        text = self._preprocess_text(text)
        if id > 0:
            time.sleep(0.5)
        temp_file_path = self.tts_client.synthesize_speech_to_file(text)
        self.get_logger().info(f"Job {id} completed.")  # type: ignore
        tts_job = TTSJob(id, temp_file_path)
        self.queue.put(tts_job)

        return temp_file_path

    def _process_queue(self):
        while rclpy.ok():
            time.sleep(0.01)
            if not self.queue.empty():
                if self.queue.queue[0][0] == self.it:
                    self.it += 1
                    tts_job = self.queue.get()
                    self.get_logger().info(  # type: ignore
                        f"Playing audio for job {tts_job.id}. {tts_job.file_path}"
                    )
                    self._play_audio(tts_job.file_path)

    def _play_audio(self, filepath: str):
        subprocess.run(
            ["ffplay", "-v", "0", "-nodisp", "-autoexit", filepath],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.get_logger().debug(f"Playing audio: {filepath}")  # type: ignore

    def _initialize_client(self) -> TTSClient:
        tts_client_param = cast(str, self.get_parameter("tts_client").get_parameter_value().string_value)  # type: ignore
        voice_param = cast(str, self.get_parameter("voice").get_parameter_value().string_value)  # type: ignore
        base_url_param = cast(str, self.get_parameter("base_url").get_parameter_value().string_value)  # type: ignore

        if tts_client_param == "opentts":
            return OpenTTSClient(
                base_url=base_url_param,
                voice=voice_param,
            )
        elif tts_client_param == "elevenlabs":
            return ElevenLabsClient(
                voice=voice_param,
                base_url=base_url_param,
            )
        else:
            raise ValueError(f"Unknown TTS client: {tts_client_param}")

    def _preprocess_text(self, text: str) -> str:
        """Remove emojis from text."""
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "]+",
            flags=re.UNICODE,
        )
        text = emoji_pattern.sub(r"", text)
        return text


def main():
    rclpy.init()

    tts_node = TTSNode()

    rclpy.spin(tts_node)

    tts_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
