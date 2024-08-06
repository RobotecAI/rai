import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


# husarion's low level driver expects big endian encoding
def reverse_bit_order(n: int) -> int:
    binary_rep = format(n, "08b")  # convert to 8-bit binary
    inverted_binary_rep = binary_rep[::-1]  # reverse the bits
    return int(inverted_binary_rep, 2)  # convert back to integer


STATE_TO_COLOR = {
    "waiting": (255, 255, 255),  # white
    "processing": (255, 134, 0),  # yellow
    "recording": (0, 255, 0),  # green
    "playing": (0, 0, 255),  # blue
}
DEFAULT_COLOR = (255, 0, 0)  # red, unknown state
PULSE_FREQUENCY = 3  # Hz


class LEDStripController(Node):

    def __init__(self):
        super().__init__("led_strip_controller")
        self.asr_state = ""
        self.hmi_state = ""
        self.tts_state = ""

        self.create_subscription(String, "/asr_status", self.asr_status_callback, 10)
        self.create_subscription(String, "/hmi_status", self.hmi_status_callback, 10)
        self.create_subscription(String, "/tts_status", self.tts_status_callback, 10)

        self.timer = self.create_timer(0.05, self.timer_callback)

        self.publisher_ = self.create_publisher(Image, "/led_strip", 10)

    def asr_status_callback(self, msg: String) -> None:
        if isinstance(msg.data, str):
            self.asr_state = msg.data

    def hmi_status_callback(self, msg: String) -> None:
        if isinstance(msg.data, str):
            self.hmi_state = msg.data

    def tts_status_callback(self, msg: String) -> None:
        if isinstance(msg.data, str):
            self.tts_state = msg.data

    def calculate_state(self) -> str:
        # priority order: recording > playing > processing > waiting
        if self.asr_state == "recording" and self.tts_state == "playing":
            self.get_logger().warn(
                "ASR is recording and TTS is playing at the same time!"
            )
            return ""

        if self.asr_state == "recording":
            return "recording"
        if self.tts_state == "playing":
            return "playing"
        if self.hmi_state == "processing":
            return "processing"
        if self.asr_state == "waiting" or self.hmi_state == "waiting":
            return "waiting"
        return ""

    def timer_callback(self):
        state = self.calculate_state()
        color = STATE_TO_COLOR.get(state, DEFAULT_COLOR)

        if state == "playing":
            t = self.get_clock().now().nanoseconds / 1e9
            value: float = np.sin(2 * np.pi * PULSE_FREQUENCY * t)
            color = np.array(color)
            color = (value * color).astype(np.uint8)

        color = list(map(reverse_bit_order, color))
        led_colors = np.full((1, 18, 3), color, dtype=np.uint8)

        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.height = 1
        msg.width = 18
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        msg.step = 18 * 3
        msg.data = led_colors.flatten().tolist()

        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LEDStripController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
