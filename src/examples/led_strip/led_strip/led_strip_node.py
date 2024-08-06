import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


def invert_bits(n):
    binary_rep = format(n, "08b")  # convert to 8-bit binary
    inverted_binary_rep = binary_rep[::-1]  # reverse the bits
    return int(inverted_binary_rep, 2)  # convert back to integer


color = (255, 255, 255)  # white
inverted_color = tuple(invert_bits(c) for c in color)

led_colors = np.full((1, 18, 3), inverted_color, dtype=np.uint8)

print("Original color:", color)
print("Inverted color:", inverted_color)
print("LED colors array:")
print(led_colors)


class LEDStripController(Node):

    def __init__(self):
        super().__init__("led_strip_controller")
        self.publisher_ = self.create_publisher(Image, "/led_strip", 10)

        self.state = "waiting"

        self.create_subscription(String, "/asr_status", self.listener_callback, 10)
        self.create_subscription(String, "/hmi_status", self.listener_callback, 10)
        self.create_subscription(String, "/tts_status", self.tts_listener_callback, 10)

        self.blink_state = False
        self.color = (255, 255, 255)  # white
        self.timer = self.create_timer(0.01, self.timer_callback)
        self.playing_state_value = 0
        self.sign = 1

    def listener_callback(self, msg):
        if msg.data in ["transcribing", "processing", "recording"]:
            self.state = msg.data

    def tts_listener_callback(self, msg):
        if msg.data == "playing":
            if not self.state == "playing":
                self.playing_state_value = 0
            self.state = msg.data
        elif msg.data == "waiting":
            if self.state == "playing":
                self.state = msg.data

    def timer_callback(self):
        if self.state == "playing":
            # publishing color/black
            self.blink_state = not self.blink_state
            color = (
                self.playing_state_value,
                0,
                self.playing_state_value,
            )  # if self.blink_state else (0, 0, 0)
            self.playing_state_value += 5 * self.sign
            if self.playing_state_value >= 255:
                self.sign = -1
            elif self.playing_state_value <= 0:
                self.sign = 1

            self.playing_state_value = max(0, (min(255, self.playing_state_value)))

        elif self.state in ["transcribing", "processing"]:
            color = (255, 134, 0)  # yellow
        elif self.state == "recording":
            color = (0, 255, 0)  # green
        else:
            color = (255, 255, 255)  # white

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
