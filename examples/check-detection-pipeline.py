# Copyright (C) 2025 Julia Jia
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

"""Diagnostic script to check if the detection pipeline is working."""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from rai_interfaces.msg import RAIDetectionArray


class DetectionPipelineChecker(Node):
    """Check if detection pipeline components are working."""

    def __init__(self):
        super().__init__("detection_pipeline_checker")
        self.camera_received = False
        self.detections_received = False
        self.detection_count = 0

    def check_camera_topic(
        self, topic: str = "/camera/image_raw", timeout: float = 5.0
    ):
        """Check if camera topic is publishing."""
        self.get_logger().info(f"Checking camera topic: {topic}")
        _ = self.create_subscription(
            Image, topic, lambda msg: setattr(self, "camera_received", True), 10
        )

        import time

        start_time = time.time()
        while time.time() - start_time < timeout:
            rclpy.spin_once(self, timeout_sec=0.5)
            if self.camera_received:
                self.get_logger().info(f"✓ Camera topic {topic} is publishing")
                return True

        self.get_logger().warn(
            f"✗ Camera topic {topic} not publishing (timeout: {timeout}s)"
        )
        return False

    def check_detection_topic(
        self, topic: str = "/detection_array", timeout: float = 10.0
    ):
        """Check if detection topic is publishing."""
        self.get_logger().info(f"Checking detection topic: {topic}")

        def detection_callback(msg: RAIDetectionArray):
            self.detections_received = True
            self.detection_count += len(msg.detections)
            self.get_logger().info(
                f"Received detection array with {len(msg.detections)} detections: "
                f"{msg.detection_classes}"
            )

        _ = self.create_subscription(RAIDetectionArray, topic, detection_callback, 10)

        import time

        start_time = time.time()
        while time.time() - start_time < timeout:
            rclpy.spin_once(self, timeout_sec=0.5)
            if self.detections_received:
                self.get_logger().info(
                    f"✓ Detection topic {topic} is publishing ({self.detection_count} total detections)"
                )
                return True

        self.get_logger().warn(
            f"✗ Detection topic {topic} not publishing (timeout: {timeout}s)"
        )
        return False

    def check_dino_service(
        self, service_name: str = "/grounding_dino/grounding_dino_classify"
    ):
        """Check if DINO service is available."""
        from rai_interfaces.srv import RAIGroundingDino

        client = self.create_client(RAIGroundingDino, service_name)

        self.get_logger().info(f"Checking DINO service: {service_name}")
        if client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info(f"✓ DINO service {service_name} is available")
            return True
        else:
            self.get_logger().warn(f"✗ DINO service {service_name} not available")
            return False


def main():
    """Run diagnostic checks."""
    rclpy.init()
    checker = DetectionPipelineChecker()

    print("\n" + "=" * 60)
    print("Detection Pipeline Diagnostic")
    print("=" * 60 + "\n")

    # Check DINO service
    dino_ok = checker.check_dino_service()
    print()

    # Check camera topic
    camera_ok = checker.check_camera_topic()
    print()

    # Check detection topic (wait longer since it depends on camera)
    if camera_ok:
        detection_ok = checker.check_detection_topic(timeout=15.0)
    else:
        print("Skipping detection topic check (camera not available)")
        detection_ok = False
    print()

    # Summary
    print("=" * 60)
    print("Summary:")
    print(f"  DINO Service: {'✓' if dino_ok else '✗'}")
    print(f"  Camera Topic: {'✓' if camera_ok else '✗'}")
    print(f"  Detection Topic: {'✓' if detection_ok else '✗'}")
    print("=" * 60)

    if not dino_ok:
        print(
            "\n⚠️  DINO service not available. Make sure perception agents are running:"
        )
        print("   python -m rai_perception.scripts.run_perception_agents")

    if not camera_ok:
        print("\n⚠️  Camera topic not publishing. Check:")
        print("   ros2 topic list | grep camera")
        print("   ros2 topic echo /camera/image_raw --once")

    if not detection_ok and camera_ok:
        print("\n⚠️  Detection topic not publishing. Check:")
        print("   - Is detection_publisher node running?")
        print("   - Check detection_publisher logs for errors")
        print("   - Verify camera topic name matches detection_publisher config")

    checker.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
