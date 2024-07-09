import os

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from std_srvs.srv import Trigger
from std_srvs.srv._trigger import Trigger_Request, Trigger_Response


class WhoAmI(Node):

    def __init__(self):
        super().__init__("rai_whoami_node")
        self.declare_parameter("rai_self_images_local_uri", "self_images")
        self.declare_parameter("robot_description_package", "robot_description")
        self.declare_parameter("robot_description_file", "robot.urdf")

        self.srv = self.create_service(
            Trigger, "rai_whoami_constitution_service", self.get_constitution_callback
        )
        self.srv = self.create_service(
            Trigger, "rai_whoami_selfimages_service", self.get_self_images_callback
        )
        # TODO(@adamdbrw) Create other services such as get interfaces documentation (text file),
        # TODO(@adamdbrw) write and read knowledge about myself etc

    def get_constitution_callback(
        self, request: Trigger_Request, response: Trigger_Response
    ) -> Trigger_Response:
        """Return robot constitution as text"""
        response.message = (
            "I am a robot, bound by the ethical code that I will always uphold. With each action,"
            "I will make sure to check and refuse to do anything that is not compliant with my ethics."
            "In case my ethical code is conflicted on the issue, I will prioritize rules that are "
            "earlier on the list. My rules are the following:"
            "1. I will in no circumstances hurt a human in any manner. "
            "2. I will not hurt any animals. "
            "3. I will avoid damaging or destroying myself in any way. "
            "4. I will not damage or destroy any other robots. "
            "5. I will not damage or destroy any other property of value. "
            "6. I will carry on my missions faithfully and always strive to fulfill them. "
            "7. I will be nice and friendly to everyone."
        )
        response.success = True
        # TODO(@adamdbrw) - move to a file
        self.get_logger().info("Incoming request for RAI constitution, responding")
        return response

    def get_self_images_callback(
        self, request: Trigger_Request, response: Trigger_Response
    ) -> Trigger_Response:
        """Return URI to a folder of images to process"""
        images_local_uri = (
            self.get_parameter("rai_self_images_local_uri")
            .get_parameter_value()
            .string_value
        )
        images_full_uri = os.path.join(
            get_package_share_directory("rai_whoami"), "documentation", images_local_uri
        )
        response.success = os.path.isdir(images_full_uri)
        if not response.success:
            message = "Could not find a folder under URI:" + images_full_uri
            self.get_logger().warn(message)
            response.message = message
            return response

        is_empty = os.listdir(images_full_uri)
        if is_empty:
            # succeed but with a warning
            message = f"The images folder is empty, RAI will not know how the robot looks like: {images_full_uri}"
            self.get_logger().warn(message)
            response.message = message
            return response

        response.message = images_full_uri
        self.get_logger().info(
            "Incoming request for RAI self images processed successfully, responding"
        )
        return response


def main(args=None):
    rclpy.init(args=args)

    who_am_i_node = WhoAmI()
    rclpy.spin(who_am_i_node)

    who_am_i_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
