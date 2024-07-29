import os

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_srvs.srv import Trigger
from std_srvs.srv._trigger import Trigger_Request, Trigger_Response


class WhoAmI(Node):

    def __init__(self):
        super().__init__("rai_whoami_node")
        self.declare_parameter("robot_description_package", Parameter.Type.STRING)

        self.srv = self.create_service(
            Trigger, "rai_whoami_constitution_service", self.get_constitution_callback
        )
        self.srv = self.create_service(
            Trigger, "rai_whoami_selfimages_service", self.get_self_images_callback
        )

        self.robot_constitution_path = os.path.join(
            get_package_share_directory("rai_whoami"),
            "description/robot_constitution.txt",
        )

        with open(self.robot_constitution_path, "r") as file:
            self.robot_constitution = file.read()
            self.get_logger().info(
                f"Robot constitution loaded from {self.robot_constitution_path}"
            )

        # TODO(@adamdbrw) Create other services such as get interfaces documentation (text file),
        # TODO(@adamdbrw) write and read knowledge about myself etc

    def get_constitution_callback(
        self, request: Trigger_Request, response: Trigger_Response
    ) -> Trigger_Response:
        """Return robot constitution as text"""
        response.message = self.robot_constitution
        response.success = True
        self.get_logger().info("Incoming request for RAI constitution, responding")
        return response

    def get_self_images_callback(
        self, request: Trigger_Request, response: Trigger_Response
    ) -> Trigger_Response:
        """Return URI to a folder of images to process"""
        images_local_uri = "description/images"
        images_full_uri = os.path.join(
            get_package_share_directory("rai_whoami"), images_local_uri
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
