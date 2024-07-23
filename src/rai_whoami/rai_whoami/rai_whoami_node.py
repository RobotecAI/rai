import os

import rclpy
from ament_index_python.packages import get_package_share_directory
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from rai_interfaces.srv._vector_store_retrieval import (
    VectorStoreRetrieval_Request,
    VectorStoreRetrieval_Response,
)
from rclpy.node import Node
from std_srvs.srv import Trigger
from std_srvs.srv._trigger import Trigger_Request, Trigger_Response

from rai_interfaces.srv import VectorStoreRetrieval


class WhoAmI(Node):

    def __init__(self):
        super().__init__("rai_whoami_node")
        self.declare_parameter("rai_self_images_local_uri")
        self.declare_parameter("robot_description_package")
        self.declare_parameter("robot_description_file")
        self.declare_parameter(
            "robot_constitution_path", "resource/default_robot_constitution.txt"
        )

        self.srv = self.create_service(
            Trigger, "rai_whoami_constitution_service", self.get_constitution_callback
        )
        self.srv = self.create_service(
            Trigger, "rai_whoami_selfimages_service", self.get_self_images_callback
        )
        self.srv = self.create_service(
            VectorStoreRetrieval,
            "rai_whoami_documentation_service",
            self.get_documentation_callback,
        )
        self.srv = self.create_service(
            Trigger,
            "rai_whoami_identity_service",
            self.get_identity_callback,
        )
        self.robot_constitution_path = (
            self.get_parameter("robot_constitution_path")
            .get_parameter_value()
            .string_value
        )

        with open(self.robot_constitution_path, "r") as file:
            self.robot_constitution = file.read()
            self.get_logger().info(
                f"Robot constitution loaded from {self.robot_constitution_path}"
            )

        # TODO(@adamdbrw) Create other services such as get interfaces documentation (text file),
        # TODO(@adamdbrw) write and read knowledge about myself etc
        self.faiss_index = self._load_documentation()

    def _load_documentation(self) -> FAISS:
        faiss_index = FAISS.load_local(
            get_package_share_directory("rai_whoami"),
            OpenAIEmbeddings(),
            allow_dangerous_deserialization=True,
        )
        return faiss_index

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

    def get_documentation_callback(
        self,
        request: VectorStoreRetrieval_Request,
        response: VectorStoreRetrieval_Response,
    ) -> VectorStoreRetrieval_Response:
        """Return documentation based on the query string"""
        query = request.query
        self.get_logger().warn(f"Query: {query}")

        if query:
            self.get_logger().info(f"Querying for documentation: {query}")
            output = self.faiss_index.similarity_search_with_score(query)
            response.message = "Query successful"
            response.success = True
            response.documents = [doc.page_content for doc, _ in output]
            response.scores = [float(score) for _, score in output]
            self.get_logger().info(f"Query successful, found {len(output)} documents")
        else:
            response.message = "No query provided"
            response.success = False

        self.get_logger().info(f"Incoming request for documentation: {query}")
        return response

    def get_identity_callback(
        self,
        request: Trigger_Request,
        response: Trigger_Response,
    ) -> Trigger_Response:
        """Return robot identity"""
        identity_path = get_package_share_directory("rai_whoami") + "/identity.txt"
        with open(identity_path, "r") as f:
            identity = f.read()
        response.success = True
        response.message = identity
        self.get_logger().info("Incoming request for RAI identity, responding")
        return response


def main(args=None):
    rclpy.init(args=args)

    who_am_i_node = WhoAmI()
    rclpy.spin(who_am_i_node)

    who_am_i_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
