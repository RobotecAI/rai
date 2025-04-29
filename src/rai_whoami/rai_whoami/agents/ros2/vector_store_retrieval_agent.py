# Copyright (C) 2025 Robotec.AI
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

import importlib.util

if importlib.util.find_spec("rclpy") is None:
    raise ImportError(
        "This is a ROS2 feature. Make sure ROS2 is installed and sourced."
    )

import logging
from pathlib import Path
from typing import Annotated

from langchain_core.embeddings import Embeddings
from rai.agents import BaseAgent
from rai.communication.ros2 import ROS2Connector

from rai_interfaces.srv import VectorStoreRetrieval
from rai_whoami.vector_db.faiss import get_faiss_client


class ROS2VectorStoreRetrievalAgent(BaseAgent):
    def __init__(
        self,
        service_name: str,
        root_dir: str,
        embeddings_model: Embeddings | None = None,
        k: Annotated[int, "The number of results to return"] = 4,
    ):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.connector = ROS2Connector()
        self.service_name = service_name
        self.root_dir = root_dir
        self.embeddings_model = embeddings_model
        self.k = k
        self.vdb_client = get_faiss_client(
            str(Path(self.root_dir) / "generated"), self.embeddings_model
        )

        self.connector.create_service(
            service_name=service_name,
            service_type="rai_interfaces/srv/VectorStoreRetrieval",
            on_request=self.service_callback,
        )
        self.logger.info(
            f"Vector store retrieval agent initialized with service name {service_name}"
        )

    def service_callback(
        self,
        request: VectorStoreRetrieval.Request,
        response: VectorStoreRetrieval.Response,
    ):
        self.logger.info(f"Received request: {request.query}")
        vdb_results = self.vdb_client.similarity_search_with_score(
            request.query, k=self.k
        )
        response.success = True
        response.message = "Success"
        response.documents = [result[0].page_content for result in vdb_results]
        response.scores = [result[1] for result in vdb_results]
        self.logger.info(f"Sending response to query: {request.query}")
        return response

    def run(self):
        pass

    def stop(self):
        self.connector.shutdown()
