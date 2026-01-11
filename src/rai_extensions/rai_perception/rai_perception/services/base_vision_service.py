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

"""Base class for ROS2 vision services.

It reads the model name from ROS2 parameters and uses the vision model registry to dynamically load the appropriate vision algorithm.

Note: This class is named "service" (not "agent") to avoid confusion with the RAI agent abstraction (rai.agents.BaseAgent).
"""

from pathlib import Path
from typing import Optional

from rai.communication.ros2 import ROS2Connector

from .weights import download_weights, load_model_with_error_handling


class BaseVisionService:
    """Base class for vision services that load models and provide ROS2 services.

    It handles model loading, weight management, and ROS2 service setup.
    """

    WEIGHTS_URL: str = ""
    DEFAULT_WEIGHTS_ROOT_PATH: Path = Path.home() / Path(".cache/rai/")
    WEIGHTS_DIR_PATH_PART: Path = Path("vision/weights")
    WEIGHTS_FILENAME: str = ""

    def __init__(
        self,
        weights_root_path: str | Path = DEFAULT_WEIGHTS_ROOT_PATH,
        ros2_name: str = "",
        ros2_connector: Optional[ROS2Connector] = None,
    ):
        # TODO: After agents are deprecated, make ros2_connector a required parameter
        # (remove Optional and default None). Services should always receive a connector.
        if not self.WEIGHTS_FILENAME:
            raise ValueError("WEIGHTS_FILENAME is not set")
        self.weights_root_path = Path(weights_root_path)
        self.weights_path = (
            self.weights_root_path / self.WEIGHTS_DIR_PATH_PART / self.WEIGHTS_FILENAME
        )
        self.weights_path.parent.mkdir(parents=True, exist_ok=True)
        self.ros2_connector = ros2_connector or ROS2Connector(
            ros2_name, executor_type="single_threaded"
        )
        self.logger = self.ros2_connector.node.get_logger()

        if not self.weights_path.exists():
            download_weights(self.weights_path, self.logger, self.WEIGHTS_URL)

    def _load_model_with_error_handling(
        self, model_class, config_path: str | Path | None = None
    ):
        """Load model with automatic error handling for corrupted weights.

        Args:
            model_class: A class that can be instantiated with weights_path and optionally config_path
            config_path: Optional path to config file

        Returns:
            The loaded model instance
        """
        return load_model_with_error_handling(
            model_class, self.weights_path, self.logger, self.WEIGHTS_URL, config_path
        )

    def _initialize_model_from_registry(
        self, get_model_func, default_model_name: str, model_type_name: str
    ):
        """Initialize model from registry based on ROS2 parameter.

        Args:
            get_model_func: Function that takes model name and returns (AlgorithmClass, config_path)
            default_model_name: Default model name if ROS2 parameter not set
            model_type_name: Type name for logging (e.g., "detection", "segmentation")

        Returns:
            Tuple of (model_instance, model_name)
        """
        from rai.communication.ros2 import get_param_value

        model_name = get_param_value(
            self.ros2_connector.node, "model_name", default=default_model_name
        )
        AlgorithmClass, config_path = get_model_func(model_name)
        self.logger.info(
            f"Loading {model_type_name} model '{model_name}' (config: {config_path})"
        )
        model_instance = self._load_model_with_error_handling(
            AlgorithmClass, config_path
        )
        self.logger.info(
            f"{self.__class__.__name__} initialized with model '{model_name}'"
        )
        return model_instance, model_name

    def _get_service_name(self, default_service_name: str) -> str:
        """Get service name from ROS2 parameter or use default.

        Args:
            default_service_name: Default service name if parameter not set

        Returns:
            Service name string
        """
        from rai.communication.ros2 import get_param_value

        return get_param_value(
            self.ros2_connector.node, "service_name", default=default_service_name
        )

    def _create_service(
        self, service_name: str, callback, service_type: str, service_type_name: str
    ):
        """Create ROS2 service and log startup.

        Args:
            service_name: ROS2 service name
            callback: Service callback function
            service_type: ROS2 service type string
            service_type_name: Human-readable service type name for logging
        """
        self.ros2_connector.create_service(
            service_name, callback, service_type=service_type
        )
        self.logger.info(f"{service_type_name} service started at '{service_name}'")

    def stop(self):
        self.ros2_connector.shutdown()
