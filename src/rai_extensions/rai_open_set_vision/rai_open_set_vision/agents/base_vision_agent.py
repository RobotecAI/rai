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


import os
import shutil
import subprocess
from pathlib import Path

from rai.agents import BaseAgent
from rai.communication.ros2 import ROS2Connector


class BaseVisionAgent(BaseAgent):
    WEIGHTS_URL: str = ""
    WEIGHTS_FILENAME: str = ""

    def __init__(
        self,
        weights_path: str | Path = Path.home() / Path(".cache/rai/"),
        ros2_name: str = "",
    ):
        super().__init__()
        self._weights_path = Path(weights_path)
        os.makedirs(self._weights_path, exist_ok=True)
        self._is_weights_path_set = False
        self._init_weight_path()
        self.weight_path = self._weights_path
        self.ros2_connector = ROS2Connector(ros2_name, executor_type="single_threaded")

    def _init_weight_path(self):
        try:
            if self.WEIGHTS_FILENAME == "":
                raise ValueError("WEIGHTS_FILENAME is not set")

            # Ensure that the self._weights_path variable is set only once
            # to prevent issues during weight re-downloading
            if not self._is_weights_path_set:
                install_path = (
                    self._weights_path / "vision" / "weights" / self.WEIGHTS_FILENAME
                )
            else:
                install_path = self._weights_path
            self._is_weights_path_set = True

            # make sure the file exists
            if install_path.exists() and install_path.is_file():
                self._weights_path = install_path
            else:
                self._remove_weights(path=install_path)
                self._download_weights(install_path)
                self._weights_path = install_path

        except Exception:
            self.logger.error("Could not find package path")
            raise Exception("Could not find package path")

    def _load_model_with_error_handling(self, model_class):
        """Load model with automatic error handling for corrupted weights.

        Args:
            model_class: A class that can be instantiated with weights_path

        Returns:
            The loaded model instance
        """
        try:
            return model_class(self._weights_path)
        except RuntimeError as e:
            self.logger.error(f"Could not load model: {e}")
            if "PytorchStreamReader" in str(e):
                self.logger.error("The weights might be corrupted. Redownloading...")
                self._remove_weights(str(self._weights_path))
                self._download_weights(self._weights_path)
                return model_class(self._weights_path)
            else:
                raise e

    def _download_weights(self, path: Path):
        try:
            os.makedirs(path.parent, exist_ok=True)
            subprocess.run(
                [
                    "wget",
                    self.WEIGHTS_URL,
                    "-O",
                    path,
                    "--progress=dot:giga",
                ]
            )
        except Exception:
            self.logger.error("Could not download weights")
            raise Exception("Could not download weights")

    def _remove_weights(self, path: str):
        # Sometimes redownloding weights bugged and created a dir
        # so check also for dir and remove it in both cases
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.isfile(path):
            os.remove(path)

    def stop(self):
        self.ros2_connector.shutdown()
