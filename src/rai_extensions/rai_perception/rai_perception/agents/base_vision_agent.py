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
import subprocess
from pathlib import Path

from rai.agents import BaseAgent
from rai.communication.ros2 import ROS2Connector


class BaseVisionAgent(BaseAgent):
    WEIGHTS_URL: str = ""
    DEFAULT_WEIGHTS_ROOT_PATH: Path = Path.home() / Path(".cache/rai/")
    WEIGHTS_DIR_PATH_PART: Path = Path("vision/weights")
    WEIGHTS_FILENAME: str = ""

    def __init__(
        self,
        weights_root_path: str | Path = DEFAULT_WEIGHTS_ROOT_PATH,
        ros2_name: str = "",
    ):
        if not self.WEIGHTS_FILENAME:
            raise ValueError("WEIGHTS_FILENAME is not set")
        super().__init__()
        self.weights_root_path = Path(weights_root_path)
        self.weights_path = (
            self.weights_root_path / self.WEIGHTS_DIR_PATH_PART / self.WEIGHTS_FILENAME
        )
        # create the directory structure
        self.weights_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.weights_path.exists():
            self._download_weights()
        self.ros2_connector = ROS2Connector(ros2_name, executor_type="single_threaded")

    def _load_model_with_error_handling(self, model_class):
        """Load model with automatic error handling for corrupted weights.

        Args:
            model_class: A class that can be instantiated with weights_path

        Returns:
            The loaded model instance
        """
        try:
            return model_class(self.weights_path)
        except RuntimeError as e:
            self.logger.error(f"Could not load model: {e}")
            if "PytorchStreamReader" in str(e):
                self.logger.error("The weights might be corrupted. Redownloading...")
                self._remove_weights()
                self._download_weights()
                return model_class(self.weights_path)
            else:
                raise e

    def _download_weights(self):
        self.logger.info(
            f"Downloading weights from {self.WEIGHTS_URL} to {self.weights_path}"
        )
        try:
            subprocess.run(
                [
                    "wget",
                    self.WEIGHTS_URL,
                    "-O",
                    str(self.weights_path),
                    "--progress=dot:giga",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            # Verify file exists and has reasonable size (> 1MB)
            if not os.path.exists(self.weights_path):
                raise Exception(f"Downloaded file not found at {self.weights_path}")
            file_size = os.path.getsize(self.weights_path)
            if file_size < 1024 * 1024:
                raise Exception(
                    f"Downloaded file is too small ({file_size} bytes), expected > 1MB"
                )
            self.logger.info(
                f"Successfully downloaded weights ({file_size / (1024 * 1024):.2f} MB)"
            )
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else e.stdout if e.stdout else str(e)
            self.logger.error(f"wget failed: {error_msg}")
            # Clean up partial download
            if os.path.exists(self.weights_path):
                os.remove(self.weights_path)
            raise Exception(f"Could not download weights: {error_msg}")
        except Exception as e:
            self.logger.error(f"Could not download weights: {e}")
            # Clean up partial download
            if os.path.exists(self.weights_path):
                os.remove(self.weights_path)
            raise

    def _remove_weights(self):
        os.remove(self.weights_path)

    def stop(self):
        self.ros2_connector.shutdown()
