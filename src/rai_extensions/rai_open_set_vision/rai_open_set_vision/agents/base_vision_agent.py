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
        self._init_weight_path()
        self.weight_path = self._weights_path
        self.ros2_connector = ROS2Connector(ros2_name)

    def _init_weight_path(self):
        try:
            if self.WEIGHTS_FILENAME == "":
                raise ValueError("WEIGHTS_FILENAME is not set")

            install_path = (
                self._weights_path / "vision" / "weights" / self.WEIGHTS_FILENAME
            )
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
        # NOTE (jm) sometimes downloding weights bugs and creates a dir
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.isfile(path):
            os.remove(path)

    def stop(self):
        self.ros2_connector.shutdown()
