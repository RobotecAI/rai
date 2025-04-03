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


import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

from rai.agents import BaseAgent
from rai.communication import ROS2ARIConnector


class BaseVisionAgent(BaseAgent):
    WEIGHTS_URL: str = ""
    WEIGHTS_FILENAME: str = ""

    def __init__(
        self,
        weights_path: str | Path = Path.home() / Path(".cache/rai/"),
        ros2_name: str = "",
        logger: Optional[logging.Logger] = None,
    ):
        self._weights_path = Path(weights_path)
        os.makedirs(self._weights_path, exist_ok=True)
        self._init_weight_path()
        self._logger = logger if logger else logging.getLogger(__name__)
        ros2_connector = ROS2ARIConnector(ros2_name)

        super().__init__(connectors={"ros2": ros2_connector})

    def _init_weight_path(self):
        try:
            if self.WEIGHTS_FILENAME == "":
                raise ValueError("WEIGHTS_FILENAME is not set")

            install_path = (
                self._weights_path / "vision" / "weights" / self.WEIGHTS_FILENAME
            )
            # make sure the file exists
            if install_path.exists():
                self._weights_path = install_path
            else:
                self._download_weights(install_path)
                self._weights_path = install_path

        except Exception:
            self._logger.error("Could not find package path")
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
            self._logger.error("Could not download weights")
            raise Exception("Could not download weights")

    def _remove_weights(self, path: Path):
        if path.exists():
            os.remove(path)

    def stop(self):
        assert isinstance(self.connectors["ros2"], ROS2ARIConnector)
        self.connectors["ros2"].shutdown()
