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

from rai_perception.services.detection_service import DetectionService
from rai_perception.services.segmentation_service import SegmentationService
from rai_perception.services.weights import download_weights

SERVICES = [DetectionService, SegmentationService]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    for service_class in SERVICES:
        weights_path = (
            service_class.DEFAULT_WEIGHTS_ROOT_PATH
            / service_class.WEIGHTS_DIR_PATH_PART
            / service_class.WEIGHTS_FILENAME
        )
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        if not weights_path.exists():
            download_weights(weights_path, logger, service_class.WEIGHTS_URL)
        else:
            logger.info(f"Weights already exist at {weights_path}, skipping download.")
