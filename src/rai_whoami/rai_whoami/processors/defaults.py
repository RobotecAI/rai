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

from typing import List

from rai_whoami.processors.postprocessors import (
    CompressorPostProcessor,
    StylePostProcessor,
)
from rai_whoami.processors.postprocessors.base import DataPostProcessor
from rai_whoami.processors.preprocessors import (
    DocsPreProcessor,
    ImagePreProcessor,
)
from rai_whoami.processors.preprocessors.base import DataPreProcessor


def get_default_preprocessors() -> List[DataPreProcessor]:
    return [
        ImagePreProcessor(),
        DocsPreProcessor(),
    ]


def get_default_postprocessors() -> List[DataPostProcessor]:
    return [
        CompressorPostProcessor(),
        StylePostProcessor(),
    ]
