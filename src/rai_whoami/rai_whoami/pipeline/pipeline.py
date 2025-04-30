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
from typing import List, Literal, Optional

from rai_whoami.models import EmbodimentInfo, EmbodimentSource
from rai_whoami.processors.postprocessors.base import DataPostProcessor
from rai_whoami.processors.preprocessors.base import DataPreProcessor
from rai_whoami.vector_db.builder import VectorDBBuilder


def merge_intermediate_outputs(
    intermediate_outputs: List[EmbodimentInfo],
) -> EmbodimentInfo:
    final_output = EmbodimentInfo(
        rules=[], capabilities=[], behaviors=[], description="", images=[]
    )
    for output in intermediate_outputs:
        final_output += output
    return final_output


class Pipeline:
    def __init__(
        self,
        preprocessors: List[DataPreProcessor],
        postprocessors: List[DataPostProcessor],
        aggregate: Literal["merge"] = "merge",
        vector_db_builder: Optional[VectorDBBuilder] = None,
    ):
        self.preprocessors = preprocessors
        self.postprocessors = postprocessors
        self.aggregate = aggregate
        self.vector_db_builder = vector_db_builder
        self.logger = logging.getLogger(__name__)

    def process(self, input: EmbodimentSource) -> EmbodimentInfo:
        if self.vector_db_builder is not None:
            self.logger.info("Building vector database.")
            self.vector_db_builder.build(input)
            self.logger.info("Vector database built.")

        self.logger.info(
            f"Processing input started. Total preprocessors: {len(self.preprocessors)}."
        )
        intermediate_outputs: List[EmbodimentInfo] = []
        for processor in self.preprocessors:
            self.logger.info(
                f"Processing input with preprocessor: {processor.__class__.__name__}"
            )
            intermediate_outputs.append(processor.process(input))

        self.logger.info(
            f"Aggregating intermediate outputs. {self.aggregate.capitalize()} strategy."
        )
        if self.aggregate == "merge":
            output = merge_intermediate_outputs(intermediate_outputs)
        else:
            raise ValueError(f"Invalid aggregate value: {self.aggregate}")

        self.logger.info(
            f"Processing output with postprocessors: {len(self.postprocessors)}."
        )
        for processor in self.postprocessors:
            self.logger.info(
                f"Processing output with postprocessor: {processor.__class__.__name__}"
            )
            output = processor.process(output)
        self.logger.info("Processing output completed.")

        return output
