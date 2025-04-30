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

from typing import TYPE_CHECKING, List, Literal, Optional

from rai_whoami.processors.postprocessors.base import DataPostProcessor
from rai_whoami.processors.preprocessors.base import DataPreProcessor

if TYPE_CHECKING:
    from rai_whoami.pipeline.pipeline import Pipeline
    from rai_whoami.vector_db.builder import VectorDBBuilder


class PipelineBuilder:
    """
    Builder class for constructing Pipeline instances in a fluent interface style.

    Example:
        builder = PipelineBuilder()
        pipeline = (builder
                   .add_preprocessor(text_preprocessor)
                   .add_preprocessor(image_preprocessor)
                   .add_postprocessor(final_processor)
                   .build())
    """

    def __init__(self) -> None:
        """Initialize an empty pipeline builder."""
        self._preprocessors: List[DataPreProcessor] = []
        self._postprocessors: List[DataPostProcessor] = []
        self._aggregate: Literal["merge"] = "merge"
        self._vector_db_builder: Optional[VectorDBBuilder] = None

    def add_preprocessor(self, processor: DataPreProcessor) -> "PipelineBuilder":
        """
        Add a preprocessor to the pipeline.

        Args:
            processor: A DataProcessor instance to add to the preprocessing stage

        Returns:
            self for method chaining
        """
        self._preprocessors.append(processor)
        return self

    def add_preprocessors(
        self, processors: List[DataPreProcessor]
    ) -> "PipelineBuilder":
        """
        Add a list of preprocessors to the pipeline.
        """
        self._preprocessors.extend(processors)
        return self

    def add_postprocessor(self, processor: DataPostProcessor) -> "PipelineBuilder":
        """
        Add a postprocessor to the pipeline.

        Args:
            processor: A DataPostProcessor instance to add to the postprocessing stage

        Returns:
            self for method chaining
        """
        self._postprocessors.append(processor)
        return self

    def add_postprocessors(
        self, processors: List[DataPostProcessor]
    ) -> "PipelineBuilder":
        """
        Add a list of postprocessors to the pipeline.
        """
        self._postprocessors.extend(processors)
        return self

    def set_aggregate_strategy(self, strategy: Literal["merge"]) -> "PipelineBuilder":
        """
        Set the aggregation strategy for the pipeline.

        Args:
            strategy: The strategy to use for aggregating preprocessor outputs.
                     Currently only "merge" is supported.

        Returns:
            self for method chaining

        Raises:
            ValueError: If an unsupported strategy is provided
        """
        if strategy != "merge":
            raise ValueError(f"Unsupported aggregate strategy: {strategy}")
        self._aggregate = strategy
        return self

    def add_vector_db_builder(self, builder: "VectorDBBuilder") -> "PipelineBuilder":
        """
        Add a vector database builder to the pipeline.
        """
        self._vector_db_builder = builder
        return self

    def build(self) -> "Pipeline":
        """
        Build and return a new Pipeline instance with the configured components.

        Returns:
            A new Pipeline instance
        """
        from rai_whoami.pipeline.pipeline import Pipeline

        return Pipeline(
            preprocessors=self._preprocessors,
            postprocessors=self._postprocessors,
            aggregate=self._aggregate,
            vector_db_builder=self._vector_db_builder,
        )

    def reset(self) -> "PipelineBuilder":
        """
        Reset the builder to its initial state.

        Returns:
            self for method chaining
        """
        self._preprocessors = []
        self._postprocessors = []
        self._aggregate = "merge"
        return self
