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

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from rai_whoami.loaders.models import EmbodimentSource


class VectorDBBuilder(ABC):
    def __init__(self, embedding: Embeddings, **kwargs: Any):
        self.embedding = embedding

    @abstractmethod
    def build(self, data: EmbodimentSource) -> VectorStore:
        """
        Build a vector database from an EmbodimentSource.
        """
