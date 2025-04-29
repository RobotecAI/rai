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

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from rai_whoami.models import EmbodimentSource


class VectorDBBuilder(ABC):
    def __init__(
        self,
        root_dir: str,
        embedding: Embeddings,
        model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Optional[Any],
    ):
        self.root_dir = Path(root_dir)
        self.embedding = embedding
        if model_kwargs is None:
            model_kwargs = {}
        self.model_kwargs = model_kwargs

    def build(self, data: EmbodimentSource) -> VectorStore:
        """
        Build a vector database from an EmbodimentSource.
        """
        db = self._build(data)
        self.dump_model_kwargs()
        return db

    @abstractmethod
    def _build(self, data: EmbodimentSource) -> VectorStore:
        """
        Build a vector database from an EmbodimentSource.
        """

    def dump_model_kwargs(self):
        (self.root_dir / "vdb_kwargs.json").write_text(json.dumps(self.model_kwargs))
