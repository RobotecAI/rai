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
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, cast

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from rai.initialization import get_embeddings_model

from rai_whoami.models import EmbodimentSource
from rai_whoami.vector_db.builder import VectorDBBuilder


class FAISSBuilder(VectorDBBuilder):
    def __init__(
        self,
        root_dir: str = "faiss/",
        embedding: Optional[Embeddings] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.root_dir = Path(root_dir)
        if embedding is None:
            embedding, model_kwargs = cast(
                Tuple[Embeddings, Dict[str, Any]],
                get_embeddings_model(return_kwargs=True),
            )
        super().__init__(
            root_dir=root_dir, embedding=embedding, model_kwargs=model_kwargs
        )

    def _build(self, data: EmbodimentSource):
        if len(data.documentation) == 0:
            raise ValueError("No documents found")
        os.makedirs(self.root_dir, exist_ok=True)
        db = FAISS.from_documents(data.documentation, self.embedding)
        db.save_local(self.root_dir.as_posix())
        c = str(db.__class__).strip("<>").replace("class '", "").replace("'", "")
        new_kwargs = {"vectorstore": {"class": c}, "embeddings": self.model_kwargs}
        self.model_kwargs = new_kwargs
        self.dump_model_kwargs()
        return db
