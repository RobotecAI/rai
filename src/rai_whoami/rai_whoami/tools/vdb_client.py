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

import inspect
import json
from importlib import import_module
from pathlib import Path
from typing import Any, Literal, Type

from langchain_community.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


def get_class_from_string(class_path: str) -> type:
    module_path, class_name = class_path.rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, class_name)


def initialize_embeddings(class_path: str, **kwargs: Any) -> Embeddings:
    c = get_class_from_string(class_path)
    kwargs = {k: kwargs[k] for k in inspect.signature(c).parameters if k in kwargs}
    return c(**kwargs)


class QueryDatabaseToolInput(BaseModel):
    query: str = Field(..., description="The query to search the database with")


class QueryDatabaseTool(BaseTool):
    name: str = "query_database"
    description: str = "Query the database with a natural language query"
    args_schema: Type[QueryDatabaseToolInput] = QueryDatabaseToolInput

    database_type: Literal["faiss"] = Field(
        default="faiss", description="The type of database to use"
    )
    root_dir: str = Field(..., description="The root directory of the database")
    embeddings_model: Embeddings | None = None

    k: int = Field(default=4, description="The number of results to return")
    vdb_client: VectorStore | None = None

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        if self.database_type == "faiss":
            from langchain_community.vectorstores import FAISS

            if self.embeddings_model is None:
                vdb_kwargs = json.load(open(Path(self.root_dir) / "vdb_kwargs.json"))
                self.embeddings_model = initialize_embeddings(
                    vdb_kwargs["embeddings"]["class"], **vdb_kwargs
                )

            self.vdb_client = FAISS.load_local(
                folder_path=self.root_dir,
                embeddings=self.embeddings_model,
                allow_dangerous_deserialization=True,
            )

    def _run(self, query: str) -> str:
        return str(self.vdb_client.similarity_search(query, k=self.k))
