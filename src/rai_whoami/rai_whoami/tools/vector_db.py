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

from pathlib import Path
from typing import Any, Literal, Type

from langchain_community.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from rai_whoami.vector_db.faiss import get_faiss_client


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
            self.vdb_client = get_faiss_client(
                str(Path(self.root_dir) / "generated"), self.embeddings_model
            )
        else:
            raise ValueError(f"Unsupported database type: {self.database_type}")

    def _run(self, query: str) -> str:
        return str(self.vdb_client.similarity_search(query, k=self.k))
