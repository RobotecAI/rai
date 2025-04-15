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

import pickle
from pathlib import Path
from typing import Any, List, TypedDict


class MultimodalArtifact(TypedDict):
    images: List[str]  # base64 encoded images
    audios: List[str]


def store_artifacts(
    tool_call_id: str, artifacts: List[Any], db_path="artifact_database.pkl"
):
    # TODO(boczekbartek): refactor
    db_path = Path(db_path)
    if not db_path.is_file():
        artifact_database = {}
        with open("artifact_database.pkl", "wb") as file:
            pickle.dump(artifact_database, file)
    with open("artifact_database.pkl", "rb") as file:
        artifact_database = pickle.load(file)
        if tool_call_id not in artifact_database:
            artifact_database[tool_call_id] = artifacts
        else:
            artifact_database[tool_call_id].extend(artifacts)
    with open("artifact_database.pkl", "wb") as file:
        pickle.dump(artifact_database, file)


def get_stored_artifacts(
    tool_call_id: str, db_path="artifact_database.pkl"
) -> List[Any]:
    # TODO(boczekbartek): refactor
    db_path = Path(db_path)
    if not db_path.is_file():
        return []

    with db_path.open("rb") as db:
        artifact_database = pickle.load(db)
        if tool_call_id in artifact_database:
            return artifact_database[tool_call_id]

    return []
