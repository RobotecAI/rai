# Copyright (C) 2026 Robotec.AI
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

from unittest.mock import patch

import numpy as np
import pytest
from langchain_core.documents import Document

from rai_interfaces.srv import VectorStoreRetrieval
from rai_whoami.agents.ros2 import vector_store_retrieval_agent


@pytest.mark.parametrize(
    "scores",
    [
        pytest.param(
            [np.float32(0.6289791), np.float32(0.5291562)], id="numpy-float32"
        ),
        pytest.param([1.5, 0.25], id="python-float"),
        pytest.param([], id="empty"),
    ],
)
def test_service_callback_serializes_scores(scores, tmp_path):
    documents = [
        Document(page_content=f"Document {index}") for index in range(len(scores))
    ]
    with (
        patch.object(vector_store_retrieval_agent, "ROS2Connector"),
        patch.object(vector_store_retrieval_agent, "get_faiss_client") as get_client,
    ):
        search = get_client.return_value.similarity_search_with_score
        search.return_value = list(zip(documents, scores))
        agent = vector_store_retrieval_agent.ROS2VectorStoreRetrievalAgent(
            service_name="rai_whoami_documentation_service",
            root_dir=str(tmp_path),
            k=2,
        )
        request = VectorStoreRetrieval.Request(query="maximum load")
        # Newer ROS generators make field validation opt-in.
        if hasattr(VectorStoreRetrieval.Response, "_check_fields"):
            response = VectorStoreRetrieval.Response(check_fields=True)
        else:
            response = VectorStoreRetrieval.Response()

        result = agent.service_callback(request, response)

    assert result is response
    assert response.success is True
    assert response.message == "Success"
    assert response.documents == [document.page_content for document in documents]
    assert list(response.scores) == [float(score) for score in scores]
    assert all(type(score) is float for score in response.scores)
    search.assert_called_once_with(request.query, k=2)
