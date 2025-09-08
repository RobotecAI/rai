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
from collections import deque
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.fake_chat_models import ParrotFakeChatModel
from langchain_core.runnables import RunnableConfig
from rai.agents.langchain import invoke_llm_with_tracing
from rai.agents.langchain.agent import LangChainAgent, newMessageBehaviorType
from rai.initialization import get_tracing_callbacks
from rai.messages import HumanMultimodalMessage


@pytest.mark.parametrize(
    "new_message_behavior,in_buffer,out_buffer,output",
    [
        ("take_all", [1, 2, 3], [], [1, 2, 3]),
        ("keep_last", [1, 2, 3], [], [3]),
        ("queue", [1, 2, 3], [2, 3], [1]),
        ("interupt_take_all", [1, 2, 3], [], [1, 2, 3]),
        ("interupt_keep_last", [1, 2, 3], [], [3]),
    ],
)
def test_reduce_messages(
    new_message_behavior: newMessageBehaviorType,
    in_buffer: List,
    out_buffer: List,
    output: List,
):
    buffer = deque(in_buffer)
    output_ = LangChainAgent._apply_reduction_behavior(new_message_behavior, buffer)
    assert output == output_
    assert buffer == deque(out_buffer)


class TestTracingConfiguration:
    """Test tracing configuration integration with langchain agents."""

    def test_tracing_with_missing_config_file(self):
        """Test that tracing gracefully handles missing config.toml file in langchain context."""
        # This should not crash even without config.toml
        callbacks = get_tracing_callbacks()
        assert len(callbacks) == 0

    def test_tracing_with_config_file_present(self, test_config_toml):
        """Test that tracing works when config.toml is present in langchain context."""
        config_path, cleanup = test_config_toml(
            langfuse_enabled=True, langsmith_enabled=False
        )

        try:
            # Mock environment variables to avoid actual API calls
            with patch.dict(
                os.environ,
                {
                    "LANGFUSE_PUBLIC_KEY": "test_key",
                    "LANGFUSE_SECRET_KEY": "test_secret",
                },
            ):
                callbacks = get_tracing_callbacks(config_path=config_path)
                # Should return 1 callback for langfuse
                assert len(callbacks) == 1
        finally:
            cleanup()


class TestInvokeLLMWithTracing:
    """Test the invoke_llm_with_tracing function."""

    def test_invoke_llm_without_tracing(self):
        """Test that invoke_llm_with_tracing works when no tracing callbacks are available."""
        # Mock LLM
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "test response"

        # Mock messages
        mock_messages = ["test message"]

        # Mock get_tracing_callbacks to return empty list (no config.toml)
        with patch(
            "rai.agents.langchain.invocation_helpers.get_tracing_callbacks"
        ) as mock_get_callbacks:
            mock_get_callbacks.return_value = []

            result = invoke_llm_with_tracing(mock_llm, mock_messages)

            mock_llm.invoke.assert_called_once_with(mock_messages, config=None)
            assert result == "test response"

    def test_invoke_llm_with_tracing(self):
        """Test that invoke_llm_with_tracing works when tracing callbacks are available."""
        # Mock LLM
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "test response"

        # Mock messages
        mock_messages = ["test message"]

        # Mock get_tracing_callbacks to return some callbacks
        with patch(
            "rai.agents.langchain.invocation_helpers.get_tracing_callbacks"
        ) as mock_get_callbacks:
            mock_get_callbacks.return_value = ["tracing_callback"]

            _ = invoke_llm_with_tracing(mock_llm, mock_messages)

            # Verify that the LLM was called with enhanced config
            mock_llm.invoke.assert_called_once()
            call_args = mock_llm.invoke.call_args
            assert call_args[0][0] == mock_messages
            assert "callbacks" in call_args[1]["config"]
            assert "tracing_callback" in call_args[1]["config"]["callbacks"]

    def test_invoke_llm_with_existing_config(self):
        """Test that invoke_llm_with_tracing preserves existing config."""
        # Mock LLM
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "test response"

        # Mock messages
        mock_messages = ["test message"]

        # Mock existing config
        existing_config = {"callbacks": ["existing_callback"]}

        # Mock get_tracing_callbacks to return some callbacks
        with patch(
            "rai.agents.langchain.invocation_helpers.get_tracing_callbacks"
        ) as mock_get_callbacks:
            mock_get_callbacks.return_value = ["tracing_callback"]

            _ = invoke_llm_with_tracing(mock_llm, mock_messages, existing_config)

            # Verify that the LLM was called with enhanced config
            mock_llm.invoke.assert_called_once()
            call_args = mock_llm.invoke.call_args
            assert call_args[0][0] == mock_messages
            assert "callbacks" in call_args[1]["config"]
            assert "existing_callback" in call_args[1]["config"]["callbacks"]
            assert "tracing_callback" in call_args[1]["config"]["callbacks"]

    def test_invoke_llm_with_callback_integration(self):
        """Test that invoke_llm_with_tracing works with a callback handler."""
        llm = ParrotFakeChatModel()
        human_msg = HumanMultimodalMessage(content="human")
        response = llm.invoke(
            [human_msg], config=RunnableConfig(callbacks=[BaseCallbackHandler()])
        )
        assert response.content == [{"type": "text", "text": "human"}]
