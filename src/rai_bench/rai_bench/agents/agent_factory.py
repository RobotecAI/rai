# Copyright (C) 2024 Robotec.AI
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
from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
)

from langchain.chat_models.base import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from rai.agents.langchain.core import ReActAgentState, create_conversational_agent
from rai.messages import HumanMultimodalMessage, SystemMessage, SystemMultimodalMessage

from rai_bench.agents.dual_agent import create_multimodal_to_tool_agent
from rai_bench.agents.plan_agent import (
    PlanExecuteState,
    create_initial_plan_execute_state,
    create_plan_execute_agent,
)
from rai_bench.agents.verification_agent import (
    TaskVerificationState,
    create_initial_task_verification_state,
    create_task_verification_agent,
)
from rai_bench.utils import get_llm_model_name

SystemPromptType = Union[str, SystemMultimodalMessage, SystemMessage]


class AgentFactory(ABC):
    @abstractmethod
    def create_agent(
        self,
        tools: List[Any],
        system_prompt: SystemPromptType,
        logger: logging.Logger | None = None,
    ) -> CompiledStateGraph:
        pass

    @staticmethod
    def clean_model_name(model_str: str) -> str:
        """Clean model name string by removing keep_alive and base_url parameters
        and replacing ':' with '-' so it can be uploaded easily to for example google drive
        """
        import re

        # Remove keep_alive and base_url parameters
        cleaned = re.sub(r"\s*keep_alive=\S+", "", model_str)
        cleaned = re.sub(r"\s*base_url=\S+", "", cleaned)
        # Clean up any extra spaces
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned.replace(":", "-")
        return cleaned

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    def create_initial_state(
        self, prompt: str, **kwargs: Dict[str, Any]
    ) -> ReActAgentState:
        """Create default initial state agents."""
        msg = HumanMultimodalMessage(content=prompt, **kwargs)
        return {"messages": [msg]}


class ConversationalAgentFactory(AgentFactory):
    """Factory for conversational agents."""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def create_agent(
        self,
        tools: List[BaseTool],
        system_prompt: SystemPromptType,
        logger: Optional[logging.Logger] = None,
    ) -> CompiledStateGraph:
        return create_conversational_agent(
            llm=self.llm, tools=tools, system_prompt=system_prompt, logger=logger
        )

    @property
    def model_name(self) -> str:
        return AgentFactory.clean_model_name(get_llm_model_name(self.llm))


class DualAgentFactory(AgentFactory):
    """Factory for dual (multimodal + tool) agents."""

    def __init__(
        self,
        multimodal_llm: BaseChatModel,
        tool_calling_llm: BaseChatModel,
    ):
        self.multimodal_llm = multimodal_llm
        self.tool_calling_llm = tool_calling_llm

    def create_agent(
        self,
        tools: List[BaseTool],
        system_prompt: SystemPromptType,
        logger: Optional[logging.Logger] = None,
        **kwargs: Any,
    ) -> CompiledStateGraph:
        return create_multimodal_to_tool_agent(
            multimodal_llm=self.multimodal_llm,
            tool_llm=self.tool_calling_llm,
            tools=tools,
            tool_system_prompt=system_prompt,
            logger=logger,
            **kwargs,
        )

    @property
    def model_name(self) -> str:
        m_llm_name = AgentFactory.clean_model_name(
            get_llm_model_name(self.multimodal_llm)
        )
        tool_llm_name = AgentFactory.clean_model_name(
            get_llm_model_name(self.multimodal_llm)
        )
        model_str = f"dual_multimodal->{m_llm_name}_tool-calling->{tool_llm_name}"
        return AgentFactory.clean_model_name(model_str)


class TaskVerificationAgentFactory(AgentFactory):
    """Factory for task verification agents."""

    def __init__(
        self,
        worker_llm: BaseChatModel,
        verification_llm: BaseChatModel,
        max_verification_attempts: int = 1,
    ):
        self.worker_llm = worker_llm
        self.verification_llm = verification_llm
        self.max_verification_attempts = max_verification_attempts

    def create_agent(
        self,
        tools: List[BaseTool],
        system_prompt: Optional[Union[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs: Any,
    ) -> CompiledStateGraph:
        return create_task_verification_agent(
            work_llm=self.worker_llm,
            verification_llm=self.verification_llm,
            tools=tools,
            system_prompt=system_prompt,
        )

    def create_initial_state(
        self, prompt: str, **kwargs: Dict[str, Any]
    ) -> TaskVerificationState:
        """Create initial state for task verification agents."""
        return create_initial_task_verification_state(
            original_task=prompt,
            messages=[HumanMultimodalMessage(content=prompt, **kwargs)],
            max_verification_attempts=self.max_verification_attempts,
        )

    @property
    def model_name(self) -> str:
        tool_llm_name = AgentFactory.clean_model_name(
            getattr(self.worker_llm, "model_name", str(self.worker_llm))
        )
        verification_llm_name = AgentFactory.clean_model_name(
            getattr(self.verification_llm, "model_name", str(self.verification_llm))
        )
        return f"verification_worker->{tool_llm_name}_verification->{verification_llm_name}"


class PlanExecuteAgentFactory(AgentFactory):
    """Factory for planning agents."""

    def __init__(
        self,
        planner_llm: BaseChatModel,
        executor_llm: BaseChatModel,
        replanner_llm: BaseChatModel,
    ):
        self.planner_llm = planner_llm
        self.executor_llm = executor_llm
        self.replanner_llm = replanner_llm

    def create_agent(
        self,
        tools: List[BaseTool],
        system_prompt: Optional[Union[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs: Any,
    ) -> CompiledStateGraph:
        return create_plan_execute_agent(
            planner_llm=self.planner_llm,
            executor_llm=self.executor_llm,
            replanner_llm=self.replanner_llm,
            tools=tools,
            system_prompt=system_prompt,
            **kwargs,
        )

    def create_initial_state(
        self, prompt: str, **kwargs: Dict[str, Any]
    ) -> PlanExecuteState:
        """Create initial state for planning agents."""
        return create_initial_plan_execute_state(
            original_task=prompt,
            messages=[HumanMultimodalMessage(content=prompt, **kwargs)],
        )

    @property
    def model_name(self) -> str:
        planner = AgentFactory.clean_model_name(
            getattr(self.planner_llm, "model_name", str(self.planner_llm))
        )
        executor = AgentFactory.clean_model_name(
            getattr(self.executor_llm, "model_name", str(self.executor_llm))
        )
        replanner = AgentFactory.clean_model_name(
            getattr(self.replanner_llm, "model_name", str(self.replanner_llm))
        )
        return f"plan&execute_planner->{planner}_executor->{executor}_replanner->{replanner}"
