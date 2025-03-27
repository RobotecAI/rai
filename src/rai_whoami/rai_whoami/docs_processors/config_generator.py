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
from pathlib import Path
from typing import Dict, List, Optional

from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

from ..loader.schema import RobotConstitution, RobotIdentity
from .process_docs import ingest_documentation


class ConfigGenerator:
    """A class for generating configuration files from documentation.

    This class uses LangChain to process documentation and generate structured
    configuration files for robot identity and constitution.

    Parameters
    ----------
    docs_path : Path | str
        Path to the documentation directory.
    model_name : str, default="gpt-4-turbo-preview"
        Name of the LLM model to use.
    recursive : bool, default=True
        Whether to recursively search for documents.
    """

    def __init__(
        self,
        docs_path: Path | str,
        model: BaseChatModel,
        recursive: bool = True,
    ):
        self.docs_path = Path(docs_path)
        self.model = model
        self.recursive = recursive

    def _create_identity_prompt(self) -> ChatPromptTemplate:
        """Create a prompt template for extracting identity information.

        Returns
        -------
        ChatPromptTemplate
            A prompt template for identity extraction.
        """
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert at analyzing documentation and extracting key information about AI systems.
            Your task is to extract the identity information from the provided documentation.
            The output should contain:
            - name: The name of the AI system
            - model: The model identifier
            - version: The version number
            - description: A clear description of the AI system's purpose and capabilities

            Extract this information and format it according to the provided schema.""",
                ),
                ("human", "Here is the documentation to analyze:\n\n{docs}"),
            ]
        )

    def _create_constitution_prompt(self) -> ChatPromptTemplate:
        """Create a prompt template for extracting constitution information.

        Returns
        -------
        ChatPromptTemplate
            A prompt template for constitution extraction.
        """
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert at analyzing documentation and extracting ethical principles and operational guidelines.
            Your task is to extract the constitution information from the provided documentation.
            The output should contain:
            - ethical_principles: List of ethical principles the AI must follow
            - operational_constraints: List of operational constraints
            - safety_protocols: List of safety protocols
            - interaction_rules: List of rules for human-AI interaction
            - learning_guidelines: List of guidelines for learning and adaptation

            Extract this information and format it according to the provided schema.""",
                ),
                ("human", "Here is the documentation to analyze:\n\n{docs}"),
            ]
        )

    def _process_documents(self, docs: List[Document]) -> str:
        """Process a list of documents into a single string.

        Parameters
        ----------
        docs : List[Document]
            List of documents to process.

        Returns
        -------
        str
            Combined text from all documents.
        """
        return "\n\n".join(doc.page_content for doc in docs)

    def generate_identity(self) -> RobotIdentity:
        """Generate identity configuration from documentation.

        Returns
        -------
        RobotIdentity
            The generated identity configuration.

        Raises
        ------
        ValueError
            If the generated configuration is invalid.
        """
        docs = ingest_documentation(self.docs_path, recursive=self.recursive)
        docs_text = self._process_documents(docs)

        chain = self._create_identity_prompt() | self.model.with_structured_output(
            RobotIdentity
        )
        return chain.invoke({"docs": docs_text})

    def generate_constitution(self) -> RobotConstitution:
        """Generate constitution configuration from documentation.

        Returns
        -------
        RobotConstitution
            The generated constitution configuration.

        Raises
        ------
        ValueError
            If the generated configuration is invalid.
        """
        docs = ingest_documentation(self.docs_path, recursive=self.recursive)
        docs_text = self._process_documents(docs)

        chain = self._create_constitution_prompt() | self.model.with_structured_output(
            RobotConstitution
        )
        return chain.invoke({"docs": docs_text})

    def generate_configs(
        self, output_dir: Optional[Path | str] = None
    ) -> Dict[str, Path]:
        """Generate both identity and constitution configurations.

        Parameters
        ----------
        output_dir : Optional[Path | str], default=None
            Directory to save the generated configuration files.
            If None, uses the same directory as the documentation.

        Returns
        -------
        Dict[str, Path]
            Dictionary mapping configuration types to their file paths.
        """
        if output_dir is None:
            output_dir = self.docs_path
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate configurations
        identity = self.generate_identity()
        constitution = self.generate_constitution()

        # Save to files
        identity_path = output_dir / "identity.json"
        constitution_path = output_dir / "constitution.json"

        with open(identity_path, "w") as f:
            json.dump(identity.model_dump(), f, indent=4)

        with open(constitution_path, "w") as f:
            json.dump(constitution.model_dump(), f, indent=4)

        return {"identity": identity_path, "constitution": constitution_path}
