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

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class VectorDBConfig(BaseModel):
    """Configuration for vector database.

    Parameters
    ----------
    type : str
        Type of vector database (e.g., "chroma", "pinecone", "weaviate").
    connection_params : Dict[str, Any]
        Connection parameters for the vector database.
    collection_name : str
        Name of the collection to use.
    embedding_model : str
        Name of the embedding model to use.
    dimension : int
        Dimension of the vectors.
    settings : Dict[str, Any]
        Additional settings for the vector database.
    """

    type: str = Field(
        description="Type of vector database (e.g., 'chroma', 'pinecone', 'weaviate')"
    )
    connection_params: Dict[str, Any] = Field(
        description="Connection parameters for the vector database"
    )
    collection_name: str = Field(description="Name of the collection to use")
    embedding_model: str = Field(description="Name of the embedding model to use")
    dimension: int = Field(description="Dimension of the vectors")
    settings: Dict[str, Any] = Field(
        description="Additional settings for the vector database"
    )


class RobotIdentity(BaseModel):
    """Robot's identity configuration.

    Parameters
    ----------
    name : str
        Name of the AI system.
    model : str
        Model identifier.
    version : str
        Version of the AI system.
    description : str
        Description of the AI system and its purpose.
    """

    name: str = Field(description="Name of the AI system")
    model: str = Field(description="Model identifier")
    version: str = Field(description="Version of the AI system")
    description: str = Field(
        description="Description of the AI system and its purpose starting with 'You' to ensure proper embodiment"
    )


class RobotConstitution(BaseModel):
    """Robot's constitution and operational rules.

    Parameters
    ----------
    ethical_principles : List[str]
        List of ethical principles the robot must follow.
    operational_constraints : List[str]
        List of operational constraints.
    safety_protocols : List[str]
        List of safety protocols.
    interaction_rules : List[str]
        List of rules for human-robot interaction.
    learning_guidelines : List[str]
        List of guidelines for learning and adaptation.
    """

    ethical_principles: List[str] = Field(
        description="List of ethical principles the robot must follow"
    )
    operational_constraints: List[str] = Field(
        description="List of operational constraints"
    )
    safety_protocols: List[str] = Field(description="List of safety protocols")
    interaction_rules: List[str] = Field(
        description="List of rules for human-robot interaction"
    )
    learning_guidelines: List[str] = Field(
        description="List of guidelines for learning and adaptation"
    )


class RobotConfig(BaseModel):
    """Complete robot configuration.

    Parameters
    ----------
    version : str
        Version of the configuration.
    environment : str
        Environment identifier (e.g., "development", "production").
    identity : RobotIdentity
        Robot's identity configuration.
    constitution : RobotConstitution
        Robot's constitution and operational rules.
    vector_db : VectorDBConfig
        Vector database configuration.
    """

    version: str = Field(description="Version of the configuration")
    environment: str = Field(
        description="Environment identifier (e.g., 'development', 'production')"
    )
    identity: RobotIdentity = Field(description="Robot's identity configuration")
    constitution: RobotConstitution = Field(
        description="Robot's constitution and operational rules"
    )
    vector_db: VectorDBConfig = Field(description="Vector database configuration")


class ConfigValidationError(Exception):
    """Exception raised when configuration validation fails."""

    pass
