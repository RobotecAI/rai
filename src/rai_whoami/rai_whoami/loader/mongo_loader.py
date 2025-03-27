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

from datetime import datetime
from typing import Optional

from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

from .base import ConfigData, ConfigLoader
from .file_loader import ConfigLoadError, ConfigValidationError


class MongoConfigLoader(ConfigLoader):
    """Configuration loader that reads from MongoDB.

    This loader supports reading configuration data from MongoDB collections
    and includes validation of the configuration structure.
    """

    def __init__(
        self,
        connection_string: str,
        database: str,
        collection: str,
        config_id: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """Initialize the MongoDB configuration loader.

        Args:
            connection_string: MongoDB connection string.
            database: Name of the database.
            collection: Name of the collection.
            config_id: ID of the configuration document.
            username: Optional username for authentication.
            password: Optional password for authentication.
        """
        self.connection_string = connection_string
        self.database = database
        self.collection = collection
        self.config_id = config_id

        # Create MongoDB client
        try:
            self.client = MongoClient(
                connection_string, username=username, password=password
            )
            # Test connection
            self.client.server_info()
        except ServerSelectionTimeoutError as e:
            raise ConfigLoadError(f"Failed to connect to MongoDB: {e}")

    def load(self) -> ConfigData:
        """Load configuration from MongoDB.

        Returns:
            ConfigData: The loaded configuration data.

        Raises:
            ConfigLoadError: If there's an error reading from MongoDB.
        """
        try:
            db = self.client[self.database]
            collection = db[self.collection]

            config_doc = collection.find_one({"_id": self.config_id})
            if not config_doc:
                raise ConfigLoadError(
                    f"Configuration not found with ID: {self.config_id}"
                )

            # Remove MongoDB _id field
            config_data = {k: v for k, v in config_doc.items() if k != "_id"}

            return ConfigData(
                data=config_data,
                source=f"mongodb://{self.database}/{self.collection}",
                version=config_data.get("version"),
                last_modified=datetime.utcnow().isoformat(),
            )
        except Exception as e:
            raise ConfigLoadError(f"Error loading configuration from MongoDB: {e}")

    def validate(self, config: ConfigData) -> bool:
        """Validate the configuration data.

        Args:
            config: The configuration data to validate.

        Returns:
            bool: True if the configuration is valid.

        Raises:
            ConfigValidationError: If the configuration is invalid.
        """
        if not isinstance(config.data, dict):
            raise ConfigValidationError("Configuration data must be a dictionary")

        required_fields = ["version", "environment"]
        for field in required_fields:
            if field not in config.data:
                raise ConfigValidationError(f"Missing required field: {field}")

        return True

    def __del__(self):
        """Clean up MongoDB connection."""
        if hasattr(self, "client"):
            self.client.close()
