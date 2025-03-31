#!/usr/bin/env python3
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

from rai_whoami.loader.manager import ConfigManager, ConfigSource


def load_from_file_example():
    """Example of loading configuration from a JSON file."""
    print("\n=== Loading configuration from file ===")

    # Get the path to the example config file
    current_dir = Path(__file__).parent
    config_path = current_dir / "config.json"

    # Create a configuration manager for file source
    manager = ConfigManager(source=ConfigSource.FILE, config_path=str(config_path))

    # Load and get the configuration
    config = manager.get_config()

    # Print some example information
    print(f"Robot Name: {config.identity.name}")
    print(f"Model: {config.identity.model}")
    print(f"Capabilities: {', '.join(config.identity.capabilities)}")
    print(f"Vector DB Type: {config.vector_db.type}")
    print(f"Environment: {config.environment}")


def load_from_mongodb_example():
    """Example of loading configuration from MongoDB."""
    print("\n=== Loading configuration from MongoDB ===")

    # Create a configuration manager for MongoDB source
    manager = ConfigManager(
        source=ConfigSource.MONGODB,
        connection_string="mongodb://localhost:27017",
        database="rai_whoami",
        collection="configs",
        config_id="default",
        username=os.getenv("MONGODB_USERNAME"),  # Optional
        password=os.getenv("MONGODB_PASSWORD"),  # Optional
    )

    # Load and get the configuration
    config = manager.get_config()

    # Print some example information
    print(f"Robot Name: {config.identity.name}")
    print(f"Model: {config.identity.model}")
    print(f"Capabilities: {', '.join(config.identity.capabilities)}")
    print(f"Vector DB Type: {config.vector_db.type}")
    print(f"Environment: {config.environment}")


def main():
    """Main function demonstrating configuration loading."""
    print("RAI Whoami Configuration Loading Examples")
    print("=======================================")

    # Example 1: Load from file
    load_from_file_example()

    # Example 2: Load from MongoDB (commented out as it requires MongoDB setup)
    # load_from_mongodb_example()


if __name__ == "__main__":
    main()
