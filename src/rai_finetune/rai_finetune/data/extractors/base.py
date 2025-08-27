# Copyright (C) 2025 Julia Jia
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
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("RAI_FINETUNE_LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class BaseDataExtractionConfig:
    """Base configuration for data extraction pipeline"""

    # Common fields
    output_file: str = "raw_data.jsonl"
    max_data_limit: int = 5000  # Maximum number of data to fetch
    start_time: Optional[Union[str, datetime]] = (
        None  # ISO format string or datetime object
    )
    stop_time: Optional[Union[str, datetime]] = (
        None  # ISO format string or datetime object
    )

    # Filtering
    filter_models: Optional[List[str]] = None
    include_fields: [List[str]] = None

    # Provider-specific field names for filtering
    model_field_name: str = "model"  # Field name containing model information, langfuse uses 'model', langsmith uses 'model_name'
    output_field_name: str = "output"  # Field name containing output data, both langfuse and langsmith use 'output'

    def __post_init__(self):
        if self.filter_models is None:
            self.filter_models = []
        self.include_fields = self._get_default_include_fields()

        # Convert datetime objects to ISO strings
        self._convert_datetime_fields()

    def _convert_datetime_fields(self):
        """Convert datetime objects to ISO strings for consistent string comparison"""
        if isinstance(self.start_time, datetime):
            self.start_time = self.start_time.isoformat()

        if isinstance(self.stop_time, datetime):
            self.stop_time = self.stop_time.isoformat()

    @abstractmethod
    def _get_default_include_fields(self) -> List[str]:
        """Return default fields to include for this provider"""
        pass

    @staticmethod
    def parse_iso_time(time_str: str) -> datetime:
        """Convert ISO format string to datetime object"""
        return datetime.fromisoformat(time_str.replace("Z", "+00:00"))

    @staticmethod
    def _normalize_timestamp(timestamp: Any) -> str:
        """Normalize timestamp to ISO format for consistent ML data processing

        Handles various timestamp formats:
        - LangSmith: "2025-08-17T06:09:30.781431" (ISO format, no timezone)
        - Langfuse: "2025-08-17T06:09:30.781431Z" (ISO format with UTC)
        - Legacy: "2025-08-17 06:09:30.781431" (space-separated, no timezone)
        """
        if timestamp is None:
            return None

        if isinstance(timestamp, datetime):
            return timestamp.isoformat() + "Z"

        if isinstance(timestamp, str):
            # Handle space-separated format (e.g., "2025-08-17 06:09:30.781431")
            if " " in timestamp and "T" not in timestamp:
                timestamp = timestamp.replace(" ", "T")

            # LangSmith timestamps often lack timezone info, so we add it
            # Examples: "2025-08-17T06:09:30.781431" -> "2025-08-17T06:09:30.781431Z"
            if not timestamp.endswith("Z") and "+" not in timestamp:
                timestamp += "Z"

            return timestamp

        # Fallback: convert to string and normalize
        timestamp_str = str(timestamp)
        if " " in timestamp_str and "T" not in timestamp_str:
            timestamp_str = timestamp_str.replace(" ", "T")
        if not timestamp_str.endswith("Z") and "+" not in timestamp_str:
            timestamp_str += "Z"
        return timestamp_str


class BaseDataExtractor(ABC):
    """Abstract base class for data extractors"""

    def __init__(self, config: BaseDataExtractionConfig):
        self.config = config
        self.client = self._initialize_client()

    @abstractmethod
    def _initialize_client(self) -> Any:
        """Initialize and return the client for the specific data source provider"""
        pass

    @abstractmethod
    def fetch_data(
        self,
        limit: Optional[int] = None,
        start_time: Optional[Union[str, datetime]] = None,
        stop_time: Optional[Union[str, datetime]] = None,
        **kwargs,
    ) -> List[Any]:
        """Fetch data with pagination. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _get_provider_name(self) -> str:
        """Return the name of the provider (e.g., 'LangSmith', 'Langfuse')"""
        pass

    def filter_by_model(self, item: Any, filter_models: List[str]) -> bool:
        """Generic model filtering logic using configurable field name"""
        if not filter_models:
            return True

        model_value = getattr(item, self.config.model_field_name, None)
        if model_value and any(
            fm.lower() in str(model_value).lower() for fm in filter_models
        ):
            return True

        return False

    def has_nonempty_output(self, item: Any) -> bool:
        """Generic check for non-empty output using configurable field name"""
        output = getattr(item, self.config.output_field_name, None)
        if output is None:
            return False

        if isinstance(output, str):
            return output.strip() != ""
        elif isinstance(output, dict):
            return bool(output)
        else:
            return bool(output)

    def data_sample_to_mapping(self, obj: Any) -> Dict[str, Any]:
        """Generic conversion of a data sample to a Python mapping (dict)."""
        # Prefer model_dump (Pydantic v2)
        if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
            data = obj.model_dump()
            if isinstance(data, dict):
                return self._sanitize_dict(data)

        # Fallback to dict() (Pydantic v1)
        if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
            data = obj.dict()
            if isinstance(data, dict):
                return self._sanitize_dict(data)

        # Fallback to __dict__
        if hasattr(obj, "__dict__"):
            return self._sanitize_dict(dict(obj.__dict__))

        # Last resort: convert to string
        return {"value": str(obj)}

    def _sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize dictionary values for JSON serialization"""
        sanitized = {}
        for key, value in data.items():
            if hasattr(
                value, "hex"
            ):  # UUID objects (common in LangSmith API responses)
                sanitized[key] = str(value)
            elif isinstance(value, datetime):  # datetime objects
                sanitized[key] = value.isoformat()
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_dict(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    self._sanitize_dict(item)
                    if isinstance(item, dict)
                    else str(item)
                    if hasattr(item, "hex")
                    else item.isoformat()
                    if isinstance(item, datetime)
                    else item
                    for item in value
                ]
            else:
                sanitized[key] = value
        return sanitized


class BaseDataExtractionPipeline(ABC):
    """Abstract base class for data extraction pipelines"""

    def __init__(self, config: BaseDataExtractionConfig):
        self.config = config
        self.extractor = self._create_extractor()

    @abstractmethod
    def _create_extractor(self) -> BaseDataExtractor:
        """Create and return the appropriate extractor instance"""
        pass

    def run(self) -> bool:
        """Main pipeline execution. Common logic shared across providers."""
        try:
            provider_name = self.extractor._get_provider_name()
            logger.info(f"🚀 Starting {provider_name} data extraction pipeline...")

            data = self.extractor.fetch_data()
            # log all fields in the first data sample
            first_sample = self.extractor.data_sample_to_mapping(data[0])
            logger.debug(
                f"Data fields from first sample: {', '.join(sorted(first_sample.keys()))}"
            )

            # Filter by model if specified
            if self.config.filter_models:
                before = len(data)
                data = [
                    obs
                    for obs in data
                    if self.extractor.filter_by_model(obs, self.config.filter_models)
                ]
                logger.info(f"Filtered data by model: {before} -> {len(data)}")

            # Filter out data with no output
            before_output = len(data)
            data = [obs for obs in data if self.extractor.has_nonempty_output(obs)]
            logger.info(f"Filtered empty-output data: {before_output} -> {len(data)}")

            if not data:
                logger.error("No data to save after filtering")
                return False

            # Print fields once (post-inclusion view)
            try:
                first_mapped = self.extractor.data_sample_to_mapping(data[0])
                if self.config.include_fields:
                    first_mapped = {
                        k: first_mapped[k]
                        for k in self.config.include_fields
                        if k in first_mapped
                    }
                logger.info(
                    "Data fields (first item, after inclusion): "
                    + ", ".join(sorted(first_mapped.keys()))
                )
            except Exception as e:
                logger.debug(f"Could not print data fields: {e}")

            self._save_data(data)
            logger.info(f"✅ Successfully saved {len(data)} data samples")
            return True
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return False

    def _save_data(self, data: List[Any]) -> None:
        """Save data to JSONL. Common implementation shared across providers."""
        output_path = self.config.output_file
        with open(output_path, "w", encoding="utf-8") as f:
            for obs in data:
                mapped = self.extractor.data_sample_to_mapping(obs)
                if self.config.include_fields:
                    mapped = {
                        k: mapped[k] for k in self.config.include_fields if k in mapped
                    }
                f.write(json.dumps(mapped, ensure_ascii=False) + "\n")

        logger.info(f"💾 Saved data to {output_path}")
