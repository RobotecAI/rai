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


import argparse
import logging
import os
from dataclasses import dataclass
from typing import Any, List, Literal, Optional

from langfuse import Langfuse

from .base import (
    BaseDataExtractionConfig,
    BaseDataExtractionPipeline,
    BaseDataExtractor,
)

logger = logging.getLogger(__name__)


@dataclass
class LangfuseDataExtractionConfig(BaseDataExtractionConfig):
    """Configuration for observation data extraction pipeline from Langfuse"""

    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    page_size: int = 50
    type_filter: Literal["GENERATION", "EVENT", "SPAN"] = "GENERATION"
    trace_id: Optional[str] = None

    # Configure provider-specific field names
    model_field_name: str = "model"  # Langfuse uses 'model'
    output_field_name: str = "output"  # Langfuse uses 'output'

    def _get_default_include_fields(self) -> List[str]:
        """Return default fields to include for Langfuse"""
        return [
            "model",
            "input",
            "output",
            "promptTokens",
            "traceId",
        ]


class LangfuseDataExtractor(BaseDataExtractor):
    """Extracts observation data from Langfuse API"""

    def _initialize_client(self) -> Langfuse:
        """Initialize Langfuse client with authentication"""
        try:
            client = Langfuse(
                public_key=self.config.langfuse_public_key,
                secret_key=self.config.langfuse_secret_key,
                host=self.config.langfuse_host,
            )
            return client
        except Exception as e:
            logger.error(f"Failed to initialize Langfuse client: {e}")
            raise

    def fetch_data(
        self,
        limit: Optional[int] = None,
        start_time: Optional[
            str
        ] = None,  # ISO format string, natively supported by Langfuse
        stop_time: Optional[
            str
        ] = None,  # ISO format string, natively supported by Langfuse
        **kwargs,
    ) -> List[Any]:
        """Fetch data samples with pagination."""
        if limit is None:
            limit = self.config.max_data_limit
        if start_time is None:
            start_time = self.config.start_time
        if stop_time is None:
            stop_time = self.config.stop_time

        # Get Langfuse-specific parameters
        type_filter = kwargs.get("type_filter", self.config.type_filter)
        trace_id = kwargs.get("trace_id", self.config.trace_id)

        logger.info(
            f"Fetching data samples from Langfuse (type={type_filter}, trace_id={trace_id}, start_time={start_time}, stop_time={stop_time})..."
        )

        all_samples: List[Any] = []
        page = 1
        page_size = self.config.page_size

        try:
            while len(all_samples) < limit:
                remaining = min(page_size, limit - len(all_samples))

                # Build API parameters
                api_params = {
                    "limit": remaining,
                    "page": page,
                    "type": type_filter,
                }

                if trace_id:
                    api_params["trace_id"] = trace_id

                data_response = self.client.api.observations.get_many(**api_params)

                if not data_response.data:
                    logger.info("No more data samples available")
                    break

                all_samples.extend(data_response.data)

                if len(data_response.data) < page_size:
                    logger.info("Reached end of available data samples")
                    break

                page += 1

            if start_time or stop_time:
                logger.info("Applying time filtering locally...")
                filtered_samples = []
                skipped_count = 0

                for sample in all_samples:
                    # Use start_time field for time filtering
                    timestamp = sample.start_time

                    if not timestamp:
                        skipped_count += 1
                        continue

                    # Normalize timestamp to ISO format for consistent comparison
                    timestamp_str = self.config._normalize_timestamp(timestamp)

                    # Apply time filtering using normalized ISO format
                    if start_time and timestamp_str < start_time:
                        continue

                    if stop_time and timestamp_str > stop_time:
                        continue

                    filtered_samples.append(sample)

                if skipped_count > 0:
                    logger.info(
                        f"Skipped {skipped_count} data samples without valid timestamps"
                    )

                # If all samples were skipped due to missing timestamps, warn and return all
                if len(filtered_samples) == 0 and len(all_samples) > 0:
                    logger.warning(
                        "No data samples with valid timestamps found. Returning all samples without time filtering."
                    )
                    all_samples = all_samples
                else:
                    all_samples = filtered_samples

            logger.info(f"Successfully fetched {len(all_samples)} data samples")
            return all_samples

        except Exception as e:
            logger.error(f"Error fetching data samples: {e}")
            raise

    def _get_provider_name(self) -> str:
        """Return the name of the provider"""
        return "Langfuse"


class LangfuseDataExtractionPipeline(BaseDataExtractionPipeline):
    """Main pipeline for extracting and saving data samples from Langfuse"""

    def _create_extractor(self) -> LangfuseDataExtractor:
        """Create and return the Langfuse extractor instance"""
        return LangfuseDataExtractor(self.config)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Extract data samples from Langfuse")
    parser.add_argument(
        "--host", default="http://localhost:3000", help="Langfuse host URL"
    )
    parser.add_argument("--public-key", help="Langfuse public key")
    parser.add_argument("--secret-key", help="Langfuse secret key")
    parser.add_argument(
        "--output",
        default="langfuse_data.jsonl",
        help="Output file for data samples",
    )
    parser.add_argument(
        "--start-time",
        help="Start time for observations (ISO format, e.g., 2024-01-01T00:00:00Z)",
    )
    parser.add_argument(
        "--stop-time",
        help="Stop time for observations (ISO format, e.g., 2024-01-02T00:00:00Z)",
    )
    parser.add_argument(
        "--type",
        dest="type_filter",
        default="GENERATION",
        help="Observation type filter (e.g., GENERATION, EVENT, SPAN)",
    )
    parser.add_argument(
        "--trace-id",
        dest="trace_id",
        default=None,
        help="Restrict to a specific trace ID",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[],
        help="Filter data samples by model name substring(s)",
    )
    parser.add_argument(
        "--include-fields",
        nargs="+",
        help="Fields to include in saved data samples",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=50,
        help="Number of data samples to fetch per API call (default: 50)",
    )

    parser.add_argument(
        "--max-data-limit",
        type=int,
        default=5000,
        help="Maximum number of data samples to fetch (default: 5000)",
    )

    args = parser.parse_args()

    # Create configuration
    config = LangfuseDataExtractionConfig(
        langfuse_host=args.host,
        langfuse_public_key=args.public_key or os.getenv("LANGFUSE_PUBLIC_KEY", ""),
        langfuse_secret_key=args.secret_key or os.getenv("LANGFUSE_SECRET_KEY", ""),
        output_file=args.output,
        max_data_limit=args.max_data_limit,
        type_filter=args.type_filter,
        trace_id=args.trace_id,
        start_time=args.start_time,
        stop_time=args.stop_time,
        filter_models=args.models,
        include_fields=args.include_fields,
        page_size=args.page_size,
    )

    # Validate required configuration
    if not config.langfuse_public_key or not config.langfuse_secret_key:
        logger.error(
            "Missing Langfuse credentials. Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY environment variables or use --public-key and --secret-key arguments"
        )
        return 1

    # Run pipeline
    pipeline = LangfuseDataExtractionPipeline(config)
    success = pipeline.run()

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
