# Copyright (C) 2025 RAI Development Team
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

# Author: Julia Jia


import argparse
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    from langfuse import Langfuse
except ImportError:
    print("Error: langfuse package not found. Install with: pip install langfuse")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ObservationExtractionConfig:
    """Configuration for observation extraction pipeline from Langfuse"""

    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    # Observation related
    output_file: str = "observations.jsonl"
    max_observations: int = 1000
    type_filter: str = "GENERATION"  # e.g., GENERATION, EVENT, SPAN
    start_time: Optional[str] = None  # ISO format string
    stop_time: Optional[str] = None  # ISO format string
    trace_id: Optional[str] = None

    # Filtering
    filter_models: Optional[List[str]] = None
    # Field name in the observation object
    include_fields: Optional[List[str]] = None

    def __post_init__(self):
        if self.filter_models is None:
            self.filter_models = []
        if self.include_fields is None:
            self.include_fields = [
                "model",
                "input",
                "output",
                "promptTokens",
                "traceId",
            ]


class LangfuseObservationExtractor:
    """Extracts observations from Langfuse API"""

    def __init__(self, config: ObservationExtractionConfig):
        self.config = config
        self.client = self._initialize_client()

    def _initialize_client(self) -> Langfuse:
        """Initialize Langfuse client with authentication"""
        try:
            client = Langfuse(
                public_key=self.config.langfuse_public_key,
                secret_key=self.config.langfuse_secret_key,
                host=self.config.langfuse_host,
            )

            # Verify connection
            if client.auth_check():
                logger.info("Successfully connected to Langfuse")
            else:
                logger.error("❌ Failed to authenticate with Langfuse")
                raise ConnectionError("Langfuse authentication failed")

            return client
        except Exception as e:
            logger.error(f"Failed to initialize Langfuse client: {e}")
            raise

    def fetch_observations(
        self,
        limit: Optional[int] = None,
        type_filter: Optional[str] = None,
        trace_id: Optional[str] = None,
        start_time: Optional[str] = None,
        stop_time: Optional[str] = None,
    ) -> List[Any]:
        """Fetch observations with pagination."""
        if limit is None:
            limit = self.config.max_observations
        if type_filter is None:
            type_filter = self.config.type_filter
        if trace_id is None:
            trace_id = self.config.trace_id
        if start_time is None:
            start_time = self.config.start_time
        if stop_time is None:
            stop_time = self.config.stop_time

        logger.info(
            f"Fetching observations from Langfuse (type={type_filter}, trace_id={trace_id}, start_time={start_time}, stop_time={stop_time})..."
        )

        all_observations: List[Any] = []
        page = 1
        page_size = 50

        try:
            while len(all_observations) < limit:
                remaining = min(page_size, limit - len(all_observations))

                observations_response = self.client.api.observations.get_many(
                    limit=remaining,
                    page=page,
                    type=type_filter,
                    trace_id=trace_id,
                )

                if not observations_response.data:
                    logger.info("No more observations available")
                    break

                all_observations.extend(observations_response.data)

                if len(observations_response.data) < remaining:
                    logger.info("Reached end of available observations")
                    break

                page += 1

            # Apply time filtering locally if specified
            if start_time or stop_time:
                filtered_observations = []
                for obs in all_observations:
                    timestamp = getattr(obs, "timestamp", None)
                    if timestamp:
                        # Convert timestamp to comparable format
                        try:
                            if hasattr(timestamp, "isoformat"):
                                obs_time = timestamp.isoformat()
                            else:
                                obs_time = str(timestamp)

                            # Apply start time filter
                            if start_time and obs_time < start_time:
                                continue

                            # Apply stop time filter
                            if stop_time and obs_time > stop_time:
                                continue

                            filtered_observations.append(obs)
                        except Exception as e:
                            logger.debug(f"Could not parse timestamp {timestamp}: {e}")
                            # Include observation if timestamp parsing fails
                            filtered_observations.append(obs)
                    else:
                        # Include observation if no timestamp
                        filtered_observations.append(obs)

                all_observations = filtered_observations

            logger.info(f"Successfully fetched {len(all_observations)} observations")
            return all_observations

        except Exception as e:
            logger.error(f"Error fetching observations: {e}")
            raise


def filter_observation_by_model(item: Any, filter_models: Optional[List[str]]) -> bool:
    """Filter observations by model value present directly or inside metadata."""
    if not filter_models:
        return True

    # Direct model attribute on observation
    model_value = getattr(item, "model", None)
    if model_value and any(
        fm.lower() in str(model_value).lower() for fm in filter_models
    ):
        return True

    # Check metadata string for model name hints
    metadata_str = getattr(item, "metadata", "")
    if metadata_str and isinstance(metadata_str, str):
        try:
            metadata = json.loads(metadata_str)
            model_name = metadata.get("ls_model_name", "")
            if any(fm.lower() in str(model_name).lower() for fm in filter_models):
                return True
        except json.JSONDecodeError:
            pass

    # Fallback: inspect nested attributes if available (e.g., parent span/generation fields)
    for attribute_name in ("observations", "children", "spans"):
        nested_items = getattr(item, attribute_name, [])
        for nested in nested_items or []:
            nested_model = getattr(nested, "model", None)
            if nested_model and any(
                fm.lower() in str(nested_model).lower() for fm in filter_models
            ):
                return True

    return False


def has_nonempty_output(item: Any) -> bool:
    """Return True if observation has a non-empty output field."""
    output = getattr(item, "output", None)
    if output is None:
        return False

    if isinstance(output, str):
        return output.strip() != ""

    if isinstance(output, (list, dict)):
        return len(output) > 0

    # Try common structured object dumps
    try:
        if hasattr(output, "model_dump") and callable(getattr(output, "model_dump")):
            return bool(output.model_dump())
    except Exception:
        pass

    try:
        if hasattr(output, "dict") and callable(getattr(output, "dict")):
            return bool(output.dict())
    except Exception:
        pass

    # Fallback: treat other types as truthy unless clearly empty / falsy
    return bool(output)


def _observation_to_mapping(obj: Any) -> Dict[str, Any]:
    """Best-effort conversion of an observation to a Python mapping (dict)."""
    # Prefer model_dump (Pydantic v2)
    try:
        if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
            data = obj.model_dump()
            if isinstance(data, dict):
                return data
    except Exception:
        pass

    # Fallback to dict() (Pydantic v1)
    try:
        if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
            data = obj.dict()
            if isinstance(data, dict):
                return data
    except Exception:
        pass

    # Fallback to __dict__
    if hasattr(obj, "__dict__"):
        try:
            data = dict(obj.__dict__)
            return data
        except Exception:
            pass

    # Last resort: json dumps/loads
    try:
        s = json.dumps(
            obj, default=lambda o: getattr(o, "__dict__", str(o)), ensure_ascii=False
        )
        maybe = json.loads(s)
        if isinstance(maybe, dict):
            return maybe
        return {"value": maybe}
    except Exception:
        return {"value": str(obj)}


def _observation_to_json(obj: Any, include_fields: Optional[List[str]] = None) -> str:
    """Serialize an observation object to a JSON string, including only selected fields if provided."""
    mapped = _observation_to_mapping(obj)
    if include_fields:
        mapped = {k: mapped[k] for k in include_fields if k in mapped}
    try:
        return json.dumps(mapped, ensure_ascii=False)
    except TypeError:
        # If something in mapped is not JSON-serializable, fallback to string for that field
        def safe_default(o: Any):
            try:
                return json.loads(
                    json.dumps(o, default=lambda x: getattr(x, "__dict__", str(x)))
                )
            except Exception:
                return str(o)

        return json.dumps(mapped, default=safe_default, ensure_ascii=False)


class ObservationExtractionPipeline:
    """Main pipeline for extracting and saving observations only"""

    def __init__(self, config: ObservationExtractionConfig):
        self.config = config
        self.extractor = LangfuseObservationExtractor(config)

    def run(self) -> bool:
        try:
            logger.info("🚀 Starting observation extraction pipeline...")

            observations = self.extractor.fetch_observations()

            # Filter by model if specified
            if self.config.filter_models:
                before = len(observations)
                observations = [
                    obs
                    for obs in observations
                    if filter_observation_by_model(obs, self.config.filter_models)
                ]
                logger.info(
                    f"Filtered observations by model: {before} -> {len(observations)}"
                )

            # Filter out observations with no output
            before_output = len(observations)
            observations = [obs for obs in observations if has_nonempty_output(obs)]
            logger.info(
                f"Filtered empty-output observations: {before_output} -> {len(observations)}"
            )

            if not observations:
                logger.error("❌ No observations to save after filtering")
                return False

            # Print fields once (post-exclusion view)
            try:
                first_mapped = _observation_to_mapping(observations[0])
                if self.config.include_fields:
                    first_mapped = {
                        k: first_mapped[k]
                        for k in self.config.include_fields
                        if k in first_mapped
                    }
                logger.info(
                    "Observation fields (first item, after inclusion): "
                    + ", ".join(sorted(first_mapped.keys()))
                )
            except Exception as e:
                logger.debug(f"Could not print observation fields: {e}")

            self._save_observations(observations)
            logger.info(f"🎉 Successfully saved {len(observations)} observations")
            return True
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return False

    def _save_observations(self, observations: List[Any]) -> None:
        """Save observations to JSONL"""
        try:
            output_path = self.config.output_file
            with open(output_path, "w", encoding="utf-8") as f:
                for obs in observations:
                    f.write(
                        _observation_to_json(obs, self.config.include_fields) + "\n"
                    )

            logger.info(f"💾 Saved observations to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save observations: {e}")
            raise


def load_config_from_env() -> ObservationExtractionConfig:
    """Load configuration from environment variables"""
    models_env = os.getenv("FILTER_MODELS", "").strip()
    models = [m for m in models_env.split(",") if m] if models_env else []
    include_env = os.getenv("include_fields", "").strip()
    include = [k for k in include_env.split(",") if k] if include_env else []

    return ObservationExtractionConfig(
        langfuse_host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
        langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
        langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
        output_file=os.getenv("OBSERVATIONS_OUTPUT_FILE", "observations.jsonl"),
        max_observations=int(os.getenv("MAX_OBSERVATIONS", "1000")),
        type_filter=os.getenv("OBSERVATION_TYPE", "GENERATION"),
        trace_id=os.getenv("TRACE_ID", None) or None,
        start_time=os.getenv("START_TIME", None) or None,
        stop_time=os.getenv("STOP_TIME", None) or None,
        filter_models=models,
        include_fields=include,
    )


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Extract observations from Langfuse")
    parser.add_argument(
        "--host", default="http://localhost:3000", help="Langfuse host URL"
    )
    parser.add_argument("--public-key", help="Langfuse public key")
    parser.add_argument("--secret-key", help="Langfuse secret key")
    parser.add_argument(
        "--output", default="observations.jsonl", help="Output file for observations"
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
        help="Filter observations by model name substring(s)",
    )
    parser.add_argument(
        "--include-fields",
        nargs="+",
        help="Fields to include in saved observations",
    )

    args = parser.parse_args()

    # Create configuration
    config = ObservationExtractionConfig(
        langfuse_host=args.host,
        langfuse_public_key=args.public_key or os.getenv("LANGFUSE_PUBLIC_KEY", ""),
        langfuse_secret_key=args.secret_key or os.getenv("LANGFUSE_SECRET_KEY", ""),
        output_file=args.output,
        max_observations=int(os.getenv("MAX_OBSERVATIONS", "1000")),
        type_filter=args.type_filter,
        trace_id=args.trace_id,
        start_time=args.start_time,
        stop_time=args.stop_time,
        filter_models=args.models,
        include_fields=args.include_fields,
    )

    # Validate required configuration
    if not config.langfuse_public_key or not config.langfuse_secret_key:
        logger.error(
            "❌ Missing Langfuse credentials. Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY environment variables or use --public-key and --secret-key arguments"
        )
        return 1

    # Run pipeline
    pipeline = ObservationExtractionPipeline(config)
    success = pipeline.run()

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
