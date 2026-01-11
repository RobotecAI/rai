# Rethinking RAI Perception

_How not to make the API so hard to use_

Perception is a critical component in robotics applications, enabling object detection, segmentation, 3D gripping point estimation, and point cloud processing for ROS2-based systems. Within the RAI Framework, `rai_perception` serves as an extension providing vision capabilities through tools (`GetDetectionTool`, `GetObjectGrippingPointsTool`), ROS2 service nodes (`GroundedSamAgent`, `GroundingDinoAgent`), and utilities that integrate with other RAI components (e.g., `rai_semap`, `rai_bench`).

As of January 2026, as the codebase grows with recent work on [3D gripping point detection](https://github.com/RobotecAI/rai/pull/694) and new perception functionality in `rai_semap`, the current API design reveals significant usability challenges because most new APIs are designed for expert-level users, requiring deep understanding of algorithms, pipeline architecture, and domain-specific concepts. The configuration supporting these APIs is also complex. The existing APIs lack support for use cases where developers need to switch to different detection models.

This document proposes a redesign that addresses some of these challenges by organizing code into a tiered API structure that supports progressive disclosure. Rather than focusing solely on API surface changes, the redesign considers cognitive load and provides clear paths from simple, agent-friendly tools to configurable components to expert-level algorithms. The goal is to make perception capabilities accessible to all users while maintaining the flexibility needed for advanced use cases.

---

## Table of Contents

-   [Current State Analysis](#current-state-analysis)
    -   [Usage Patterns](#usage-patterns)
    -   [Audiences and Roles](#audiences-and-roles)
    -   [Abstraction Levels and Complexity](#abstraction-levels-and-complexity)
    -   [Usability Concerns](#usability-concerns)
    -   [Error Handling Analysis](#error-handling-analysis)
-   [Proposed Design](#proposed-design)
    -   [Tiered API Structure](#tiered-api-structure)
    -   [Folder Structure](#folder-structure)
    -   [API Analysis and Recommendations](#api-analysis-and-recommendations)
    -   [Use Cases and Impact](#use-cases-and-impact)
    -   [Validation Scenarios](#validation-scenarios)
-   [Future Work](#future-work)
    -   [Data Collection for Fine-Tuning](#data-collection-for-fine-tuning)
    -   [Observability](#observability)

---

## Current State Analysis

### Usage Patterns

1. Tools in RAI Agents:

    - `GetDetectionTool`: Used in manipulation demos (`examples/manipulation-demo-v2.py`, `examples/rosbot-xl-demo.py`) for object detection
    - `GetObjectGrippingPointsTool`: Used in manipulation scenarios for 3D gripping point estimation
    - `GetDistanceToObjectsTool`: Used for distance calculations to detected objects

2. ROS2 Service Nodes:

    - `GroundedSamAgent`, `GroundingDinoAgent`: Standalone ROS2 service nodes providing detection/segmentation services
    - Launched via `rai_bringup/launch/openset.launch.py` and `rai_perception.scripts.run_perception_agents`
    - Note: The "agent" terminology here is a breakage from the RAI agents abstraction and should be renamed (e.g., to "service" or "node") to reduce confusion with RAI's agent concept.

3. Integration with rai_semap:

    - `rai_semap` uses `rai_perception.ros2.perception_utils.extract_pointcloud_from_bbox` for semantic mapping
    - Detection results flow from `rai_perception` services into `rai_semap` for map annotation

4. Benchmarking:
    - Used in `rai_bench` for tool-calling agent evaluation and manipulation benchmarks

### Audiences and Roles

`rai_perception` serves four identified roles within the RAI Framework:

-   Application Developers (High-level): Use tools (`GetDetectionTool`, `GetObjectGrippingPointsTool`) in agents to build applications with minimal configuration
-   LLM agents (Runtime consumers): Consume perception tools at runtime via tool-calling mechanisms (tools implement LangChain `BaseTool` with `name`, `description`, `args_schema` for LLM understanding)
-   Extension Developers (Mid-level): Extend tools or create custom perception pipelines by working within existing framework layers
-   Core Developers (Low-level): Implement new vision services (extending `BaseVisionAgent`, which should be renamed to avoid confusion with RAI's agent concept) or low-level algorithms from scratch

### Abstraction Levels and Complexity

Multi-tier abstraction with significant complexity:

**High-level (Application Developer / LLM Agent):**

-   Simple tools: `GetDetectionTool(camera_topic="/camera/rgb", object_names=["cup"])` - intent-based API
-   Example: `GetObjectGrippingPointsTool(object_name="box")` - minimal required input, configuration via ROS2 parameters
-   Abstraction: Hides detection pipeline, point cloud processing, filtering algorithms

**Mid-level (Extension Developer):**

-   Configuration classes: `PointCloudFilterConfig`, `GrippingPointEstimatorConfig`, `PointCloudFromSegmentationConfig` - expose 10+ algorithm parameters
-   Example: Must understand filtering strategies (`density_based`, `aggressive_outlier_removal`, `cluster_based`, `conservative_outlier_removal` - Note: Algorithm-specific names like `dbscan`, `isolation_forest` have been replaced with domain-oriented names in 2025), RANSAC parameters, percentile thresholds
-   Abstraction: Exposes algorithm selection and tuning but hides implementation details

**Low-level (Core Developer / Algorithm Expert):**

-   Core algorithms: `depth_to_point_cloud()`, `GrippingPointEstimator`, `PointCloudFilter` - direct algorithm implementation
-   Example: Must implement filtering strategies, plane detection, outlier removal from scratch
-   Abstraction: Minimal - all algorithm logic must be understood and implemented

> **Abstraction Gap Concern**: Large jump from high-level tool usage (simple `object_name` parameter) to mid-level configuration (10+ algorithm parameters with domain-specific knowledge required).

### Usability Concerns

1. **Configuration complexity**: `GetObjectGrippingPointsTool` requires 6+ ROS2 parameters (`target_frame`, `source_frame`, `camera_topic`, `depth_topic`, `camera_info_topic`, `timeout_sec`) plus 3 configuration objects with 10+ parameters each
2. **Algorithm knowledge requirement**: Mid-level users must understand ML/computer vision concepts (DBSCAN, RANSAC, Isolation Forest, percentiles) to configure effectively
3. **Hidden dependencies**: Tool initialization depends on ROS2 parameters that must be set before tool creation—no clear error if missing
4. **Pipeline complexity**: `GetObjectGrippingPointsTool` orchestrates 4-stage pipeline (detection → segmentation → point cloud → filtering → estimation) with no visibility into intermediate stages
5. **Progressive evaluation difficulty**: Cannot test individual pipeline stages—must run full pipeline to see results
    - Partial solution: Debug mode (`debug=True`) added to `GetObjectGrippingPointsTool` publishes intermediate results to ROS2 topics and logs stage-level metadata, enabling visualization of pipeline stages in RVIZ. Next step: Extend debug mode to other tools and consider exposing intermediate results as optional return values for programmatic access.
6. **Parameter discovery**: Configuration options scattered across multiple config classes—no single source of truth for all parameters
7. **Error messages**: Algorithm-specific errors (e.g., RANSAC failures, filtering edge cases) may not provide actionable guidance
8. **Domain correspondence gap**: Parameter names like `if_contamination`, `lof_n_neighbors` don't clearly map to perception domain concepts (Note: This has been addressed in 2025 by replacing algorithm-specific names with semantic names like `outlier_fraction` and `neighborhood_size` - see API Analysis section for details)
9. **Code duplication**: Significant duplication exists across tools at the infrastructure level:

    - Service client creation (`_call_gdino_node`, `_call_gsam_node`) duplicated across multiple files
    - Image message retrieval (`_get_image_message`) duplicated in three locations
    - Parameter retrieval with defaults duplicated for `conversion_ratio` and `outlier_sigma_threshold`
    - Camera intrinsics extraction duplicated in `segmentation_tools.py` and `perception_utils.py`
    - Future result retrieval helpers duplicated in `segmentation_tools.py`

    This creates maintenance burden, inconsistent error handling, and potential bugs when fixes aren't applied uniformly. The duplication exists below the documented abstraction layers, representing infrastructure-level concerns not addressed by the current architecture.

> **Note**: `GetSegmentationTool` and `GetGrabbingPointTool` exist but are not documented in Usage Patterns. `GetSegmentationTool` does not inherit from `BaseTool`, conflicting with the claim that tools implement LangChain `BaseTool`.

### Error Handling Analysis

**Current State:**

1. **Tool-level error handling**: Tools catch `RaiTimeoutError` and return user-friendly messages (e.g., `gripping_points_tools.py` line 195-197). Generic `Exception` is re-raised without context (line 198-199).

2. **Parameter validation**: Inconsistent patterns:

    - `gripping_points_tools.py`: Raises `ValueError` with clear message if required ROS2 params missing (line 116-118)
    - `gdino_tools.py`: Catches `ParameterUninitializedException`, logs warning, uses defaults (line 231-235)
    - `segmentation_tools.py`: Generic `Exception("Received wrong message")` (line 94, 221)

3. **Service call errors**: Tools return string error messages instead of raising exceptions (e.g., `gdino_tools.py` line 172: "Service call failed. Can't get detections.").

4. **Algorithm errors**: Low-level algorithm failures (RANSAC, filtering) propagate as generic exceptions without actionable context.

5. **RAI framework patterns**:
    - `RaiTimeoutError`: Custom exception for timeouts (used in `gripping_points_tools.py`)
    - `ToolRunner`: Catches `ValidationError` (Pydantic) and generic `Exception`, returns `ToolMessage` with `status="error"` (rai_core)
    - `BaseVisionAgent`: Handles model loading errors with automatic retry for corrupted weights

**What to Keep:**

1. **Custom timeout exception**: `RaiTimeoutError` provides clear timeout semantics and is handled gracefully in tools.

2. **Parameter validation with defaults**: Pattern of catching `ParameterUninitializedException`/`ParameterNotDeclaredException` and using sensible defaults (e.g., `gdino_tools.py` line 231-235) is appropriate for optional parameters.

3. **Tool-level error recovery**: Tools catching exceptions and returning user-friendly messages (rather than crashing) enables LLM agents to handle errors gracefully.

4. **Pydantic validation**: Using Pydantic `BaseModel` for input validation provides automatic error messages for invalid inputs.

**What to Change:**

1. **Standardize exception types**: Replace generic `Exception` with domain-specific exceptions that provide structured error information:

    - `ROS2ServiceError`: Service unavailable, call failed, timeout (general ROS2 error, reusable across framework)
        - **Additional info**: Service name, timeout duration, service state (exists/ready/unavailable), underlying error type, retry suggestions
        - **Value over generic Exception**: Enables error-specific handling (retry vs fail-fast), provides actionable recovery suggestions, includes service diagnostics
    - `ROS2ParameterError`: Missing/invalid ROS2 parameters (general ROS2 error, reusable across framework)
        - **Additional info**: Parameter name, expected type/value, where to set it (launch file, param file), related parameters, default value if available
        - **Value over generic Exception**: Enables early validation at initialization, provides setup guidance, helps diagnose configuration issues
    - `PerceptionAlgorithmError`: Algorithm-specific failures (RANSAC, filtering) - domain-specific
        - **Additional info**: Algorithm stage, input data characteristics, alternative strategies, parameter suggestions
        - **Value over generic Exception**: Enables automatic strategy switching, provides algorithm-specific recovery paths
    - `PerceptionValidationError`: Input validation failures beyond Pydantic - domain-specific
        - **Additional info**: Validation rule violated, input value, valid range/options, context about why validation failed
        - **Value over generic Exception**: Enables targeted validation feedback, helps LLM agents understand constraints

2. **Improve error messages for LLM agents**: Error messages should be:

    - Actionable: Suggest fixes (e.g., "Try with quality='high'")
    - Contextual: Include what failed and why
    - Structured: Return error details in tool response format

3. **Early validation**: Move parameter validation to tool initialization (like `gripping_points_tools.py`) rather than runtime checks with generic exceptions.

4. **Pipeline stage visibility**: Add intermediate error reporting for multi-stage pipelines (detection → segmentation → filtering → estimation) so failures can be diagnosed.

5. **Error recovery suggestions**: Tools should suggest alternative approaches when algorithms fail (e.g., "RANSAC plane fitting failed. Try using strategy='centroid' instead of 'top_plane'").

6. **Consistent error handling pattern**: Standardize error handling across all tools:
    - Required params: Raise `ROS2ParameterError` at initialization
    - Optional params: Use defaults with warning logs
    - Service calls: Raise `ROS2ServiceError` with retry suggestions
    - Algorithm failures: Raise `PerceptionAlgorithmError` with strategy suggestions
    - Timeouts: Catch `RaiTimeoutError`, return user-friendly message

**Implementation Plan:**

1. Exception hierarchy implemented:

    - `ROS2ServiceError` and `ROS2ParameterError` defined in `rai_core/rai/communication/ros2/exceptions.py` for framework-wide reuse
    - `PerceptionError`, `PerceptionAlgorithmError`, `PerceptionValidationError` defined in `rai_perception/components/exceptions.py`:

    ```python
    # In rai_core/rai/communication/ros2/exceptions.py (for framework-wide use):
    class ROS2ServiceError(Exception):
        def __init__(self, service_name: str, timeout_sec: float,
                     service_state: str = None, suggestion: str = None,
                     underlying_error: Exception = None):
            self.service_name = service_name
            self.timeout_sec = timeout_sec
            self.service_state = service_state  # "exists", "ready", "unavailable"
            self.suggestion = suggestion  # "Check if service is running", "Try increasing timeout"
            self.underlying_error = underlying_error
            super().__init__(f"Service {service_name} error: {service_state or 'unavailable'}")

    class ROS2ParameterError(Exception):
        def __init__(self, param_name: str, expected_type: str = None,
                     expected_value: str = None, suggestion: str = None,
                     default_value: Any = None):
            self.param_name = param_name
            self.expected_type = expected_type
            self.expected_value = expected_value
            self.suggestion = suggestion  # "Set in launch file", "Check config YAML"
            self.default_value = default_value
            super().__init__(f"Parameter {param_name} error: {suggestion or 'missing or invalid'}")

    # Export in rai_core/rai/communication/ros2/__init__.py:
    # from .exceptions import ROS2ServiceError, ROS2ParameterError

    # In rai_perception/components/exceptions.py:
    class PerceptionError(Exception): ...

    class PerceptionAlgorithmError(PerceptionError):
        def __init__(self, algorithm_stage: str, strategy: str = None,
                     suggestion: str = None, input_info: dict = None):
            self.algorithm_stage = algorithm_stage  # "ransac", "filtering", "estimation"
            self.strategy = strategy  # "top_plane", "isolation_forest"
            self.suggestion = suggestion  # "Try strategy='centroid'"
            self.input_info = input_info  # {"point_count": 100, "noise_level": "high"}
            super().__init__(f"Algorithm error at {algorithm_stage}: {suggestion or 'failed'}")

    class PerceptionValidationError(PerceptionError):
        def __init__(self, validation_rule: str, input_value: Any,
                     valid_range: str = None, suggestion: str = None):
            self.validation_rule = validation_rule
            self.input_value = input_value
            self.valid_range = valid_range
            self.suggestion = suggestion
            super().__init__(f"Validation failed: {validation_rule}")
    ```

2. Update all tools to use custom exceptions and provide actionable error messages.

3. Add error context to tool responses (error type, suggestion, retry strategy).

4. Implement pipeline stage error reporting for multi-stage operations.

---

## Proposed Design

### Tiered API Structure

The tiered API structure organizes code into three abstraction levels:

**High-level layer (Agent-friendly):** Simple, intent-based tools in `tools/` with minimal required arguments. For example, `GetObjectGrippingPointsTool(object_name="cup")` only requires the object name—camera topics, filter configs, and estimation strategies are handled via ROS2 parameters or sensible defaults. The agent doesn't need to understand the pipeline or choose between `isolation_forest` vs `dbscan` strategies.

**Mid-level layer (Configurable):** Configurable components in `components/` that expose key parameters for tuning behavior. For example, `PointCloudFilter` and `GrippingPointEstimator` with their Config classes (`PointCloudFilterConfig`, `GrippingPointEstimatorConfig`) allow users to configure filtering strategies and estimation methods without implementing algorithms from scratch.

**Low-level layer (Expert control):** Core algorithms in `algorithms/` providing direct access to model inference and processing stages. For example, `GDBoxer` (detection algorithm), `GDSegmenter` (segmentation algorithm), and `depth_to_point_cloud` for users who need full control over every parameter.

### Folder Structure

The folder structure organizes code by abstraction level to support progressive disclosure and reduce cognitive load. Each folder maps to a specific audience and abstraction tier, making it easier to discover appropriate APIs and understand component relationships.

```
rai_core/rai/
├── config/                      # General configuration utilities
│   ├── loader.py               # Unified YAML/Python config loading
│   └── merger.py               # Config merging with precedence: defaults → ROS2 params → overrides
│
└── communication/ros2/           # ROS2 communication infrastructure
    └── parameters.py            # get_param_value() helper for extracting ROS2 parameter values

rai_perception/
├── models/                    # Domain abstraction layer (model registry/interfaces)
│   ├── detection.py          # DetectionModel registry, base classes
│   │                         # Maps: "grounding_dino" → (AlgorithmClass, config_path)
│   │                         # Registry provides config_path; algorithm loads its own config
│   │                         # TODO: Implement registry with get_model(name), list_available_models()
│   └── segmentation.py       # SegmentationModel registry, base classes
│                             # Maps: "grounded_sam" → (AlgorithmClass, config_path)
│                             # Combined models register in both with capability="detection+segmentation"
│                             # TODO: Implement capability-based registration
│
├── tools/                     # High-level: LLM agent tools (BaseTool instances)
│   ├── gdino_tools.py        # GetDetectionTool, GetDistanceToObjectsTool
│   │                         # TODO: Read service_name from ROS2 param /detection_tool/service_name
│   │                         # (default: "/detection") instead of hardcoding GDINO_SERVICE_NAME
│   ├── gripping_points_tools.py # GetObjectGrippingPointsTool
│   └── segmentation_tools.py  # GetSegmentationTool, GetGrabbingPointTool (deprecated)
│                             # TODO: Read service_name from ROS2 params instead of hardcoding
│
├── components/                # Mid-level: Configurable components & inter-package APIs
│   ├── perception_utils.py   # 3D pose computation and point cloud extraction from 2D detections
│   │                         # Functions: compute_3d_pose_from_bbox(), extract_pointcloud_from_bbox(), enhance_detection_with_3d_pose()
│   ├── detection_publisher.py # ROS2 node: subscribes to camera images, calls detection service, publishes detections
│   ├── exceptions.py         # Perception-specific exceptions with rich error metadata for LLM agents
│   │                         # Exceptions: PerceptionError, PerceptionAlgorithmError, PerceptionValidationError
│   ├── perception_presets.py # Semantic preset definitions: "default_grasp", "precise_grasp", "top_grasp"
│   │                         # API: get_preset(), apply_preset(), list_presets()
│   │                         # Note: General config utilities (loader, merger, get_param_value) are in rai_core
│   └── gripping_points.py    # Point cloud processing components: PointCloudFromSegmentation, PointCloudFilter, GrippingPointEstimator
│                             # Strategy pattern: dbscan, isolation_forest, centroid, top_plane, biggest_plane
│
├── algorithms/                # Low-level: Core algorithms
│   ├── boxer.py              # GDBoxer (detection algorithm)
│   │                         # Algorithm loads its own config from config_path provided by registry
│   │                         # Example: GDBoxer(weights_path, config_path="configs/gdino_config.py")
│   ├── segmenter.py          # GDSegmenter (segmentation algorithm)
│   │                         # Algorithm loads its own config (self-contained)
│   └── point_cloud.py        # depth_to_point_cloud (extract from gripping_points.py or segmentation_tools.py)
│
├── services/                  # ROS2 service nodes (agents)
│   ├── base_vision_agent.py  # BaseVisionAgent
│   ├── detection_agent.py    # Model-agnostic DetectionAgent
│   │                         # Reads ROS2 param: /detection_agent/model_name (default: "grounding_dino")
│   │                         # Exposes service at: /detection_agent/service_name (default: "/detection")
│   │                         # TODO: Refactor grounding_dino.py to use this pattern
│   └── segmentation_agent.py # Model-agnostic SegmentationAgent
│                             # Reads ROS2 param: /segmentation_agent/model_name (default: "grounded_sam")
│                             # Exposes service at: /segmentation_agent/service_name (default: "/segmentation")
│                             # TODO: Refactor grounded_sam.py to use this pattern
│
├── configs/                   # Configuration files (user-facing runtime settings)
│   ├── detection_publisher.yaml
│   ├── perception_utils.yaml
│   ├── seg_config.yml
│   └── gdino_config.py
│                             # Algorithm configs loaded by algorithms themselves (self-contained)
│                             # Registry points to config_path; algorithm handles loading
│
├── scripts/                   # Utility scripts
│   └── run_perception_agents.py
│
└── examples/                  # Example code
    └── talker.py
```

This structure aligns with cognitive dimensions—abstraction level (tools → components → algorithms), penetrability (easy to find code by tier), domain correspondence (domain concepts visible within each tier), and role expressiveness (clear mapping to audiences). All configs are consolidated in `configs/` as user-facing runtime settings, reducing cognitive load from separating deployment vs algorithm configs.

**Configuration Management Flow:**

The configuration system follows a multi-tier approach with clear separation of concerns:

1. **Algorithm Configs (Low-level)**: Algorithms load their own configs from `config_path` provided by model registry. Configs are self-contained (e.g., `gdino_config.py`). Flow: `models/registry` → `config_path` → `algorithms/boxer.py` loads config.

2. **ROS2 Parameters (Deployment)**: ROS2 parameters handle deployment settings (topics, frames, timeouts, service names). Retrieved via `rai.communication.ros2.get_param_value()`. Flow: `configs/*.yaml` → ROS2 param system → components/services.

3. **Component Configs (Mid-level)**: Pydantic Config classes (`PointCloudFilterConfig`, `GrippingPointEstimatorConfig`) handle algorithm parameters. Defined in `components/gripping_points.py`, instantiated from ROS2 params or defaults.

4. **Presets (High-level)**: Semantic presets map user-friendly names to component configs. Tools use presets internally. Flow: `tools/*.py` → `components/perception_presets.py` → `rai.config.merger` → component configs.

5. **Config Merging**: `rai.config.merger` (in `rai_core`) handles precedence: defaults → ROS2 params → user overrides. Ensures consistent configuration resolution.

**Configuration Infrastructure Status:**

-   `rai.config.loader` (in `rai_core/rai/config/`): Unified YAML/Python config loading implemented. Replaces manual YAML loading in nodes. Handles ROS2 config pattern: `{node_name}: ros__parameters: {...}`.

-   `rai.config.merger` (in `rai_core/rai/config/`): Config merging logic implemented. Merges with precedence: defaults → ROS2 params → user overrides. Supports nested configs.

-   `rai.communication.ros2.get_param_value()` (in `rai_core/rai/communication/ros2/`): Helper function for extracting ROS2 parameter values with automatic type conversion.

-   `components/perception_presets.py`: Semantic presets implemented. Provides presets: "default_grasp", "precise_grasp", "top_grasp" that map to component configs. Required for high-level tool API simplification.

**Implementation steps:**

1. Model Registry: Implement `models/detection.py` and `models/segmentation.py` with capability-based registries
2. Configuration Infrastructure: ✅ Implemented:
    - `rai.config.loader` (in `rai_core/rai/config/`): Unified YAML/Python config loading to replace manual loading in nodes
    - `rai.config.merger` (in `rai_core/rai/config/`): Config merging with precedence (defaults → ROS2 params → overrides)
    - `components/perception_presets.py`: Semantic presets for high-level tools (quality="precise_grasp", approach="top_grasp")
    - `components/exceptions.py`: Perception-specific exceptions with rich metadata
3. Component Migration: ✅ Partially implemented, needs migration:
    - `components/exceptions.py`: ✅ Implemented
    - `components/perception_presets.py`: ✅ Implemented
    - `components/perception_utils.py`: ✅ Implemented (currently in `ros2/perception_utils.py`, needs move)
    - `components/detection_publisher.py`: ✅ Implemented (currently in `ros2/detection_publisher.py`, needs move)
    - `components/gripping_points.py`: ✅ Implemented (previously `pcl_detection.py`)
    - TODO: Migrate files to `components/` directory and update imports
4. Model-Agnostic Services: Refactor `grounding_dino.py` and `grounded_sam.py` to `detection_agent.py` and `segmentation_agent.py` that read model from ROS2 params
5. Tool Updates: Update tools to read service_name from ROS2 params (via parameter registry) instead of hardcoding, and use presets for semantic configuration
6. Configuration Loading: Algorithms load their own configs from paths provided by registry (self-contained approach)
7. Migration Path: Keep existing `grounding_dino.py`/`grounded_sam.py` as wrappers for backward compatibility during transition

### API Analysis and Recommendations

The current API design has several strengths: a two-tier configuration system separates deployment settings (ROS2 parameters for topics, frames, timeouts) from algorithm parameters (Pydantic models for thresholds, strategies), and the strategy pattern provides flexibility with multiple filtering (`dbscan`, `isolation_forest`) and estimation strategies (`centroid`, `top_plane`, `ransac_plane`) without requiring API changes.

However, the API has significant complexity concerns for LLM agents. It exposes too many low-level configuration knobs (e.g., `if_contamination`, `dbscan_eps`, `dbscan_min_samples` - Note: These have been replaced with semantic names like `outlier_fraction`, `cluster_radius_m`, `min_cluster_size` in 2025), creating a combinatorial explosion of choices that agents struggle with. The pipeline abstraction also leaks—tools expose intermediate stages (`PointCloudFromSegmentation`, `GrippingPointEstimator`, `point_cloud_filter`), requiring users to understand the full pipeline structure. Additionally, the boundary between ROS2 parameters and Pydantic models is unclear, making it difficult to know when to use each.

The tiered API structure addresses these concerns through:

1. High-level tools (`tools/`) should use named presets over raw parameters. Instead of exposing all algorithm parameters, tools like `GetObjectGrippingPointsTool` should support semantic presets (e.g., `quality="precise_grasp"`, `approach="top_grasp"`) that internally map to appropriate component configurations.

    **Example - Presets improve penetrability**: The `perception_presets.py` module demonstrates how presets make component relationships discoverable. Users can call `list_presets()` to see available options (`["default_grasp", "precise_grasp", "top_grasp"]`), and preset names like `"precise_grasp"` are self-documenting—no need to read implementation to understand what they do. The module-level docstring explicitly documents the component pipeline flow (`PointCloudFromSegmentation → PointCloudFilter → GrippingPointEstimator`), making relationships discoverable without inspecting code. This addresses the penetrability dimension: users can explore and understand API components without reading implementation details.

    **Example - Consistent naming improves consistency**: Input schema classes follow the pattern `{ToolName}Input` (e.g., `GetObjectGrippingPointsToolInput`, `GetDetectionToolInput`, `GetDistanceToObjectsInput`). Once users learn this pattern, they can infer all input schema names without looking them up. This addresses the consistency dimension: users can apply knowledge from one part of the API to understand other parts.

    **Example - Consistent parameter handling improves consistency**: Tools use a standardized `_load_parameters()` method called in `model_post_init()` with parameter prefixes (e.g., `perception.gripping_points.*`, `perception.distance_to_objects.*`). Once users learn this pattern from one tool, they can infer how all tools handle ROS2 parameters—parameters are loaded at initialization with auto-declaration, type checking, and consistent error handling. This eliminates the need to read implementation details to understand parameter handling across different tools.

2. Mid-level components (`components/`) should use semantic parameter names. Configuration classes like `PointCloudFilterConfig` and `GrippingPointEstimatorConfig` should expose parameters that describe outcomes (e.g., `noise_handling="aggressive"`) rather than algorithm names (e.g., `strategy="isolation_forest"`).

    **Note on Domain Correspondence Changes (2025):** The `PointCloudFilterConfig` has been updated to use domain-oriented parameter names and strategy names:

    - **Parameter names**: Algorithm-specific names (`if_contamination`, `lof_n_neighbors`, `dbscan_eps`) replaced with semantic names (`outlier_fraction`, `neighborhood_size`, `cluster_radius_m`)
    - **Strategy names**: Algorithm names (`"isolation_forest"`, `"dbscan"`, `"lof"`) replaced with domain-oriented names (`"aggressive_outlier_removal"`, `"density_based"`, `"conservative_outlier_removal"`)

    **Potential Controversy**: These changes may be controversial because ML-specific terminology (e.g., `"isolation_forest"`, `if_contamination`) can be more useful to ML engineers who understand the underlying algorithms and want direct control over algorithm parameters. However, the semantic naming approach prioritizes application developers who need to configure perception pipelines without deep ML knowledge. The mapping from semantic names to algorithms is documented in code comments, allowing ML engineers to understand which algorithm is used while keeping the API accessible to non-ML developers.

    **Trade-off**: The tiered API design attempts to balance both needs:

    - **High-level tools**: Use semantic presets (`"precise_grasp"`, `"default_grasp"`) that hide algorithm details
    - **Mid-level components**: Use semantic parameter names (`outlier_fraction` instead of `if_contamination`) but document algorithm mapping
    - **Low-level algorithms**: Direct algorithm access remains available for ML engineers who need full control

    This approach improves domain correspondence (API components map clearly to robotics domain concepts) while maintaining access to algorithm-level control for expert users.

    **Note on Role Expressiveness Improvements (2025):** Tools have been enhanced to make pipeline/data flow and service dependencies explicit:

    - **Pipeline visibility**: Tools now expose `pipeline_stages` class attributes and `get_pipeline_info()` methods that document the internal pipeline stages (e.g., `GetObjectGrippingPointsTool` documents its 3-stage pipeline: Point Cloud Extraction → Point Cloud Filtering → Gripping Point Estimation). This makes it clear what stages a tool executes and helps users understand tool behavior and debug pipeline issues.

    - **Service dependency clarity**: Tools now expose `required_services` class attributes, `get_service_info()` methods, and `check_service_dependencies()` methods that document which ROS2 services are required and their current availability status. This makes it clear that tools depend on services (e.g., `GetDetectionTool` requires `DetectionService`), helps users understand deployment requirements, and provides better error messages when services are unavailable.

    These improvements address the role expressiveness dimension by making the relationship between tools, components, and services apparent without requiring users to read implementation details.

3. Results should include confidence and metadata. Tools should return confidence scores, strategy used, and alternative options to help LLM agents make better decisions about retrying or adjusting approaches.

This approach maintains the flexible low-level API in `algorithms/` for power users while providing progressive disclosure: simple for agents, powerful for experts.

### Renaming Agents to Services

The classes previously named "agents" (e.g., `GroundingDinoAgent`, `GroundedSamAgent`) are being renamed to "services" (e.g., `DetectionService`, `SegmentationService`) for two key reasons:

1. **Abstraction Confusion**: These classes were incorrectly named "agents" despite being ROS2 service nodes, not RAI agents. The RAI framework has a distinct `rai.agents.BaseAgent` abstraction for high-level agent orchestration that uses connectors and tools. Calling ROS2 service nodes "agents" creates confusion about the architecture and makes it unclear what these classes actually do.

2. **ROS2-Specific Implementation**: These classes are tightly coupled to ROS2 infrastructure—they create `ROS2Connector` instances, use ROS2 parameters (`rclpy.parameter.Parameter`), expose ROS2 services, and cannot work with other connector types. They are ROS2 service nodes, not abstracted agents that could work with different communication backends.

The new naming clarifies that these are ROS2 service nodes that provide vision capabilities, while real RAI agents (if needed) would use these services as tools/resources rather than inheriting from them.

### Use Cases and Impact

#### Existing Use Cases

**1. Tools in RAI Agents**

Tools used:

-   `GetDetectionTool`: Used in manipulation demos (`examples/manipulation-demo-v2.py`, `examples/rosbot-xl-demo.py`) for object detection
-   `GetObjectGrippingPointsTool`: Used in manipulation scenarios for 3D gripping point estimation
-   `GetDistanceToObjectsTool`: Used for distance calculations to detected objects

Impact of proposed design:

-   Positive: No breaking changes for `GetDetectionTool`/`GetDistanceToObjectsTool` - tools read service_name from ROS2 params (backward compatible with defaults)
-   Positive: `GetObjectGrippingPointsTool` requires no changes - already uses ROS2 params for configuration, config objects remain in `components/`
-   Positive: Model flexibility - can switch detection models via ROS2 param without code changes
-   Issue: Service name changes - examples hardcode service names in `wait_for_ros2_services()` calls (e.g., `"/grounding_dino_classify"`). Need to update to use parameter registry or configurable service names
-   Issue: Backward compatibility - existing launch files/configs assume specific service names. Migration path needed
-   Issue: Tool initialization - tools must read ROS2 params at initialization. If params not set, need sensible defaults matching current behavior

Migration strategy:

-   Tools default to current service names (`"/grounding_dino_classify"`, `"/grounded_sam_segment"`) if params not set
-   Examples can continue working without changes
-   New deployments can use model-agnostic service names via params

**2. ROS2 Service Nodes**

-   `GroundedSamAgent`, `GroundingDinoAgent`: Standalone ROS2 service nodes providing detection/segmentation services
-   Launched via `rai_bringup/launch/openset.launch.py` and `rai_perception.scripts.run_perception_agents`

Impact of proposed design:

-   Breaking change: Services refactored to `detection_agent.py` and `segmentation_agent.py` (model-agnostic)
-   Migration needed: `run_perception_agents.py` imports `GroundedSamAgent`, `GroundingDinoAgent` - must update to use new service classes or wrappers
-   Launch files: May need updates if they reference specific agent classes
-   Backward compatibility: Keep `grounding_dino.py`/`grounded_sam.py` as wrappers during transition

**3. Integration with rai_semap**

-   `rai_semap` uses `rai_perception.ros2.perception_utils.extract_pointcloud_from_bbox` for semantic mapping
-   Detection results flow from `rai_perception` services into `rai_semap` for map annotation

Impact of proposed design:

-   No breaking change: `perception_utils.py` moves to `components/` but import path can be maintained via `__init__.py` or alias (`from rai_perception.components import perception_utils` or keep `rai_perception.ros2.perception_utils` as alias)
-   Service integration: `rai_semap` subscribes to detection topics/services - service names may change if using model-agnostic approach
-   Potential issue: If service names change, `rai_semap` configs may need updates (but defaults maintain current names)

**4. Benchmarking**

-   Used in `rai_bench` for tool-calling agent evaluation and manipulation benchmarks

Impact of proposed design:

-   No breaking change: Tools remain in `tools/`, import paths unchanged
-   Service dependencies: Benchmarks wait for specific service names (e.g., `"/grounding_dino_classify"`) - may need updates if service names change, but defaults maintain compatibility
-   Model flexibility: Can test different models via ROS2 params without code changes

#### New Use Cases

**Using a Different Detection Model**

Workflow:

1. Developer wants different detection model → Check `models/detection.py` registry
2. Find available models: `list_available_models()` shows "grounding_dino", "yolo", etc.
3. Set ROS2 param: `/detection_agent/model_name = "yolo"` (or desired model)
4. Service reads param, queries registry: `get_model("yolo")` → returns (AlgorithmClass, config_path)
5. Service instantiates: `algorithm = AlgorithmClass(weights_path, config_path=config_path)`
6. Algorithm loads its own config (self-contained)
7. Tools read `/detection_tool/service_name` from parameter registry (default: "/detection")
8. Use tool normally - no code changes needed

Remaining issues:

1. Error handling: If `model_name` param doesn't match registry, runtime error only. No validation at param declaration time.
2. Multiple instances: Running multiple models simultaneously unclear. Need multiple service instances with different service names, but no documented pattern.
3. Model registration: Adding a new model requires creating algorithm, registering in registry, adding config - process is clear but not documented as workflow.

### Validation Scenarios

The tiered API structure should be validated through the following scenarios:

**High-Level Tier (`tools/`):**

-   Zero-shot agent usage: LLM agents can use tools like `GetObjectGrippingPointsTool(object_name="cup")` without examples or parameter tuning
-   Error recovery: Tools return actionable error messages with suggestions (e.g., "Try with quality='high'")
-   Common task success: 80%+ of manipulation tasks solvable with high-level API alone, 1-2 tool calls per task

**Mid-Level Tier (`components/`):**

-   Edge case handling: Components like `PointCloudFilter` and `GrippingPointEstimator` handle non-standard objects with semantic parameters (e.g., `noise_handling="aggressive"`)
-   Strategy comparison: Users can experiment with different approaches without understanding low-level algorithms
-   Environment adaptation: Same API works across different robot/camera setups via configuration

**Low-Level Tier (`algorithms/`):**

-   Custom pipelines: Users can inject custom algorithms and compose pipeline stages independently
-   Algorithm reproduction: Every stage is accessible and configurable for research use cases

**Tier Transitions:**

-   Gradual complexity: Users can progress from simple to advanced without rewriting code
-   API discoverability: Docstrings and type hints guide users to appropriate tiers

**Performance:**

-   Abstraction overhead: High-level tools should have <5% overhead compared to direct algorithm usage

**Success Metrics:**

-   90% of new users succeed with high-level tier
-   15% of power users utilize low-level tier
-   > 85% LLM agent success rate
-   Natural progression between tiers over time

---

## Future Work

### Data Collection for Fine-Tuning

Considerations:

1. What to collect: input images, tool calls, expected vs actual outputs, error messages
2. Where to collect: tool level? service level? both?
3. Data format: What format for perception failure data? How does it map to fine-tuning format (ChatML)?
4. Storage: Where to store collected data? File system? Database? Integration with rai_finetune?
5. Filtering: How to filter useful failures vs noise? Need criteria for what constitutes "failure data"
6. Privacy: Image data collection - need mechanisms for handling sensitive data
7. Integration: How does this integrate with existing rai_finetune data extraction pipelines?

Remaining issues:

1. Collection point: Unclear where to instrument - tools? services? both? Need clear collection API
2. Data format: What format for perception failure data? How does it map to fine-tuning format (ChatML)?
3. Storage location: Where to store collected data? File system? Database? Integration with rai_finetune?
4. Selective collection: How to filter useful failures vs noise? Need criteria for what constitutes "failure data"
5. Privacy/sensitivity: Image data collection - need mechanisms for handling sensitive data
6. Integration: How does this integrate with existing rai_finetune data extraction pipelines?

Design compatibility:

-   No blocking points: Current tiered structure (tools → components → algorithms) provides clear instrumentation points at each level
-   ROS2 parameter helpers (`rai.communication.ros2.get_param_value()`) can be extended for data collection configuration
-   Component-based design allows adding collection hooks without breaking existing APIs
-   Consideration: Need to design collection API that works across all abstraction tiers without adding overhead to high-level tools

### Observability

Considerations:

1. What to observe: tool call latency, success/failure rates, detection accuracy, service health
2. Instrumentation points: tools? services? algorithms? all levels?
3. Observability data: metrics, logs, traces, or all?
4. Integration: existing RAI tracing (Langfuse, LangSmith) or separate system?
5. Visualization: where to view collected data?

Remaining issues:

1. Instrumentation points: Unclear where to add observability - tools? services? algorithms? Need consistent pattern across abstraction levels
2. Data collection: What metrics to collect? (latency, accuracy, failure rates, model performance) Need standardized metric definitions
3. Integration: How to integrate with existing RAI tracing infrastructure (Langfuse/LangSmith)? Or separate observability system?
4. Performance overhead: Observability adds overhead - need configurable levels (minimal vs detailed)
5. Data storage: Where to store observability data? Separate from application logs? Integration with existing logging?
6. OpenTelemetry integration: Future direction - standard observability protocol. Need to design for potential OpenTelemetry integration without breaking current approach

Design compatibility:

-   No blocking points: Tiered structure naturally supports instrumentation at each level
-   Service layer (`services/`) provides centralized point for service-level metrics
-   Component abstraction allows adding observability decorators/wrappers without changing core logic
-   Parameter registry can include observability configuration (enable/disable, verbosity levels)
-   Consideration: Need to ensure observability doesn't leak into high-level tool APIs (keep tools simple for LLM agents)
-   Consideration: Design should support optional observability - not required for basic usage

### Adding New Models: YOLO and Florence-2

**YOLO (Detection Model):**

Design Compatibility:

-   Fits tiered architecture: Add `YOLOBoxer` in `algorithms/`, register in `models/detection.py`
-   No blocking points: Follows same pattern as `GDBoxer`
-   API difference: Uses class IDs (closed vocabulary) instead of text prompts (open vocabulary)
-   Config loading: Typically YAML or no config (vs Python config for GroundingDINO)

Work Needed:

1. Create `algorithms/yolo_boxer.py` with `YOLOBoxer` class
2. Implement `get_boxes(image, class_ids, confidence_threshold)` - different signature than GroundingDINO
3. Add class ID mapping utility (COCO class names → IDs)
4. Register in `models/detection.py` registry
5. Update `algorithms/__init__.py` to export `YOLOBoxer`
6. Document model-specific config loading (YOLO typically doesn't need config file)

**Florence-2 (Unified Vision-Language Model):**

Design Compatibility:

-   Fits tiered architecture: Add `Florence2Algorithm` in `algorithms/`
-   No blocking points: Can register in both detection and segmentation registries
-   Service architecture: Current separate services (`DetectionService`, `SegmentationService`) work but don't leverage unified model efficiently
-   Task abstraction: Florence-2 uses task prompts (`"OD"`, `"SEG"`) - different from current task-specific models

Work Needed:

1. Create `algorithms/florence2.py` with `Florence2Algorithm` class
2. Implement both `get_boxes()` and `get_segmentation()` methods (same class, different task prompts)
3. Handle Hugging Face model loading (no config file needed)
4. Parse location tokens to bounding boxes/masks (Florence-2 output format)
5. Register in both `models/detection.py` and `models/segmentation.py` registries
6. Optional optimization: Consider unified service or capability-based registry for future multi-task models

Future Considerations:

-   Capability-based registry: Design doc mentions `capability="detection+segmentation"` - Florence-2 is good use case
-   Unified service: Current architecture works but unified service would be more efficient for multi-task models
-   Task abstraction: Consider `Task` interface/enum for models that support multiple tasks

### External Dependencies: Hydra Configuration System

**Current State:**

The `GDSegmenter` algorithm uses Hydra for configuration management. Hydra is an external dependency (from Facebook Research) that provides configuration composition, overrides, and validation. The SAM2 model library (`sam2`) requires Hydra for loading model configurations.

**Design Decision:**

For the default case (`config_path=None`), we use full file system paths instead of Hydra's config module discovery system. This decision was made to:

1. **Preserve high-level API simplicity**: Application developers using tools should not encounter Hydra-specific errors or need to understand Hydra's package structure requirements.

2. **Reliability**: Full path loading avoids Hydra's config module discovery mechanism, which requires proper package setup (`__init__.py` configuration, correct file extensions, etc.) and can fail in ways that leak implementation details to high-level users.

3. **Progressive disclosure**: Default configs work simply (full path), while advanced users can still use Hydra's module system for config composition and overrides when needed.

**Trade-offs:**

-   **Pros**: Simple, reliable defaults; no abstraction leakage; works out-of-the-box
-   **Cons**: Bypasses Hydra's module system for defaults; less flexible for config composition/overrides in default case; path-dependent (may break if package structure changes)

**Future Considerations:**

1. **Hydra package setup**: If we want to leverage Hydra's full capabilities (config composition, overrides), we would need to properly set up `rai_perception.configs` as a Hydra config package with appropriate `__init__.py` and file structure.

2. **Alternative approaches**: Consider whether SAM2's Hydra dependency is necessary for our use case, or if we can load configs directly and pass structured configs to SAM2.

3. **Abstraction layer**: The algorithm layer (low-level) currently exposes Hydra as an implementation detail. Consider wrapping Hydra initialization in a config loader abstraction that hides Hydra from higher layers.

4. **Documentation**: Users who want to customize configs should understand when they're using Hydra's module system vs file system paths, and what capabilities each approach provides.

**Design Principle Alignment:**

This approach aligns with the tiered API design: high-level users get simple, reliable defaults without understanding Hydra, while mid-level users can still access Hydra's advanced features when needed. The key is that implementation details (Hydra) don't leak to the high-level API.

---
