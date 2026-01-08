# Rethinking RAI Perception

_How not to make the API so damn hard to use_

Perception is a critical component in robotics applications, enabling object detection, segmentation, 3D gripping point estimation, and point cloud processing for ROS2-based systems. Within the RAI Framework, `rai_perception` serves as an extension providing vision capabilities through tools (`GetDetectionTool`, `GetObjectGrippingPointsTool`), ROS2 service nodes (`GroundedSamAgent`, `GroundingDinoAgent`), and utilities that integrate with other RAI components (e.g., `rai_semap`, `rai_bench`).

As the codebase grows with recent work on [3D gripping point detection](https://github.com/RobotecAI/rai/pull/694) and new perception functionality in `rai_semap`, the current API design reveals significant usability challenges. Most APIs are designed for expert-level users, requiring deep understanding of algorithms, pipeline architecture, and domain-specific concepts. When cognitive load kills usability, application developers and LLM agents are left struggling with interfaces that should be simple and intent-based.

This document proposes a redesign that addresses some of these challenges by organizing code into a tiered API structure that supports progressive disclosure. Rather than focusing solely on API surface changes, the redesign considers cognitive load and provides clear paths from simple, agent-friendly tools to configurable components to expert-level algorithms. The goal is to make perception capabilities accessible to all users while maintaining the flexibility needed for advanced use cases.

---

## Table of Contents

-   [Current State Analysis](#current-state-analysis)
    -   [Usage Patterns](#usage-patterns)
    -   [Audiences and Roles](#audiences-and-roles)
    -   [Abstraction Levels and Complexity](#abstraction-levels-and-complexity)
    -   [Usability Concerns](#usability-concerns)
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
-   Core Developers (Low-level): Implement new vision agents (extending `BaseVisionAgent`) or low-level algorithms from scratch

### Abstraction Levels and Complexity

Multi-tier abstraction with significant complexity:

**High-level (Application Developer / LLM Agent):**

-   Simple tools: `GetDetectionTool(camera_topic="/camera/rgb", object_names=["cup"])` - intent-based API
-   Example: `GetObjectGrippingPointsTool(object_name="box")` - minimal required input, configuration via ROS2 parameters
-   Abstraction: Hides detection pipeline, point cloud processing, filtering algorithms

**Mid-level (Extension Developer):**

-   Configuration classes: `PointCloudFilterConfig`, `GrippingPointEstimatorConfig`, `PointCloudFromSegmentationConfig` - expose 10+ algorithm parameters
-   Example: Must understand filtering strategies (`dbscan`, `isolation_forest`, `kmeans_largest_cluster`, `lof`), RANSAC parameters, percentile thresholds
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
6. **Parameter discovery**: Configuration options scattered across multiple config classes—no single source of truth for all parameters
7. **Error messages**: Algorithm-specific errors (e.g., RANSAC failures, filtering edge cases) may not provide actionable guidance
8. **Domain correspondence gap**: Parameter names like `if_contamination`, `lof_n_neighbors` don't clearly map to perception domain concepts
9. **Code duplication**: Significant duplication exists across tools at the infrastructure level:

    - Service client creation (`_call_gdino_node`, `_call_gsam_node`) duplicated across multiple files
    - Image message retrieval (`_get_image_message`) duplicated in three locations
    - Parameter retrieval with defaults duplicated for `conversion_ratio` and `outlier_sigma_threshold`
    - Camera intrinsics extraction duplicated in `segmentation_tools.py` and `perception_utils.py`
    - Future result retrieval helpers duplicated in `segmentation_tools.py`

    This creates maintenance burden, inconsistent error handling, and potential bugs when fixes aren't applied uniformly. The duplication exists below the documented abstraction layers, representing infrastructure-level concerns not addressed by the current architecture.

> **Note**: `GetSegmentationTool` and `GetGrabbingPointTool` exist but are not documented in Usage Patterns. `GetSegmentationTool` does not inherit from `BaseTool`, conflicting with the claim that tools implement LangChain `BaseTool`.

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
├── tools/                    # High-level: LLM agent tools (BaseTool instances)
│   ├── gdino_tools.py        # GetDetectionTool, GetDistanceToObjectsTool
│   │                         # TODO: Read service_name from ROS2 param /detection_tool/service_name
│   │                         # (default: "/detection") instead of hardcoding GDINO_SERVICE_NAME
│   ├── pcl_detection_tools.py # GetObjectGrippingPointsTool
│   └── segmentation_tools.py  # GetSegmentationTool, GetGrabbingPointTool (deprecated)
│                             # TODO: Read service_name from ROS2 params instead of hardcoding
│
├── components/                # Mid-level: Configurable components & inter-package APIs
│   ├── perception_utils.py   # Inter-package APIs: extract_pointcloud_from_bbox, compute_3d_pose_from_bbox, enhance_detection_with_3d_pose
│   ├── detection_publisher.py # DetectionPublisher (ROS2 node)
│   ├── parameters.py         # ParameterRegistry: ROS2 parameter documentation/discovery
│   │                         # Lists all ROS2 params: name, type, default, description, valid values
│   │                         # Provides: get_param_info(name), list_all_params(), validate_param(name, value)
│   │                         # TODO: Implement parameter registry for discoverability and validation
│   └── pcl_detection.py      # PointCloudFromSegmentation, PointCloudFilter, GrippingPointEstimator + Config classes
│
├── algorithms/                # Low-level: Core algorithms
│   ├── boxer.py              # GDBoxer (detection algorithm)
│   │                         # Algorithm loads its own config from config_path provided by registry
│   │                         # Example: GDBoxer(weights_path, config_path="configs/gdino_config.py")
│   ├── segmenter.py          # GDSegmenter (segmentation algorithm)
│   │                         # Algorithm loads its own config (self-contained)
│   └── point_cloud.py        # depth_to_point_cloud (extract from pcl_detection.py or segmentation_tools.py)
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

**Implementation steps:**

1. Model Registry: Implement `models/detection.py` and `models/segmentation.py` with capability-based registries
2. Parameter Registry: Implement `components/parameters.py` with `ParameterRegistry` class for ROS2 parameter documentation, discovery, and validation
3. Model-Agnostic Services: Refactor `grounding_dino.py` and `grounded_sam.py` to `detection_agent.py` and `segmentation_agent.py` that read model from ROS2 params
4. Tool Updates: Update tools to read service_name from ROS2 params (via parameter registry) instead of hardcoding
5. Configuration Loading: Algorithms load their own configs from paths provided by registry (self-contained approach)
6. Migration Path: Keep existing `grounding_dino.py`/`grounded_sam.py` as wrappers for backward compatibility during transition

### API Analysis and Recommendations

The current API design has several strengths: a two-tier configuration system separates deployment settings (ROS2 parameters for topics, frames, timeouts) from algorithm parameters (Pydantic models for thresholds, strategies), and the strategy pattern provides flexibility with multiple filtering (`dbscan`, `isolation_forest`) and estimation strategies (`centroid`, `top_plane`, `ransac_plane`) without requiring API changes.

However, the API has significant complexity concerns for LLM agents. It exposes too many low-level configuration knobs (e.g., `if_contamination`, `dbscan_eps`, `dbscan_min_samples`), creating a combinatorial explosion of choices that agents struggle with. The pipeline abstraction also leaks—tools expose intermediate stages (`PointCloudFromSegmentation`, `GrippingPointEstimator`, `point_cloud_filter`), requiring users to understand the full pipeline structure. Additionally, the boundary between ROS2 parameters and Pydantic models is unclear, making it difficult to know when to use each.

The tiered API structure addresses these concerns through:

1. High-level tools (`tools/`) should use named presets over raw parameters. Instead of exposing all algorithm parameters, tools like `GetObjectGrippingPointsTool` should support semantic presets (e.g., `quality="high"`, `approach="top_down"`) that internally map to appropriate component configurations.

2. Mid-level components (`components/`) should use semantic parameter names. Configuration classes like `PointCloudFilterConfig` and `GrippingPointEstimatorConfig` should expose parameters that describe outcomes (e.g., `noise_handling="aggressive"`) rather than algorithm names (e.g., `strategy="isolation_forest"`).

3. Results should include confidence and metadata. Tools should return confidence scores, strategy used, and alternative options to help LLM agents make better decisions about retrying or adjusting approaches.

This approach maintains the flexible low-level API in `algorithms/` for power users while providing progressive disclosure: simple for agents, powerful for experts.

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
3. Check `components/parameters.py` for required ROS2 params
4. Set ROS2 param: `/detection_agent/model_name = "yolo"` (or desired model)
5. Service reads param, queries registry: `get_model("yolo")` → returns (AlgorithmClass, config_path)
6. Service instantiates: `algorithm = AlgorithmClass(weights_path, config_path=config_path)`
7. Algorithm loads its own config (self-contained)
8. Tools read `/detection_tool/service_name` from parameter registry (default: "/detection")
9. Use tool normally - no code changes needed

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
-   Parameter registry (`components/parameters.py`) can be extended to include data collection configuration
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

---
