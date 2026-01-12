# Rethinking RAI Perception

Last updated: Jan 2026

_How not to make the API so hard to use_

Perception is a critical component in robotics applications, enabling object detection, segmentation, 3D gripping point estimation, and point cloud processing for ROS2-based systems. Within the RAI Framework, `rai_perception` serves as an extension providing vision capabilities through tools (`GetDetectionTool`, `GetObjectGrippingPointsTool`), ROS2 service nodes (`DetectionService`, `SegmentationService`), and utilities that integrate with other RAI components (e.g., `rai_semap`, `rai_bench`).

As the codebase grows with recent work on [3D gripping point detection](https://github.com/RobotecAI/rai/pull/694) and new perception functionality in `rai_semap`, the current API design reveals significant usability challenges because most new APIs are designed for expert-level users, requiring deep understanding of algorithms, pipeline architecture, and domain-specific concepts. The configuration supporting these APIs is also complex. The existing APIs lack support for use cases where developers need to switch to different detection models.

This document explores a potential redesign approach that addresses some of these challenges by organizing code into a tiered API structure that supports progressive disclosure. The exploration draws on design principles from [`api_design_considerations.md`](../../../docs/api_design_considerations.md) and illustrates concepts with examples from refactored `rai_perception` implementation.

Rather than focusing solely on API surface changes, the exploration considers cognitive load and provides clear paths from simple, agent-friendly tools to configurable components to expert-level algorithms, balancing accessibility with flexibility. Another goal is to lay out the foundation for enabling developers to switch between different detection models without code changes (see the "Switching Between Detection Models" use case below).

---

## Table of Contents

-   [Current State Analysis](#current-state-analysis-as-of-jan-2026)
    -   [Audiences and Roles](#audiences-and-roles)
    -   [Usability Concerns](#usability-concerns)
-   [Proposed Design](#proposed-design)
    -   [Tiered API Structure](#tiered-api-structure)
    -   [Folder Structure](#folder-structure)
    -   [Usability Improvements Along Cognitive Dimensions](#usability-improvements-along-cognitive-dimensions)
        -   [Progressive Disclosure](#progressive-disclosure)
        -   [Configuration Management](#configuration-management)
        -   [Progressive Evaluation](#progressive-evaluation)
        -   [Penetrability](#penetrability)
        -   [Consistency](#consistency)
        -   [Domain Correspondence](#domain-correspondence)
        -   [Role Expressiveness](#role-expressiveness)
    -   [Why Agents are Renamed as Services](#why-agents-are-renamed-as-services)
    -   [Use Cases and Impact Evaluation](#use-cases-and-impact-evaluation)
        -   [Existing Use Cases](#existing-use-cases)
        -   [New Use Case: Switching Between Detection Models](#new-use-case-switching-between-detection-models)
        -   [Adding New Models to the Registry](#adding-new-models-to-the-registry)
-   [Good to Have (deferred)](#good-to-have-deferred)
-   [Future Work](#future-work)
    -   [Data Collection for Fine-Tuning](#data-collection-for-fine-tuning)
    -   [Observability](#observability)

---

## Current State Analysis (As of Jan 2026)

### Audiences and Roles

`rai_perception` serves four roles within the RAI Framework:

-   Application Developers (High-level): Use tools (`GetDetectionTool`, `GetObjectGrippingPointsTool`) in agents to build applications with minimal configuration
-   LLM agents (Runtime consumers): Consume perception tools at runtime via tool-calling mechanisms (tools implement LangChain `BaseTool` with `name`, `description`, `args_schema` for LLM understanding)
-   Extension Developers (Mid-level): Extend tools or create custom perception pipelines by working within existing framework layers
-   Core Developers (Low-level): Implement new vision services (extending `BaseVisionAgent`, which should be renamed to avoid confusion with RAI's agent concept) or low-level algorithms from scratch

### Usability Concerns

The multi-tier abstraction (high-level tools, mid-level components, low-level algorithms) reveals significant usability challenges:

1. Abstraction gap: Large jump from high-level tool usage (simple `object_name` parameter) to mid-level configuration (10+ algorithm parameters with domain-specific knowledge required)
2. Configuration complexity: `GetObjectGrippingPointsTool` requires 6+ ROS2 parameters (`target_frame`, `source_frame`, `camera_topic`, `depth_topic`, `camera_info_topic`, `timeout_sec`) plus 3 configuration objects with 10+ parameters each
3. Algorithm knowledge requirement: Mid-level users must understand ML/computer vision concepts (DBSCAN, RANSAC, Isolation Forest, percentiles) to configure effectively
4. Hidden dependencies: Tool initialization depends on ROS2 parameters that must be set before tool creation, no clear error if missing
5. Pipeline complexity: `GetObjectGrippingPointsTool` orchestrates a 3-stage pipeline (point cloud extraction → filtering → estimation) with no visibility into intermediate stages
6. Progressive evaluation difficulty: Cannot test individual pipeline stages, must run full pipeline to see results. Debug mode (`debug=True`) partially addresses this by publishing intermediate results to ROS2 topics for visualization in RVIZ.
7. Current error handling is inconsistent: tools catch `RaiTimeoutError` and return user-friendly messages, but generic `Exception` is re-raised without context. Parameter validation patterns vary across tools, and algorithm failures propagate as generic exceptions without actionable guidance.

---

## Proposed Design

The proposed design addresses the usability concerns identified above through a tiered API structure that supports progressive disclosure. This section outlines the architecture and explores how it improves usability along key cognitive dimensions.

### Tiered API Structure

The tiered API structure organizes code into three abstraction levels:

-   _High-level layer (Agent-friendly):_ Tools in `tools/` like `GetObjectGrippingPointsTool(object_name="cup")` and `GetDetectionTool(camera_topic="/camera/rgb", object_names=["cup"])` with minimal required arguments, hiding pipeline details.

-   _Mid-level layer (Configurable):_ Components in `components/` like `PointCloudFilter` and `GrippingPointEstimator` with Config classes (`PointCloudFilterConfig`, `GrippingPointEstimatorConfig`) that expose semantic parameters (e.g., `strategy="aggressive_outlier_removal"`, `outlier_fraction=0.05`) and support presets via `perception_presets.py`.

-   _Low-level layer (Expert control):_ Algorithms in `algorithms/` like `GDBoxer` and `GDSegmenter` that load their own configs and provide direct access to model inference and processing stages.

This tiered structure is reflected in the folder organization, where each abstraction level maps to a specific directory.

### Folder Structure

The folder structure supports progressive disclosure and reduces cognitive load by organizing code by abstraction level. Each folder maps to a specific audience and tier, making it easier to discover appropriate APIs and understand component relationships:

```
rai_core/rai/
└── communication/ros2/        # ROS2 communication infrastructure
    ├── parameters.py         # get_param_value() helper for extracting ROS2 parameter values
    └── exceptions.py         # ROS2ServiceError, ROS2ParameterError exception classes

rai_perception/
├── models/                    # Model registry and interfaces
├── tools/                     # High-level LLM agent tools (BaseTool instances)
├── components/                # Mid-level configurable components and inter-package APIs
├── algorithms/                # Low-level core algorithms
├── services/                  # Model-agnostic ROS2 service nodes
├── configs/                   # Configuration files
├── scripts/                   # Utility scripts
└── examples/                  # Example code

Deprecations, see later section for rationale
rai_perception/
├── agents/                    # Legacy ROS2 service nodes (deprecated, use services/)
└── vision_markup/             # Vision markup utilities (deprecated, use algorithms/)

```

This structure aligns with cognitive dimensions: abstraction level (tools → components → algorithms), penetrability (easy to find code by tier), domain correspondence (domain concepts visible within each tier), and role expressiveness (clear mapping to audiences). All configs are consolidated in `configs/` as user-facing runtime settings, reducing cognitive load from separating deployment vs algorithm configs.

### Usability Improvements Along Cognitive Dimensions

The following sections detail how the proposed design addresses usability concerns through improvements along key cognitive dimensions, with specific implementation examples from the refactored `rai_perception` codebase.

#### Progressive Disclosure

The tiered API structure spans three abstraction levels for progressive disclosure: agent-friendly tools in `tools/`, configurable components in `components/`, and expert-level algorithms in `algorithms/`. Each layer exposes only the knobs that each audience needs, letting agents issue intent-level commands, integrators adjust meaningful parameters, and experts directly control core implementations without wrestling with irrelevant details.

#### Configuration Management

The configuration system uses a multi-tier approach: algorithm configs (loaded from model registry), ROS2 parameters (deployment settings), component configs (Pydantic classes for algorithm tuning), and presets (semantic mappings for tools). Each tier serves a specific audience and abstraction level, reducing cognitive load by hiding irrelevant configuration details.

_Configuration Infrastructure Highlights:_

-   `rai.communication.ros2.get_param_value()`: Helper function for extracting ROS2 parameter values with automatic type conversion
-   `components/perception_presets.py`: Semantic presets ("default_grasp", "precise_grasp", "top_grasp") that map to component configs, enabling high-level tool API simplification

_Note on External Dependencies: Hydra Configuration System_

_Challenge:_ The `GDSegmenter` algorithm uses Hydra (required by SAM2) for configuration management. Hydra's config module discovery system requires proper package setup and can fail in ways that leak implementation details to high-level users, breaking API simplicity.

_Solution:_ For the default case (`config_path=None`), we use full file system paths instead of Hydra's config module discovery system. This preserves high-level API simplicity—application developers using tools don't encounter Hydra-specific errors or need to understand Hydra's package structure requirements. Advanced users can still use Hydra's module system for config composition and overrides when needed, maintaining progressive disclosure.

#### Progressive Evaluation

_Current limitation:_ Cannot test individual pipeline stages—must run full pipeline to see results. This makes debugging and incremental development difficult.

_Partial solution implemented:_ Debug mode (`debug=True`) added to `GetObjectGrippingPointsTool` publishes intermediate results to ROS2 topics and logs stage-level metadata, enabling visualization of pipeline stages in RVIZ. This allows users to inspect intermediate pipeline outputs without modifying code.

_Future improvements:_ Extend debug mode to other tools and consider exposing intermediate results as optional return values for programmatic access. This would enable progressive evaluation: users can test individual pipeline stages without running the full pipeline, supporting incremental debugging and validation.

#### Penetrability

Penetrability refers to the ease of exploring and understanding API components without reading implementation details. The tiered design improves penetrability through discoverable presets and self-documenting component relationships.

_Presets improve penetrability:_ The `perception_presets.py` module demonstrates how presets make component relationships discoverable. Users can call `list_presets()` to see available options (`["default_grasp", "precise_grasp", "top_grasp"]`), and preset names like `"precise_grasp"` are self-documenting—no need to read implementation to understand what they do. The module-level docstring explicitly documents the component pipeline flow (`PointCloudFromSegmentation → PointCloudFilter → GrippingPointEstimator`), making relationships discoverable without inspecting code.

High-level tools (`tools/`) use named presets over raw parameters. Instead of exposing all algorithm parameters, tools like `GetObjectGrippingPointsTool` should support semantic presets (e.g., `quality="precise_grasp"`, `approach="top_grasp"`) that internally map to appropriate component configurations.

#### Consistency

Consistency refers to how much can be inferred once part of the API is learned. The tiered design establishes consistent patterns that reduce the need to read implementation details.

_Consistent naming pattern:_ Input schema classes follow the pattern `{ToolName}Input` (e.g., `GetObjectGrippingPointsToolInput`, `GetDetectionToolInput`, `GetDistanceToObjectsInput`). Once users learn this pattern, they can infer all input schema names without looking them up.

_Consistent parameter handling:_ Tools use a standardized `_load_parameters()` method called in `model_post_init()` with parameter prefixes (e.g., `perception.gripping_points.*`, `perception.distance_to_objects.*`). Once users learn this pattern from one tool, they can infer how all tools handle ROS2 parameters—parameters are loaded at initialization with auto-declaration, type checking, and consistent error handling. This eliminates the need to read implementation details to understand parameter handling across different tools.

#### Domain Correspondence

Domain correspondence refers to how clearly API components map to the robotics domain. The tiered design uses semantic naming that reflects robotics concepts rather than algorithm implementation details.

_Semantic parameter names:_ Mid-level components (`components/`) use semantic parameter names. Configuration classes like `PointCloudFilterConfig` and `GrippingPointEstimatorConfig` expose parameters that describe outcomes (e.g., `outlier_fraction=0.05`) rather than algorithm names (e.g., `if_contamination=0.05`).

_Domain Correspondence Changes (2025):_ The `PointCloudFilterConfig` has been updated to use domain-oriented parameter names and strategy names:

-   Parameter names: Algorithm-specific names (`if_contamination`, `lof_n_neighbors`, `dbscan_eps`) replaced with semantic names (`outlier_fraction`, `neighborhood_size`, `cluster_radius_m`)
-   Strategy names: Algorithm names (`"isolation_forest"`, `"dbscan"`, `"lof"`) replaced with domain-oriented names (`"aggressive_outlier_removal"`, `"density_based"`, `"conservative_outlier_removal"`)

_Trade-off:_ The tiered API design attempts to balance both needs:

-   High-level tools: Use semantic presets (`"precise_grasp"`, `"default_grasp"`) that hide algorithm details
-   Mid-level components: Use semantic parameter names (`outlier_fraction` instead of `if_contamination`) but document algorithm mapping
-   Low-level algorithms: Direct algorithm access remains available for ML engineers who need full control

This approach improves domain correspondence (API components map clearly to robotics domain concepts) while maintaining access to algorithm-level control for expert users.

#### Role Expressiveness

Role expressiveness refers to how apparent the relationship between components and the program is. The tiered design makes pipeline stages and service dependencies explicit, improving understanding of component relationships.

_Role Expressiveness Improvements (2025):_ Tools have been enhanced to make pipeline/data flow and service dependencies explicit:

-   Pipeline visibility: Tools now expose `pipeline_stages` class attributes and `get_pipeline_info()` methods that document the internal pipeline stages (e.g., `GetObjectGrippingPointsTool` documents its 3-stage pipeline: Point Cloud Extraction → Point Cloud Filtering → Gripping Point Estimation). This makes it clear what stages a tool executes and helps users understand tool behavior and debug pipeline issues.

-   Service dependency clarity: Tools now expose `required_services` class attributes, `get_service_info()` methods, and `check_service_dependencies()` methods that document which ROS2 services are required and their current availability status. This makes it clear that tools depend on services (e.g., `GetDetectionTool` requires `DetectionService`), helps users understand deployment requirements, and provides better error messages when services are unavailable.

These improvements address the role expressiveness dimension by making the relationship between tools, components, and services apparent without requiring users to read implementation details.

_Future improvements:_ Results should include confidence and metadata. Tools should return confidence scores, strategy used, and alternative options to help LLM agents make better decisions about retrying or adjusting approaches.

#### Why Agents are Renamed as Services

The classes previously named "agents" (e.g., `GroundingDinoAgent`, `GroundedSamAgent`) are being renamed to "services" (e.g., `DetectionService`, `SegmentationService`) for two key reasons:

1. Abstraction Confusion: These classes were incorrectly named "agents" despite being ROS2 service nodes, not RAI agents. The RAI framework has a distinct `rai.agents.BaseAgent` abstraction for high-level agent orchestration that uses connectors and tools. Calling ROS2 service nodes "agents" creates confusion about the architecture and makes it unclear what these classes actually do.

2. ROS2-Specific Implementation: These classes are tightly coupled to ROS2 infrastructure—they create `ROS2Connector` instances, use ROS2 parameters (`rclpy.parameter.Parameter`), expose ROS2 services, and cannot work with other connector types. They are ROS2 service nodes, not abstracted agents that could work with different communication backends.

The new naming clarifies that these are ROS2 service nodes that provide vision capabilities, while real RAI agents (if needed) would use these services as tools/resources rather than inheriting from them.

### Use Cases and Impact Evaluation

This section evaluates how the proposed design impacts existing use cases and enables new capabilities.

#### Existing Use Cases

_1. Tools in RAI Agents_

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

_2. ROS2 Service Nodes_

-   `DetectionService`, `SegmentationService`: Model-agnostic ROS2 service nodes providing detection/segmentation services
-   Launched via `rai_perception.scripts.run_perception_services`
-   Legacy agents (`GroundedSamAgent`, `GroundingDinoAgent`) are deprecated and located in `agents/` folder

_3. Integration with Other RAI Components_

-   `rai_semap`: Uses `rai_perception.ros2.perception_utils.extract_pointcloud_from_bbox` for semantic mapping; detection results flow from `rai_perception` services into `rai_semap` for map annotation
-   `rai_bench`: Used for tool-calling agent evaluation and manipulation benchmarks

Impact of proposed design:

-   No breaking changes: Tools remain in `tools/`, import paths unchanged; `perception_utils.py` moves to `components/` but import path can be maintained via `__init__.py` or alias
-   Service dependencies: Both `rai_semap` and `rai_bench` may reference service names that changed to model-agnostic defaults, but defaults maintain compatibility
-   Model flexibility: Can test different models via ROS2 params without code changes

#### New Use Case: Switching Between Detection Models

The tiered API structure enables a new capability: switching detection models without code changes through the model registry pattern. This addresses the limitation identified in the current state where APIs lack support for model switching.

_Workflow:_

1. Check available models: `list_available_models()` in `models/detection.py` registry shows "grounding_dino", "yolo", etc.
2. Set ROS2 parameter: `/detection_service/model_name = "yolo"` (or desired model)
3. Service reads parameter and queries registry: `get_model("yolo")` returns `(AlgorithmClass, config_path)`
4. Service instantiates algorithm: `algorithm = AlgorithmClass(weights_path, config_path=config_path)`
5. Algorithm loads its own config (self-contained)
6. Tools read service name from parameter registry (default: "/detection")
7. Use tools normally—no code changes needed

_Remaining Issues:_

-   Error handling: If `model_name` doesn't match registry, only runtime error occurs. No validation at parameter declaration time.
-   Multiple instances: Running multiple models simultaneously requires multiple service instances with different service names, but no documented pattern exists.
-   Model registration: Adding a new model requires creating algorithm, registering in registry, and adding config—process is clear but not documented as workflow.

#### Adding New Models to the Registry

The tiered architecture supports extensibility by enabling new models to be added following the registry pattern. This section illustrates the process with concrete examples:

_YOLO (Detection Model):_

-   Design compatibility: Fits tiered architecture—add `YOLOBoxer` in `algorithms/`, register in `models/detection.py`. Follows same pattern as `GDBoxer`.
-   Key differences: Uses class IDs (closed vocabulary) instead of text prompts (open vocabulary). Config loading typically uses YAML or no config (vs Python config for GroundingDINO).
-   Implementation steps: Create `algorithms/yolo_boxer.py`, implement `get_boxes(image, class_ids, confidence_threshold)`, add COCO class ID mapping utility, register in registry, export in `algorithms/__init__.py`.

_Florence-2 (Unified Vision-Language Model):_

-   Design compatibility: Fits tiered architecture—add `Florence2Algorithm` in `algorithms/`. Can register in both detection and segmentation registries.
-   Key differences: Uses task prompts (`"OD"`, `"SEG"`) instead of task-specific models. Current separate services (`DetectionService`, `SegmentationService`) work but don't leverage unified model efficiently.
-   Implementation steps: Create `algorithms/florence2.py`, implement both `get_boxes()` and `get_segmentation()` methods, handle Hugging Face model loading, parse location tokens to bounding boxes/masks, register in both registries.
-   Future considerations: Capability-based registry (`capability="detection+segmentation"`) and unified service architecture would be more efficient for multi-task models.

---

## Good to Have (deferred)

## Good to Have (deferred)

_Config Utilities (`rai.config`):_

-   `load_yaml_config()`: Unified YAML loading with ROS2 parameter extraction (`{node_name}: ros__parameters: {...}`). Would reduce boilerplate in `detection_publisher.py` but current manual loading is readable and explicit.
-   `get_config_path()`: Standardizes ROS2 parameter path resolution with defaults. Would eliminate duplication but only used in one place currently.
-   `merge_nested_configs()`: Handles nested dict merging correctly vs. simple `.update()`. Useful if presets become nested; current flat presets work fine with manual updates.
-   Trade-off: Utilities reduce duplication but add abstraction layer. Current manual approach is explicit and maintainable for single usage. Main benefit is consistency if adopted across packages.

_Multi-Stage Pipeline Service Failures:_

-   Tools like `GetObjectGrippingPointsTool` call multiple services in sequence (detection → segmentation). Currently, failures don't indicate which pipeline stage failed.
-   Use case: Raise `ROS2ServiceError` with pipeline stage info (e.g., "Failed at detection stage", "Failed at segmentation stage") to enable stage-specific recovery strategies.
-   Benefit: Enables automatic retry at specific stages, provides clearer diagnostics for multi-stage operations, helps LLM agents understand partial failures.

---

## Future Work

### Data Collection for Fine-Tuning

Design compatibility:

-   No blocking points: Current tiered structure (tools → components → algorithms) provides clear instrumentation points at each level
-   ROS2 parameter helpers (`rai.communication.ros2.get_param_value()`) can be extended for data collection configuration
-   Component-based design allows adding collection hooks without breaking existing APIs
-   Consideration: Need to design collection API that works across all abstraction tiers without adding overhead to high-level tools

### Observability

Design compatibility:

-   No blocking points: Tiered structure naturally supports instrumentation at each level
-   Service layer (`services/`) provides centralized point for service-level metrics
-   Component abstraction allows adding observability decorators/wrappers without changing core logic
-   Parameter registry can include observability configuration (enable/disable, verbosity levels)
-   Consideration: Need to ensure observability doesn't leak into high-level tool APIs (keep tools simple for LLM agents)
-   Consideration: Design should support optional observability - not required for basic usage
