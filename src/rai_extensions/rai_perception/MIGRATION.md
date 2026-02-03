# Migration Guide

PR [#750 feat: redesign rai_perception API with tiered structure and improve 3D gripping point detection](https://github.com/RobotecAI/rai/pull/750) has introduced several breaking changes, including class deprecation and service name changes. This guide provides step-by-step instructions for migrating to the new codebase.

At a high level, the migration can be a gradual process. Several backward compatibility measures have been introduced to reduce the impact on existing demos and applications.

**Quick Start:** If you're migrating existing code, start with the [Breaking Changes](#breaking-changes) section, then review [Backward Compatibility Measures](#backward-compatibility-measures) to understand your migration options. The [Post-PR Follow-ups](#post-pr-follow-ups) section outlines future improvements and deprecation timelines.

## Table of Contents

-   [Breaking Changes](#breaking-changes)
    -   [Agent Service Renaming and Service Name Changes](#agent-service-renaming-and-service-name-changes)
    -   [Agents and Vision Markup Deprecation](#agents-and-vision-markup-deprecation)
    -   [Related Package Updates](#related-package-updates)
    -   [Backward Compatibility Measures](#backward-compatibility-measures)
-   [Post-PR Follow-ups](#post-pr-follow-ups)
    -   [Deprecation Timeline for Legacy Classes and Modules](#deprecation-timeline-for-legacy-classes-and-modules)
    -   [Service Name Updates in rai_bench](#service-name-updates-in-rai_bench)
    -   [Preset Selection for GetObjectGrippingPointsTool](#preset-selection-for-getobjectgrippingpointstool)
    -   [GetObjectPositionsTool Name Collision Warning](#getobjectpositionstool-name-collision-warning)
-   [Additional Issues Found But Not Directly Related to PR](#additional-issues-found-but-not-directly-related-to-pr)
    -   [Architectural Improvements](#architectural-improvements)
        -   [Generic Detection Tools Abstraction](#generic-detection-tools-abstraction)
        -   [Progressive Evaluation: GetDistanceToObjectsTool](#progressive-evaluation-getdistancetoobjectstool)

---

## Breaking Changes

### Agent Service Renaming and Service Name Changes

The refactoring introduces model-agnostic naming for both classes and service names, moving away from model-specific terminology.

_Class Renaming:_

Old classes (deprecated):

-   `GroundingDinoAgent` → `DetectionService`
-   `GroundedSamAgent` → `SegmentationService`
-   `BaseVisionAgent` → `BaseVisionService`

_Rationale:_ These classes were named "agents" despite being ROS2 service nodes. The RAI framework has a distinct `rai.agents.BaseAgent` abstraction for high-level agent orchestration. Calling ROS2 service nodes "agents" may create confusion about the architecture.

_Service Name Changes:_

Old defaults:

-   Detection service: `"/grounding_dino_classify"` → `"/detection"`
-   Segmentation service: `"/grounded_sam_segment"` → `"/segmentation"`

_Current delegation implementation:_ The deprecated agent classes (`GroundingDinoAgent`, `GroundedSamAgent`) are now thin compatibility wrappers that delegate to the new service classes. They emit deprecation warnings and will be removed in a future version. The delegation is implemented via the `create_service_wrapper()` helper function in `agents/_helpers.py`.

_Impact:_ Code that hardcodes the old class names or service names will need updates. Tools now read service names from ROS2 parameters (defaulting to the new names), making the system model-agnostic. For backward compatibility during migration, the `enable_legacy_service_names` flag can be used to register both old and new service names simultaneously (see [Legacy Service Name Handling](#legacy-service-name-handling) for details).

_Migration:_

-   Update code to use `DetectionService` and `SegmentationService` directly
-   For launch files, use `rai_perception.scripts.run_perception_services` instead of `run_perception_agents`
-   Update launch files and configuration to use the new service names, or set ROS2 parameters to override defaults

### Agents and Vision Markup Deprecation

The following modules are deprecated:

-   `rai_perception.agents` - Legacy ROS2 service nodes (deprecated, use `services/`)
-   `rai_perception.vision_markup` - Vision markup utilities (deprecated, use `algorithms/`)

These modules remain in the codebase for backward compatibility but will be removed in a future version. New code should use:

-   `services/` for ROS2 service nodes (`DetectionService`, `SegmentationService`)
-   `algorithms/` for core vision algorithms

### Related Package Updates

The following packages require updates to align with the new service naming:

-   **`rai_bench`**: Update hardcoded service names in configuration files and mocked interfaces (see [Service Name Updates in rai_bench](#service-name-updates-in-rai_bench) for details)
-   **Examples**: Update manipulation demos to use new service names
-   **`rai_sim`**: May require similar service name updates (to be verified)

**Note:** For detailed migration steps for `rai_bench`, see the [Service Name Updates in rai_bench](#service-name-updates-in-rai_bench) section in Post-PR Follow-ups.

### Backward Compatibility Measures

#### Legacy Service Name Handling

_Status:_ Implemented with backward compatibility support.

_Overview:_ The perception services support both legacy and new service names simultaneously to maintain backward compatibility during the migration period.

_Configuration:_

The `enable_legacy_service_names` parameter controls whether legacy service names are registered alongside the new generic names:

-   **Default:** `true` (for backward compatibility)
-   **Environment Variable:** `ENABLE_LEGACY_SERVICE_NAMES` (set to `"true"` or `"false"`)
-   **ROS2 Parameter:** `enable_legacy_service_names` (boolean)
-   **Launch File:** `src/rai_bringup/launch/openset.launch.py` declares this as a launch argument

The parameter is read from the environment variable first, then falls back to the ROS2 parameter if not set.

_Service Name Registration:_

When `enable_legacy_service_names=true`:

-   Detection service registers both: `/detection` (new) and `/grounding_dino_classify` (legacy)
-   Segmentation service registers both: `/segmentation` (new) and `/grounded_sam_segment` (legacy)

When `enable_legacy_service_names=false`:

-   Only new service names are registered: `/detection` and `/segmentation`

_Agent Version Compatibility:_

-   **v1 agents** (legacy tools): Use legacy service names (`/grounding_dino_classify`, `/grounded_sam_segment`)
-   **v2 agents** (new tools): Use new service names (`/detection`, `/segmentation`)

For applications that need to support both v1 and v2 agents (e.g., `manipulation-demo-streamlit.py`), set `enable_legacy_service_names=true` in the launch file to ensure both naming schemes are available.

_Migration Path:_

1. **Phase 1 (Current):** Legacy names enabled by default, both v1 and v2 work
2. **Phase 2 (Future):** Legacy names disabled by default, only v2 works
3. **Phase 3 (Future):** Legacy names removed entirely

## Post-PR Follow-ups

The following items are planned improvements and follow-up work identified during the migration. Items are organized by priority, with urgent migration tasks listed first.

### Deprecation Timeline for Legacy Classes and Modules

_Status:_ Planned deprecation schedule.

_Overview:_ The following deprecated classes and modules will be removed in a future release:

**Deprecated Classes:**

-   `GroundingDinoAgent` → Use `DetectionService`
-   `GroundedSamAgent` → Use `SegmentationService`
-   `BaseVisionAgent` → Use `BaseVisionService`

**Deprecated Modules:**

-   `rai_perception.agents` → Use `rai_perception.services`
-   `rai_perception.vision_markup` → Use `rai_perception.algorithms`

**Deprecation Schedule:**

-   **Current (Phase 1):** Deprecated classes/modules emit warnings but remain functional
-   **Future Release (Phase 2):** Deprecated classes/modules will be removed from the codebase
-   **Action Required:** Migrate to new classes and modules before Phase 2

_Note:_ The exact release date for Phase 2 will be announced in a future release. Users are encouraged to migrate as soon as possible to avoid breaking changes.

### Service Name Updates in rai_bench

_Status:_ Changes required for compatibility.

_Issue:_ `rai_bench` package uses hardcoded service names that need to be updated to match the generic service names used in `rai_perception` tools.

_Changes Required:_

1. **`src/rai_bench/rai_bench/manipulation_o3de/predefined/configs/o3de_config.yaml`**

    - Update `required_robotic_ros2_interfaces.services`:
        - `/grounding_dino_classify` → `/detection`
        - `/grounded_sam_segment` → `/segmentation`

2. **`src/rai_bench/rai_bench/tool_calling_agent/mocked_ros2_interfaces.py`**

    - Update `CUSTOM_SERVICES_AND_TYPES` dictionary:
        - `/grounding_dino_classify` → `/detection`
        - `/grounded_sam_segment` → `/segmentation`

3. **`src/rai_bench/rai_bench/tool_calling_agent/tasks/custom_interfaces.py`**
    - Update constants:
        - `GROUNDING_DINO_SERVICE = "/grounding_dino_classify"` → `GROUNDING_DINO_SERVICE = "/detection"`
        - `GROUNDED_SAM_SERVICE = "/grounded_sam_segment"` → `GROUNDED_SAM_SERVICE = "/segmentation"`

_Rationale:_ These changes align `rai_bench` with the model-agnostic service naming convention used in `rai_perception`. Tools now read service names from ROS2 parameters (defaulting to `/detection` and `/segmentation`), making the system model-agnostic and allowing easy switching between detection/segmentation models without code changes.

### Preset Selection for GetObjectGrippingPointsTool

_Status:_ Feature request.

_Issue:_ Currently, users must manually apply presets (e.g., `default_grasp`, `top_grasp`, `precise_grasp`) by calling `apply_preset()` and passing configs to the tool constructor. This requires code changes and prevents runtime configuration.

_Current Workflow:_

```python
from rai_perception.components.perception_presets import apply_preset

filter_config, estimator_config = apply_preset("top_grasp")
tool = GetObjectGrippingPointsTool(
    connector=connector,
    filter_config=filter_config,
    estimator_config=estimator_config
)
```

_Preliminary Thinking:_

1. **Constructor Parameter:** Add optional `preset` parameter to `GetObjectGrippingPointsTool`:

    ```python
    tool = GetObjectGrippingPointsTool(
        connector=connector,
        preset="top_grasp"  # Simple preset selection
    )
    ```

2. **ROS2 Parameter Support:** Add `perception.gripping_points.preset` parameter for runtime configuration:

    ```python
    node.declare_parameter("perception.gripping_points.preset", "default_grasp")
    ```

3. **Priority Logic:** If both `preset` and explicit `filter_config`/`estimator_config` are provided, explicit configs take precedence (or raise error for clarity).

_Benefits:_

-   Simplifies preset selection (no need to call `apply_preset()` manually)
-   Enables runtime configuration via ROS2 parameters
-   Maintains backward compatibility (preset is optional, defaults to `"default_grasp"`)
-   Allows easy switching between presets without code changes

_Estimated Effort:_ Small to medium (30 minutes - 2 hours depending on scope)

### GetObjectPositionsTool Name Collision Warning

_Status:_ Implemented with warning mechanism.

_Issue:_ The deprecated `GetObjectPositionsTool` in `rai.tools.ros2.manipulation.custom` and the new `GetObjectPositionsTool` in `rai_perception.tools` both use the same tool name `"get_object_positions"`. If both tools are registered in the same agent, the newer tool will silently overwrite the deprecated one, which could cause unexpected behavior.

_Current Implementation:_

The deprecated `GetObjectPositionsTool` in `src/rai_core/rai/tools/ros2/manipulation/custom.py` includes a warning mechanism:

-   When the deprecated tool is initialized, it checks if `rai_perception` is installed
-   If the new tool class is available, it emits a warning that both tools exist and may conflict
-   The warning advises users to migrate to the new tool from `rai_perception`

_Impact:_

-   Both tools can coexist for backward compatibility
-   Users will see warnings when using the deprecated tool if `rai_perception` is installed
-   If both tools are registered, the one registered last will be used (typically the new one)

_Migration:_

-   Update code to use `GetObjectPositionsTool` from `rai_perception.tools` instead of `rai.tools.ros2.manipulation.custom`
-   Remove the deprecated tool from tool lists once migration is complete
-   The deprecated tool will be removed in a future version

_Note:_ The deprecated tool remains exported from `rai.tools.ros2` for backward compatibility but should not be used in new code.

## Additional Issues Found But Not Directly Related to PR

The following issues were identified during the migration but are not directly related to PR #750. These are organized into architectural improvements and operational issues.

### Architectural Improvements

#### Generic Detection Tools Abstraction

_Problem:_ Several modules are tightly coupled to GroundingDINO model and service interface:

-   `tools/gdino_tools.py`: Hardcoded `RAIGroundingDino` service type, model-specific parameters (`box_threshold`, `text_threshold`) as class fields, direct parsing of `RAIGroundingDino.Response` structure
-   `components/gripping_points.py`: Uses `RAIGroundingDino` service type, `box_threshold`/`text_threshold` in `PointCloudFromSegmentationConfig`, DINO-specific method names (`_call_gdino_node`)
-   `tools/segmentation_tools.py`: Uses `RAIGroundingDino` service type, `box_threshold`/`text_threshold` parameters, DINO-specific method names

This prevents easy migration to other models (e.g., YOLO) without code changes. The coupling includes service types, parameter names, and response structures that are all GroundingDINO-specific.

#### Proposed Solution: Service-Level Abstraction

1. Generic Service Interface (`rai_interfaces/`): Create `RAIDetection.srv` with generic fields:

    - Request: `source_img`, `object_names[]`, `model_params` (dict/JSON)
    - Response: `RAIDetectionArray` (already exists)

2. Service Adapter Layer (`services/detection_service.py`): Convert generic `model_params` → model-specific parameters:

    - GroundingDINO: extract `box_threshold`, `text_threshold`
    - YOLO: extract `confidence_threshold`, `nms_threshold`, etc.
    - Update service to use `RAIDetection` interface

3. Tool/Component Updates:
    - `tools/gdino_tools.py`: Replace `RAIGroundingDino` → `RAIDetection` service type, replace hardcoded parameter fields → `model_params` dict
    - `components/gripping_points.py`: Abstract detection parameters, rename DINO-specific methods, use generic service interface
    - `tools/segmentation_tools.py`: Abstract detection parameters, rename DINO-specific methods, use generic service interface
    - All modules: Parse `RAIDetectionArray` directly (already returned by service)

_Benefits:_ Tools and components become model-agnostic. New models only require registry entry + parameter mapping in service. Minimal code changes for model switching.

_Note:_ This requires careful evaluation of parameter abstractions, service interface changes, and backward compatibility. The current PR scope is large (120+ files), so this will be addressed in a follow-up PR after merging.

#### Alternative Approach: Model-Specific Services/Tools

_Approach:_ Create separate services and tools for each model (e.g., `GroundingDinoDetectionService`, `YoloDetectionService`, `GroundingDinoDetectionTool`, `YoloDetectionTool`) while keeping shared infrastructure (registry, `BaseVisionService`, utilities).

_Complexity:_ Lower implementation complexity (no parameter abstraction layer), faster to implement but may result in code duplication and require code changes to switch models. Suitable when model count is low (2-3 models) or models have significantly different interfaces.

#### Progressive Evaluation: GetDistanceToObjectsTool

_Measure:_ [Progressive Evaluation](docs/api_design_considerations.md#progressive-evaluation-ability-to-test-partially-completed-code)

_Module:_ `tools/gdino_tools.py` - `GetDistanceToObjectsTool`

_Issue:_ Multi-stage pipeline (detection → depth extraction → distance calculation) lacks ability to test or inspect intermediate stages, making debugging difficult when distance results are incorrect.

_Proposed:_ Add optional `debug` parameter to expose intermediate results:

-   Publish detection bounding boxes to ROS2 topic for visualization
-   Log detection confidence scores and counts per stage
-   Optional depth ROI visualization to debug bounding box alignment and outlier filtering
