# Follow-ups After Refactoring

## Table of Contents

-   [Breaking Changes](#breaking-changes)
    -   [Agent Service Renaming and Service Name Changes](#agent-service-renaming-and-service-name-changes)
    -   [Agents and Vision Markup Deprecation](#agents-and-vision-markup-deprecation)
    -   [Related Package Updates](#related-package-updates)
-   [Post-PR Follow-ups](#post-pr-follow-ups)
    -   [Generic Detection Tools Abstraction](#generic-detection-tools-abstraction)
    -   [Progressive Evaluation: GetDistanceToObjectsTool](#progressive-evaluation-getdistancetoobjectstool)
    -   [Service Name Updates in rai_bench](#service-name-updates-in-rai_bench)

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

_Current delegation implementation:_ The deprecated agent classes (`GroundingDinoAgent`, `GroundedSamAgent`) are now thin compatibility wrappers that delegate to the new service classes. They emit deprecation warnings and will be removed in a future version. The delegation is implemented via `create_service_wrapper()` helper function in `agents/_helpers.py`, which:

-   Creates a `ROS2Connector` instance
-   Instantiates the corresponding service class (`DetectionService` or `SegmentationService`)
-   Maintains backward compatibility by preserving the old service names (`grounding_dino_classify`, `grounded_sam_segment`) while delegating to the new services

_Impact:_ Code that hardcodes the old class names or service names will need updates. Tools now read service names from ROS2 parameters (defaulting to the new names), making the system model-agnostic.

_Migration:_

-   Update code to use `DetectionService` and `SegmentationService` directly
-   For launch files, use `rai_perception.scripts.run_perception_services` instead of `run_perception_agents`
-   Update launch files and configuration to use the new service names, or set ROS2 parameters to override defaults
-   See `MIGRATION.md` for detailed migration steps

### Agents and Vision Markup Deprecation

The following modules are deprecated:

-   `rai_perception.agents` - Legacy ROS2 service nodes (deprecated, use `services/`)
-   `rai_perception.vision_markup` - Vision markup utilities (deprecated, use `algorithms/`)

These modules remain in the codebase for backward compatibility but will be removed in a future version. New code should use:

-   `services/` for ROS2 service nodes (`DetectionService`, `SegmentationService`)
-   `algorithms/` for core vision algorithms

### Related Package Updates

The following packages require updates to align with the new service naming:

-   `rai_bench`: Update hardcoded service names in configuration files and mocked interfaces
-   Examples: Update manipulation demos to use new service names

These changes are tracked in branch `jj/feat/3dpipe_and_usability` and need to be merged after `rai_perception` service name changes are finalized.

## Post-PR Follow-ups

### Generic Detection Tools Abstraction

_Problem:_ `gdino_tools.py` is tightly coupled to GroundingDINO model and service interface:

-   Hardcoded `RAIGroundingDino` service type
-   Model-specific parameters (`box_threshold`, `text_threshold`) as class fields
-   Direct parsing of `RAIGroundingDino.Response` structure

This prevents easy migration to other models (e.g., YOLO) without code changes.

_Proposed Solution: Service-Level Abstraction_

1. Generic Service Interface (`rai_interfaces/`): Create `RAIDetection.srv` with generic fields:

    - Request: `source_img`, `object_names[]`, `model_params` (dict/JSON)
    - Response: `RAIDetectionArray` (already exists)

2. Service Adapter Layer (`services/detection_service.py`): Convert generic `model_params` → model-specific parameters:

    - GroundingDINO: extract `box_threshold`, `text_threshold`
    - YOLO: extract `confidence_threshold`, `nms_threshold`, etc.
    - Update service to use `RAIDetection` interface

3. Tool Updates (`tools/gdino_tools.py`):
    - Replace `RAIGroundingDino` → `RAIDetection` service type
    - Replace hardcoded parameter fields → `model_params` dict
    - Parse `RAIDetectionArray` directly (already returned by service)

_Benefits:_ Tools become model-agnostic. New models only require registry entry + parameter mapping in service. Minimal tool code changes for model switching.

### Progressive Evaluation: GetDistanceToObjectsTool

_Measure:_ [Progressive Evaluation](docs/api_design_considerations.md#progressive-evaluation-ability-to-test-partially-completed-code)

_Module:_ `tools/gdino_tools.py` - `GetDistanceToObjectsTool`

_Issue:_ Multi-stage pipeline (detection → depth extraction → distance calculation) lacks ability to test or inspect intermediate stages, making debugging difficult when distance results are incorrect.

_Proposed:_ Add optional `debug` parameter to expose intermediate results:

-   Publish detection bounding boxes to ROS2 topic for visualization
-   Log detection confidence scores and counts per stage
-   Optional depth ROI visualization to debug bounding box alignment and outlier filtering

### Service Name Updates in rai_bench

_Status:_ Changes made in branch `jj/feat/3dpipe_and_usability` but not yet merged to main.

_Issue:_ `rai_bench` package uses hardcoded service names that need to be updated to match the generic service names used in `rai_perception` tools.

_Changes Required:_

1. `src/rai_bench/rai_bench/manipulation_o3de/predefined/configs/o3de_config.yaml`

    - Update `required_robotic_ros2_interfaces.services`:
        - `/grounding_dino_classify` → `/detection`
        - `/grounded_sam_segment` → `/segmentation`

2. `src/rai_bench/rai_bench/tool_calling_agent/mocked_ros2_interfaces.py`

    - Update `CUSTOM_SERVICES_AND_TYPES` dictionary:
        - `/grounding_dino_classify` → `/detection`
        - `/grounded_sam_segment` → `/segmentation`

3. `src/rai_bench/rai_bench/tool_calling_agent/tasks/custom_interfaces.py`
    - Update constants:
        - `GROUNDING_DINO_SERVICE = "/grounding_dino_classify"` → `GROUNDING_DINO_SERVICE = "/detection"`
        - `GROUNDED_SAM_SERVICE = "/grounded_sam_segment"` → `GROUNDED_SAM_SERVICE = "/segmentation"`

_Rationale:_ These changes align `rai_bench` with the model-agnostic service naming convention used in `rai_perception`. Tools now read service names from ROS2 parameters (defaulting to `/detection` and `/segmentation`), making the system model-agnostic and allowing easy switching between detection/segmentation models without code changes.

_Note:_ These changes are currently in a feature branch and need to be merged to main after `rai_perception` service name changes are finalized.

_Issue:_ `rai_sim` package may also need update.

-   [PR #750: feat: redesign rai_perception API with tiered structure and improve 3D gripping point detection](https://github.com/RobotecAI/rai/pull/750)
-   [tests/rai_sim](https://github.com/RobotecAI/rai/tree/main/tests/rai_sim) - Tests for rai_sim package
