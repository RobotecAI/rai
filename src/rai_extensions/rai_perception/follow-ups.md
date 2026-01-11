# After refactoring is done

## Generic Detection Tools Abstraction

problem

`gdino_tools.py` is tightly coupled to GroundingDINO model and service interface:

-   Hardcoded `RAIGroundingDino` service type
-   Model-specific parameters (`box_threshold`, `text_threshold`) as class fields
-   Direct parsing of `RAIGroundingDino.Response` structure

This prevents easy migration to other models (e.g., YOLO) without code changes.

## Proposed Solution: Service-Level Abstraction (Option A)

### Abstraction Layers Needed

1. **Generic Service Interface** (`rai_interfaces/`)

    - Create `RAIDetection.srv` with generic fields:
        - Request: `source_img`, `object_names[]`, `model_params` (dict/JSON)
        - Response: `RAIDetectionArray` (already exists)

2. **Service Adapter Layer** (`services/detection_service.py`)

    - Convert generic `model_params` → model-specific parameters
    - GroundingDINO: extract `box_threshold`, `text_threshold`
    - YOLO: extract `confidence_threshold`, `nms_threshold`, etc.
    - Update service to use `RAIDetection` interface

3. **Tool Updates** (`tools/gdino_tools.py`)
    - Replace `RAIGroundingDino` → `RAIDetection` service type
    - Replace hardcoded parameter fields → `model_params` dict
    - Parse `RAIDetectionArray` directly (already returned by service)

### Benefits

-   Tools become model-agnostic
-   New models only require registry entry + parameter mapping in service
-   Minimal tool code changes for model switching

### Scope

-   New service interface definition
-   Service adapter logic (~50-100 lines)
-   Tool refactoring (~100-150 lines)
-   Model parameter registry extension

## Progressive Evaluation: GetDistanceToObjectsTool

**Measure:** [Progressive Evaluation](docs/api_design_considerations.md#progressive-evaluation-ability-to-test-partially-completed-code)

**Module:** `tools/gdino_tools.py` - `GetDistanceToObjectsTool`

**Issue:** Multi-stage pipeline (detection → depth extraction → distance calculation) lacks ability to test or inspect intermediate stages, making debugging difficult when distance results are incorrect.

**Proposed:** Add optional `debug` parameter to expose intermediate results:

-   Publish detection bounding boxes to ROS2 topic for visualization
-   Log detection confidence scores and counts per stage
-   Optional depth ROI visualization to debug bounding box alignment and outlier filtering

**Scope:** Medium effort (~145-190 lines), similar to `gripping_points_tools.py` debug mode implementation.

## Service Name Updates in rai_bench

**Status:** Changes made in branch `jj/feat/3dpipe_and_usability` but not yet merged to main.

**Issue:** `rai_bench` package uses hardcoded service names that need to be updated to match the generic service names used in `rai_perception` tools.

**Changes Required:**

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

**Rationale:** These changes align `rai_bench` with the model-agnostic service naming convention used in `rai_perception`. Tools now read service names from ROS2 parameters (defaulting to `/detection` and `/segmentation`), making the system model-agnostic and allowing easy switching between detection/segmentation models without code changes.

**Note:** These changes are currently in a feature branch and need to be merged to main after `rai_perception` service name changes are finalized.
