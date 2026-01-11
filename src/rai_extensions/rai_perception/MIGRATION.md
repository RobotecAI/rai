# Service Name Migration Guide

## Breaking Change: Service Name Defaults

**Date**: January 2026  
**Version**: Breaking change in rai_perception

### Summary

Service name defaults have been changed from model-specific names to generic, model-agnostic names to support the new model-agnostic architecture.

**Old defaults:**

-   Detection service: `"grounding_dino_classify"`
-   Segmentation service: `"grounded_sam_segment"`

**New defaults:**

-   Detection service: `"/detection"`
-   Segmentation service: `"/segmentation"`

### Why This Change?

1. **Model-agnostic architecture**: Services can now use different models (not just GroundingDINO/GroundedSAM)
2. **Consistency**: Services and tools now use the same default names
3. **Flexibility**: Easier to run multiple instances with different service names

### What Changed?

#### Services

`DetectionService` and `SegmentationService` now default to:

-   `/detection` (was `"grounding_dino_classify"`)
-   `/segmentation` (was `"grounded_sam_segment"`)

#### Tools

All perception tools (`GetDetectionTool`, `GetObjectGrippingPointsTool`, etc.) now default to:

-   `/detection` (was `"grounding_dino_classify"`)
-   `/segmentation` (was `"grounded_sam_segment"`)

### Migration Steps

#### Option 1: Update Launch Files (Recommended)

Set service names in launch files to match your deployment:

```python
# In your launch file
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

detection_service = Node(
    package="rai_perception",
    executable="detection_service",
    parameters=[{
        "service_name": "/detection",  # or your custom name
    }]
)
```

#### Option 2: Set ROS2 Parameters

Set parameters before service/tool initialization:

```python
# For services
node.declare_parameter("service_name", "/detection")

# For tools
node.declare_parameter("/detection_tool/service_name", "/detection")
node.declare_parameter("/segmentation_tool/service_name", "/segmentation")
```

#### Option 3: Use Old Names via Parameters

If you need backward compatibility, explicitly set old names:

```python
# In launch file or Python code
node.declare_parameter("service_name", "grounding_dino_classify")  # for DetectionService
node.declare_parameter("/detection_tool/service_name", "grounding_dino_classify")  # for tools
```

### Files Updated

The following files have been updated to use new defaults:

**Core:**

-   `rai_perception/tools/gdino_tools.py`
-   `rai_perception/tools/segmentation_tools.py`
-   `rai_perception/tools/gripping_points_tools.py`
-   `rai_perception/components/gripping_points.py`
-   `rai_perception/components/pcl_detection.py`

**Examples:**

-   `examples/manipulation-demo.py`
-   `examples/manipulation-demo-v2.py` (uses `wait_for_perception_dependencies` utility)

**Configs:**

-   `rai_perception/configs/detection_publisher.yaml`
-   `rai_semap/rai_semap/ros2/config/detection_publisher.yaml`

**Tests:**

-   All test files updated to use new service names

### Backward Compatibility

**Constants still available:**

-   `GDINO_SERVICE_NAME = "grounding_dino_classify"` (in `rai_perception/__init__.py`)
-   `GSAM_SERVICE_NAME = "grounded_sam_segment"` (in `rai_perception/__init__.py`)

These constants are kept for reference but are no longer used as defaults. You can still use them when setting parameters explicitly.

### Multiple Apps / Instances

For multiple applications running simultaneously, use namespaced parameters:

```python
# App 1
node.declare_parameter("/app1/detection_tool/service_name", "/app1/detection")
node.declare_parameter("/app1/segmentation_tool/service_name", "/app1/segmentation")

# App 2
node.declare_parameter("/app2/detection_tool/service_name", "/app2/detection")
node.declare_parameter("/app2/segmentation_tool/service_name", "/app2/segmentation")
```

### Verification

After migration, verify services are accessible:

```bash
# Check services are running
ros2 service list | grep -E "(detection|segmentation)"

# Should show:
# /detection
# /segmentation
# (or your custom names)
```

## Breaking Change: Parameter Prefix and Frame Names

**Date**: January 2026  
**Version**: Breaking change in rai_perception

### Summary

The ROS2 parameter prefix for `GetObjectGrippingPointsTool` has been renamed and frame name defaults may need to be overridden for your deployment.

**Parameter prefix change:**

-   Old: `"pcl.detection.gripping_points"`
-   New: `"perception.gripping_points"`

**Frame name defaults:**

-   `target_frame`: `"base_link"` (may need to be `"panda_link0"` for O3DE, or your robot's base frame)
-   `source_frame`: `"camera_link"` (may need to be `"RGBDCamera5"` for O3DE, or your camera frame)

### Why This Change?

1. **Clearer naming**: `perception.gripping_points` better reflects the module structure
2. **Deployment-specific frames**: Frame names vary by robot/simulation, so defaults may need overrides

### What Changed?

#### Parameter Prefix

The constant `PCL_DETECTION_PARAM_PREFIX` has been renamed to `GRIPPING_POINTS_TOOL_PARAM_PREFIX`:

-   Old: `PCL_DETECTION_PARAM_PREFIX = "pcl.detection.gripping_points"`
-   New: `GRIPPING_POINTS_TOOL_PARAM_PREFIX = "perception.gripping_points"`

All parameters now use the new prefix:

-   `perception.gripping_points.target_frame`
-   `perception.gripping_points.source_frame`
-   `perception.gripping_points.camera_topic`
-   `perception.gripping_points.depth_topic`
-   `perception.gripping_points.camera_info_topic`
-   `perception.gripping_points.timeout_sec`
-   `perception.gripping_points.conversion_ratio`

#### Frame Names

Default frame names are generic (`base_link`, `camera_link`) and may not match your robot/simulation. You must set the correct frame names for your deployment.

### Migration Steps

#### Update Parameter Prefix References

If you have code that references the old constant or prefix:

```python
# Old
from rai_perception.tools.gripping_points_tools import PCL_DETECTION_PARAM_PREFIX
node.declare_parameter(f"{PCL_DETECTION_PARAM_PREFIX}.camera_topic", "/camera/image")

# New
from rai_perception.tools.gripping_points_tools import GRIPPING_POINTS_TOOL_PARAM_PREFIX
node.declare_parameter(f"{GRIPPING_POINTS_TOOL_PARAM_PREFIX}.camera_topic", "/camera/image")
```

#### Set Correct Frame Names

**For O3DE simulation:**

```python
node.declare_parameter("perception.gripping_points.target_frame", "panda_link0")
node.declare_parameter("perception.gripping_points.source_frame", "RGBDCamera5")
```

**For your robot:**

```python
# Find your robot's frame names using:
# ros2 run tf2_ros tf2_echo <source_frame> <target_frame>
# or
# ros2 run tf2_tools view_frames

node.declare_parameter("perception.gripping_points.target_frame", "your_base_frame")
node.declare_parameter("perception.gripping_points.source_frame", "your_camera_frame")
```

**Complete example for O3DE:**

```python
from rai_perception.tools.gripping_points_tools import GRIPPING_POINTS_TOOL_PARAM_PREFIX

node.declare_parameter(f"{GRIPPING_POINTS_TOOL_PARAM_PREFIX}.camera_topic", "/color_image5")
node.declare_parameter(f"{GRIPPING_POINTS_TOOL_PARAM_PREFIX}.depth_topic", "/depth_image5")
node.declare_parameter(f"{GRIPPING_POINTS_TOOL_PARAM_PREFIX}.camera_info_topic", "/color_camera_info5")
node.declare_parameter(f"{GRIPPING_POINTS_TOOL_PARAM_PREFIX}.target_frame", "panda_link0")
node.declare_parameter(f"{GRIPPING_POINTS_TOOL_PARAM_PREFIX}.source_frame", "RGBDCamera5")
```

### Verification

After setting frame names, verify the transform is available:

```bash
# Check if transform exists
ros2 run tf2_ros tf2_echo <source_frame> <target_frame>

# Or use tf2_tools to visualize the tree
ros2 run tf2_tools view_frames
```

If you see errors like "Could not find transform from camera_link to base_link", check:

1. Frame names are correct for your deployment
2. TF tree is being published (check `ros2 topic echo /tf` or `/tf_static`)
3. Parameters are set before tool initialization

## Breaking Change: GetDistanceToObjectsTool Parameter Prefix

**Date**: January 2026  
**Version**: Breaking change in rai_perception

### Summary

`GetDistanceToObjectsTool` now uses prefixed ROS2 parameters for consistency with other tools. Parameters are loaded at initialization instead of during `_run()`.

**Parameter prefix:**

-   Old: No prefix (bare parameter names)
-   New: `"perception.distance_to_objects"`

**Parameter names:**

-   Old: `"outlier_sigma_threshold"`, `"conversion_ratio"`
-   New: `"perception.distance_to_objects.outlier_sigma_threshold"`, `"perception.distance_to_objects.conversion_ratio"`

### Why This Change?

1. **Consistency**: All tools now use parameter prefixes for clear ownership
2. **Initialization**: Parameters loaded at tool creation, not during execution
3. **Auto-declaration**: Parameters auto-declared with defaults if not set

### What Changed?

#### Parameter Loading

Parameters are now loaded in `model_post_init()` via `_load_parameters()` method, matching the pattern used by `GetObjectGrippingPointsTool`.

#### Parameter Names

All parameters now use the `perception.distance_to_objects` prefix:

-   `perception.distance_to_objects.outlier_sigma_threshold` (default: `1.0`)
-   `perception.distance_to_objects.conversion_ratio` (default: `0.001`)

### Migration Steps

#### Update Parameter Declarations

**Old code:**

```python
node.declare_parameter("outlier_sigma_threshold", 1.0)
node.declare_parameter("conversion_ratio", 0.001)
```

**New code:**

```python
node.declare_parameter("perception.distance_to_objects.outlier_sigma_threshold", 1.0)
node.declare_parameter("perception.distance_to_objects.conversion_ratio", 0.001)
```

#### Launch Files

Update launch files to use prefixed parameter names:

```python
from launch_ros.actions import Node

Node(
    package="your_package",
    executable="your_node",
    parameters=[{
        "perception.distance_to_objects.outlier_sigma_threshold": 1.0,
        "perception.distance_to_objects.conversion_ratio": 0.001,
    }]
)
```

### Backward Compatibility

**Auto-declaration**: If old parameter names are not found, the tool will auto-declare new prefixed parameters with defaults. However, to avoid confusion, update your code to use the new parameter names.

**Default values remain the same:**

-   `outlier_sigma_threshold`: `1.0`
-   `conversion_ratio`: `0.001`

### Files Updated

-   `rai_perception/tools/gdino_tools.py` - Added `_load_parameters()` and parameter prefix
-   `tests/rai_perception/tools/test_gdino_tools.py` - Updated to use prefixed parameters

### Questions?

If you encounter issues:

1. Check service logs for actual service names being used
2. Verify ROS2 parameters are set correctly
3. Use `tool.get_config()` to inspect tool configuration
4. Check launch files match service/tool parameter settings
5. Verify frame names match your TF tree using `ros2 run tf2_ros tf2_echo`
