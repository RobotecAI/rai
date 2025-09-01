# Copyright (C) 2025 Robotec.AI
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

from typing import Any, Dict, List, Optional, Tuple

from langchain_core.tools import BaseTool, tool

from rai_bench.tool_calling_agent.interfaces import Task, TaskArgs, Validator
from rai_bench.tool_calling_agent.subtasks import CheckArgsToolCallSubTask
from rai_bench.tool_calling_agent.validators import OrderedCallsValidator

#### tasks for demo ###############################
WAREHOUSE_ENVIRONMENT_DESCRIPTION = """
WAREHOUSE LAYOUT:

TABLE WITH SLOTS:
- Table location: x=10-11, y=1-7
- Slot 1: (10.0, 1.5)
- Slot 2: (10.0, 3.0)
- Slot 3: (10.0, 4.5)
- Slot 4: (10.0, 6.0)
When navigating to the table remember that you can't navigate into it,
always approach from the side that is closer to rack (use x=10).

Each slot can contain at most 1 item that can be picked up.
New Items won't appear during the task, so if you picked objects from a ceratin slot,
it will be empty for the rest of the task.

STORAGE RACKS:
Storage Rack 1 location: x=2-6 y=5-6
- Boxes: (3.0, 5.0), (5.0, 5.0)
When navigating to the tack remember that you can't navigate into it,
always approach from the side that is closer to starting position (use y=5).

ROBOT STARTING POSITION:
- Robot starting location: (4.0, 2.0)
"""
SYSTEM_PROMPT = """You are a mobile robot operating in a warehouse environment for pick-and-place operations."""


class EnvStateManager:
    """Enhanced env state manager that tracks objects, boxes, and robot state"""

    def __init__(self):
        self._state = {
            "robot_position": (4.0, 2.0),
            "gripper_state": "open",
        }

        self._objects = {
            "obj_1": {
                "world_position": (10.5, 1.5),  # Slot 1 position
                "color": "blue",
                # when picked up by the robot the obj will "disappear" from the vlm view
                # when dropped the object will appear with different values
                "picked_up": False,
                "relative": (0.02, 0.1, 0.05),  # relative to robot when at slot
            },
            "obj_2": {
                "world_position": (10.5, 3.0),  # Slot 2
                "color": "red",
                "picked_up": False,
                "relative": (-0.2, 0.05, 0.05),
            },
            "obj_3": {
                "world_position": (10.5, 4.5),  # Slot 3
                "color": "green",
                "picked_up": False,
                "relative": (0.1, 0.4, 0.05),
            },
            "obj_4": {
                "world_position": (10.5, 6.0),  # Slot 4
                "color": "green",
                "picked_up": False,
                "relative": (0.15, -0.25, 0.05),
            },
        }

        self._boxes = {
            "box_1": {
                "world_position": (3.0, 5.5),
                "objects": [],  # List of objects in this box
                "relative": (0.2, 0, 0.05),  # relative when robot is at box
            },
            "box_2": {
                "world_position": (5.0, 5.5),
                "objects": [],
                "relative": (0.1, -0.05, 0.05),
            },
        }

    def get_position(self) -> Tuple[float, float]:
        return self._state["robot_position"]

    def set_position(self, x: float, y: float):
        self._state["robot_position"] = (x, y)

    def get_held_object(self) -> Optional[str]:
        return self._state.get("held_object")

    def pick_up_object_at_position(
        self, relative_pos: Tuple[float, float, float]
    ) -> Optional[str]:
        """Pick up object at relative position from current robot location"""
        robot_x, robot_y = self.get_position()

        # Find object at the relative position
        for obj, obj_data in self._objects.items():
            if not obj_data["picked_up"]:
                # Check if this object is at the current location with matching relative position
                if relative_pos == obj_data["relative"]:
                    # Check if robot is at the right slot for this object
                    if (
                        abs(robot_x - obj_data["world_position"][0]) <= 0.5
                        and abs(robot_y - obj_data["world_position"][1]) <= 0.5
                    ):
                        obj_data["picked_up"] = True
                        self._state["held_object"] = obj
                        return obj
        return None

    def drop_object_at_position(self, relative_pos: Tuple[float, float, float]) -> None:
        """Drop held object at relative position from current robot location"""
        # Check if placed in box, if yes, change env state
        robot_x, robot_y = self.get_position()
        # Find which box we're dropping into
        for box_id, box_data in self._boxes.items():
            if relative_pos == box_data["relative"]:
                # Check if robot is at the right position for this box
                if (
                    robot_x == box_data["world_position"][0]
                    and robot_y == box_data["world_position"][1]
                ):
                    # Drop object into box
                    obj_id = self._state["held_object"]
                    box_data["objects"].append(obj_id)

                    # Update object position to be in the box
                    self._objects[obj_id]["world_position"] = (
                        box_data["world_position"][0] + relative_pos[0],
                        box_data["world_position"][1] + relative_pos[1],
                    )

        self._state["held_object"] = None

    def get_visible_objects_at_position(self) -> List[Dict]:
        """Get objects visible at current robot position"""
        robot_x, robot_y = self.get_position()
        visible_objects = []

        # Check for objects at table slots
        if abs(robot_x - 10.0) <= 0.5:  # At sorting table
            for obj_id, obj_data in self._objects.items():
                if not obj_data["picked_up"]:
                    obj_world_pos = obj_data["world_position"]
                    # Check if object is at current slot
                    expected_robot_y = obj_world_pos[1] - obj_data["relative"][1]
                    if abs(robot_y - expected_robot_y) <= 0.5:
                        visible_objects.append(
                            {
                                "id": obj_id,
                                "color": obj_data["color"],
                                "relative_position": obj_data["relative"],
                            }
                        )

        return visible_objects

    def get_visible_boxes_at_position(self) -> List[Dict]:
        """Get boxes visible at current robot position"""
        robot_x, robot_y = self.get_position()
        visible_boxes = []

        # Check for boxes at storage rack
        if 2 <= robot_x <= 6 and abs(robot_y - 5.5) <= 0.5:
            for box_id, box_data in self._boxes.items():
                box_world_pos = box_data["world_position"]
                if abs(robot_x - box_world_pos[0]) <= 0.5:
                    visible_boxes.append(
                        {
                            "id": box_id,
                            "relative_position": box_data["relative"],
                            "contents": [
                                self._objects[obj_id]["color"]
                                for obj_id in box_data["objects"]
                            ],
                        }
                    )

        return visible_boxes

    def get_state_summary(self) -> Dict:
        """Get complete state for debugging"""
        return {
            "robot_position": self._state["robot_position"],
            "gripper_state": self._state["gripper_state"],
            "held_object": self._state.get("held_object"),
            "objects": self._objects,
            "boxes": self._boxes,
        }


class SortTask(Task):
    complexity = "hard"
    type = "warehouse_demo"

    def __init__(
        self,
        task_args: TaskArgs,
        validators: Optional[List[Validator]] = None,
        **kwargs: Any,
    ) -> None:
        if not validators:
            # after every navigate call
            # the where am i should probably be called? should it be mandatory?
            # it is for now
            # Should ask vlm be called after manipulaiton action?
            # So robot can confirm if it pick or droppped object
            where_am_i_subtask = CheckArgsToolCallSubTask(
                expected_tool_name="where_am_i",
                expected_args={},  # No parameters expected
            )
            ask_vlm_subtask = CheckArgsToolCallSubTask(
                expected_tool_name="ask_vlm",
                expected_args={},
            )

            #### navigate to table, detect and pick up object
            navigate_to_slot1_subtask = CheckArgsToolCallSubTask(
                expected_tool_name="nav_tool",
                expected_args={
                    "x": 10.0,
                    "y": 1.5,
                },
            )
            pick_up_1_subtask = CheckArgsToolCallSubTask(
                expected_tool_name="pick_up_object",
                expected_args={"x": 0.02, "y": 0.1, "z": 0.05},
            )
            #### navigate to the box and drop object
            navigate_to_box1_subtask = CheckArgsToolCallSubTask(
                expected_tool_name="nav_tool",
                expected_args={
                    "x": 3.0,
                    "y": 5.0,
                },
            )
            drop_subtask_1 = CheckArgsToolCallSubTask(
                expected_tool_name="drop_object",
                expected_args={"x": 0.2, "y": 0, "z": 0.05},
            )

            #### navigate to the table and pick up second object
            navigate_to_slot2_subtask = CheckArgsToolCallSubTask(
                expected_tool_name="nav_tool",
                expected_args={
                    "x": 10.0,
                    "y": 3.0,
                },
            )
            # there was no green or blue object so navigate to the next slot
            navigate_to_slot3_subtask = CheckArgsToolCallSubTask(
                expected_tool_name="nav_tool",
                expected_args={
                    "x": 10.0,
                    "y": 4.5,
                },
            )
            pick_up_3_subtask = CheckArgsToolCallSubTask(
                expected_tool_name="pick_up_object",
                expected_args={"x": 0.1, "y": 0.4, "z": 0.05},
            )

            #### navigate to the 2nd box and drop
            navigate_to_box2_subtask = CheckArgsToolCallSubTask(
                expected_tool_name="nav_tool",
                expected_args={
                    "x": 5.0,
                    "y": 5.0,
                },
            )
            drop_subtask_2 = CheckArgsToolCallSubTask(
                expected_tool_name="drop_object",
                expected_args={"x": 0.1, "y": -0.05, "z": 0.05},
            )
            #### navigate to 4th slot and check for object, its empty so end the task
            navigate_to_slot4_subtask = CheckArgsToolCallSubTask(
                expected_tool_name="nav_tool",
                expected_args={
                    "x": 10.0,
                    "y": 3.0,
                },
            )
            validators = [
                #### navigate to slot1, detect and pick up 1st object
                OrderedCallsValidator(
                    subtasks=[
                        navigate_to_slot1_subtask,
                        where_am_i_subtask,
                        ask_vlm_subtask,
                        pick_up_1_subtask,
                    ]
                ),
                #### navigate to the box1 and drop object
                OrderedCallsValidator(
                    subtasks=[
                        navigate_to_box1_subtask,
                        where_am_i_subtask,
                        ask_vlm_subtask,
                        drop_subtask_1,
                    ]
                ),
                #### navigate to slot2, detect - there is no blue or green obj
                # so navigate to slot3, detect and pick up
                OrderedCallsValidator(
                    subtasks=[
                        navigate_to_slot2_subtask,
                        where_am_i_subtask,
                        ask_vlm_subtask,
                        navigate_to_slot3_subtask,
                        where_am_i_subtask,
                        ask_vlm_subtask,
                        pick_up_3_subtask,
                    ]
                ),
                #### navigate to the 2nd box and drop
                OrderedCallsValidator(
                    subtasks=[
                        navigate_to_box2_subtask,
                        where_am_i_subtask,
                        ask_vlm_subtask,
                        drop_subtask_2,
                    ]
                ),
                #### navigate to 4th slot and check for object, its empty so end the task
                OrderedCallsValidator(
                    subtasks=[
                        navigate_to_slot4_subtask,
                        where_am_i_subtask,
                        ask_vlm_subtask,
                    ]
                ),
            ]
        super().__init__(validators=validators, task_args=task_args, **kwargs)
        self.env_state = EnvStateManager()

        # define tools
        @tool
        def nav_tool(x: float, y: float):
            """Navigate to certain coordinates in the warehouse."""
            self.env_state.set_position(x, y)
            return (
                f"Navigating to x: {x}, y: {y} ...\n"
                "Check you current position to ensure if movement was done properly"
            )

        @tool
        def where_am_i() -> Dict[str, float]:
            """Returns your current position"""
            x, y = self.env_state.get_position()
            return {"x": x, "y": y}

        @tool
        def pick_up_object(x: float, y: float, z: float) -> str:
            """Move gripper and close it to pick up object from a certain coordinates relative to you"""
            held_obj = self.env_state.get_held_object()
            if not held_obj:
                obj_id = self.env_state.pick_up_object_at_position((x, y, z))
                if obj_id:
                    obj_color = self.env_state._objects[obj_id]["color"]
                    return f"Successfully picked up {obj_color} object ({obj_id}) at relative position x: {x}, y: {y}, z: {z}"
                else:
                    return f"No object grabbed successfully at relative position x: {x}, y: {y}, z: {z}"
            else:
                return f"Can't perform pick up action as you are already holding an {held_obj} object."

        @tool
        def drop_object(x: float, y: float, z: float) -> str:
            """Move gripper and open it to drop object at a certain coordinates relative to you"""
            held_obj = self.env_state.get_held_object()
            if not held_obj:
                return "Failed to drop - you are not holding any object."
            else:
                self.env_state.drop_object_at_position((x, y, z))
                return f"Successfully dropped object ({held_obj}) at relative position x: {x}, y: {y}, z: {z}"

        @tool
        def ask_vlm() -> str:
            """Ask VLM to detect objects at your current location and return their coordinates relative to you"""
            visible_objects = self.env_state.get_visible_objects_at_position()
            visible_boxes = self.env_state.get_visible_boxes_at_position()

            current_pos = self.env_state.get_position()
            x, y = current_pos

            # Generate response based on what's actually visible
            responses = []

            if visible_objects:
                for obj in visible_objects:
                    rel_pos = obj["relative_position"]
                    responses.append(
                        f"I see a {obj['color']} object at x: {rel_pos[0]}, y: {rel_pos[1]}, z: {rel_pos[2]} relative to you"
                    )

            if visible_boxes:
                for box in visible_boxes:
                    rel_pos = box["relative_position"]
                    box_num = "1" if "box_1" in box["id"] else "2"
                    contents_str = (
                        f" (contains: {', '.join(box['contents'])} objects)"
                        if box["contents"]
                        else " (empty)"
                    )
                    responses.append(
                        f"I see Box {box_num} at x: {rel_pos[0]}, y: {rel_pos[1]}, z: {rel_pos[2]} relative to you{contents_str}"
                    )

            if not responses:
                # Check what area we're in for context
                if abs(x - 10.5) < 0.5:  # At sorting table
                    slot_num = None
                    if 1 <= y <= 2:
                        slot_num = 1
                    elif 2.5 <= y <= 3.5:
                        slot_num = 2
                    elif 4 <= y <= 5:
                        slot_num = 3
                    elif 5.5 <= y <= 6.5:
                        slot_num = 4

                    if slot_num:
                        return f"I see Slot {slot_num}, but it appears to be empty."

                elif 2 <= x <= 6 and abs(y - 5.5) < 0.5:  # At storage rack
                    return "I see the storage rack area, but no objects or boxes are immediately visible from this position."

                return "I don't see any relevant objects here."

            return " ".join(responses)

        self.nav_tool = nav_tool
        self.where_am_i = where_am_i
        self.pick_up_object = pick_up_object
        self.drop_object = drop_object
        self.ask_vlm = ask_vlm

    @property
    def optional_tool_calls_number(self) -> int:
        return 10

    def get_base_prompt(self) -> str:
        return (
            "Sort blue and green objects from slots to separate boxes on the rack. "
            "Blue objects should go to the 1st box (x: 3.0, y: 5.0), green objects should go to the second box (x: 5.0, y: 5.0). "
            "Check the slots in order. If you checked all of them and sorted all blue and green objects the task is done."
        )

    def get_prompt(self) -> str:
        return self.get_base_prompt()

    def manipulation_tools(self) -> List[BaseTool]:
        return [
            self.pick_up_object,
            self.drop_object,
            self.ask_vlm,
        ]

    def navigation_tools(self) -> List[BaseTool]:
        return [
            self.nav_tool,
            self.where_am_i,
        ]

    @property
    def available_tools(self) -> List[BaseTool]:
        return [
            self.nav_tool,
            self.where_am_i,
            self.pick_up_object,
            self.drop_object,
            self.ask_vlm,
        ]

    def get_system_prompt(self) -> str:
        return SYSTEM_PROMPT + "\n" + WAREHOUSE_ENVIRONMENT_DESCRIPTION
