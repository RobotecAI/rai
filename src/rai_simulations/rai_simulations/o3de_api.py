import rclpy
from geometry_msgs.msg import Point, Pose, Quaternion
from tf2_geometry_msgs import do_transform_pose
from tf2_ros import Buffer, TransformListener

from rai.communication.ros2.connectors import ROS2ARIConnector, ROS2ARIMessage


class O3DEAPI:
    def __init__(self, connector: ROS2ARIConnector):
        self.connector = connector
        self.entity_ids = {}

        self.tf2_buffer = Buffer()
        self.tf2_listener = TransformListener(self.tf2_buffer, self.connector._node)

    def get_available_spawnable_names(self) -> list[str]:
        msg = ROS2ARIMessage({})

        response = self.connector.service_call(
            msg,
            target="get_available_spawnable_names",
            msg_type="gazebo_msgs/srv/GetWorldProperties",
        )

        return response.payload.model_names

    def spawn_entity(self, spawnable_name: str, name: str, pose: Pose) -> None:
        msg_content = {
            "name": spawnable_name,
            "xml": "",
            "robot_namespace": name,
            "initial_pose": {
                "position": {
                    "x": pose.position.x,
                    "y": pose.position.y,
                    "z": pose.position.z,
                },
                "orientation": {
                    "x": pose.orientation.x,
                    "y": pose.orientation.y,
                    "z": pose.orientation.z,
                    "w": pose.orientation.w,
                },
            },
        }

        msg = ROS2ARIMessage(payload=msg_content)

        response = self.connector.service_call(
            msg, target="spawn_entity", msg_type="gazebo_msgs/srv/SpawnEntity"
        )

        self.entity_ids[name] = response.payload.status_message
        print(response.payload.status_message)

    def delete_entity(self, name: str) -> None:
        msg_content = {"name": self.entity_ids[name]}

        msg = ROS2ARIMessage(payload=msg_content)

        self.connector.service_call(
            msg, target="delete_entity", msg_type="gazebo_msgs/srv/DeleteEntity"
        )

    def get_entity_pose(self, name: str) -> Pose:
        entity_frame = name + "/"
        return self.transform_pose(Pose(), entity_frame, entity_frame + "odom")

    def transform_pose(self, pose: Pose, source_frame: str, target_frame: str) -> Pose:
        transform = self.tf2_buffer.lookup_transform(
            source_frame=source_frame,
            target_frame=target_frame,
            time=rclpy.time.Time(),
            timeout=rclpy.time.Duration(seconds=5.0),
        )
        pose = do_transform_pose(pose, transform)
        return pose


def main():
    rclpy.init()

    connector = ROS2ARIConnector()
    api = O3DEAPI(connector)

    spawnable_names = api.get_available_spawnable_names()

    print(spawnable_names[1], spawnable_names[2])

    api.spawn_entity(
        spawnable_names[1],
        "entity1",
        api.transform_pose(
            Pose(
                position=Point(x=0.5, y=0.0, z=0.05),
                orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
            "world",
            "odom",
        ),
    )
    import time

    time.sleep(3)
    api.spawn_entity(
        spawnable_names[2],
        "entity2",
        api.transform_pose(
            Pose(
                position=Point(x=0.5, y=0.3, z=0.05),
                orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
            "world",
            "odom",
        ),
    )

    print(api.transform_pose(api.get_entity_pose("entity1"), "odom", "world"))
    api.delete_entity("entity1")

    time.sleep(3)

    print(api.transform_pose(api.get_entity_pose("entity2"), "odom", "world"))
    api.delete_entity("entity2")

    connector.shutdown()


if __name__ == "__main__":
    main()
