# Robot identity (rai_whoami)

An agent works better when it knows what robot it runs on: what it looks like, what it can do, what rules it follows and what its documentation says. `rai_whoami` builds that from a directory of documentation, images and URDFs into an `EmbodimentInfo` (description, rules, capabilities, behaviors, images) and optionally a vector database of the documentation.

The [robot manifest](manifest.md) can include the whoami description next to the live ROS 2 graph.

## Example: Franka Emika Panda

1.  Set up RAI with the `rai_whoami` package (the [developer environment](../getting_started/dev_env.md) includes it).

2.  Fill a `panda/` directory:

    -   `panda/images`: [this image](https://robodk.com/robot/img/Franka-Emika-Panda-robot.png)
    -   `panda/documentation`: [this document](https://github.com/user-attachments/files/16417196/Franka.Emika.Panda.robot.-.RoboDK.pdf)
    -   `panda/urdf`: [this URDF](https://github.com/frankaemika/franka_ros/blob/develop/franka_description/robots/panda/panda.urdf.xacro)

    Every directory is optional, but the more the builder gets, the better the result.

3.  Build the embodiment info:

    ```bash
    python src/rai_whoami/rai_whoami/build_whoami.py panda/ --build-vector-db
    ```

    The builder calls the `complex` model and, with `--build-vector-db`, the `embeddings` model from [the settings](../getting_started/configuration.md#models). Cloud embeddings cost money; `embeddings = "ollama:nomic-embed-text"` runs locally. `--compress` and `--style` add post-processing passes.

4.  Check `panda/generated/info.json`. It holds `rules`, `capabilities`, `behaviors`, `description` and base64 `images`.

## Use it in Python

```python
from rai.agents import ReActAgent
from rai_whoami import EmbodimentInfo
from rai_whoami.tools import QueryDatabaseTool

info = EmbodimentInfo.from_directory("panda/")  # reads panda/generated/info.json
docs = QueryDatabaseTool(root_dir="panda/")     # searches the vector database

agent = ReActAgent("complex", [docs], system_prompt=info.to_langchain())
```

`to_langchain()` returns a `SystemMessage` with the description, rules, capabilities and behaviors, and the robot's images.

## Use it over ROS 2

```bash
python src/rai_whoami/rai_whoami/scripts/ros2_embodiment_service.py panda/ &
python src/rai_whoami/rai_whoami/scripts/ros2_vector_store_retrieval_service.py panda/
```

```bash
ros2 service call /rai_whoami_embodiment_info_service rai_interfaces/srv/EmbodimentInfo
ros2 service call /rai_whoami_documentation_service rai_interfaces/srv/VectorStoreRetrieval "query: 'maximum load'"
```

If both calls answer, the whoami package is ready.
