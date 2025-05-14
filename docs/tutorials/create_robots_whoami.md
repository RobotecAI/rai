# Robot's identity within RAI

RAI Agent needs to understand what kind of robot it is running on. This includes its looks, purpose,
ethical code, equipment, capabilities and documentation. To configure RAI for your robot, provide
contents for your robot's so called `whoami` package.

## Configuration example - Franka Emika Panda arm

1.  Setup the repository using [quick setup guide](../setup/install.md)

2.  Fill in the `panda/` folder with data:

    2.1. Save [this image](https://robodk.com/robot/img/Franka-Emika-Panda-robot.png) into `panda/images`

    2.2. Save [this document](https://github.com/user-attachments/files/16417196/Franka.Emika.Panda.robot.-.RoboDK.pdf) in `panda/documentation`

    2.3. Save [this urdf](https://github.com/frankaemika/franka_ros/blob/develop/franka_description/robots/panda/panda.urdf.xacro) in `panda/urdf`

3.  Build the embodiment info using `build_whoami.py`:

    ```shell
    python src/rai_whoami/rai_whoami/build_whoami.py panda/ --build-vector-db
    ```

    !!! important "Vector database"

        Building the vector database with cloud vendors might lead to costs. Consider using
        the local `ollama` provider for this task. The embedding model can be configured in
        `config.toml` (`ollama` works locally, see
        [docs/setup/vendors.md](../setup/vendors.md#ollama)).

4.  Examine the generated files

After running the build command, inspect the generated files in the `panda/generated` directory. The
folder should contain a info.json file containing:

-   `rules`: List of rules
-   `capabilities`: List of capabilities
-   `behaviors`: List of behaviors
-   `description`: Description of the robot
-   `images`: Base64 encoded images

## Testing (ROS 2)

You can test the generated package by using the RAI Whoami services:

1. Using the RAI Whoami services:

Run the RAI Whoami services:

```bash
python src/rai_whoami/rai_whoami/scripts/ros2_embodiment_service.py panda/ &
python src/rai_whoami/rai_whoami/scripts/ros2_vector_store_retrieval_service.py panda/
```

With the services running, you can query the robot's identity and vector database:

```bash
# Get robot's identity
ros2 service call /rai_whoami_embodiment_info_service rai_interfaces/srv/EmbodimentInfo

# Query the vector database
ros2 service call /rai_whoami_documentation_service rai_interfaces/srv/VectorStoreRetrieval "query: 'maximum load'"
```

If your service calls succeed and you can access the embodiment info and vector database, your
robot's whoami package has been properly initialized.

2. Alternatively, you can use the RAI Whoami tools directly in your Python code:

```python
from rai_whoami import EmbodimentInfo
from rai_whoami.tools import QueryDatabaseTool

# Load embodiment info
info = EmbodimentInfo.from_directory("panda/generated")

# Create a system prompt for your LLM
system_prompt = info.to_langchain()

# Use the vector database tool
query_tool = QueryDatabaseTool(root_dir="panda/generated")
query_tool._run(query="maximum load")
```
