# Streamlit UI

`rai.ui` has two Streamlit pages. Both need the `ui` extra (`pip install "rai-core[ui]"`).

## Chat with an agent

`rai.ui.streamlit.chat` renders a chat page for a `ReActAgent` from its [event stream](../concepts/agents.md#events): model text appears after each model call, every tool call is a status block with its effect and result (denials stand out), and images from tool artifacts are shown inline.

```python
# app.py, run with: streamlit run app.py
import streamlit as st

from rai.agents import ReActAgent
from rai.communication.ros2 import ROS2Connector
from rai.tools.ros2 import ROS2Toolkit
from rai.ui.streamlit import chat


@st.cache_resource
def make_agent() -> ReActAgent:
    return ReActAgent("complex", ROS2Toolkit(connector=ROS2Connector()).get_tools())


st.set_page_config(page_title="My robot", page_icon=":robot:")
chat(make_agent(), title="My robot", greeting="Hi! What can I do for you?")
```

-   Create the agent once per server with `st.cache_resource`: Streamlit reruns the script on every interaction.
-   Each prompt runs one `agent.ainvoke` in the script thread. Do not also run the agent with `rai.run`.
-   The conversation the model sees is `agent.history`; the page keeps its own list of turns per browser session. "Clear conversation" in the sidebar clears both (`rai.ui.streamlit.reset`).
-   `prepare=` turns the typed prompt into what `ainvoke` gets, for example a message with the current camera image attached:

```python
from rai.communication.ros2 import ROS2Connector, encode_image
from rai.messages import human

connector = ROS2Connector()


def with_camera(prompt: str):
    frame = connector.sync.receive("/camera/image_raw", timeout=2.0)
    return human(prompt, images=[encode_image(frame)])


chat(make_agent(), prepare=with_camera)
```

`examples/agents/streamlit_react.py` is a complete app, and the [manipulation demo](../demos/manipulation.md) has a Streamlit front end.

## Configurator

The configurator edits the `[models]` and `[tracing]` tables of `rai.toml` and checks that each model can be created. Other tables in the file are kept as they are.

```bash
streamlit run "$(python -c 'import rai.ui.configurator as m; print(m.__file__)')"
```

In the developer environment: `streamlit run src/rai_core/rai/ui/configurator.py`. See [Configuration](../getting_started/configuration.md).
