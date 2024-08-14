# Human Robot Interface via Streamlit

> [!IMPORTANT]
> Streamlit interface is based on OpenAI models. It is expected to have OPENAI_API_KEY environment variable populated.

## Running example

When your robot's whoami package is ready, run the following:

```bash
streamlit run src/rai_hmi/rai_hmi/streamlit_hmi_node.py <my_robot_whoami> # eg husarion_whoami
```

> [!NOTE]
> Agent's responses can take longer time for complex tasks.
