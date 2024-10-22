# Human-Robot Interface via Streamlit

## Running the Example

When your robot's whoami package is ready, run the following command:

```bash
streamlit run src/rai_hmi/rai_hmi/text_hmi.py <my_robot_whoami> # e.g., rosbot_xl_whoami
```

> [!NOTE]
> The agent's responses may take longer for complex tasks.

## Customization

Currently, customization capabilities are limited due to the internal API design. We are planning to deliver a solution for seamless expansion in the near future.

If you want to customize the available tools, you can do so by editing the `src/rai_hmi/rai_hmi/agent.py` file.

If you have a RaiStateBasedLlmNode running (see e.g., [examples/rosbot-xl-demo.py](examples/rosbot-xl-demo.py)), the Streamlit GUI will communicate with the running node via task_tools defined in the `rai_hmi/rai_hmi/agent.py` file.
