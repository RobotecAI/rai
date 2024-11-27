# `rai_state_logs`

Customizable log digests for LLM-driven debugging, summaries and state updates

## How it works?

- By default logs with level >= WARN are collected.

### Parameters

- with `clear_on_retrieval` set to `false` logs are cleared after each retrieval.
- when `filters` param contains a list of keywords, `INFO` logs with keywords are
  collected.
- `max_lines` - lets you set the maximum number of lines to be stored in the log
- `include_meta`, - lets you include meta information in the log

Defaults can be checked in `src/rai_state_logs_node.cpp` or launchfile: `./launch/rai_state_logs.launch.py`

## Running with `ros2 launch`

```shell
ros2 launch rai_state_logs rai_state_logs.launch.py
```

Using keywords for filters:

```shell
ros2 launch rai_state_logs rai_state_logs.launch.py filters:="[\"Keyword1\", \"Keyword2\"]"
```
