# `rai_state_logs`

Runtime log digests for LLM-driven debugging, summaries and state updates.

## How does it work?

The node collects logs in the system based on severity and, optionally, a user-set filter, up to a certain line count, keeping a rolling window.

- By default logs with level >= WARN are collected.

### Parameters

- with `clear_on_retrieval` set to `false` logs are cleared after each retrieval (service call).
- when `filters` param contains a list of strings, logs of any severity that include these sub-strings are
  collected. By default, filters are empty, which means logs are collected by severity (warning or higher) alone.
- `max_lines` - lets you set the maximum number of lines to be stored in the log. 512 by default. When this number is exceeded, oldest logs are removed from the rolling window.
- `include_meta`, - lets you include meta information in the log, which is its severity, timestamp and source code location. True by default.

See launchfile: `./launch/rai_state_logs.launch.py` for the easiest way to set these parameters.

## Running with `ros2 launch`

```shell
ros2 launch rai_state_logs rai_state_logs.launch.py
```

Using keywords for filters:

```shell
ros2 launch rai_state_logs rai_state_logs.launch.py filters:="[\"Keyword1\", \"Keyword2\"]"
```
