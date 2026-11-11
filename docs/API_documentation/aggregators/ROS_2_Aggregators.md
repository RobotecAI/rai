# ROS 2 Aggregators

`rai.aggregators.ros2` provides:

| Aggregator     | State                                                                                          |
| -------------- | ---------------------------------------------------------------------------------------------- |
| `RosoutDigest` | Distinct `/rosout` lines of the last 60 s at INFO or above, with repeat counts                 |
| `LastImage`    | The newest camera frame, shown to the model as an image                                        |
| `ImageSummary` | A vision model's description of the newest frame, or with `frames=3` of what changed over them |

`ImageSummary` takes `model=` (a chat model or a `rai.llm.get_llm` role, default `"simple"`) and `prompt=`. Images are encoded only for the frames sent to the model.

::: rai.aggregators.ros2.RosoutDigest

::: rai.aggregators.ros2.LastImage

::: rai.aggregators.ros2.ImageSummary

## See Also

-   [Aggregators Overview](overview.md)
