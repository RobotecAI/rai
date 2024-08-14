# Human Robot Interface via Voice

> [!IMPORTANT]
> RAI_ASR is based on OpenAI Whisper model. It is expected to have OPENAI_API_KEY environment variable populated.

## Running example

When your robot's whoami package is ready, run the following:

> [!TIP]
> Make sure rai_whoami is running.

### OpenTTS

```bash
ros2 launch rai_bringup hri.launch.py tts_vendor:=opentts robot_description_package:=<robot_description_package>

```

> [!NOTE]
> Run OpenTTS with `docker run -it -p 5500:5500 synesthesiam/opentts:en--no-espeak`

### ElevenLabs

```bash
ros2 launch rai_bringup hri.launch.py robot_description_package:=<robot_description_package>
```
