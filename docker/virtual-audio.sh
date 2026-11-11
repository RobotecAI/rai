#!/usr/bin/env bash
# Start virtual audio devices for the sound tests: JACK with the dummy driver and a PulseAudio
# null sink. PortAudio on Lyrical fails to initialize when no PulseAudio server is running.
jackd -d dummy -r 44100 >/dev/null 2>&1 &
pulseaudio --daemonize --exit-idle-time=-1 >/dev/null 2>&1
sleep 1
