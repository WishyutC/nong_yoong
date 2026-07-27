# Raspberry Pi hardware acceptance

Automated tests verify the software boundary, state transitions, persistence,
streaming response format, and workstation Thai TTS. The following checks must
run on the actual Raspberry Pi, USB audio adapter, microphone, and speaker.

## Bring-up

1. Run `python scripts/doctor.py` and resolve every missing Pi voice component.
2. Confirm the microphone and speaker with `arecord -l`, `aplay -l`, a short
   recording, and playback.
3. Start the workstation server and verify `/health/ready` from the Pi.
4. Run `nong-yoong-client --mode keyboard` and complete ten Thai turns.
5. Confirm that generated speech is audible and the state returns to `IDLE`
   only after playback finishes.
6. Install/copy the TFLite wake model, run wake mode, and tune the threshold
   using real background noise.

## Required test set

Collect 300–500 consented Thai utterances without retaining raw voice by
default. Include quiet and noisy rooms, motors/fans on and off, distances of
0.5/1/2/3 metres, regional accents, age ranges, Thai-English code switching,
and attempted interruption while the robot speaks.

Record aggregate metrics:

- Thai character error rate
- command/task success rate
- STT real-time factor and endpoint delay
- API/LLM latency
- TTS time to first completed WAV
- end-to-end turn latency
- peak Pi RAM, CPU, temperature, and dropped audio frames
- wake false accepts and false rejects per hour

Command success is the primary product metric. Emotion or wellness metadata
must be presented only as a low-confidence signal, never a diagnosis, and must
require consent before sensitive data is stored.
