"""State-driven Raspberry Pi conversation orchestrator."""

from __future__ import annotations

from pathlib import Path
from time import monotonic
from uuid import uuid4

from thai_voice_robot.client.capture import NoSpeechDetected, VoiceActivityRecorder
from thai_voice_robot.client.events import EventLogger
from thai_voice_robot.client.http_client import WorkstationClient
from thai_voice_robot.client.playback import AudioPlayer
from thai_voice_robot.client.stt import ThaiZipformerRecognizer
from thai_voice_robot.contracts import ConversationRequest, ConversationResponse
from thai_voice_robot.runtime_state import (
    InvalidStateTransition,
    RobotState,
    RobotStateMachine,
)


class VoiceRobot:
    def __init__(
        self,
        *,
        workstation: WorkstationClient,
        player: AudioPlayer,
        events: EventLogger,
        audio_dir: Path,
        recorder: VoiceActivityRecorder | None = None,
        recognizer: ThaiZipformerRecognizer | None = None,
        retain_microphone_audio: bool = False,
    ) -> None:
        self.workstation = workstation
        self.player = player
        self.events = events
        self.audio_dir = audio_dir
        self.recorder = recorder
        self.recognizer = recognizer
        self.retain_microphone_audio = retain_microphone_audio
        self.state = RobotStateMachine()
        self.session_id = str(uuid4())

    def start(self, *, check_health: bool = True) -> None:
        if check_health:
            self.workstation.health()
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.state.transition(RobotState.IDLE, expected=RobotState.STARTING)
        self.events.log("started", session_id=self.session_id)

    def new_session(self) -> str:
        if self.state.state is not RobotState.IDLE:
            raise InvalidStateTransition("a session can change only while idle")
        self.session_id = str(uuid4())
        self.events.log("session_started", session_id=self.session_id)
        return self.session_id

    def _finish_request(self, request: ConversationRequest) -> ConversationResponse:
        started = monotonic()
        current = self.state.state
        if current is RobotState.RECORDING:
            self.state.transition(RobotState.THINKING, expected=RobotState.RECORDING)
        elif current is RobotState.IDLE:
            self.state.transition(RobotState.THINKING, expected=RobotState.IDLE)
        else:
            raise InvalidStateTransition(f"cannot send request while {current.value}")

        self.events.log(
            "user_transcript",
            request_id=request.request_id,
            session_id=request.session_id,
            language=request.language,
            text=request.text,
        )
        response = self.workstation.respond(request)
        audio_path = self.workstation.download_audio(response, self.audio_dir / "responses")
        self.events.log(
            "assistant_response",
            request_id=request.request_id,
            session_id=request.session_id,
            text=response.response_text,
            audio_path=str(audio_path),
        )

        self.state.transition(RobotState.SPEAKING, expected=RobotState.THINKING)
        self.player.play(audio_path)
        self.state.transition(RobotState.IDLE, expected=RobotState.SPEAKING)
        self.events.log(
            "turn_completed",
            request_id=request.request_id,
            session_id=request.session_id,
            elapsed_seconds=round(monotonic() - started, 6),
        )
        return response

    def process_text(self, text: str) -> ConversationResponse:
        request = ConversationRequest.new(text, session_id=self.session_id, language="th")
        try:
            return self._finish_request(request)
        except Exception as exc:
            self._recover(exc, request.request_id)
            raise

    def process_voice_activation(self) -> ConversationResponse | None:
        if self.recorder is None or self.recognizer is None:
            raise RuntimeError("voice mode requires a recorder and Thai recognizer")
        request_id = str(uuid4())
        microphone_path = self.audio_dir / "microphone" / f"{request_id}.wav"
        self.state.transition(RobotState.RECORDING, expected=RobotState.IDLE)
        self.events.log(
            "recording_started", request_id=request_id, session_id=self.session_id
        )
        try:
            self.recorder.record(microphone_path)
            transcript = self.recognizer.transcribe(microphone_path).strip()
            if not transcript:
                self.events.log(
                    "empty_transcript", request_id=request_id, session_id=self.session_id
                )
                self.state.transition(RobotState.IDLE, expected=RobotState.RECORDING)
                return None
            request = ConversationRequest(
                request_id=request_id,
                session_id=self.session_id,
                text=transcript,
                language="th",
            )
            return self._finish_request(request)
        except NoSpeechDetected as exc:
            self.events.log(
                "no_speech", request_id=request_id, session_id=self.session_id, detail=str(exc)
            )
            self.state.transition(RobotState.IDLE, expected=RobotState.RECORDING)
            return None
        except Exception as exc:
            self._recover(exc, request_id)
            raise
        finally:
            if not self.retain_microphone_audio:
                microphone_path.unlink(missing_ok=True)

    def _recover(self, exc: Exception, request_id: str | None = None) -> None:
        self.events.log(
            "turn_error",
            request_id=request_id,
            session_id=self.session_id,
            error_type=type(exc).__name__,
            detail=str(exc),
        )
        if self.state.state is RobotState.STOPPING:
            return
        try:
            self.state.transition(RobotState.ERROR)
        except InvalidStateTransition:
            pass
        if self.state.state is RobotState.ERROR:
            self.state.transition(RobotState.IDLE, expected=RobotState.ERROR)

    def stop(self) -> None:
        if self.state.state is not RobotState.STOPPING:
            self.state.transition(RobotState.STOPPING)
        self.events.log("stopped", session_id=self.session_id)
