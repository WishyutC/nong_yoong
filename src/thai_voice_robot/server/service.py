"""Serialized workstation conversation pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from threading import RLock
from time import monotonic

from thai_voice_robot.config import ServerSettings
from thai_voice_robot.contracts import (
    AudioDescriptor,
    ConversationRequest,
    ConversationResponse,
)
from thai_voice_robot.server.backends import ChatMessage, LLMBackend, SYSTEM_PROMPT
from thai_voice_robot.server.store import ConversationStore
from thai_voice_robot.server.tts import TTSBackend


class ResponseService:
    def __init__(
        self,
        settings: ServerSettings,
        store: ConversationStore,
        llm: LLMBackend,
        tts: TTSBackend,
    ) -> None:
        self.settings = settings
        self.store = store
        self.llm = llm
        self.tts = tts
        self._lock = RLock()
        self.settings.audio_dir.mkdir(parents=True, exist_ok=True)

    def respond(self, request: ConversationRequest) -> ConversationResponse:
        with self._lock:
            existing = self.store.get_response(request.request_id)
            if existing is not None:
                if existing.session_id != request.session_id:
                    raise ValueError("request_id already belongs to another session")
                if self.store.get_request_text(request.request_id) != request.text:
                    raise ValueError("request_id was already used for different text")
                return existing

            started = monotonic()
            history = self.store.recent_messages(
                request.session_id, self.settings.max_history_messages
            )
            messages = [ChatMessage("system", SYSTEM_PROMPT), *history]
            messages.append(ChatMessage("user", request.text))

            llm_started = monotonic()
            response_text = self.llm.generate(messages).strip()
            llm_seconds = monotonic() - llm_started

            final_path = self.settings.audio_dir / f"{request.request_id}.wav"
            temporary_path = Path(f"{final_path}.part.wav")
            try:
                tts_started = monotonic()
                wav_info = self.tts.synthesize(response_text, temporary_path)
                tts_seconds = monotonic() - tts_started
                os.replace(temporary_path, final_path)
            finally:
                temporary_path.unlink(missing_ok=True)

            relative_url = f"/v1/audio/{request.request_id}.wav"
            audio_url = (
                f"{self.settings.public_base_url}{relative_url}"
                if self.settings.public_base_url
                else relative_url
            )
            audio = AudioDescriptor(
                url=audio_url,
                sample_rate_hz=wav_info.sample_rate_hz,
                channels=wav_info.channels,
            )
            result = ConversationResponse(
                request_id=request.request_id,
                session_id=request.session_id,
                response_text=response_text,
                audio=audio,
            )
            latency = {
                "llm_seconds": round(llm_seconds, 6),
                "tts_seconds": round(tts_seconds, 6),
                "total_seconds": round(monotonic() - started, 6),
            }
            self.store.save_turn(
                request_id=request.request_id,
                session_id=request.session_id,
                user_text=request.text,
                response_text=response_text,
                audio_path=final_path,
                audio=audio,
                latency=latency,
            )
            return result
