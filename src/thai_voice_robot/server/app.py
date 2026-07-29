"""FastAPI application for workstation LLM, Thai TTS, and history."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from thai_voice_robot.config import ServerSettings
from thai_voice_robot.contracts import ConversationRequest
from thai_voice_robot.server.backends import (
    BackendUnavailable,
    EchoLLM,
    OllamaLLM,
)
from thai_voice_robot.server.service import ResponseService
from thai_voice_robot.server.store import ConversationStore
from thai_voice_robot.server.tts import (
    EspeakThaiTTS,
    MmsThaiTTS,
    ThonburianThaiTTS,
)
from thai_voice_robot.server.worker import SerializedWorker


class ResponseRequestModel(BaseModel):
    request_id: UUID
    session_id: UUID
    text: str = Field(min_length=1, max_length=12_000)
    language: str = Field(default="th", min_length=2, max_length=16)

    def to_domain(self) -> ConversationRequest:
        return ConversationRequest(
            request_id=str(self.request_id),
            session_id=str(self.session_id),
            text=self.text,
            language=self.language,
        )


class LegacyPromptModel(BaseModel):
    prompt: str = Field(min_length=1, max_length=12_000)
    request_id: UUID | None = None
    session_id: UUID | None = None
    language: str = "th"


def _build_service(settings: ServerSettings) -> ResponseService:
    if settings.llm_backend == "echo":
        llm = EchoLLM()
    elif settings.llm_backend == "ollama":
        llm = OllamaLLM(
            settings.ollama_url,
            settings.ollama_model,
            settings.request_timeout_seconds,
        )
    else:
        raise ValueError(f"unsupported LLM backend: {settings.llm_backend}")

    if settings.tts_backend == "espeak":
        tts = EspeakThaiTTS(settings.espeak_voice, settings.espeak_speed)
    elif settings.tts_backend == "mms":
        tts = MmsThaiTTS(settings.mms_model)
    elif settings.tts_backend == "thonburian":
        tts = ThonburianThaiTTS(
            settings.thonburian_url, settings.thonburian_timeout_seconds
        )
    else:
        raise ValueError(f"unsupported TTS backend: {settings.tts_backend}")

    return ResponseService(
        settings=settings,
        store=ConversationStore(settings.database_path),
        llm=llm,
        tts=tts,
    )


def create_app(
    settings: ServerSettings | None = None,
    service: ResponseService | None = None,
) -> FastAPI:
    active_settings = settings or ServerSettings.from_env()
    active_service = service or _build_service(active_settings)
    app = FastAPI(title="Nong Yoong Thai Voice Robot API", version="1.0.0")
    app.state.settings = active_settings
    app.state.service = active_service
    conversation_worker = SerializedWorker()
    app.state.conversation_worker = conversation_worker

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(active_settings.cors_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.exception_handler(BackendUnavailable)
    async def backend_error_handler(
        request: Request, exc: BackendUnavailable
    ) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={
                "request_id": request.headers.get("x-request-id"),
                "code": "backend_unavailable",
                "detail": str(exc),
            },
        )

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/health/ready")
    async def ready() -> JSONResponse:
        llm_ok, llm_detail = active_service.llm.health()
        tts_ok, tts_detail = active_service.tts.health()
        ready_state = llm_ok and tts_ok
        return JSONResponse(
            status_code=200 if ready_state else 503,
            content={
                "status": "ready" if ready_state else "not_ready",
                "components": {
                    "llm": {"ready": llm_ok, "detail": llm_detail},
                    "tts": {"ready": tts_ok, "detail": tts_detail},
                },
            },
        )

    @app.post("/v1/responses")
    async def generate_response(data: ResponseRequestModel) -> dict[str, Any]:
        try:
            result = await conversation_worker.run(active_service.respond, data.to_domain())
            return result.to_payload()
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/v1/audio/{request_id}.wav")
    async def get_audio(request_id: UUID) -> StreamingResponse:
        audio_path = active_service.store.get_audio_path(str(request_id))
        if audio_path is None or not audio_path.is_file():
            raise HTTPException(status_code=404, detail="audio not found")

        async def stream_file():
            with Path(audio_path).open("rb") as source:
                while chunk := source.read(64 * 1024):
                    yield chunk

        return StreamingResponse(
            stream_file(),
            media_type="audio/wav",
            headers={
                "Content-Length": str(audio_path.stat().st_size),
                "Content-Disposition": f'inline; filename="{request_id}.wav"',
            },
        )

    @app.get("/v1/sessions/{session_id}/events")
    async def session_events(session_id: UUID, limit: int = 100) -> list[dict[str, Any]]:
        safe_limit = min(max(limit, 1), 500)
        return [
            {
                "request_id": event.request_id,
                "session_id": event.session_id,
                "role": event.role,
                "text": event.text,
                "audio_url": event.audio_url,
                "created_at": event.created_at,
                "metadata": event.metadata,
            }
            for event in active_service.store.session_events(str(session_id), safe_limit)
        ]

    @app.get("/v1/dashboard/summary")
    async def dashboard_summary() -> dict[str, Any]:
        return active_service.store.summary()

    @app.post("/generate")
    async def legacy_generate(data: LegacyPromptModel) -> dict[str, Any]:
        """Temporary migration adapter. It intentionally does not return Base64."""

        request_id = str(data.request_id or uuid4())
        session_id = str(data.session_id or uuid4())
        result = await conversation_worker.run(
            active_service.respond,
            ConversationRequest(
                request_id=request_id,
                session_id=session_id,
                text=data.prompt,
                language=data.language,
            ),
        )
        return {
            "request_id": result.request_id,
            "session_id": result.session_id,
            "response_text": result.response_text,
            "audio_url": result.audio.url,
            "audio": result.audio.to_payload()
            if hasattr(result.audio, "to_payload")
            else {
                "url": result.audio.url,
                "content_type": result.audio.content_type,
                "encoding": result.audio.encoding,
                "sample_rate_hz": result.audio.sample_rate_hz,
                "channels": result.audio.channels,
            },
        }

    return app


app = create_app()
