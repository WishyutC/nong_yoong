"""Dependency-free value objects for the v1 Pi/workstation API boundary."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping
from uuid import UUID, uuid4


def _uuid_text(value: str | UUID, field: str) -> str:
    try:
        return str(UUID(str(value)))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"{field} must be a UUID") from exc


@dataclass(frozen=True, slots=True)
class ConversationRequest:
    request_id: str
    session_id: str
    text: str
    language: str = "th"

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _uuid_text(self.request_id, "request_id"))
        object.__setattr__(self, "session_id", _uuid_text(self.session_id, "session_id"))
        clean_text = self.text.strip()
        if not clean_text:
            raise ValueError("text must not be empty")
        object.__setattr__(self, "text", clean_text)
        if not self.language.strip():
            raise ValueError("language must not be empty")

    @classmethod
    def new(
        cls,
        text: str,
        *,
        session_id: str | UUID | None = None,
        language: str = "th",
    ) -> ConversationRequest:
        return cls(
            request_id=str(uuid4()),
            session_id=_uuid_text(session_id or uuid4(), "session_id"),
            text=text,
            language=language,
        )

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ConversationRequest:
        return cls(
            request_id=str(payload.get("request_id", "")),
            session_id=str(payload.get("session_id", "")),
            text=str(payload.get("text", "")),
            language=str(payload.get("language", "th")),
        )


@dataclass(frozen=True, slots=True)
class AudioDescriptor:
    url: str
    content_type: str = "audio/wav"
    encoding: str = "pcm_s16le"
    sample_rate_hz: int = 24_000
    channels: int = 1

    def __post_init__(self) -> None:
        if not self.url.strip():
            raise ValueError("audio url must not be empty")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if self.channels != 1:
            raise ValueError("v1 playback audio must be mono")

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> AudioDescriptor:
        return cls(
            url=str(payload.get("url", "")),
            content_type=str(payload.get("content_type", "audio/wav")),
            encoding=str(payload.get("encoding", "pcm_s16le")),
            sample_rate_hz=int(payload.get("sample_rate_hz", 0)),
            channels=int(payload.get("channels", 0)),
        )


@dataclass(frozen=True, slots=True)
class ConversationResponse:
    request_id: str
    session_id: str
    response_text: str
    audio: AudioDescriptor

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _uuid_text(self.request_id, "request_id"))
        object.__setattr__(self, "session_id", _uuid_text(self.session_id, "session_id"))
        if not self.response_text.strip():
            raise ValueError("response_text must not be empty")

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ConversationResponse:
        audio = payload.get("audio")
        if not isinstance(audio, Mapping):
            raise ValueError("audio must be an object")
        return cls(
            request_id=str(payload.get("request_id", "")),
            session_id=str(payload.get("session_id", "")),
            response_text=str(payload.get("response_text", "")),
            audio=AudioDescriptor.from_payload(audio),
        )
