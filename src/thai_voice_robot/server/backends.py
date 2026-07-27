"""Pluggable local-LLM backends for the workstation service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import requests


SYSTEM_PROMPT = """คุณคือน้องยูง ผู้ช่วยหุ่นยนต์ที่เป็นมิตรและพูดภาษาไทยเป็นหลัก
ตอบให้กระชับ เป็นธรรมชาติ และเหมาะสำหรับการอ่านออกเสียง
ห้ามอ้างว่าเป็นแพทย์หรือวินิจฉัยสุขภาพจิต หากผู้ใช้พูดถึงอันตรายเร่งด่วน
ให้แนะนำให้ติดต่อบริการฉุกเฉินหรือบุคคลที่ไว้ใจได้อย่างชัดเจน
อย่าแสดงกระบวนการคิดภายใน และอย่าใช้รูปแบบ Markdown ที่ซับซ้อน"""


class BackendUnavailable(RuntimeError):
    """Raised when an inference or speech backend cannot serve a request."""


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str
    content: str


class LLMBackend(Protocol):
    def generate(self, messages: Sequence[ChatMessage]) -> str: ...

    def health(self) -> tuple[bool, str]: ...


class OllamaLLM:
    def __init__(self, base_url: str, model: str, timeout: float = 180.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._session = requests.Session()

    def generate(self, messages: Sequence[ChatMessage]) -> str:
        try:
            response = self._session.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": message.role, "content": message.content}
                        for message in messages
                    ],
                    "stream": False,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            text = str(payload.get("message", {}).get("content", "")).strip()
            if not text:
                raise BackendUnavailable("Ollama returned an empty response")
            return text
        except (requests.RequestException, ValueError, TypeError) as exc:
            raise BackendUnavailable(f"Ollama request failed: {exc}") from exc

    def health(self) -> tuple[bool, str]:
        try:
            response = self._session.get(f"{self.base_url}/api/tags", timeout=3)
            response.raise_for_status()
            models = {
                str(item.get("name", "")).split(":")[0]
                for item in response.json().get("models", [])
            }
            configured = self.model.split(":")[0]
            if configured not in models:
                return False, f"model {self.model!r} is not installed"
            return True, f"Ollama model {self.model} is available"
        except (requests.RequestException, ValueError, TypeError) as exc:
            return False, f"Ollama unavailable: {exc}"


class EchoLLM:
    """Deterministic local backend used for setup checks and automated tests."""

    def generate(self, messages: Sequence[ChatMessage]) -> str:
        user_text = next(
            (message.content for message in reversed(messages) if message.role == "user"),
            "",
        )
        return f"ได้ยินแล้วครับ คุณพูดว่า {user_text}"

    def health(self) -> tuple[bool, str]:
        return True, "echo backend ready"
