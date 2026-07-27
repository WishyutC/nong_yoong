"""Bounded-memory workstation API and WAV streaming client."""

from __future__ import annotations

import os
import time
import wave
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests

from thai_voice_robot.contracts import ConversationRequest, ConversationResponse


class WorkstationAPIError(RuntimeError):
    pass


def validate_playback_wav(path: Path) -> None:
    try:
        with wave.open(str(path), "rb") as wav_file:
            if wav_file.getnchannels() != 1:
                raise WorkstationAPIError("playback WAV must be mono")
            if wav_file.getsampwidth() != 2 or wav_file.getcomptype() != "NONE":
                raise WorkstationAPIError("playback WAV must be uncompressed PCM16")
            if wav_file.getnframes() <= 0:
                raise WorkstationAPIError("playback WAV is empty")
    except wave.Error as exc:
        raise WorkstationAPIError(f"invalid playback WAV: {exc}") from exc


class WorkstationClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 240.0,
        chunk_size: int = 64 * 1024,
        max_audio_bytes: int = 100 * 1024 * 1024,
        retries: int = 2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.chunk_size = chunk_size
        self.max_audio_bytes = max_audio_bytes
        self.retries = retries
        self._session = requests.Session()

    def health(self) -> dict:
        try:
            response = self._session.get(
                f"{self.base_url}/health/ready", timeout=min(self.timeout, 5)
            )
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise WorkstationAPIError(f"workstation health check failed: {exc}") from exc
        if response.status_code != 200:
            raise WorkstationAPIError(f"workstation is not ready: {payload}")
        return payload

    def respond(self, request: ConversationRequest) -> ConversationResponse:
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = self._session.post(
                    f"{self.base_url}/v1/responses",
                    json=request.to_payload(),
                    headers={"X-Request-ID": request.request_id},
                    timeout=self.timeout,
                )
                if response.status_code in {502, 503, 504} and attempt < self.retries:
                    time.sleep(0.5 * (2**attempt))
                    continue
                response.raise_for_status()
                result = ConversationResponse.from_payload(response.json())
                if result.request_id != request.request_id:
                    raise WorkstationAPIError("response request_id does not match request")
                if result.session_id != request.session_id:
                    raise WorkstationAPIError("response session_id does not match request")
                return result
            except (requests.RequestException, ValueError, TypeError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(0.5 * (2**attempt))
                    continue
        raise WorkstationAPIError(f"workstation request failed: {last_error}")

    def download_audio(
        self,
        response: ConversationResponse,
        destination_dir: Path,
    ) -> Path:
        audio_url = urljoin(f"{self.base_url}/", response.audio.url)
        configured = urlparse(self.base_url)
        resolved = urlparse(audio_url)
        if (configured.scheme, configured.netloc) != (resolved.scheme, resolved.netloc):
            raise WorkstationAPIError("audio URL points outside the configured workstation")

        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / f"{response.request_id}.wav"
        temporary = destination.with_suffix(".wav.part")
        total = 0
        try:
            with self._session.get(audio_url, stream=True, timeout=self.timeout) as download:
                download.raise_for_status()
                with temporary.open("wb") as output:
                    for chunk in download.iter_content(chunk_size=self.chunk_size):
                        if not chunk:
                            continue
                        total += len(chunk)
                        if total > self.max_audio_bytes:
                            raise WorkstationAPIError("audio response exceeds configured limit")
                        output.write(chunk)
            validate_playback_wav(temporary)
            os.replace(temporary, destination)
            return destination
        except requests.RequestException as exc:
            raise WorkstationAPIError(f"audio download failed: {exc}") from exc
        finally:
            temporary.unlink(missing_ok=True)
