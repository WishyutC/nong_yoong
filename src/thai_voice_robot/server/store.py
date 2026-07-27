"""Structured conversation storage with an idempotent response index."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any

from thai_voice_robot.contracts import AudioDescriptor, ConversationResponse
from thai_voice_robot.server.backends import ChatMessage


@dataclass(frozen=True, slots=True)
class ConversationEvent:
    request_id: str
    session_id: str
    role: str
    text: str
    audio_url: str | None
    created_at: str
    metadata: dict[str, Any]


class ConversationStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                    text TEXT NOT NULL,
                    audio_url TEXT,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_events_session
                    ON events(session_id, id);
                CREATE INDEX IF NOT EXISTS idx_events_request
                    ON events(request_id, id);

                CREATE TABLE IF NOT EXISTS responses (
                    request_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    response_text TEXT NOT NULL,
                    audio_path TEXT NOT NULL,
                    audio_url TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    encoding TEXT NOT NULL,
                    sample_rate_hz INTEGER NOT NULL,
                    channels INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    latency_json TEXT NOT NULL DEFAULT '{}'
                );
                """
            )

    def get_response(self, request_id: str) -> ConversationResponse | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM responses WHERE request_id = ?", (request_id,)
            ).fetchone()
        if row is None:
            return None
        return ConversationResponse(
            request_id=row["request_id"],
            session_id=row["session_id"],
            response_text=row["response_text"],
            audio=AudioDescriptor(
                url=row["audio_url"],
                content_type=row["content_type"],
                encoding=row["encoding"],
                sample_rate_hz=row["sample_rate_hz"],
                channels=row["channels"],
            ),
        )

    def get_audio_path(self, request_id: str) -> Path | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT audio_path FROM responses WHERE request_id = ?", (request_id,)
            ).fetchone()
        return Path(row["audio_path"]) if row else None

    def get_request_text(self, request_id: str) -> str | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT text FROM events
                WHERE request_id = ? AND role = 'user'
                ORDER BY id LIMIT 1
                """,
                (request_id,),
            ).fetchone()
        return str(row["text"]) if row else None

    def recent_messages(self, session_id: str, limit: int) -> list[ChatMessage]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT role, text FROM (
                    SELECT id, role, text
                    FROM events
                    WHERE session_id = ? AND role IN ('user', 'assistant')
                    ORDER BY id DESC
                    LIMIT ?
                ) ORDER BY id ASC
                """,
                (session_id, limit),
            ).fetchall()
        return [ChatMessage(role=row["role"], content=row["text"]) for row in rows]

    def save_turn(
        self,
        *,
        request_id: str,
        session_id: str,
        user_text: str,
        response_text: str,
        audio_path: Path,
        audio: AudioDescriptor,
        latency: dict[str, float],
    ) -> None:
        now = datetime.now(UTC).isoformat()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO events
                    (request_id, session_id, role, text, audio_url, created_at, metadata_json)
                VALUES (?, ?, 'user', ?, NULL, ?, ?)
                """,
                (request_id, session_id, user_text, now, "{}"),
            )
            connection.execute(
                """
                INSERT INTO events
                    (request_id, session_id, role, text, audio_url, created_at, metadata_json)
                VALUES (?, ?, 'assistant', ?, ?, ?, ?)
                """,
                (request_id, session_id, response_text, audio.url, now, json.dumps(latency)),
            )
            connection.execute(
                """
                INSERT INTO responses
                    (request_id, session_id, response_text, audio_path, audio_url,
                     content_type, encoding, sample_rate_hz, channels, created_at, latency_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    session_id,
                    response_text,
                    str(audio_path),
                    audio.url,
                    audio.content_type,
                    audio.encoding,
                    audio.sample_rate_hz,
                    audio.channels,
                    now,
                    json.dumps(latency),
                ),
            )

    def session_events(self, session_id: str, limit: int = 100) -> list[ConversationEvent]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT request_id, session_id, role, text, audio_url, created_at, metadata_json
                FROM events WHERE session_id = ? ORDER BY id DESC LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [
            ConversationEvent(
                request_id=row["request_id"],
                session_id=row["session_id"],
                role=row["role"],
                text=row["text"],
                audio_url=row["audio_url"],
                created_at=row["created_at"],
                metadata=json.loads(row["metadata_json"]),
            )
            for row in reversed(rows)
        ]

    def summary(self) -> dict[str, Any]:
        with self._lock, self._connect() as connection:
            event_count = connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            response_count = connection.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
            session_count = connection.execute(
                "SELECT COUNT(DISTINCT session_id) FROM events"
            ).fetchone()[0]
        return {
            "event_count": event_count,
            "conversation_turns": response_count,
            "session_count": session_count,
        }
