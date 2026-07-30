"""Dependency-free text and timing helpers for the Qwen3-TTS sidecar."""

from __future__ import annotations

import re
import unicodedata


def clean_text(text: str) -> str:
    """Remove speech-hostile formatting while retaining sentence boundaries."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[*_`#]+", "", text)
    kept = [
        character
        for character in text
        if character == "\n"
        or unicodedata.category(character) not in {"Cc", "Cf", "Cs", "Co", "So"}
    ]
    lines = [" ".join(line.split()) for line in "".join(kept).splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _split_oversized_text(text: str, max_chars: int) -> list[str]:
    """Split at spaces where possible, with a Unicode-safe hard limit fallback."""

    if len(text) <= max_chars:
        return [text]
    words = text.split()
    if len(words) > 1:
        chunks: list[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if current and len(candidate) > max_chars:
                chunks.extend(_split_oversized_text(current, max_chars))
                current = word
            else:
                current = candidate
        if current:
            chunks.extend(_split_oversized_text(current, max_chars))
        return chunks

    chunks = []
    remaining = text
    while len(remaining) > max_chars:
        cut = max_chars
        while cut > max_chars - 8 and unicodedata.category(remaining[cut]).startswith("M"):
            cut -= 1
        chunks.append(remaining[:cut])
        remaining = remaining[cut:]
    if remaining:
        chunks.append(remaining)
    return chunks


def split_text_chunks(text: str, max_chars: int = 160) -> list[str]:
    """Create bounded synthesis units at formatting and sentence boundaries."""

    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")
    cleaned = clean_text(text)
    if not cleaned:
        return []
    segments: list[str] = []
    # Newlines already preserve formatted list items. Avoid splitting on a
    # period because numbered Thai/English lists commonly contain ``1.``.
    sentence_boundary = re.compile(r"(?<=[!?。！？])\s+")
    for line in cleaned.splitlines():
        for sentence in sentence_boundary.split(line):
            sentence = sentence.strip()
            if sentence:
                segments.extend(_split_oversized_text(sentence, max_chars))

    chunks: list[str] = []
    current = ""
    for segment in segments:
        candidate = f"{current} {segment}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = segment
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def silence_sample_count(sample_rate: int, padding_ms: int) -> int:
    """Return the exact number of samples needed for a silence duration."""

    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than zero")
    if padding_ms < 0:
        raise ValueError("padding_ms cannot be negative")
    return int(sample_rate * padding_ms / 1000)
