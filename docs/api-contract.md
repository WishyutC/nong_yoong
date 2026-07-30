# Workstation voice API contract (v1)

The first implementation keeps text/LLM/TTS as one serialized workstation
operation, but removes WAV data from JSON. All IDs are UUID strings.

## Authentication

Set the same `NONG_YOONG_API_TOKEN` secret on the workstation and every client.
Authenticated requests include:

```http
Authorization: Bearer <token>
```

When a token is configured, missing or incorrect credentials return `401` with
`WWW-Authenticate: Bearer`. Only `GET /health/live` remains public. Generate a
deployment secret with `openssl rand -hex 32`; never commit its value.

## Generate a response

`POST /v1/responses`

Example:

```bash
curl https://api.wishayutc.org/v1/responses \
  --header "Authorization: Bearer $NONG_YOONG_API_TOKEN" \
  --header "Content-Type: application/json" \
  --data '{
    "request_id": "7a2e7093-d4a7-4ad1-8515-af1c82b40c1d",
    "session_id": "bb1bd0a7-6f24-4bbc-a981-621836db8b8c",
    "text": "สวัสดี น้องยูง",
    "language": "th"
  }'
```

Request:

```json
{
  "request_id": "7a2e7093-d4a7-4ad1-8515-af1c82b40c1d",
  "session_id": "bb1bd0a7-6f24-4bbc-a981-621836db8b8c",
  "text": "สวัสดี น้องยูง",
  "language": "th"
}
```

Response (`200 OK`):

```json
{
  "request_id": "7a2e7093-d4a7-4ad1-8515-af1c82b40c1d",
  "session_id": "bb1bd0a7-6f24-4bbc-a981-621836db8b8c",
  "response_text": "สวัสดีครับ วันนี้เป็นอย่างไรบ้างครับ",
  "audio": {
    "url": "/v1/audio/7a2e7093-d4a7-4ad1-8515-af1c82b40c1d.wav",
    "content_type": "audio/wav",
    "encoding": "pcm_s16le",
    "sample_rate_hz": 24000,
    "channels": 1
  }
}
```

The workstation generates mono signed PCM16 WAV directly. The Pi streams the
audio URL to disk or its player in bounded chunks and must not decode Base64 or
run FFmpeg/pydub conversion.

The audio URL is relative unless `NONG_YOONG_PUBLIC_BASE_URL` is configured.
The Pi resolves relative URLs against its configured workstation URL and
rejects audio URLs that unexpectedly point to a different host.

`request_id` is the idempotency and tracing key. Retrying the same request ID
must return the same completed result rather than generating a second reply.
`session_id` selects conversation memory. A new physical conversation gets a
new session ID; every turn within it reuses that ID.

## Fetch audio

`GET /v1/audio/{request_id}.wav`

The response is `audio/wav`. It should support normal HTTP streaming and may
support byte ranges later. Unknown or expired IDs return `404`.

## Health

`GET /health/live` only proves that the API process is running.

`GET /health/ready` returns `200` only when the LLM and Thai TTS backends can
accept work; otherwise it returns `503` with component status details.

## History and dashboard

`GET /v1/sessions/{session_id}/events` returns structured user and assistant
events for one session. `GET /v1/dashboard/summary` returns real event, turn,
and session counts. The initial deployment uses SQLite on the workstation; its
schema is deliberately structured so it can be moved to PostgreSQL later.

## Errors and compatibility

Errors use JSON with `request_id`, a stable machine-readable `code`, and a safe
human-readable `detail`. The legacy `POST /generate` endpoint can remain during
migration as an adapter, but new Pi code uses `/v1/responses`. Audio is never
stored indefinitely by default; retention and consent policy are configured on
the workstation.
