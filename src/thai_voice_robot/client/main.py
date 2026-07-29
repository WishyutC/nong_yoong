"""Command-line entry point for Raspberry Pi and workstation smoke tests."""

from __future__ import annotations

import argparse
import sys

from thai_voice_robot.client.activation import KeyboardActivator, TFLiteWakeWordActivator
from thai_voice_robot.client.app import VoiceRobot
from thai_voice_robot.client.capture import VoiceActivityRecorder
from thai_voice_robot.client.events import EventLogger
from thai_voice_robot.client.http_client import WorkstationClient
from thai_voice_robot.client.playback import NullAudioPlayer, SystemAudioPlayer
from thai_voice_robot.client.stt import ThaiZipformerRecognizer
from thai_voice_robot.config import ClientSettings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Nong Yoong Thai voice robot client")
    parser.add_argument(
        "--mode",
        choices=("text", "keyboard", "wake"),
        help="text test, keyboard voice activation, or TFLite wake word",
    )
    parser.add_argument("--once", help="send one text prompt and exit")
    parser.add_argument("--no-playback", action="store_true")
    parser.add_argument("--skip-health-check", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = ClientSettings.from_env()
    mode = args.mode or ("text" if args.once else settings.activation_mode)
    workstation = WorkstationClient(
        settings.server_url,
        timeout=settings.request_timeout_seconds,
        chunk_size=settings.stream_chunk_bytes,
    )
    player = (
        NullAudioPlayer()
        if args.no_playback or not settings.playback_enabled
        else SystemAudioPlayer(settings.output_device, settings.audio_player)
    )
    voice_mode = mode in {"keyboard", "wake"}
    robot = VoiceRobot(
        workstation=workstation,
        player=player,
        events=EventLogger(settings.events_path),
        audio_dir=settings.audio_dir,
        recorder=VoiceActivityRecorder(
            sample_rate=settings.sample_rate,
            aggressiveness=settings.vad_aggressiveness,
            silence_seconds=settings.silence_seconds,
            speech_start_timeout_seconds=settings.speech_start_timeout_seconds,
            max_utterance_seconds=settings.max_utterance_seconds,
            input_device=settings.input_device,
        )
        if voice_mode
        else None,
        recognizer=ThaiZipformerRecognizer(settings.stt_model_dir) if voice_mode else None,
        retain_microphone_audio=settings.retain_microphone_audio,
    )

    try:
        robot.start(check_health=not args.skip_health_check)
        print(f"Session: {robot.session_id}")
        if args.once:
            response = robot.process_text(args.once)
            print(f"Nong Yoong: {response.response_text}")
            return

        if mode == "text":
            print("Text mode. Enter /new for a new session or /quit to stop.")
            while True:
                text = input("You: ").strip()
                if not text:
                    continue
                if text == "/quit":
                    break
                if text == "/new":
                    print(f"New session: {robot.new_session()}")
                    continue
                response = robot.process_text(text)
                print(f"Nong Yoong: {response.response_text}")
        else:
            activator = (
                KeyboardActivator()
                if mode == "keyboard"
                else TFLiteWakeWordActivator(
                    settings.wake_model_path,
                    sample_rate=settings.sample_rate,
                    threshold=settings.wake_threshold,
                    invert_probability=settings.wake_invert_probability,
                    input_device=settings.input_device,
                )
            )
            while True:
                activator.wait_for_activation()
                response = robot.process_voice_activation()
                if response is not None:
                    print(f"Nong Yoong: {response.response_text}")
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    finally:
        try:
            robot.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
