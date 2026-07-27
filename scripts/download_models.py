#!/usr/bin/env python3
"""Download only the Thai Zipformer INT8 runtime files from the official archive."""

from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.request
from pathlib import Path


MODEL_NAME = "sherpa-onnx-zipformer-thai-2024-06-20"
MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    f"{MODEL_NAME}.tar.bz2"
)
REQUIRED_FILES = {
    "encoder-epoch-12-avg-5.int8.onnx": 100_000_000,
    "decoder-epoch-12-avg-5.onnx": 1_000_000,
    "joiner-epoch-12-avg-5.int8.onnx": 500_000,
    "tokens.txt": 10_000,
}


def verify(target: Path) -> bool:
    missing = [
        name
        for name, minimum_size in REQUIRED_FILES.items()
        if not (target / name).is_file() or (target / name).stat().st_size < minimum_size
    ]
    if missing:
        print(f"Missing or incomplete: {', '.join(missing)}")
        return False
    print(f"Thai STT model ready: {target}")
    return True


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading official Thai STT archive to {destination}")
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    temporary.replace(destination)


def extract_runtime_files(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    found: set[str] = set()
    with tarfile.open(archive, "r:bz2") as source:
        for member in source.getmembers():
            filename = Path(member.name).name
            if not member.isfile() or filename not in REQUIRED_FILES:
                continue
            extracted = source.extractfile(member)
            if extracted is None:
                continue
            temporary = (target / filename).with_suffix(Path(filename).suffix + ".part")
            with temporary.open("wb") as output:
                shutil.copyfileobj(extracted, output, length=1024 * 1024)
            temporary.replace(target / filename)
            found.add(filename)
            print(f"Extracted {filename}")
    missing = set(REQUIRED_FILES) - found
    if missing:
        raise RuntimeError(f"official archive did not contain: {sorted(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--keep-archive", action="store_true")
    parser.add_argument(
        "--copy-wake-model",
        type=Path,
        help="optional path to the existing keyword_model.tflite",
    )
    args = parser.parse_args()
    models_dir = args.models_dir.expanduser().resolve()
    target = models_dir / MODEL_NAME

    if not verify(target):
        archive = args.archive or (models_dir / f"{MODEL_NAME}.tar.bz2")
        archive = archive.expanduser().resolve()
        if not archive.is_file():
            download(MODEL_URL, archive)
        extract_runtime_files(archive, target)
        if not args.keep_archive and args.archive is None:
            archive.unlink(missing_ok=True)
        if not verify(target):
            raise SystemExit(1)

    if args.copy_wake_model:
        source = args.copy_wake_model.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        wake_target = models_dir / "keyword_model.tflite"
        shutil.copy2(source, wake_target)
        print(f"Wake-word model copied to {wake_target}")


if __name__ == "__main__":
    main()
