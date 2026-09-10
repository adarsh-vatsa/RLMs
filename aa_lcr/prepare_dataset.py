"""Download and validate the pinned public AA-LCR dataset."""

from __future__ import annotations

import argparse
import json
import shutil
import unicodedata
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from aa_lcr.dataset import (
    ARCHIVE_SHA256,
    DATASET_RELEASES,
    DEFAULT_DATA_DIR,
    load_questions,
    sha256_file,
    validate_dataset,
)


QUESTIONS_FILENAME = "AA-LCR_Dataset.csv"
ARCHIVE_FILENAME = "AA-LCR_extracted-text.zip"


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    request = urllib.request.Request(
        url, headers={"User-Agent": "adarsh-rlms-aa-lcr/1"}
    )
    try:
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            temporary.open("wb") as out,
        ):
            shutil.copyfileobj(response, out)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _ensure_file(path: Path, url: str, expected_sha256: str) -> None:
    if path.is_file() and sha256_file(path) == expected_sha256:
        return
    _download(url, path)
    actual = sha256_file(path)
    if actual != expected_sha256:
        path.unlink(missing_ok=True)
        raise ValueError(
            f"SHA-256 mismatch for {path}: expected {expected_sha256}, got {actual}"
        )


def _decoded_member_name(name: str) -> PurePosixPath:
    try:
        name = name.encode("cp437").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return PurePosixPath(unicodedata.normalize("NFC", name))


def _extract_archive(archive_path: Path, extract_root: Path) -> None:
    staging_root = extract_root.with_name(extract_root.name + ".extracting")
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True)
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            member = _decoded_member_name(info.filename)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"Unsafe ZIP member: {info.filename}")
            destination = staging_root.joinpath(*member.parts)
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, destination.open("wb") as output:
                shutil.copyfileobj(source, output)
    if extract_root.exists():
        shutil.rmtree(extract_root)
    staging_root.replace(extract_root)


def prepare_dataset(
    data_dir: Path | None = None, *, dataset_version: str = "1.0.0"
) -> dict:
    release = DATASET_RELEASES[dataset_version]
    if data_dir is None:
        data_dir = (
            DEFAULT_DATA_DIR
            if dataset_version == "1.0.0"
            else DEFAULT_DATA_DIR / f"v{dataset_version}"
        )
    data_dir = Path(data_dir)
    questions_path = data_dir / QUESTIONS_FILENAME
    archive_path = data_dir / ARCHIVE_FILENAME
    extract_root = data_dir / "extracted_text"
    documents_root = extract_root / "lcr"
    manifest_path = data_dir / "dataset_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("dataset_revision") != release["revision"]:
            raise ValueError(
                "Dataset directory contains another revision; use a separate --data-dir"
            )
    if (
        questions_path.exists()
        and sha256_file(questions_path) != release["questions_sha256"]
    ):
        raise ValueError(
            "Dataset directory contains a different CSV; use a separate --data-dir"
        )

    base_url = (
        "https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR/resolve/"
        f"{release['revision']}"
    )
    questions_url = f"{base_url}/{QUESTIONS_FILENAME}"
    archive_url = f"{base_url}/extracted_text/{ARCHIVE_FILENAME}"

    _ensure_file(questions_path, questions_url, release["questions_sha256"])
    _ensure_file(archive_path, archive_url, ARCHIVE_SHA256)
    _extract_archive(archive_path, extract_root)

    questions = load_questions(questions_path)
    validation = validate_dataset(
        questions,
        documents_root,
        expected_question_count=100,
    )
    manifest = {
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "dataset_repo": "ArtificialAnalysis/AA-LCR",
        "dataset_version": dataset_version,
        "dataset_revision": release["revision"],
        "questions_url": questions_url,
        "questions_path": str(questions_path),
        "questions_sha256": sha256_file(questions_path),
        "archive_url": archive_url,
        "archive_path": str(archive_path),
        "archive_sha256": sha256_file(archive_path),
        "documents_root": str(documents_root),
        **validation,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and validate AA-LCR")
    parser.add_argument(
        "--dataset-version", choices=sorted(DATASET_RELEASES), default="1.0.0"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Defaults to benchmark_data/aa_lcr for 1.0.0, or its v1.1 subdirectory",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    prepare_dataset(args.data_dir, dataset_version=args.dataset_version)


if __name__ == "__main__":
    main()
