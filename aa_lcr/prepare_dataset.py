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

from aa_lcr.dataset import load_questions, sha256_file, validate_dataset


DATASET_REVISION = "bdae010bbce259820c0e34c1d7cce210d966fb75"
QUESTIONS_FILENAME = "AA-LCR_Dataset.csv"
ARCHIVE_FILENAME = "AA-LCR_extracted-text.zip"
QUESTIONS_SHA256 = "2f90d9c30cfb4dd8df2c0f46547c384065e4c76917bd347a9a97bf797235c1ea"
ARCHIVE_SHA256 = "5e839249826f6b9bd5324f0d139089c9dc481ccb3f212a6dfad00c51045d9d8a"
BASE_URL = (
    "https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR/resolve/"
    f"{DATASET_REVISION}"
)
QUESTIONS_URL = f"{BASE_URL}/{QUESTIONS_FILENAME}"
ARCHIVE_URL = f"{BASE_URL}/extracted_text/{ARCHIVE_FILENAME}"


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


def prepare_dataset(data_dir: Path) -> dict:
    data_dir = Path(data_dir)
    questions_path = data_dir / QUESTIONS_FILENAME
    archive_path = data_dir / ARCHIVE_FILENAME
    extract_root = data_dir / "extracted_text"
    documents_root = extract_root / "lcr"

    _ensure_file(questions_path, QUESTIONS_URL, QUESTIONS_SHA256)
    _ensure_file(archive_path, ARCHIVE_URL, ARCHIVE_SHA256)
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
        "dataset_revision": DATASET_REVISION,
        "questions_url": QUESTIONS_URL,
        "questions_path": str(questions_path),
        "questions_sha256": sha256_file(questions_path),
        "archive_url": ARCHIVE_URL,
        "archive_path": str(archive_path),
        "archive_sha256": sha256_file(archive_path),
        "documents_root": str(documents_root),
        **validation,
    }
    manifest_path = data_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and validate AA-LCR")
    parser.add_argument("--data-dir", type=Path, default=Path("benchmark_data/aa_lcr"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    prepare_dataset(args.data_dir)


if __name__ == "__main__":
    main()
