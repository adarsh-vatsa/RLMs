"""AA-LCR dataset loading and validation."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATA_DIR = Path("benchmark_data/aa_lcr")
DEFAULT_QUESTIONS_CSV = DEFAULT_DATA_DIR / "AA-LCR_Dataset.csv"
DEFAULT_DOCUMENTS_ROOT = DEFAULT_DATA_DIR / "extracted_text" / "lcr"
DEFAULT_DATASET_MANIFEST = DEFAULT_DATA_DIR / "dataset_manifest.json"

REQUIRED_COLUMNS = {
    "document_category",
    "document_set_id",
    "question_id",
    "question",
    "answer",
    "data_source_filenames",
    "data_source_urls",
    "input_tokens",
}


@dataclass(frozen=True)
class Question:
    document_category: str
    document_set_id: str
    question_id: str
    question: str
    answer: str
    data_source_filenames: tuple[str, ...]
    data_source_urls: tuple[str, ...]
    input_tokens: int

    @property
    def case_id(self) -> str:
        try:
            suffix = f"{int(self.question_id):03d}"
        except ValueError:
            suffix = self.question_id
        return f"aa_lcr_{suffix}"


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _required_text(row: dict[str, str], field: str, location: str) -> str:
    value = str(row.get(field) or "").strip()
    if not value:
        raise ValueError(f"{location} has an empty {field}")
    return value


def _semicolon_values(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(";") if part.strip())


def load_questions(path: Path = DEFAULT_QUESTIONS_CSV) -> list[Question]:
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(
                f"{path} is missing required columns: {', '.join(missing)}"
            )

        questions: list[Question] = []
        seen_ids: set[str] = set()
        document_sets: dict[str, tuple[str, tuple[str, ...]]] = {}
        for line_number, raw in enumerate(reader, start=2):
            location = f"{path}:{line_number}"
            question_id = _required_text(raw, "question_id", location)
            if question_id in seen_ids:
                raise ValueError(f"Duplicate question_id in {path}: {question_id}")
            seen_ids.add(question_id)

            category = _required_text(raw, "document_category", location)
            document_set_id = _required_text(raw, "document_set_id", location)
            filenames = _semicolon_values(
                _required_text(raw, "data_source_filenames", location)
            )
            if not filenames:
                raise ValueError(f"{location} has no data source filenames")
            set_contract = (category, filenames)
            previous = document_sets.setdefault(document_set_id, set_contract)
            if previous != set_contract:
                raise ValueError(
                    f"Document set {document_set_id} has inconsistent category or file order"
                )

            try:
                input_tokens = int(_required_text(raw, "input_tokens", location))
            except ValueError as exc:
                raise ValueError(f"{location} has invalid input_tokens") from exc
            questions.append(
                Question(
                    document_category=category,
                    document_set_id=document_set_id,
                    question_id=question_id,
                    question=_required_text(raw, "question", location),
                    answer=_required_text(raw, "answer", location),
                    data_source_filenames=filenames,
                    data_source_urls=_semicolon_values(
                        str(raw.get("data_source_urls") or "")
                    ),
                    input_tokens=input_tokens,
                )
            )
    return questions


def document_paths(question: Question, documents_root: Path) -> list[Path]:
    set_root = (
        Path(documents_root) / question.document_category / question.document_set_id
    )
    paths = [set_root / filename for filename in question.data_source_filenames]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        listed = ", ".join(str(path) for path in missing[:5])
        suffix = "" if len(missing) <= 5 else f" (+{len(missing) - 5} more)"
        raise FileNotFoundError(
            f"Missing {len(missing)} documents for {question.document_set_id}: {listed}{suffix}"
        )
    return paths


def load_documents(question: Question, documents_root: Path) -> list[tuple[str, str]]:
    return [
        (path.name, path.read_text(encoding="utf-8"))
        for path in document_paths(question, documents_root)
    ]


def validate_dataset(
    questions: list[Question],
    documents_root: Path,
    *,
    expected_question_count: int | None = None,
) -> dict:
    if (
        expected_question_count is not None
        and len(questions) != expected_question_count
    ):
        raise ValueError(
            f"Expected {expected_question_count} questions, found {len(questions)}"
        )
    unique_paths: set[Path] = set()
    for question in questions:
        unique_paths.update(document_paths(question, documents_root))
    return {
        "question_count": len(questions),
        "document_set_count": len({question.document_set_id for question in questions}),
        "referenced_document_count": len(unique_paths),
    }


def group_questions(questions: list[Question]) -> list[list[Question]]:
    order: list[str] = []
    grouped: dict[str, list[Question]] = {}
    for question in questions:
        if question.document_set_id not in grouped:
            order.append(question.document_set_id)
            grouped[question.document_set_id] = []
        grouped[question.document_set_id].append(question)
    return [grouped[document_set_id] for document_set_id in order]


def build_scope_hash(documents: list[tuple[str, str]]) -> str:
    hasher = hashlib.sha256(b"aa_lcr_document_scope_v1\0")
    for filename, text in documents:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        hasher.update(filename.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(normalized.encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def load_dataset_manifest(path: Path = DEFAULT_DATASET_MANIFEST) -> dict:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def dataset_signature(
    questions_csv_sha256: str,
    archive_sha256: str,
    questions: list[Question],
) -> str:
    hasher = hashlib.sha256()
    hasher.update(f"{questions_csv_sha256}\n{archive_sha256}\n".encode("utf-8"))
    for question in questions:
        hasher.update(
            json.dumps(
                {
                    "question_id": question.question_id,
                    "document_set_id": question.document_set_id,
                    "question": question.question,
                    "answer": question.answer,
                    "filenames": question.data_source_filenames,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        )
        hasher.update(b"\n")
    return hasher.hexdigest()
