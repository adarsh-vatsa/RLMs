"""Build route-labeled legal cache benchmark suites from CUAD-style data."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from cache_bench.run_benchmark import _coerce_text, _sanitize_path_segment


CUAD_DATA_URL = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"
EXCLUDED_CLAUSE_LABELS = {
    "Document Name",
}
SELECTION_STRATEGIES = {"sequential", "balanced-corpora"}


@dataclass(frozen=True)
class LegalRecord:
    source_id: str
    title: str
    context: str
    question: str
    answers: list[str]
    answer_starts: list[int]
    clause_label: str


def _iter_raw_records(raw: object) -> Iterable[dict]:
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                yield item
        return

    if not isinstance(raw, dict):
        return

    for key in ("records", "samples", "data", "train", "test", "validation"):
        value = raw.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    yield item


def load_raw_records(path: Path) -> list[dict]:
    """Load CUAD-style records from JSON or JSONL without requiring datasets."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Legal dataset file not found: {path}")

    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows: list[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                item = json.loads(stripped)
                if not isinstance(item, dict):
                    raise ValueError(f"JSONL row {line_no} must be an object")
                rows.append(item)
        return rows

    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = list(_iter_raw_records(raw))
    if not rows:
        raise ValueError("Legal dataset JSON did not contain records")
    return rows


def _iter_cuad_source_file(path: Path, split_name: str) -> Iterable[dict]:
    """Yield flattened CUAD examples using the logic from legal_bench/cuad_qa.py."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    for example in raw.get("data", []):
        if not isinstance(example, dict):
            continue
        title = _coerce_text(example.get("title"))
        for paragraph in example.get("paragraphs", []):
            if not isinstance(paragraph, dict):
                continue
            context = _coerce_text(paragraph.get("context"))
            for qa in paragraph.get("qas", []):
                if not isinstance(qa, dict):
                    continue
                answers = qa.get("answers") or []
                if not isinstance(answers, list):
                    answers = []
                answer_starts = []
                answer_texts = []
                for answer in answers:
                    if not isinstance(answer, dict):
                        continue
                    answer_text = _coerce_text(answer.get("text"))
                    if not answer_text:
                        continue
                    answer_texts.append(answer_text)
                    answer_start = answer.get("answer_start")
                    if isinstance(answer_start, int):
                        answer_starts.append(answer_start)
                    elif isinstance(answer_start, str) and answer_start.strip().isdigit():
                        answer_starts.append(int(answer_start.strip()))
                yield {
                    "id": _coerce_text(qa.get("id")),
                    "title": title,
                    "context": context,
                    "question": _coerce_text(qa.get("question")),
                    "answers": {
                        "answer_start": answer_starts,
                        "text": answer_texts,
                    },
                    "split": split_name,
                }


def _find_cuad_file(root: Path, filename: str) -> Path:
    direct = root / filename
    if direct.exists():
        return direct
    matches = sorted(root.rglob(filename))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find CUAD source file '{filename}' under {root}")


def _load_cuad_source_dir(root: Path, split: str) -> list[dict]:
    records: list[dict] = []
    split_files = {
        "train": "train_separate_questions.json",
        "test": "test.json",
    }
    split_names = list(split_files.keys()) if split == "all" else [split]

    for split_name in split_names:
        if split_name not in split_files:
            raise ValueError("CUAD split must be 'train', 'test', or 'all'")
        source_file = _find_cuad_file(root, split_files[split_name])
        records.extend(_iter_cuad_source_file(source_file, split_name))
    return records


def load_cuad_source_records(
    source_path: Path | None = None,
    split: str = "all",
    data_url: str = CUAD_DATA_URL,
) -> list[dict]:
    """Load CUAD records directly from the official source zip or extracted files.

    This replaces Hugging Face dataset-script loading, which is no longer
    supported by current `datasets` releases. The parsing mirrors the
    `_generate_examples` implementation in `legal_bench/cuad_qa.py`.
    """
    if source_path:
        source_path = Path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"CUAD source path not found: {source_path}")
        if source_path.is_dir():
            return _load_cuad_source_dir(source_path, split)
        if source_path.suffix.lower() == ".zip":
            with tempfile.TemporaryDirectory() as tmpdir:
                with zipfile.ZipFile(source_path) as zip_file:
                    zip_file.extractall(tmpdir)
                return _load_cuad_source_dir(Path(tmpdir), split)
        return load_raw_records(source_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "cuad_data.zip"
        urllib.request.urlretrieve(data_url, zip_path)
        with zipfile.ZipFile(zip_path) as zip_file:
            zip_file.extractall(tmpdir)
        return _load_cuad_source_dir(Path(tmpdir), split)


def load_huggingface_cuad_records(
    dataset_name: str = "",
    split: str = "all",
) -> list[dict]:
    """Backward-compatible alias for the old CLI path.

    Dataset scripts are no longer supported by current Hugging Face `datasets`
    releases, so this function ignores Hub dataset names and parses CUAD from
    the official source zip or from a local source path.
    """
    source_path = Path(dataset_name) if dataset_name else None
    if source_path and source_path.exists():
        return load_cuad_source_records(source_path=source_path, split=split)
    return load_cuad_source_records(split=split)



def _coerce_answers(value: object) -> tuple[list[str], list[int]]:
    if isinstance(value, dict):
        text_value = value.get("text", [])
        start_value = value.get("answer_start", [])
    else:
        text_value = value
        start_value = []

    if isinstance(text_value, list):
        answers = [_coerce_text(item) for item in text_value]
    else:
        answers = [_coerce_text(text_value)]
    answers = [answer for answer in answers if answer]

    starts: list[int] = []
    if isinstance(start_value, list):
        for item in start_value:
            if isinstance(item, int):
                starts.append(item)
            elif isinstance(item, str) and item.strip().isdigit():
                starts.append(int(item.strip()))
    elif isinstance(start_value, int):
        starts.append(start_value)
    elif isinstance(start_value, str) and start_value.strip().isdigit():
        starts.append(int(start_value.strip()))

    return answers, starts


def _extract_clause_label(question: str) -> str:
    question = _coerce_text(question)
    match = re.search(r'related to\s+"([^"]+)"', question, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()

    match = re.search(r"related to\s+(.+?)\s+that should", question, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip(" .")

    compact = re.sub(r"\s+", " ", question).strip(" ?.")
    return compact[:80] or "legal_clause"


def normalize_cuad_records(raw_records: Sequence[dict]) -> list[LegalRecord]:
    """Normalize CUAD-QA records into the fields needed by the suite builder."""
    records: list[LegalRecord] = []
    seen: set[tuple[str, str, str]] = set()
    seen_clause_labels: set[tuple[str, str]] = set()

    for idx, raw in enumerate(raw_records, start=1):
        title = _coerce_text(raw.get("title") or raw.get("contract_title"))
        context = _coerce_text(raw.get("context") or raw.get("text"))
        question = _coerce_text(raw.get("question") or raw.get("input"))
        source_id = _coerce_text(raw.get("id") or raw.get("source_id")) or f"record_{idx:05d}"
        answers, starts = _coerce_answers(raw.get("answers") or raw.get("answer") or raw.get("expected_output"))

        if not title or not context or not question or not answers:
            continue

        clause_label = _extract_clause_label(question)
        if clause_label in EXCLUDED_CLAUSE_LABELS:
            continue
        clause_key = (title, clause_label)
        if clause_key in seen_clause_labels:
            continue

        key = (title, question, answers[0])
        if key in seen:
            continue
        seen.add(key)
        seen_clause_labels.add(clause_key)

        records.append(
            LegalRecord(
                source_id=source_id,
                title=title,
                context=context,
                question=question,
                answers=answers,
                answer_starts=starts,
                clause_label=clause_label,
            )
        )

    return records


def _paraphrase_question(question: str, clause_label: str) -> str:
    label = _coerce_text(clause_label)
    if label:
        return f"Which contract language should counsel review for the {label} issue?"

    stripped = _coerce_text(question).rstrip("?")
    if stripped.lower().startswith("highlight the parts"):
        return stripped.replace("Highlight the parts", "Identify the relevant provisions", 1) + "."
    return f"Please answer this contract-review question in equivalent terms: {stripped}?"


def _summary_seed_question(record: LegalRecord) -> str:
    return (
        "Summarize the contract evidence relevant to this review issue, including the "
        f"exact supporting clause if present: {record.question}"
    )


def _case_metadata(record: LegalRecord, route: str, seed_record: LegalRecord | None = None) -> dict:
    seed = seed_record or record
    return {
        "source_dataset": "cuad_qa",
        "source_record_id": record.source_id,
        "source_title": record.title,
        "seed_record_id": seed.source_id,
        "clause_label": record.clause_label,
        "answer_starts": record.answer_starts,
        "route_generation": route,
    }


def _find_miss_seed(record: LegalRecord, records_by_title: dict[str, list[LegalRecord]]) -> LegalRecord | None:
    for candidate in records_by_title.get(record.title, []):
        if candidate.source_id != record.source_id and candidate.question != record.question:
            return candidate
    return None


def select_legal_records(
    records: Sequence[LegalRecord],
    max_records: int = 0,
    selection_strategy: str = "sequential",
    max_corpora: int = 5,
    records_per_corpus: int = 3,
) -> list[LegalRecord]:
    """Select normalized legal records before route-case generation."""
    if selection_strategy not in SELECTION_STRATEGIES:
        raise ValueError(
            "selection_strategy must be one of: "
            f"{', '.join(sorted(SELECTION_STRATEGIES))}"
        )

    if selection_strategy == "sequential":
        selected = list(records)
        if max_records > 0:
            selected = selected[:max_records]
        return selected

    if max_corpora <= 0:
        raise ValueError("--max-corpora must be positive for balanced-corpora selection")
    if records_per_corpus <= 0:
        raise ValueError("--records-per-corpus must be positive for balanced-corpora selection")

    records_by_title: dict[str, list[LegalRecord]] = {}
    for record in records:
        records_by_title.setdefault(record.title, []).append(record)

    selected: list[LegalRecord] = []
    corpora_selected = 0
    for bucket in records_by_title.values():
        if len(bucket) < records_per_corpus:
            continue
        selected.extend(bucket[:records_per_corpus])
        corpora_selected += 1
        if corpora_selected >= max_corpora:
            break
    return selected


def build_legal_cache_suite(
    records: Sequence[LegalRecord],
    suite_name: str = "legal_cache_suite_cuad_v1",
    max_records: int = 0,
    selection_strategy: str = "sequential",
    max_corpora: int = 5,
    records_per_corpus: int = 3,
) -> dict:
    """Build a cache_bench-compatible suite from normalized legal records."""
    selected = select_legal_records(
        records,
        max_records=max_records,
        selection_strategy=selection_strategy,
        max_corpora=max_corpora,
        records_per_corpus=records_per_corpus,
    )
    if not selected:
        raise ValueError("No usable legal records were provided")

    corpora_by_id: dict[str, dict] = {}
    records_by_title: dict[str, list[LegalRecord]] = {}
    for record in selected:
        records_by_title.setdefault(record.title, []).append(record)
        corpus_id = _sanitize_path_segment(record.title, max_len=120)
        if corpus_id not in corpora_by_id:
            corpora_by_id[corpus_id] = {
                "corpus_id": corpus_id,
                "docs": [
                    {
                        "filename": f"{corpus_id}.txt",
                        "text": record.context,
                    }
                ],
                "metadata": {
                    "source_dataset": "cuad_qa",
                    "source_title": record.title,
                },
            }

    cases: list[dict] = []
    route_counts = {"exact": 0, "semantic": 0, "knowledge": 0, "miss": 0}

    for record in selected:
        corpus_id = _sanitize_path_segment(record.title, max_len=120)
        base_id = _sanitize_path_segment(record.source_id, max_len=90)
        title_prefix = record.clause_label or "Contract clause"

        route_counts["exact"] += 1
        cases.append(
            {
                "case_id": f"{base_id}__exact",
                "title": f"{title_prefix}: exact reuse",
                "scenario": "legal_exact_positive",
                "corpus_id": corpus_id,
                "seed_queries": [record.question],
                "eval_query": record.question,
                "expected_answers": record.answers,
                "expected_answer_mode": "contains",
                "expected_cache_type": "exact",
                "expected_from_cache": True,
                "metadata": _case_metadata(record, "exact"),
            }
        )

        route_counts["semantic"] += 1
        cases.append(
            {
                "case_id": f"{base_id}__semantic",
                "title": f"{title_prefix}: semantic paraphrase",
                "scenario": "legal_semantic_positive",
                "corpus_id": corpus_id,
                "seed_queries": [record.question],
                "eval_query": _paraphrase_question(record.question, record.clause_label),
                "expected_answers": record.answers,
                "expected_answer_mode": "contains",
                "expected_cache_type": "semantic",
                "expected_from_cache": True,
                "metadata": _case_metadata(record, "semantic"),
            }
        )

        route_counts["knowledge"] += 1
        cases.append(
            {
                "case_id": f"{base_id}__knowledge",
                "title": f"{title_prefix}: knowledge reuse",
                "scenario": "legal_knowledge_positive",
                "corpus_id": corpus_id,
                "seed_queries": [_summary_seed_question(record)],
                "eval_query": record.question,
                "expected_answers": record.answers,
                "expected_answer_mode": "contains",
                "expected_cache_type": "knowledge",
                "expected_from_cache": True,
                "metadata": _case_metadata(record, "knowledge"),
            }
        )

        miss_seed = _find_miss_seed(record, records_by_title)
        if miss_seed is not None:
            route_counts["miss"] += 1
            cases.append(
                {
                    "case_id": f"{base_id}__miss",
                    "title": f"{title_prefix}: different clause should miss",
                    "scenario": "legal_semantic_negative",
                    "corpus_id": corpus_id,
                    "seed_queries": [miss_seed.question],
                    "eval_query": record.question,
                    "expected_answers": record.answers,
                    "expected_answer_mode": "contains",
                    "expected_cache_type": "miss",
                    "expected_from_cache": False,
                    "metadata": _case_metadata(record, "miss", seed_record=miss_seed),
                }
            )

    return {
        "suite_name": suite_name,
        "source": {
            "dataset": "theatticusproject/cuad-qa",
            "format": "cache_bench_fixture_v1",
            "selection_strategy": selection_strategy,
            "records_selected": len(selected),
            "corpora_selected": len(corpora_by_id),
            "max_corpora": max_corpora if selection_strategy == "balanced-corpora" else 0,
            "records_per_corpus": records_per_corpus if selection_strategy == "balanced-corpora" else 0,
            "route_counts": route_counts,
        },
        "corpora": list(corpora_by_id.values()),
        "cases": cases,
    }


def write_suite(suite: dict, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(suite, indent=2), encoding="utf-8")


def build_suite_from_path(
    input_path: Path,
    output_path: Path,
    suite_name: str = "legal_cache_suite_cuad_v1",
    max_records: int = 0,
    selection_strategy: str = "sequential",
    max_corpora: int = 5,
    records_per_corpus: int = 3,
) -> dict:
    raw_records = load_raw_records(input_path)
    records = normalize_cuad_records(raw_records)
    suite = build_legal_cache_suite(
        records,
        suite_name=suite_name,
        max_records=max_records,
        selection_strategy=selection_strategy,
        max_corpora=max_corpora,
        records_per_corpus=records_per_corpus,
    )
    write_suite(suite, output_path)
    return suite


def build_suite_from_cuad_source(
    output_path: Path,
    source_path: Path | None = None,
    split: str = "all",
    suite_name: str = "legal_cache_suite_cuad_v1",
    max_records: int = 0,
    selection_strategy: str = "sequential",
    max_corpora: int = 5,
    records_per_corpus: int = 3,
) -> dict:
    raw_records = load_cuad_source_records(
        source_path=source_path,
        split=split,
    )
    records = normalize_cuad_records(raw_records)
    suite = build_legal_cache_suite(
        records,
        suite_name=suite_name,
        max_records=max_records,
        selection_strategy=selection_strategy,
        max_corpora=max_corpora,
        records_per_corpus=records_per_corpus,
    )
    write_suite(suite, output_path)
    return suite


def build_suite_from_huggingface(
    output_path: Path,
    dataset_name: str = "",
    split: str = "all",
    suite_name: str = "legal_cache_suite_cuad_v1",
    max_records: int = 0,
    selection_strategy: str = "sequential",
    max_corpora: int = 5,
    records_per_corpus: int = 3,
) -> dict:
    """Backward-compatible wrapper for the old --from-huggingface option."""
    source_path = Path(dataset_name) if dataset_name else None
    if source_path and not source_path.exists():
        source_path = None
    return build_suite_from_cuad_source(
        output_path=output_path,
        source_path=source_path,
        split=split,
        suite_name=suite_name,
        max_records=max_records,
        selection_strategy=selection_strategy,
        max_corpora=max_corpora,
        records_per_corpus=records_per_corpus,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a LegalBench/CUAD cache-route suite")
    parser.add_argument("--input-path", default="", help="CUAD-QA JSON or JSONL export")
    parser.add_argument(
        "--from-cuad-source",
        action="store_true",
        help="Download/parse the official CUAD source zip instead of --input-path",
    )
    parser.add_argument(
        "--cuad-source-path",
        default="",
        help="Optional local CUAD source zip, extracted directory, JSON, or JSONL path",
    )
    parser.add_argument("--cuad-split", default="all", help="CUAD split to load: train, test, or all")
    parser.add_argument(
        "--from-huggingface",
        action="store_true",
        help="Deprecated alias for --from-cuad-source; dataset scripts are no longer used",
    )
    parser.add_argument(
        "--hf-dataset",
        default="",
        help="Deprecated. If this is a local path it is treated as --cuad-source-path; Hub ids are ignored.",
    )
    parser.add_argument("--hf-split", default="", help="Deprecated alias for --cuad-split")
    parser.add_argument(
        "--output-path",
        default="benchmark_data/legal_bench/legal_cache_suite_cuad_v1.json",
        help="Where to write the generated cache-route suite JSON",
    )
    parser.add_argument("--suite-name", default="legal_cache_suite_cuad_v1")
    parser.add_argument("--max-records", type=int, default=0, help="Optional record cap")
    parser.add_argument(
        "--selection-strategy",
        choices=sorted(SELECTION_STRATEGIES),
        default="sequential",
        help=(
            "Record selection before route-case generation. "
            "Use balanced-corpora for deterministic contract variety."
        ),
    )
    parser.add_argument(
        "--max-corpora",
        type=int,
        default=5,
        help="Maximum contracts to select with --selection-strategy balanced-corpora",
    )
    parser.add_argument(
        "--records-per-corpus",
        type=int,
        default=3,
        help="Clause records per contract with --selection-strategy balanced-corpora",
    )
    args = parser.parse_args()

    if args.from_cuad_source or args.from_huggingface:
        source_path_text = args.cuad_source_path or args.hf_dataset
        split = args.cuad_split
        if args.hf_split:
            split = args.hf_split
        source_path = Path(source_path_text) if source_path_text else None
        if source_path and not source_path.exists():
            source_path = None
        suite = build_suite_from_cuad_source(
            output_path=Path(args.output_path),
            source_path=source_path,
            split=split,
            suite_name=args.suite_name,
            max_records=args.max_records,
            selection_strategy=args.selection_strategy,
            max_corpora=args.max_corpora,
            records_per_corpus=args.records_per_corpus,
        )
    else:
        if not args.input_path:
            raise SystemExit("Either --input-path or --from-cuad-source is required")
        suite = build_suite_from_path(
            input_path=Path(args.input_path),
            output_path=Path(args.output_path),
            suite_name=args.suite_name,
            max_records=args.max_records,
            selection_strategy=args.selection_strategy,
            max_corpora=args.max_corpora,
            records_per_corpus=args.records_per_corpus,
        )
    print(
        "[LEGAL-BENCH] Wrote suite "
        f"{args.output_path} with {len(suite['corpora'])} corpora and {len(suite['cases'])} cases"
    )


if __name__ == "__main__":
    main()
