"""Prepare explicitly selected released MRCR v2 files without truncation."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import urllib.request

from aa_lcr.prompting import chat_token_count
from mrcr_v2.dataset import (
    DATASET_VERSION, DEFAULT_DATA_DIR, DEFAULT_MODEL, band_urls, file_hash,
    load_tokenizer, text_hash, tokenizer_metadata, validate_bounds,
)
from mrcr_v2.prompting import messages, split_prompt


def _download(url: str, destination: Path) -> None:
    temporary = destination.with_suffix(".download")
    request = urllib.request.Request(url, headers={"User-Agent": "adarsh-rlms-mrcr-v2/1"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    temporary.replace(destination)


def prepare_dataset(args, *, tokenizer_factory=load_tokenizer, downloader=_download) -> dict:
    validate_bounds(args.min_source_tokens, args.max_source_tokens)
    if args.max_rows < 0:
        raise ValueError("--max-rows must be non-negative")
    if bool(args.input_csv) == bool(args.download_bands):
        raise ValueError("Specify either --input-csv or --download-bands")
    urls = band_urls(args.download_bands, args.needles) if args.download_bands else []
    tokenizer = tokenizer_factory(args.executor_model, args.tokenizer_revision)
    metadata = tokenizer_metadata(tokenizer, args.executor_model)
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=False)
    sources = []
    if urls:
        raw_dir = data_dir / "raw"
        raw_dir.mkdir()
        for url in urls:
            path = raw_dir / url.rsplit("/", 1)[-1]
            downloader(url, path)
            sources.append({"path": str(path.resolve()), "url": url, "sha256": file_hash(path)})
    else:
        for path in sorted({Path(path).resolve() for path in args.input_csv}):
            sources.append({"path": str(path), "url": None, "sha256": file_hash(path)})

    csv.field_size_limit(sys.maxsize)
    seen = {}
    selected = []
    source_ids = set()
    read_count = eligible_count = duplicate_count = 0
    questions_path = data_dir / "questions.jsonl"
    with questions_path.open("w", encoding="utf-8") as output:
        for source in sources:
            with Path(source["path"]).open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                required = {"queries", "answer", "view_ops", "context_len", "num_relevant"}
                if not required.issubset(reader.fieldnames or []):
                    raise ValueError(f"Missing MRCR CSV columns in {source['path']}")
                for line, row in enumerate(reader, start=2):
                    read_count += 1
                    prompt, question, answer = row["queries"], row["view_ops"], row["answer"]
                    prefix, body = split_prompt(prompt, question)
                    if not answer or len(answer.strip()) < 12:
                        raise ValueError(f"Invalid MRCR reference at {source['path']}:{line}")
                    if int(row["num_relevant"]) != args.needles:
                        raise ValueError("CSV needle count does not match --needles")
                    count = chat_token_count(tokenizer, messages(prompt))
                    case_id = text_hash(prompt)
                    answer_hash = text_hash(answer)
                    if case_id in seen:
                        if seen[case_id] != answer_hash:
                            raise ValueError(f"Conflicting references for {case_id}")
                        duplicate_count += 1
                        continue
                    seen[case_id] = answer_hash
                    if not args.min_source_tokens <= count <= args.max_source_tokens:
                        continue
                    eligible_count += 1
                    if args.max_rows and len(selected) >= args.max_rows:
                        continue
                    source_id = text_hash(prefix + body)
                    if source_id not in source_ids:
                        directory = data_dir / "sources" / source_id
                        directory.mkdir(parents=True)
                        (directory / "prefix.txt").write_text(prefix, encoding="utf-8", newline="")
                        (directory / "context.txt").write_text(body, encoding="utf-8", newline="")
                        source_ids.add(source_id)
                    record = {
                        "case_id": case_id, "source_id": source_id,
                        "question": question, "answer": answer,
                        "full_rendered_input_tokens": count,
                        "published_context_len": int(row["context_len"]),
                        "num_relevant": args.needles,
                        "source_file_sha256": source["sha256"], "source_row": line,
                        "upstream_metadata": {key: value for key, value in row.items() if key not in required},
                    }
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    selected.append(case_id)
    if not selected:
        raise ValueError("No MRCR examples matched the requested source bounds")
    manifest = {
        "version": DATASET_VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
        "sources": sources, "tokenizer": metadata, "needles": args.needles,
        "min_source_tokens": args.min_source_tokens, "max_source_tokens": args.max_source_tokens,
        "max_rows": args.max_rows, "selection_order": "sorted_input_files_then_csv_row",
        "coverage_scope": "supplied_files_only", "rows_read": read_count,
        "duplicate_rows": duplicate_count, "eligible_unique_rows": eligible_count,
        "selected_example_ids": selected, "source_count": len(source_ids),
        "questions_sha256": file_hash(questions_path),
    }
    (data_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[MRCR] Prepared {len(selected)} examples / {len(source_ids)} transcripts in {data_dir}")
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-csv", type=Path, action="append")
    source.add_argument("--download-bands", help="Official Gemini bands, e.g. 65536:131072,131072:262144")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--min-source-tokens", type=int, required=True)
    parser.add_argument("--max-source-tokens", type=int, required=True)
    parser.add_argument("--needles", type=int, choices=[2, 4, 8], default=8)
    parser.add_argument("--executor-model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--max-rows", type=int, default=0)
    return parser


if __name__ == "__main__":
    prepare_dataset(build_arg_parser().parse_args())
