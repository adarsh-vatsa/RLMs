"""Small metadata records referencing shared MRCR transcripts on disk."""

import hashlib
import json
from pathlib import Path

from mrcr_v2.prompting import PROMPT_VERSION


DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B"
DEFAULT_DATA_DIR = Path("benchmark_data/mrcr_v2")
DATASET_VERSION = 1
BASE_URL = "https://storage.googleapis.com/mrcr_v2"
OFFICIAL_BANDS = [(2**power, 2**(power + 1)) for power in range(12, 23)]


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_bounds(minimum: int, maximum: int) -> None:
    if minimum <= 0 or maximum < minimum:
        raise ValueError("Source token bounds must be positive with minimum <= maximum")


def load_tokenizer(model: str, revision: str | None = None):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=True)


def tokenizer_metadata(tokenizer, model: str) -> dict:
    template = tokenizer.chat_template
    backend = getattr(tokenizer, "backend_tokenizer", None)
    vocabulary = backend.to_str() if backend is not None else json.dumps(tokenizer.get_vocab(), sort_keys=True)
    return {
        "model": model,
        "revision": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
        "class": type(tokenizer).__name__,
        "chat_template_sha256": text_hash(json.dumps(template, sort_keys=True)),
        "tokenizer_sha256": text_hash(vocabulary),
        "special_tokens_sha256": text_hash(json.dumps(tokenizer.special_tokens_map, sort_keys=True, default=str)),
        "prompt_version": PROMPT_VERSION,
        "enable_thinking": False,
    }


def band_urls(raw: str, needles: int) -> list[str]:
    bands = []
    for item in raw.split(","):
        try:
            lower, upper = (int(value) for value in item.strip().split(":"))
        except ValueError as exc:
            raise ValueError("Download bands must be comma-separated LOWER:UPPER pairs") from exc
        if (lower, upper) not in OFFICIAL_BANDS:
            raise ValueError(f"Not an official MRCR band: {lower}:{upper}")
        bands.append((lower, upper))
    return [
        f"{BASE_URL}/mrcr_v2p1_{needles}needle_in_({lower},{upper})_dynamic_fewshot_text_style_fast.csv"
        for lower, upper in sorted(set(bands))
    ]


def read_source(data_dir: Path, source_id: str) -> tuple[str, str]:
    directory = data_dir / "sources" / source_id
    with (directory / "prefix.txt").open(encoding="utf-8", newline="") as handle:
        prefix = handle.read()
    with (directory / "context.txt").open(encoding="utf-8", newline="") as handle:
        body = handle.read()
    if text_hash(prefix + body) != source_id:
        raise ValueError(f"Prepared transcript changed: {source_id}")
    return prefix, body


def load_prepared(data_dir: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((data_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    if manifest["version"] != DATASET_VERSION:
        raise ValueError("Unsupported MRCR prepared dataset version")
    questions_path = data_dir / "questions.jsonl"
    if file_hash(questions_path) != manifest["questions_sha256"]:
        raise ValueError("Prepared questions changed; prepare a new dataset")
    with questions_path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if [row["case_id"] for row in rows] != manifest["selected_example_ids"]:
        raise ValueError("Prepared example IDs do not match the manifest")
    return manifest, rows
