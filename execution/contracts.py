from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable


PIPELINE_VERSION = "shared_direct_hybrid_v1"
EVIDENCE_SELECTIONS = ("dense", "bm25", "fusion", "random", "position")


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class Document:
    id: str
    text: str


@dataclass(frozen=True)
class Task:
    """Solver input only. References and scoring metadata belong outside this record."""

    case_id: str
    source_id: str
    documents: tuple[Document, ...]
    query: str
    render: Callable[[list[dict] | None], list[dict]]
    prompt_version: str
    choices: tuple[str, ...] = ()
    required_prefix: str = ""
    fixed_instructions: str = ""
    legacy_docs_dir: Path | None = None


@dataclass(frozen=True)
class Config:
    mode: str
    max_input_tokens: int
    max_output_tokens: int
    context_window_tokens: int
    executor_model: str = ""
    profile: str = "common"
    direct_overflow: str = "unsupported"
    evidence_order: str = "source"
    merge_overlaps: bool = True
    exact_offsets: bool = True
    child_tokens: int = 7500
    child_overlap_tokens: int = 750
    cache_read: bool = False
    cache_write: bool = False
    cache_matching: str = "semantic"
    rerank_top: int = 0
    evidence_selection: str = "dense"
    selection_seed: int = 0
    neighbor_children: int = 0
    document_floor: bool = False
    packing: str = "stop"
    max_retries: int = 5

    def __post_init__(self):
        if self.mode not in {"direct", "hybrid"}:
            raise ValueError("Execution mode must be direct or hybrid")
        if min(self.max_input_tokens, self.max_output_tokens, self.context_window_tokens) <= 0:
            raise ValueError("Executor token budgets must be positive")
        if self.max_input_tokens + self.max_output_tokens > self.context_window_tokens:
            raise ValueError("Input plus output budget must not exceed the context window")
        if self.child_tokens <= 0 or not 0 <= self.child_overlap_tokens < self.child_tokens:
            raise ValueError("Require child tokens > overlap >= 0")
        if self.direct_overflow not in {"unsupported", "middle", "middle-marked", "head", "error"}:
            raise ValueError("Invalid direct overflow policy")
        if self.evidence_selection not in EVIDENCE_SELECTIONS:
            raise ValueError("Invalid evidence selection")
        if self.packing not in {"stop", "skip", "fill"}:
            raise ValueError("Invalid packing policy")
        if self.neighbor_children < 0:
            raise ValueError("Neighbor children must be non-negative")
        if self.rerank_top and self.evidence_selection in {"random", "position"}:
            raise ValueError("Reranking requires a query-ranked evidence selection")
        if ((self.evidence_selection != "dense" or self.neighbor_children or self.document_floor
                or self.packing != "stop") and not (self.exact_offsets and self.merge_overlaps)):
            raise ValueError("Evidence selection, neighbors, document floor and packing require merged source order")
        if self.evidence_order not in {"source", "score"}:
            raise ValueError("Invalid evidence order")
        if self.merge_overlaps and self.evidence_order != "source":
            raise ValueError("Overlap merging requires source order")
        if self.cache_matching not in {"exact", "semantic"}:
            raise ValueError("Invalid cache matching policy")
        if self.max_retries < 1 or self.rerank_top < 0:
            raise ValueError("Invalid retry or reranking limit")

    def metadata(self):
        settings = {"pipeline_version": PIPELINE_VERSION, **asdict(self)}
        return {**settings, "config_sha256": fingerprint(settings)}

    def cache_identity(self):
        settings = self.metadata()
        for key in ("cache_read", "cache_write", "config_sha256"):
            settings.pop(key)
        return settings


def add_arguments(parser):
    parser.add_argument("--execution-profile", choices=("legacy", "common"), default="legacy")
    parser.add_argument("--answer-cache-read", action="store_true", default=None)
    parser.add_argument("--no-answer-cache-read", action="store_false", dest="answer_cache_read")
    parser.add_argument("--answer-cache-write", action="store_true", default=None)
    parser.add_argument("--no-answer-cache-write", action="store_false", dest="answer_cache_write")
    parser.add_argument("--cache-matching", choices=("exact", "semantic"), default="semantic")
    parser.add_argument("--evidence-order", choices=("source", "score"))
    parser.add_argument("--merge-overlaps", action="store_true", default=None)
    parser.add_argument("--no-merge-overlaps", action="store_false", dest="merge_overlaps")
    parser.add_argument("--direct-overflow", choices=("unsupported", "middle", "middle-marked", "head", "error"))
    parser.add_argument("--pipeline-rerank-top", type=int, default=0)
    parser.add_argument("--evidence-selection", choices=EVIDENCE_SELECTIONS, default="dense")
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument("--neighbor-children", type=int, default=0)
    parser.add_argument("--document-floor", action="store_true")
    parser.add_argument("--packing", choices=("stop", "skip", "fill"), default="stop")
    parser.add_argument("--cache-verifier-model")
    parser.add_argument("--cache-verifier-base-url")
    parser.add_argument("--cache-verifier-api-key-env", default="")


def resolve_config(args, *, mode, legacy_cache=False, legacy_source_order=False,
                   legacy_overflow="unsupported", **budgets):
    # Older programmatic callers supply a Namespace containing only their original options.
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    add_arguments(parser)
    for key, value in vars(parser.parse_args([])).items():
        if not hasattr(args, key):
            setattr(args, key, value)
    common = args.execution_profile == "common"
    source_order = common or legacy_source_order
    order = args.evidence_order or ("source" if source_order else "score")
    cache = legacy_cache and not common
    return Config(
        mode=mode, profile=args.execution_profile,
        direct_overflow=args.direct_overflow or ("unsupported" if common else legacy_overflow),
        evidence_order=order,
        merge_overlaps=(order == "source") if args.merge_overlaps is None else args.merge_overlaps,
        exact_offsets=common or legacy_source_order or order == "source",
        child_tokens=getattr(args, "child_tokens", 7500),
        child_overlap_tokens=getattr(args, "child_overlap_tokens", 750),
        cache_read=cache if args.answer_cache_read is None else args.answer_cache_read,
        cache_write=cache if args.answer_cache_write is None else args.answer_cache_write,
        cache_matching=args.cache_matching, rerank_top=args.pipeline_rerank_top,
        evidence_selection=args.evidence_selection, selection_seed=args.selection_seed,
        neighbor_children=args.neighbor_children, document_floor=args.document_floor, packing=args.packing,
        max_retries=getattr(args, "max_retries", 1), **budgets,
    )
