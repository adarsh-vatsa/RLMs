"""Structured memoization primitives for recursive language-model agents.

This module is intentionally independent from the heavier semantic cache stack.
It implements the first practical slice of "dynamic programming for language":
exact task/scope replay, negative memoization, and compositional coverage over
addressable context intervals.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


REUSE_EXACT_ANSWER = "exact_answer"
REUSE_SUPPORTING_FACT = "supporting_fact"
REUSE_SEARCH_HINT = "search_hint"
REUSE_AGGREGATION_COMPONENT = "aggregation_component"
REUSE_RULED_OUT = "ruled_out"

VERIFIER_NOT_REQUIRED = "not_required"
VERIFIER_UNVERIFIED = "unverified"
VERIFIER_SUPPORTED = "supported"
VERIFIER_REJECTED = "rejected"


def _normalize_text(value: str) -> str:
    return " ".join((value or "").strip().lower().split())


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _stable_hash(value: Any, length: int = 16) -> str:
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    return digest[:length]


def coerce_confidence(value: Any, default: float = 1.0) -> float:
    """Coerce numeric or label-style confidence values into [0, 1]."""
    if isinstance(value, str):
        label = value.strip().lower()
        label_map = {
            "very high": 0.95,
            "high": 0.9,
            "medium": 0.6,
            "moderate": 0.6,
            "low": 0.3,
            "very low": 0.1,
            "none": 0.0,
            "unknown": default,
        }
        if label in label_map:
            return max(0.0, min(1.0, float(label_map[label])))
        if label.endswith("%"):
            try:
                return max(0.0, min(1.0, float(label[:-1].strip()) / 100.0))
            except ValueError:
                pass
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = float(default)
    if 1.0 < numeric <= 100.0:
        numeric = numeric / 100.0
    return max(0.0, min(1.0, numeric))


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


_SEARCH_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "which",
    "who",
    "with",
}


def _search_tokens(value: str) -> Tuple[str, ...]:
    tokens = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", (value or "").lower())
    return tuple(token for token in tokens if token not in _SEARCH_STOPWORDS and len(token) > 1)


@dataclass(frozen=True)
class TaskSpec:
    """A normalized description of the language subproblem being memoized."""

    prompt: str
    task_type: str = "generic"
    output_contract: str = "text"
    constraints: Mapping[str, Any] = field(default_factory=dict)

    @property
    def normalized_prompt(self) -> str:
        return _normalize_text(self.prompt)

    def signature_payload(self) -> Dict[str, Any]:
        return {
            "prompt": self.normalized_prompt,
            "task_type": _normalize_text(self.task_type),
            "output_contract": _normalize_text(self.output_contract),
            "constraints": dict(self.constraints),
        }

    def signature(self) -> str:
        return _stable_hash(self.signature_payload())

    def exact_match(self, other: "TaskSpec") -> bool:
        return self.signature_payload() == other.signature_payload()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "task_type": self.task_type,
            "output_contract": self.output_contract,
            "constraints": dict(self.constraints),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaskSpec":
        return cls(
            prompt=str(data.get("prompt", "")),
            task_type=str(data.get("task_type", "generic")),
            output_contract=str(data.get("output_contract", "text")),
            constraints=dict(data.get("constraints", {})),
        )


@dataclass(frozen=True)
class ContextScope:
    """An addressable interval of a corpus.

    The interval is half-open: [start, end). The unit can be "chunk", "char",
    "token", or any caller-defined coordinate system, as long as a store uses it
    consistently.
    """

    corpus_id: str
    document_id: str
    start: int
    end: int
    unit: str = "chunk"
    content_hash: str = ""

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("ContextScope.start must be non-negative")
        if self.end < self.start:
            raise ValueError("ContextScope.end must be greater than or equal to start")

    @property
    def length(self) -> int:
        return self.end - self.start

    def signature_payload(self) -> Dict[str, Any]:
        return {
            "corpus_id": self.corpus_id,
            "document_id": self.document_id,
            "start": self.start,
            "end": self.end,
            "unit": self.unit,
            "content_hash": self.content_hash,
        }

    def signature(self) -> str:
        return _stable_hash(self.signature_payload())

    def compatible_with(self, other: "ContextScope") -> bool:
        if (
            self.corpus_id != other.corpus_id
            or self.document_id != other.document_id
            or self.unit != other.unit
        ):
            return False
        if self.content_hash and other.content_hash and self.content_hash != other.content_hash:
            return False
        return True

    def exact_match(self, other: "ContextScope") -> bool:
        return self.compatible_with(other) and self.start == other.start and self.end == other.end

    def contains(self, other: "ContextScope") -> bool:
        return self.compatible_with(other) and self.start <= other.start and self.end >= other.end

    def overlaps(self, other: "ContextScope") -> bool:
        return self.compatible_with(other) and max(self.start, other.start) < min(self.end, other.end)

    def clipped_to(self, outer: "ContextScope") -> Optional["ContextScope"]:
        if not self.overlaps(outer):
            return None
        return ContextScope(
            corpus_id=outer.corpus_id,
            document_id=outer.document_id,
            start=max(self.start, outer.start),
            end=min(self.end, outer.end),
            unit=outer.unit,
            content_hash=outer.content_hash,
        )

    def subtract_covered(self, covered: Iterable["ContextScope"]) -> List["ContextScope"]:
        """Return sub-scopes of this interval not covered by compatible scopes."""
        intervals: List[Tuple[int, int]] = []
        for scope in covered:
            clipped = scope.clipped_to(self)
            if clipped is not None and clipped.length > 0:
                intervals.append((clipped.start, clipped.end))

        if not intervals:
            return [self] if self.length > 0 else []

        intervals.sort()
        merged: List[Tuple[int, int]] = []
        for start, end in intervals:
            if not merged or start > merged[-1][1]:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))

        missing: List[ContextScope] = []
        cursor = self.start
        for start, end in merged:
            if cursor < start:
                missing.append(
                    ContextScope(
                        corpus_id=self.corpus_id,
                        document_id=self.document_id,
                        start=cursor,
                        end=start,
                        unit=self.unit,
                        content_hash=self.content_hash,
                    )
                )
            cursor = max(cursor, end)
        if cursor < self.end:
            missing.append(
                ContextScope(
                    corpus_id=self.corpus_id,
                    document_id=self.document_id,
                    start=cursor,
                    end=self.end,
                    unit=self.unit,
                    content_hash=self.content_hash,
                )
            )
        return missing

    def to_dict(self) -> Dict[str, Any]:
        return self.signature_payload()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ContextScope":
        return cls(
            corpus_id=str(data.get("corpus_id", "")),
            document_id=str(data.get("document_id", "")),
            start=int(data.get("start", 0)),
            end=int(data.get("end", 0)),
            unit=str(data.get("unit", "chunk")),
            content_hash=str(data.get("content_hash", "")),
        )


@dataclass(frozen=True)
class EvidenceSpan:
    """Source evidence that supports a memoized result."""

    document_id: str
    start: int
    end: int
    unit: str = "char"
    text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "start": self.start,
            "end": self.end,
            "unit": self.unit,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceSpan":
        return cls(
            document_id=str(data.get("document_id", "")),
            start=int(data.get("start", 0)),
            end=int(data.get("end", 0)),
            unit=str(data.get("unit", "char")),
            text=str(data.get("text", "")),
        )


@dataclass(frozen=True)
class ContextChunk:
    """A persisted raw context chunk addressable by corpus/document/chunk index."""

    corpus_id: str
    document_id: str
    chunk_index: int
    text: str
    unit: str = "chunk"
    content_hash: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "corpus_id": self.corpus_id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "unit": self.unit,
            "content_hash": self.content_hash,
            "metadata": dict(self.metadata),
        }


@dataclass
class MemoEntry:
    """A reusable unit of language-model work."""

    task: TaskSpec
    scope: ContextScope
    result: str
    result_type: str = "answer"
    reusable_as: Sequence[str] = field(default_factory=lambda: (REUSE_EXACT_ANSWER,))
    evidence: Sequence[EvidenceSpan] = field(default_factory=tuple)
    dependencies: Sequence[str] = field(default_factory=tuple)
    confidence: float = 1.0
    verifier_status: str = VERIFIER_NOT_REQUIRED
    metadata: Mapping[str, Any] = field(default_factory=dict)
    entry_id: str = ""
    created_at: str = ""

    def __post_init__(self) -> None:
        self.reusable_as = tuple(dict.fromkeys(str(v) for v in self.reusable_as))
        self.evidence = tuple(self.evidence)
        self.dependencies = tuple(str(v) for v in self.dependencies)
        self.metadata = dict(self.metadata)
        self.confidence = coerce_confidence(self.confidence)
        if not self.created_at:
            self.created_at = _utc_now()
        if not self.entry_id:
            self.entry_id = self._derive_id()

    def _derive_id(self) -> str:
        return _stable_hash(
            {
                "task": self.task.signature_payload(),
                "scope": self.scope.signature_payload(),
                "result": self.result,
                "result_type": self.result_type,
                "reusable_as": list(self.reusable_as),
                "dependencies": list(self.dependencies),
            },
            length=24,
        )

    @property
    def is_rejected(self) -> bool:
        return self.verifier_status == VERIFIER_REJECTED

    @property
    def is_negative(self) -> bool:
        return REUSE_RULED_OUT in self.reusable_as or self.result_type == "not_found"

    @property
    def fragment_kind(self) -> str:
        """Return the planner-facing role of this memo fragment."""
        if self.is_negative:
            return "ruled_out_region"
        if REUSE_SEARCH_HINT in self.reusable_as:
            return "search_hint"
        if REUSE_SUPPORTING_FACT in self.reusable_as:
            return "supporting_fact"
        if REUSE_AGGREGATION_COMPONENT in self.reusable_as:
            return "aggregation_component"
        if REUSE_EXACT_ANSWER in self.reusable_as:
            return "exact_answer"
        return self.result_type or "unknown"

    def can_cover_scope(self) -> bool:
        cover_modes = {
            REUSE_EXACT_ANSWER,
            REUSE_SUPPORTING_FACT,
            REUSE_AGGREGATION_COMPONENT,
            REUSE_RULED_OUT,
        }
        return not self.is_rejected and bool(cover_modes.intersection(self.reusable_as))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "created_at": self.created_at,
            "task": self.task.to_dict(),
            "scope": self.scope.to_dict(),
            "result": self.result,
            "result_type": self.result_type,
            "reusable_as": list(self.reusable_as),
            "evidence": [item.to_dict() for item in self.evidence],
            "dependencies": list(self.dependencies),
            "confidence": self.confidence,
            "verifier_status": self.verifier_status,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemoEntry":
        return cls(
            entry_id=str(data.get("entry_id", "")),
            created_at=str(data.get("created_at", "")),
            task=TaskSpec.from_dict(data.get("task", {})),
            scope=ContextScope.from_dict(data.get("scope", {})),
            result=str(data.get("result", "")),
            result_type=str(data.get("result_type", "answer")),
            reusable_as=tuple(data.get("reusable_as", (REUSE_EXACT_ANSWER,))),
            evidence=tuple(EvidenceSpan.from_dict(item) for item in data.get("evidence", [])),
            dependencies=tuple(data.get("dependencies", ())),
            confidence=coerce_confidence(data.get("confidence", 1.0)),
            verifier_status=str(data.get("verifier_status", VERIFIER_NOT_REQUIRED)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class MemoCandidate:
    """A candidate memo entry returned by a cheap retrieval pass."""

    entry: MemoEntry
    score: float
    matched_terms: Tuple[str, ...] = ()


@dataclass(frozen=True)
class MemoReusePlan:
    """Planner output for a requested task and scope."""

    task: TaskSpec
    scope: ContextScope
    exact_entry: Optional[MemoEntry] = None
    reusable_entries: Tuple[MemoEntry, ...] = ()
    negative_entries: Tuple[MemoEntry, ...] = ()
    hint_entries: Tuple[MemoEntry, ...] = ()
    missing_scopes: Tuple[ContextScope, ...] = ()

    @property
    def has_exact_replay(self) -> bool:
        return self.exact_entry is not None

    @property
    def is_complete(self) -> bool:
        return self.has_exact_replay or len(self.missing_scopes) == 0

    @property
    def requires_model_calls(self) -> bool:
        return not self.is_complete

    @property
    def covered_scopes(self) -> Tuple[ContextScope, ...]:
        return tuple(entry.scope for entry in self.reusable_entries + self.negative_entries)

    @staticmethod
    def _merge_intervals(intervals: Sequence[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
        if not intervals:
            return ()
        sorted_intervals = sorted(intervals)
        merged: List[Tuple[int, int]] = []
        for start, end in sorted_intervals:
            if end <= start:
                continue
            if not merged or start > merged[-1][1]:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        return tuple(merged)

    @property
    def covered_intervals(self) -> Tuple[Tuple[int, int], ...]:
        if self.has_exact_replay:
            return ((self.scope.start, self.scope.end),) if self.scope.length > 0 else ()

        intervals: List[Tuple[int, int]] = []
        for entry in self.reusable_entries + self.negative_entries:
            if not entry.can_cover_scope():
                continue
            clipped = entry.scope.clipped_to(self.scope)
            if clipped is not None and clipped.length > 0:
                intervals.append((clipped.start, clipped.end))
        return self._merge_intervals(intervals)

    @property
    def requested_length(self) -> int:
        return self.scope.length

    @property
    def covered_length(self) -> int:
        return sum(end - start for start, end in self.covered_intervals)

    @property
    def missing_length(self) -> int:
        return sum(scope.length for scope in self.missing_scopes)

    @property
    def coverage_ratio(self) -> float:
        if self.requested_length <= 0:
            return 1.0
        return min(1.0, max(0.0, self.covered_length / float(self.requested_length)))

    @staticmethod
    def _count_fragment_kinds(entries: Sequence[MemoEntry]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for entry in entries:
            counts[entry.fragment_kind] = counts.get(entry.fragment_kind, 0) + 1
        return counts

    def to_telemetry(self) -> Dict[str, Any]:
        """Return JSON-safe planner telemetry for tests and benchmark rows."""
        covering_entries = tuple(
            entry
            for entry in self.reusable_entries + self.negative_entries
            if entry.can_cover_scope()
        )
        all_entries = self.reusable_entries + self.negative_entries + self.hint_entries
        return {
            "task_signature": self.task.signature(),
            "scope": self.scope.to_dict(),
            "scope_length": self.scope.length,
            "has_exact_replay": self.has_exact_replay,
            "is_complete": self.is_complete,
            "requires_model_calls": self.requires_model_calls,
            "coverage_ratio": round(self.coverage_ratio, 6),
            "covered_length": self.covered_length,
            "missing_length": self.missing_length,
            "covered_intervals": [list(item) for item in self.covered_intervals],
            "missing_scopes": [scope.to_dict() for scope in self.missing_scopes],
            "exact_entry_id": self.exact_entry.entry_id if self.exact_entry else None,
            "reusable_entry_ids": [entry.entry_id for entry in self.reusable_entries],
            "negative_entry_ids": [entry.entry_id for entry in self.negative_entries],
            "hint_entry_ids": [entry.entry_id for entry in self.hint_entries],
            "reusable_count": len(self.reusable_entries),
            "negative_count": len(self.negative_entries),
            "hint_count": len(self.hint_entries),
            "fragment_kind_counts": self._count_fragment_kinds(all_entries),
            "covering_fragment_kind_counts": self._count_fragment_kinds(covering_entries),
            "evidence_span_count": sum(len(entry.evidence) for entry in all_entries),
            "dependency_edge_count": sum(len(entry.dependencies) for entry in all_entries),
        }


class MemoStore:
    """In-memory store with JSON persistence and structural reuse planning."""

    def __init__(self, entries: Iterable[MemoEntry] = ()) -> None:
        self.entries: Dict[str, MemoEntry] = {}
        for entry in entries:
            self.add(entry)

    def add(self, entry: MemoEntry) -> MemoEntry:
        self.entries[entry.entry_id] = entry
        return entry

    def reject_entry(self, entry: MemoEntry, *, reason: str = "rejected") -> MemoEntry:
        """Mark an entry as rejected while preserving it for audit/history."""
        metadata = dict(entry.metadata)
        metadata["rejection_reason"] = reason
        metadata["rejected_at"] = _utc_now()
        entry.metadata = metadata
        entry.verifier_status = VERIFIER_REJECTED
        return self.add(entry)

    def parents(self, entry_id: str) -> List[MemoEntry]:
        """Return entries that depend on the given entry."""
        return [
            entry
            for entry in self.entries.values()
            if str(entry_id) in entry.dependencies
        ]

    def children(self, entry_id: str) -> List[MemoEntry]:
        """Return dependency entries used by the given entry."""
        entry = self.entries.get(str(entry_id))
        if entry is None:
            return []
        return [
            self.entries[dependency_id]
            for dependency_id in entry.dependencies
            if dependency_id in self.entries
        ]

    def invalidate_scope(
        self,
        scope: ContextScope,
        *,
        task: Optional[TaskSpec] = None,
        reason: str = "scope invalidated",
        propagate_dependencies: bool = True,
    ) -> Tuple[MemoEntry, ...]:
        """Reject non-rejected memo entries that overlap a changed context scope.

        Passing a scope with an empty content hash invalidates matching document
        ranges regardless of version. Passing a specific content hash only
        invalidates entries compatible with that version.
        """
        queue = list(self.candidates_for(task, scope) if task is not None else self.scope_candidates(scope))
        rejected: List[MemoEntry] = []
        seen: set[str] = set()

        while queue:
            entry = queue.pop(0)
            if entry.entry_id in seen or entry.is_rejected:
                continue
            seen.add(entry.entry_id)
            rejected_entry = self.reject_entry(entry, reason=reason)
            rejected.append(rejected_entry)
            if propagate_dependencies:
                queue.extend(parent for parent in self.parents(entry.entry_id) if not parent.is_rejected)
        return tuple(rejected)

    def add_answer(
        self,
        task: TaskSpec,
        scope: ContextScope,
        result: str,
        *,
        evidence: Sequence[EvidenceSpan] = (),
        confidence: float = 1.0,
        dependencies: Sequence[str] = (),
        reusable_as: Sequence[str] = (REUSE_EXACT_ANSWER,),
        verifier_status: str = VERIFIER_NOT_REQUIRED,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> MemoEntry:
        return self.add(
            MemoEntry(
                task=task,
                scope=scope,
                result=result,
                result_type="answer",
                reusable_as=reusable_as,
                evidence=evidence,
                dependencies=dependencies,
                confidence=confidence,
                verifier_status=verifier_status,
                metadata=metadata or {},
            )
        )

    def add_negative(
        self,
        task: TaskSpec,
        scope: ContextScope,
        *,
        reason: str = "not_found",
        confidence: float = 1.0,
        evidence: Sequence[EvidenceSpan] = (),
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> MemoEntry:
        merged_metadata = dict(metadata or {})
        merged_metadata.setdefault("reason", reason)
        return self.add(
            MemoEntry(
                task=task,
                scope=scope,
                result=reason,
                result_type="not_found",
                reusable_as=(REUSE_RULED_OUT,),
                evidence=evidence,
                confidence=confidence,
                verifier_status=VERIFIER_NOT_REQUIRED,
                metadata=merged_metadata,
            )
        )

    def find_exact(self, task: TaskSpec, scope: ContextScope) -> Optional[MemoEntry]:
        for entry in self.entries.values():
            if (
                not entry.is_rejected
                and REUSE_EXACT_ANSWER in entry.reusable_as
                and entry.task.exact_match(task)
                and entry.scope.exact_match(scope)
            ):
                return entry
        return None

    def candidates_for(self, task: TaskSpec, scope: ContextScope) -> List[MemoEntry]:
        candidates = []
        for entry in self.entries.values():
            if entry.is_rejected:
                continue
            if not entry.task.exact_match(task):
                continue
            if not entry.scope.compatible_with(scope):
                continue
            if entry.scope.overlaps(scope) or entry.scope.contains(scope) or scope.contains(entry.scope):
                candidates.append(entry)
        candidates.sort(key=lambda item: (item.scope.start, item.scope.end, item.entry_id))
        return candidates

    def scope_candidates(self, scope: ContextScope) -> List[MemoEntry]:
        """Return non-rejected entries with compatible overlapping scope, regardless of task."""
        candidates = []
        for entry in self.entries.values():
            if entry.is_rejected:
                continue
            if not entry.scope.compatible_with(scope):
                continue
            if entry.scope.overlaps(scope) or entry.scope.contains(scope) or scope.contains(entry.scope):
                candidates.append(entry)
        candidates.sort(key=lambda item: (item.scope.start, item.scope.end, item.entry_id))
        return candidates

    def _candidate_text(self, entry: MemoEntry) -> str:
        evidence_text = " ".join(item.text for item in entry.evidence)
        metadata_text = _canonical_json(dict(entry.metadata)) if entry.metadata else ""
        return " ".join(
            [
                entry.task.prompt,
                entry.task.task_type,
                entry.task.output_contract,
                entry.result,
                " ".join(entry.reusable_as),
                evidence_text,
                metadata_text,
            ]
        )

    def ranked_text_candidates(
        self,
        query: str,
        scope: Optional[ContextScope] = None,
        limit: int = 20,
    ) -> List[MemoCandidate]:
        """Cheap candidate generation over memo entries.

        This is intentionally not a verifier. It only gathers plausible memo
        entries for a later planner/verifier to inspect.
        """
        query_terms = set(_search_tokens(query))
        if scope is not None:
            entries = self.scope_candidates(scope)
        else:
            entries = [entry for entry in self.entries.values() if not entry.is_rejected]

        candidates: List[MemoCandidate] = []
        for entry in entries:
            entry_terms = set(_search_tokens(self._candidate_text(entry)))
            matched = tuple(sorted(query_terms.intersection(entry_terms)))
            score = float(len(matched))
            if scope is not None and entry.scope.compatible_with(scope):
                if entry.scope.exact_match(scope):
                    score += 3.0
                elif entry.scope.overlaps(scope) or entry.scope.contains(scope) or scope.contains(entry.scope):
                    score += 2.0
            if REUSE_EXACT_ANSWER in entry.reusable_as:
                score += 0.25
            if REUSE_SUPPORTING_FACT in entry.reusable_as:
                score += 0.25
            if score <= 0:
                continue
            candidates.append(MemoCandidate(entry=entry, score=score, matched_terms=matched))

        candidates.sort(
            key=lambda item: (
                item.score,
                item.entry.confidence,
                -item.entry.scope.start,
                item.entry.entry_id,
            ),
            reverse=True,
        )
        return candidates[: max(0, int(limit))]

    def _dedupe_covering_entries(self, entries: Sequence[MemoEntry]) -> Tuple[MemoEntry, ...]:
        """Keep one covering entry per exact scope to avoid double aggregation."""
        by_scope: Dict[Tuple[str, str, int, int, str, str], MemoEntry] = {}

        def rank(entry: MemoEntry) -> Tuple[int, int, float]:
            exact = 1 if REUSE_EXACT_ANSWER in entry.reusable_as else 0
            dependency_count = len(entry.dependencies)
            return (exact, dependency_count, entry.confidence)

        for entry in entries:
            key = (
                entry.scope.corpus_id,
                entry.scope.document_id,
                entry.scope.start,
                entry.scope.end,
                entry.scope.unit,
                entry.scope.content_hash,
            )
            existing = by_scope.get(key)
            if existing is None or rank(entry) > rank(existing):
                by_scope[key] = entry

        return tuple(sorted(by_scope.values(), key=lambda item: (item.scope.start, item.scope.end, item.entry_id)))

    def plan_reuse(self, task: TaskSpec, scope: ContextScope) -> MemoReusePlan:
        exact = self.find_exact(task, scope)
        if exact is not None:
            return MemoReusePlan(
                task=task,
                scope=scope,
                exact_entry=exact,
                reusable_entries=(exact,),
                missing_scopes=(),
            )

        reusable: List[MemoEntry] = []
        negative: List[MemoEntry] = []
        hints: List[MemoEntry] = []

        for entry in self.candidates_for(task, scope):
            if REUSE_SEARCH_HINT in entry.reusable_as:
                hints.append(entry)
            if entry.is_negative:
                negative.append(entry)
            elif entry.can_cover_scope():
                reusable.append(entry)

        reusable = list(self._dedupe_covering_entries(reusable))
        covered = [entry.scope for entry in reusable + negative if entry.can_cover_scope()]
        missing = tuple(scope.subtract_covered(covered))

        return MemoReusePlan(
            task=task,
            scope=scope,
            exact_entry=None,
            reusable_entries=tuple(reusable),
            negative_entries=tuple(negative),
            hint_entries=tuple(hints),
            missing_scopes=missing,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"entries": [entry.to_dict() for entry in self.entries.values()]}

    def stats(self) -> Dict[str, Any]:
        entries = list(self.entries.values())

        def increment(counts: Dict[str, int], key: str) -> None:
            counts[key] = counts.get(key, 0) + 1

        by_fragment_kind: Dict[str, int] = {}
        by_result_type: Dict[str, int] = {}
        by_verifier_status: Dict[str, int] = {}
        by_reuse_mode: Dict[str, int] = {}
        dependency_edges = 0
        evidence_spans = 0
        confidence_sum = 0.0

        for entry in entries:
            increment(by_fragment_kind, entry.fragment_kind)
            increment(by_result_type, entry.result_type)
            increment(by_verifier_status, entry.verifier_status)
            for mode in entry.reusable_as:
                increment(by_reuse_mode, mode)
            dependency_edges += len(entry.dependencies)
            evidence_spans += len(entry.evidence)
            confidence_sum += entry.confidence

        return {
            "entry_count": len(entries),
            "by_fragment_kind": by_fragment_kind,
            "by_result_type": by_result_type,
            "by_verifier_status": by_verifier_status,
            "by_reuse_mode": by_reuse_mode,
            "dependency_edge_count": dependency_edges,
            "evidence_span_count": evidence_spans,
            "average_confidence": round(confidence_sum / max(len(entries), 1), 6),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemoStore":
        return cls(MemoEntry.from_dict(item) for item in data.get("entries", []))

    def save(self, path: Path) -> Path:
        path = Path(path)
        if path.suffix:
            out_path = path
            out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)
            out_path = path / "memo_entries.json"
        out_path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return out_path

    @classmethod
    def load(cls, path: Path) -> "MemoStore":
        path = Path(path)
        in_path = path if path.suffix else path / "memo_entries.json"
        return cls.from_dict(json.loads(in_path.read_text(encoding="utf-8")))


class DuckDBMemoStore(MemoStore):
    """DuckDB-backed memo store for larger local runs.

    DuckDB handles durable structured lookup. Python still owns the interval
    coverage math and reuse planning so behavior stays aligned with MemoStore.
    """

    def __init__(self, database_path: Path | str = ":memory:") -> None:
        self.database_path = str(database_path)
        self._duckdb = self._load_duckdb()
        self.conn = self._duckdb.connect(self.database_path)
        self._ensure_schema()
        self.entries = _DuckDBEntryView(self)

    @staticmethod
    def _load_duckdb():
        try:
            import duckdb
        except ImportError as exc:
            raise RuntimeError(
                "DuckDBMemoStore requires the optional 'duckdb' package"
            ) from exc
        return duckdb

    def _ensure_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memo_entries (
                entry_id TEXT PRIMARY KEY,
                task_signature TEXT NOT NULL,
                prompt TEXT NOT NULL,
                normalized_prompt TEXT NOT NULL,
                task_type TEXT NOT NULL,
                output_contract TEXT NOT NULL,
                constraints_json TEXT NOT NULL,
                corpus_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                scope_start BIGINT NOT NULL,
                scope_end BIGINT NOT NULL,
                unit TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                result TEXT NOT NULL,
                result_type TEXT NOT NULL,
                reusable_as_json TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                dependencies_json TEXT NOT NULL,
                confidence DOUBLE NOT NULL,
                verifier_status TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                entry_json TEXT NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memo_lookup
            ON memo_entries (
                task_signature,
                corpus_id,
                document_id,
                unit,
                content_hash,
                scope_start,
                scope_end
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memo_task
            ON memo_entries (task_signature)
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memo_edges (
                parent_entry_id TEXT NOT NULL,
                child_entry_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                PRIMARY KEY (parent_entry_id, child_entry_id, edge_type)
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memo_edges_child
            ON memo_edges (child_entry_id)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memo_edges_parent
            ON memo_edges (parent_entry_id)
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS context_chunks (
                corpus_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                chunk_index BIGINT NOT NULL,
                unit TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                text TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (corpus_id, document_id, unit, content_hash, chunk_index)
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_context_chunks_scope
            ON context_chunks (corpus_id, document_id, unit, content_hash, chunk_index)
            """
        )

    def _row_from_entry(self, entry: MemoEntry) -> Tuple[Any, ...]:
        constraints_json = _canonical_json(dict(entry.task.constraints))
        reusable_json = _canonical_json(list(entry.reusable_as))
        evidence_json = _canonical_json([item.to_dict() for item in entry.evidence])
        dependencies_json = _canonical_json(list(entry.dependencies))
        metadata_json = _canonical_json(dict(entry.metadata))
        entry_json = json.dumps(entry.to_dict(), sort_keys=True, ensure_ascii=False)
        return (
            entry.entry_id,
            entry.task.signature(),
            entry.task.prompt,
            entry.task.normalized_prompt,
            entry.task.task_type,
            entry.task.output_contract,
            constraints_json,
            entry.scope.corpus_id,
            entry.scope.document_id,
            entry.scope.start,
            entry.scope.end,
            entry.scope.unit,
            entry.scope.content_hash,
            entry.result,
            entry.result_type,
            reusable_json,
            evidence_json,
            dependencies_json,
            entry.confidence,
            entry.verifier_status,
            metadata_json,
            entry.created_at,
            entry_json,
        )

    def add(self, entry: MemoEntry) -> MemoEntry:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO memo_entries VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            self._row_from_entry(entry),
        )
        self.conn.execute(
            "DELETE FROM memo_edges WHERE parent_entry_id = ? AND edge_type = 'depends_on'",
            (entry.entry_id,),
        )
        for dependency_id in entry.dependencies:
            self.add_edge(
                parent_entry_id=entry.entry_id,
                child_entry_id=dependency_id,
                edge_type="depends_on",
                metadata={"source": "entry.dependencies"},
            )
        return entry

    def add_edge(
        self,
        parent_entry_id: str,
        child_entry_id: str,
        edge_type: str = "depends_on",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO memo_edges VALUES (?, ?, ?, ?, ?)
            """,
            (
                str(parent_entry_id),
                str(child_entry_id),
                str(edge_type),
                _utc_now(),
                _canonical_json(dict(metadata or {})),
            ),
        )

    def graph_edges(self, entry_id: Optional[str] = None, direction: str = "out") -> List[Dict[str, Any]]:
        where = ""
        params: Tuple[Any, ...] = ()
        if entry_id is not None:
            if direction == "in":
                where = "WHERE child_entry_id = ?"
            else:
                where = "WHERE parent_entry_id = ?"
            params = (entry_id,)
        rows = self.conn.execute(
            f"""
            SELECT parent_entry_id, child_entry_id, edge_type, created_at, metadata_json
            FROM memo_edges
            {where}
            ORDER BY created_at, parent_entry_id, child_entry_id, edge_type
            """,
            params,
        ).fetchall()
        return [
            {
                "parent_entry_id": row[0],
                "child_entry_id": row[1],
                "edge_type": row[2],
                "created_at": row[3],
                "metadata": json.loads(row[4]),
            }
            for row in rows
        ]

    def parents(self, entry_id: str) -> List[MemoEntry]:
        rows = self.conn.execute(
            """
            SELECT e.entry_json
            FROM memo_edges edge
            JOIN memo_entries e ON e.entry_id = edge.parent_entry_id
            WHERE edge.child_entry_id = ?
            ORDER BY edge.created_at, edge.parent_entry_id
            """,
            (entry_id,),
        ).fetchall()
        return [self._entry_from_json(row[0]) for row in rows]

    def children(self, entry_id: str) -> List[MemoEntry]:
        rows = self.conn.execute(
            """
            SELECT e.entry_json
            FROM memo_edges edge
            JOIN memo_entries e ON e.entry_id = edge.child_entry_id
            WHERE edge.parent_entry_id = ?
            ORDER BY edge.created_at, edge.child_entry_id
            """,
            (entry_id,),
        ).fetchall()
        return [self._entry_from_json(row[0]) for row in rows]

    def lineage(self, entry_id: str, max_depth: int = 10) -> Dict[str, Any]:
        seen = set()

        def walk(current_id: str, depth: int) -> Dict[str, Any]:
            node = {"entry_id": current_id, "children": []}
            if depth >= max_depth or current_id in seen:
                return node
            seen.add(current_id)
            for child in self.children(current_id):
                node["children"].append(walk(child.entry_id, depth + 1))
            return node

        return walk(entry_id, 0)

    def _entry_from_json(self, raw: str) -> MemoEntry:
        return MemoEntry.from_dict(json.loads(raw))

    def _all_entries(self) -> List[MemoEntry]:
        rows = self.conn.execute(
            "SELECT entry_json FROM memo_entries ORDER BY created_at, entry_id"
        ).fetchall()
        return [self._entry_from_json(row[0]) for row in rows]

    def upsert_context_chunks(
        self,
        *,
        corpus_id: str,
        document_id: str,
        chunks: Sequence[str],
        content_hash: str = "",
        unit: str = "chunk",
        start_index: int = 0,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> int:
        """Persist raw context chunks for later structured retrieval."""
        rows = []
        for offset, text in enumerate(chunks):
            rows.append(
                (
                    str(corpus_id),
                    str(document_id),
                    int(start_index + offset),
                    str(unit),
                    str(content_hash or ""),
                    str(text),
                    _canonical_json(dict(metadata or {})),
                    _utc_now(),
                )
            )
        if not rows:
            return 0
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO context_chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return len(rows)

    def _context_chunk_from_row(self, row: Tuple[Any, ...]) -> ContextChunk:
        return ContextChunk(
            corpus_id=str(row[0]),
            document_id=str(row[1]),
            chunk_index=int(row[2]),
            unit=str(row[3]),
            content_hash=str(row[4]),
            text=str(row[5]),
            metadata=json.loads(row[6]),
        )

    def fetch_context_range(
        self,
        *,
        corpus_id: str,
        document_id: str,
        start: int,
        end: int,
        content_hash: str = "",
        unit: str = "chunk",
    ) -> List[ContextChunk]:
        """Fetch a half-open raw context chunk range from DuckDB."""
        rows = self.conn.execute(
            """
            SELECT corpus_id, document_id, chunk_index, unit, content_hash, text, metadata_json
            FROM context_chunks
            WHERE corpus_id = ?
              AND document_id = ?
              AND unit = ?
              AND (content_hash = ? OR content_hash = '' OR ? = '')
              AND chunk_index >= ?
              AND chunk_index < ?
            ORDER BY chunk_index
            """,
            (
                str(corpus_id),
                str(document_id),
                str(unit),
                str(content_hash or ""),
                str(content_hash or ""),
                int(start),
                int(end),
            ),
        ).fetchall()
        return [self._context_chunk_from_row(row) for row in rows]

    def search_context_chunks(
        self,
        query: str,
        *,
        corpus_id: str,
        document_id: Optional[str] = None,
        content_hash: Optional[str] = None,
        unit: str = "chunk",
        limit: int = 20,
    ) -> List[Tuple[ContextChunk, float, Tuple[str, ...]]]:
        """Lexically retrieve persisted context chunks with parameterized SQL."""
        tokens = _search_tokens(query)
        if not tokens:
            return []

        score_parts = []
        score_params: List[Any] = []
        for token in tokens:
            score_parts.append("CASE WHEN lower(text) LIKE ? THEN 1 ELSE 0 END")
            score_params.append(f"%{token}%")
        score_sql = " + ".join(score_parts)

        where = ["corpus_id = ?", "unit = ?"]
        where_params: List[Any] = [str(corpus_id), str(unit)]
        if document_id is not None:
            where.append("document_id = ?")
            where_params.append(str(document_id))
        if content_hash is not None:
            where.append("(content_hash = ? OR content_hash = '' OR ? = '')")
            where_params.extend([str(content_hash or ""), str(content_hash or "")])

        rows = self.conn.execute(
            f"""
            SELECT *
            FROM (
                SELECT
                    corpus_id,
                    document_id,
                    chunk_index,
                    unit,
                    content_hash,
                    text,
                    metadata_json,
                    ({score_sql}) AS score
                FROM context_chunks
                WHERE {" AND ".join(where)}
            )
            WHERE score > 0
            ORDER BY score DESC, document_id, chunk_index
            LIMIT ?
            """,
            tuple(score_params + where_params + [int(limit)]),
        ).fetchall()
        return [
            (
                self._context_chunk_from_row(row[:7]),
                float(row[7]),
                tuple(token for token in tokens if token in str(row[5]).lower()),
            )
            for row in rows
        ]

    def context_chunk_count(self, corpus_id: Optional[str] = None) -> int:
        if corpus_id is None:
            return int(self.conn.execute("SELECT count(*) FROM context_chunks").fetchone()[0])
        return int(
            self.conn.execute(
                "SELECT count(*) FROM context_chunks WHERE corpus_id = ?",
                (str(corpus_id),),
            ).fetchone()[0]
        )

    def find_exact(self, task: TaskSpec, scope: ContextScope) -> Optional[MemoEntry]:
        rows = self.conn.execute(
            """
            SELECT entry_json
            FROM memo_entries
            WHERE task_signature = ?
              AND corpus_id = ?
              AND document_id = ?
              AND scope_start = ?
              AND scope_end = ?
              AND unit = ?
              AND (content_hash = ? OR content_hash = '' OR ? = '')
            ORDER BY created_at, entry_id
            """,
            (
                task.signature(),
                scope.corpus_id,
                scope.document_id,
                scope.start,
                scope.end,
                scope.unit,
                scope.content_hash,
                scope.content_hash,
            ),
        ).fetchall()
        for row in rows:
            entry = self._entry_from_json(row[0])
            if (
                not entry.is_rejected
                and REUSE_EXACT_ANSWER in entry.reusable_as
                and entry.task.exact_match(task)
                and entry.scope.exact_match(scope)
            ):
                return entry
        return None

    def candidates_for(self, task: TaskSpec, scope: ContextScope) -> List[MemoEntry]:
        rows = self.conn.execute(
            """
            SELECT entry_json
            FROM memo_entries
            WHERE task_signature = ?
              AND corpus_id = ?
              AND document_id = ?
              AND unit = ?
              AND (content_hash = ? OR content_hash = '' OR ? = '')
              AND scope_start < ?
              AND scope_end > ?
            ORDER BY scope_start, scope_end, entry_id
            """,
            (
                task.signature(),
                scope.corpus_id,
                scope.document_id,
                scope.unit,
                scope.content_hash,
                scope.content_hash,
                scope.end,
                scope.start,
            ),
        ).fetchall()
        candidates = [self._entry_from_json(row[0]) for row in rows]
        return [
            entry
            for entry in candidates
            if not entry.is_rejected
            and entry.task.exact_match(task)
            and entry.scope.compatible_with(scope)
            and (entry.scope.overlaps(scope) or entry.scope.contains(scope) or scope.contains(entry.scope))
        ]

    def scope_candidates(self, scope: ContextScope) -> List[MemoEntry]:
        rows = self.conn.execute(
            """
            SELECT entry_json
            FROM memo_entries
            WHERE corpus_id = ?
              AND document_id = ?
              AND unit = ?
              AND (content_hash = ? OR content_hash = '' OR ? = '')
              AND scope_start < ?
              AND scope_end > ?
            ORDER BY scope_start, scope_end, entry_id
            """,
            (
                scope.corpus_id,
                scope.document_id,
                scope.unit,
                scope.content_hash,
                scope.content_hash,
                scope.end,
                scope.start,
            ),
        ).fetchall()
        candidates = [self._entry_from_json(row[0]) for row in rows]
        return [
            entry
            for entry in candidates
            if not entry.is_rejected
            and entry.scope.compatible_with(scope)
            and (entry.scope.overlaps(scope) or entry.scope.contains(scope) or scope.contains(entry.scope))
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {"entries": [entry.to_dict() for entry in self._all_entries()]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DuckDBMemoStore":
        store = cls(":memory:")
        for item in data.get("entries", []):
            store.add(MemoEntry.from_dict(item))
        return store

    def save(self, path: Path) -> Path:
        path = Path(path)
        if path.suffix == ".duckdb":
            if self.database_path == ":memory:":
                target = DuckDBMemoStore(path)
                for entry in self._all_entries():
                    target.add(entry)
                target.close()
            return path
        return super().save(path)

    def close(self) -> None:
        self.conn.close()


class _DuckDBEntryView:
    """Small mapping-like view so existing len(store.entries) code works."""

    def __init__(self, store: DuckDBMemoStore) -> None:
        self.store = store

    def __len__(self) -> int:
        return int(self.store.conn.execute("SELECT count(*) FROM memo_entries").fetchone()[0])

    def values(self) -> List[MemoEntry]:
        return self.store._all_entries()

    def __iter__(self):
        return iter(entry.entry_id for entry in self.store._all_entries())

    def __contains__(self, entry_id: str) -> bool:
        row = self.store.conn.execute(
            "SELECT 1 FROM memo_entries WHERE entry_id = ? LIMIT 1",
            (entry_id,),
        ).fetchone()
        return row is not None
