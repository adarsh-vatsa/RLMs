"""High-parity NoLiMa data expansion utilities.

This module reproduces the core NoLiMa benchmark mechanics needed by this
repository: needle-set expansion, haystack placement across depth sweeps,
and deterministic sample generation.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple


DEFAULT_TASK_TEMPLATE = (
    "You will answer a question based on the following book snippet:\n\n{haystack}\n\n"
    "Use the information provided in the book snippet to answer the question. "
    "Your answer should be short and based on either explicitly stated facts "
    "or strong, logical inferences.\n\nQuestion: {question}\n\n"
    "Return only the final answer with no additional explanation or reasoning."
)


def _coerce_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value).strip()
    return ""


def _coerce_list_of_text(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            text = _coerce_text(item)
            if text:
                out.append(text)
        return out
    text = _coerce_text(value)
    return [text] if text else []


def _replace_args(template: object, args: Sequence[object]) -> str:
    text = _coerce_text(template)
    if not text:
        return ""
    for i, arg in enumerate(args, start=1):
        text = text.replace("{" + str(i) + "}", _coerce_text(arg))
    return text


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def depth_percent_values(intervals: int) -> List[float]:
    if intervals <= 1:
        return [0.0]
    step = 100.0 / float(intervals - 1)
    return [round(i * step, 6) for i in range(intervals)]


@dataclass(frozen=True)
class NoLiMaCase:
    exp_id: str
    test_name: str
    question_type: str
    needle: str
    retrieval_question: str
    gold_answers: List[str]
    character_set: List[str]
    distractor: Optional[str]
    system_prompt: str
    task_template: str
    seed_offset: int


@dataclass(frozen=True)
class HaystackAsset:
    path: Path
    name: str
    sha256: str
    total_chars: int


def load_needle_set_cases(
    needle_set_path: Path,
    max_cases: int = 0,
) -> Tuple[List[NoLiMaCase], str]:
    """Expand needle-set definitions into concrete benchmark cases."""
    needle_set_path = Path(needle_set_path)
    if not needle_set_path.exists():
        raise FileNotFoundError(f"Needle-set file not found: {needle_set_path}")

    needle_set_hash = _sha256_file(needle_set_path)
    raw = json.loads(needle_set_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Needle-set JSON must be a list of experiment configs")

    cases: List[NoLiMaCase] = []
    for exp_idx, exp_cfg in enumerate(raw, start=1):
        if not isinstance(exp_cfg, dict):
            continue

        exp_id = _coerce_text(exp_cfg.get("id")) or f"{exp_idx:04d}"
        seed_match = re.match(r"(\d+)", exp_id)
        seed_offset = int(seed_match.group(1)) if seed_match else exp_idx

        system_prompt = _coerce_text(exp_cfg.get("system_prompt"))
        task_template = _coerce_text(exp_cfg.get("task_template")) or DEFAULT_TASK_TEMPLATE
        character_set = _coerce_list_of_text(exp_cfg.get("character_set"))

        questions = exp_cfg.get("questions")
        tests = exp_cfg.get("tests")
        distractors = exp_cfg.get("distractors") if isinstance(exp_cfg.get("distractors"), dict) else {}

        if not isinstance(questions, dict) or not isinstance(tests, dict):
            continue

        needle_template = _coerce_text(exp_cfg.get("needle"))

        for question_type, question_template in questions.items():
            question_type_text = _coerce_text(question_type)
            question_template_text = _coerce_text(question_template)
            if not question_type_text or not question_template_text:
                continue

            for test_id, test_cfg in tests.items():
                if not isinstance(test_cfg, dict):
                    continue

                input_args = test_cfg.get("input_args")
                if not isinstance(input_args, list):
                    input_args = []

                full_needle = _replace_args(needle_template, input_args)
                full_question = _replace_args(question_template_text, input_args)

                distractor_template = distractors.get(question_type_text)
                full_distractor = None
                if distractor_template is not None:
                    full_distractor = _replace_args(distractor_template, input_args)
                    if not full_distractor:
                        full_distractor = None

                gold_answers = _coerce_list_of_text(test_cfg.get("gold_answers"))
                test_name = f"{exp_id}_{_coerce_text(test_id)}_{question_type_text}"

                if not full_needle or not full_question:
                    continue

                cases.append(
                    NoLiMaCase(
                        exp_id=exp_id,
                        test_name=test_name,
                        question_type=question_type_text,
                        needle=full_needle,
                        retrieval_question=full_question,
                        gold_answers=gold_answers,
                        character_set=character_set,
                        distractor=full_distractor,
                        system_prompt=system_prompt,
                        task_template=task_template,
                        seed_offset=seed_offset,
                    )
                )

    if max_cases > 0:
        cases = cases[:max_cases]

    if not cases:
        raise ValueError("No runnable cases were produced from the needle-set file")

    return cases, needle_set_hash


def list_haystack_assets(haystack_dir: Path, max_haystacks: int = 0) -> List[HaystackAsset]:
    """Collect and hash haystack files used for placement."""
    haystack_dir = Path(haystack_dir)
    if not haystack_dir.exists():
        raise FileNotFoundError(f"Haystack directory not found: {haystack_dir}")

    assets: List[HaystackAsset] = []
    for path in sorted(haystack_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        assets.append(
            HaystackAsset(
                path=path,
                name=path.stem,
                sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                total_chars=len(text),
            )
        )

    if max_haystacks > 0:
        assets = assets[:max_haystacks]

    if not assets:
        raise ValueError(f"No .txt haystack files found under: {haystack_dir}")

    return assets


class BookHaystack:
    """Approximate NoLiMa book haystack placement mechanics for this repo."""

    def __init__(self, book_path: Path) -> None:
        self.path = Path(book_path)
        self.text = self.path.read_text(encoding="utf-8", errors="ignore")
        if not self.text.strip():
            raise ValueError(f"Haystack file is empty: {self.path}")

        self._token_spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", self.text)]
        if not self._token_spans:
            raise ValueError(f"Haystack has no tokens: {self.path}")
        self._token_starts = [start for start, _ in self._token_spans]

        newline_char_positions = [0]
        for idx, ch in enumerate(self.text):
            if ch == "\n":
                newline_char_positions.append(min(idx + 1, len(self.text)))

        valid_char_positions: List[int] = []
        valid_token_positions: List[int] = []
        seen = set()
        for char_pos in newline_char_positions:
            token_pos = bisect_left(self._token_starts, char_pos)
            token_pos = min(max(token_pos, 0), len(self._token_spans) - 1)
            key = (char_pos, token_pos)
            if key in seen:
                continue
            seen.add(key)
            valid_char_positions.append(char_pos)
            valid_token_positions.append(token_pos)

        self._valid_char_positions = valid_char_positions
        self._valid_token_positions = valid_token_positions

    def get_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def total_tokens(self) -> int:
        return len(self._token_spans)

    def _choose_anchor_index(
        self,
        context_length: int,
        depth: float,
        shift: int,
        static_depth: float,
    ) -> Tuple[int, float]:
        if static_depth < 0:
            denominator = float(max(context_length + 1, 1))
            best_idx = None
            best_delta = None
            for idx, token_pos in enumerate(self._valid_token_positions):
                delta = (float(token_pos - shift) / denominator) - depth
                if delta < 0:
                    continue
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    best_idx = idx
            if best_idx is None:
                best_idx = len(self._valid_token_positions) - 1
            token_pos = self._valid_token_positions[best_idx]
            return best_idx, float(token_pos) / float(max(self.total_tokens, 1))

        if shift > 0:
            raise ValueError("Shift is not supported with static depth")

        target = static_depth * float(max(self.total_tokens - 1, 1))
        best_idx = 0
        best_gap = None
        for idx, token_pos in enumerate(self._valid_token_positions):
            gap = abs(float(token_pos) - target)
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best_idx = idx

        token_pos = self._valid_token_positions[best_idx]
        return best_idx, float(token_pos) / float(max(self.total_tokens, 1))

    def _slice_with_needle(
        self,
        anchor_index: int,
        needle: str,
        context_length: int,
        depth: float,
    ) -> Tuple[str, int]:
        anchor_token = self._valid_token_positions[anchor_index]
        anchor_char = self._valid_char_positions[anchor_index]

        context_len = max(1, int(context_length))
        start_token = anchor_token - int(round(context_len * depth))
        start_token = max(0, start_token)

        end_token = start_token + context_len
        if end_token > self.total_tokens:
            end_token = self.total_tokens
            start_token = max(0, end_token - context_len)

        if end_token <= start_token:
            end_token = min(self.total_tokens, start_token + 1)

        start_char = self._token_spans[start_token][0]
        end_char = self._token_spans[end_token - 1][1]
        anchor_char = max(start_char, min(anchor_char, end_char))

        pre_haystack = self.text[start_char:anchor_char]
        if "\n" in pre_haystack:
            pre_haystack = pre_haystack[: pre_haystack.rfind("\n") + 1]
        post_haystack = self.text[anchor_char:end_char]

        merged = f"{pre_haystack} {needle}\n{post_haystack}".strip()
        token_depth = max(0, anchor_token - start_token)
        return merged, token_depth

    def _generate_w_needle_placement(
        self,
        needle: str,
        context_length: int,
        shift: int = 0,
        depth: float = 0.5,
        static_depth: float = -1,
    ) -> Dict[str, object]:
        anchor_index, real_static_depth = self._choose_anchor_index(
            context_length=context_length,
            depth=depth,
            shift=shift,
            static_depth=static_depth,
        )
        text, token_depth = self._slice_with_needle(
            anchor_index=anchor_index,
            needle=needle,
            context_length=context_length,
            depth=depth,
        )
        return {
            "text": text,
            "static_depth": real_static_depth,
            "token_depth": token_depth,
            "depth": depth,
            "context_length_wo_needle": int(context_length),
        }

    def generate_w_needle_placement(
        self,
        needle: str,
        context_length: int,
        shift: int = 0,
        depth: float = 0.5,
        static_depth: float = -1,
        distractor: Optional[str] = None,
        distractor_free_zone: float = 0.2,
        rng: Optional[random.Random] = None,
    ) -> Dict[str, object]:
        """Generate one placement sample in the haystack for the target depth."""
        base = self._generate_w_needle_placement(
            needle=needle,
            context_length=context_length,
            shift=shift,
            depth=depth,
            static_depth=static_depth,
        )

        if not distractor:
            return base

        if static_depth >= 0:
            raise ValueError("Static depth placement with distractor is not supported")

        if distractor_free_zone < 0 or distractor_free_zone > 0.25:
            raise ValueError("Distractor free zone must be in [0, 0.25]")

        rand = rng or random
        left_available = max(0.0, depth - 2.0 * distractor_free_zone)
        right_available = max(0.0, 1.0 - (depth + 2.0 * distractor_free_zone))
        span = left_available + right_available

        if span <= 0:
            distractor_depth = 0.8 if depth <= 0.5 else 0.2
        else:
            draw = rand.random() * span
            if draw > left_available:
                distractor_depth = depth + distractor_free_zone + (draw - left_available)
            else:
                distractor_depth = draw + distractor_free_zone

        text = _coerce_text(base.get("text"))
        token_spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        if token_spans:
            token_idx = int(round(distractor_depth * float(max(len(token_spans) - 1, 1))))
            token_idx = max(0, min(token_idx, len(token_spans) - 1))
            insert_at = token_spans[token_idx][0]
        else:
            insert_at = 0

        text_with_distractor = (
            text[:insert_at] + " " + distractor + "\n" + text[insert_at:]
        ).strip()

        base["text"] = text_with_distractor
        base["distractor_depth"] = round(distractor_depth, 6)
        return base


def build_dataset_signature(
    needle_set_hash: str,
    cases: Sequence[NoLiMaCase],
    haystacks: Sequence[HaystackAsset],
    lengths: Sequence[int],
    depth_intervals: int,
    seed: int,
) -> str:
    """Build a stable signature for cache namespace and manifest reproducibility."""
    hasher = hashlib.sha256()
    payload = {
        "needle_set_hash": needle_set_hash,
        "depth_intervals": int(depth_intervals),
        "seed": int(seed),
        "lengths": list(lengths),
        "cases": [
            {
                "test_name": c.test_name,
                "question_type": c.question_type,
                "needle": c.needle,
                "retrieval_question": c.retrieval_question,
                "gold_answers": c.gold_answers,
                "character_set": c.character_set,
                "distractor": c.distractor,
            }
            for c in cases
        ],
        "haystacks": [
            {
                "name": h.name,
                "sha256": h.sha256,
                "total_chars": h.total_chars,
            }
            for h in haystacks
        ],
    }
    hasher.update(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return hasher.hexdigest()[:16]


def iter_nolima_samples(
    cases: Sequence[NoLiMaCase],
    haystacks: Sequence[HaystackAsset],
    lengths: Sequence[int],
    depth_intervals: int,
    seed: int,
    shift: int = 0,
    static_depth: float = -1,
) -> Iterator[Dict[str, object]]:
    """Yield deterministic NoLiMa evaluation samples for all placements."""
    depth_values = depth_percent_values(depth_intervals)

    for haystack_no, haystack in enumerate(haystacks):
        book = BookHaystack(haystack.path)

        for case in cases:
            case_rng = random.Random(seed + case.seed_offset + haystack_no)

            for length in lengths:
                for depth_idx, depth_percent in enumerate(depth_values):
                    needle = case.needle
                    retrieval_question = case.retrieval_question
                    expected_answers = list(case.gold_answers)
                    selected_character = ""

                    if "{CHAR}" in needle or "{CHAR}" in retrieval_question:
                        if not case.character_set:
                            raise ValueError(
                                f"Case {case.test_name} requires character_set but none is provided"
                            )
                        selected_character = str(case_rng.choice(case.character_set))
                        needle = needle.replace("{CHAR}", selected_character)
                        retrieval_question = retrieval_question.replace("{CHAR}", selected_character)
                        expected_answers = [selected_character]

                    placement = book.generate_w_needle_placement(
                        needle=needle,
                        context_length=int(length),
                        shift=shift,
                        depth=float(depth_percent) / 100.0,
                        static_depth=static_depth,
                        distractor=case.distractor,
                        rng=case_rng,
                    )

                    expected_value: object
                    if not expected_answers:
                        expected_value = ""
                    elif len(expected_answers) == 1:
                        expected_value = expected_answers[0]
                    else:
                        expected_value = expected_answers

                    sample_id = (
                        f"{case.test_name}__{haystack.name}__L{int(length)}"
                        f"__D{depth_idx:02d}"
                    )

                    yield {
                        "id": sample_id,
                        "task": case.question_type,
                        "test_name": case.test_name,
                        "exp_id": case.exp_id,
                        "length": str(int(length)),
                        "depth_percent": depth_percent,
                        "depth_index": depth_idx,
                        "question": retrieval_question,
                        "context": _coerce_text(placement.get("text")),
                        "expected_answer": expected_value,
                        "gold_answers": expected_answers,
                        "selected_character": selected_character,
                        "haystack_name": haystack.name,
                        "haystack_path": str(haystack.path),
                        "haystack_hash": haystack.sha256,
                        "needle": needle,
                        "question_type": case.question_type,
                        "system_prompt": case.system_prompt,
                        "task_template": case.task_template,
                        "distractor": case.distractor or "",
                        "placement_metadata": {
                            "static_depth": placement.get("static_depth"),
                            "token_depth": placement.get("token_depth"),
                            "context_length_wo_needle": placement.get("context_length_wo_needle"),
                            "distractor_depth": placement.get("distractor_depth"),
                        },
                    }
