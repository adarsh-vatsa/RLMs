"""Shared strict-Qwen prompt and tokenizer helpers for LongBench-v2."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


STRICT_MCQ_SYSTEM_PROMPT = (
    "You are solving a long-context multiple-choice question using ONLY the "
    "provided documents. The query includes choices A, B, C, and D. Silently "
    "check each option against the documents before answering. The correct "
    "choice must satisfy every constraint in the question and every substantive "
    "claim in the answer choice. It must be directly supported by the documents, "
    "not just compatible with them. Reject choices that are only partially "
    "supported, too narrow, too broad, overstate the evidence, add unsupported "
    "causal claims, skip required implications, or are merely mentioned in the "
    "documents. If more than one option seems plausible, choose the option best "
    "supported by the overall evidence and the exact wording of the question. "
    "Return exactly one capital letter: A, B, C, or D. Do not explain."
)


def token_ids(value: Any) -> list[int]:
    """Extract a single sequence of token IDs from common HF result shapes."""
    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise ValueError("Tokenizer result does not contain input_ids")
        value = value["input_ids"]
    elif hasattr(value, "input_ids"):
        value = value.input_ids
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return list(value)


def chat_token_count(tokenizer: Any, messages: list[dict]) -> int:
    """Count the rendered non-thinking Qwen chat request exactly."""
    return len(
        token_ids(
            tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                enable_thinking=False,
            )
        )
    )


def build_strict_mcq_messages(context: str, query: str) -> list[dict]:
    return [
        {"role": "system", "content": STRICT_MCQ_SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\n{query}"},
    ]

