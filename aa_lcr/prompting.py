"""Official AA-LCR prompting, grading, and token-budget helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


PROMPT_VERSION = "aa_lcr_official_v1"
GRADER_PROMPT_VERSION = "aa_lcr_equality_v1"
GRADER_PROMPT_VERSIONS = (GRADER_PROMPT_VERSION, "aa_lcr_equality_v1.1")

# ArtificialAnalysis/AA-LCR README at 9a77ef56b717057ade24ceab4d273712a0b4f19e.
GRADER_SYSTEM_V1_1 = """Decide whether the CANDIDATE ANSWER is correct or incorrect against the OFFICIAL ANSWER.
Note the following points when assessing correctness:

- Numbers should still match when they are the same value written differently, e.g., a
  percentage, a count of percentage points, and the equivalent decimal fraction are the same
  value: 0.675, "67.5%" and "67.5 percentage points" all match. So do different scales
  (thousand, million, bn) and different notations (thousands separators, currency symbols,
  LaTeX markup, and numbers written as words).
- Where the question asks for a particular format (e.g., a percentage, a number of decimal
  places, a unit, a rounding, or an ordering) the CANDIDATE ANSWER must meet it. If the
  question asks for no particular format, accept any equivalent form.
- In cases where the question asks for an ordered list, a title, honorific or article added
  to an entry in the CANDIDATE ANSWER can change where that entry sorts. Accept the ordering
  if it is correct either with those additions or without them.
- Grade the value the CANDIDATE ANSWER finally commits to, and it must commit to one. Values
  reached while working, and alternatives it considers and sets aside, do not count. If it
  offers several values without selecting one, it is incorrect even if one of them is right.
  Hedging is fine as long as one clearly definitive answer is given."""

GRADER_USER_V1_1 = """Assess whether the following CANDIDATE ANSWER is CORRECT or INCORRECT.
For the CANDIDATE ANSWER to be correct, it must be consistent with the OFFICIAL ANSWER.

The question, for reference only: START QUESTION {question}

END QUESTION

The OFFICIAL ANSWER: {official_answer}

END OFFICIAL ANSWER

BEGIN CANDIDATE ANSWER TO ASSESS

{candidate_answer}

END CANDIDATE ANSWER TO ASSESS

Reply as JSON, with a verdict of CORRECT or INCORRECT."""


def build_prompt(documents: list[str], question: str) -> str:
    documents_text = "\n\n".join(
        f"BEGIN DOCUMENT {index}:\n{document}\nEND DOCUMENT {index}"
        for index, document in enumerate(documents, start=1)
    )
    return (
        "BEGIN INPUT DOCUMENTS\n\n"
        f"{documents_text}\n\n"
        "END INPUT DOCUMENTS\n\n"
        "Answer the following question using the input documents provided above.\n"
        "START QUESTION\n\n"
        f"{question}\n\n"
        "END QUESTION\n"
    )


def build_messages(documents: list[str], question: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": build_prompt(documents, question)}]


def build_grader_prompt(
    question: str, official_answer: str, candidate_answer: str
) -> str:
    return (
        "Assess whether the following CANDIDATE ANSWER is CORRECT or INCORRECT.\n"
        "For the CANDIDATE ANSWER to be correct, it must be consistent with the "
        "OFFICIAL ANSWER.\n\n"
        f"The question, for reference only: {question}\n"
        f"The OFFICIAL ANSWER: {official_answer}\n"
        f"CANDIDATE ANSWER TO ASSESS: {candidate_answer}\n\n"
        "Reply only with CORRECT or INCORRECT."
    )


def build_grader_messages(
    question: str,
    official_answer: str,
    candidate_answer: str,
    version: str = GRADER_PROMPT_VERSION,
) -> list[dict[str, str]]:
    if version == GRADER_PROMPT_VERSION:
        return [
            {
                "role": "user",
                "content": build_grader_prompt(
                    question, official_answer, candidate_answer
                ),
            }
        ]
    if version != "aa_lcr_equality_v1.1":
        raise ValueError(f"Unknown grader prompt version: {version}")
    return [
        {"role": "system", "content": GRADER_SYSTEM_V1_1},
        {
            "role": "user",
            "content": GRADER_USER_V1_1.format(
                question=question,
                official_answer=official_answer,
                candidate_answer=candidate_answer,
            ),
        },
    ]


def parse_grade(text: str, version: str = GRADER_PROMPT_VERSION) -> str:
    if version == "aa_lcr_equality_v1.1":
        try:
            payload = json.loads(text)
        except (ValueError, TypeError):
            return ""
        if not isinstance(payload, dict) or not isinstance(payload.get("verdict"), str):
            return ""
        return (
            payload["verdict"] if payload["verdict"] in {"CORRECT", "INCORRECT"} else ""
        )
    if version != GRADER_PROMPT_VERSION:
        raise ValueError(f"Unknown grader prompt version: {version}")
    grade = str(text or "").strip().upper()
    return grade if grade in {"CORRECT", "INCORRECT"} else ""


def non_thinking_extra_body(*, grader: bool = False) -> dict:
    body: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": False}}
    if grader:
        body["structured_outputs"] = {"choice": ["CORRECT", "INCORRECT"]}
    return body


def token_ids(value: Any) -> list[int]:
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


def chat_token_count(tokenizer: Any, messages: list[dict[str, str]]) -> int:
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


def prepare_direct_messages(
    documents: list[str],
    question: str,
    tokenizer: Any,
    max_input_tokens: int,
    *,
    allow_truncation: bool,
) -> tuple[list[dict[str, str]], dict]:
    messages = build_messages(documents, question)
    original_tokens = chat_token_count(tokenizer, messages)
    if original_tokens <= max_input_tokens:
        return messages, {
            "prompt_truncated": False,
            "prompt_tokens_before_truncation": original_tokens,
            "prompt_tokens_after_truncation": original_tokens,
            "prompt_tokens_removed": 0,
        }
    if not allow_truncation:
        raise ValueError(
            f"Rendered prompt uses {original_tokens} tokens, exceeding the "
            f"{max_input_tokens}-token input budget"
        )

    user_ids = token_ids(
        tokenizer.encode(messages[0]["content"], add_special_tokens=False)
    )
    overhead = chat_token_count(tokenizer, [{"role": "user", "content": ""}])
    keep_tokens = min(len(user_ids), max(0, max_input_tokens - overhead))
    while True:
        head = keep_tokens // 2
        tail = keep_tokens - head
        kept_ids = user_ids[:head] + (user_ids[-tail:] if tail else [])
        content = tokenizer.decode(kept_ids, skip_special_tokens=True)
        truncated = [{"role": "user", "content": content}]
        rendered_tokens = chat_token_count(tokenizer, truncated)
        if rendered_tokens <= max_input_tokens:
            break
        if keep_tokens == 0:
            raise ValueError(
                "Chat template overhead exceeds the configured input budget"
            )
        keep_tokens = max(0, keep_tokens - max(1, rendered_tokens - max_input_tokens))

    return truncated, {
        "prompt_truncated": True,
        "prompt_tokens_before_truncation": original_tokens,
        "prompt_tokens_after_truncation": rendered_tokens,
        "prompt_tokens_removed": original_tokens - rendered_tokens,
    }


def pack_retrieved_children(
    tokenizer: Any,
    question: str,
    results: list[dict],
    max_input_tokens: int,
) -> tuple[list[dict[str, str]], dict]:
    selected: list[dict] = []
    rendered_tokens = chat_token_count(tokenizer, build_messages([], question))
    for result in results:
        candidate = [*selected, result]
        messages = build_messages([item["text"] for item in candidate], question)
        candidate_tokens = chat_token_count(tokenizer, messages)
        if candidate_tokens > max_input_tokens:
            break
        selected = candidate
        rendered_tokens = candidate_tokens
    if not selected:
        raise RuntimeError("No retrieved child fits within the configured input budget")

    ranges = []
    for result in selected:
        metadata = result.get("metadata") or {}
        ranges.append(
            {
                "filename": metadata.get("filename"),
                "child_index": metadata.get("child_index", metadata.get("chunk_index")),
                "token_start": metadata.get("token_start"),
                "token_end": metadata.get("token_end"),
                "char_start": metadata.get("char_start"),
                "char_end": metadata.get("char_end"),
                "score": round(float(result.get("score") or 0.0), 8),
            }
        )
    return build_messages([item["text"] for item in selected], question), {
        "rendered_input_tokens": rendered_tokens,
        "faiss_candidate_count": len(results),
        "selected_child_count": len(selected),
        "dropped_child_count": len(results) - len(selected),
        "selected_evidence_ranges": ranges,
    }


def prompt_contract_metadata(grader_version: str = GRADER_PROMPT_VERSION) -> dict:
    if grader_version == GRADER_PROMPT_VERSION:
        grader_template = build_grader_prompt(
            "{question}", "{official_answer}", "{candidate_answer}"
        )
    else:
        grader_template = json.dumps(
            build_grader_messages(
                "{question}", "{official_answer}", "{candidate_answer}", grader_version
            ),
            ensure_ascii=False,
            sort_keys=True,
        )
    return {
        "prompt_version": PROMPT_VERSION,
        "prompt_template_sha256": hashlib.sha256(
            build_prompt(["{document}"], "{question}").encode("utf-8")
        ).hexdigest(),
        "grader_prompt_version": grader_version,
        "grader_prompt_template_sha256": hashlib.sha256(
            grader_template.encode("utf-8")
        ).hexdigest(),
    }
