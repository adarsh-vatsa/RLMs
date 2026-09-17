"""Preserve MRCR text prompts and pack evidence in source order."""

from aa_lcr.prompting import chat_token_count


PROMPT_VERSION = "mrcr_v2_source_order_v1"
FEWSHOT_END = "======== EXAMPLE 3 ========\n\n"
OMITTED = "\n\n[... omitted conversation ...]\n\n"


def messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]


def split_prompt(prompt: str, question: str) -> tuple[str, str]:
    if not question or not prompt.endswith(question):
        raise ValueError("MRCR view_ops must be an exact suffix of queries")
    if not question.startswith("User: Prepend ") or not question.rstrip().endswith("Assistant:"):
        raise ValueError("Unsupported MRCR final instruction format")
    prefix, separator, body = prompt[:-len(question)].partition(FEWSHOT_END)
    if not separator or not prefix.startswith("Here are some examples of conversations"):
        raise ValueError("Unsupported MRCR few-shot format; expected text-style EXAMPLE 3")
    if not body.lstrip().startswith("User: "):
        raise ValueError("Missing MRCR conversation body")
    return prefix + separator, body


def merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def render_evidence(prefix: str, body: str, question: str, ranges: list[tuple[int, int]]) -> str:
    parts = [prefix]
    cursor = 0
    for start, end in ranges:
        if start > cursor:
            parts.append(OMITTED)
        parts.append(body[start:end])
        cursor = end
    if cursor < len(body):
        parts.append(OMITTED)
    parts.append(question)
    return "".join(parts)


def pack_evidence(tokenizer, prefix: str, body: str, question: str,
                  results: list[dict], max_input_tokens: int) -> tuple[list[dict], dict]:
    selected = []
    ranges = []
    for result in results:
        metadata = result["metadata"]
        start, end = metadata.get("char_start"), metadata.get("char_end")
        if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(body):
            raise ValueError("Retrieved child is missing valid source offsets")
        if result["text"] != body[start:end]:
            raise ValueError("Retrieved text does not match its source offsets")
        candidate_ranges = merge_ranges([*ranges, (start, end)])
        candidate = messages(render_evidence(prefix, body, question, candidate_ranges))
        if chat_token_count(tokenizer, candidate) > max_input_tokens:
            break
        ranges = candidate_ranges
        selected.append(metadata.get("child_index", metadata.get("chunk_index")))
    if not selected:
        raise ValueError("No retrieved child fits the MRCR input budget")
    request = messages(render_evidence(prefix, body, question, ranges))
    return request, {
        "final_rendered_input_tokens": chat_token_count(tokenizer, request),
        "selected_child_indices": selected,
        "selected_evidence_ranges": [{"char_start": start, "char_end": end} for start, end in ranges],
        "candidate_count": len(results),
    }
