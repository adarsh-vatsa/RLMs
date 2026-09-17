"""Preserve MRCR text prompts and pack evidence in source order."""


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


def pack_evidence(tokenizer, prefix: str, body: str, question: str,
                  results: list[dict], max_input_tokens: int) -> tuple[list[dict], dict]:
    from execution.contracts import Document
    from execution.packing import pack, render_document_slices

    document = Document("source", body)
    request, info = pack(tokenizer,
        lambda evidence: messages(prefix + render_document_slices(document, evidence, OMITTED) + question),
        (document,), results, max_input_tokens)
    info["selected_evidence_ranges"] = [
        {"char_start": item["char_start"], "char_end": item["char_end"]}
        for item in info["selected_evidence_ranges"]]
    return request, info
