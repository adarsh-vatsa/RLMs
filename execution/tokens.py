from collections.abc import Mapping


def token_ids(value):
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


def chat_token_count(tokenizer, messages):
    return len(token_ids(tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, enable_thinking=False,
    )))


TRUNCATION_ROUTES = {"middle": "middle_truncated", "middle-marked": "middle_marked_truncated", "head": "head_truncated"}
TRUNCATION_MARKER = "\n\n[... omitted ...]\n\n"


def route(mode, full_tokens, budget, overflow="unsupported"):
    if full_tokens <= budget:
        return "direct_fit"
    if mode == "hybrid":
        return "dense_child_packed"
    if overflow == "error":
        raise ValueError(f"Rendered prompt uses {full_tokens} tokens, exceeding the {budget}-token input budget")
    return TRUNCATION_ROUTES.get(overflow, "unsupported_context")


def truncate_prompt(tokenizer, messages, budget, policy="middle"):
    """Truncate the final user message, preserving other roles.

    `middle` joins an equal head and tail silently (the LongBench protocol),
    `middle-marked` puts TRUNCATION_MARKER at the cut, and `head` keeps only the start.
    """
    result = [dict(message) for message in messages]
    ids = token_ids(tokenizer.encode(result[-1]["content"], add_special_tokens=False))
    marker = TRUNCATION_MARKER if policy == "middle-marked" else ""
    result[-1]["content"] = marker
    keep = max(0, min(len(ids), budget - chat_token_count(tokenizer, result)))
    while True:
        head = keep if policy == "head" else keep // 2
        tail = ids[len(ids) - (keep - head):] if keep > head else []
        if marker:
            text = (tokenizer.decode(ids[:head], skip_special_tokens=True) + marker
                    + tokenizer.decode(tail, skip_special_tokens=True))
        else:
            text = tokenizer.decode(ids[:head] + tail, skip_special_tokens=True)
        result[-1]["content"] = text
        count = chat_token_count(tokenizer, result)
        if count <= budget:
            return result
        if keep == 0:
            raise ValueError("Fixed instructions and chat template exceed the input budget")
        keep = max(0, keep - max(1, count - budget))
