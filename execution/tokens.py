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


def route(mode, full_tokens, budget, overflow="unsupported"):
    if full_tokens <= budget:
        return "direct_fit"
    if mode == "hybrid":
        return "dense_child_packed"
    if overflow == "error":
        raise ValueError(f"Rendered prompt uses {full_tokens} tokens, exceeding the {budget}-token input budget")
    return "middle_truncated" if overflow == "middle" else "unsupported_context"


def truncate_middle(tokenizer, messages, budget):
    """Legacy head/tail truncation of the final user message, preserving other roles."""
    result = [dict(message) for message in messages]
    ids = token_ids(tokenizer.encode(result[-1]["content"], add_special_tokens=False))
    result[-1]["content"] = ""
    keep = max(0, min(len(ids), budget - chat_token_count(tokenizer, result)))
    while True:
        head, tail = keep // 2, keep - keep // 2
        result[-1]["content"] = tokenizer.decode(ids[:head] + (ids[-tail:] if tail else []), skip_special_tokens=True)
        count = chat_token_count(tokenizer, result)
        if count <= budget:
            return result
        if keep == 0:
            raise ValueError("Fixed instructions and chat template exceed the input budget")
        keep = max(0, keep - max(1, count - budget))
