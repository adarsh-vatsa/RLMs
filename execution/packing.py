from execution.tokens import chat_token_count


OMITTED = "\n\n[... omitted source ...]\n\n"
SLACK_TOKENS = 16


def source_ranges(documents, selected, merge=True):
    positions = {doc.id: (index, doc.text) for index, doc in enumerate(documents)}
    ranges = []
    for result in selected:
        meta = result["metadata"]
        doc_id = meta.get("document_id", meta.get("filename", documents[0].id))
        if doc_id not in positions:
            raise ValueError("Retrieved child refers to an unknown document")
        start, end = meta.get("char_start"), meta.get("char_end")
        body = positions[doc_id][1]
        if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(body):
            raise ValueError("Retrieved child is missing valid source offsets")
        if result["text"] != body[start:end]:
            raise ValueError("Retrieved text does not match its source offsets")
        ranges.append({"document_id": doc_id, "char_start": start, "char_end": end})
    ranges.sort(key=lambda item: (positions[item["document_id"]][0], item["char_start"]))
    merged = []
    for item in ranges:
        if (merge and merged and merged[-1]["document_id"] == item["document_id"]
                and item["char_start"] <= merged[-1]["char_end"]):
            merged[-1]["char_end"] = max(merged[-1]["char_end"], item["char_end"])
        else:
            merged.append(dict(item))
    return [dict(item, text=positions[item["document_id"]][1][item["char_start"]:item["char_end"]]) for item in merged]


def render_document_slices(document, evidence, separator=OMITTED):
    parts = []
    cursor = 0
    for item in evidence:
        if item["document_id"] != document.id:
            continue
        if item["char_start"] != cursor:
            parts.append(separator)
        parts.append(item["text"])
        cursor = item["char_end"]
    if cursor < len(document.text):
        parts.append(separator)
    return "".join(parts)


def _uncovered(covered, start, end):
    pieces = []
    for left, right in covered:
        if left >= end:
            break
        if right <= start:
            continue
        if left > start:
            pieces.append((start, left))
        start = max(start, right)
    if start < end:
        pieces.append((start, end))
    return pieces


def _cover(covered, start, end):
    merged = []
    for left, right in sorted([*covered, (start, end)]):
        if merged and left <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
        else:
            merged.append((left, right))
    return merged


def pack(tokenizer, render, documents, results, budget, *, order="source", merge=True, policy="stop"):
    """Pack ranked results into the budget.

    `stop` ends at the first result that does not fit, `skip` keeps trying lower-ranked
    results, and `fill` also trims the best remaining result into the leftover budget.
    The complete prompt is re-counted only when cheap token bounds cannot decide. The
    bounds allow `overhead` tokens per result for document headers, omission markers and
    tokenization at fragment boundaries, so decisions match counting every step.
    """
    if not results:
        raise ValueError("No retrieved child fits the configured input budget")
    merged = order == "source" and merge
    renders = 0
    selected, covered = [], {}
    empty = chat_token_count(tokenizer, [{"role": "user", "content": ""}])

    def text_tokens(text):
        return chat_token_count(tokenizer, [{"role": "user", "content": text}]) - empty if text else 0

    def evidence_for(items):
        return source_ranges(documents, items, merge) if order == "source" else [
            {**item.get("metadata", {}), "text": item["text"], "score": item.get("score", 0)} for item in items]

    def count(items):
        nonlocal renders
        renders += 1
        return chat_token_count(tokenizer, render(evidence_for(items)))

    def new_ranges(item):
        meta = item["metadata"]
        return _uncovered(covered.get(meta.get("document_id"), []), meta["char_start"], meta["char_end"])

    def gain(item):
        if not merged:
            return text_tokens(item["text"])
        offset = item["metadata"]["char_start"]
        return sum(text_tokens(item["text"][a - offset:b - offset]) for a, b in new_ranges(item))

    def accept(item):
        selected.append(item)
        if merged:
            meta = item["metadata"]
            key = meta.get("document_id")
            covered[key] = _cover(covered.get(key, []), meta["char_start"], meta["char_end"])

    first = results[0]
    middle = len(first["text"]) // 2
    probe_end = min(len(first["text"]), middle + 16)
    probe_start = middle if probe_end > middle else 0
    offset = first["metadata"].get("char_start", 0)
    probe = {"text": first["text"][probe_start:probe_end], "score": 0, "metadata": {
        **first["metadata"], "char_start": offset + probe_start, "char_end": offset + probe_end}}
    base = count([])
    overhead = max(0, count([probe]) - base - text_tokens(probe["text"])) + SLACK_TOKENS
    upper = lower = base
    stop_reason = "candidates_exhausted"
    for item in results:
        tokens = gain(item)
        if not tokens:
            accept(item)
            continue
        if upper + tokens + overhead <= budget:
            accept(item)
            upper += tokens + overhead
            lower += max(0, tokens - overhead)
            continue
        if lower + tokens - overhead <= budget:
            exact = count([*selected, item])
            if exact <= budget:
                accept(item)
                upper = lower = exact
                continue
        if policy == "stop":
            stop_reason = "next_candidate_overflow"
            break
    trimmed = 0
    if policy == "fill":
        remaining = budget - count(selected)
        for item in results:
            ranges = new_ranges(item)
            if not ranges:
                continue
            meta = item["metadata"]
            start, end = ranges[0]
            piece = item["text"][start - meta["char_start"]:end - meta["char_start"]]
            characters_per_token = len(piece) / max(1, text_tokens(piece))
            keep = min(len(piece), int((remaining - overhead) * characters_per_token))
            while keep > 0:
                part = {**item, "text": piece[:keep], "metadata": {**meta, "char_start": start, "char_end": start + keep}}
                exact = count([*selected, part])
                if exact <= budget:
                    accept(part)
                    trimmed = keep
                    break
                keep -= max(1, int((exact - budget) * characters_per_token))
            break
    if not selected:
        raise ValueError("No retrieved child fits the configured input budget")
    evidence = evidence_for(selected)
    request = render(evidence)
    return request, {
        "final_rendered_input_tokens": chat_token_count(tokenizer, request),
        "selected_evidence_ranges": [{key: value for key, value in item.items() if key != "text"} for item in evidence],
        "selected_child_indices": [item["metadata"].get("child_index", item["metadata"].get("chunk_index")) for item in selected],
        "candidate_count": len(results), "packing_stop_reason": stop_reason,
        "packing_renders": renders, "packing_trimmed_chars": trimmed,
    }
