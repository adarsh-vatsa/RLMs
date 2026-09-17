from execution.tokens import chat_token_count


OMITTED = "\n\n[... omitted source ...]\n\n"


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


def pack(tokenizer, render, documents, results, budget, *, order="source", merge=True):
    selected = []
    evidence = []
    stop_reason = "candidates_exhausted"
    for result in results:
        candidate = [*selected, result]
        candidate_evidence = source_ranges(documents, candidate, merge) if order == "source" else [
            {**item.get("metadata", {}), "text": item["text"], "score": item.get("score", 0)} for item in candidate
        ]
        if chat_token_count(tokenizer, render(candidate_evidence)) > budget:
            stop_reason = "next_candidate_overflow"
            break
        selected, evidence = candidate, candidate_evidence
    if not selected:
        raise ValueError("No retrieved child fits the configured input budget")
    request = render(evidence)
    return request, {
        "final_rendered_input_tokens": chat_token_count(tokenizer, request),
        "selected_evidence_ranges": [{key: value for key, value in item.items() if key != "text"} for item in evidence],
        "selected_child_indices": [item["metadata"].get("child_index", item["metadata"].get("chunk_index")) for item in selected],
        "candidate_count": len(results), "packing_stop_reason": stop_reason,
    }
