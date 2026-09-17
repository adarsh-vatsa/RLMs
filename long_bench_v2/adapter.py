from execution.contracts import Document, Task
from execution.packing import render_document_slices
from long_bench_v2.qwen_prompt import build_strict_mcq_messages, STRICT_MCQ_SYSTEM_PROMPT


def solver_task(case_id, source_id, context, query, *, legacy=False, docs_dir=None):
    document = Document("context.txt", context)

    def render(evidence):
        if evidence is None:
            text = context
        elif legacy:
            text = "\n\n---\n\n".join(item["text"] for item in evidence)
        else:
            text = render_document_slices(document, evidence)
        return build_strict_mcq_messages(text, query)

    return Task(case_id, source_id, (document,), query, render,
                "strict_qwen_mcq_v1" + ("_children_v1" if legacy else "_source_fragments_v1"),
                choices=("A", "B", "C", "D"), fixed_instructions=STRICT_MCQ_SYSTEM_PROMPT,
                legacy_docs_dir=docs_dir)
