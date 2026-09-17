from execution.contracts import Document, Task
from execution.packing import render_document_slices
from mrcr_v2.prompting import OMITTED, PROMPT_VERSION, messages


def solver_task(case_id, source_id, prefix, body, question):
    document = Document(source_id, body)

    def render(evidence):
        context = body if evidence is None else render_document_slices(document, evidence, OMITTED)
        return messages(prefix + context + question)

    marker = question.removeprefix("User: Prepend ").split(" to the ", 1)[0]
    return Task(case_id, source_id, (document,), question, render, PROMPT_VERSION,
                required_prefix=marker, fixed_instructions=prefix)
