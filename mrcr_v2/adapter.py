import re

from execution.contracts import Document, Task
from execution.packing import render_document_slices
from mrcr_v2.prompting import OMITTED, PROMPT_VERSION, messages


# Retrieval uses only what the matching requests share (format, topic and style):
# the marker is random and the ordinal never appears in the conversation.
QUESTION = re.compile(r"User: Prepend (\S+) to the \w+ (.+?)\. Do not include any other text in your response\.")


def solver_task(case_id, source_id, prefix, body, question):
    document = Document(source_id, body)
    match = QUESTION.match(question)
    if not match:
        raise ValueError("Unsupported MRCR final instruction format")
    marker, request = match.groups()

    def render(evidence):
        context = body if evidence is None else render_document_slices(document, evidence, OMITTED)
        return messages(prefix + context + question)

    return Task(case_id, source_id, (document,), request, render, PROMPT_VERSION,
                required_prefix=marker, fixed_instructions=prefix)
