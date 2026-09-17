from itertools import groupby

from execution.contracts import Document, Task
from execution.packing import render_document_slices
from aa_lcr.prompting import PROMPT_VERSION, build_messages, build_prompt


def solver_task(case_id, source_id, documents, question, *, legacy=False, docs_dir=None):
    sources = tuple(Document(name, text) for name, text in documents)

    def render(evidence):
        if evidence is None:
            return build_messages([doc.text for doc in sources], question)
        if legacy:
            return build_messages([item["text"] for item in evidence], question)
        positions = {doc.id: (index, doc) for index, doc in enumerate(sources, 1)}
        texts, numbers = [], []
        for doc_id, fragments in groupby(evidence, key=lambda item: item["document_id"]):
            number, document = positions[doc_id]
            numbers.append(number)
            texts.append(render_document_slices(document, list(fragments)))
        return [{"role": "user", "content": build_prompt(texts, question, document_numbers=numbers)}]

    version = PROMPT_VERSION + ("_children_v1" if legacy else "_source_fragments_v1")
    return Task(case_id, source_id, sources, question, render, version, legacy_docs_dir=docs_dir)
