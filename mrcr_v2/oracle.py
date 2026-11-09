"""MRCR oracle: a hybrid run that puts the chunks holding the needle turns first.

It orders chunks with the gold answer, so it is a ceiling, not a method. The prompt is
built exactly as in any hybrid run; only the order in which chunks are offered to the
packer uses gold data.
"""

import re

from execution.pipeline import Pipeline


ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth")
ASKED = re.compile(r"User: Prepend \S+ to the (\w+) (.+?) in an? (\w+) style\. ")
NEXT_TURN = "\n\nUser: "


def needle_spans(body, question, answer, count):
    """Character spans of the turns that repeat the asked request, in conversation order, and the target's index.

    The question names the request and its ordinal; the gold answer confirms which turn it is.
    """
    ordinal, subject, style = ASKED.match(question).groups()
    request = f"User: Write a {subject} in {style} style.\n\nAssistant: "
    spans = []
    for match in re.finditer(re.escape(request), body):
        end = body.find(NEXT_TURN, match.end())
        spans.append((match.start(), end if end >= 0 else len(body)))
    target = ORDINALS.index(ordinal)
    if len(spans) != count or answer.strip()[12:].strip() not in body[slice(*spans[target])]:
        raise ValueError("The needle turns or the gold answer were not found where the question says")
    return spans, target


class OraclePipeline(Pipeline):
    def __init__(self, *args, gold, **kwargs):
        super().__init__(*args, **kwargs)
        self.gold = gold

    def _candidates(self, task, controller):
        row = self.gold[task.case_id]
        spans, target = needle_spans(task.documents[0].text, row["question"], row["answer"], row["num_relevant"])
        spans.insert(0, spans.pop(target))

        def needle(item):
            meta = item["metadata"]
            return next((index for index, (start, end) in enumerate(spans)
                         if meta["char_start"] < end and start < meta["char_end"]), len(spans))

        # A stable sort keeps the recipe's ranking within the needle chunks and among the rest.
        return sorted(super()._candidates(task, controller), key=needle)
