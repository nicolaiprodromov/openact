"""Ask: the whole situation, Yes/No or Amount (§6.1). One question per context: a one-row table."""

from __future__ import annotations

from .. import pipeline
from .base import (KEY, Spec, confidence, lock_answer, number_text, overrides_inputs, shared_sockets, state_text,
                   warnings)

ANSWERS = ("Optional answer set. Yes / No: 'yes: <when>; no: <when>'. Amount: 2-16 levels, lowest first, e.g. "
           "'none; dusting; patchy; thick; buried'. Or a .csv path. Empty: Yes / No, or Not at all ... Extremely")


class AskSpec(Spec):
    name, idname, label = "ask", "GeometryNodeOpenActAsk", "Ask"
    WORDS = ("Question", "Context", "Answers", "Overrides", "Key")
    SOCKETS = (
        ('INPUT', "Question", 'NodeSocketString',
         "The question, e.g. 'Is this scene set at night?' (Amount mode: 'How weathered should it look?')", "word"),
        ('INPUT', "Context", 'NodeSocketString', "Optional facts the question is about, e.g. a scene description",
         "word"),
        ('OUTPUT', "Answer", 'NodeSocketFloat',
         "Yes / No: probability of yes (0-1). Amount: how much (0-1). 0 until answered"),
        ('OUTPUT', "Yes", 'NodeSocketBool', "True when Answer is 0.5 or more"),
        ('OUTPUT', "Confidence", 'NodeSocketFloat', "How sure the answer is (0 = no idea, 1 = certain)"),
    )
    FALLBACK = {"Answer": 0.0, "Yes": False, "Confidence": 0.0}
    empty_hint = "Type a question"

    def interface(self, node):
        from ..inner import SocketDef  # imports bpy
        from ..sources import csv_inputs

        return [*shared_sockets(), *(SocketDef(*args) for args in self.SOCKETS), *csv_inputs("Answers", ANSWERS),
                *overrides_inputs(), SocketDef(*KEY)]

    def answers(self, node, ctx):
        from .. import sources  # imports bpy

        return sources.answer_set(node, ctx, "Answers", sources.KINDS[node.answer_type], spec="ask")

    def listed(self, node, ctx):
        from .. import lists  # imports bpy

        source = lists.csv_source(node, "Answers", ctx)  # an inline set is in gn already
        if source is None:
            return None
        doc = lists.load_csv(node, source)
        return doc.path, source.delimiter, doc.stamp

    def plan(self, node, ctx):
        from ..evaluator import Hint  # imports bpy

        gn = self.gn(node, ctx)
        if not gn[0].strip():
            raise Hint(self.empty_hint)
        answers = self.answers(node, ctx)
        job = self.job(gn, node.answer_type, node.balance_order, node.uid, answers)
        job.problems = warnings(answers.problems)
        return job

    def job(self, gn, answer_type, debias, uid="", answers=None):
        """The Job for these words and answer set (csvfmt.AnswerSet; None: the default Yes / No or five levels).
        Pure, so the pure suites plan exactly like the pass."""
        from .. import sources  # pure without bpy

        kind = sources.KINDS[answer_type]
        answers = answers or sources.default_set(kind)
        question, context = gn[0].strip(), gn[1].strip()
        options = tuple((option.name, option.when) for option in answers.options)
        asked = sources.question(kind, question, answers)
        columns = (("amount", "FLOAT"), ("c", "FLOAT")) if kind == "amount" else (("p", "FLOAT"), ("c", "FLOAT"))
        state = state_text(context)
        cell = pipeline.Cell(pipeline.cell_key(asked["type"], state, question, asked["criteria"], debias), state, asked,
                             pipeline.cost(len(options), debias))
        row = pipeline.Row(question, pipeline.wkey(self.name, answer_type, [question, context], [], options),
                           cell=cell)
        shape = pipeline.TableShape(1, columns)
        csv = [answers.doc.text_hash] if answers.doc is not None else []
        return pipeline.Job(uid, tuple(gn), pipeline.words_key(gn, csv, answer_type, shape.digest(), [row.wkey]),
                            shape, [row], answer_type=answer_type, debias=debias,
                            options=tuple(option.name for option in answers.options))

    def values(self, index, answer):
        if "amount" in answer:  # parse() gave score / (L - 1), as sources.amount_value() does for custom levels
            return {"amount": answer["amount"], "c": confidence('AMOUNT', answer["probabilities"])}
        return {"p": answer["p"], "c": confidence('YES_NO', (answer["p"], 1.0 - answer["p"]))}

    OVERRIDE_KEYS = ("question",)

    def lock_values(self, job, rows, answer):
        value = lock_answer(job.answer_type, answer, job.options)
        column = "amount" if job.answer_type == 'AMOUNT' else "p"
        return value if isinstance(value, str) else [{column: value, "c": 1.0}]

    def answer_text(self, job, index, values):
        return number_text(values["amount"] if "amount" in values else values["p"], job.options)

    def single(self, rows):
        values = rows[0].values
        answer = values["amount"] if "amount" in values else values["p"]
        return {"Answer": answer, "Yes": answer >= 0.5, "Confidence": values["c"]}

    def summary(self, entry):
        if entry.answer_type == 'AMOUNT':
            return f"Amount {entry.single['Answer']:.2f}"
        p = entry.single["Answer"]
        return f"{'Yes' if p >= 0.5 else 'No'}  ({p * 100:.0f}% yes)"

    def single_values(self, node):
        """Every output is a single value, a column of the entry's header row (§8.6)."""
        from ..inner import Column  # imports bpy

        return tuple(Column(output, name, data_type, self.FALLBACK[output]) for output, name, data_type in (
            ("Answer", "answer", 'FLOAT'), ("Yes", "yes", 'BOOLEAN'), ("Confidence", "conf", 'FLOAT')))

    def build(self, b, gi, go, entries):
        for j, column in enumerate(self.single_values(b.node)):
            b.link(b.single(column, 0, 200 - 200 * j), go.inputs[column.output])
