"""Ask: the whole situation, Yes/No or Amount (§6.1). One question per context: a one-row table."""

from __future__ import annotations

from .. import pipeline
from .base import Spec, confidence, state_text

AMOUNT_LEVELS = ("Not at all", "Slightly", "Moderately", "Very", "Extremely")
YES_NO = (("yes", ""), ("no", ""))  # without Answers the server offers "Yes." and "No."


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

    def plan(self, node, ctx):
        from ..evaluator import Hint  # imports bpy

        gn = self.gn(node, ctx)
        if not gn[0].strip():
            raise Hint(self.empty_hint)
        return self.job(gn, node.answer_type, node.balance_order, node.uid)

    def job(self, gn, answer_type, debias, uid=""):
        """The Job for these words. Pure, so the pure suites plan exactly like the pass."""
        question, context = gn[0].strip(), gn[1].strip()
        if answer_type == 'AMOUNT':
            options = tuple((level, "") for level in AMOUNT_LEVELS)
            asked = {"type": "score", "instructions": question, "criteria": list(AMOUNT_LEVELS)}
            columns = (("amount", "FLOAT"), ("c", "FLOAT"))
        else:
            options = YES_NO
            asked = {"type": "noul", "instructions": question}
            columns = (("p", "FLOAT"), ("c", "FLOAT"))
        state = state_text(context)
        cell = pipeline.Cell(pipeline.cell_key(asked["type"], state, question, asked.get("criteria"), debias), state,
                             asked, pipeline.cost(len(options), debias))
        row = pipeline.Row(question, pipeline.wkey(self.name, answer_type, [question, context], [], options),
                           cell=cell)
        shape = pipeline.TableShape(1, columns)
        return pipeline.Job(uid, tuple(gn), pipeline.words_key(gn, [], answer_type, shape.digest(), [row.wkey]),
                            shape, [row], answer_type=answer_type, debias=debias)

    def values(self, index, answer):
        if "amount" in answer:
            return {"amount": answer["amount"], "c": confidence('AMOUNT', answer["probabilities"])}
        return {"p": answer["p"], "c": confidence('YES_NO', (answer["p"], 1.0 - answer["p"]))}

    def single(self, rows):
        values = rows[0].values
        answer = values["amount"] if "amount" in values else values["p"]
        return {"Answer": answer, "Yes": answer >= 0.5, "Confidence": values["c"]}

    def summary(self, entry):
        if entry.answer_type == 'AMOUNT':
            return f"Amount {entry.single['Answer']:.2f}"
        p = entry.single["Answer"]
        return f"{'Yes' if p >= 0.5 else 'No'}  ({p * 100:.0f}% yes)"

    def build(self, b, gi, go, entries):
        index = b.match_chain(self.word_sockets(gi), [e.gn for e in entries])
        for name, data_type, y in (("Answer", 'FLOAT', 200), ("Yes", 'BOOLEAN', 0), ("Confidence", 'FLOAT', -200)):
            b.lookup(index, data_type, self.FALLBACK[name], [e.single[name] for e in entries], go.inputs[name], y)
