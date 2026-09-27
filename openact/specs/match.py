"""Match: each named item, Yes/No or Amount (§6.3). One question per item, all sharing one state: one row each.

PerItemSpec holds what Match and Classify share: the list source and the answer set, the rows of the per-item cells
(sources.per_item_cells(), the §10.2 templates), the 8192-row estimate (§5.4), and the per-element table (§8.1).
"""

from __future__ import annotations

from .. import pipeline
from ..csvfmt import _n
from .base import KEY, UNSURE_BELOW, Spec, confidence, lock_answer, number_text, overrides_inputs, warnings

ROW_LIMIT = 8192  # estimated scored rows per context, sum(cell.cost): refused above, before anything is sent (§5.4)


class PerItemSpec(Spec):
    """A table with one row per item of the node's list (§6.3, §6.4). Every output is per element, read at ID."""

    ANSWERS = ""  # the answer-set word: "Answers" or "Labels"
    COLUMNS: tuple = ()  # (name, type) of each answer column of a row
    NEUTRAL: dict = {}  # the row of an empty material slot: never asked, never flagged (§6.3)

    def kind(self, node) -> str:
        """The answer-set kind: "yes_no", "amount" or "one_of"."""
        raise NotImplementedError

    def head(self, node, question: str, answers: str) -> list:
        """The inputs both nodes have: Question, Context, the list source's sockets, the answer set and ID."""
        from .. import sources
        from ..inner import SocketDef  # imports bpy

        return [
            SocketDef('INPUT', "Question", 'NodeSocketString', question, "word"),
            SocketDef('INPUT', "Context", 'NodeSocketString', "The situation all items share, e.g. a scene description",
                      "word"),
            *sources.list_inputs(getattr(node, "list_source", 'TEXT')),
            *sources.csv_inputs(self.ANSWERS, answers),
            SocketDef('INPUT', "ID", 'NodeSocketInt', "Which item each element is (0 = the first). Unlinked: the "
                      "element's index", kind="field", default_input='INDEX', hide_value=True),
        ]

    def resolved(self, node, ctx):
        """The node's items (lists.ItemList) and answer set in ctx. Raises Hint, Pending or Unresolvable."""
        from .. import lists, sources  # imports bpy

        items = lists.resolve_items(node, ctx, spec=self.name)
        answers = sources.answer_set(node, ctx, self.ANSWERS, self.kind(node), spec=self.name,
                                     empty_hint=self.empty_hint)
        return items, answers

    def listed(self, node, ctx):
        """The list and the answer-set CSV in the fingerprint: a collection's children, a CSV's stamp."""
        from ..evaluator import Hint, Unresolvable  # imports bpy

        try:
            items, answers = self.resolved(node, ctx)
        except (Hint, Unresolvable) as error:  # plan() raises it again, and the plan cache keeps it
            return f"{type(error).__name__}: {error}"
        doc = answers.doc
        return items.fingerprint, doc and (doc.path, getattr(answers.source, "delimiter", None), doc.stamp)

    def table(self, gn, answer_type, rows, docs, debias, uid="", job=pipeline.Job, **extra):
        """The Job of these per-item rows (sources.per_item_cells()). docs: the CSVs read, whose text enters the words
        key. Raises Unresolvable above ROW_LIMIT estimated scored rows."""
        total = sum(row.cell.cost for row in rows if row.cell is not None)
        if total > ROW_LIMIT:
            from ..evaluator import Unresolvable  # imports bpy

            raise Unresolvable(f"{self.label} needs about {_n(total)} scored rows (limit {_n(ROW_LIMIT)}): use fewer "
                               f"items or {self.ANSWERS.lower()}, or turn off Balance Option Order")
        shape = pipeline.TableShape(len(rows), self.COLUMNS)
        hashes = [doc.text_hash for doc in docs if doc is not None]
        words_key = pipeline.words_key(gn, hashes, answer_type, shape.digest(), [row.wkey for row in rows])
        return job(uid, tuple(gn), words_key, shape, rows, answer_type=answer_type, debias=debias, **extra)

    def offered(self, entry, column):
        """(name, value) of the rows that are items, without empty material slots."""
        return [(name, value) for name, value in zip(entry.rows.name, entry.rows.v[column]) if name]

    def unsure(self, entry) -> int:
        """Model rows below UNSURE_BELOW: a pending or stale row of a table so far is no answer (§5.4)."""
        return sum(src == "m" and c < UNSURE_BELOW for src, c in zip(entry.rows.src, entry.rows.v["c"]))


class MatchSpec(PerItemSpec):
    name, idname, label = "match", "GeometryNodeOpenActMatch", "Match"
    WORDS = ("Question", "Context", "Items", "Answers", "Overrides", "Key")
    ANSWERS = "Answers"
    COLUMNS = (("p", "FLOAT"), ("c", "FLOAT"))
    NEUTRAL = {"p": 0.0, "c": 0.0}
    empty_hint = "Type a question"

    def kind(self, node):
        from .. import sources

        return sources.KINDS[node.answer_type]

    def interface(self, node):
        from ..inner import SocketDef  # imports bpy

        return [
            *self.head(node, "Asked about every item, e.g. 'Would you find it in a kitchen?' (Amount: 'How heavy "
                             "is it?')",
                       "Optional, as on Ask: 'yes: ...; no: ...', Amount levels lowest first, or a .csv path"),
            SocketDef('INPUT', "Threshold", 'NodeSocketFloat',
                      "Selection is true where Probability is at least this (0-1). May be a field"),
            SocketDef('OUTPUT', "Probability", 'NodeSocketFloat',
                      "Yes / No: probability of yes for item ID. Amount: its amount (0-1). 0 until answered"),
            SocketDef('OUTPUT', "Selection", 'NodeSocketBool', "Probability >= Threshold, for an answered item"),
            SocketDef('OUTPUT', "Confidence", 'NodeSocketFloat',
                      "How sure the answer for item ID is (0 = no idea, 1 = certain)"),
            *overrides_inputs(),
            SocketDef(*KEY),
        ]

    def per_element(self, node):
        from ..inner import Column  # imports bpy

        return Column("Probability", "p", 'FLOAT', 0.0), Column("Confidence", "c", 'FLOAT', 0.0)

    def plan(self, node, ctx):
        from ..evaluator import Hint  # imports bpy

        gn = self.gn(node, ctx)
        if not gn[0].strip():
            raise Hint(self.empty_hint)
        items, answers = self.resolved(node, ctx)
        job = self.job(gn, node.answer_type, items, answers, node.balance_order, node.uid)
        job.problems = warnings(items.problems + answers.problems)  # the pass shows them (§17.1)
        return job

    def job(self, gn, answer_type, items, answers, debias, uid=""):
        """The Job for these words, items (an ItemList or its items) and answer set. Pure, like Ask's and Pick's."""
        from .. import sources

        rows = sources.per_item_cells(self.name, answer_type, items, answers, gn[0], gn[1], debias, self.NEUTRAL)
        return self.table(gn, answer_type, rows, (getattr(items, "doc", None), answers.doc), debias, uid,
                          options=tuple(option.name for option in answers.options))

    def lock_values(self, job, rows, answer):
        value = lock_answer(job.answer_type, answer, job.options)
        return value if isinstance(value, str) else [{"p": value, "c": 1.0}]

    def answer_text(self, job, index, values):
        return number_text(values["p"], job.options)

    def values(self, index, answer):
        if "amount" in answer:
            return {"p": answer["amount"], "c": confidence('AMOUNT', answer["probabilities"])}
        return {"p": answer["p"], "c": confidence('YES_NO', (answer["p"], 1.0 - answer["p"]))}

    def summary(self, entry):
        values = [p for _name, p in self.offered(entry, "p")]
        if entry.answer_type == 'AMOUNT':
            mean = sum(values) / len(values) if values else 0.0
            return f"{len(values)} items · mean {mean:.2f} · {self.unsure(entry)} unsure"
        return f"{sum(p >= 0.5 for p in values)} of {len(values)} yes · {self.unsure(entry)} unsure"

    def build(self, b, gi, go, entries):
        from ..inner import sock  # imports bpy

        b.match_chain(b.words(gi, self.WORDS), [e.gn for e in entries])

        def selection(valid, values):  # Selection = valid AND p >= Threshold (§6.3); Threshold may be a field
            at_least = b.new("FunctionNodeCompare", 900, -1300)
            at_least.data_type = 'FLOAT'
            at_least.operation = 'GREATER_EQUAL'
            b.link(values["Probability"], sock(at_least.inputs, "A"))
            b.link(gi.outputs["Threshold"], sock(at_least.inputs, "B"))
            b.link(b._and(valid, at_least.outputs[0], 1050, -1300), go.inputs["Selection"])

        b.after_elements.append(selection)