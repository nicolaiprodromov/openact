"""Classify: each named item, one of several labels (§6.4). One choice question per item over all the labels, all
sharing one state; more than 16 labels run as a tournament on the server (§10.3), which the row estimate counts."""

from __future__ import annotations

import dataclasses

from .. import pipeline
from .base import KEY, confidence, named, overrides_inputs, warnings
from .match import PerItemSpec

FACE = "openact classify payload"  # evaluator.MEMO key: the payload the pass's first planned context of a node reads


@dataclasses.dataclass
class LabelsJob(pipeline.Job):
    """A Classify Job, with what each label bakes into the rows that choose it (§8.1). Its options are the label
    names, in row order."""

    payload: tuple = ()  # (header, type) of each numeric column of the Labels CSV
    baked: tuple = ()  # per label, its payload values in header order


class ClassifySpec(PerItemSpec):
    name, idname, label = "classify", "GeometryNodeOpenActClassify", "Classify"
    WORDS = ("Question", "Context", "Items", "Labels", "Overrides", "Key")
    ANSWERS = "Labels"
    COLUMNS = (("label", "INT"), ("c", "FLOAT"))
    NEUTRAL = {"label": -1, "c": 0.0}
    FIXED = ("Label", "Confidence")  # the outputs that are not payload
    empty_hint = "Add labels"

    def kind(self, node):
        return "one_of"

    def payload_outputs(self, node) -> list:
        """(header, type) of the node's payload outputs. They follow its Labels CSV, set by sync()."""
        tree = node.node_tree
        if tree is None:
            return []
        return [(item.name, 'INT' if item.socket_type == 'NodeSocketInt' else 'FLOAT')
                for item in tree.interface.items_tree
                if item.item_type == 'SOCKET' and item.in_out == 'OUTPUT' and item.name not in self.FIXED]

    def interface(self, node, payload=None):
        from ..inner import SocketDef  # imports bpy

        if payload is None:  # keep the payload outputs the node has
            payload = self.payload_outputs(node)
        return [
            *self.head(node, "Optional instruction. Empty: 'Which label fits this item best?'",
                       "The labels (required): 'metal: steel, iron; wood: planks, bark', or a .csv path with "
                       "answer, when and numeric payload columns"),
            SocketDef('OUTPUT', "Label", 'NodeSocketInt',
                      "Row of the chosen label for item ID (0 = the first). -1 until answered"),
            SocketDef('OUTPUT', "Confidence", 'NodeSocketFloat', "How clearly the label wins over the runner-up (0-1)"),
            *(SocketDef('OUTPUT', header, 'NodeSocketInt' if kind == 'INT' else 'NodeSocketFloat',
                        f"The Labels CSV's '{header}' of the chosen label") for header, kind in payload),
            *overrides_inputs(),
            SocketDef(*KEY),
        ]

    def per_element(self, node):
        from ..inner import Column  # imports bpy

        return (Column("Label", "label", 'INT', -1), Column("Confidence", "c", 'FLOAT', 0.0),
                *(Column(header, f"pl_{k}", kind, 0 if kind == 'INT' else 0.0, payload=True)
                  for k, (header, kind) in enumerate(self.payload_outputs(node))))

    def plan(self, node, ctx):
        gn = self.gn(node, ctx)
        items, labels = self.resolved(node, ctx)
        job = self.job(gn, items, labels, node.balance_order, node.uid)
        job.problems = warnings(items.problems + labels.problems)  # the pass shows them (§17.1)
        return self.locked(node, ctx, job)  # after the warnings: locked() appends its own

    def job(self, gn, items, labels, debias, uid=""):
        """The Job for these words, items (an ItemList or its items) and labels. Pure, like Ask's and Pick's."""
        from .. import sources

        rows = sources.per_item_cells(self.name, 'ONE_OF', items, labels, gn[0], gn[1], debias, self.NEUTRAL)
        payload = tuple(labels.doc.num_cols.items()) if labels.doc is not None else ()
        return self.table(gn, 'ONE_OF', rows, (getattr(items, "doc", None), labels.doc), debias, uid, job=LabelsJob,
                          options=tuple(option.name for option in labels.options), payload=payload,
                          baked=tuple(tuple(option.payload[header] for header, _type in payload)
                                      for option in labels.options))

    def values(self, index, answer):
        probabilities = answer["probabilities"]
        label = max(range(len(probabilities)), key=probabilities.__getitem__)  # the first of equal ones, as the server
        return {"label": label, "c": confidence('ONE_OF', probabilities)}

    def lock_values(self, job, rows, answer):
        label = named(answer, job.options)
        return f"'{answer}' is not one of the labels" if label is None else [{"label": label, "c": 1.0}]

    def answer_text(self, job, index, values):
        label = values["label"]
        return job.options[label] if 0 <= label < len(job.options) else ""

    def resolve(self, job, entries, last_shown="", partial=False):
        """Spec.resolve(), with each row's payload baked from its label (denormalised, §8.1). A numbers-only edit of
        the Labels CSV keeps the words key: the stored entry comes back with the new payload, and nothing is asked."""
        from .. import evaluator  # imports bpy

        if evaluator.MEMO is not None:  # a pass: sync() follows the plan, so the outputs exist before the first answer
            evaluator.MEMO.setdefault((FACE, job.uid), list(job.payload))
        entry, missing, failed = super().resolve(job, entries, last_shown, partial)
        if entry is None:
            return entry, missing, failed
        payload, baked = job.payload, job.baked
        shape = dict(entry.shape, labels=list(job.options), payload=[list(column) for column in payload])
        columns = {header: [baked[label][j] if 0 <= label < len(baked) else (0 if kind == 'INT' else 0.0)
                            for label in entry.rows.v["label"]] for j, (header, kind) in enumerate(payload)}
        if shape != entry.shape or columns != entry.rows.payload:
            entry = dataclasses.replace(entry, shape=shape, rows=dataclasses.replace(entry.rows, payload=columns))
        return entry, missing, failed

    def sync(self, node, _entries):
        """Give the node the payload outputs of the Labels CSV its first planned context reads, as resolve() recorded
        it in this pass. Outside a pass, or while no context plans, the node keeps its outputs and the links to them."""
        from .. import evaluator  # imports bpy
        from ..inner import sync_interface

        wanted = (evaluator.MEMO or {}).get((FACE, node.uid))
        if wanted is not None and wanted != self.payload_outputs(node):
            sync_interface(node, self.interface(node, wanted))

    def summary(self, entry):
        rows = self.offered(entry, "label")
        used = len({label for _name, label in rows if label >= 0})
        return f"{len(rows)} items · {used} label{'' if used == 1 else 's'} used · {self.unsure(entry)} unsure"

    def preview(self, entry):
        """(item, label name) rows for the sidebar."""
        names = entry.shape.get("labels", [])
        return [(name, names[label] if 0 <= label < len(names) else str(label))
                for name, label in self.offered(entry, "label")]

    def build(self, b, gi, go, entries):
        b.match_chain(b.words(gi, self.WORDS), [e.gn for e in entries])  # every output is a per-element column