"""Pick: the whole situation, one entry of a list (§6.2). One choice question per context, one row per item."""

from __future__ import annotations

import dataclasses

from .. import pipeline
from .base import KEY, Spec, confidence, named, option_text, overrides_inputs, state_text, warnings

PICK_STATE = "Choose one of the listed items."


@dataclasses.dataclass
class PickJob(pipeline.Job):
    list_source: str = 'TEXT'  # under MATERIALS the entry also stores the picked material's name (§8.6)


class PickSpec(Spec):
    name, idname, label = "pick", "GeometryNodeOpenActPick", "Pick"
    WORDS = ("Prompt", "Context", "Items", "Overrides", "Key")
    INPUTS = (
        ('INPUT', "Prompt", 'NodeSocketString', "What to pick, e.g. 'something to sit on'", "word"),
        ('INPUT', "Context", 'NodeSocketString',
         "Optional situation the pick is for, e.g. 'a beach bar at noon'. Sent to the model", "word"),
    )
    ID = ('INPUT', "ID", 'NodeSocketInt',
          "The entry Selection and Probability describe. Unlinked: the element's index", "field", False, 'INDEX', True)
    OUTPUTS = (
        ('OUTPUT', "Index", 'NodeSocketInt',
         "Position of the picked entry, -1 until answered. Matches Collection Info order and material indices"),
        ('OUTPUT', "Item", 'NodeSocketString', "Name of the picked entry"),
        ('OUTPUT', "Confidence", 'NodeSocketFloat', "How clearly the pick wins over the runner-up (0-1)"),
        ('OUTPUT', "Selection", 'NodeSocketBool', "True for the element whose ID is the picked entry"),
        ('OUTPUT', "Probability", 'NodeSocketFloat', "How well entry ID fits (0-1); 0 outside the list"),
    )
    EXTRA = {  # list_source -> its extra output
        'COLLECTION': ('OUTPUT', "Instance", 'NodeSocketGeometry', "The picked collection child as an instance"),
        'MATERIALS': ('OUTPUT', "Material", 'NodeSocketMaterial', "The picked slot's material"),
    }
    FALLBACK = {"Index": -1, "Item": "", "Confidence": 0.0}
    empty_hint = "Type a prompt and items"

    def per_element(self, node):
        from ..inner import Column  # imports bpy

        return (Column("Probability", "prob", 'FLOAT', 0.0, linked_only=True),)

    def interface(self, node):
        from ..inner import SocketDef  # imports bpy
        from ..sources import list_inputs

        source = getattr(node, "list_source", 'TEXT')
        extra = (self.EXTRA[source],) if source in self.EXTRA else ()
        return [*(SocketDef(*args) for args in self.INPUTS), *list_inputs(source), SocketDef(*self.ID),
                *(SocketDef(*args) for args in (*self.OUTPUTS, *extra)), *overrides_inputs(), SocketDef(*KEY)]

    def items(self, node, ctx):
        from ..lists import resolve_items  # imports bpy

        return resolve_items(node, ctx, spec="pick")

    def listed(self, node, ctx):
        from ..evaluator import Hint, Unresolvable  # imports bpy

        try:
            return self.items(node, ctx).fingerprint
        except (Hint, Unresolvable) as error:  # plan() raises it again after its own checks; the plan cache keeps it
            return f"{type(error).__name__}: {error}"

    def plan(self, node, ctx):
        from ..evaluator import Hint  # imports bpy

        gn = self.gn(node, ctx)
        if not gn[0].strip():
            raise Hint("Type a prompt")
        items = self.items(node, ctx)
        gn = gn[:2] + (items.raw,) + gn[3:]  # "" under Collection, Materials and Attributes (§8.2)
        job = self.job(gn, items.items, node.balance_order, node.uid, items.source,
                       [items.doc.text_hash] if items.doc is not None else [])
        job.problems = warnings(items.problems)
        return job

    def job(self, gn, items, debias, uid="", source='TEXT', csv_hashes=()):
        """The Job for these words and items (lists.Item, or names). Pure, so the pure suites plan exactly like the
        pass. Entries that can't be offered (an empty material slot) keep their ID with probability 0; a list with
        one offered entry is decided by code and never sent (4.d)."""
        from ..lists import Item  # pure without bpy

        items = [Item(item) if isinstance(item, str) else item for item in items]
        prompt, context = gn[0].strip(), gn[1].strip()
        offered = [(i, item) for i, item in enumerate(items) if item.offered]
        options = tuple((item.name, item.when) for _i, item in offered)
        criteria = {str(i): option_text(item) for i, item in offered}
        instructions = f"Which item best fits: {prompt.rstrip('?').strip()}?"
        state = state_text(context, PICK_STATE)
        cell = None
        if len(offered) > 1:
            cell = pipeline.Cell(pipeline.cell_key("choice", state, instructions, criteria, debias),
                                 state, {"type": "choice", "instructions": instructions, "criteria": criteria},
                                 pipeline.cost(len(options), debias))
        rows = []
        for item in items:
            wkey = pipeline.wkey(self.name, 'ONE_OF', [prompt, context], [item.name, item.when], options)
            if cell is not None and item.offered:
                rows.append(pipeline.Row(item.name, wkey, cell=cell))
            else:  # code rows: never asked, and ask "" (no model's settings behind them)
                rows.append(pipeline.Row(item.name, wkey, "code", {"prob": float(item.offered)}, ""))
        shape = pipeline.TableShape(len(rows), (("prob", "FLOAT"),))
        words_key = pipeline.words_key(gn, list(csv_hashes), 'ONE_OF', shape.digest(), [row.wkey for row in rows])
        return PickJob(uid, tuple(gn), words_key, shape, rows, answer_type='ONE_OF', debias=debias,
                       list_source=source)

    def rows(self, job, answers):
        """As Spec.rows(), but the server's probabilities come in the order of the offered IDs (criteria keys)."""
        rows = []
        for index, row in enumerate(job.rows):
            answer = answers.get(row.cell.key) if row.source == "pending" and row.cell is not None else None
            if answer is not None:
                position = list(row.cell.question["criteria"]).index(str(index))
                row = dataclasses.replace(row, source="model", values={"prob": answer["probabilities"][position]},
                                          ask=self.ask_key(job.debias, answer["model"]))
            rows.append(row)
        return rows

    OVERRIDE_KEYS = ("prompt",)

    def override_units(self, job):
        """One unit per prompt: a lock forces the picked item (§7.6)."""
        return [((job.gn[0].strip(),), tuple(range(len(job.rows))))] if job.rows else []

    def lock_values(self, job, rows, answer):
        chosen = named(answer, [row.name for row in job.rows])  # an empty material slot has no name to match
        if chosen is None:
            return f"'{answer}' is not an item of the list"
        return [{"prob": float(i == chosen)} for i in rows]

    def export_unit(self, job, rows, table):
        if not all(table[i][1] for i in rows):
            return "", None, "pending"
        probabilities = [table[i][1]["prob"] for i in rows]
        best = max(range(len(rows)), key=probabilities.__getitem__)
        return job.rows[rows[best]].name, confidence('ONE_OF', probabilities), table[rows[best]][0]

    def single(self, rows):
        asked = [row for row in rows if row.cell is not None]
        if asked and all(row.source == "pending" for row in asked):  # a partial table with no answer yet (§5.4)
            return dict(self.FALLBACK)
        probabilities = [row.values["prob"] for row in rows]
        index = max(range(len(rows)), key=probabilities.__getitem__)  # the first of equal ones, as the server picks
        return {"Index": index, "Item": rows[index].name, "Confidence": confidence('ONE_OF', probabilities)}

    def entry(self, job, rows, model=""):
        entry = super().entry(job, rows, model)
        if getattr(job, "list_source", "") == 'MATERIALS':
            entry.single["Material"] = entry.single["Item"]  # by name, resolved at build time (§8.6)
        return entry

    def summary(self, entry):
        p = entry.rows.v["prob"][entry.single["Index"]]
        return f"{entry.single['Item']}  ({p * 100:.0f}%)"

    def build(self, b, gi, go, entries):
        from ..inner import sock  # imports bpy

        index = b.match_chain(b.words(gi, self.WORDS), [e.gn for e in entries])
        picked = b.lookup(index, 'INT', -1, [e.single["Index"] for e in entries], go.inputs["Index"], 300)
        b.lookup(index, 'STRING', "", [e.single["Item"] for e in entries], go.inputs["Item"], 100)
        b.lookup(index, 'FLOAT', 0.0, [e.single["Confidence"] for e in entries], go.inputs["Confidence"], -100)
        ident = next((s for s in gi.outputs if s.name == "ID"), None)
        if ident is not None and "Selection" in go.inputs:
            b.selection(ident, picked, go.inputs["Selection"])
        if "Material" in go.inputs:
            b.lookup(index, 'MATERIAL', None, [e.single.get("Material") for e in entries], go.inputs["Material"], -250)
        collection = next((s for s in gi.outputs if s.name == "Collection"), None)
        if collection is None or "Instance" not in go.inputs:
            return
        info = b.new("GeometryNodeCollectionInfo", 150, -350)
        b.link(collection, sock(info.inputs, "Collection"))  # the list's collection, live per context
        sock(info.inputs, "Separate Children").default_value = True
        sock(info.inputs, "Reset Children").default_value = True
        position = b.new("GeometryNodeInputIndex", 150, -560)
        other = b.new("FunctionNodeCompare", 350, -520)
        other.data_type = 'INT'
        other.operation = 'NOT_EQUAL'
        b.link(position.outputs[0], sock(other.inputs, "A_INT"))
        b.link(picked, sock(other.inputs, "B_INT"))
        delete = b.new("GeometryNodeDeleteGeometry", 550, -350)
        delete.domain = 'INSTANCE'
        b.link(info.outputs[0], sock(delete.inputs, "Geometry"))
        b.link(other.outputs[0], sock(delete.inputs, "Selection"))
        b.link(delete.outputs[0], go.inputs["Instance"])
