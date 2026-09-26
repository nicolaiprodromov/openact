"""Pick: the whole situation, one entry of a list (§6.2). One choice question per context, one row per item."""

from __future__ import annotations

from .. import natsort, pipeline
from .base import Spec, confidence, option_text, state_text

MAX_ITEMS = 255
PICK_STATE = "Choose one of the listed items."


def parse_items(text):
    """Items are separated by commas, or by new lines when there are any (e.g. Import Text)."""
    if not text:
        return []
    parts = text.splitlines() if ("\n" in text or "\r" in text) else text.split(",")
    return [part.strip() for part in parts if part.strip()]


def collection_items(collection):
    """Children in the order Collection Info (Separate Children) instances them."""
    names = [obj.name for obj in collection.objects] + [child.name for child in collection.children]
    return sorted(names, key=natsort.sort_key)


class PickSpec(Spec):
    name, idname, label = "pick", "GeometryNodeOpenActPick", "Pick"
    WORDS = ("Prompt", "Context", "Items", "Overrides", "Key")
    SOCKETS = (
        ('INPUT', "Prompt", 'NodeSocketString', "What to pick, e.g. 'something to sit on'", "word"),
        ('INPUT', "Items", 'NodeSocketString',
         "Items separated by commas (or one per line, e.g. from Import Text). Ignored when a Collection is set",
         "word"),
        ('INPUT', "Collection", 'NodeSocketCollection',
         "Pick one of this collection's children instead, in the same order as Collection Info", "word"),
        ('OUTPUT', "Index", 'NodeSocketInt',
         "Position of the picked item, -1 until answered. Matches Collection Info / Pick Instance order"),
        ('OUTPUT', "Item", 'NodeSocketString', "Name of the picked item"),
        ('OUTPUT', "Confidence", 'NodeSocketFloat', "How clearly the pick wins over the runner-up (0-1)"),
        ('OUTPUT', "Instance", 'NodeSocketGeometry',
         "The picked collection child as an instance (empty when picking from text)"),
    )
    FALLBACK = {"Index": -1, "Item": "", "Confidence": 0.0}
    empty_hint = "Type a prompt and items"

    def collection(self, node, ctx):
        import bpy

        from ..evaluator import eval_input

        value = eval_input(node.inputs["Collection"], ctx)
        return value if isinstance(value, bpy.types.Collection) else None

    def listed(self, node, ctx):
        collection = self.collection(node, ctx)
        return None if collection is None else collection_items(collection)

    def plan(self, node, ctx):
        from ..evaluator import Hint, Unresolvable  # imports bpy

        gn = self.gn(node, ctx)
        children = self.listed(node, ctx)
        items = parse_items(gn[2]) if children is None else children
        if not gn[0].strip():
            raise Hint("Type a prompt")
        if not items:
            raise Hint("Type items separated by commas, or set a collection" if children is None else
                       "The collection is empty")
        if len(items) > MAX_ITEMS:
            raise Unresolvable(f"Too many items ({len(items)}); the limit is {MAX_ITEMS}")
        return self.job(gn, items, node.balance_order, node.uid)

    def job(self, gn, items, debias, uid=""):
        """The Job for these words and items. Pure, so the pure suites plan exactly like the pass."""
        prompt, context = gn[0].strip(), gn[1].strip()
        options = tuple((name, "") for name in items)
        criteria = {str(i): option_text(option) for i, option in enumerate(options)}
        instructions = f"Which item best fits: {prompt.rstrip('?').strip()}?"
        state = state_text(context, PICK_STATE)
        cell = pipeline.Cell(pipeline.cell_key("choice", state, instructions, list(criteria.items()), debias), state,
                             {"type": "choice", "instructions": instructions, "criteria": criteria},
                             pipeline.cost(len(options), debias))
        rows = [pipeline.Row(name, pipeline.wkey(self.name, 'ONE_OF', [prompt, context], [name, when], options),
                             cell=cell) for name, when in options]
        shape = pipeline.TableShape(len(rows), (("prob", "FLOAT"),))
        words_key = pipeline.words_key(gn, [], 'ONE_OF', shape.digest(), [row.wkey for row in rows])
        return pipeline.Job(uid, tuple(gn), words_key, shape, rows, answer_type='ONE_OF', debias=debias)

    def values(self, index, answer):
        return {"prob": answer["probabilities"][index]}

    def single(self, rows):
        probabilities = [row.values["prob"] for row in rows]
        index = max(range(len(rows)), key=probabilities.__getitem__)  # the first of equal ones, as the server picks
        return {"Index": index, "Item": rows[index].name, "Confidence": confidence('ONE_OF', probabilities)}

    def summary(self, entry):
        p = entry.rows.v["prob"][entry.single["Index"]]
        return f"{entry.single['Item']}  ({p * 100:.0f}%)"

    def build(self, b, gi, go, entries):
        from ..inner import sock  # imports bpy

        index = b.match_chain(self.word_sockets(gi), [e.gn for e in entries])
        picked = b.lookup(index, 'INT', -1, [e.single["Index"] for e in entries], go.inputs["Index"], 300)
        b.lookup(index, 'STRING', "", [e.single["Item"] for e in entries], go.inputs["Item"], 100)
        b.lookup(index, 'FLOAT', 0.0, [e.single["Confidence"] for e in entries], go.inputs["Confidence"], -100)
        info = b.new("GeometryNodeCollectionInfo", 150, -350)
        b.link(gi.outputs["Collection"], sock(info.inputs, "Collection"))
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
