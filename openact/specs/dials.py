"""Dials: one prompt, many named settings (§6.6).

Every setting is one question about the Prompt, and all of them travel in one request. A context's answers are one
row per setting. The node has one output per setting, synchronised by name and type, and reads them from a
one-row-per-entry table object at k - 1 (§8.4, §8.6): each setting's value, its min and max, and the entry's lowest
confidence. Amount settings go through a Map Range into their min/max. Those are per-entry data (shape.settings,
§14.2), so contexts may use different ranges, and a min/max edit carries every answer over (0 requests).
"""

from __future__ import annotations

import dataclasses

from .. import csvfmt, pipeline
from .ask import AMOUNT_LEVELS
from .base import KEY, Spec, confidence, option_text

SOCKETS = {"yes_no": 'NodeSocketBool', "amount": 'NodeSocketFloat', "one_of": 'NodeSocketInt'}
KINDS = {socket: kind for kind, socket in SOCKETS.items()}
DATA = {"yes_no": 'BOOLEAN', "amount": 'FLOAT', "one_of": 'INT'}
FALLBACK = {"yes_no": False, "amount": 0.0, "one_of": -1}
TYPES = {"yes_no": "yes/no", "amount": "amount", "one_of": "one-of"}  # as the Settings CSV writes them
TIPS = {"yes_no": "True when the answer is yes", "amount": "How much, mapped from 0-1 into the setting's min-max",
        "one_of": "Position of the chosen answer, -1 until answered"}
COLUMNS = (("value", "FLOAT"), ("c", "FLOAT"))  # value: p (yes/no), 0-1 (amount) or the chosen index (one-of)
WARNINGS = {}  # node uid -> the rename warning, shown on the node until its settings change again
FACE = "openact dials outputs"  # evaluator.MEMO key: the outputs the pass's first context of a node wants


@dataclasses.dataclass(frozen=True)
class DialsShape(pipeline.TableShape):
    """A Dials table: its settings with this entry's min/max, in output order, as shape.settings (§14.2)."""

    settings: tuple = ()  # (name, type, min, max) per row

    def as_dict(self) -> dict:
        return dict(super().as_dict(), settings=[{"name": name, "type": kind, "min": low, "max": high}
                                                 for name, kind, low, high in self.settings])


def outputs_of(entry) -> list[tuple[str, str]]:
    """(name, type) of each setting of a stored entry, in output order."""
    return [(s["name"], s["type"]) for s in entry.shape.get("settings", ())]


def mapped(kind, value, low=0.0, high=1.0):
    """A row's value as its output shows it (§6.6)."""
    if kind == "yes_no":
        return value >= 0.5
    if kind == "one_of":
        return int(round(value))
    return low + value * (high - low)


class DialsSpec(Spec):
    name, idname, label = "dials", "GeometryNodeOpenActDials", "Dials"
    WORDS = ("Prompt", "Settings", "Overrides", "Key")
    empty_hint = "Add settings"

    def interface(self, node, outputs=None):
        from ..inner import SocketDef  # imports bpy
        from ..sources import csv_inputs

        outputs = self.outputs(node) if outputs is None else outputs
        return [
            SocketDef('INPUT', "Prompt", 'NodeSocketString',
                      "The situation every setting is judged against, e.g. 'A stormy night in a fishing village'",
                      "word"),
            *csv_inputs("Settings", "Inline 'Rain (yes/no): Is it raining?; Wind (amount): How windy is it?', or a "
                                    ".csv path with columns setting, type, question, answers, min, max"),
            *(SocketDef('OUTPUT', name, SOCKETS[kind], TIPS[kind]) for name, kind in outputs),
            SocketDef('OUTPUT', "Confidence", 'NodeSocketFloat', "The lowest confidence among the settings (0-1)"),
            SocketDef(*KEY),
        ]

    def outputs(self, node) -> list[tuple[str, str]]:
        """(name, type) of the setting outputs the node has now, in order."""
        tree = node.node_tree
        items = tree.interface.items_tree if tree is not None else ()
        return [(item.name, KINDS[item.socket_type]) for item in items if item.item_type == 'SOCKET'
                and item.in_out == 'OUTPUT' and item.name != "Confidence" and item.socket_type in KINDS]

    def fallback(self, name):
        return 0.0 if name == "Confidence" else None

    # ------------------------------------------------------------------------------------------ planning

    def source(self, node, ctx):
        """The Settings CSV of ctx: (source, doc), or None for inline settings."""
        from .. import lists  # imports bpy

        source = lists.csv_source(node, "Settings", ctx)
        return None if source is None else (source, lists.load_csv(node, source, schema="settings"))

    def listed(self, node, ctx):
        found = self.source(node, ctx)  # the file's stamp: an edit re-plans with the same words
        return None if found is None else [found[1].path, found[0].delimiter, list(found[1].stamp)]

    def plan(self, node, ctx):
        from .. import lists  # imports bpy
        from ..evaluator import Hint, Unresolvable

        gn = self.gn(node, ctx)
        found = self.source(node, ctx)
        if found is not None:
            source, doc = found
            try:
                settings, hashes = csvfmt.parse_settings(doc), [doc.text_hash]
            except csvfmt.CsvError as error:
                raise lists.file_error(doc, error) from None
        elif not gn[1].strip():
            raise Hint(self.empty_hint)
        else:
            try:
                settings, hashes = csvfmt.parse_inline_settings(gn[1]), []
            except csvfmt.CsvError as error:
                raise Unresolvable(f"Settings: {error}") from None
        if not gn[0].strip():
            raise Hint("Type a prompt")
        return self.job(gn, settings, node.balance_order, node.uid, hashes)

    def job(self, gn, settings, debias, uid="", csv_hashes=()):
        """The Job for these words and settings: one row per setting, every question about the Prompt, so one request.
        Pure, so the pure suites plan exactly like the pass. min/max enter only the shape, never a wkey."""
        prompt = gn[0].strip()
        rows = []
        for setting in settings:
            question = self.question(setting)
            criteria = question.get("criteria")
            ordered = list(criteria.items()) if isinstance(criteria, dict) else criteria
            options = 2 if criteria is None or setting.type == "yes_no" else len(criteria)
            cell = pipeline.Cell(pipeline.cell_key(question["type"], prompt, setting.question, ordered, debias),
                                 prompt, question, pipeline.cost(options, debias))
            wkey = pipeline.wkey(self.name, setting.type, [prompt], [setting.question], list(setting.answers))
            rows.append(pipeline.Row(setting.name, wkey, cell=cell))
        shape = DialsShape(len(rows), COLUMNS, tuple((s.name, s.type, s.min, s.max) for s in settings))
        words_key = pipeline.words_key(gn, list(csv_hashes), "", shape.digest(), [row.wkey for row in rows])
        return pipeline.Job(uid, tuple(gn), words_key, shape, rows, debias=debias)

    @staticmethod
    def question(setting) -> dict:
        """§10.2: the setting's question; yes/no -> noul, amount -> score over its levels, one-of -> choice."""
        if setting.type == "yes_no":
            asked = {"type": "noul", "instructions": setting.question}
            if setting.answers:  # without whens the server offers "Yes." and "No.", as for Ask
                whens = {name.casefold(): when for name, when in setting.answers}
                asked["criteria"] = {"true": whens.get("yes") or "Yes.", "false": whens.get("no") or "No."}
            return asked
        if setting.type == "amount":
            levels = [option_text(option) for option in setting.answers] or list(AMOUNT_LEVELS)
            return {"type": "score", "instructions": setting.question, "criteria": levels}
        return {"type": "choice", "instructions": setting.question,
                "criteria": {str(i): option_text(option) for i, option in enumerate(setting.answers)}}

    def resolve(self, job, entries, last_shown=""):
        from .. import evaluator  # imports bpy

        if evaluator.MEMO is not None:  # in a pass: every live context must agree on the outputs (§6.0)
            wanted = [(name, kind) for name, kind, _low, _high in job.shape.settings]
            first = evaluator.MEMO.setdefault((FACE, job.uid), wanted)
            if first != wanted:
                one, other = ([f"'{name}' ({TYPES[kind]})" for name, kind in a if (name, kind) not in b]
                              for a, b in ((first, wanted), (wanted, first)))
                what = f"{', '.join(one) or '-'} vs {', '.join(other) or '-'}" if one or other else "another order"
                return None, [], [f"Dials: contexts use different settings ({what}); setting names and types must "
                                  f"agree"]
        return super().resolve(job, entries, last_shown)

    # ------------------------------------------------------------------------------------------- answers

    def values(self, index, answer):
        if "amount" in answer:
            return {"value": answer["amount"], "c": confidence('AMOUNT', answer["probabilities"])}
        if "p" in answer:
            return {"value": answer["p"], "c": confidence('YES_NO', (answer["p"], 1.0 - answer["p"]))}
        probabilities = answer["probabilities"]
        chosen = max(range(len(probabilities)), key=probabilities.__getitem__)
        return {"value": float(chosen), "c": confidence('ONE_OF', probabilities)}

    def entry(self, job, rows, model=""):
        entry = super().entry(job, rows, model)
        entry.single = {s["name"]: mapped(s["type"], value, s["min"], s["max"])
                        for s, value in zip(entry.shape["settings"], entry.rows.v["value"])}
        entry.single["Confidence"] = min(entry.rows.v["c"], default=0.0)
        return entry

    def summary(self, entry):
        count = len(entry.rows.name)
        return f"{count} setting{'s' if count != 1 else ''} · lowest confidence {entry.single['Confidence']:.2f}"

    # ------------------------------------------------------------------------------- the inner group

    def build(self, b, gi, go, entries):
        from .. import tables  # imports bpy
        from ..inner import sock

        node = b.node
        outputs = self.sync(node, entries)
        k = b.match_chain(b.words(gi, self.WORDS), [e.gn for e in entries])
        if k is None:  # nothing stored: every output shows its fallback
            zero = b.new("FunctionNodeInputInt", -200, 400)
            zero.integer = 0
            k = zero.outputs[0]
        info = b.new("GeometryNodeObjectInfo", -200, -200)
        info.name = tables.NODE
        info.inputs["Object"].default_value = b.table
        row = b.new("FunctionNodeIntegerMath", -200, 200)
        row.operation = 'SUBTRACT'
        b.link(k, row.inputs[0])
        row.inputs[1].default_value = 1
        live = b._compare('GREATER_THAN', k, 0, -200, 100)
        geometry, index = info.outputs["Geometry"], row.outputs[0]
        for j, (name, kind) in enumerate([*outputs, ("Confidence", "conf")]):
            y = 300 - 250 * j
            column, data_type = ("conf", 'FLOAT') if kind == "conf" else (f"v{j}", DATA[kind])
            value = self._sample(b, geometry, column, data_type, index, 0, y)
            if kind == "amount":
                scale = b.new("ShaderNodeMapRange", 350, y)
                scale.data_type, scale.interpolation_type, scale.clamp = 'FLOAT', 'LINEAR', False
                b.link(value, scale.inputs["Value"])
                b.link(self._sample(b, geometry, f"min{j}", 'FLOAT', index, 0, y - 80), scale.inputs["To Min"])
                b.link(self._sample(b, geometry, f"max{j}", 'FLOAT', index, 0, y - 160), scale.inputs["To Max"])
                value = scale.outputs["Result"]
            switch = b.new("GeometryNodeSwitch", 550, y)
            switch.input_type = data_type
            b.link(live, sock(switch.inputs, "Switch"))
            sock(switch.inputs, "False").default_value = 0.0 if kind == "conf" else FALLBACK[kind]
            b.link(value, sock(switch.inputs, "True"))
            b.link(switch.outputs[0], go.inputs[name])

    @staticmethod
    def _sample(b, geometry, column, data_type, index, x, y):
        """One value of the table object's row `index`: Named Attribute and Sample Index (a single value, §8.4)."""
        from .. import tables  # imports bpy
        from ..inner import _enabled

        attribute = b.new("GeometryNodeInputNamedAttribute", x, y)
        attribute.data_type = data_type
        attribute.inputs["Name"].default_value = tables.ATTR + column
        sampled = b.new("GeometryNodeSampleIndex", x + 170, y)
        sampled.data_type = data_type
        b.link(geometry, sampled.inputs["Geometry"])
        b.link(_enabled(attribute.outputs, "Attribute"), _enabled(sampled.inputs, "Value"))
        b.link(index, sampled.inputs["Index"])
        return _enabled(sampled.outputs, "Value")

    def sync(self, node, entries):
        """Give the node the outputs of its newest entry (§6.0): matched by name and type, so a renamed or retyped
        setting replaces its socket, and a link to it is dropped with a warning. Returns the outputs."""
        from ..inner import sync_interface  # imports bpy

        have = self.outputs(node)
        newest = max(entries, key=lambda entry: entry.t, default=None)
        wanted = have if newest is None else outputs_of(newest)
        if wanted != have:
            linked = {socket.name for socket in node.outputs if socket.is_linked}
            gone = [name for name, kind in have if (name, kind) not in wanted and name in linked]
            sync_interface(node, self.interface(node, wanted))
            if gone:
                WARNINGS[node.uid] = f"Setting '{gone[0]}' renamed or retyped: reconnect its output"
            else:
                WARNINGS.pop(node.uid, None)
        return wanted

    def entry_table(self, node, entries):
        """The one-row-per-entry table (§8.4): row e holds materialised entry e's value of each output setting (its
        fallback where the entry lacks it), min and max for Amount, and its lowest confidence. None when the newest
        entry wants other outputs than the node has, so build() must run."""
        outputs = self.outputs(node)
        newest = max(entries, key=lambda entry: entry.t, default=None)
        if newest is not None and outputs_of(newest) != outputs:
            return None
        columns = {"conf": [min(entry.rows.v.get("c", ()), default=0.0) for entry in entries]}
        types = {"conf": 'FLOAT'}
        for j, (name, kind) in enumerate(outputs):
            cells = [self._cell(entry, name, kind) for entry in entries]
            columns[f"v{j}"], types[f"v{j}"] = [cell[0] for cell in cells], DATA[kind]
            if kind == "amount":
                columns[f"min{j}"], columns[f"max{j}"] = [cell[1] for cell in cells], [cell[2] for cell in cells]
                types[f"min{j}"] = types[f"max{j}"] = 'FLOAT'
        return columns, types

    @staticmethod
    def _cell(entry, name, kind):
        """(value, min, max) of an entry's setting, with the value in its table type."""
        for i, setting in enumerate(entry.shape.get("settings", ())):
            if (setting["name"], setting["type"]) == (name, kind):
                value = entry.rows.v["value"][i]
                return (mapped(kind, value) if kind != "amount" else float(value)), setting["min"], setting["max"]
        return FALLBACK[kind], 0.0, 1.0
