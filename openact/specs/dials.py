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
from .base import KEY, Spec, confidence, named, number_text, overrides_inputs, warnings, yes_no_text

SOCKETS = {"yes_no": 'NodeSocketBool', "amount": 'NodeSocketFloat', "one_of": 'NodeSocketInt'}
KINDS = {socket: kind for kind, socket in SOCKETS.items()}
DATA = {"yes_no": 'BOOLEAN', "amount": 'FLOAT', "one_of": 'INT'}
FALLBACK = {"yes_no": False, "amount": 0.0, "one_of": -1}
TYPES = {"yes_no": "yes/no", "amount": "amount", "one_of": "one-of"}  # as the Settings CSV writes them
TIPS = {"yes_no": "True when the answer is yes", "amount": "How much, mapped from 0-1 into the setting's min-max",
        "one_of": "Position of the chosen answer, -1 until answered"}
COLUMNS = (("value", "FLOAT"), ("c", "FLOAT"))  # value: p (yes/no), 0-1 (amount) or the chosen index (one-of)
WARNINGS = {}  # node uid -> the warning under its status: a setting lost its link
RENAMED = {}  # node uid -> (the linked setting that went, the settings that came with it): what ends its warning
FACES = {}  # node uid -> the (name, type) settings its contexts last agreed on (agree()), sorted, for resolve()


@dataclasses.dataclass(frozen=True)
class DialsShape(pipeline.TableShape):
    """A Dials table: its settings with this entry's min/max, in output order, as shape.settings (§14.2)."""

    settings: tuple = ()  # (name, type, min, max) per row

    def as_dict(self) -> dict:
        return dict(super().as_dict(), settings=[{"name": name, "type": kind, "min": low, "max": high}
                                                 for name, kind, low, high in self.settings])


def _outputs(settings):
    """(name, type) of each setting (DialsShape.settings), in order: the outputs they give the node. Their min/max
    are per-entry data (§6.0)."""
    return [(name, kind) for name, kind, _low, _high in settings]


def differ(first_label, first, label, wanted) -> str:
    """The §6.0 error for two contexts whose (name, type) lists differ in more than order: what only one has."""
    ours, theirs = ([f"'{name}' ({TYPES[kind]})" for name, kind in a if (name, kind) not in b]
                    for a, b in ((first, wanted), (wanted, first)))
    if ours and theirs:
        what = f"{', '.join(ours)} vs {', '.join(theirs)}"
    else:
        what = f"only '{first_label if ours else label}' has {', '.join(ours or theirs)}"
    return f"Dials: '{first_label}' and '{label}' use different settings ({what}); setting names and types must agree"


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
            *overrides_inputs(),
            SocketDef(*KEY),
        ]

    def outputs(self, node) -> list[tuple[str, str]]:
        """(name, type) of the setting outputs the node has now, in order."""
        tree = node.node_tree
        items = tree.interface.items_tree if tree is not None else ()
        return [(item.name, KINDS[item.socket_type]) for item in items if item.item_type == 'SOCKET'
                and item.in_out == 'OUTPUT' and item.name != "Confidence" and item.socket_type in KINDS]

    def fallback(self, name):
        """What an output gives a node that reads it as a word while this node shows a hint: Confidence 0.0 (§6.0).
        A setting's type can't be told from its name, so its reader waits, as it does before the first answer."""
        from ..evaluator import Pending  # imports bpy

        if name == "Confidence":
            return 0.0
        raise Pending()

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
        found, problems = self.source(node, ctx), ()
        if found is not None:
            source, doc = found
            try:
                settings, hashes = csvfmt.parse_settings(doc), [doc.text_hash]
            except csvfmt.CsvError as error:
                raise lists.file_error(doc, error) from None
            problems = warnings(doc.problems) + ((source.warning,) if source.warning else ())  # shown (§17.1)
        elif not gn[1].strip():
            raise Hint(self.empty_hint)
        else:
            try:
                settings, hashes = csvfmt.parse_inline_settings(gn[1]), []
            except csvfmt.CsvError as error:
                raise Unresolvable(f"Settings: {error}") from None
        if not gn[0].strip():
            raise Hint("Type a prompt")
        job = self.job(gn, settings, node.balance_order, node.uid, hashes)
        job.problems = problems
        return job

    def job(self, gn, settings, debias, uid="", csv_hashes=()):
        """The Job for these words and settings: one row per setting, every question about the Prompt, so one request.
        Each setting is asked as Ask asks its answer set (§10.2). Pure, so the pure suites plan exactly like the pass.
        min/max enter only the shape, never a wkey."""
        from .. import sources  # pure without bpy

        prompt = gn[0].strip()
        rows = []
        for setting in settings:
            answers = (csvfmt.AnswerSet(tuple(csvfmt.Option(name, when, "", {}) for name, when in setting.answers), None)
                       if setting.answers else sources.default_set(setting.type))  # one-of always has answers
            question = sources.question(setting.type, setting.question, answers)
            criteria = question["criteria"]
            cell = pipeline.Cell(pipeline.cell_key(question["type"], prompt, setting.question, criteria, debias),
                                 prompt, question, pipeline.cost(len(criteria), debias))
            wkey = pipeline.wkey(self.name, setting.type, [prompt], [setting.question], list(setting.answers))
            rows.append(pipeline.Row(setting.name, wkey, cell=cell))
        shape = DialsShape(len(rows), COLUMNS, tuple((s.name, s.type, s.min, s.max) for s in settings))
        words_key = pipeline.words_key(gn, list(csv_hashes), "", shape.digest(), [row.wkey for row in rows])
        return pipeline.Job(uid, tuple(gn), words_key, shape, rows, debias=debias,
                            options=tuple(tuple(name for name, _when in s.answers) for s in settings))

    def agree(self, node, jobs, unknown=False, _entries=()):
        """§6.0, once per pass before any context is answered: every context that plans must plan the same settings,
        by name and type in any order, since each entry fills the outputs by name (_cell()). Jobs that agree give the
        node the outputs of the first one's settings, so they exist before the first answer, and "" is returned; with
        no job, the node keeps its outputs and their links. Otherwise the error naming the first two contexts that
        differ is returned, and the node, its entries and its graph stay as they are. unknown: a live context has no
        job, and may still show the outputs as they are (engine._process()): they stay, and None is returned unless
        the jobs fit them.

        A renamed or retyped setting replaces its socket, and its dropped link warns until that output is back and
        linked (undo, File > Revert), a socket that came with it is linked, or the outputs change again without it."""
        from ..inner import sync_interface  # imports bpy

        uid, have = node.uid, self.outputs(node)
        wanted = have
        if jobs:
            (first_ctx, first), *others = jobs
            wanted = _outputs(first.shape.settings)
            for ctx, job in others:
                theirs = _outputs(job.shape.settings)
                if sorted(theirs) != sorted(wanted):
                    return differ(first_ctx.label(), wanted, ctx.label(), theirs)
            if unknown:
                if sorted(wanted) != sorted(have):
                    return None
                wanted = have
            FACES[uid] = sorted(wanted)
        if wanted != have:
            linked = {socket.name for socket in node.outputs if socket.is_linked}
            gone = [(name, kind) for name, kind in have if (name, kind) not in wanted and name in linked]
            sync_interface(node, self.interface(node, wanted))
            if gone:
                RENAMED[uid] = (gone[0], [pair for pair in wanted if pair not in have])
            elif RENAMED.get(uid, (None,))[0] not in wanted:  # a setting typed back keeps its warning: no link yet
                RENAMED.pop(uid, None)
        gone, came = RENAMED.get(uid, (None, ()))
        linked = {socket.name for socket in node.outputs if socket.is_linked}
        if (gone in wanted and gone[0] in linked) or any(name in linked for name, _kind in came):
            del RENAMED[uid]
        if uid in RENAMED:
            WARNINGS[uid] = f"Setting '{gone[0]}' renamed or retyped: reconnect its output"
        else:
            WARNINGS.pop(uid, None)
        return ""

    def resolve(self, job, entries, last_shown=""):
        """Spec.resolve(), except for a job whose settings differ from the ones the node's contexts last agreed on
        (agree(), §6.0), which the pass doesn't answer: the sidebar and chained reads get the entry stored for its
        words, the last good table the graph still shows, and nothing is asked."""
        agreed = FACES.get(job.uid)
        if agreed is not None and sorted(_outputs(job.shape.settings)) != agreed:
            return next((entry for entry in entries if entry.gn == job.gn), None), [], []
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

    def node_warnings(self, node):
        return (WARNINGS[node.uid],) if node.uid in WARNINGS else ()

    # ------------------------------------------------------------------------------- the inner group

    def build(self, b, gi, go, entries):
        from ..inner import sock  # imports bpy

        node = b.node
        outputs = self.outputs(node)
        k = b.match_chain(b.words(gi, self.WORDS), [e.gn for e in entries])
        if k is None:  # nothing stored: every output shows its fallback
            zero = b.new("FunctionNodeInputInt", -200, 400)
            zero.integer = 0
            k = zero.outputs[0]
        row = b.new("FunctionNodeIntegerMath", -200, 200)
        row.operation = 'SUBTRACT'
        b.link(k, row.inputs[0])
        row.inputs[1].default_value = 1
        live = b._compare('GREATER_THAN', k, 0, -200, 100)
        geometry, index = b.table_geometry(), row.outputs[0]  # a single-value index gives single values (§8.4)
        for j, (name, kind) in enumerate([*outputs, ("Confidence", "conf")]):
            y = 300 - 250 * j
            column, data_type = ("conf", 'FLOAT') if kind == "conf" else (f"v{j}", DATA[kind])
            value = b.sample(geometry, column, data_type, index, 0, y)
            if kind == "amount":
                scale = b.new("ShaderNodeMapRange", 350, y)
                scale.data_type, scale.interpolation_type, scale.clamp = 'FLOAT', 'LINEAR', False
                b.link(value, scale.inputs["Value"])
                b.link(b.sample(geometry, f"min{j}", 'FLOAT', index, 0, y - 80), scale.inputs["To Min"])
                b.link(b.sample(geometry, f"max{j}", 'FLOAT', index, 0, y - 160), scale.inputs["To Max"])
                value = scale.outputs["Result"]
            switch = b.new("GeometryNodeSwitch", 550, y)
            switch.input_type = data_type
            b.link(live, sock(switch.inputs, "Switch"))
            sock(switch.inputs, "False").default_value = 0.0 if kind == "conf" else FALLBACK[kind]
            b.link(value, sock(switch.inputs, "True"))
            b.link(switch.outputs[0], go.inputs[name])

    def layout(self, node, entries):
        """What build() bakes besides the entries' words (inner._built()): output j reads table column j, so a
        Settings CSV whose rows only move needs a new graph, while a min/max edit stays values only."""
        return self.outputs(node)

    def entry_table(self, node, entries):
        """The one-row-per-entry table (§8.4): row e holds materialised entry e's value of each output setting (its
        fallback where the entry lacks it), min and max for Amount, and its lowest confidence."""
        outputs = self.outputs(node)
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

    # ------------------------------------------------------------------ the Overrides CSV (§7.6)

    OVERRIDE_KEYS = ("setting",)

    def lock_values(self, job, rows, answer):
        _name, kind, low, high = job.shape.settings[rows[0]]
        if kind == "yes_no":
            value = 1.0 if answer.casefold() == "yes" else 0.0 if answer.casefold() == "no" else None
            return f"'{answer}' is not yes or no" if value is None else [{"value": value, "c": 1.0}]
        if kind == "amount":  # in output units, mapped back to 0-1
            number = csvfmt.number(answer)
            if number is None or not min(low, high) <= number <= max(low, high):
                return f"'{answer}' is not a number from {low:g} to {high:g}"
            return [{"value": (number - low) / (high - low) if high != low else 0.0, "c": 1.0}]
        chosen = named(answer, job.options[rows[0]])
        return f"'{answer}' is not one of the answers" if chosen is None else [{"value": float(chosen), "c": 1.0}]

    def answer_text(self, job, index, values):
        _name, kind, low, high = job.shape.settings[index]
        if kind == "yes_no":
            return yes_no_text(values["value"])
        if kind == "amount":
            return number_text(mapped(kind, values["value"], low, high))
        options, chosen = job.options[index], int(round(values["value"]))
        return options[chosen] if 0 <= chosen < len(options) else ""
