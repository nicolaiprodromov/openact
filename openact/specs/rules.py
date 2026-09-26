"""Rules: each measured situation (§6.5, §9). One question per cell, a combination of factor levels; Geometry Nodes
finds each element's cell from its factor sockets and reads that cell's row: the Outcome and its payload (One-of,
optionally dithered), or Probability / Amount blended over up to 3 Amount factors (Yes/No, Amount).

The maths is rules_core's, which stays pure; this module turns it into Jobs, entries and the inner graph. Pure when
imported: the methods that read a node import the evaluator, and with it bpy.
"""

from __future__ import annotations

import dataclasses
import itertools
import os

from .. import csvfmt, pipeline, rules_core
from .base import KEY, Spec, confidence, lock_answer, named, number_text, overrides_inputs, warnings, yes_no_text

FACTORS_HINT = "Set a Factors CSV (sidebar: New Rules Template)"
OUTCOMES_HINT = "Add outcomes"
UNSURE_BELOW = 0.30  # 4.e; the preference arrives with PR-22
PAYLOAD = "pl_"  # table column prefix of a payload output, followed by its position
HEAD = ("Question", "Context", "Factors", "Factors CSV", "Outcomes", "Outcomes CSV", "Overrides", "Overrides CSV",
        "Dither", "Seed", "Key")  # inputs that are not factors
SOCKETS = {"amount": 'NodeSocketFloat', "category": 'NodeSocketInt', "flag": 'NodeSocketBool'}
MODES = {"Outcome": 'ONE_OF', "Probability": 'YES_NO', "Amount": 'AMOUNT'}  # the output that tells the mode
DECLARED = {}  # node uid -> (factors ((name, kind), ...), payload ((name, type), ...)) of its last plan


@dataclasses.dataclass(frozen=True)
class RulesShape(pipeline.TableShape):
    """A Rules table's shape (§14.2): everything build() needs, so no CSV file is."""

    factors: tuple = ()  # declare() of each factor
    payload: tuple = ()  # One-of: (name, "INT" | "FLOAT", value per outcome row)

    def as_dict(self):
        return {**super().as_dict(), "factors": [dict(f) for f in self.factors],
                "payload": [[name, kind, list(values)] for name, kind, values in self.payload]}


def declare(factor) -> dict:
    """What the graph bakes of a factor (§6.0): kind, levels, category and flag values, custom band edges."""
    custom = factor.kind == "amount" and factor.levels[0].band is not None
    return {"name": factor.name, "kind": factor.kind, "levels": [level.name for level in factor.levels],
            "values": None if factor.kind == "amount" else [level.value for level in factor.levels],
            "edges": list(rules_core.band_edges(factor)) if custom else None}


def factor_of(data) -> csvfmt.Factor:
    """declare() back into a csvfmt.Factor, for rules_core."""
    edges, values = data.get("edges"), data.get("values")
    levels = tuple(csvfmt.Level(name, "", values[b] if values else None, (edges[b], edges[b + 1]) if edges else None, 0)
                   for b, name in enumerate(data["levels"]))
    return csvfmt.Factor(data["name"], data["kind"], levels, 0)


class RulesSpec(Spec):
    name, idname, label = "rules", "GeometryNodeOpenActRules", "Rules"
    WORDS = ("Question", "Context", "Factors", "Outcomes", "Overrides", "Key")
    empty_hint = FACTORS_HINT

    # ------------------------------------------------------------------ sockets

    def declared(self, node):
        """(factors, payload) the interface shows: the last plan's, else the newest stored entry's."""
        found = DECLARED.get(node.uid)
        if found is not None:
            return found
        from .. import store

        entries = [entry for entry in store.load(node).entries if entry.shape.get("factors")]
        if not entries:
            return (), ()
        shape = max(entries, key=lambda entry: entry.t).shape
        return (tuple((f["name"], f["kind"]) for f in shape["factors"]),
                tuple((p[0], p[1]) for p in shape.get("payload") or ()))

    def interface(self, node):
        from .. import sources
        from ..inner import SocketDef

        factors, payload = self.declared(node)
        mode = node.answer_type
        defs = [
            SocketDef('INPUT', "Question", 'NodeSocketString',
                      "What to decide for each cell, e.g. 'What happens to this cell in the next generation?'",
                      kind="word"),
            SocketDef('INPUT', "Context", 'NodeSocketString', "The situation every cell shares", kind="word"),
            *sources.csv_inputs("Factors", "A Factors CSV: factor,kind,level,when,value,from,to"),
            *sources.csv_inputs("Outcomes", "The outcomes: inline, or a CSV with answer,when[,if][,payload...]"),
            *overrides_inputs(),
        ]
        defs += [SocketDef('INPUT', name, SOCKETS[kind], f"Factor '{name}' ({kind}), per element", kind="field")
                 for name, kind in factors]
        if mode == 'ONE_OF':
            defs += [SocketDef('INPUT', "Dither", 'NodeSocketFloat', "0-1: blend outcomes near Amount band edges"),
                     SocketDef('INPUT', "Seed", 'NodeSocketInt', "Seed for Dither")]
            outputs = [("Outcome", 'NodeSocketInt', "Row of the chosen outcome, -1 until answered"),
                       ("Confidence", 'NodeSocketFloat', "How sure the cell's answer is (0-1)")]
            outputs += [(name, 'NodeSocketInt' if kind == "INT" else 'NodeSocketFloat',
                         f"The Outcomes CSV's '{name}' of the chosen outcome") for name, kind in payload]
        elif mode == 'YES_NO':
            outputs = [("Probability", 'NodeSocketFloat', "P(yes), blended over Amount factors"),
                       ("Yes", 'NodeSocketBool', "True when Probability is 0.5 or more"),
                       ("Confidence", 'NodeSocketFloat', "How sure the answer is (0-1)")]
        else:
            outputs = [("Amount", 'NodeSocketFloat', "How much (0-1), blended over Amount factors"),
                       ("Confidence", 'NodeSocketFloat', "How sure the answer is (0-1)")]
        defs += [SocketDef('OUTPUT', name, socket_type, description) for name, socket_type, description in outputs]
        return defs + [SocketDef(*KEY)]

    def _declare(self, node, factors, answers):
        """Remember what this plan declares; when the interface differs, sync it and rebuild (§6.0)."""
        from .. import inner, store
        from ..evaluator import Pending

        payload = ()
        if node.answer_type == 'ONE_OF' and answers.doc is not None:
            payload = tuple(answers.doc.num_cols.items())
        DECLARED[node.uid] = (tuple((f.name, f.kind) for f in factors), payload)
        wanted = self.interface(node)
        expected = [(d.in_out, d.name, d.socket_type) for d in sorted(wanted, key=lambda d: d.in_out != 'OUTPUT')]
        if node.node_tree is None or _face(node.node_tree) == expected:
            return
        try:
            inner.sync_interface(node, wanted)
            inner.build(node, store.load(node))
        except (AttributeError, RuntimeError) as error:  # a draw callback may not write: the pass will
            raise Pending() from error

    # ------------------------------------------------------------------ planning

    def listed(self, node, ctx):
        """The CSV files plan() reads, by path and stamp, so an edit plans again (§5.3)."""
        from .. import lists
        from ..evaluator import Hint, Pending, Unresolvable

        seen = []
        for name in ("Factors", "Outcomes"):
            try:
                source = lists.csv_source(node, name, ctx)
                path = lists.resolve_path(source.path, node) if source is not None else ""
                stat = os.stat(path) if path else None
                seen.append([path, source and source.delimiter, stat and stat.st_mtime_ns, stat and stat.st_size])
            except (Hint, Pending, Unresolvable, OSError) as error:
                seen.append(str(error))
        return seen

    def plan(self, node, ctx):
        from .. import lists, sources
        from ..evaluator import Hint, Unresolvable

        gn = self.gn(node, ctx)
        source = lists.csv_source(node, "Factors", ctx)
        if source is None:
            raise Hint(FACTORS_HINT)
        doc = lists.load_csv(node, source, schema="factors")
        try:
            factors = csvfmt.parse_factors(doc)
        except csvfmt.CsvError as error:
            raise lists.file_error(doc, error) from None
        kind = sources.KINDS[node.answer_type]
        answers = sources.answer_set(node, ctx, "Outcomes", kind, spec="rules", empty_hint=OUTCOMES_HINT)
        hashes = [doc.text_hash] + ([answers.doc.text_hash] if answers.doc is not None else [])
        try:
            job = self.job(gn, factors, answers, node.answer_type, node.balance_order, hashes, node.uid)
        except rules_core.RulesHint as hint:
            raise Hint(str(hint)) from None
        except rules_core.RulesError as error:
            raise Unresolvable(str(error)) from None
        except csvfmt.CsvError as error:  # a bad `if`
            if answers.doc is None:
                raise Unresolvable(f"Outcomes: {error}") from None
            raise lists.file_error(answers.doc, error) from None
        job.problems = (warnings(doc.problems + answers.problems) + ((source.warning,) if source.warning else ())
                        + job.problems)  # the CSVs' warnings, then the table's
        self._declare(node, factors, answers)
        return self.locked(node, ctx, job)

    def job(self, gn, factors, answers, answer_type, debias, hashes=(), uid=""):
        """The Job for these words, factors and outcomes (None: the default Yes/No or Amount levels). Pure, so the
        pure suites plan exactly like the pass. Raises RulesHint, RulesError or CsvError."""
        question, context = gn[0].strip(), gn[1].strip()
        planned = rules_core.plan_table(factors, answer_type, answers, question, context, debias)
        options = answers.options if answers is not None else rules_core.default_options(answer_type)
        rows = []
        for cell in planned.cells:
            facts = [[f.name, f.levels[b].name, f.levels[b].when] for f, b in zip(factors, cell.levels)]
            offered = [[options[r].name, options[r].when] for r in cell.allowed]
            wkey = pipeline.wkey(self.name, answer_type, [question, context], facts, offered)
            if cell.source != "model":
                rows.append(pipeline.Row(cell.name, wkey, cell.source, dict(cell.values)))
                continue
            asked = cell.question
            ordered = list(asked["criteria"].items()) if isinstance(asked["criteria"], dict) else asked["criteria"]
            key = pipeline.cell_key(asked["type"], planned.state, asked["instructions"], ordered, debias)
            rows.append(pipeline.Row(cell.name, wkey, cell=pipeline.Cell(key, planned.state, asked, cell.cost)))
        if answer_type == 'ONE_OF':
            columns = (("outcome", "INT"), ("c", "FLOAT"))
        else:
            columns = ((rules_core.value_column(answer_type), "FLOAT"), ("c", "FLOAT"), ("ok", "FLOAT"))
        payload = ()
        if answer_type == 'ONE_OF' and answers is not None and answers.doc is not None:
            payload = tuple((name, kind, tuple(option.payload.get(name, 0) for option in options))
                            for name, kind in answers.doc.num_cols.items())
        shape = RulesShape(len(rows), columns, tuple(declare(f) for f in factors), payload)
        words_key = pipeline.words_key(gn, list(hashes), answer_type, shape.digest(), [row.wkey for row in rows])
        return pipeline.Job(uid, tuple(gn), words_key, shape, rows, answer_type=answer_type, debias=debias,
                            options=tuple(option.name for option in options), problems=warnings(planned.warnings),
                            errors=(planned.error,) if planned.error else ())

    # ------------------------------------------------------------------ the Overrides CSV (§7.6)

    def override_keys(self, job):
        return [f["name"] for f in job.shape.factors]  # one column per factor, holding level names

    def override_units(self, job):
        levels = itertools.product(*(f["levels"] for f in job.shape.factors))  # row order is cell order
        return [(cells, (i,)) for i, cells in enumerate(levels)]

    def lock_values(self, job, rows, answer):
        if job.answer_type == 'ONE_OF':
            outcome = named(answer, job.options)
            return f"'{answer}' is not an outcome" if outcome is None else [{"outcome": outcome, "c": 1.0}]
        value = lock_answer(job.answer_type, answer, job.options)
        return value if isinstance(value, str) else [{rules_core.value_column(job.answer_type): value, "c": 1.0,
                                                      "ok": 1.0}]

    def answer_text(self, job, index, values):
        if job.answer_type == 'ONE_OF':
            outcome = values["outcome"]
            return job.options[outcome] if 0 <= outcome < len(job.options) else ""
        value = values[rules_core.value_column(job.answer_type)]
        return number_text(value) if job.answer_type == 'AMOUNT' else yes_no_text(value)

    # ------------------------------------------------------------------ answers

    def rows(self, job, answers):
        """As Spec.rows(); a One-of answer's winner is mapped back to its outcome row through the criteria keys."""
        out = []
        for row in job.rows:
            answer = answers.get(row.cell.key) if row.source == "pending" and row.cell is not None else None
            if answer is not None:
                row = dataclasses.replace(row, source="model", ask=self.ask_key(job.debias, answer["model"]),
                                          values=self.answer_values(job.answer_type, row.cell, answer))
            out.append(row)
        return out

    @staticmethod
    def answer_values(answer_type, cell, answer):
        if answer_type == 'ONE_OF':
            probabilities = answer["probabilities"]
            best = max(range(len(probabilities)), key=probabilities.__getitem__)  # the first of equal ones
            return {"outcome": int(list(cell.question["criteria"])[best]), "c": confidence('ONE_OF', probabilities)}
        if answer_type == 'AMOUNT':
            return {"amount": answer["amount"], "c": confidence('AMOUNT', answer["probabilities"]), "ok": 1.0}
        return {"p": answer["p"], "c": confidence('YES_NO', (answer["p"], 1.0 - answer["p"])), "ok": 1.0}

    def entry(self, job, rows, model=""):
        """Spec.entry(), plus One-of's payload denormalised per cell from its outcome (rows.payload)."""
        asked = next((row.ask for row in rows if row.ask), "")  # code and error rows were asked nothing
        rows = [row if row.ask is not None else dataclasses.replace(row, ask=asked) for row in rows]
        entry = super().entry(job, rows, model)
        outcomes = entry.rows.v.get("outcome")
        if outcomes is not None:
            entry.rows.payload = {
                name: [values[o] if 0 <= o < len(values) else (-1 if kind == "INT" else 0.0) for o in outcomes]
                for name, kind, values in getattr(job.shape, "payload", ())}
        return entry

    def single(self, rows):
        return {"Confidence": sum(row.values["c"] for row in rows) / len(rows) if rows else 0.0}

    def summary(self, entry):
        src, confidences = entry.rows.src, entry.rows.v.get("c", [])
        unsure = sum(1 for s, c in zip(src, confidences) if s == "m" and c < UNSURE_BELOW)
        text = f"{len(src)} cells · {unsure} unsure"
        if "e" in src:
            text += f" · {src.count('e')} with no outcome"
        return text

    # ------------------------------------------------------------------ the table object

    def per_element(self, node):
        """The table columns (inner.Column): One-of's are looked up at the cell by inner's per_element(); the
        Yes/No and Amount ones and the category map are read by build() itself (output "")."""
        from ..inner import Column

        mode = _mode(node.node_tree) if node.node_tree is not None else None
        columns = [Column("", "map", 'INT', -1)]
        if mode == 'ONE_OF':
            columns += [Column("Outcome", "outcome", 'INT', -1), Column("Confidence", "c", 'FLOAT', 0.0)]
            payload = [out for out in node.outputs
                       if out.name not in MODES and out.name != "Confidence" and out.type in ('INT', 'VALUE')]
            for k, out in enumerate(payload):  # by position: a header may be too long for an attribute name (§8.5)
                integer = out.type == 'INT'
                columns.append(Column(out.name, f"{PAYLOAD}{k}", 'INT' if integer else 'FLOAT', -1 if integer else 0.0,
                                      payload=True))
        elif mode is not None:
            columns += [Column("", rules_core.value_column(mode), 'FLOAT', 0.0), Column("", "c", 'FLOAT', 0.0),
                        Column("", "ok", 'FLOAT', 0.0)]
        return tuple(columns)

    def table_rows(self, node, entries, columns):
        """inner.flat_rows(), then the category maps after every cell row (§9.2)."""
        from ..inner import flat_rows

        tree = node.node_tree
        flat, fallbacks = flat_rows(entries, columns), {column.name: column.fallback for column in columns}
        for rows in _maps(_reference(tree, entries, _mode(tree))):
            for name, values in flat.items():
                values.extend(rows if name == "map" else [fallbacks[name]] * len(rows))
        return flat

    def layout(self, node, entries):
        """What build() bakes from the entries besides their words (inner._built()): the reference entry's
        factors, and the cell rows that the category maps follow."""
        tree = node.node_tree
        reference = _reference(tree, entries, _mode(tree))
        return reference and [reference.shape["factors"], sum(len(entry.rows.name) for entry in entries)]

    # ------------------------------------------------------------------ the graph

    def build(self, b, gi, go, entries):
        b.match_chain(self.word_sockets(gi), [e.gn for e in entries])
        mode = _mode(b.tree)
        reference = _reference(b.tree, entries, mode)
        if reference is None:
            if mode == 'ONE_OF':  # nothing answered for these factors: every element reads the fallbacks
                b.index = _node(b, "FunctionNodeInputInt", -300, -700, integer=-1).outputs[0]
            return
        graph = _Graph(b, gi, go, entries, reference)
        if mode == 'ONE_OF':
            b.index = graph.one_of()
        else:
            graph.blend(mode)


# ---------------------------------------------------------------------- the interface and the reference entry


def _face(tree):
    return [(i.in_out, i.name, i.socket_type) for i in tree.interface.items_tree if i.item_type == 'SOCKET']


def _mode(tree):
    names = {i.name for i in tree.interface.items_tree if i.item_type == 'SOCKET' and i.in_out == 'OUTPUT'}
    return next((mode for name, mode in MODES.items() if name in names), None)


def _factor_sockets(tree):
    kinds = {socket: kind for kind, socket in SOCKETS.items()}
    return [(i.name, kinds[i.socket_type]) for i in tree.interface.items_tree
            if i.item_type == 'SOCKET' and i.in_out == 'INPUT' and i.name not in HEAD and i.socket_type in kinds]


def _reference(tree, entries, mode):
    """The entry whose shape the graph bakes: the newest one of the node's answer type whose factors are the
    interface's. Entries of other shapes stay in the match chain but read misaddressed rows (§6.0 is left)."""
    face = _factor_sockets(tree)
    fitting = [e for e in entries if e.answer_type == mode and e.shape.get("factors")
               and [(f["name"], f["kind"]) for f in e.shape["factors"]] == face]
    return max(fitting, key=lambda e: e.t, default=None)


def _maps(reference):
    """The map rows of each non-contiguous category factor, in factor order."""
    if reference is None:
        return []
    maps = (rules_core.category_map(factor_of(f)) for f in reference.shape["factors"] if f["kind"] == "category")
    return [rows for rows in maps if rows is not None]


# ---------------------------------------------------------------------- node helpers


def _node(b, idname, x, y, **settings):
    node = b.new(idname, x, y)
    for name, value in settings.items():
        setattr(node, name, value)
    return node


def _feed(b, socket, value):
    """Link a socket, or set a number or bool."""
    if isinstance(value, (bool, int, float)):
        socket.default_value = value
    elif value is not None:
        b.link(value, socket)


def _enabled(sockets, name):
    return next(s for s in sockets if s.name == name and s.enabled)


def _sock(sockets, identifier):
    return next(s for s in sockets if s.identifier == identifier)


def _math(b, operation, x, y, *values, clamp=False):
    node = _node(b, "ShaderNodeMath", x, y, operation=operation, use_clamp=clamp)
    for socket, value in zip(node.inputs, values):
        _feed(b, socket, value)
    return node.outputs[0]


def _int(b, operation, x, y, *values):
    node = _node(b, "FunctionNodeIntegerMath", x, y, operation=operation)
    for socket, value in zip(node.inputs, values):
        _feed(b, socket, value)
    return node.outputs[0]


def _cmp(b, data_type, operation, first, second, x, y):
    node = _node(b, "FunctionNodeCompare", x, y, data_type=data_type, operation=operation)
    names = ("A", "B") if data_type == 'FLOAT' else ("A_INT", "B_INT")
    _feed(b, _sock(node.inputs, names[0]), first)
    _feed(b, _sock(node.inputs, names[1]), second)
    return node.outputs[0]


def _switch(b, data_type, condition, false, true, x, y):
    node = _node(b, "GeometryNodeSwitch", x, y, input_type=data_type)
    for name, value in (("Switch", condition), ("False", false), ("True", true)):
        _feed(b, _sock(node.inputs, name), value)
    return node.outputs[0]


def _and(b, first, second, x, y):
    if first is None or second is None:
        return second if first is None else first
    node = _node(b, "FunctionNodeBooleanMath", x, y, operation='AND')
    b.link(first, node.inputs[0])
    b.link(second, node.inputs[1])
    return node.outputs[0]


class _Graph:
    """The per-element graph of one reference shape: level indices (§9.2), the cell index (§9.3), dithering
    (§9.7) and the corner blend (§9.6). rules_core holds the Python reference of each."""

    def __init__(self, b, gi, go, entries, reference):
        self.b, self.gi, self.go = b, gi, go
        self.factors = [factor_of(f) for f in reference.shape["factors"]]
        self.steps = rules_core.strides(rules_core.level_counts(self.factors))
        self.sockets = [gi.outputs.get(f.name) for f in self.factors]
        self.counts = [len(entry.rows.name) for entry in entries]
        self.bases = list(itertools.accumulate([0] + self.counts))[:-1]
        self.map_base, at = {}, sum(self.counts)  # the maps follow every cell row (table_rows())
        for j, factor in enumerate(self.factors):
            rows = rules_core.category_map(factor) if factor.kind == "category" else None
            if rows is not None:
                self.map_base[j] = at
                at += len(rows)
        self.clamped = {}

    def value(self, j):
        """Factor j's socket, or its default when the interface lacks it."""
        socket = self.sockets[j]
        if socket is not None:
            return socket
        return {"amount": 0.0, "category": 0, "flag": False}[self.factors[j].kind]

    def x(self, j):
        """x = clamp(v, 0, 1) of Amount factor j."""
        if j not in self.clamped:
            node = _node(self.b, "ShaderNodeClamp", -700, -300 * j)
            node.inputs["Value"].default_value = 0.0
            _feed(self.b, node.inputs["Value"], self.value(j))
            self.clamped[j] = node.outputs[0]
        return self.clamped[j]

    def band(self, factor, x, left, y):
        """§9.2 amount: min(floor(x n), n - 1) for equal widths, else the count of interior edges at or below x."""
        b, n = self.b, len(factor.levels)
        if factor.levels[0].band is None:
            floor = _node(b, "FunctionNodeFloatToInt", left + 150, y, rounding_mode='FLOOR')
            b.link(_math(b, 'MULTIPLY', left, y, x, float(n)), floor.inputs[0])
            return _int(b, 'MINIMUM', left + 300, y, floor.outputs[0], n - 1)
        total = 0
        for e, level in enumerate(factor.levels[1:]):
            above = _cmp(b, 'FLOAT', 'GREATER_EQUAL', x, level.band[0], left, y - 40 * e)
            total = _int(b, 'ADD', left + 150, y - 40 * e, above, total)
        return total

    def level(self, j):
        """(b, valid) of factor j, undithered; valid None when always valid (§9.2)."""
        b, factor, value, y = self.b, self.factors[j], self.value(j), -300 * j
        if factor.kind == "flag":
            zero = next(i for i, level in enumerate(factor.levels) if level.value == 0)
            return _switch(b, 'INT', value, zero, 1 - zero, -500, y), None
        if factor.kind == "category":
            n, rows = len(factor.levels), rules_core.category_map(factor)
            if rows is None:
                level = _int(b, 'SUBTRACT', -500, y, value, factor.levels[0].value)
                valid = _and(b, _cmp(b, 'INT', 'GREATER_EQUAL', level, 0, -350, y),
                             _cmp(b, 'INT', 'LESS_THAN', level, n, -350, y - 40), -200, y)
                return level, valid
            inside = _and(b, _cmp(b, 'INT', 'GREATER_EQUAL', value, 0, -650, y),
                          _cmp(b, 'INT', 'LESS_EQUAL', value, len(rows) - 1, -650, y - 40), -500, y)
            at = _int(b, 'ADD', -650, y - 80, value, self.map_base[j])
            mapped = b.sample(b.table_geometry(), "map", 'INT', at, -500, y - 120)
            level = _switch(b, 'INT', inside, -1, mapped, -200, y)
            return level, _cmp(b, 'INT', 'GREATER_EQUAL', level, 0, -50, y)
        return self.band(factor, self.x(j), -550, y), None

    def cell(self, levels, start=0):
        """§9.3: start + the sum of b_j * stride_j; -1 unless every factor is valid (start = 0 only)."""
        b, acc, valid = self.b, start, None
        for j, ((level, ok), step) in enumerate(zip(levels, self.steps)):
            acc = _int(b, 'MULTIPLY_ADD', 0, -300 * j, level, step, acc)
            valid = _and(b, valid, ok, 0, -300 * j - 100)
        if valid is None:
            return acc
        return _switch(b, 'INT', valid, -1, acc, 150, 0)

    def one_of(self):
        """The One-of cell (§9.3), each Amount coordinate jittered by (r - 0.5) Dither widths of its band (§9.7).
        The Random Value branch is always built: Dither 0 multiplies it away."""
        b = self.b
        seed, dither = self.gi.outputs.get("Seed"), self.gi.outputs.get("Dither")
        levels = []
        for j, factor in enumerate(self.factors):
            if factor.kind != "amount":
                levels.append(self.level(j))
                continue
            x, y = self.x(j), -300 * j
            random = _node(b, "FunctionNodeRandomValue", -700, y - 150, data_type='FLOAT')
            b.link(_int(b, 'ADD', -850, y - 150, seed if seed is not None else 0,
                        rules_core.DITHER_SEED_STEP * j), random.inputs["Seed"])
            offset = _math(b, 'SUBTRACT', -550, y - 150, _enabled(random.outputs, "Value"), 0.5)
            jitter = _math(b, 'MULTIPLY', -400, y - 150, offset, dither if dither is not None else 0.0)
            moved = _node(b, "ShaderNodeMath", -250, y - 100, operation='MULTIPLY_ADD', use_clamp=True)
            b.link(jitter, moved.inputs[0])
            b.link(x, moved.inputs[2])
            n = len(factor.levels)
            if factor.levels[0].band is None:
                moved.inputs[1].default_value = 1.0 / n
            else:
                edges = rules_core.band_edges(factor)
                widths = [edges[k + 1] - edges[k] for k in range(n)]
                b.lookup(self.band(factor, x, -550, y + 100), 'FLOAT', widths[0], widths[1:], moved.inputs[1], y)
            levels.append((self.band(factor, moved.outputs[0], -100, y - 100), None))
        return self.cell(levels)

    def axis(self, j):
        """(b0, t) of interpolated factor j (§9.6); b0 may be the number 0."""
        b, factor, x, y = self.b, self.factors[j], self.x(j), -300 * j
        n = len(factor.levels)
        if factor.levels[0].band is None:
            u = _node(b, "ShaderNodeClamp", -400, y)
            b.link(_math(b, 'MULTIPLY_ADD', -550, y, x, float(n), -0.5), u.inputs["Value"])
            u.inputs["Min"].default_value, u.inputs["Max"].default_value = 0.0, float(n - 1)
            floor = _node(b, "FunctionNodeFloatToInt", -250, y, rounding_mode='FLOOR')
            b.link(u.outputs[0], floor.inputs[0])
            b0 = _int(b, 'MINIMUM', -100, y, floor.outputs[0], n - 2)
            return b0, _math(b, 'SUBTRACT', 50, y, u.outputs[0], b0)
        centres = list(rules_core.band_centres(factor))
        b0 = 0
        for e, centre in enumerate(centres[1:n - 1]):
            b0 = _int(b, 'ADD', -400, y - 40 * e, _cmp(b, 'FLOAT', 'GREATER_EQUAL', x, centre, -550, y - 40 * e), b0)
        offset = _node(b, "ShaderNodeMath", -100, y, operation='SUBTRACT')  # x - m_b0
        span = _node(b, "ShaderNodeMath", -100, y - 100, operation='SUBTRACT')  # m_b0+1 - m_b0
        b.link(x, offset.inputs[0])
        if isinstance(b0, int):
            offset.inputs[1].default_value = span.inputs[1].default_value = centres[0]
            span.inputs[0].default_value = centres[1]
        else:
            b.link(b.lookup(b0, 'FLOAT', centres[0], centres[1:], offset.inputs[1], y), span.inputs[1])
            b.lookup(_int(b, 'ADD', -250, y - 100, b0, 1), 'FLOAT', centres[0], centres[1:], span.inputs[0], y - 100)
        return b0, _math(b, 'DIVIDE', 50, y, offset.outputs[0], span.outputs[0], clamp=True)

    def blend(self, mode):
        """Yes/No and Amount (§9.6): the 2^Ka corner rows weighted by w * ok and renormalised; the fallbacks when
        a category value matches no level, no entry matches, or the answered corners weigh <= 1e-6."""
        from ..inner import BASE, COUNT

        b, go = self.b, self.go
        interpolated = rules_core.interpolated_factors(self.factors)
        start = _node(b, "FunctionNodeIntegerMath", 300, 300, operation='ADD')
        b.lookup(b.k, 'INT', 0, self.bases, start.inputs[0], 300).node.name = BASE
        start.inputs[1].default_value = 0
        matched = _node(b, "FunctionNodeCompare", 300, 150, data_type='INT', operation='GREATER_THAN')
        b.lookup(b.k, 'INT', 0, self.counts, _sock(matched.inputs, "A_INT"), 150).node.name = COUNT
        valid, flat = matched.outputs[0], start.outputs[0]
        for j in range(len(self.factors)):
            if j not in interpolated:
                level, ok = self.level(j)
                flat = _int(b, 'MULTIPLY_ADD', 450, -300 * j, level, self.steps[j], flat)
                valid = _and(b, valid, ok, 450, -300 * j - 100)
        corners = [(flat, None)]  # (flat row, weight; None = 1), s in itertools.product order
        for j in interpolated:
            b0, t = self.axis(j)
            b1 = b0 + 1 if isinstance(b0, int) else _int(b, 'ADD', 200, -300 * j, b0, 1)
            rest = _math(b, 'SUBTRACT', 200, -300 * j - 100, 1.0, t)
            grown = []
            for flat, weight in corners:
                for level, share in ((b0, rest), (b1, t)):
                    row = _int(b, 'MULTIPLY_ADD', 600, -120 * len(grown), level, self.steps[j], flat)
                    grown.append((row, share if weight is None else _math(b, 'MULTIPLY', 750, -120 * len(grown),
                                                                         weight, share)))
            corners = grown
        geometry, value = b.table_geometry(), rules_core.value_column(mode)
        attributes = {name: b.attribute(name, 'FLOAT', 900, 400 - 100 * i) for i, name in enumerate((value, "c", "ok"))}
        total = numerator = confidence_sum = None
        for i, (row, weight) in enumerate(corners):
            y = -300 * i
            v, c, ok = (b.sample(geometry, name, 'FLOAT', row, 1000, y - 80 * r, attributes[name])
                        for r, name in enumerate((value, "c", "ok")))
            share = ok if weight is None else _math(b, 'MULTIPLY', 1300, y, weight, ok)
            numerator = _math(b, 'MULTIPLY', 1450, y, share, v) if numerator is None else \
                _math(b, 'MULTIPLY_ADD', 1450, y, share, v, numerator)
            confidence_sum = _math(b, 'MULTIPLY', 1450, y - 80, share, c) if confidence_sum is None else \
                _math(b, 'MULTIPLY_ADD', 1450, y - 80, share, c, confidence_sum)
            total = share if total is None else _math(b, 'ADD', 1450, y - 160, total, share)
        valid = _and(b, valid, _cmp(b, 'FLOAT', 'GREATER_THAN', total, rules_core.BLEND_EPSILON, 1600, 200), 1750, 200)
        result = _switch(b, 'FLOAT', valid, 0.0, _math(b, 'DIVIDE', 1750, 0, numerator, total), 1900, 0)
        b.link(result, go.inputs["Probability" if mode == 'YES_NO' else "Amount"])
        b.link(_switch(b, 'FLOAT', valid, 0.0, _math(b, 'DIVIDE', 1750, -150, confidence_sum, total), 1900, -150),
               go.inputs["Confidence"])
        if mode == 'YES_NO':
            b.link(_cmp(b, 'FLOAT', 'GREATER_EQUAL', result, 0.5, 2050, 100), go.inputs["Yes"])