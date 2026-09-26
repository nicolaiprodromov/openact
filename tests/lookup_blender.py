"""The inner group built from a node's entries, with no server: the match chain, entry-level values,
values-only updates and rebuilds, and single-value sampling from a table object. With --bench, also the
main-thread budgets of §8.4, as BENCH lines: inner.build() and update_values() for the Ask and Pick
shapes and the contract spike's stand-ins for the demo and worst shapes of the nodes still to come; and,
recorded without a gate, the planning cost of a 4096-cell context (§5.3).

Run:  blender -b --factory-startup --python tests/lookup_blender.py [-- --bench]
      OPENACT_BENCH_SHAPE=demo|worst limits --bench to one set of shapes (default: both).
"""

import contextlib
import itertools
import os
import sys
import time

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import check, finish, sock, store  # noqa: E402
from openact import engine, evaluator, inner, nodes  # noqa: E402  (bpy must be imported first)
from openact import prefs as P  # noqa: E402
from openact import store as S  # noqa: E402  (store() is the harness's attribute writer)
from openact.inner import SocketDef  # noqa: E402

P._Defaults.auto_start = False  # nothing here may start a model server
nodes.register()
ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
MATERIALISED_MAX = 16  # the cap of §8.1: every worst shape has this many entries
DIALS_MAX_SETTINGS = 32  # §7.5
BUDGET_MS = {"build demo": 100, "build demo interpolated": 250, "build worst": 1000,  # §8.4
             "update_values": 30, "update_values all": 150}

# ------------------------------------------------------------------------------ the scene

tree = bpy.data.node_groups.new("Lookup", 'GeometryNodeTree')
for name, socket_type in (("Geometry", 'NodeSocketGeometry'), ("Question", 'NodeSocketString'),
                          ("Prompt", 'NodeSocketString'), ("Items", 'NodeSocketString'),
                          ("Item", 'NodeSocketString')):
    tree.interface.new_socket(name, in_out='INPUT', socket_type=socket_type)
tree.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
N, L = tree.nodes, tree.links
g_in, g_out = N.new("NodeGroupInput"), N.new("NodeGroupOutput")
ask = N.new(engine.ASK_ID)
L.new(g_in.outputs["Question"], ask.inputs["Question"])
pick = N.new(engine.PICK_ID)
L.new(g_in.outputs["Prompt"], pick.inputs["Prompt"])
L.new(g_in.outputs["Items"], pick.inputs["Items"])
item_ok = N.new("FunctionNodeCompare")  # Pick's Item against the item this object expects
item_ok.data_type = 'STRING'
L.new(pick.outputs["Item"], sock(item_ok.inputs, "A_STR"))
L.new(g_in.outputs["Item"], sock(item_ok.inputs, "B_STR"))
geometry = g_in.outputs["Geometry"]
for name, output, data_type in (("answer", ask.outputs["Answer"], 'FLOAT'), ("yes", ask.outputs["Yes"], 'BOOLEAN'),
                                ("conf", ask.outputs["Confidence"], 'FLOAT'), ("index", pick.outputs["Index"], 'INT'),
                                ("pconf", pick.outputs["Confidence"], 'FLOAT'), ("item_ok", item_ok.outputs[0], 'BOOLEAN')):
    geometry = store(tree, geometry, output, name, data_type)
L.new(geometry, g_out.inputs[0])
ids = {i.name: i.identifier for i in tree.interface.items_tree if getattr(i, "in_out", "") == 'INPUT'}
WORDS = {  # object -> the words its modifier gives the tree
    "A": {"Question": "Is it night?", "Prompt": "something to sit on", "Items": "table, chair", "Item": "chair"},
    "B": {"Question": "Is it cold?", "Prompt": "something to read", "Items": "book, lamp", "Item": "book"},
    "C": {"Question": "Is it loud?", "Prompt": "something to eat", "Items": "apple, pear", "Item": ""},
}
objects = {}
for key, words in WORDS.items():
    mesh = bpy.data.meshes.new(key)
    mesh.from_pydata([(0, 0, 0)], [], [])
    obj = objects[key] = bpy.data.objects.new(key, mesh)
    bpy.context.scene.collection.objects.link(obj)
    mod = obj.modifiers.new("Lookup", 'NODES')
    mod.node_group = tree
    for name, value in words.items():
        mod[ids[name]] = value


def ask_entry(key, answer, confidence):
    return S.Entry(words_key=key, gn=(WORDS[key]["Question"], ""), kind="ask", answer_type='YES_NO',
                   single={"Answer": answer, "Yes": answer >= 0.5, "Confidence": confidence})


def pick_entry(key, index, confidence):
    words = WORDS[key]
    names = [name.strip() for name in words["Items"].split(",")]
    return S.Entry(words_key=key, gn=(words["Prompt"], words["Items"]), kind="pick", answer_type='ONE_OF',
                   single={"Index": index, "Item": names[index], "Confidence": confidence})


def read(key):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    evaluated = objects[key].evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        return {name: round(a.data[0].value, 6) if a.data_type == 'FLOAT' else a.data[0].value
                for name, a in mesh.attributes.items() if name in ("answer", "yes", "conf", "index", "pconf", "item_ok")}
    finally:
        evaluated.to_mesh_clear()


def pointers(node):
    return [n.as_pointer() for n in node.node_tree.nodes]


# --------------------------------------------------------------------------- the checks

def phase_lookup():
    asks = S.NodeStore([ask_entry("A", 0.875, 0.75), ask_entry("B", 0.25, 0.5)])
    picks = S.NodeStore([pick_entry("A", 1, 0.5), pick_entry("B", 0, 0.25)])
    inner.build(ask, asks)
    inner.build(pick, picks)
    got = {key: read(key) for key in WORDS}
    check("build: each context reads its own entry",
          got["A"] == {"answer": 0.875, "yes": True, "conf": 0.75, "index": 1, "pconf": 0.5, "item_ok": True}
          and got["B"] == {"answer": 0.25, "yes": False, "conf": 0.5, "index": 0, "pconf": 0.25, "item_ok": True}, got)
    check("build: unmatched words read the fallbacks",
          got["C"] == {"answer": 0.0, "yes": False, "conf": 0.0, "index": -1, "pconf": 0.0, "item_ok": True}, got["C"])
    placed = {n.bl_idname: tuple(n.location) for n in ask.node_tree.nodes}
    check("build: locations are written in one batch",
          placed["NodeGroupInput"] == (-900, 0) and placed["NodeGroupOutput"] == (800, 0)
          and sorted(n.location.y for n in ask.node_tree.nodes if n.bl_idname == "GeometryNodeIndexSwitch")
          == [-200, 0, 200], placed)

    before = pointers(ask), pointers(pick)
    asks.entries[1] = ask_entry("B", 0.625, 0.25)
    picks.entries[0] = pick_entry("A", 0, 0.75)
    updated = inner.update_values(ask, asks), inner.update_values(pick, picks)
    got = {key: read(key) for key in WORDS}
    check("update_values: new values of the same entries, no rebuild",
          updated == (True, True) and (pointers(ask), pointers(pick)) == before, updated)
    check("update_values: each context reads its entry's new values",
          got["A"] == {"answer": 0.875, "yes": True, "conf": 0.75, "index": 0, "pconf": 0.75, "item_ok": False}
          and got["B"] == {"answer": 0.625, "yes": True, "conf": 0.25, "index": 0, "pconf": 0.25, "item_ok": True},
          got)

    grown = S.NodeStore(asks.entries + [ask_entry("C", 1.0, 1.0)])
    refused = inner.update_values(ask, grown)
    check("update_values: other entries need build(), and nothing is written",
          refused is False and pointers(ask) == before[0] and read("C")["answer"] == 0.0, (refused, read("C")))
    inner.build(ask, grown)
    got = {key: read(key) for key in WORDS}
    check("build: the new entry reaches its context", got["C"]["answer"] == 1.0 and got["C"]["yes"] is True, got["C"])
    inner.build(ask, grown)
    inner.build(pick, picks)
    check("build: a rebuild from the same entries leaves every output unchanged",
          {key: read(key) for key in WORDS} == got, got)


def phase_pass():
    """The pass keeps entries through store.py: schema 1 is converted on read, schema 2 is written, and the
    live set it passes to save() keeps a live entry that eviction would otherwise remove."""
    live = bpy.data.node_groups.new("Pass", 'GeometryNodeTree')
    live.interface.new_socket("Question", in_out='INPUT', socket_type='NodeSocketString')
    node_ = live.nodes.new(engine.ASK_ID)
    live.links.new(live.nodes.new("NodeGroupInput").outputs["Question"], node_.inputs["Question"])
    for name in ("Old words", "New words"):
        user = bpy.data.objects.new(name, bpy.data.meshes.new(name))
        user.modifiers.new("Pass", 'NODES').node_group = live
        user.modifiers["Pass"][live.interface.items_tree["Question"].identifier] = name
    requests = {ctx.obj.name: engine.AskSpec.request(node_, ctx) for ctx in evaluator.find_uses()[live.as_pointer()]}
    old = S.Entry(words_key=requests["Old words"].key, gn=requests["Old words"].gn, t=1.0, kind="ask",
                  answer_type='YES_NO', single={"Answer": 0.25, "Yes": False, "Confidence": 0.5})
    warm = [S.Entry(words_key=f"warm {i}", gn=(f"Warm {i}?", ""), t=10.0 + i, kind="ask", answer_type='YES_NO',
                    single={"Answer": 0.5, "Yes": True, "Confidence": 0.0}) for i in range(S.MAX_ENTRIES + 2)]
    S.save(node_, S.NodeStore([old] + warm))
    engine.MEMORY[requests["New words"].key] = engine.AskSpec.parse(requests["New words"],
                                                                     {"answer": {"type": "noul", "noul": 0.875}})
    try:
        engine.run_pass()
    finally:
        engine.MEMORY.clear()
    saved = S.load(node_).entries
    check("pass: a new answer is stored, and eviction keeps the oldest entry because its words are live",
          node_.entries.startswith('{"schema":2,') and [e.gn[0] for e in saved[:1] + saved[-1:]] == ["Old words", "New words"]
          and [e.gn[0] for e in saved[1:-1]] == [f"Warm {i}?" for i in range(2, S.MAX_ENTRIES + 2)],
          [e.gn[0] for e in saved])
    check("pass: the inner group shows the stored answers", engine.STATUS[node_.uid][0] == 'CHECKMARK'
          and inner.update_values(node_, S.load(node_)), engine.STATUS[node_.uid])
    node_.entries = ('[{"key": "k", "gn": ["Old words", ""], "t": 1.0,'
                     ' "a": {"mode": "YES_NO", "value": 1.0, "confidence": 1.0}}]')  # v0.1.0, schema 1
    engine.MEMORY[requests["New words"].key] = engine.AskSpec.parse(requests["New words"],
                                                                     {"answer": {"type": "noul", "noul": 0.875}})
    try:
        engine.run_pass()
    finally:
        engine.MEMORY.clear()
    saved = [(e.gn, e.legacy) for e in S.load(node_).entries]
    check("pass: a schema 1 list is converted on read, its entry kept, and the store written as schema 2",
          node_.entries.startswith('{"schema":2,')
          and saved == [(("Old words", "", "", "", ""), True), (("New words", ""), False)], saved)
    engine.reask(node_)
    check("Ask Again: the node's answers are forgotten, saved as schema 2 and the group rebuilt",
          node_.entries == '{"schema":2,"entries":[]}' and inner.update_values(node_, S.NodeStore()), node_.entries)


# ------------------------------------------------------------------ stand-ins for later nodes
# The contract spike (§8.4): graphs with the node types and counts that the nodes still to come will
# build for their demos and at their documented limits, built through the real inner.build() on an
# Ask node that stands in for them.

TABLE = bpy.data.objects.new(".OpenAct Table bench", bpy.data.meshes.new(".OpenAct Table bench"))
MATERIALS = [bpy.data.materials.new(f"Bench {i}") for i in range(MATERIALISED_MAX)]
INTEGER, REAL = "FunctionNodeIntegerMath", "ShaderNodeMath"


def enabled(sockets, name):
    return next(s for s in sockets if s.name == name and s.enabled)


def chain(b, gi, words, entries):
    """The joined-key match chain of §8.2: one Join Strings, then a Compare and a Switch per entry."""
    join = b.new("GeometryNodeStringJoin", -900, 600)
    join.inputs["Delimiter"].default_value = "\x1f"
    for word in reversed(words):  # Join Strings joins the most recently linked first
        b.link(gi.outputs[word], join.inputs["Strings"])
    k = None
    for i, entry in enumerate(entries, start=1):
        compare = b.new("FunctionNodeCompare", -700, 600 - 30 * i)
        compare.data_type, compare.operation = 'STRING', 'EQUAL'
        b.link(join.outputs[0], sock(compare.inputs, "A_STR"))
        sock(compare.inputs, "B_STR").default_value = "\x1f".join(entry.gn)
        switch = b.new("GeometryNodeSwitch", -500, 600 - 30 * i)
        switch.input_type = 'INT'
        b.link(compare.outputs[0], sock(switch.inputs, "Switch"))
        if k is None:
            sock(switch.inputs, "False").default_value = 0
        else:
            b.link(k, sock(switch.inputs, "False"))
        sock(switch.inputs, "True").default_value = i
        k = switch.outputs[0]
    return k


def node(b, idname, x, y, **settings):
    made = b.new(idname, x, y)
    for name, value in settings.items():
        setattr(made, name, value)
    return made


def math(b, idname, operation, x, y, *values):
    """A math node whose inputs are linked from sockets or set to numbers, in order."""
    made = node(b, idname, x, y, operation=operation)
    for socket, value in zip(made.inputs, values):
        if isinstance(value, (int, float)):
            socket.default_value = value
        else:
            b.link(value, socket)
    return made.outputs[0]


def sample(b, table, column, data_type, index, x, y):
    """One lookup on the table object (§8.3): Named Attribute and Sample Index."""
    attribute = node(b, "GeometryNodeInputNamedAttribute", x, y, data_type=data_type)
    attribute.inputs["Name"].default_value = column
    sampled = node(b, "GeometryNodeSampleIndex", x + 150, y, data_type=data_type)
    b.link(table, sampled.inputs["Geometry"])
    b.link(enabled(attribute.outputs, "Attribute"), enabled(sampled.inputs, "Value"))
    b.link(index, sampled.inputs["Index"])
    return enabled(sampled.outputs, "Value")


def guarded(b, valid, value, data_type, fallback, x, y):
    """valid ? value : fallback."""
    switch = node(b, "GeometryNodeSwitch", x, y, input_type=data_type)
    b.link(valid, sock(switch.inputs, "Switch"))
    sock(switch.inputs, "False").default_value = fallback
    b.link(value, sock(switch.inputs, "True"))
    return switch.outputs[0]


def flat_index(b, k, entries, rows, index, x, y):
    """flat = base + index and valid = 0 <= index < count, with base and count on k (§8.3)."""
    flat = node(b, INTEGER, x + 400, y, operation='ADD')
    b.lookup(k, 'INT', 0, [rows * e for e in range(len(entries))], flat.inputs[0], y)
    b.link(index, flat.inputs[1])
    below = node(b, "FunctionNodeCompare", x + 400, y - 150, data_type='INT', operation='LESS_THAN')
    b.lookup(k, 'INT', 0, [rows] * len(entries), sock(below.inputs, "B_INT"), y - 150)
    b.link(index, sock(below.inputs, "A_INT"))
    above = node(b, "FunctionNodeCompare", x + 400, y - 300, data_type='INT', operation='GREATER_EQUAL')
    b.link(index, sock(above.inputs, "A_INT"))
    valid = math(b, "FunctionNodeBooleanMath", 'AND', x + 600, y - 200, above.outputs[0], below.outputs[0])
    return flat.outputs[0], valid


def table_geometry(b, table=TABLE):
    info = b.new("GeometryNodeObjectInfo", -300, -900)
    info.inputs["Object"].default_value = table
    return info.outputs["Geometry"]


def words_iface(words):
    return tuple(SocketDef('INPUT', w, 'NodeSocketString', kind="word") for w in words)


class RulesStandIn:
    """Rules in Amount mode (§9.2, §9.6): three Amount factors of `bands` bands, interpolated over 8
    corners with 3 lookups each (value, c, ok), weights and the renormalised blend. Custom edges cost a
    Compare and an Add per edge; equal-width bands a fixed graph."""

    label, words = "Rules", ("Question", "Context", "Factors", "Outcomes", "Overrides", "Key")
    iface = words_iface(words) + tuple(SocketDef('INPUT', f"f{j}", 'NodeSocketFloat', kind="field") for j in range(3)) + (
        SocketDef('OUTPUT', "Amount", 'NodeSocketFloat'), SocketDef('OUTPUT', "Confidence", 'NodeSocketFloat'))

    def __init__(self, bands, custom_edges):
        self.bands, self.custom_edges = bands, custom_edges

    def build(self, b, gi, go, entries):
        n = self.bands
        k = chain(b, gi, self.words, entries)
        table = table_geometry(b)
        lower, weight = [], []  # per factor: (b0, b0 + 1) and (1 - t, t)
        for j in range(3):
            y = -1200 * j
            x = node(b, "ShaderNodeClamp", -900, y)
            b.link(gi.outputs[f"f{j}"], x.inputs["Value"])
            if self.custom_edges:  # b0 = the number of edges at or below x; t from the centres around it
                total = 0
                for e in range(n - 1):
                    above = node(b, "FunctionNodeCompare", -700, y - 40 * e, operation='GREATER_EQUAL')
                    b.link(x.outputs[0], sock(above.inputs, "A"))
                    sock(above.inputs, "B").default_value = (e + 1) / n
                    total = math(b, INTEGER, 'ADD', -500, y - 40 * e, above.outputs[0], total)
                upper = math(b, INTEGER, 'ADD', -300, y, total, 1)
                offset = node(b, REAL, -100, y, operation='SUBTRACT')
                span = node(b, REAL, -100, y - 100, operation='SUBTRACT')
                centres = [(i + 0.5) / n for i in range(n)]
                low = b.lookup(total, 'FLOAT', 0.0, centres, offset.inputs[1], y)
                b.link(low, span.inputs[1])
                b.lookup(upper, 'FLOAT', 0.0, centres, span.inputs[0], y - 100)
                b.link(x.outputs[0], offset.inputs[0])
                t = math(b, REAL, 'DIVIDE', 100, y, offset.outputs[0], span.outputs[0])
                t.node.use_clamp = True
            else:  # u = clamp(x n - 1/2, 0, n - 1), b0 = min(floor(u), n - 2), t = u - b0
                u = node(b, "ShaderNodeClamp", -500, y)
                b.link(math(b, REAL, 'MULTIPLY_ADD', -700, y, x.outputs[0], n, -0.5), u.inputs["Value"])
                u.inputs["Max"].default_value = n - 1
                floor = node(b, "FunctionNodeFloatToInt", -300, y, rounding_mode='FLOOR')
                b.link(u.outputs[0], floor.inputs[0])
                total = math(b, INTEGER, 'MINIMUM', -100, y, floor.outputs[0], n - 2)
                upper = math(b, INTEGER, 'ADD', 100, y, total, 1)
                t = math(b, REAL, 'SUBTRACT', 100, y - 100, u.outputs[0], total)
            lower.append((total, upper))
            weight.append((math(b, REAL, 'SUBTRACT', 300, y, 1.0, t), t))
        sums = {"w": [], "v": [], "c": []}
        base = None
        for c, s in enumerate(itertools.product((0, 1), repeat=3)):  # the 8 corners
            y = -400 * c
            cell = math(b, INTEGER, 'MULTIPLY_ADD', 500, y, lower[1][s[1]], n, lower[2][s[2]])
            cell = math(b, INTEGER, 'MULTIPLY_ADD', 700, y, lower[0][s[0]], n * n, cell)
            flat = node(b, INTEGER, 900, y, operation='ADD')
            if base is None:
                base = b.lookup(k, 'INT', 0, [n ** 3 * e for e in range(len(entries))], flat.inputs[0], y)
            else:
                b.link(base, flat.inputs[0])
            b.link(cell, flat.inputs[1])
            v, conf, ok = (sample(b, table, column, 'FLOAT', flat.outputs[0], 1100, y - 120 * i)
                           for i, column in enumerate(("oa_amount", "oa_c", "oa_ok")))
            w = math(b, REAL, 'MULTIPLY', 1500, y, weight[0][s[0]], weight[1][s[1]])
            w = math(b, REAL, 'MULTIPLY', 1700, y, w, weight[2][s[2]])
            w_ok = math(b, REAL, 'MULTIPLY', 1900, y, w, ok)
            sums["w"].append(w_ok)
            sums["v"].append(math(b, REAL, 'MULTIPLY', 2100, y, w_ok, v))
            sums["c"].append(math(b, REAL, 'MULTIPLY', 2100, y - 100, w_ok, conf))
        totals = {}
        for name, terms in sums.items():
            total = terms[0]
            for term in terms[1:]:
                total = math(b, REAL, 'ADD', 2300, 0, total, term)
            totals[name] = total
        valid = node(b, "FunctionNodeCompare", 2500, 200, operation='GREATER_THAN')
        b.link(totals["w"], sock(valid.inputs, "A"))
        sock(valid.inputs, "B").default_value = 1e-6
        for name, output in (("v", "Amount"), ("c", "Confidence")):
            value = math(b, REAL, 'DIVIDE', 2500, 0, totals[name], totals["w"])
            b.link(guarded(b, valid.outputs[0], value, 'FLOAT', 0.0, 2700, 0), go.inputs[output])


class RulesOneOfStandIn:
    """Rules in One-of mode (§9.2, §9.3, §9.7): three equal-width Amount factors of `bands` bands, each
    dithered with its own Random Value, the row-major cell index, and lookups of outcome, c and the
    payload columns."""

    label, words = "Rules", ("Question", "Context", "Factors", "Outcomes", "Overrides", "Key")
    iface = words_iface(words) + tuple(SocketDef('INPUT', f"f{j}", 'NodeSocketFloat', kind="field") for j in range(3)) + (
        SocketDef('INPUT', "Dither", 'NodeSocketFloat'), SocketDef('INPUT', "Seed", 'NodeSocketInt'),
        SocketDef('OUTPUT', "Outcome", 'NodeSocketInt'), SocketDef('OUTPUT', "Confidence", 'NodeSocketFloat'),
        SocketDef('OUTPUT', "instance", 'NodeSocketInt'), SocketDef('OUTPUT', "scale", 'NodeSocketFloat'))

    def __init__(self, bands):
        self.bands = bands

    def build(self, b, gi, go, entries):
        n = self.bands
        k = chain(b, gi, self.words, entries)
        levels = []
        for j in range(3):  # x' = clamp(x + (r - 0.5) Dither / n), b = min(floor(x' n), n - 1)
            y = -600 * j
            x = node(b, "ShaderNodeClamp", -900, y)
            b.link(gi.outputs[f"f{j}"], x.inputs["Value"])
            random = node(b, "FunctionNodeRandomValue", -900, y - 200, data_type='FLOAT')
            b.link(math(b, INTEGER, 'ADD', -1100, y - 200, gi.outputs["Seed"], 7919 * j), random.inputs["Seed"])
            jitter = math(b, REAL, 'SUBTRACT', -700, y - 200, enabled(random.outputs, "Value"), 0.5)
            jitter = math(b, REAL, 'MULTIPLY', -500, y - 200, jitter, gi.outputs["Dither"])
            moved = math(b, REAL, 'MULTIPLY_ADD', -300, y, jitter, 1 / n, x.outputs[0])
            moved.node.use_clamp = True
            band = node(b, "FunctionNodeFloatToInt", -100, y, rounding_mode='FLOOR')
            b.link(math(b, REAL, 'MULTIPLY', -200, y, moved, n), band.inputs[0])
            levels.append(math(b, INTEGER, 'MINIMUM', 100, y, band.outputs[0], n - 1))
        cell = math(b, INTEGER, 'MULTIPLY_ADD', 300, 0, levels[1], n, levels[2])
        cell = math(b, INTEGER, 'MULTIPLY_ADD', 500, 0, levels[0], n * n, cell)
        flat = node(b, INTEGER, 700, 0, operation='ADD')
        b.lookup(k, 'INT', 0, [n ** 3 * e for e in range(len(entries))], flat.inputs[0], 0)
        b.link(cell, flat.inputs[1])
        table = table_geometry(b)
        for i, (column, data_type, output) in enumerate((("oa_outcome", 'INT', "Outcome"), ("oa_c", 'FLOAT', "Confidence"),
                                                          ("oa_pl_0", 'INT', "instance"), ("oa_pl_1", 'FLOAT', "scale"))):
            b.link(sample(b, table, column, data_type, flat.outputs[0], 900, -150 * i), go.inputs[output])


class DialsStandIn:
    """Dials with `settings` Amount settings (§8.4, §8.6): each one's value, min and max sampled from the
    one-row-per-entry table at k - 1 behind a Switch on k > 0, then Map Range; and the Confidence."""

    label, words = "Dials", ("Prompt", "Settings", "Overrides", "Key")

    def __init__(self, settings):
        self.settings = settings
        self.iface = words_iface(self.words) + tuple(
            SocketDef('OUTPUT', f"s{s}", 'NodeSocketFloat') for s in range(settings)) + (
            SocketDef('OUTPUT', "Confidence", 'NodeSocketFloat'),)

    def build(self, b, gi, go, entries):
        k = chain(b, gi, self.words, entries)
        table = table_geometry(b)
        row = math(b, INTEGER, 'SUBTRACT', -300, 0, k, 1)
        live = node(b, "FunctionNodeCompare", -300, -100, data_type='INT', operation='GREATER_THAN')
        b.link(k, sock(live.inputs, "A_INT"))
        for s in range(self.settings):
            y = -250 * s
            value, low, high = (sample(b, table, f"oa_{column}_{s}", 'FLOAT', row, 0, y - 80 * i)
                                for i, column in enumerate(("v", "min", "max")))
            mapped = node(b, "ShaderNodeMapRange", 400, y, clamp=False)
            b.link(value, mapped.inputs["Value"])
            b.link(low, mapped.inputs["To Min"])
            b.link(high, mapped.inputs["To Max"])
            b.link(guarded(b, live.outputs[0], mapped.outputs["Result"], 'FLOAT', 0.0, 600, y), go.inputs[f"s{s}"])
        confidence = sample(b, table, "oa_conf", 'FLOAT', row, 0, 300)
        b.link(guarded(b, live.outputs[0], confidence, 'FLOAT', 0.0, 600, 300), go.inputs["Confidence"])


class PickPayloadStandIn:
    """Pick at its limits (§8.4): Index, Item and Confidence, 16 payload columns and Material on k, each
    an Index Switch of 17 items, plus the per-element prob lookup (Probability, Selection) on ID."""

    label, words = "Pick", ("Prompt", "Context", "Items", "Overrides", "Key")
    singles = [("Index", 'INT', -1), ("Item", 'STRING', ""), ("Confidence", 'FLOAT', 0.0)] + [
        (f"p{c}", 'FLOAT', 0.0) for c in range(16)] + [("Material", 'MATERIAL', None)]
    iface = words_iface(words) + (
        SocketDef('INPUT', "ID", 'NodeSocketInt', kind="field", default_input='INDEX', hide_value=True),) + tuple(
        SocketDef('OUTPUT', name, {'INT': 'NodeSocketInt', 'STRING': 'NodeSocketString', 'FLOAT': 'NodeSocketFloat',
                                   'MATERIAL': 'NodeSocketMaterial'}[data_type]) for name, data_type, _ in singles) + (
        SocketDef('OUTPUT', "Selection", 'NodeSocketBool'), SocketDef('OUTPUT', "Probability", 'NodeSocketFloat'))

    def build(self, b, gi, go, entries):
        k = chain(b, gi, self.words, entries)
        picked = [b.lookup(k, data_type, fallback, [e.single[name] for e in entries], go.inputs[name], 300 - 150 * i)
                  for i, (name, data_type, fallback) in enumerate(self.singles)][0]
        flat, valid = flat_index(b, k, entries, 255, gi.outputs["ID"], -300, -3000)
        probability = sample(b, table_geometry(b), "oa_prob", 'FLOAT', flat, 300, -3000)
        b.link(guarded(b, valid, probability, 'FLOAT', 0.0, 600, -3000), go.inputs["Probability"])
        same = node(b, "FunctionNodeCompare", 300, -3300, data_type='INT', operation='EQUAL')
        b.link(gi.outputs["ID"], sock(same.inputs, "A_INT"))
        b.link(picked, sock(same.inputs, "B_INT"))
        answered = node(b, "FunctionNodeCompare", 300, -3450, data_type='INT', operation='GREATER_EQUAL')
        b.link(picked, sock(answered.inputs, "A_INT"))
        b.link(math(b, "FunctionNodeBooleanMath", 'AND', 600, -3300, same.outputs[0], answered.outputs[0]),
               go.inputs["Selection"])


class ClassifyStandIn:
    """Classify with `payload` payload columns (§8.3, §8.4): label, c and each payload column, a
    per-element lookup on ID behind the valid switch. Match has the same graph with only p and c."""

    label, words = "Classify", ("Question", "Context", "Items", "Labels", "Overrides", "Key")

    def __init__(self, payload):
        self.columns = [("oa_label", 'INT', -1), ("oa_c", 'FLOAT', 0.0)] + [
            (f"oa_pl_{c}", 'FLOAT', 0.0) for c in range(payload)]
        self.iface = words_iface(self.words) + (
            SocketDef('INPUT', "ID", 'NodeSocketInt', kind="field", default_input='INDEX', hide_value=True),) + tuple(
            SocketDef('OUTPUT', column, 'NodeSocketInt' if data_type == 'INT' else 'NodeSocketFloat')
            for column, data_type, _ in self.columns)

    def build(self, b, gi, go, entries):
        k = chain(b, gi, self.words, entries)
        flat, valid = flat_index(b, k, entries, 255, gi.outputs["ID"], -300, 0)
        table = table_geometry(b)
        for i, (column, data_type, fallback) in enumerate(self.columns):
            value = sample(b, table, column, data_type, flat, 300, -150 * i)
            b.link(guarded(b, valid, value, data_type, fallback, 700, -150 * i), go.inputs[column])


class SampledCount:
    """One Dials-like output: an INT sampled from a table object at k - 1, a single-value index."""

    label, words = "Dials", ("Prompt",)
    iface = words_iface(words) + (SocketDef('OUTPUT', "Count", 'NodeSocketInt'),)

    def __init__(self, table):
        self.table = table

    def build(self, b, gi, go, entries):
        row = math(b, INTEGER, 'SUBTRACT', -300, 0, chain(b, gi, self.words, entries), 1)
        b.link(sample(b, table_geometry(b, self.table), "oa_count", 'INT', row, 0, 0), go.inputs["Count"])


@contextlib.contextmanager
def standing_in(node_, spec):
    """Build a stand-in shape with the real inner functions, on an Ask node whose spec it replaces."""
    saved = inner.SPECS[node_.bl_idname]
    inner.SPECS[node_.bl_idname] = spec
    try:
        inner.sync_interface(node_, spec.iface)
        yield
    finally:
        inner.SPECS[node_.bl_idname] = saved


def phase_single_value_sample():
    """§8.4 regression guard: Sample Index at a single-value index gives a single value, so a Dials output
    can drive a single-value socket such as Grid's Vertices X."""
    mesh = bpy.data.meshes.new(".OpenAct Table sampled")
    mesh.from_pydata([(0, 0, 0)] * 3, [], [])
    mesh.attributes.new("oa_count", 'INT', 'POINT').data.foreach_set("value", [5, 7, 9])
    table = bpy.data.objects.new(mesh.name, mesh)
    grid_tree = bpy.data.node_groups.new("Sampled", 'GeometryNodeTree')
    grid_tree.interface.new_socket("Prompt", in_out='INPUT', socket_type='NodeSocketString')
    grid_tree.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    gi, go = grid_tree.nodes.new("NodeGroupInput"), grid_tree.nodes.new("NodeGroupOutput")
    stand = grid_tree.nodes.new(engine.ASK_ID)
    grid = grid_tree.nodes.new("GeometryNodeMeshGrid")
    grid.inputs["Vertices Y"].default_value = 2
    obj = bpy.data.objects.new("Sampled", bpy.data.meshes.new("Sampled"))
    bpy.context.scene.collection.objects.link(obj)
    mod = obj.modifiers.new("Sampled", 'NODES')
    mod.node_group = grid_tree
    mod[grid_tree.interface.items_tree["Prompt"].identifier] = "second"
    with standing_in(stand, SampledCount(table)):
        grid_tree.links.new(gi.outputs["Prompt"], stand.inputs["Prompt"])
        link = grid_tree.links.new(stand.outputs["Count"], grid.inputs["Vertices X"])
        grid_tree.links.new(grid.outputs["Mesh"], go.inputs["Geometry"])
        inner.build(stand, S.NodeStore([S.Entry(gn=("first",)), S.Entry(gn=("second",))]))
        depsgraph = bpy.context.evaluated_depsgraph_get()
        depsgraph.update()
        evaluated = obj.evaluated_get(depsgraph)
        count = len(evaluated.to_mesh().vertices)
        evaluated.to_mesh_clear()
        shape = getattr(stand.outputs["Count"], "inferred_structure_type", 'SINGLE')  # 5.0+ only
        check("Sample Index at a single-value index drives Grid's Vertices X (§8.4)",
              link.is_valid and count == 7 * 2 and shape == 'SINGLE', (link.is_valid, count, shape))


# ------------------------------------------------------------------------------ the bench

def timed(fn, runs=5):
    """The fastest of the runs, in milliseconds: the one least slowed by other work on the machine."""
    times = []
    for _ in range(runs):
        started = time.perf_counter()
        fn()
        times.append((time.perf_counter() - started) * 1000)
    return min(times)


def bench_line(path, elements, ms, budget=None):
    print(f"BENCH {path} {elements} {ms:.1f}", flush=True)
    if budget is not None:
        check(f"bench: {path} within {budget} ms", ms <= budget, f"{ms:.1f} ms, {elements} elements")


def bench_shape(label, node_, store_a, store_b, shape, budget=None):
    """build() from store_a, then update_values() alternating with store_b, whose values all differ: once
    with every entry changed (every entry-level socket written), once with one entry changed."""
    ms = timed(lambda: inner.build(node_, store_a))
    bench_line(f"build/{label}/{shape}", len(node_.node_tree.nodes), ms, BUDGET_MS[budget or "build " + shape])
    if store_b is None:
        return
    switches = [n for n in node_.node_tree.nodes if n.bl_idname == "GeometryNodeIndexSwitch"
                and any(link.to_node.bl_idname == "NodeGroupOutput" for link in n.outputs[0].links)]
    one = S.NodeStore(list(store_a.entries))
    one.entries[-1] = store_b.entries[-1]
    for variant, other, writes, limit in (
            ("all-entries", store_b, len(switches) * len(store_a.entries), BUDGET_MS["update_values all"]),
            ("one-entry", one, len(switches), BUDGET_MS["update_values"])):
        stores = itertools.cycle((other, store_a))
        ms = timed(lambda: check_update(node_, next(stores)), runs=9)
        bench_line(f"update_values/{label}/{shape}/{variant}", writes, ms, limit)


def check_update(node_, node_store):
    if not inner.update_values(node_, node_store):
        raise AssertionError("update_values() refused the entries build() laid out")


def ask_store(count, offset=0.0):
    return S.NodeStore([S.Entry(words_key=str(i), gn=(f"Question {i}?", f"Context {i}"), single={
        "Answer": (i + offset) / 40, "Yes": i % 2 == bool(offset), "Confidence": (i + 2 * offset) / 50})
        for i in range(count)])


def pick_store(count, offset=0):
    return S.NodeStore([S.Entry(words_key=str(i), gn=(f"Prompt {i}", "table, chair, lamp, rug, bookshelf"), single={
        "Index": (i + offset) % 5, "Item": f"item {(i + offset) % 5}", "Confidence": (i + offset) / 40})
        for i in range(count)])


def pick_payload_store(count, offset=0):
    entries = []
    for i in range(count):
        single = {"Index": (i + offset) % 5, "Item": f"item {(i + offset) % 5}", "Confidence": (i + offset) / 40,
                  "Material": MATERIALS[(i + offset) % len(MATERIALS)]}
        single.update({f"p{c}": (i * 16 + c + offset) / 1000 for c in range(16)})
        entries.append(S.Entry(words_key=str(i), gn=(f"Prompt {i}", "", "//props.csv", "", ""), single=single))
    return S.NodeStore(entries)


def words_store(spec, count):
    """Entries whose words fill every word of the spec; stand-ins need no values."""
    return S.NodeStore([S.Entry(words_key=str(e), gn=tuple(f"{word} of context {e}" for word in spec.words))
                        for e in range(count)])


def plan_4096():
    """What planning a changed 4096-cell Rules context costs in Python (§5.3): each cell's instructions,
    its row words key and its cell key, then the entry's words key, hashed as engine._key() does."""
    factors = [[(f"band {i} of {j}", f"the value lies in the {i}th of sixteen bands of factor {j}") for i in range(16)]
               for j in ("temperature", "moisture", "altitude")]
    options = [(f"outcome {o}", f"when the {o}th outcome's condition holds for this situation") for o in range(7)]
    question = "How much snow covers the ground in this situation?"
    state = "No additional context.\nEach question describes one situation with these facts:\n" + "\n".join(
        f"- {j}: " + " | ".join(name for name, _when in levels) for j, levels in enumerate(factors))
    criteria = {str(o): f"{name}: {when}" for o, (name, when) in enumerate(options)}
    wkeys = []
    for levels in itertools.product(*factors):
        facts = "\n".join(f"- {j}: {name} ({when})" for j, (name, when) in enumerate(levels))
        instructions = f"Situation:\n{facts}\n{question}"
        wkeys.append(engine._key("row", "rules", 'AMOUNT', ["", question], levels, options))
        engine._key("cell", "score", state, instructions, criteria, True)
    return engine._key("words", ["", question], [], 'AMOUNT', "shape", wkeys, "")


def phase_bench(shapes):
    stand_ins = []
    if "demo" in shapes:  # the demo scenes of §18.4, one context each (Ask and Pick: the prototype's two)
        bench_shape("ask", ask, ask_store(2), ask_store(2, 1.0), "demo")
        bench_shape("pick", pick, pick_store(2), pick_store(2, 1), "demo")
        stand_ins += [("classify-materials", ClassifyStandIn(2), 1, "demo", "build demo"),
                      ("rules-biome", RulesOneOfStandIn(3), 1, "demo", "build demo"),
                      ("rules-snow", RulesStandIn(3, custom_edges=False), 1, "demo", "build demo interpolated"),
                      ("dials-weather", DialsStandIn(5), 1, "demo", "build demo")]
    if "worst" in shapes:  # every shape at the MATERIALISED_MAX cap
        bench_shape("ask", ask, ask_store(MATERIALISED_MAX), ask_store(MATERIALISED_MAX, 1.0), "worst")
        bench_shape("pick", pick, pick_store(MATERIALISED_MAX), pick_store(MATERIALISED_MAX, 1), "worst")
        stand_ins += [("pick-payload", PickPayloadStandIn(), None, "worst", "build worst"),
                      ("classify", ClassifyStandIn(16), MATERIALISED_MAX, "worst", "build worst"),
                      ("dials", DialsStandIn(DIALS_MAX_SETTINGS), MATERIALISED_MAX, "worst", "build worst"),
                      ("rules", RulesStandIn(16, custom_edges=True), MATERIALISED_MAX, "worst", "build worst")]
    for label, spec, entries, shape, budget in stand_ins:
        stand = N.new(engine.ASK_ID)
        with standing_in(stand, spec):
            if entries is None:
                bench_shape(label, stand, pick_payload_store(MATERIALISED_MAX), pick_payload_store(MATERIALISED_MAX, 1),
                            shape, budget)
            else:
                bench_shape(label, stand, words_store(spec, entries), None, shape, budget)
    bench_line("plan/rules", 4096, timed(plan_4096, runs=3))  # recorded only: the pass's TIMING covers planning


phase_lookup()
phase_pass()
phase_single_value_sample()
if "--bench" in ARGS:
    chosen = os.environ.get("OPENACT_BENCH_SHAPE", "")
    phase_bench({chosen} if chosen in ("demo", "worst") else {"demo", "worst"})
finish()
