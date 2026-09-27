"""Dials (§6.6) in a headless Blender.

Without arguments no model is needed: fake answers stand in for it. The weather settings plan into one request, and
their outputs exist before the first answer, showing their fallbacks; the one-row-per-entry table object gives each
output its setting's answer, Amount mapped into min-max; a min/max edit is a values-only update with 0 requests; two
contexts with different ranges each get their own values; contexts with other setting names give the status error
naming both, and every context keeps its last good table and every output its link, whichever context changed, also
after an object rename, while the same settings in another order agree; meanwhile the sidebar's table and the export
of a context whose settings moved give each setting its own answer; an output drives Grid's Vertices X through a
valid link; renaming a setting warns (in the CSV or inline) until its new socket is linked, or it comes back linked
(File > Revert), and typed back it warns until linked again; after Ask Again, or while an edited question is asked
again, the outputs and their links stay; a CSV whose rows only move keeps each output on its own setting; a new
setting gets its output before its answer; a node reading a setting as a word waits while Dials shows a hint; while a
context has no job (a missing Settings CSV, an inline error), every output keeps its link, another context's new
prompt is still asked and shown, and a rename in it waits.

With -- --model PORT GPU the real model answers the weather dials on that port and GPU: the storm prompt gives
Rain = true, a new prompt is answered in about 0.6 s, and switching back is instant.

Run:  blender -b --factory-startup --python tests/dials_blender.py [-- --model 18769 0]
"""

import contextlib
import os
import shutil
import sys
import tempfile
import time

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import ROOT, check, drive, finish, store  # noqa: E402
from openact import engine, inner, nodes, ops, pipeline, server, tables, ui  # noqa: E402  (bpy must be imported first)
from openact import prefs as P  # noqa: E402
from openact import store as S  # noqa: E402  (store() is the harness's attribute writer)
from openact.specs import dials  # noqa: E402

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
DIALS = dials.DialsSpec.idname
WEATHER = os.path.join(ROOT, "tests", "fixtures", "weather_dials.csv")
STORM, CALM = "A stormy night in a fishing village", "A calm summer morning"
TMP = tempfile.mkdtemp(prefix="openact_dials_")
OUTPUTS = [("Rain", 'BOOLEAN'), ("Wind speed", 'VALUE'), ("Time of day", 'INT'), ("Street lights", 'BOOLEAN'),
           ("Fog", 'VALUE'), ("Confidence", 'VALUE')]
STORED = {'BOOLEAN': 'BOOLEAN', 'VALUE': 'FLOAT', 'INT': 'INT'}  # socket type -> Store Named Attribute type
FAKE = {  # instructions -> what the fake model answers, parsed as pipeline.parse() gives it
    "Is it raining?": {"p": 0.9},
    "Are the street lights on?": {"p": 0.2},
    "How windy is it?": {"probabilities": [0.0, 0.0, 0.2, 0.6, 0.2], "amount": 0.75},
    "How foggy is it?": {"probabilities": [0.0, 0.25, 0.5, 0.25, 0.0], "amount": 0.5},
    "What time of day is it?": {"probabilities": [0.05, 0.05, 0.1, 0.8]},
}
# Confidence = the lowest: Fog's 1 - sigma / sigma_u = 1 - sqrt(0.5 / 2) = 0.5
EXPECTED = {"Rain": True, "Wind speed": 22.5, "Time of day": 3, "Street lights": False, "Fog": 0.5, "Confidence": 0.5}
FALLBACKS = {"Rain": False, "Wind speed": 0.0, "Time of day": -1, "Street lights": False, "Fog": 0.0, "Confidence": 0.0}
ASK = "GeometryNodeOpenActAsk"

nodes.register()
ops.register()


# ------------------------------------------------------------------------------------------ helpers

def weather(name, edit=None):
    """A copy of the weather CSV in the temporary folder, with (old, new) text replaced."""
    with open(WEATHER, encoding="utf-8") as handle:
        text = handle.read()
    if edit:
        assert edit[0] in text, edit
        text = text.replace(*edit)
    path = os.path.join(TMP, name)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return path


def rewrite(path, lines):
    """Write the lines over the file with a later mtime: rows that only move keep its size, so its stamp could stay."""
    later = os.stat(path).st_mtime_ns + 10**9
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
    os.utime(path, ns=(later, later))


def scene(name, users):
    """A tree whose Prompt and Settings come from the modifier, used by one object per (name, prompt, settings)."""
    tree = bpy.data.node_groups.new(name, 'GeometryNodeTree')
    for socket, kind in (("Geometry", 'NodeSocketGeometry'), ("Prompt", 'NodeSocketString'),
                         ("Settings", 'NodeSocketString')):
        tree.interface.new_socket(socket, in_out='INPUT', socket_type=kind)
    tree.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    gi, go = tree.nodes.new("NodeGroupInput"), tree.nodes.new("NodeGroupOutput")
    node = tree.nodes.new(DIALS)
    tree.links.new(gi.outputs["Prompt"], node.inputs["Prompt"])
    tree.links.new(gi.outputs["Settings"], node.inputs["Settings"])
    tree.links.new(gi.outputs["Geometry"], go.inputs["Geometry"])
    objects = {key: user(tree, key, prompt, settings) for key, prompt, settings in users}
    return tree, node, objects


def user(tree, key, prompt, settings):
    mesh = bpy.data.meshes.new(key)
    mesh.from_pydata([(0, 0, 0)], [], [])
    obj = bpy.data.objects.new(key, mesh)
    bpy.context.scene.collection.objects.link(obj)
    mod = obj.modifiers.new("Dials", 'NODES')
    mod.node_group = tree
    words(obj, Prompt=prompt, Settings=settings)
    return obj


def words(obj, **values):
    mod = obj.modifiers["Dials"]
    for name, value in values.items():
        mod[mod.node_group.interface.items_tree[name].identifier] = value
    obj.update_tag()


def wire(tree, node):
    """Store every Dials output on the points, named after it, for read()."""
    gi = next(n for n in tree.nodes if n.bl_idname == "NodeGroupInput")
    go = next(n for n in tree.nodes if n.bl_idname == "NodeGroupOutput")
    geometry = gi.outputs["Geometry"]
    for x, socket in enumerate(node.outputs):
        geometry = store(tree, geometry, socket, socket.name, STORED[socket.type], 300 + 200 * x)
    tree.links.new(geometry, go.inputs["Geometry"])


def read(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        return {name: round(a.data[0].value, 4) if a.data_type == 'FLOAT' else a.data[0].value
                for name, a in mesh.attributes.items() if a.domain == 'POINT' and name[:1] != "." and name != "position"}
    finally:
        evaluated.to_mesh_clear()


@contextlib.contextmanager
def asking(calls):
    """Each Job the passes inside would send goes into calls as (job, missing cells); nothing reaches a server."""
    submit = engine._submit
    engine._submit = lambda job, missing: calls.append((job, list(missing)))
    try:
        yield
    finally:
        engine._submit = submit


@contextlib.contextmanager
def rebuilds(rebuilt):
    """Lists the nodes the passes inside rebuild: they call build() only when update_values() refuses (§8.8)."""
    build = inner.build

    def recorded(node_, store_, live=None, default=None):
        rebuilt.append(node_.name)
        build(node_, store_, live, default)

    inner.build = recorded
    try:
        yield
    finally:
        inner.build = build


def answer(cells):
    for cell in cells:
        pipeline.remember(cell.key, dict(FAKE[cell.question["instructions"]], model="fake"))


def settle(times=3):
    for _ in range(times):
        engine.run_pass()


def pointers(node):
    return [n.as_pointer() for n in node.node_tree.nodes]


def linked(node):
    return [(s.name, s.is_linked) for s in node.outputs]


# --------------------------------------------------------------------------------------- no model

def phase_nomodel():
    P._Defaults.auto_start = False  # nothing here may start a model server
    csv_a = weather("a.csv")
    tree, node, users = scene("Dials", [("A", STORM, csv_a)])
    calls = []
    with asking(calls):
        engine.run_pass()
    mine = [(job, missing) for job, missing in calls if job.uid == node.uid]
    kinds = sorted(cell.question["type"] for _job, missing in mine for cell in missing)
    check("plan: 5 settings travel in 1 request, sharing the Prompt as state",
          len(mine) == 1 and kinds == ["choice", "noul", "noul", "score", "score"]
          and {cell.state for cell in mine[0][1]} == {STORM}, (len(mine), kinds))
    got = [(s.name, s.type) for s in node.outputs]
    check("plan: before the first answer the node has one output per setting, then Confidence (§6.0)",
          got == OUTPUTS, got)
    wire(tree, node)
    got = read(users["A"])
    check("plan: until then each output shows its fallback: False, 0.0 or -1", got == FALLBACKS, got)
    answer(mine[0][1] if mine else [])
    calls.clear()
    with asking(calls):
        settle()
    got = [(s.name, s.type) for s in node.outputs]
    check("answers: one output per setting, by name and type, in list order, then Confidence", got == OUTPUTS, got)
    got = read(users["A"])
    check("answers: each output shows its setting's answer, Amount mapped into min-max; Confidence is the lowest",
          got == EXPECTED, got)
    table = tables.find(node)
    names = sorted(a.name for a in table.data.attributes if a.name.startswith(tables.ATTR)) if table else []
    check("table: one row per entry, each setting's value, min/max for Amount and conf, in no collection",
          table is not None and len(table.data.vertices) == 1 and not table.users_collection and names == sorted(
              ["oa_conf", "oa_v0", "oa_v1", "oa_min1", "oa_max1", "oa_v2", "oa_v3", "oa_v4", "oa_min4", "oa_max4"]),
          names)
    status = engine.STATUS[node.uid]
    check("status: the node sums up its settings", (status.icon, status.text) == (
        'CHECKMARK', "5 settings · lowest confidence 0.50"), status)
    weather("a.csv", ("\nFog,", "\n\nFog,"))  # a blank line between rows: a warning, the settings still usable
    with asking(calls):
        settle()
    icon, text = engine.STATUS[node.uid].icon, engine.STATUS[node.uid].text
    check("status: a Settings CSV warning shows after the summary, and nothing is asked",
          icon == 'ERROR' and text.startswith("5 settings · lowest confidence 0.50 · 1 warning: a.csv: Line 6 is "
                                              "blank") and not calls, (icon, text, len(calls)))

    # a min/max edit: the rows carry over (their wkeys hold no min/max), the shape changes, the graph doesn't
    before, rebuilt = pointers(node), []
    weather("a.csv", (",0,30", ",10,50"))
    calls.clear()
    with asking(calls), rebuilds(rebuilt):
        settle()
    ranges = [(s["name"], s["min"], s["max"]) for s in S.load(node).entries[0].shape["settings"]][1]
    check("min/max edit: 0 requests, a values-only update (no rebuild), stored as the entry's shape.settings",
          not calls and not rebuilt and pointers(node) == before and ranges == ("Wind speed", 10.0, 50.0),
          (len(calls), rebuilt, ranges))
    got = read(users["A"])
    check("min/max edit: the new range shows (10 + 0.75 x 40)", got["Wind speed"] == 40.0, got)

    # a second context with its own range for the same settings
    csv_b = weather("b.csv", (",0,30", ",0,100"))
    users["B"] = user(tree, "B", STORM, csv_b)
    calls.clear()
    with asking(calls):
        settle()
    a, b = read(users["A"]), read(users["B"])
    check("two contexts: the same questions ask nothing more", not calls, len(calls))
    check("two contexts: each maps with its own min/max from its own table row",
          a == dict(EXPECTED, **{"Wind speed": 40.0}) and b == dict(EXPECTED, **{"Wind speed": 75.0})
          and len(tables.find(node).data.vertices) == 2, (a, b))

    # contexts must agree on the setting names and types (§6.0). Otherwise the node shows the error and answers no
    # context, so each keeps its last good table and the outputs their links, whichever context is first
    good, before, links = (read(users["A"]), read(users["B"])), pointers(node), linked(node)
    tail = "; setting names and types must agree"

    def settled(step, error=None):
        rebuilt = []
        calls.clear()
        with asking(calls), rebuilds(rebuilt):
            settle()
        got = (read(users["A"]), read(users["B"]))
        record, warning = engine.STATUS[node.uid], dials.WARNINGS.get(node.uid)
        status = record.icon, record.text
        shown = ('CANCEL', error) if error else ('CHECKMARK', "5 settings · lowest confidence 0.50   +1 more")
        check(f"shape agreement ({step}): {'the status error names both contexts' if error else 'a checkmark'}, and "
              "each context keeps its last good table, every output its link and the graph, with 0 requests",
              status == shown and not warning and got == good and linked(node) == links and not rebuilt
              and pointers(node) == before and not calls, (status, warning, got, linked(node), rebuilt, len(calls)))

    weather("b.csv", ("Wind speed,amount", "Gust,amount"))
    settled("the second context renames a setting",
            "Dials: 'A > Dials' and 'B > Dials' use different settings ('Wind speed' (amount) vs 'Gust' (amount))" + tail)
    shown = [entry and entry.shape["settings"][1]["name"] for _label, _job, entry in engine.answers_for(node)]
    check("shape agreement: the sidebar shows B's last good table too", shown == ["Wind speed", "Wind speed"], shown)
    users["A"].name = "Harbour"  # contexts go by object name: B, which differs, is now the first
    settled("an object rename makes the other context first",
            "Dials: 'B > Dials' and 'Harbour > Dials' use different settings ('Gust' (amount) vs 'Wind speed' (amount))"
            + tail)
    users["A"].name = "A"
    weather("b.csv", (",0,30", ",0,100"))
    settled("B agrees again")
    weather("a.csv", ("Wind speed,amount", "Gust,amount"))
    settled("the first context renames a setting",
            "Dials: 'A > Dials' and 'B > Dials' use different settings ('Gust' (amount) vs 'Wind speed' (amount))" + tail)
    weather("a.csv", (",0,30", ",10,50"))
    settled("A agrees again")
    with open(csv_b, encoding="utf-8") as handle:
        header, rain, wind, time_, street, fog = handle.read().split("\n")[:6]
    rewrite(csv_b, [header, rain, fog, wind, time_, street])  # each entry fills the outputs by name
    settled("the same settings in another order agree")
    rewrite(csv_b, [header, wind, rain, time_, street, fog.replace("Fog,", "Mist,")])  # moved, and one renamed
    settled("B moves Wind speed first and renames Fog",
            "Dials: 'A > Dials' and 'B > Dials' use different settings ('Fog' (amount) vs 'Mist' (amount))" + tail)
    label, job, entry = engine.answers_for(node)[1]
    table = [row[:2] for row in ui.table_rows(engine.SPECS[DIALS], job, entry)]
    target = os.path.join(TMP, "b_overrides.csv")
    ops.write_overrides(node, 1, target)
    node.inputs["Overrides"].default_value = ""  # the export filled the empty socket: the words stay as they were
    with open(target, encoding="utf-8") as handle:
        written = [line.split(",")[:2] for line in handle.read().splitlines()[1:]]
    check("shape agreement: meanwhile B's table and its export give each setting its own answer, from B's last good "
          "table by wkey (Mist asks what Fog asked)",
          label == "B > Dials" and table == [("Wind speed", "75"), ("Rain", "yes"), ("Time of day", "night"),
                                             ("Street lights", "no"), ("Mist", "0.5")]
          and written[-1] == ["Mist", "0.5"], (label, table, written))
    rewrite(csv_b, [header, rain, fog, wind, time_, street])
    settled("B agrees again, in its own order")

    # with two entries too: the pass replaces the entry in place, so the graph's entry order holds
    before, rebuilt = pointers(node), []
    weather("a.csv", (",0,30", ",20,60"))
    calls.clear()
    with asking(calls), rebuilds(rebuilt):
        settle()
    got = read(users["A"])
    check("min/max edit, two entries: still values only (20 + 0.75 x 40)",
          not calls and not rebuilt and pointers(node) == before and got["Wind speed"] == 50.0,
          (len(calls), rebuilt, got))

    phase_grid(csv_a)

    # renaming a linked setting replaces its socket, drops the link and warns
    bpy.data.objects.remove(users.pop("B"))
    weather("a.csv", ("Fog,amount", "Mist,amount"))
    with asking(calls):
        settle()
    names = [s.name for s in node.outputs]
    check("rename: the node warns once, and the renamed setting gets a new socket",
          dials.WARNINGS.get(node.uid) == "Setting 'Fog' renamed or retyped: reconnect its output"
          and "Mist" in names and "Fog" not in names and not calls, (dials.WARNINGS.get(node.uid), names, len(calls)))
    status = engine.STATUS[node.uid]
    check("rename: the warning is in the node's status record",
          status.state == "warning" and "Setting 'Fog' renamed or retyped: reconnect its output"
          in [problem.message for problem in status.problems], (status.text, status.problems))
    phase_inline()


def phase_inline():
    """An inline rename changes the words: the new entry is appended while the old one stays, just left. The outputs
    follow the settings, and the warning about the dropped link ends once the new socket is linked (§6.6)."""
    inline = "Rain (yes/no): Is it raining?; {} (amount): How foggy is it?"
    tree, node, users = scene("DialsInline", [("I", STORM, inline.format("Fog"))])
    calls = []
    with asking(calls):
        settle()  # the cells are the CSV's: cached
    wire(tree, node)
    words(users["I"], Settings=inline.format("Mist"))
    with asking(calls):
        settle()
    names = [s.name for s in node.outputs]
    check("inline rename: the outputs follow the settings, and the node warns",
          names == ["Rain", "Mist", "Confidence"] and len(S.load(node).entries) == 2 and not calls
          and dials.WARNINGS.get(node.uid) == "Setting 'Fog' renamed or retyped: reconnect its output",
          (names, dials.WARNINGS.get(node.uid), len(calls)))
    orphan = next(n for n in tree.nodes if n.bl_idname == "GeometryNodeStoreNamedAttribute"
                  and n.inputs["Name"].default_value == "Fog")
    tree.links.new(node.outputs["Mist"], next(s for s in orphan.inputs if s.name == "Value" and s.enabled))
    with asking(calls):
        settle()
    check("inline rename: once the socket that replaced Fog is linked, the warning ends",
          dials.WARNINGS.get(node.uid) is None, dials.WARNINGS.get(node.uid))
    words(users["I"], Settings=inline.format("Fog"))  # back to the stored entry: the store itself doesn't change
    with asking(calls):
        settle()
    names = [s.name for s in node.outputs]
    check("inline rename back: the stored entry's outputs return, with 0 requests, and the linked Mist warns",
          names == ["Rain", "Fog", "Confidence"] and not calls
          and dials.WARNINGS.get(node.uid) == "Setting 'Mist' renamed or retyped: reconnect its output",
          (names, len(calls), dials.WARNINGS.get(node.uid)))
    words(users["I"], Settings=inline.format("Mist"))  # typed back: a new socket, without the link undo brings back
    with asking(calls):
        settle()
    check("inline rename typed back: Mist returns without its link, so the node still warns",
          not node.outputs["Mist"].is_linked and not calls
          and dials.WARNINGS.get(node.uid) == "Setting 'Mist' renamed or retyped: reconnect its output",
          (linked(node), len(calls), dials.WARNINGS.get(node.uid)))
    tree.links.new(node.outputs["Mist"], next(s for s in orphan.inputs if s.name == "Value" and s.enabled))
    with asking(calls):
        settle()
    check("inline rename typed back: once Mist is linked again, the warning ends", dials.WARNINGS.get(node.uid) is None,
          dials.WARNINGS.get(node.uid))
    phase_live_outputs()


def settle_answered(calls):
    """Passes that answer every Amount cell they ask for with the same middling answer, until nothing is asked."""
    for _ in range(3):
        calls.clear()
        with asking(calls):
            engine.run_pass()
        for _job, missing in calls:
            for cell in missing:
                pipeline.remember(cell.key, {"probabilities": [0.0, 0.25, 0.5, 0.25, 0.0], "amount": 0.5,
                                             "model": "fake"})
    calls.clear()
    with asking(calls):
        settle(2)


def phase_live_outputs():
    """The outputs follow the settings the node's contexts plan (§6.0): after Ask Again, or while a question edited in
    the Settings CSV (the same words) is asked again, an older stored entry with other settings never takes them over,
    so the links to them survive."""
    tree, node, users = scene("DialsReask", [("R", CALM, "Fog (amount): How foggy is the bay?")])
    calls = []
    settle_answered(calls)
    words(users["R"], Settings="Mist (amount): How misty is the bay?")
    settle_answered(calls)
    wire(tree, node)
    linked = [s.name for s in node.outputs if s.is_linked]
    engine.reask(node)
    right_after = [s.name for s in node.outputs if s.is_linked]
    settle_answered(calls)
    got = [s.name for s in node.outputs if s.is_linked]
    check("Ask Again: another stored entry's settings never take the outputs over, so their links survive",
          linked == right_after == got == ["Mist", "Confidence"] and len(S.load(node).entries) == 2
          and not dials.WARNINGS.get(node.uid), (linked, right_after, got, dials.WARNINGS.get(node.uid)))

    def settings(name, question):
        path = os.path.join(TMP, f"{name.lower()}.csv")
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(f"setting,type,question,answers,min,max\n{name},amount,{question},,,\n")
        return path

    fog, mist = settings("Fog", "How foggy is the pier?"), settings("Mist", "How misty is the pier?")
    tree, node, users = scene("DialsEdit", [("E", CALM, fog)])
    settle_answered(calls)
    words(users["E"], Settings=mist)
    settle_answered(calls)
    wire(tree, node)
    settings("Mist", "How misty is the pier right now?")  # the same setting and words, a new question
    with asking(calls):
        settle()
    asked = {cell.question["instructions"] for _job, missing in calls for cell in missing}
    meanwhile = [s.name for s in node.outputs if s.is_linked]
    settle_answered(calls)
    got = [s.name for s in node.outputs if s.is_linked]
    check("Settings CSV edit: while its question is asked again, and after, the outputs and their links stay",
          asked == {"How misty is the pier right now?"} and meanwhile == got == ["Mist", "Confidence"]
          and not dials.WARNINGS.get(node.uid), (asked, meanwhile, got, dials.WARNINGS.get(node.uid)))
    phase_new_setting()


def phase_new_setting():
    """§6.0: a setting added to the Settings gets its output at once, before its first answer, and the other outputs
    keep their links."""
    inline = "Fog (amount): How foggy is the harbour?"
    tree, node, users = scene("DialsAdd", [("N", CALM, inline)])
    calls = []
    settle_answered(calls)
    wire(tree, node)
    words(users["N"], Settings=inline + "; Snow (yes/no): Is it snowing in the harbour?")
    calls.clear()
    with asking(calls):
        settle()
    asked = {cell.question["instructions"] for _job, missing in calls for cell in missing}
    got = [(s.name, s.type, s.is_linked) for s in node.outputs]
    check("new setting: its output comes before its answer, and the other outputs keep their links",
          asked == {"Is it snowing in the harbour?"}
          and got == [("Fog", 'VALUE', True), ("Snow", 'BOOLEAN', False), ("Confidence", 'VALUE', True)], (asked, got))
    phase_reorder()


def phase_reorder():
    """A Settings CSV edit that only moves rows (§5.4): every row carries over by wkey, with 0 requests, and the graph,
    whose output j reads table column j, is laid out again once, so each output keeps its own setting's value."""
    path = weather("reorder.csv")
    tree, node, users = scene("DialsReorder", [("R", STORM, path)])
    calls = []
    with asking(calls):
        settle()  # the weather cells are cached
    wire(tree, node)
    with open(path, encoding="utf-8") as handle:
        header, rain, wind, time_, street, fog = handle.read().split("\n")[:6]
    for label, order in (("Fog moved up", [rain, fog, wind, time_, street]),
                         ("Wind speed and Fog swapped", [rain, fog, time_, street, wind])):
        rewrite(path, [header] + order)
        calls.clear()
        rebuilt = []
        with asking(calls), rebuilds(rebuilt):
            settle()
        names, got = [s.name for s in node.outputs], read(users["R"])
        asked = [job for job, _missing in calls if job.uid == node.uid]  # the new setting's Snow is still unanswered
        check(f"reorder ({label}): each output keeps its own setting's value, with 0 requests and one rebuild",
              got == EXPECTED and names == [row.split(",")[0] for row in order] + ["Confidence"] and not asked
              and len(rebuilt) == 1, (got, names, len(asked), rebuilt))
    phase_chained_hint()


def phase_chained_hint():
    """§6.0: while Dials shows a hint (here: no Prompt), a node that reads a setting as a word waits, as it does before
    the first answer. An Amount read through Value to String gave it an internal error, and a One-of read through an
    Index Switch read item 0, where Geometry Nodes gives -1."""
    tree = bpy.data.node_groups.new("DialsChain", 'GeometryNodeTree')
    tree.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    dial = tree.nodes.new(DIALS)
    dial.inputs["Prompt"].default_value = STORM
    dial.inputs["Settings"].default_value = weather("chain.csv")
    calls = []
    with asking(calls):
        settle()  # the weather cells are cached
    text = tree.nodes.new("FunctionNodeValueToString")
    tree.links.new(dial.outputs["Fog"], text.inputs["Value"])
    amount = tree.nodes.new(ASK)
    tree.links.new(text.outputs["String"], amount.inputs["Question"])
    switch = tree.nodes.new("GeometryNodeIndexSwitch")
    switch.data_type = 'STRING'
    while len(switch.index_switch_items) < 4:
        switch.index_switch_items.new()
    items = [s for s in switch.inputs if s.identifier not in ("Index", "__extend__")]
    for item, question in zip(items, ("Is it dawn?", "Is it day?", "Is it dusk?", "Is it night?")):
        item.default_value = question
    tree.links.new(dial.outputs["Time of day"], switch.inputs["Index"])
    one_of = tree.nodes.new(ASK)
    tree.links.new(switch.outputs[0], one_of.inputs["Question"])
    calls.clear()
    with asking(calls):
        settle()
    planned = {job.uid: job.rows[0].name for job, _missing in calls if job.uid in (amount.uid, one_of.uid)}
    check("chained: while Dials is answered, both readers plan from its values",
          planned.get(one_of.uid) == "Is it night?" and amount.uid in planned, planned)
    dial.inputs["Prompt"].default_value = ""
    calls.clear()
    with asking(calls):
        settle()
    status = engine.STATUS.get(amount.uid)
    check("chained hint: an Amount read through Value to String waits, with no internal error",
          status is not None and status.icon in ('PAUSE', 'SORTTIME') and "Internal error" not in status.text, status)
    asked = [job.rows[0].name for job, _missing in calls if job.uid == one_of.uid]
    check("chained hint: a One-of read through an Index Switch waits too, instead of asking about item 0",
          not asked and engine.STATUS[one_of.uid].icon in ('PAUSE', 'SORTTIME'), (asked, engine.STATUS[one_of.uid]))
    phase_hold()


def phase_hold():
    """§6.0: a context without a job (its Settings CSV missing, an inline error) may still show the outputs as they
    are, so they stay, with their links. Another context's new prompt, which fits them, is still asked and shown;
    a rename, which would change them, waits, and gives the error once that context plans again."""
    csv_a, csv_b = weather("hold_a.csv"), weather("hold_b.csv")
    tree, node, users = scene("DialsHold", [("HA", STORM, csv_a), ("HB", STORM, csv_b)])
    calls = []

    def passes(change, times=3):
        """Make a change and run the passes: (status, what both objects read, the links, the cells this node asks)."""
        change()
        calls.clear()
        with asking(calls):
            settle(times)
        status = engine.STATUS[node.uid]
        return ((status.icon, status.text), (read(users["HA"]), read(users["HB"])), linked(node),
                [cell for job, missing in calls if job.uid == node.uid for cell in missing])

    passes(lambda: None)  # the weather cells are cached
    wire(tree, node)
    good, links = (read(users["HA"]), read(users["HB"])), linked(node)
    os.remove(csv_a)
    waited = passes(lambda: words(users["HB"], Prompt=CALM), 1)
    for cell in waited[3]:  # the calm morning is dry
        pipeline.remember(cell.key, dict(FAKE[cell.question["instructions"]], model="fake",
                                         **({"p": 0.1} if cell.question["instructions"] == "Is it raining?" else {})))
    got = passes(lambda: None)
    check("hold: while a Settings CSV is missing, another context's new prompt is asked, and shown once answered; "
          "every output keeps its link, and the object whose CSV is missing its last table",
          waited[0] == got[0] == ('CANCEL', f"Can't find {os.path.normpath(csv_a)}") and len(waited[3]) == 5
          and not got[3] and got[1] == (good[0], dict(good[1], Rain=False)) and got[2] == links, (waited, got))
    got = passes(lambda: (weather("hold_a.csv"), words(users["HB"], Prompt=STORM)))
    check("hold: the CSV back, a checkmark, and both objects show their tables again, 0 requests",
          got == (('CHECKMARK', "5 settings · lowest confidence 0.50   +1 more"), good, links, []), got)
    renamed = ("Dials: 'HA > Dials' and 'HB > Dials' use different settings ('Wind speed' (amount) vs 'Gust' (amount));"
               " setting names and types must agree")
    got = (passes(lambda: weather("hold_b.csv", ("Wind speed,amount", "Gust,amount"))),
           passes(lambda: words(users["HA"], Settings="Rain (yes/no): Is it raining?; Rain (yes/no): Is it wet?")),
           passes(lambda: words(users["HA"], Settings=csv_a)))
    check("hold: while the context whose settings the outputs show has an inline error, another context's rename "
          "waits: every output keeps its link and the renamed context its last table, with 0 requests; fixed, the "
          "error is back",
          got[0] == got[2] == (('CANCEL', renamed), good, links, []) and got[1][0] == (
              'CANCEL', "Settings: Entry 2: 'Rain' appears twice") and got[1][1][1] == good[1] and got[1][2:] == (
              links, []) and not dials.WARNINGS.get(node.uid), got)
    got = passes(lambda: weather("hold_b.csv"))
    check("hold: the rename undone, a checkmark", got[0][0] == 'CHECKMARK' and got[1:] == (good, links, []), got)
    phase_revert()


def phase_revert():
    """§6.6: File > Revert to the file saved before a rename brings the setting back with its link, as undo does, and
    the rename warning ends. It opens a file, so it runs last."""
    inline = "Rain (yes/no): Is it raining?; {} (amount): How foggy is it?"
    tree, node, users = scene("DialsRevert", [("V", STORM, inline.format("Fog"))])
    calls = []
    with asking(calls):
        settle()  # the cells are the CSV's: cached
    wire(tree, node)
    path = os.path.join(TMP, "revert.blend")
    bpy.ops.wm.save_as_mainfile(filepath=path)
    words(users["V"], Settings=inline.format("Mist"))
    with asking(calls):
        settle()
    uid, warned = node.uid, dials.WARNINGS.get(node.uid)
    bpy.ops.wm.open_mainfile(filepath=path)
    engine.on_load_post()
    calls.clear()
    with asking(calls):
        settle()
    node = next(n for n in bpy.data.node_groups["DialsRevert"].nodes if n.bl_idname == DIALS)
    fog = node.outputs.get("Fog")
    check("revert: Fog is back with its link, and its rename warning ends, with 0 requests",
          warned == "Setting 'Fog' renamed or retyped: reconnect its output" and node.uid == uid
          and fog is not None and fog.is_linked and dials.WARNINGS.get(uid) is None
          and not [job for job, _missing in calls if job.uid == uid],
          (warned, node.uid == uid, fog is not None and fog.is_linked, dials.WARNINGS.get(uid)))


def phase_grid(csv):
    """§8.4: Sample Index at a single-value index gives a single value, so a Dials output drives Grid's Vertices X."""
    tree = bpy.data.node_groups.new("DialsGrid", 'GeometryNodeTree')
    tree.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    node = tree.nodes.new(DIALS)
    node.inputs["Prompt"].default_value = STORM
    node.inputs["Settings"].default_value = csv
    grid = tree.nodes.new("GeometryNodeMeshGrid")
    grid.inputs["Vertices Y"].default_value = 2
    tree.links.new(grid.outputs["Mesh"], tree.nodes.new("NodeGroupOutput").inputs["Geometry"])
    obj = bpy.data.objects.new("Grid", bpy.data.meshes.new("Grid"))
    bpy.context.scene.collection.objects.link(obj)
    obj.modifiers.new("Grid", 'NODES').node_group = tree
    calls = []
    with asking(calls):
        settle()
    link = tree.links.new(node.outputs["Time of day"], grid.inputs["Vertices X"])
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    evaluated = obj.evaluated_get(depsgraph)
    count = len(evaluated.to_mesh().vertices)
    evaluated.to_mesh_clear()
    shape = getattr(node.outputs["Time of day"], "inferred_structure_type", 'SINGLE')  # 5.0+ only
    check("grid: a Dials output feeds Grid's Vertices X through a valid link (3 x 2 vertices)",
          link.is_valid and count == 6 and shape == 'SINGLE' and not calls, (link.is_valid, count, shape, len(calls)))


# ------------------------------------------------------------------------------------------ model

def phase_model(port, gpu):
    P._Defaults.port, P._Defaults.gpu, P._Defaults.auto_start = port, gpu, True
    engine.register()
    check("runtime detected", P.runtime_installed(P.runtime_root()), P.runtime_root())
    tree, node, users = scene("Weather", [("Village", STORM, WEATHER)])
    village = users["Village"]
    started = time.time()

    def shown(prompt):
        return getattr(engine.STATUS.get(node.uid), "icon", "") == 'CHECKMARK' and not pipeline.PENDING and any(
            entry.gn[0] == prompt for entry in S.load(node).entries) and len(node.outputs) == len(OUTPUTS)

    engine.schedule(immediate=True)
    finished = drive(engine, lambda: shown(STORM), timeout=300)
    print(f"TIMING dials first answers {time.time() - started:.1f}s (model start included)", flush=True)
    check("model: the weather dials are answered", finished, engine.STATUS.get(node.uid))
    if not finished:
        return
    wire(tree, node)
    storm = read(village)
    print("storm:", storm, flush=True)
    check("model: the storm prompt gives Rain = true", storm.get("Rain") is True, storm)

    words(village, Prompt=CALM)
    engine.schedule(immediate=True)
    t0 = time.time()
    finished = drive(engine, lambda: shown(CALM), timeout=60)
    elapsed = time.time() - t0
    calm = read(village)
    print("calm:", calm, flush=True)
    print(f"TIMING dials new prompt {elapsed:.2f}s (budget 0.6 s)", flush=True)
    if finished and elapsed > 0.6:
        print(f"WARN dials new prompt took {elapsed:.2f}s, above 0.6 s", flush=True)
    check("model: a new prompt is answered in about 0.6 s (fails above twice that)", finished and elapsed <= 1.2,
          f"{elapsed:.2f}s")

    words(village, Prompt=STORM)
    engine.schedule(immediate=True)
    t0 = time.time()
    drive(engine, lambda: read(village) == storm, timeout=10)
    check("model: switching back is instant", read(village) == storm, f"{time.time() - t0:.2f}s")
    server.stop()
    check("server stopped", not server.is_ready())


if "--model" in ARGS:
    at = ARGS.index("--model")
    phase_model(int(ARGS[at + 1]), int(ARGS[at + 2]))
else:
    phase_nomodel()
shutil.rmtree(TMP, ignore_errors=True)
finish()
