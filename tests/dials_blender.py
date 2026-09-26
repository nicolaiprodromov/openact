"""Dials (§6.6) in a headless Blender.

Without arguments no model is needed: fake answers stand in for it. The weather settings plan into one request; the
one-row-per-entry table object gives each output its setting's answer, Amount mapped into min-max; a min/max edit is
a values-only update with 0 requests; two contexts with different ranges each get their own values; different setting
names give the shape-agreement error; an output drives Grid's Vertices X through a valid link; renaming a setting warns.

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
from openact import engine, inner, nodes, ops, pipeline, server, tables  # noqa: E402  (bpy must be imported first)
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
def values_first(rebuilt):
    """The pass wiring of §8.8: update_values() first, build() only when it refuses (rebuilt nodes are listed)."""
    build = inner.build

    def wired(node_, store_, live=None, default=None):
        if not inner.update_values(node_, store_, live, default):
            rebuilt.append(node_.name)
            build(node_, store_, live, default)

    inner.build = wired
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
    check("plan: before the first answer the node has only Confidence",
          [s.name for s in node.outputs] == ["Confidence"], [s.name for s in node.outputs])
    answer(mine[0][1] if mine else [])
    calls.clear()
    with asking(calls):
        settle()
    got = [(s.name, s.type) for s in node.outputs]
    check("answers: one output per setting, by name and type, in list order, then Confidence", got == OUTPUTS, got)
    wire(tree, node)
    got = read(users["A"])
    check("answers: each output shows its setting's answer, Amount mapped into min-max; Confidence is the lowest",
          got == EXPECTED, got)
    table = tables.find(node)
    names = sorted(a.name for a in table.data.attributes if a.name.startswith(tables.ATTR)) if table else []
    check("table: one row per entry, each setting's value, min/max for Amount and conf, in no collection",
          table is not None and len(table.data.vertices) == 1 and not table.users_collection and names == sorted(
              ["oa_conf", "oa_v0", "oa_v1", "oa_min1", "oa_max1", "oa_v2", "oa_v3", "oa_v4", "oa_min4", "oa_max4"]),
          names)
    check("status: the node sums up its settings", engine.STATUS[node.uid] == (
        'CHECKMARK', "5 settings · lowest confidence 0.50"), engine.STATUS.get(node.uid))

    # a min/max edit: the rows carry over (their wkeys hold no min/max), the shape changes, the graph doesn't
    before, rebuilt = pointers(node), []
    weather("a.csv", (",0,30", ",10,50"))
    calls.clear()
    with asking(calls), values_first(rebuilt):
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

    # contexts must agree on the setting names and types
    weather("b.csv", ("Wind speed,amount", "Gust,amount"))
    with asking(calls):
        settle()
    icon, text = engine.STATUS[node.uid]
    check("shape agreement: different setting names give the error", (icon, text) == (
        'ERROR', "Dials: contexts use different settings ('Wind speed' (amount) vs 'Gust' (amount)); setting names "
                 "and types must agree"), (icon, text))
    check("shape agreement: the agreeing context keeps its values", read(users["A"])["Wind speed"] == 40.0)
    weather("b.csv", (",0,30", ",0,100"))
    with asking(calls):
        settle()
    check("shape agreement: agreeing again clears the error, with 0 requests",
          engine.STATUS[node.uid][0] == 'CHECKMARK' and not calls and read(users["B"])["Wind speed"] == 75.0,
          (engine.STATUS[node.uid], len(calls)))

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
        return engine.STATUS.get(node.uid, ("",))[0] == 'CHECKMARK' and not pipeline.PENDING and any(
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
