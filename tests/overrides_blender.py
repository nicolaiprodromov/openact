"""The Overrides CSV (§7.6) in a headless Blender, with no model: fake answers go into MEMORY, and engine._submit is
patched to count what the passes would ask.

- Every node has the Overrides word and its CSV companion, hidden like Key.
- Match through the pass: export fills the empty socket; locks send 0 requests; the orphaned-lock warning; editing a
  lock is a values-only update; an invalid answer is an error and the model's answer stays; the export merge keeps
  locked rows verbatim and user columns; a target file held open reports the PermissionError message.
- Ask, Pick, Classify and Dials (an amount in output units) fully locked: 0 requests.
- Rules: a locked corner takes part in interpolation, against rules_core.blend.

Run:  blender -b --factory-startup --python tests/overrides_blender.py
"""

import csv
import os
import shutil
import sys
import tempfile

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import check, finish, store  # noqa: E402
from openact import csvfmt, engine, evaluator, nodes, ops, pipeline  # noqa: E402
from openact import prefs as P  # noqa: E402
from openact import rules_core as rc  # noqa: E402
from openact import store as S  # noqa: E402
from openact.specs import SPECS  # noqa: E402
from openact.specs import rules as R  # noqa: E402

P._Defaults.auto_start = False  # nothing here may start a model server
nodes.register()
ops.register()
TMP = tempfile.mkdtemp(prefix="openact_overrides_")
POINTS = 4
_stamp = [1_700_000_000 * 10**9]


def write(name, text):
    """A file in TMP, with a new mtime every time: an edit within one clock tick still changes the stamp."""
    path = os.path.join(TMP, name)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    _stamp[0] += 10**9
    os.utime(path, ns=(_stamp[0], _stamp[0]))
    return path


def rows_of(path):
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


class Capture:
    """Patches engine._submit: the missing cells the passes want are recorded, and nothing is queued."""

    def __enter__(self):
        self.cells = []
        self.saved = engine._submit
        engine._submit = lambda job, missing: self.cells.extend((job.uid, cell) for cell in missing)
        return self

    def __exit__(self, *_exc):
        engine._submit = self.saved

    @property
    def asked(self):
        return len(self.cells)

    def of(self, node):
        """The missing cells of this node the passes would ask."""
        return [cell for uid, cell in self.cells if uid == node.uid]


def new_tree(name):
    tree = bpy.data.node_groups.new(name, 'GeometryNodeTree')
    tree.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    tree.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    return tree


def user(name, tree, attributes=None):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([(i, 0, 0) for i in range(POINTS)], [], [])
    for attribute, values in (attributes or {}).items():
        mesh.attributes.new(attribute, 'FLOAT', 'POINT').data.foreach_set("value", values)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.modifiers.new("OpenAct", 'NODES').node_group = tree
    return obj


def read(obj, name):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        found = mesh.attributes.get(name)
        return [round(d.value, 4) for d in found.data] if found else None
    finally:
        evaluated.to_mesh_clear()


def entry_of(node):
    """The stored entry of the node's first context's words (the Overrides path is one of them)."""
    jobs = engine._jobs(node)
    return next((e for e in S.load(node).entries if jobs and e.gn == jobs[0][1].gn), None)


def status(node):
    return engine.STATUS.get(node.uid, ("", ""))


# ------------------------------------------------------------------------------ sockets

def phase_sockets():
    tree = new_tree("Sockets")
    missing = []
    for idname in SPECS:
        node = tree.nodes.new(idname)
        names = [s.name for s in node.inputs]
        ok = "Overrides" in names and "Overrides CSV" in names and names.index("Overrides CSV") == names.index(
            "Overrides") + 1 and node.inputs["Overrides"].hide and node.inputs["Overrides CSV"].hide
        if not ok:
            missing.append((idname, names))
    check("sockets: all six nodes have Overrides and Overrides CSV below it, hidden like Key", not missing
          and len(SPECS) == 6, missing)
    bpy.data.node_groups.remove(tree)


# ------------------------------------------------------------------------------ Match through the pass

def phase_match():
    tree = new_tree("MatchLocks")
    node = tree.nodes.new(SPECS["GeometryNodeOpenActMatch"].idname)
    node.inputs["Question"].default_value = "Would you find it in a kitchen?"
    node.inputs["Items"].default_value = "pan, sofa, kettle"
    gi, go = tree.nodes.new("NodeGroupInput"), tree.nodes.new("NodeGroupOutput")
    tree.links.new(store(tree, gi.outputs["Geometry"], node.outputs["Probability"], "p", 'FLOAT'), go.inputs[0])
    obj = user("MatchUser", tree)
    with Capture() as capture:
        engine.run_pass()
    job = engine._jobs(node)[0][1]
    fake = {"pan": 0.9, "sofa": 0.1, "kettle": 0.8}
    for row in job.rows:
        pipeline.remember(row.cell.key, {"p": fake[row.name], "model": "fake"})
    check("match: without Overrides the 3 items are asked", len(capture.cells) == 3, len(capture.cells))
    engine.run_pass()
    check("match: the model's answers show", read(obj, "p") == [0.9, 0.1, 0.8, 0.0], read(obj, "p"))

    path = ops.write_overrides(node, 0, os.path.join(TMP, "kitchen_overrides.csv"))
    check("export: an empty, unlinked socket takes the chosen file", node.inputs["Overrides"].default_value == path
          and os.path.isfile(path), node.inputs["Overrides"].default_value)
    check("export: one row per item, the model's answers unlocked, with confidence, unsure and source",
          rows_of(path) == [["item", "answer", "lock", "model_answer", "confidence", "unsure", "source"],
                            ["pan", "yes", "0", "yes", "0.80", "0", "model"],
                            ["sofa", "no", "0", "no", "0.80", "0", "model"],
                            ["kettle", "yes", "0", "yes", "0.60", "0", "model"]], rows_of(path))
    with Capture() as capture:
        engine.run_pass()
    check("export: the exported file locks nothing: the table and its words key stay", capture.asked == 0
          and entry_of(node) is not None and entry_of(node).rows.src == "mmm", capture.asked)

    header = "item,answer,lock,model_answer,confidence,unsure,source,note\n"
    write("kitchen_overrides.csv", header + "pan,no,1,yes,0.80,0,model,mine\nsofa,no,0,no,0.80,0,model,check\n"
          "kettle,yes,0,yes,0.60,0,model,\nGhost,yes,1,,,,,old\n")
    with Capture() as capture:
        engine.run_pass()
    entry = entry_of(node)
    check("locks: a locked row sends 0 requests and shows at once, with source lock and confidence 1",
          capture.asked == 0 and entry is not None and entry.rows.src == "lmm" and entry.rows.v["p"] == [0.0, 0.1, 0.8]
          and entry.rows.v["c"][0] == 1.0 and read(obj, "p") == [0.0, 0.1, 0.8, 0.0],
          (capture.asked, entry and (entry.rows.src, entry.rows.v), read(obj, "p")))
    check("locks: a lock whose key matches no row is warned about as orphaned",
          status(node)[0] == 'ERROR' and "orphaned lock 'ghost' matches no row" in status(node)[1]
          and "line 5" in status(node)[1], status(node))

    before = [n.as_pointer() for n in node.node_tree.nodes]
    words_key = entry.words_key
    write("kitchen_overrides.csv", header + "pan,yes,1,yes,0.80,0,model,mine\nsofa,no,0,no,0.80,0,model,check\n"
          "kettle,yes,0,yes,0.60,0,model,\nGhost,yes,1,,,,,old\n")
    with Capture() as capture:
        engine.run_pass()
    entry = entry_of(node)
    check("locks: editing a lock is a values-only update (new words key, same nodes, 0 requests)",
          capture.asked == 0 and entry is not None and entry.words_key != words_key
          and entry.rows.v["p"][0] == 1.0 and [n.as_pointer() for n in node.node_tree.nodes] == before
          and read(obj, "p") == [1.0, 0.1, 0.8, 0.0], (capture.asked, read(obj, "p")))

    write("kitchen_overrides.csv", header + "pan,maybe,1,yes,0.80,0,model,mine\nsofa,no,0,no,0.80,0,model,check\n"
          "kettle,yes,0,yes,0.60,0,model,\n")
    with Capture() as capture:
        engine.run_pass()
    entry = entry_of(node)
    check("locks: an invalid answer is an error on its row, and the model's answer stays (0 requests)",
          capture.asked == 0 and entry is not None and entry.rows.src == "mmm" and entry.rows.v["p"][0] == 0.9
          and status(node)[0] == 'ERROR' and "line 2: 'maybe' is not yes, no or a number from 0 to 1"
          in status(node)[1], (capture.asked, status(node)))

    write("kitchen_overrides.csv", "note,item,lock,answer\nmine,pan,x,Yes\ncheck,sofa,,no\nold,Ghost,1,yes\n")
    engine.run_pass()
    ops.write_overrides(node)
    check("export merge: locked rows verbatim (orphans too), user columns kept, the table's rows in order",
          rows_of(path) == [["item", "answer", "lock", "model_answer", "confidence", "unsure", "source", "note"],
                            ["pan", "Yes", "x", "", "", "", "", "mine"],
                            ["sofa", "no", "0", "no", "0.80", "0", "model", "check"],
                            ["kettle", "yes", "0", "yes", "0.60", "0", "model", ""],
                            ["Ghost", "yes", "1", "", "", "", "", "old"]], rows_of(path))

    handle = open(path, encoding="utf-8")  # Windows refuses to replace a file another handle holds, as Excel does
    replace = ops.os.replace
    if sys.platform != "win32":
        def refuse(*_args):
            raise PermissionError(13, "Permission denied")
        ops.os.replace = refuse
    try:
        ops.write_overrides(node)
        message = "written"
    except ops.OverridesError as error:
        message = str(error)
    finally:
        ops.os.replace = replace
        handle.close()
    check("export: a target file held by another program reports the PermissionError message, no .tmp left",
          message == ops.PERMISSION == "Close the file in the other program (e.g. Excel) and try again."
          and not os.path.exists(path + ".tmp"), message)


# ------------------------------------------------------------------------------ the other nodes, fully locked

def phase_nodes():
    tree = new_tree("Locked")
    ask = tree.nodes.new("GeometryNodeOpenActAsk")
    ask.inputs["Question"].default_value = "Is it night?"
    ask.inputs["Overrides"].default_value = write("ask.csv", "question,answer,lock\n IS IT NIGHT? ,yes,1\n")
    pick = tree.nodes.new("GeometryNodeOpenActPick")
    pick.inputs["Prompt"].default_value = "something to sit on"
    pick.inputs["Items"].default_value = "table, lamp, armchair"
    pick.inputs["Overrides"].default_value = write("pick.csv", "prompt,answer,lock\nsomething to sit on,Lamp,1\n")
    classify = tree.nodes.new("GeometryNodeOpenActClassify")
    classify.inputs["Items"].default_value = "steel, oak"
    classify.inputs["Labels"].default_value = "metal; wood"
    classify.inputs["Overrides"].default_value = write("classify.csv", "item,answer,lock\nsteel,wood,1\noak,METAL,1\n")
    dial = tree.nodes.new("GeometryNodeOpenActDials")
    dial.inputs["Prompt"].default_value = "A stormy night"
    dial.inputs["Settings"].default_value = write(
        "settings.csv", "setting,type,question,answers,min,max\nWind,amount,How windy is it?,,0,100\n"
                        "Rain,yes/no,Is it raining?,,,\nTime,one-of,What time is it?,dawn: early; dusk: late,,\n")
    dial.inputs["Overrides"].default_value = write("dials.csv", "setting,answer,lock\nWind,25,1\nRain,yes,1\n"
                                                               "Time,Dusk,1\n")
    with Capture() as capture:
        engine.run_pass()
        engine.run_pass()
    got = {n.name: entry_of(n) for n in (ask, pick, classify, dial)}
    check("nodes: Ask, Pick, Classify and Dials fully locked send 0 requests", capture.asked == 0
          and all(got.values()), (capture.asked, {k: status(n) for k, n in zip(got, (ask, pick, classify, dial))}))
    e = got[ask.name]
    check("nodes: Ask locks by question (trimmed, ignoring case): Answer 1, Confidence 1",
          e is not None and e.single == {"Answer": 1.0, "Yes": True, "Confidence": 1.0}, e and e.single)
    e = got[pick.name]
    check("nodes: Pick locks by prompt and forces the item by name: probability 1, the others 0",
          e is not None and e.single["Index"] == 1 and e.single["Item"] == "lamp" and e.single["Confidence"] == 1.0
          and e.rows.v["prob"] == [0.0, 1.0, 0.0] and e.rows.src == "lll", e and (e.single, e.rows.v))
    e = got[classify.name]
    check("nodes: Classify locks by item to a label name", e is not None and e.rows.v["label"] == [1, 0]
          and e.rows.v["c"] == [1.0, 1.0], e and e.rows.v)
    e = got[dial.name]
    check("nodes: Dials amounts are written in output units (25 of 0-100 is 0.25), one-of by option name",
          e is not None and e.rows.v["value"] == [0.25, 1.0, 1.0] and e.single["Wind"] == 25.0
          and e.single["Rain"] is True and e.single["Time"] == 1 and e.single["Confidence"] == 1.0,
          e and (e.rows.v, e.single))
    path = ops.write_overrides(dial)
    job = engine._jobs(dial)[0][1]
    spec = SPECS[dial.bl_idname]
    check("export: Dials keeps its locked rows verbatim, and writes amounts in output units, one-of by name",
          rows_of(path)[1:] == [["Wind", "25", "1", "", "", "", ""], ["Rain", "yes", "1", "", "", "", ""],
                                ["Time", "Dusk", "1", "", "", "", ""]]
          and [spec.answer_text(job, i, {"value": v}) for i, v in enumerate((0.625, 0.2, 0.0))]
          == ["62.5", "no", "dawn"], rows_of(path))
    write("dials.csv", "setting,answer,lock\nWind,250,1\nRain,yes,1\nTime,Dusk,1\n")
    with Capture() as capture:
        engine.run_pass()
    job = engine._jobs(dial)[0][1]
    check("nodes: a Dials amount outside min-max is an error on its row, which the model answers instead",
          len(capture.cells) == 1 and job.rows[0].source == "pending"
          and any("line 2: '250' is not a number from 0 to 100" in p for p in job.problems),
          (len(capture.cells), job.problems))


# ------------------------------------------------------------------------------ Rules: a locked corner

def phase_rules():
    factors = write("warmth_factors.csv", "factor,kind,level,when\nwarmth,amount,cold,freezing\n"
                                          "warmth,amount,mild,cool to warm\nwarmth,amount,hot,hot all year\n")
    xs = [0.1, 0.4, 0.55, 0.8]
    tree = new_tree("RulesLock")
    gi, go = tree.nodes.new("NodeGroupInput"), tree.nodes.new("NodeGroupOutput")
    rules = tree.nodes.new(R.RulesSpec.idname)
    rules.answer_type = 'YES_NO'
    rules.inputs["Question"].default_value = "Does it snow here?"
    rules.inputs["Factors"].default_value = factors
    rules.inputs["Overrides"].default_value = write("rules_overrides.csv", "warmth,answer,lock\nMild,0.7,1\n")
    obj = user("RulesUser", tree, {"in_warmth": xs})
    with Capture() as capture:
        engine.run_pass()
    ctx = next(c for c in evaluator.find_uses()[tree.as_pointer()] if c.obj == obj)
    job = SPECS[R.RulesSpec.idname].plan(rules, ctx)
    check("rules: the locked cell is never asked; the other two are",
          len(capture.of(rules)) == 2 and [row.source for row in job.rows] == ["pending", "lock", "pending"]
          and job.rows[1].values == {"p": 0.7, "c": 1.0, "ok": 1.0}, (len(capture.of(rules)), job.rows[1]))
    for row, p in zip(job.rows, (0.9, None, 0.1)):
        if p is not None:
            pipeline.remember(row.cell.key, {"p": p, "model": "fake"})
    with Capture() as capture:
        engine.run_pass()
    attribute = tree.nodes.new("GeometryNodeInputNamedAttribute")
    attribute.data_type = 'FLOAT'
    attribute.inputs["Name"].default_value = "in_warmth"
    tree.links.new(next(s for s in attribute.outputs if s.name == "Attribute" and s.enabled), rules.inputs["warmth"])
    tree.links.new(store(tree, gi.outputs["Geometry"], rules.outputs["Probability"], "p", 'FLOAT'), go.inputs[0])
    got = read(obj, "p")
    table = [(0.9, 0.8, 1.0), (0.7, 1.0, 1.0), (0.1, 0.8, 1.0)]
    parsed = csvfmt.parse_factors(csvfmt.read_csv(factors, schema="factors"))
    expected = [round(rc.blend(parsed, [x], table)[0], 4) for x in xs]
    entry = entry_of(rules)
    check("rules: a locked corner takes part in interpolation (rules_core.blend), 0 requests after the answers",
          not capture.of(rules) and entry is not None and entry.rows.src == "mlm" and got == expected,
          (len(capture.of(rules)), got, expected))


try:
    phase_sockets()
    phase_match()
    phase_nodes()
    phase_rules()
except Exception:
    import traceback

    traceback.print_exc()
    check("no exception", False)
finally:
    shutil.rmtree(TMP, ignore_errors=True)
finish()
