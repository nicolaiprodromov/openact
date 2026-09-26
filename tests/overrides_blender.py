"""The Overrides CSV (§7.6) in a headless Blender, with no model: fake answers go into MEMORY, and engine._submit and
server.ask are patched to count what the passes would ask.

- Every node has the Overrides word and its CSV companion, hidden like Key.
- Match through the pass: export fills the empty socket; locking an exported row keeps the model's probability; locks
  send 0 requests; the orphaned-lock warning; editing a lock is a values-only update; an invalid answer is an error and
  the model's answer stays; the export merge keeps locked rows verbatim and user columns; a target file held open
  reports the PermissionError message.
- Ask, Pick, Classify and Dials (an amount in output units) fully locked: 0 requests.
- Rules: a locked corner takes part in interpolation, against rules_core.blend.
- The export never loses or changes rows it doesn't own, byte for byte: it creates a missing file and refuses one with
  errors; of duplicate keys the locked row wins and the others stay; a name twice in the table is written once; a CSV
  that isn't an Overrides CSV for the node (no key column, or an Items or Settings CSV no node reads: no 'lock'
  column), or is a node's input, is refused; every context sharing the file adds its rows, and a per-object file gets
  only its object's; a BOM, a ';' delimiter and a read-only flag are respected; a leftover .tmp is replaced; an
  untrusted file can't write outside its folder.
- A factor can't be named after an Overrides column; numeric level names lock by name, and exported numbers read back
  as the same value.

Run:  blender -b --factory-startup --python tests/overrides_blender.py
"""

import csv
import inspect
import os
import shutil
import stat
import sys
import tempfile
import traceback

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import check, finish, store  # noqa: E402
from openact import csvfmt, engine, evaluator, nodes, ops, pipeline, server, trust  # noqa: E402
from openact import prefs as P  # noqa: E402
from openact import rules_core as rc  # noqa: E402
from openact import store as S  # noqa: E402
from openact.specs import SPECS  # noqa: E402
from openact.specs import dials as D  # noqa: E402
from openact.specs import rules as R  # noqa: E402
from openact.specs.base import lock_answer  # noqa: E402

P._Defaults.auto_start = False  # nothing here may start a model server
nodes.register()
ops.register()
TMP = tempfile.mkdtemp(prefix="openact_overrides_")
POINTS = 4
_stamp = [1_700_000_000 * 10**9]
HEADER = ["item", "answer", "lock", "model_answer", "confidence", "unsure", "source"]
KITCHEN = {"pan": {"p": 0.9}, "sofa": {"p": 0.1}, "kettle": {"p": 0.8}}  # fake Yes/No answers by item
MODEL_ROWS = [["pan", "0.9", "0", "0.9", "0.80", "0", "model"], ["sofa", "0.1", "0", "0.1", "0.80", "0", "model"],
              ["kettle", "0.8", "0", "0.8", "0.60", "0", "model"]]  # how the export writes KITCHEN


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


def bytes_of(path):
    with open(path, "rb") as handle:
        return handle.read()


def csv_text(rows, delimiter=","):
    """The text the export writes for these rows: no cell here needs quoting."""
    return "".join(delimiter.join(row) + "\n" for row in rows)


def export(node, index=0, path=None):
    """"written", or the message of the OverridesError the export raised."""
    try:
        ops.write_overrides(node, index, path)
    except ops.OverridesError as error:
        return str(error)
    return "written"


class Capture:
    """Patches engine._submit and server.ask: the missing cells the passes want, and any request, are recorded."""

    def __enter__(self):
        self.cells, self.requests = [], []
        self.saved = engine._submit, server.ask
        engine._submit = lambda job, missing: self.cells.extend((job.uid, cell) for cell in missing)
        server.ask = lambda payload, on_done, timeout=180.0: self.requests.append(payload)
        return self

    def __exit__(self, *_exc):
        engine._submit, server.ask = self.saved

    @property
    def asked(self):
        return len(self.cells) + len(self.requests)

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


def match_node(name, items="pan, sofa, kettle", answer_type='YES_NO', answers=""):
    """A Match node asking the kitchen question, on a tree of its own that one object uses."""
    tree = new_tree(name)
    node = tree.nodes.new(SPECS["GeometryNodeOpenActMatch"].idname)
    node.answer_type = answer_type
    node.inputs["Question"].default_value = "Would you find it in a kitchen?"
    node.inputs["Answers"].default_value = answers
    if items is not None:
        node.inputs["Items"].default_value = items
    gi, go = tree.nodes.new("NodeGroupInput"), tree.nodes.new("NodeGroupOutput")
    tree.links.new(store(tree, gi.outputs["Geometry"], node.outputs["Probability"], "p", 'FLOAT'), go.inputs[0])
    user(name + "User", tree)
    return node


def answer(node, answers):
    """Plan the node's contexts, answer their cells with fake answers by row name, and show them."""
    with Capture():
        engine.run_pass()
    for _ctx, job in engine._jobs(node):
        for row in job.rows:
            if row.cell is not None:
                pipeline.remember(row.cell.key, dict(answers[row.name], model="fake"))
    engine.run_pass()


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
          rows_of(path) == [HEADER, *MODEL_ROWS], rows_of(path))
    with Capture() as capture:
        engine.run_pass()
    check("export: the exported file locks nothing: the table and its words key stay", capture.asked == 0
          and entry_of(node) is not None and entry_of(node).rows.src == "mmm", capture.asked)
    exported = rows_of(path)
    exported[1][2] = "1"  # the artist locks pan's row as it was exported
    write("kitchen_overrides.csv", csv_text(exported))
    with Capture() as capture:
        engine.run_pass()
    entry = entry_of(node)
    check("export: locking an exported Yes/No row keeps the model's probability (0.9, not 1)",
          capture.asked == 0 and entry is not None and entry.rows.src == "lmm" and entry.rows.v["p"] == [0.9, 0.1, 0.8],
          (capture.asked, entry and (entry.rows.src, entry.rows.v["p"])))

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
          rows_of(path) == [[*HEADER, "note"], ["pan", "Yes", "x", "", "", "", "", "mine"],
                            [*MODEL_ROWS[1], "check"], [*MODEL_ROWS[2], ""], ["Ghost", "yes", "1", "", "", "", "", "old"]],
          rows_of(path))

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
    source = inspect.getsource(D)
    check("dials: no merge-history tags, and one import from .base",
          "PR-21" not in source and source.count("from .base import") == 1)


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
    rules_overrides = write("rules_overrides.csv", "warmth,answer,lock\nMild,0.7,1\n")
    rules.inputs["Overrides"].default_value = rules_overrides
    obj = user("RulesUser", tree, {"in_warmth": xs})
    with Capture() as capture:
        engine.run_pass()
    ctx = next(c for c in evaluator.find_uses()[tree.as_pointer()] if c.obj == obj)
    spec = SPECS[R.RulesSpec.idname]
    job = engine._plan(spec, rules, ctx)
    check("rules: plan() leaves the Overrides CSV to engine._plan(), which applies it to every spec's plan once",
          [row.source for row in spec.plan(rules, ctx).rows] == ["pending"] * 3)
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
    got = export(rules)
    check("rules: the export writes Yes/No answers as probabilities, by factor level, and the locked corner verbatim",
          got == "written" and rows_of(rules_overrides) == [
              ["warmth", *HEADER[1:]], ["cold", "0.9", "0", "0.9", "0.80", "0", "model"],
              ["Mild", "0.7", "1", "", "", "", ""], ["hot", "0.1", "0", "0.1", "0.80", "0", "model"]],
          (got, rows_of(rules_overrides)))


# ------------------------------------------------------------------------------ export: a missing or broken file

def phase_missing():
    node = match_node("Missing")
    answer(node, KITCHEN)
    target = os.path.join(TMP, "typed_overrides.csv")
    node.inputs["Overrides"].default_value = target
    engine.run_pass()
    check("missing: a typed Overrides path to a file that doesn't exist yet is the node's error",
          status(node)[0] == 'ERROR' and "Can't find" in status(node)[1], status(node))
    got = export(node)
    check("missing: the export creates that file, with the model's answers",
          got == "written" and os.path.isfile(target) and bytes_of(target) == csv_text([HEADER, *MODEL_ROWS]).encode(),
          (got, os.path.isfile(target) and rows_of(target)))
    with Capture() as capture:
        engine.run_pass()
    check("missing: the node then reads the new file: no error, nothing asked",
          status(node)[0] == 'CHECKMARK' and not capture.of(node), (status(node), len(capture.of(node))))
    broken = write("locked_twice.csv", "item,answer,lock\npan,yes,1\nPan,no,1\nsofa,no,0\n")
    before = bytes_of(broken)
    node.inputs["Overrides"].default_value = broken
    engine.run_pass()
    got = export(node)
    check("missing: a file with errors is refused with its own error, and left as it was",
          got == "locked_twice.csv: Line 3: 'Pan' is locked twice (also line 2). Fix it, or write another file"
          and bytes_of(broken) == before, got)


# ------------------------------------------------------------------------------ export: duplicate keys

def phase_duplicates():
    node = match_node("Twice")
    answer(node, KITCHEN)
    path = write("twice_overrides.csv", "item,answer,lock,note\nsofa,no,0,draft\nSofa,yes,1,mine\n")
    node.inputs["Overrides"].default_value = path
    engine.run_pass()
    entry = entry_of(node)
    check("duplicates: of two rows with one key, the locked one applies",
          entry is not None and entry.rows.src == "mlm" and entry.rows.v["p"] == [0.9, 1.0, 0.8],
          entry and (entry.rows.src, entry.rows.v["p"]))
    got = export(node)
    check("duplicates: the export keeps the locked row verbatim, and the other row with its key as it was",
          got == "written" and bytes_of(path) == csv_text([
              [*HEADER, "note"], [*MODEL_ROWS[0], ""], ["Sofa", "yes", "1", "", "", "", "", "mine"],
              [*MODEL_ROWS[2], ""], ["sofa", "no", "0", "", "", "", "", "draft"]]).encode(), (got, rows_of(path)))
    with Capture() as capture:
        engine.run_pass()
    entry = entry_of(node)
    check("duplicates: after the export the lock still applies, and nothing is asked",
          not capture.of(node) and entry is not None and entry.rows.src == "mlm" and entry.rows.v["p"] == [0.9, 1.0, 0.8],
          (len(capture.of(node)), entry and (entry.rows.src, entry.rows.v["p"])))

    palette = bpy.data.objects.new("Palette", bpy.data.meshes.new("Palette"))
    steel, wood = bpy.data.materials.new("Steel"), bpy.data.materials.new("Wood")
    for material in (steel, wood, steel):  # one material in two slots, as after joining meshes
        palette.data.materials.append(material)
    node = match_node("Slots", items=None)
    node.list_source = 'MATERIALS'
    node.inputs["Object"].default_value = palette
    answer(node, {"Steel": {"p": 0.9}, "Wood": {"p": 0.2}})
    path = os.path.join(TMP, "slots_overrides.csv")
    steel_row, wood_row = ["Steel", "0.9", "0", "0.9", "0.80", "0", "model"], ["Wood", "0.2", "0", "0.2", "0.60", "0",
                                                                                 "model"]
    got = export(node, 0, path)
    check("duplicates: a material in two slots gets one row",
          got == "written" and bytes_of(path) == csv_text([HEADER, steel_row, wood_row]).encode(), (got, rows_of(path)))
    locked = ["Steel", "0.5", "1", *steel_row[3:]]
    write("slots_overrides.csv", csv_text([HEADER, locked, wood_row]))
    engine.run_pass()
    entry = entry_of(node)
    check("duplicates: its lock applies to both slots", entry is not None and entry.rows.src == "lml"
          and entry.rows.v["p"] == [0.5, 0.2, 0.5], entry and (entry.rows.src, entry.rows.v["p"]))
    got = export(node)
    engine.run_pass()
    entry = entry_of(node)
    check("duplicates: exporting again writes that lock once, verbatim, and the node still reads the file",
          got == "written" and bytes_of(path) == csv_text([HEADER, locked, wood_row]).encode()
          and status(node)[0] == 'CHECKMARK' and entry is not None and entry.rows.src == "lml",
          (got, rows_of(path), status(node)))


# ------------------------------------------------------------------------------ export: files it must not merge into

def phase_unrelated():
    node = match_node("Unrelated")
    answer(node, KITCHEN)
    contacts = write("contacts.csv", "name,phone,email\nAlice,555-0100,alice@example.com\n"
                                     "Bob,555-0199,bob@example.com\n")
    before = bytes_of(contacts)
    got = export(node, 0, contacts)
    check("unrelated: an existing CSV without the node's key column is refused and left as it was",
          got == "contacts.csv is not an Overrides CSV for this node (no 'item' column): choose another file"
          and bytes_of(contacts) == before and node.inputs["Overrides"].default_value == "", got)
    inventory = write("inventory.csv", "item,when\npan,for cooking\nsofa,to sit on\nkettle,boils water\n")
    before = bytes_of(inventory)
    got = export(node, 0, inventory)  # e.g. another .blend's Items CSV: no node here reads it
    check("unrelated: an Items CSV no node reads is refused although it has the 'item' column (no 'lock'), and left as "
          "it was", got == "inventory.csv is not an Overrides CSV (no 'lock' column): choose another file"
          and bytes_of(inventory) == before and node.inputs["Overrides"].default_value == "", got)
    tree = new_tree("UnrelatedAsk")
    ask = tree.nodes.new("GeometryNodeOpenActAsk")
    ask.inputs["Question"].default_value = "Is it windy?"
    user("UnrelatedAskUser", tree)
    answer(ask, {"Is it windy?": {"p": 0.7}})
    settings = write("weather_settings.csv", "setting,type,question,answers,min,max\n"
                                             "Wind,amount,How windy is it?,,0,100\nRain,yes/no,Is it raining?,,,\n")
    before = bytes_of(settings)
    got = export(ask, 0, settings)
    check("unrelated: a Settings CSV no node reads is refused for Ask, whose 'question' key column it has, and left as "
          "it was", got == "weather_settings.csv is not an Overrides CSV (no 'lock' column): choose another file"
          and bytes_of(settings) == before and ask.inputs["Overrides"].default_value == "", got)
    items = write("kitchen_items.csv", "item,when\npan,for cooking\nsofa,to sit on\nkettle,boils water\n")
    node.inputs["Items"].default_value = items
    before = bytes_of(items)
    got = export(node, 0, items)
    check("unrelated: the node's own Items CSV is refused and left as it was, although it has an 'item' column",
          got == f"kitchen_items.csv is the Items CSV of '{node.name}': choose another file"
          and bytes_of(items) == before, got)
    flag = bpy.ops.openact.write_overrides.get_rna_type().properties.get("check_existing")
    check("unrelated: the file dialog flags an existing file (check_existing)",
          flag is not None and flag.default is True and flag.is_hidden, flag)


# ------------------------------------------------------------------------------ export: several contexts

def phase_contexts():
    tree = new_tree("Shared")
    tree.interface.new_socket("Question", in_out='INPUT', socket_type='NodeSocketString')
    ask = tree.nodes.new("GeometryNodeOpenActAsk")
    gi, go = tree.nodes.new("NodeGroupInput"), tree.nodes.new("NodeGroupOutput")
    tree.links.new(gi.outputs["Question"], ask.inputs["Question"])
    tree.links.new(gi.outputs["Geometry"], go.inputs[0])
    question = tree.interface.items_tree["Question"].identifier
    for name, text in (("Night", "Is it night?"), ("Rain", "Is it raining?")):
        user(name, tree).modifiers["OpenAct"][question] = text
    header, foggy = ["question", *HEADER[1:], "note"], ["Is it foggy?", "no", "0", "", "", "", "", "from the old shot"]
    path = write("shared_overrides.csv", csv_text([
        header, ["Is it night?", "yes", "0", "yes", "0.80", "0", "model", "check at dusk"],
        ["Is it raining?", "no", "0", "no", "0.80", "0", "model", "ask the director"], foggy]))
    ask.inputs["Overrides"].default_value = path
    answer(ask, {"Is it night?": {"p": 0.9}, "Is it raining?": {"p": 0.2}})
    rows = {"Is it night?": ["Is it night?", "0.9", "0", "0.9", "0.80", "0", "model", "check at dusk"],
            "Is it raining?": ["Is it raining?", "0.2", "0", "0.2", "0.60", "0", "model", "ask the director"]}
    order = [job.rows[0].name for _ctx, job in engine._jobs(ask)]
    expected = csv_text([header, *(rows[name] for name in order), foggy]).encode()
    got = [export(ask, 0), bytes_of(path)]
    check("contexts: the export writes every context's question with its own answer, and keeps the notes and the "
          "file's other rows", sorted(order) == sorted(rows) and got == ["written", expected], (order, got))
    got = [export(ask, 1), bytes_of(path)]
    check("contexts: exporting the other context writes the same file", got == ["written", expected], got)

    tree = new_tree("PerObject")
    ask = tree.nodes.new("GeometryNodeOpenActAsk")
    gi, go = tree.nodes.new("NodeGroupInput"), tree.nodes.new("NodeGroupOutput")
    for name in ("Question", "Overrides"):
        tree.interface.new_socket(name, in_out='INPUT', socket_type='NodeSocketString')
        tree.links.new(gi.outputs[name], ask.inputs[name])
    tree.links.new(gi.outputs["Geometry"], go.inputs[0])
    question, overrides = (tree.interface.items_tree[name].identifier for name in ("Question", "Overrides"))
    moon_file = write("moon_overrides.csv", "question,answer,lock\nIs the moon up?,yes,1\n")
    sun, moon = user("Sun", tree), user("Moon", tree)
    sun.modifiers["OpenAct"][question] = "Is the sun up?"
    moon.modifiers["OpenAct"][question], moon.modifiers["OpenAct"][overrides] = "Is the moon up?", moon_file
    answer(ask, {"Is the sun up?": {"p": 0.7}})  # the moon's question is locked
    sun_file = os.path.join(TMP, "sun_overrides.csv")
    sun.modifiers["OpenAct"][overrides] = sun_file  # per object, through Group Input (§7.6 Scope)
    engine.run_pass()
    before = bytes_of(moon_file)
    got = export(ask, [ctx.obj for ctx in evaluator.find_uses()[tree.as_pointer()]].index(sun))
    check("contexts: a per-object Overrides file that doesn't exist yet is created with its object's row; the other "
          "object's file is left as it was",
          got == "written" and os.path.isfile(sun_file) and bytes_of(sun_file) == csv_text(
              [["question", *HEADER[1:]], ["Is the sun up?", "0.7", "0", "0.7", "0.40", "0", "model"]]).encode()
          and bytes_of(moon_file) == before, (got, os.path.isfile(sun_file) and rows_of(sun_file)))


# ------------------------------------------------------------------------------ Rules: factor names

def phase_clash():
    factors = write("clash_factors.csv", "factor,kind,level,value,when\nsource,category,river,0,Water from a river.\n"
                                         "source,category,lake,1,Water from a lake.\n")
    tree = new_tree("Clash")
    rules = tree.nodes.new(R.RulesSpec.idname)
    rules.answer_type = 'YES_NO'
    rules.inputs["Question"].default_value = "Is the water safe to drink?"
    rules.inputs["Factors"].default_value = factors
    user("ClashUser", tree)
    with Capture():
        engine.run_pass()
    target = os.path.join(TMP, "clash_overrides.csv")
    got = export(rules, 0, target)
    check("rules: a factor named after an Overrides CSV column is an error, so no export writes that column twice",
          status(rules)[0] == 'ERROR'
          and "Line 2: rename factor 'source': it clashes with an Overrides CSV column" in status(rules)[1]
          and got == "Nothing to export: the node has no answered context" and not os.path.exists(target),
          (status(rules), got))


# ------------------------------------------------------------------------------ numbers: level names and digits

def phase_numbers():
    got = [lock_answer('AMOUNT', text, ("1", "2", "3", "4", "5")) for text in ("1", "3", "5", "0.5")]
    check("numbers: an Amount lock names a level before it reads a number 0-1 ('1' of 1-5 is the lowest)",
          got == [0.0, 0.5, 1.0, 0.5], got)
    node = match_node("Levels", answer_type='AMOUNT', answers="0; 1; 2; 3")
    certain = [0.0, 0.0, 0.0, 1.0]
    answer(node, {"pan": {"amount": 1.0, "probabilities": certain}, "sofa": {"amount": 0.25, "probabilities": certain},
                  "kettle": {"amount": 0.5, "probabilities": certain}})
    path = os.path.join(TMP, "levels_overrides.csv")
    got = export(node, 0, path)
    exported = rows_of(path) if os.path.isfile(path) else [[]]
    check("numbers: an exported amount that would read as a level name is written in full ('1.0', not level '1')",
          got == "written" and [row[1] for row in exported[1:]] == ["1.0", "0.25", "0.5"], (got, exported))
    locked = [[row[0], row[1], "1", *row[3:]] for row in exported[1:]]
    locked[1][1] = "2"  # sofa locked by a level's name
    write("levels_overrides.csv", csv_text([exported[0], *locked]))
    with Capture() as capture:
        engine.run_pass()
    entry = entry_of(node)
    check("numbers: locking the exported amounts keeps them, and level '2' of '0; 1; 2; 3' locks as 2/3",
          not capture.of(node) and entry is not None and entry.rows.src == "lll"
          and entry.rows.v["p"] == [1.0, 2 / 3, 0.5], (len(capture.of(node)), entry and (entry.rows.src,
                                                                                         entry.rows.v["p"]), status(node)))
    dials = SPECS[D.DialsSpec.idname]
    job = dials.job(("A grain of sand", "", "", ""), [csvfmt.Setting("Tiny", "amount", "How big?", (), 0.0, 0.01, 0)],
                    False)
    text = dials.answer_text(job, 0, {"value": 0.37})
    back = dials.lock_values(job, (0,), text)
    check("numbers: a small range keeps its digits (0.37 of 0-0.01 exports 0.0037, which locks as 0.37)",
          text == "0.0037" and round(back[0]["value"], 9) == 0.37, (text, back))
    got = [lock_answer('YES_NO', "1_0e-1"), lock_answer('YES_NO', "\u0660.\u0665"), dials.lock_values(job, (0,), "1_0e-3")]
    check("numbers: a lock's number reads as every other number in these files do ('1_0e-1' and Arabic digits don't)",
          all(isinstance(value, str) for value in got), got)


# ------------------------------------------------------------------------------ export: the file itself

def phase_files():
    node = match_node("Files")
    answer(node, KITCHEN)
    path = os.path.join(TMP, "bom_overrides.csv")
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        handle.write("item,answer,lock,note\npan,yes,0,caf\u00e9\n")
    node.inputs["Overrides"].default_value = path
    engine.run_pass()
    written = b"\xef\xbb\xbf" + csv_text([[*HEADER, "note"], [*MODEL_ROWS[0], "caf\u00e9"], [*MODEL_ROWS[1], ""],
                                          [*MODEL_ROWS[2], ""]]).encode()
    got = export(node)
    check("files: a file that starts with a BOM, as Excel's CSV UTF-8 writes it, keeps it",
          got == "written" and bytes_of(path) == written, (got, bytes_of(path)[:60]))
    with open(path + ".tmp", "w", encoding="utf-8") as handle:
        handle.write("half an export")
    got = export(node)
    check("files: a .tmp left by an export that stopped half-way is replaced, and none is left",
          got == "written" and bytes_of(path) == written and not os.path.exists(path + ".tmp"), got)
    os.chmod(path, stat.S_IREAD)
    try:
        got = export(node)
    finally:
        os.chmod(path, stat.S_IREAD | stat.S_IWRITE)
    check("files: a read-only file is refused as read-only, and left as it was",
          got == f"Can't write {path}: the file is read-only" and bytes_of(path) == written
          and not os.path.exists(path + ".tmp"), got)
    semicolon = write("semicolon_overrides.csv", "item;answer;lock;note\npan;yes;0;a,b\n")
    node.inputs["Overrides"].default_value = semicolon
    engine.run_pass()
    got = export(node)
    check("files: a file written with ';' keeps it", got == "written" and bytes_of(semicolon) == csv_text(
        [[*HEADER, "note"], [*MODEL_ROWS[0], "a,b"], [*MODEL_ROWS[1], ""], [*MODEL_ROWS[2], ""]], ";").encode(),
        (got, bytes_of(semicolon)))
    node.inputs["Overrides"].default_value = ""
    project, far = os.path.join(TMP, "project"), os.path.join(TMP, "far_overrides.csv")
    os.makedirs(project)
    trusted, cwd = trust.is_trusted, os.getcwd()
    trust.is_trusted = lambda: False  # a file received from someone else: unsaved, its folder is the working folder
    os.chdir(project)
    try:
        got = export(node, 0, far)
    finally:
        os.chdir(cwd)
        trust.is_trusted = trusted
    check("files: an untrusted file can't write outside its folder, and the message says writing",
          got == f"Blocked writing {far}: outside this file's folder. Allow this file (or trust its folder) to write it"
          and not os.path.exists(far), got)


try:
    for phase in (phase_sockets, phase_match, phase_nodes, phase_rules, phase_missing, phase_duplicates,
                  phase_unrelated, phase_contexts, phase_clash, phase_numbers, phase_files):
        try:
            phase()
        except Exception:  # the later phases still run: each builds its own trees
            traceback.print_exc()
            check(f"{phase.__name__}: no exception", False)
finally:
    shutil.rmtree(TMP, ignore_errors=True)
finish()
