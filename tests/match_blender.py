"""Match and Classify (§6.3, §6.4), with no server by default:
- synthetic tables: Match's Probability and Confidence per element, Selection = valid AND p >= Threshold with Threshold
  driven by a field; Classify's Label, Confidence and payload per element; invalid IDs give -1 and the fallbacks;
- the planned jobs: 40 items sharing one state, with the §10.2 template text; 255 labels x 250 items refused by the
  8192-row estimate before anything is sent; the tournament cost of 20 labels;
- the pass from plan to table with cached answers: a Labels CSV's payload denormalised per row (a numbers-only edit
  re-bakes it and asks nothing), and an empty material slot that is never asked;
- a draw (the sidebar's answers_for(), where Blender refuses ID writes) never writes a Classify node: after a Labels
  CSV path is typed, a header renamed or a column removed, the next pass syncs the payload outputs, also after an undo
  that keeps the plans;
- two contexts must agree on their Labels CSVs' payload headers and types, in any order (§6.0): other ones give the
  error, and the outputs, their links and the last good table stay, also while a CSV has an error; while a context
  has no job (a missing CSV, an item-list error, a Context waiting for a Pick), the outputs and their links stay,
  and another context's new words are still asked and shown.
With -- --model PORT GPU, also a spot check with the real model on that port and GPU: pan = yes and sofa = no for
"Would you find it in a kitchen?"; "steel" -> metal and "oak planks" -> wood.

Run:  blender -b --factory-startup --python tests/match_blender.py [-- --model 18767 1]
"""

import os
import shutil
import sys
import tempfile
import time

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import check, drive, enable_addon, finish, store  # noqa: E402

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
MODEL = [int(arg) for arg in ARGS[ARGS.index("--model") + 1:][:2]] if "--model" in ARGS else None
if MODEL is not None:  # enabled like e2e_blender.py: its preferences choose the port and GPU
    engine, server, P = enable_addon()
    PREFS = P.get_prefs()
    PREFS.port, PREFS.gpu, PREFS.auto_start = MODEL[0], MODEL[1], False  # no server before the model phase
else:
    from openact import engine, nodes, ops, server  # noqa: E402
    from openact import prefs as P  # noqa: E402

    P._Defaults.auto_start = False  # nothing here may start a model server
    nodes.register()
    ops.register()
from openact import csvfmt, evaluator, inner, lists, pipeline, sources, tables  # noqa: E402
from openact import store as S  # noqa: E402
from openact.specs.classify import ClassifySpec  # noqa: E402
from openact.specs.match import MatchSpec  # noqa: E402

MATCH, CLASSIFY = engine.SPECS[MatchSpec.idname], engine.SPECS[ClassifySpec.idname]
FAMILIES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "material_families.csv")
KITCHEN = "Would you find it in a kitchen?"
POINTS = 6
PAYLOAD = [("roughness", 'FLOAT'), ("metallic", 'INT')]  # the numeric columns of material_families.csv
FAMILY_ANSWERS = {"steel": [0.9, 0.05, 0.03, 0.02], "oak planks": [0.1, 0.8, 0.05, 0.05],  # over its 4 labels
                  "marble": [0.05, 0.05, 0.85, 0.05], "silk": [0.1, 0.1, 0.1, 0.7]}


# ------------------------------------------------------------------------------ scenes

def make_tree(name, idname, words=()):
    """A tree whose OpenAct node takes `words` from the modifier, and ID = index + Shift (a modifier input)."""
    tree = bpy.data.node_groups.new(name, 'GeometryNodeTree')
    tree.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    for word in words:
        tree.interface.new_socket(word, in_out='INPUT', socket_type='NodeSocketString')
    tree.interface.new_socket("Shift", in_out='INPUT', socket_type='NodeSocketInt')
    tree.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    gi = tree.nodes.new("NodeGroupInput")
    node = tree.nodes.new(idname)
    for word in words:
        tree.links.new(gi.outputs[word], node.inputs[word])
    shifted = tree.nodes.new("FunctionNodeIntegerMath")
    shifted.operation = 'ADD'
    tree.links.new(tree.nodes.new("GeometryNodeInputIndex").outputs[0], shifted.inputs[0])
    tree.links.new(gi.outputs["Shift"], shifted.inputs[1])
    tree.links.new(shifted.outputs[0], node.inputs["ID"])
    return tree, node


def store_outputs(tree, node, outputs):
    """Store each of the node's outputs {name: data type} as a point attribute of that name."""
    geometry = next(n for n in tree.nodes if n.bl_idname == "NodeGroupInput").outputs["Geometry"]
    for output, data_type in outputs.items():
        geometry = store(tree, geometry, node.outputs[output], output, data_type)
    tree.links.new(geometry, tree.nodes.new("NodeGroupOutput").inputs[0])


def user(name, tree, attributes=None, **inputs):
    """An object of POINTS points using the tree, with modifier inputs by name and FLOAT point attributes."""
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([(i, 0, 0) for i in range(POINTS)], [], [])
    for attribute, values in (attributes or {}).items():
        mesh.attributes.new(attribute, 'FLOAT', 'POINT').data.foreach_set("value", values)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    mod = obj.modifiers.new("OpenAct", 'NODES')
    mod.node_group = tree
    idents = {i.name: i.identifier for i in tree.interface.items_tree if getattr(i, "in_out", "") == 'INPUT'}
    for key, value in inputs.items():
        mod[idents[key]] = value
    obj.update_tag()
    return obj


def read(obj, names):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        return {name: [round(d.value, 4) if isinstance(d.value, float) else d.value for d in mesh.attributes[name].data]
                if name in mesh.attributes else None for name in names}
    finally:
        evaluated.to_mesh_clear()


def entry(kind, gn, v, payload=None):
    n = len(next(iter(v.values())))
    shape = {"rows": n, "columns": {name: "INT" if name == "label" else "FLOAT" for name in v}}
    if payload:
        shape["payload"] = [list(column) for column in PAYLOAD]
    return S.Entry(words_key="|".join(gn), gn=gn, t=1.0, kind=kind, answer_type='ONE_OF' if kind == "classify" else
                   'YES_NO', asks=["a"], shape=shape,
                   rows=S.Rows(name=[f"item {i}" for i in range(n)], wkey=[""] * n, ask=[0] * n, src="m" * n, v=v,
                               payload=payload or {}))


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

    def keys(self, node):
        return {cell.key for uid, cell in self.cells if uid == node.uid}


DRAWN = {}  # what drawn() runs, and what came of it


class OPENACT_TEST_OT_drawn(bpy.types.Operator):
    """Runs DRAWN["fn"] in poll(), where Blender refuses ID writes as it does in a panel's draw()."""

    bl_idname = "openact_test.drawn"
    bl_label = "Drawn"

    @classmethod
    def poll(cls, context):
        try:
            DRAWN["result"] = ("ok", DRAWN["fn"]())
        except Exception as error:  # noqa: BLE001 - in a real draw, the sidebar would lose its answers and buttons
            DRAWN["result"] = (type(error).__name__, str(error))
        return False


bpy.utils.register_class(OPENACT_TEST_OT_drawn)


def drawn(fn):
    """fn() as a draw callback runs it: ("ok", its value), or (the error's type, its message)."""
    DRAWN.update(fn=fn, result=None)
    bpy.ops.openact_test.drawn.poll()
    return DRAWN["result"]


def face(node):
    """What a write to the node would change: its outputs (name, type, tooltip), the links from them, its group."""
    group = node.node_tree
    outputs = [(i.name, i.socket_type, i.description) for i in group.interface.items_tree
               if i.item_type == 'SOCKET' and i.in_out == 'OUTPUT']
    links = sorted(link.from_socket.name for link in node.id_data.links if link.from_node == node)
    return outputs, links, len(group.nodes), group.get(inner.BUILT)


# ------------------------------------------------------------------------------ synthetic tables

def phase_match_table():
    tree, node = make_tree("MatchTable", MatchSpec.idname, ("Question", "Items"))
    threshold = tree.nodes.new("GeometryNodeInputNamedAttribute")
    threshold.data_type = 'FLOAT'
    threshold.inputs["Name"].default_value = "thr"
    tree.links.new(next(s for s in threshold.outputs if s.name == "Attribute" and s.enabled), node.inputs["Threshold"])
    store_outputs(tree, node, {"Probability": 'FLOAT', "Selection": 'BOOLEAN', "Confidence": 'FLOAT'})
    users = {"MA": user("MA", tree, {"thr": [0.95, 0.5, 0.2, 0.5, 0.0, 0.0]}, Question=KITCHEN,
                        Items="pan, sofa, kettle, lamp", Shift=0),
             "MB": user("MB", tree, {"thr": [0.0, 0.3, 0.3, 0.6, 0.0, 0.0]}, Question="Is it heavy?",
                        Items="anvil, feather, brick", Shift=-1)}
    entries = [entry("match", (KITCHEN, "", "pan, sofa, kettle, lamp", "", "", ""),
                     {"p": [0.9, 0.6, 0.3, 0.45], "c": [0.8, 0.2, 0.4, 0.1]}),
               entry("match", ("Is it heavy?", "", "anvil, feather, brick", "", "", ""),
                     {"p": [0.95, 0.05, 0.7], "c": [0.9, 0.9, 0.4]})]
    node_store, live = S.NodeStore(entries), [e.gn for e in entries]
    inner.build(node, node_store, live)
    names = ("Probability", "Selection", "Confidence")
    got = {key: read(obj, names) for key, obj in users.items()}
    check("match table: two contexts read Probability and Confidence per element at base + ID (MB's ID = index - 1)",
          got["MA"]["Probability"] == [0.9, 0.6, 0.3, 0.45, 0.0, 0.0]
          and got["MA"]["Confidence"] == [0.8, 0.2, 0.4, 0.1, 0.0, 0.0]
          and got["MB"]["Probability"] == [0.0, 0.95, 0.05, 0.7, 0.0, 0.0]
          and got["MB"]["Confidence"] == [0.0, 0.9, 0.9, 0.4, 0.0, 0.0], got)
    check("match table: Selection = p >= Threshold, with Threshold a per-point field (Named Attribute)",
          got["MA"]["Selection"][:4] == [False, True, True, False] and got["MB"]["Selection"][1:4] == [True, False, True],
          {key: value["Selection"] for key, value in got.items()})
    check("match table: invalid IDs (-1, past the rows) are never selected, even where Threshold is 0",
          got["MA"]["Selection"][4:] == [False, False] and got["MB"]["Selection"][0] is False
          and got["MB"]["Selection"][4:] == [False, False], {key: value["Selection"] for key, value in got.items()})
    before = [n.as_pointer() for n in node.node_tree.nodes]
    entries[0].rows.v["p"][3] = 0.55
    updated = inner.update_values(node, node_store, live)
    got = read(users["MA"], names)
    check("match table: a values-only update moves lamp above its Threshold, with the same nodes",
          updated and [n.as_pointer() for n in node.node_tree.nodes] == before and got["Probability"][3] == 0.55
          and got["Selection"][:4] == [False, True, True, True], (updated, got))


def phase_classify_table():
    tree, node = make_tree("ClassifyTable", ClassifySpec.idname, ("Items", "Labels"))
    inner.sync_interface(node, CLASSIFY.interface(node, PAYLOAD))
    outputs = {"Label": 'INT', "Confidence": 'FLOAT', "roughness": 'FLOAT', "metallic": 'INT'}
    store_outputs(tree, node, outputs)
    users = {"CA": user("CA", tree, Items="steel, oak planks, granite", Labels="metal; wood; stone", Shift=0),
             "CB": user("CB", tree, Items="silk, iron", Labels="fabric; metal", Shift=-1)}
    entries = [entry("classify", ("", "", "steel, oak planks, granite", "metal; wood; stone", "", ""),
                     {"label": [0, 1, 2], "c": [0.9, 0.8, 0.7]}, {"roughness": [0.35, 0.6, 0.8], "metallic": [1, 0, 0]}),
               entry("classify", ("", "", "silk, iron", "fabric; metal", "", ""),
                     {"label": [0, 1], "c": [0.6, 0.95]}, {"roughness": [0.9, 0.35], "metallic": [0, 1]})]
    inner.build(node, S.NodeStore(entries), [e.gn for e in entries])
    got = {key: read(obj, outputs) for key, obj in users.items()}
    check("classify table: Label, Confidence and each payload column per element, from the row's own label",
          got["CA"] == {"Label": [0, 1, 2, -1, -1, -1], "Confidence": [0.9, 0.8, 0.7, 0.0, 0.0, 0.0],
                        "roughness": [0.35, 0.6, 0.8, 0.0, 0.0, 0.0], "metallic": [1, 0, 0, 0, 0, 0]}
          and got["CB"]["Label"][1:3] == [0, 1] and got["CB"]["roughness"][1:3] == [0.9, 0.35]
          and got["CB"]["metallic"][1:3] == [0, 1], got)
    check("classify table: an invalid ID gives Label -1, Confidence 0 and payload 0",
          [got["CB"][name][0] for name in outputs] == [-1, 0.0, 0.0, 0]
          and all(got[key][name][4:] == [got[key][name][4]] * 2 and got[key][name][4] in (-1, 0)
                  for key in got for name in outputs), got)
    table = tables.find(node)
    types = {a.name: a.data_type for a in table.data.attributes if a.name.startswith(tables.ATTR)}
    check("classify table: the table object holds oa_label (INT), oa_c, and the payload as oa_pl_0 (FLOAT) and "
          "oa_pl_1 (INT)", types == {"oa_label": 'INT', "oa_c": 'FLOAT', "oa_pl_0": 'FLOAT', "oa_pl_1": 'INT'}, types)


# ------------------------------------------------------------------------------ planned jobs

def phase_jobs():
    Item = lists.Item
    items = [Item(f"thing {i}", "found near a sink" if i % 2 else "") for i in range(40)]
    gn = (KITCHEN, "A small flat", "", "", "", "")
    job = MATCH.job(gn, 'YES_NO', items, sources.default_set("yes_no"), True, "forty")
    cells = [row.cell for row in job.rows]
    check("jobs: 40 items plan into one Job of 40 rows, each asking its own cell",
          len(job.rows) == 40 and all(cells) and len({cell.key for cell in cells}) == 40, len(job.rows))
    check("jobs: every cell shares the job's one state, the Context", {cell.state for cell in cells} == {"A small flat"},
          {cell.state for cell in cells if cell})
    check("jobs: the exact template text (§10.2): 'Item: {name}' + ' — {when}' + '\\n{question}', noul Yes./No.",
          cells[0].question == {"type": "noul", "instructions": f"Item: thing 0\n{KITCHEN}",
                                "criteria": {"true": "Yes.", "false": "No."}}
          and cells[1].question["instructions"] == f"Item: thing 1 — found near a sink\n{KITCHEN}", cells[1].question)
    with Capture() as capture:
        pipeline.submit(job, cells, job.priority)
    for cell in cells:
        pipeline.PENDING.pop(cell.key, None)
    sent = [question for payload in capture.requests for question in payload["questions"].values()]
    check("jobs: the requests carry the one state, the 40 questions and the node's debias",
          capture.requests and all(p["state"] == "A small flat" and p["options"] == {"debias": True}
                                   for p in capture.requests) and sent == [cell.question for cell in cells],
          (len(capture.requests), len(sent)))
    empty = MATCH.job((KITCHEN, "", "", "", "", ""), 'YES_NO', items[:2], sources.default_set("yes_no"), False)
    amount = MATCH.job(("How heavy is it?", "", "", "", "", ""), 'AMOUNT', items[:1], sources.default_set("amount"),
                       True)
    check("jobs: no Context gives the state 'No additional context.'; Amount asks a score over the levels",
          {row.cell.state for row in empty.rows} == {"No additional context."}
          and amount.rows[0].cell.question == {"type": "score", "instructions": "Item: thing 0\nHow heavy is it?",
                                               "criteria": list(sources.AMOUNT_LEVELS)},
          amount.rows[0].cell.question)
    labels = csvfmt.parse_inline_set("metal: steel, iron; wood: planks", "one_of")
    job = CLASSIFY.job(("", "a workshop", "", "", "", ""), [Item("steel")], labels, True)
    check("jobs: Classify's template, its default question and a choice over the labels",
          job.rows[0].cell.question == {"type": "choice", "instructions": "Item: steel\nWhich label fits this item best?",
                                        "criteria": {"0": "metal: steel, iron", "1": "wood: planks"}}
          and job.rows[0].cell.state == "a workshop", job.rows[0].cell.question)


def labels_of(count):
    return csvfmt.parse_inline_set("; ".join(f"label {i}" for i in range(count)), "one_of")


def phase_limits():
    gn = ("", "", "", "", "", "")
    items = [lists.Item(f"item {i}") for i in range(250)]
    try:
        CLASSIFY.job(gn, items, labels_of(255), True)
        refused = "not refused"
    except evaluator.Unresolvable as error:
        refused = str(error)
    check("limits: 255 labels x 250 items (34 rows x 250 = 8 500 > 8 192) are refused by the row estimate",
          "8 500" in refused and "8 192" in refused, refused)
    cheaper = CLASSIFY.job(gn, items, labels_of(255), False)
    check("limits: without Balance Option Order the same table needs 17 x 250 = 4 250 rows and is planned",
          sum(row.cell.cost for row in cheaper.rows) == 4250)

    tree, node = make_tree("Refused", ClassifySpec.idname, ("Items", "Labels"))  # a typed value holds 1023 bytes
    store_outputs(tree, node, {})
    user("RF", tree, Items=", ".join(f"item {i}" for i in range(250)),
         Labels="; ".join(f"label {i}" for i in range(255)), Shift=0)
    with Capture() as capture:
        engine.run_pass()
    status = engine.STATUS.get(node.uid)
    check("limits: the node shows the error before anything is sent", not capture.keys(node) and not capture.requests
          and status is not None and status.icon == 'CANCEL' and "8 500" in status.text, status)

    def costs(count, debias, n=3):
        return [row.cell.cost for row in CLASSIFY.job(gn, items[:n], labels_of(count), debias).rows]

    got = (costs(20, True), costs(20, False), costs(16, True), costs(17, True), sum(costs(20, True, 5)))
    check("tournament: 20 labels cost 2 x (2 groups + 1) = 6 scored rows per item (3 without balancing; 16 labels 2, "
          "17 labels 6), so 5 items over 20 labels estimate 30 rows",
          got == ([6] * 3, [3] * 3, [2] * 3, [6] * 3, 30) and pipeline.cost(20, True) == 6, got)


# ------------------------------------------------------------------------------ the pass

def phase_pass_classify():
    """plan() -> Job -> resolve() with cached answers -> store -> inner.build(), per-element columns included."""
    folder = tempfile.mkdtemp(prefix="openact-match-")
    try:
        path = os.path.join(folder, "families.csv")
        shutil.copyfile(FAMILIES, path)
        tree, node = make_tree("ClassifyPass", ClassifySpec.idname)
        node.inputs["Items"].default_value = "steel, oak planks, marble, silk"
        node.inputs["Labels"].default_value = path
        obj = user("CP", tree, Shift=0)
        with Capture() as capture:
            engine.run_pass()
        outputs = [(s.name, s.bl_idname) for s in node.outputs]
        check("pass: the Labels CSV's numeric columns become payload outputs as soon as the node plans",
              outputs == [("Label", 'NodeSocketInt'), ("Confidence", 'NodeSocketFloat'),
                          ("roughness", 'NodeSocketFloat'), ("metallic", 'NodeSocketInt')], outputs)
        store_outputs(tree, node, {"Label": 'INT', "Confidence": 'FLOAT', "roughness": 'FLOAT', "metallic": 'INT'})
        job = CLASSIFY.plan(node, evaluator.find_uses()[tree.as_pointer()][0])
        check("pass: one choice question per item over the 4 labels, all 4 missing cells asked in one job",
              capture.keys(node) == {row.cell.key for row in job.rows} and len(job.rows) == 4
              and all(len(row.cell.question["criteria"]) == 4 for row in job.rows), len(capture.keys(node)))
        for row in job.rows:
            pipeline.remember(row.cell.key, {"probabilities": FAMILY_ANSWERS[row.name], "model": "fake"})
        with Capture() as capture:
            engine.run_pass()
        names = ("Label", "Confidence", "roughness", "metallic")
        got = read(obj, names)
        check("pass: cached answers become the table: Label and the payload per element, denormalised per row",
              got == {"Label": [0, 1, 2, 3, -1, -1], "Confidence": [0.85, 0.7, 0.8, 0.6, 0.0, 0.0],
                      "roughness": [0.35, 0.6, 0.8, 0.9, 0.0, 0.0], "metallic": [1, 0, 0, 0, 0, 0]}
              and not capture.keys(node), got)
        stored = S.load(node).entries
        check("pass: the stored entry holds rows.payload per row, and shape the payload headers and label names",
              len(stored) == 1 and stored[0].rows.payload == {"roughness": [0.35, 0.6, 0.8, 0.9], "metallic": [1, 0, 0, 0]}
              and stored[0].shape["payload"] == [["roughness", "FLOAT"], ["metallic", "INT"]]
              and stored[0].shape["labels"] == ["metal", "wood", "stone", "fabric"],
              stored and (stored[0].rows.payload, stored[0].shape))
        status = engine.STATUS[node.uid]
        check("pass: the status summarises the table", (status.icon, status.text) == (
            'CHECKMARK', "4 items · 4 labels used · 0 unsure"), status)
        words_key = stored[0].words_key
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(text.replace("0.35", "0.355"))
        with Capture() as capture:
            engine.run_pass()
        got = read(obj, ("roughness",))["roughness"]
        stored = S.load(node).entries
        check("pass: a numbers-only edit of the Labels CSV re-bakes the payload per row and asks nothing",
              got[:4] == [0.355, 0.6, 0.8, 0.9] and not capture.keys(node) and len(stored) == 1
              and stored[0].words_key == words_key, (got, len(capture.keys(node))))
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(text.replace("0.35", "0.355").replace("\nfabric,", "\n\nfabric,"))
        with Capture() as capture:
            engine.run_pass()
        icon, status = engine.STATUS[node.uid].icon, engine.STATUS[node.uid].text
        check("pass: a Labels CSV warning (a blank line) shows after the summary, and asks nothing",
              icon == 'ERROR' and status.startswith("4 items · 4 labels used · 0 unsure · 1 warning: families.csv: "
                                                    "Line 5 is blank") and not capture.keys(node),
              (icon, status))
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def rewire(tree, node, outputs):
    """store_outputs() for these outputs, in place of the Store Named Attribute chain the tree has."""
    for old in [n for n in tree.nodes if n.bl_idname in ("GeometryNodeStoreNamedAttribute", "NodeGroupOutput")]:
        tree.nodes.remove(old)
    store_outputs(tree, node, outputs)


def phase_draw_classify():
    """The sidebar reads answers_for() from a draw, which may never write the node: it plans nothing and reads the last
    pass's record. The pass syncs the payload outputs on its next visit, whether it plans then or reuses a plan."""
    folder = tempfile.mkdtemp(prefix="openact-draw-")
    try:
        path = os.path.join(folder, "families.csv")
        shutil.copyfile(FAMILIES, path)
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        tree, node = make_tree("ClassifyDraw", ClassifySpec.idname)
        node.inputs["Items"].default_value = "steel, oak planks, marble, silk"
        node.inputs["Labels"].default_value = "metal; wood; stone; fabric"
        obj = user("CD", tree, Shift=0)
        with Capture():
            engine.run_pass()
        check("draw: the stand-in draw refuses ID writes, as Blender's draw callbacks do",
              drawn(lambda: setattr(node, "label", "drawn"))[0] == "AttributeError" and node.label == "", node.label)

        node.inputs["Labels"].default_value = path  # typed: the sidebar may draw before the debounced pass
        before = face(node)
        got = drawn(lambda: engine.answers_for(node))
        check("draw: typing a Labels CSV path, then a draw: answers_for() raises nothing and writes nothing",
              got[0] == "ok" and face(node) == before, (got[0], face(node)[0]))
        with Capture():
            engine.run_pass()
        check("draw: the next pass plans and adds the payload outputs",
              face(node)[0][2:] == [("roughness", 'NodeSocketFloat', "The Labels CSV's 'roughness' of the chosen label"),
                                    ("metallic", 'NodeSocketInt', "The Labels CSV's 'metallic' of the chosen label")],
              face(node)[0])
        store_outputs(tree, node, {"Label": 'INT', "roughness": 'FLOAT', "metallic": 'INT'})
        for row in CLASSIFY.plan(node, evaluator.find_uses()[tree.as_pointer()][0]).rows:
            pipeline.remember(row.cell.key, {"probabilities": FAMILY_ANSWERS[row.name], "model": "fake"})
        with Capture():
            engine.run_pass()
        snapshot = node.entries  # what an undo to before the edits below restores

        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(text.replace("roughness,metallic", "roughness,metalness"))
        before = face(node)
        got = drawn(lambda: engine.answers_for(node))
        check("draw: after a Labels CSV header rename, a draw shows the answers and leaves the outputs, their links "
              "and the group as they were", got[0] == "ok" and got[1] and got[1][0][2] is not None
              and face(node) == before, (got[0], face(node)[:2]))
        with Capture() as capture:
            engine.run_pass()
        tips = {name: tip for name, _type, tip in face(node)[0]}
        rewire(tree, node, {"Label": 'INT', "roughness": 'FLOAT', "metalness": 'INT'})
        got = read(obj, ("Label", "roughness", "metalness"))
        check("draw: the next pass renames the payload output, with its tooltip, and asks nothing",
              [s.name for s in node.outputs] == ["Label", "Confidence", "roughness", "metalness"]
              and tips.get("metalness") == "The Labels CSV's 'metalness' of the chosen label"
              and got == {"Label": [0, 1, 2, 3, -1, -1], "roughness": [0.35, 0.6, 0.8, 0.9, 0.0, 0.0],
                          "metalness": [1, 0, 0, 0, 0, 0]} and not capture.keys(node), (tips, got))

        with open(path, "w", encoding="utf-8", newline="") as handle:  # the last column removed
            handle.write("\n".join(line.rsplit(",", 1)[0] for line in text.splitlines()) + "\n")
        before = face(node)
        got = drawn(lambda: engine.answers_for(node))
        check("draw: after a Labels CSV column is removed, a draw writes nothing either",
              got[0] == "ok" and face(node) == before, (got[0], face(node)[:2]))
        with Capture() as capture:
            engine.run_pass()
        got = read(obj, ("Label", "roughness"))
        check("draw: the next pass drops the payload output and rebuilds the group: each element reads its own label",
              [s.name for s in node.outputs] == ["Label", "Confidence", "roughness"]
              and got == {"Label": [0, 1, 2, 3, -1, -1], "roughness": [0.35, 0.6, 0.8, 0.9, 0.0, 0.0]}
              and not capture.keys(node), got)

        node.entries = snapshot  # undo: the node and its group as they were; the file and the plans stay
        inner.sync_interface(node, CLASSIFY.interface(node, PAYLOAD))
        inner.build(node, S.load(node))
        with Capture():
            engine.run_pass()
        got = read(obj, ("Label", "roughness"))
        check("draw: after an undo that keeps the plans, the next pass syncs the payload outputs to the file again",
              [s.name for s in node.outputs] == ["Label", "Confidence", "roughness"]
              and got == {"Label": [0, 1, 2, 3, -1, -1], "roughness": [0.35, 0.6, 0.8, 0.9, 0.0, 0.0]},
              ([s.name for s in node.outputs], got))
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def phase_agree_classify():
    """§6.0: the contexts of one Classify must read the same payload headers and types, in any order. Labels CSVs that
    share them with other values each show their own. Other headers or types give the error naming both contexts,
    and nothing is asked, stored or rebuilt: the outputs, the links to them and the last good table stay, also while
    either CSV has an error (§7.8). A context without a job (a CSV with errors, an item-list error, a Context that
    waits for a Pick) keeps the outputs and their links as they are, and another context's new words that fit them
    are still asked and shown."""
    folder = tempfile.mkdtemp(prefix="openact-agree-")
    try:
        with open(FAMILIES, encoding="utf-8") as handle:
            text = handle.read()
        paths = [os.path.join(folder, name) for name in ("a.csv", "b.csv")]
        bodies = dict(zip(paths, (text, text.replace("0.35,1", "0.4,1"))))  # the same headers, other values

        def write(path, body):
            with open(path, "w", encoding="utf-8", newline="") as handle:
                handle.write(body)

        for path, body in bodies.items():
            write(path, body)
        items = "steel, oak planks, marble, silk"
        tree, node = make_tree("ClassifyAgree", ClassifySpec.idname, ("Labels", "Items"))
        tree.interface.new_socket("Where", in_out='INPUT', socket_type='NodeSocketString')
        pick = tree.nodes.new(engine.PICK_ID)  # its answer is the Context: while it has none, the context waits
        pick.inputs["Items"].default_value = "a museum, a kitchen"
        tree.links.new(next(n for n in tree.nodes if n.bl_idname == "NodeGroupInput").outputs["Where"],
                       pick.inputs["Prompt"])
        tree.links.new(pick.outputs["Item"], node.inputs["Context"])
        objs = [user(name, tree, Labels=path, Items=items, Where="Where is it?", Shift=0)
                for name, path in zip(("CGA", "CGB"), paths)]
        uses = evaluator.find_uses()[tree.as_pointer()]
        idents = {i.name: i.identifier for i in tree.interface.items_tree if getattr(i, "in_out", "") == 'INPUT'}
        answers = dict(FAMILY_ANSWERS, granite=[0.05, 0.05, 0.85, 0.05])

        def use(obj, **words):
            for word, value in words.items():
                obj.modifiers["OpenAct"][idents[word]] = value
            obj.update_tag()

        def picked(ctx):
            """The Pick answers "a museum" to the prompt ctx has now."""
            cell = next(row.cell for row in engine.SPECS[engine.PICK_ID].plan(pick, ctx).rows if row.cell is not None)
            pipeline.remember(cell.key, {"probabilities": [0.9, 0.1], "model": "fake"})

        def answered(ctx):
            for row in CLASSIFY.plan(node, ctx).rows:
                pipeline.remember(row.cell.key, {"probabilities": answers[row.name], "model": "fake"})

        picked(uses[0])  # both objects have the same prompt
        with Capture():
            engine.run_pass()
        for ctx in uses:
            answered(ctx)
        with Capture():
            engine.run_pass()
        store_outputs(tree, node, {"Label": 'INT', "roughness": 'FLOAT', "metallic": 'INT'})
        names, summary = ("Label", "roughness", "metallic"), "4 items · 4 labels used · 0 unsure   +1 more"
        before = [read(obj, names) for obj in objs]
        check("agreement: two contexts whose Labels CSVs share the payload headers, with other values, each read "
              "their own payload", before == [{"Label": [0, 1, 2, 3, -1, -1], "metallic": [1, 0, 0, 0, 0, 0],
                                               "roughness": [0.35, 0.6, 0.8, 0.9, 0.0, 0.0]},
                                              {"Label": [0, 1, 2, 3, -1, -1], "metallic": [1, 0, 0, 0, 0, 0],
                                               "roughness": [0.4, 0.6, 0.8, 0.9, 0.0, 0.0]}]
              and (engine.STATUS[node.uid].icon, engine.STATUS[node.uid].text) == ('CHECKMARK', summary),
              (before, engine.STATUS.get(node.uid)))
        first, second = (CLASSIFY.gn(node, ctx)[3] for ctx in uses)  # first-wins would follow the first's CSV
        stored, shown = [(entry.gn, entry.words_key) for entry in S.load(node).entries], face(node)

        def after(change):
            """Make a change and run a pass: (status, cells asked, whether the entries, the outputs, their links and
            the group stayed, what both objects read)."""
            change()
            with Capture() as capture:
                engine.run_pass()
            kept = [(entry.gn, entry.words_key) for entry in S.load(node).entries] == stored and face(node) == shown
            status = engine.STATUS.get(node.uid)
            return ((status.icon, status.text) if status else None, len(capture.keys(node)), kept,
                    [read(obj, names) for obj in objs])

        def edited(path, body):
            """Write a context's CSV and run a pass: after()."""
            return after(lambda: write(path, body))

        def broken(path):
            return ('CANCEL', f"{os.path.basename(path)}: Two columns are named 'Answer'"), 0, True, before

        labels = [ctx.label() for ctx in uses]
        conflict = "Classify: '{}' and '{}' use Labels CSVs with different payload columns ({})".format
        headers = (('CANCEL', conflict(*labels, "'hardness' (INT) vs 'roughness' (FLOAT), 'metallic' (INT)")), 0, True,
                   before)
        hardness = "answer,when,hardness\n" + "".join(f"{line.rsplit(',', 2)[0]},{value}\n" for line, value
                                                      in zip(text.splitlines()[1:], (9, 3, 7, 1)))
        got = edited(first, hardness)
        check("agreement: a Labels CSV with other payload columns gives the error naming both contexts; nothing is "
              "asked, stored or rebuilt, and the outputs, their links and both objects' values stay", got == headers,
              got)
        got = edited(second, "answer,Answer\nx,y\n"), edited(second, bodies[second])
        check("agreement: while the CSV whose columns the outputs show has an error, the outputs, their links and both "
              "objects' values stay; readable again, the error is back", got == (broken(second), headers), got)
        got = edited(first, "answer,Answer\nx,y\n")
        check("agreement: while the other CSV has an error, the outputs, their links and both objects' values stay",
              got == broken(first), got)
        got = edited(first, text.replace("0.35,1", "0.35,1.5"))
        check("agreement: readable again, with a payload column of another type: the error names it",
              got == (('CANCEL', conflict(*labels, "'metallic' (FLOAT) vs 'metallic' (INT)")), 0, True, before), got)
        got = edited(first, bodies[first])
        check("agreement: once both agree again, each object reads its own payload with its links, 0 requests",
              got == (('CHECKMARK', summary), 0, True, before), got)

        # the hold is on the outputs, not on the answers: another context's new words that fit them are answered
        os.remove(first)
        waited = after(lambda: use(objs[1], Items=items + ", granite"))
        answered(uses[1])
        got = after(lambda: None)
        granite = {"Label": [0, 1, 2, 3, 2, -1], "metallic": [1, 0, 0, 0, 0, 0],
                   "roughness": [0.4, 0.6, 0.8, 0.9, 0.8, 0.0]}
        check("agreement: while a Labels CSV is missing, another context's new item is asked, and shown once answered; "
              "the outputs keep their links, and the object whose CSV is missing its last table",
              waited[0] == got[0] == ('CANCEL', f"Can't find {os.path.normpath(first)}") and waited[1] == 1
              and got[1] == 0 and face(node)[:2] == shown[:2] and got[3] == [before[0], granite], (waited, got))
        got = after(lambda: (write(first, bodies[first]), use(objs[1], Items=items)))
        check("agreement: once the CSV is back, both contexts show their tables again with a checkmark, 0 requests",
              got[0] == ('CHECKMARK', summary) and got[1] == 0 and got[3] == before, got)
        stored, shown = [(entry.gn, entry.words_key) for entry in S.load(node).entries], face(node)
        got = edited(first, hardness)
        check("agreement: the other payload columns give the error again", got == headers, got)
        got = after(lambda: use(objs[1], Items="steel, steel, marble, silk")), after(lambda: use(objs[1], Items=items))
        check("agreement: while the context whose columns the outputs show has an item-list error, the outputs, their "
              "links, the entries and the other object's values stay; fixed, the error is back",
              got[0][:3] == (('CANCEL', "Items: Entry 2: 'steel' appears twice"), 0, True)
              and got[0][3][0] == before[0] and got[1] == headers, got)
        got = after(lambda: use(objs[1], Where="Where is it now?"))
        picked(uses[1])
        got = got, after(lambda: None)
        check("agreement: while that context's Context waits for the Pick, the outputs, their links, the entries and "
              "the other object's values stay; answered, the error is back",
              got[0][:3] == (('PAUSE', "Model server is stopped (OpenAct tab)"), 0, True)
              and got[0][3][0] == before[0] and got[1] == headers, got)
        reordered = "".join(f"{head},{metallic},{roughness}\n" for head, roughness, metallic
                            in (line.rsplit(",", 2) for line in text.splitlines()))
        got = edited(first, reordered)
        check("agreement: the same payload columns in another order agree: each object reads its own payload, the "
              "outputs follow the first context's order and keep their links, 0 requests",
              got[0] == ('CHECKMARK', summary) and got[1] == 0 and got[3] == before and face(node)[1] == shown[1]
              and [s.name for s in node.outputs] == ["Label", "Confidence", "metallic", "roughness"],
              (got, [s.name for s in node.outputs]))
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def phase_pass_match():
    palette = bpy.data.objects.new("Palette", bpy.data.meshes.new("Palette"))
    for name in ("Steel", None, "Oak"):
        palette.data.materials.append(bpy.data.materials.new(name) if name else None)
    tree, node = make_tree("MatchPass", MatchSpec.idname)
    node.list_source = 'MATERIALS'
    node.inputs["Object"].default_value = palette
    node.inputs["Question"].default_value = "Is this material metal?"
    store_outputs(tree, node, {"Probability": 'FLOAT', "Selection": 'BOOLEAN', "Confidence": 'FLOAT'})
    obj = user("MP", tree, Shift=0)
    check("pass: a new Match has Threshold 0.5, and the Materials source swaps Items for Object",
          node.inputs["Threshold"].default_value == 0.5 and "Items" not in node.inputs and "Object" in node.inputs,
          [s.name for s in node.inputs])
    with Capture() as capture:
        engine.run_pass()
    job = MATCH.plan(node, evaluator.find_uses()[tree.as_pointer()][0])
    check("pass: the empty material slot is a fallback row, never asked; the two materials are",
          [(row.name, row.source, row.cell is None) for row in job.rows]
          == [("Steel", "pending", False), ("", "code", True), ("Oak", "pending", False)]
          and capture.keys(node) == {job.rows[0].cell.key, job.rows[2].cell.key}, len(capture.keys(node)))
    pipeline.remember(job.rows[0].cell.key, {"p": 0.9, "model": "fake"})
    pipeline.remember(job.rows[2].cell.key, {"p": 0.1, "model": "fake"})
    with Capture() as capture:
        engine.run_pass()
    got = read(obj, ("Probability", "Selection", "Confidence"))
    check("pass: ID = material index; the empty slot reads the fallbacks and Selection uses Threshold 0.5",
          got == {"Probability": [0.9, 0.0, 0.1, 0.0, 0.0, 0.0], "Selection": [True] + [False] * 5,
                  "Confidence": [0.8, 0.0, 0.8, 0.0, 0.0, 0.0]} and not capture.keys(node), got)
    status = engine.STATUS[node.uid]
    check("pass: the status counts the materials only", (status.icon, status.text) == (
        'CHECKMARK', "1 of 2 yes · 0 unsure"), status)


# ------------------------------------------------------------------------------ the real model

def phase_model(port, gpu):
    for obj in list(bpy.data.objects):  # only this scene is asked
        bpy.data.objects.remove(obj)
    for group in list(bpy.data.node_groups):
        bpy.data.node_groups.remove(group)
    check("model: runtime detected", P.runtime_installed(P.runtime_root(PREFS)), P.runtime_root(PREFS))
    PREFS.auto_start = True
    tree = bpy.data.node_groups.new("Model", 'GeometryNodeTree')
    tree.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    tree.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    match = tree.nodes.new(MatchSpec.idname)
    match.inputs["Question"].default_value = KITCHEN
    match.inputs["Items"].default_value = "pan, sofa"
    classify = tree.nodes.new(ClassifySpec.idname)
    classify.inputs["Items"].default_value = "steel, oak planks"
    classify.inputs["Labels"].default_value = FAMILIES
    geometry = tree.nodes.new("NodeGroupInput").outputs["Geometry"]
    geometry = store(tree, geometry, match.outputs["Probability"], "kitchen", 'FLOAT')
    geometry = store(tree, geometry, classify.outputs["Label"], "family", 'INT')
    tree.links.new(geometry, tree.nodes.new("NodeGroupOutput").inputs[0])
    obj = user("Model", tree)
    engine.schedule(immediate=True)
    started = time.time()
    finished = drive(engine, lambda: not pipeline.PENDING and all(
        getattr(engine.STATUS.get(n.uid), "icon", "") == 'CHECKMARK' for n in (match, classify)), timeout=300)
    for n in (match, classify):
        print(f"  {n.name}: {engine.STATUS.get(n.uid)}", flush=True)
    check("model: both nodes answered", finished, f"{time.time() - started:.1f}s, server {server.status()}")
    got = read(obj, ("kitchen", "family"))
    print("model:", got, flush=True)
    kitchen, family = got["kitchen"] or [None, None], got["family"] or [None, None]
    check(f"model: pan = yes and sofa = no for '{KITCHEN}'",
          None not in kitchen[:2] and kitchen[0] >= 0.5 > kitchen[1], kitchen[:2])
    check("model: 'steel' -> metal (0) and 'oak planks' -> wood (1)", family[:2] == [0, 1], family[:2])
    server.stop()
    check("model: server stopped", not server.is_ready())


def main():
    print("BLENDER", bpy.app.version_string, f"model on port {MODEL[0]}, GPU {MODEL[1]}" if MODEL else "no model",
          flush=True)
    phase_match_table()
    phase_classify_table()
    phase_jobs()
    phase_limits()
    phase_pass_classify()
    phase_draw_classify()
    phase_agree_classify()
    phase_pass_match()
    if MODEL is not None:
        phase_model(*MODEL)
    finish()


main()