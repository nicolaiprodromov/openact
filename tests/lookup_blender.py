"""The inner group and table object built from a node's entries, with no server: the hash lookup (each entry found by
the key of its words among the table's header rows, and the default a miss shows), single values and per-element rows
of all six nodes, values-only updates, appends and rebuilds, more than 16 word sets on one node, words whose keys
collide, and per-element values with an unlinked ID in a second Blender, without the add-on (G6). With --bench, also
the main-thread budgets of §8.7, as BENCH lines: inner.build(), update_values() and append_entry() on the demo shapes
and the worst ones the limits allow; and, recorded without a gate, the planning cost of a 4096-cell context (§5.3). As
§18.2 checks timings, a measurement above its budget prints a WARN line and fails only above twice the budget.

Run:  blender -b --factory-startup --python tests/lookup_blender.py [-- --bench]
      OPENACT_BENCH_SHAPE=demo|worst limits --bench to one set of shapes (default: both).
"""

import contextlib
import dataclasses
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import ROOT, check, finish, sock, store, trigger  # noqa: E402
from openact import engine, evaluator, inner, lists, nodes, ops, pipeline, tables, words  # noqa: E402  (bpy first)
from openact import prefs as P  # noqa: E402
from openact import store as S  # noqa: E402  (store() is the harness's attribute writer)
from openact.specs.base import Spec  # noqa: E402
from openact.specs.classify import ClassifySpec  # noqa: E402
from openact.specs.dials import DialsSpec  # noqa: E402
from openact.specs.match import MatchSpec  # noqa: E402
from openact.specs.rules import RulesSpec  # noqa: E402

P._Defaults.auto_start = False  # nothing here may start a model server
nodes.register()
ops.register()
ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
MANY = S.MAX_ENTRIES  # the warm entries a node keeps (§14.7): as many blocks in the bench's largest tables
DIALS_MAX_SETTINGS = 32  # §7.5
BUDGET_MS = {"build demo": 100, "build demo interpolated": 250, "build worst": 1000,  # §8.7
             "update_values": 10, "update_values worst": 100, "append": 5, "append worst": 20}

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
for key, given in WORDS.items():
    mesh = bpy.data.meshes.new(key)
    mesh.from_pydata([(0, 0, 0)], [], [])
    obj = objects[key] = bpy.data.objects.new(key, mesh)
    bpy.context.scene.collection.objects.link(obj)
    mod = obj.modifiers.new("Lookup", 'NODES')
    mod.node_group = tree
    for name, value in given.items():
        mod[ids[name]] = value


def ask_entry(key, answer, confidence):
    return S.Entry(words_key=key, gn=(WORDS[key]["Question"], "", "", "", ""), kind="ask", answer_type='YES_NO',
                   single={"Answer": answer, "Yes": answer >= 0.5, "Confidence": confidence})


def pick_entry(key, index, confidence):
    words = WORDS[key]
    names = [name.strip() for name in words["Items"].split(",")]
    rows = S.Rows(name=names, wkey=[""] * len(names), ask=[0] * len(names), src="m" * len(names),
                  v={"prob": [float(r == index) for r in range(len(names))]})
    return S.Entry(words_key=key, gn=(words["Prompt"], "", words["Items"], "", ""), kind="pick", answer_type='ONE_OF',
                   asks=["a"], rows=rows, single={"Index": index, "Item": names[index], "Confidence": confidence})


def asked(question, p, t, key=""):
    """The entry the pass stores once an Ask whose only words are this Question and Key is answered p, last used at
    t."""
    spec = engine.SPECS[engine.ASK_ID]
    job = spec.job((question, "", "", "", key), 'YES_NO', True)
    rows = spec.rows(job, {job.rows[0].cell.key: {"p": p, "model": "fake"}})
    return dataclasses.replace(spec.entry(job, rows, "fake"), t=t)


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
          and placed["GeometryNodeAttributeStatistic"] == (250, 200), placed)

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
    added = inner.update_values(ask, grown)
    got = {key: read(key) for key in WORDS}
    check("update_values: a new entry is new rows of the table, values only: the same nodes, and its context reads it",
          added is True and pointers(ask) == before[0] and got["C"]["answer"] == 1.0 and got["C"]["yes"] is True,
          (added, got["C"]))

    def broken(*_args, **_kwargs):
        raise ValueError("the table can't be written")

    write_rows, tables.write_rows = tables.write_rows, broken
    try:
        inner.build(ask, grown)
        raised = False
    except ValueError:
        raised = True
    finally:
        tables.write_rows = write_rows
    check("build: after a build that raised, update_values() refuses even the entries built before it",
          raised and inner.update_values(ask, grown) is False, raised)
    inner.build(ask, grown)
    inner.build(pick, picks)
    check("build: a rebuild from the same entries leaves every output unchanged",
          {key: read(key) for key in WORDS} == got, got)


def phase_positions():
    """The key text holds each word at its position (§8.2): a text Pick's entry is found only by its prompt and Items
    text, a collection Pick's entry only by its prompt with no Items text. An entry with too few words, not a tuple the
    node makes, is found by none, and the entries after it still are."""
    collection = S.Entry(words_key="B", gn=(WORDS["B"]["Prompt"], "", "", "", ""), kind="pick", answer_type='ONE_OF',
                         single={"Index": 0, "Item": "book", "Confidence": 0.5})
    short = S.Entry(words_key="short", gn=(WORDS["C"]["Prompt"], WORDS["C"]["Items"]), kind="pick",
                    answer_type='ONE_OF', single={"Index": 1, "Item": "pear", "Confidence": 0.5})
    inner.build(pick, S.NodeStore([pick_entry("A", 1, 0.5), collection, short, pick_entry("C", 0, 0.5)]))
    got = {key: read(key)["index"] for key in WORDS}
    mod = objects["C"].modifiers["Lookup"]
    for key in ("A", "B"):  # C with that prompt and no Items text: a collection Pick's words
        mod[ids["Prompt"]], mod[ids["Items"]] = WORDS[key]["Prompt"], ""
        objects["C"].update_tag()
        got[f"C with {key}'s prompt"] = read("C")["index"]
    mod[ids["Prompt"]], mod[ids["Items"]] = WORDS["C"]["Prompt"], WORDS["C"]["Items"]
    objects["C"].update_tag()
    check("word positions: each entry is found by its own words at their positions, an entry with too few words by "
          "none, and the entries after it still are",
          got == {"A": 1, "B": -1, "C": 0, "C with A's prompt": -1, "C with B's prompt": 0}, got)


@contextlib.contextmanager
def asking(node_, questions):
    """The cell keys that the passes inside ask for node_ go into `questions`; nothing reaches a server."""
    submit = engine._submit
    engine._submit = lambda job, missing: questions.extend(cell.key for cell in missing if job.uid == node_.uid)
    try:
        yield
    finally:
        engine._submit = submit


def phase_pass():
    """The pass works on Jobs by the validity rule (§5.4) and keeps entries through store.py. A stored entry whose
    words key holds is reused, and a cached answer becomes a new entry; a new PROMPT_VERSION or Balance Option Order
    asks nothing, and a new Key carries every row over. Schema 2 is written, the live set it passes to save() keeps
    a live entry that eviction would otherwise remove, and a newer OpenAct's JSON is left as it is."""
    live = bpy.data.node_groups.new("Pass", 'GeometryNodeTree')
    live.interface.new_socket("Question", in_out='INPUT', socket_type='NodeSocketString')
    node_ = live.nodes.new(engine.ASK_ID)
    live.links.new(live.nodes.new("NodeGroupInput").outputs["Question"], node_.inputs["Question"])
    for name in ("Old words", "New words"):
        user = bpy.data.objects.new(name, bpy.data.meshes.new(name))
        user.modifiers.new("Pass", 'NODES').node_group = live
        user.modifiers["Pass"][live.interface.items_tree["Question"].identifier] = name
    spec = engine.SPECS[engine.ASK_ID]
    jobs = {ctx.obj.name: spec.plan(node_, ctx) for ctx in evaluator.find_uses()[live.as_pointer()]}
    old_rows = spec.rows(jobs["Old words"], {jobs["Old words"].rows[0].cell.key: {"p": 0.25, "model": "fake"}})
    old = dataclasses.replace(spec.entry(jobs["Old words"], old_rows, "fake"), t=1.0)
    warm = [S.Entry(words_key=f"warm {i}", gn=(f"Warm {i}?", "", "", "", ""), t=10.0 + i, kind="ask",
                    answer_type='YES_NO', single={"Answer": 0.5, "Yes": True, "Confidence": 0.0})
            for i in range(S.MAX_ENTRIES + 2)]
    S.save(node_, S.NodeStore([old] + warm))
    new_cell = jobs["New words"].rows[0].cell.key
    pipeline.remember(new_cell, {"p": 0.875, "model": "fake"})
    try:
        engine.run_pass()
    finally:
        pipeline.forget([new_cell])
    saved = S.load(node_).entries
    check("pass: a new answer is stored, and eviction keeps the oldest entry because its words are live",
          node_.entries.startswith('{"schema":2,') and [e.gn[0] for e in saved[:1] + saved[-1:]] == ["Old words", "New words"]
          and [e.gn[0] for e in saved[1:-1]] == [f"Warm {i}?" for i in range(2, S.MAX_ENTRIES + 2)],
          [e.gn[0] for e in saved])
    check("pass: the table object holds every stored entry, a header row each, and a values-only update keeps it",
          engine.STATUS[node_.uid].icon == 'CHECKMARK'
          and inner.update_values(node_, S.load(node_), [job.gn for job in jobs.values()])
          and len(tables.header_rows(node_)) == len(saved), (engine.STATUS[node_.uid], len(saved)))

    before, questions = node_.entries, []
    with asking(node_, questions):
        Spec.PROMPT_VERSION += 1
        node_.balance_order = False
        try:
            engine.run_pass()
        finally:
            Spec.PROMPT_VERSION -= 1
            node_.balance_order = True
    check("pass: a new PROMPT_VERSION and Balance Option Order ask nothing and keep every answer",
          not questions and node_.entries == before and engine.STATUS[node_.uid].icon == 'CHECKMARK', len(questions))
    node_.inputs["Key"].default_value = "left"
    with asking(node_, questions):
        engine.run_pass()
    keyed = {e.gn[0]: e.single["Answer"] for e in S.load(node_).entries if e.gn[-1] == "left"}
    check("pass: a new Key carries every row over from what each context showed, and asks nothing",
          not questions and keyed == {"Old words": 0.25, "New words": 0.875}, (len(questions), keyed))
    node_.inputs["Key"].default_value = ""

    S.save(node_, S.NodeStore([old]))  # only the live words' entries, which Ask Again then forgets
    pipeline.remember(new_cell, {"p": 0.875, "model": "fake"})
    try:
        with asking(node_, questions):
            engine.run_pass()
    finally:
        pipeline.forget([new_cell])
    saved = [(e.gn, e.single["Answer"]) for e in S.load(node_).entries]
    check("pass: a stored entry answers its words and a cached answer the others, with 0 requests, written as schema 2",
          node_.entries.startswith('{"schema":2,') and not questions
          and saved == [(("Old words", "", "", "", ""), 0.25), (("New words", "", "", "", ""), 0.875)],
          (questions, saved))
    engine.reask(node_)
    check("Ask Again: the node's answers are forgotten, saved as schema 2, and its table holds none",
          node_.entries == '{"schema":2,"entries":[]}' and inner.update_values(node_, S.NodeStore())
          and not tables.header_rows(node_), node_.entries)

    newer = '{"schema":3,"entries":[{"written by":"a newer OpenAct"}]}'
    node_.entries = newer
    group, asked = pointers(node_), []
    submit, engine._submit = engine._submit, lambda job, _missing: asked.append(job.uid)
    try:
        engine.run_pass()
    finally:
        engine._submit = submit
    status = engine.STATUS[node_.uid]
    check("pass: a newer OpenAct's answers show an error, and are not asked for, written over or rebuilt",
          node_.entries == newer and node_.uid not in asked and pointers(node_) == group
          and (status.icon, status.text) == ('CANCEL', S.newer_message(S.load(node_))), status)
    try:
        bpy.ops.openact.reask(uid=node_.uid)
        shown = ""
    except RuntimeError as error:  # how an operator's ERROR report reaches a Python caller
        shown = str(error).strip()
    check("Ask Again: a newer OpenAct's answers are kept, and the operator says why",
          node_.entries == newer and "saved by a newer OpenAct" in shown and "Traceback" not in shown, shown)


def phase_pick_pass():
    """Pick through the pass (§6.2, §8.2, §6.0): a Collection-source Pick matches its stored entry, the absent Items
    joined as a constant ""; Probability at the picked ID is the stored top probability; switching back to a stored
    prompt asks nothing; two contexts that differ only by collection get "only one answer can show" until each has
    its own Key, and then the context already answered asks nothing."""
    group = bpy.data.node_groups.new("PickPass", 'GeometryNodeTree')
    for name, socket_type in (("Geometry", 'NodeSocketGeometry'), ("Prompt", 'NodeSocketString'),
                              ("Collection", 'NodeSocketCollection'), ("Key", 'NodeSocketString')):
        group.interface.new_socket(name, in_out='INPUT', socket_type=socket_type)
    group.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    gi, go = group.nodes.new("NodeGroupInput"), group.nodes.new("NodeGroupOutput")
    picker = group.nodes.new(engine.PICK_ID)
    picker.list_source = 'COLLECTION'  # swaps Items for Collection
    for name in ("Prompt", "Collection", "Key"):
        group.links.new(gi.outputs[name], picker.inputs[name])
    geometry = gi.outputs["Geometry"]
    for name, output, data_type in (("prob", "Probability", 'FLOAT'), ("sel", "Selection", 'BOOLEAN'),
                                    ("idx", "Index", 'INT')):
        geometry = store(group, geometry, picker.outputs[output], name, data_type)
    group.links.new(geometry, go.inputs[0])
    trigger(picker)  # it asks only while its Trigger is on (§6.0)
    idents = {i.name: i.identifier for i in group.interface.items_tree if getattr(i, "in_out", "") == 'INPUT'}
    lists_ = {}
    for name, children in (("Shelf", ("book", "lamp", "vase")), ("Kitchen", ("cup", "pan", "plate"))):
        lists_[name] = bpy.data.collections.new(name)
        for child in children:
            lists_[name].objects.link(bpy.data.objects.new(child, None))
    users = {}

    def use(name, **words):
        if name not in users:
            mesh = bpy.data.meshes.new(name)
            mesh.from_pydata([(i, 0, 0) for i in range(4)], [], [])
            users[name] = bpy.data.objects.new(name, mesh)
            bpy.context.scene.collection.objects.link(users[name])
            users[name].modifiers.new("PickPass", 'NODES').node_group = group
        for word, value in words.items():
            users[name].modifiers["PickPass"][idents[word]] = value
        users[name].update_tag()

    def cell(name):
        ctx = next(c for c in evaluator.find_uses()[group.as_pointer()] if c.obj.name == name)
        return engine.SPECS[engine.PICK_ID].plan(picker, ctx).rows[0].cell.key

    def answer(name, probabilities):
        key = cell(name)
        pipeline.remember(key, {"probabilities": probabilities, "model": "fake"})
        return key

    light, flowers, questions = "something that gives light", "something to put flowers in", []
    use("P1", Prompt=light, Collection=lists_["Shelf"])
    shelf = answer("P1", [0.125, 0.75, 0.125])
    with asking(picker, questions):
        engine.run_pass()
    stored, got = S.load(picker).entries, table_read(users["P1"])
    check("pass, Pick (Collection): the answer is stored for words with Items \"\", and the inner group matches it",
          not questions and [e.gn for e in stored] == [(light, "", "", "", "")] and got["idx"] == [1] * 4
          and engine.STATUS[picker.uid].icon == 'CHECKMARK', (engine.STATUS[picker.uid], got))
    top = stored[0].rows.v["prob"] if stored else [0.0]
    check("pass, Pick: Probability at the picked ID is the stored top probability, and Selection marks that ID",
          got["prob"][1] == max(top) == 0.75 and got["prob"] == [0.125, 0.75, 0.125, 0.0]
          and got["sel"] == [False, True, False, False], got)

    use("P1", Prompt=flowers)
    vase = answer("P1", [0.0625, 0.0625, 0.875])
    with asking(picker, questions):
        engine.run_pass()
    edited = table_read(users["P1"])["idx"]
    pipeline.forget([shelf, vase])  # from here on only the stored entries can answer
    use("P1", Prompt=light)
    with asking(picker, questions):
        engine.run_pass()
    check("pass, Pick: switching back to a stored prompt sends 0 requests and shows its answer again",
          edited == [2] * 4 and not questions and table_read(users["P1"])["idx"] == [1] * 4, (edited, questions))

    use("P2", Prompt=light, Collection=lists_["Kitchen"])
    with asking(picker, questions):
        engine.run_pass()
    status = engine.STATUS[picker.uid]
    check("pass, Pick: two contexts that differ only by collection: \"only one answer can show\", naming the Key",
          status.icon == 'CANCEL' and "only one answer can show" in status.text and "Key" in status.text
          and not questions, status)
    use("P1", Key="shelf")
    use("P2", Key="kitchen")
    with asking(picker, questions):
        engine.run_pass()
    asked = list(questions)
    kitchen = answer("P2", [0.5, 0.25, 0.25])
    try:
        with asking(picker, questions):
            engine.run_pass()
    finally:
        pipeline.forget([kitchen])
    got = table_read(users["P1"])["idx"], table_read(users["P2"])["idx"]
    check("pass, Pick: with a Key each, both show their own answer; the context already answered sends 0 requests",
          asked == [kitchen] and questions == asked and got == ([1] * 4, [0] * 4)
          and engine.STATUS[picker.uid].icon == 'CHECKMARK', (asked == [kitchen], got, engine.STATUS[picker.uid]))


def phase_stored_default():
    """A single context whose words changed keeps showing its own last answer while it waits, a stored one too, never
    another stored entry's (§8.2). The words key "" names no entry, not even one stored without a words key."""
    group = bpy.data.node_groups.new("Stored", 'GeometryNodeTree')
    for name, socket_type in (("Geometry", 'NodeSocketGeometry'), ("Question", 'NodeSocketString')):
        group.interface.new_socket(name, in_out='INPUT', socket_type=socket_type)
    group.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    gi, go = group.nodes.new("NodeGroupInput"), group.nodes.new("NodeGroupOutput")
    asker = group.nodes.new(engine.ASK_ID)
    group.links.new(gi.outputs["Question"], asker.inputs["Question"])
    group.links.new(store(group, gi.outputs["Geometry"], asker.outputs["Answer"], "answer", 'FLOAT'), go.inputs[0])
    S.save(asker, S.NodeStore([asked("Is it night?", 0.875, 1.0), asked("Is it cold?", 0.125, 2.0)]))
    mesh = bpy.data.meshes.new("Stored")
    mesh.from_pydata([(0, 0, 0)], [], [])
    user = bpy.data.objects.new("Stored", mesh)
    bpy.context.scene.collection.objects.link(user)
    user.modifiers.new("Stored", 'NODES').node_group = group
    ident = group.interface.items_tree["Question"].identifier

    def shown(question):
        user.modifiers["Stored"][ident] = question
        user.update_tag()
        with asking(asker, []):  # nothing answers: edited words wait
            engine.run_pass()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        depsgraph.update()
        evaluated = user.evaluated_get(depsgraph)
        result = evaluated.to_mesh()
        try:
            return round(result.attributes["answer"].data[0].value, 6)
        finally:
            evaluated.to_mesh_clear()

    before, after = shown("Is it cold?"), shown("Is it cold outside?")
    check("stored entries: a single context whose words changed keeps showing its own stored answer, never the other "
          "stored entry's", before == 0.125 and after == 0.125, (before, after))
    live = S.Entry(words_key="live", gn=("Live?", "", "", "", ""))
    keyless = [S.Entry(words_key="", gn=(f"Keyless {i}?", "", "", "", ""), t=float(i)) for i in range(3)]
    inner.build(asker, S.NodeStore(keyless + [live]), [live.gn], "")
    default = tables.read(asker, tables.DEFAULT)
    check("entries without a words key: the default \"\" names none of them, so a miss shows the fallbacks",
          default is not None and default[0] == -1 and len(tables.header_rows(asker)) == 4, default)
    bpy.data.objects.remove(user)
    bpy.data.node_groups.remove(group)


def phase_table_only():
    """Rules in Yes/No mode declares only table-only columns, and before its first answer nothing samples them: the
    group still holds its table object, so the next pass is values only, with no new object and no pass loop."""
    group = bpy.data.node_groups.new("TableOnly", 'GeometryNodeTree')
    rules = group.nodes.new(RulesSpec.idname)
    rules.answer_type = 'YES_NO'
    rules.inputs["Question"].default_value = "Is it snowy?"
    rules.inputs["Factors"].default_value = os.path.join(ROOT, "tests", "fixtures", "biome_factors.csv")
    build, rebuilt = inner.build, []

    def recorded(node_, *args):
        rebuilt.append(node_ == rules)
        build(node_, *args)

    with asking(rules, []):
        engine.run_pass()  # plans: the factor sockets appear, and the cells wait for the model
        held = tables.find(rules)
        inner.build = recorded
        try:
            engine.run_pass()
        finally:
            inner.build = build
    check("table-only columns: an unanswered Rules Yes/No node holds its table object, and the next pass is values "
          "only", held is not None and tables.find(rules) == held and not any(rebuilt), (held, rebuilt))
    bpy.data.node_groups.remove(group)
    tables.collect_garbage(list(engine.iter_nodes()))  # its table goes now: phase_table_copies counts what goes


def user_of(group, name, points=1):
    """An object of `points` points in the scene whose modifier uses the group."""
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([(i, 0, 0) for i in range(points)], [], [])
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.modifiers.new(name, 'NODES').node_group = group
    return obj


def phase_carry():
    """Carry-over by wkey (§5.4) in a new session, where only the stored entry can answer: a text edit of a Match's
    Items CSV (the same words) asks only the row whose wkey changed, and nothing more once its answer arrives.
    Meanwhile the table so far shows (stale-while-refilling), while the old entry stays stored, hidden: undoing the
    edit before the answer arrives asks nothing and shows the stored answers, also in a reopened file."""
    folder = tempfile.mkdtemp(prefix="openact-carry-")
    path = os.path.join(folder, "items.csv")

    def items(lamp, stamp, table="a flat surface"):  # each edit gets a new stamp, which the fingerprint holds
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(f"answer,when\nchair,a seat\ntable,{table}\nlamp,{lamp}\n")
        os.utime(path, ns=(stamp, stamp))

    group = bpy.data.node_groups.new("Carry", 'GeometryNodeTree')
    group.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    group.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    matcher = group.nodes.new(MatchSpec.idname)
    matcher.inputs["Question"].default_value = "Would you find it in a kitchen?"
    matcher.inputs["Items"].default_value = path
    geometry = store(group, group.nodes.new("NodeGroupInput").outputs["Geometry"], matcher.outputs["Probability"],
                     "prob", 'FLOAT')
    group.links.new(geometry, group.nodes.new("NodeGroupOutput").inputs[0])
    trigger(matcher)  # it asks only while its Trigger is on (§6.0)
    user = user_of(group, "Carry", 3)

    def cells():
        return [row.cell.key for row in
                engine.SPECS[MatchSpec.idname].plan(matcher, evaluator.find_uses()[group.as_pointer()][0]).rows]

    def keys():
        return [entry.words_key for entry in S.load(matcher).entries]

    items("gives light", 10 ** 18)
    old, questions, extra = cells(), [], []
    for key, p in zip(old, (0.875, 0.75, 0.25)):
        pipeline.remember(key, {"p": p, "model": "fake"})
    try:
        with asking(matcher, questions):
            engine.run_pass()
        stored, before = table_read(user)["prob"], keys()
        pipeline.forget(old)  # a new session, in which the CSV was edited while Blender was closed
        engine.on_load_post()
        items("gives light at night", 2 * 10 ** 18)  # only lamp's wkey changes
        with asking(matcher, questions):
            engine.run_pass()
        edited, meanwhile, lamp = list(questions), table_read(user)["prob"], cells()[2]
        status = engine.STATUS[matcher.uid]
        check("carry-over, new session: a text edit asks only the row whose wkey changed; meanwhile the carried rows "
              "show their answers and the edited one its old answer, marked stale (§5.4), and the old entry stays "
              "stored", stored == [0.875, 0.75, 0.25] and edited == [lamp] and meanwhile == stored and keys() == before
              and (status.rows, status.answered, status.stale, status.pending) == (3, 2, 1, 0),
              (stored, len(edited), meanwhile, keys() == before, (status.rows, status.answered, status.stale)))
        questions.clear()
        items("gives light", 3 * 10 ** 18)  # undone before lamp's answer arrives
        with asking(matcher, questions):
            engine.run_pass()
        check("carry-over: undoing the edit before its answer arrives asks nothing and shows the stored answers",
              not questions and table_read(user)["prob"] == stored, (len(questions), table_read(user)["prob"]))
        items("gives light at night", 4 * 10 ** 18)
        with asking(matcher, questions):
            engine.run_pass()
        pipeline.remember(lamp, {"p": 0.125, "model": "fake"})
        questions.clear()
        with asking(matcher, questions):
            engine.run_pass()
        check("carry-over: once the changed row is answered nothing more is asked, and the others keep their stored "
              "answers", not questions and table_read(user)["prob"] == [0.875, 0.75, 0.125],
              (len(questions), table_read(user)["prob"]))
        answered = keys()
        items("gives light", 5 * 10 ** 18, "a flat surface to eat at")  # table's answer is known, lamp's isn't
        extra.append(cells()[1])
        pipeline.remember(extra[0], {"p": 0.5, "model": "fake"})
        with asking(matcher, questions):
            engine.run_pass()
        partial = table_read(user)["prob"]
        engine.on_load_post()  # the file saved and reopened meanwhile
        items("gives light at night", 6 * 10 ** 18)  # undone
        questions.clear()
        with asking(matcher, questions):
            engine.run_pass()
        check("carry-over: undoing an edit whose table so far showed a new answer, in a reopened file, asks nothing "
              "and shows the stored answers again", partial == [0.875, 0.5, 0.125] and keys() == answered
              and not questions and table_read(user)["prob"] == [0.875, 0.75, 0.125],
              (partial, keys() == answered, len(questions), table_read(user)["prob"]))
    finally:
        pipeline.forget(old + cells() + extra)
        shutil.rmtree(folder, ignore_errors=True)


def phase_live_set():
    """The live set (§5.3 step 6, §8.1): a context whose Answers CSV now has errors keeps showing its last good table,
    so save() never evicts that entry, even when another context stores a new one in the same pass. Two contexts with
    the same words show one entry, and in a new session a new Key on either carries its rows over, with 0 requests."""
    folder = tempfile.mkdtemp(prefix="openact-live-")
    path = os.path.join(folder, "a.csv")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write("answer,when\nyes,it is dark\nno,it is light\n")
    group = bpy.data.node_groups.new("LiveSet", 'GeometryNodeTree')
    group.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    for name in ("Answers", "Key"):
        group.interface.new_socket(name, in_out='INPUT', socket_type='NodeSocketString')
    group.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    gi = group.nodes.new("NodeGroupInput")
    group.links.new(gi.outputs["Geometry"], group.nodes.new("NodeGroupOutput").inputs[0])
    asker = group.nodes.new(engine.ASK_ID)
    asker.inputs["Question"].default_value = "Is it night?"
    for name in ("Answers", "Key"):
        group.links.new(gi.outputs[name], asker.inputs[name])
    idents = {i.name: i.identifier for i in group.interface.items_tree if getattr(i, "in_out", "") == 'INPUT'}
    users = {name: user_of(group, name) for name in ("LA", "LB", "LC")}

    def use(name, **words):
        for word, value in words.items():
            users[name].modifiers[name][idents[word]] = value
        users[name].update_tag()

    def cell(name):
        ctx = next(c for c in evaluator.find_uses()[group.as_pointer()] if c.obj.name == name)
        return engine.SPECS[engine.ASK_ID].plan(asker, ctx).rows[0].cell.key

    use("LA", Answers=path, Key="")
    use("LB", Answers="", Key="")
    use("LC", Answers="", Key="")  # LB's words exactly
    keys = [cell("LA"), cell("LB")]
    for key in keys:
        pipeline.remember(key, {"p": 0.875, "model": "fake"})
    try:
        engine.run_pass()
        pipeline.forget(keys)  # a new session: only the stored entries can answer
        engine.on_load_post()
        engine.run_pass()
        asked = {}
        for name in ("LC", "LB"):
            questions = []
            use(name, Key=name.lower())
            with asking(asker, questions):
                engine.run_pass()
            asked[name] = len(questions)
            use(name, Key="")
            engine.run_pass()
        check("same words, new session: a new Key on either context carries its rows over, 0 requests",
              asked == {"LC": 0, "LB": 0}, asked)

        node_store = S.load(asker)
        errored = next(e.gn for e in node_store.entries if e.gn[2] == path)
        for entry in node_store.entries:
            entry.t = 1.0  # the least recently used: the first to go
        node_store.entries += [S.Entry(words_key=f"warm {i}", gn=(f"Warm {i}?", "", "", "", ""), t=10.0 + i, kind="ask",
                                       answer_type='YES_NO', single={"Answer": 0.5, "Yes": True, "Confidence": 0.0})
                               for i in range(S.MAX_ENTRIES + 1)]
        S.save(asker, node_store)
        with open(path, "w", encoding="utf-8", newline="") as handle:  # LA's CSV breaks while Blender is closed
            handle.write('name;;;\n"unclosed\n')
        engine.on_load_post()
        pipeline.remember(keys[1], {"p": 0.25, "model": "fake"})
        use("LB", Key="b")  # LB stores a new entry in the same pass
        engine.run_pass()
        kept = [e.gn for e in S.load(asker).entries]
        check("live set: the entry of a context whose CSV now has errors is never evicted, as it still shows",
              errored in kept and ("Warm 0?", "", "", "", "") not in kept and engine.STATUS[asker.uid].icon == 'CANCEL',
              (errored in kept, len(kept), engine.STATUS[asker.uid]))
    finally:
        pipeline.forget(keys)
        shutil.rmtree(folder, ignore_errors=True)


def phase_last_shown():
    """The default a miss shows names a stored entry by its words key: a single context showing one gets new words
    that no model answers, and keeps showing its own answer, never another question's (§8.2)."""
    group = bpy.data.node_groups.new("LastShown", 'GeometryNodeTree')
    group.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    group.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    asker = group.nodes.new(engine.ASK_ID)
    geometry = store(group, group.nodes.new("NodeGroupInput").outputs["Geometry"], asker.outputs["Answer"], "prob",
                     'FLOAT')
    group.links.new(geometry, group.nodes.new("NodeGroupOutput").inputs[0])
    user = user_of(group, "LastShown")
    S.save(asker, S.NodeStore([asked("Is it cold?", 0.125, 1.0), asked("Is it night?", 0.875, 2.0)]))
    got = []
    with asking(asker, []):
        for question in ("Is it night?", "Is it dark?"):
            asker.inputs["Question"].default_value = question
            engine.run_pass()
            got.append(table_read(user)["prob"][0])
    check("stored entries: a context shows its own, and with new words keeps showing it while they wait, never "
          "another question's stored answer", got == [0.875, 0.875], got)


def phase_statuses():
    """A Pick with nothing typed asks for its prompt first: its list's hint waits for plan(). A node that reads a newer
    OpenAct's answers as words says why, rather than waiting for them."""
    group = bpy.data.node_groups.new("Statuses", 'GeometryNodeTree')
    picker = group.nodes.new(engine.PICK_ID)
    reader = group.nodes.new(engine.ASK_ID)
    engine.run_pass()
    empty = engine.STATUS[picker.uid]
    picker.inputs["Prompt"].default_value = "something to sit on"
    picker.inputs["Items"].default_value = "table, chair"
    picker.entries = '{"schema":3,"entries":[]}'
    group.links.new(picker.outputs["Item"], reader.inputs["Question"])
    with asking(reader, []):
        engine.run_pass()
    status = engine.STATUS[reader.uid]
    check("status: an empty Pick asks for its prompt first", (empty.icon, empty.text) == ('INFO', "Type a prompt"),
          empty)
    check("status: a node that reads a newer OpenAct's answers as words says why, rather than waiting",
          status.icon == 'CANCEL' and "newer OpenAct" in status.text, status)


# ------------------------------------------------------------------ all six nodes, appends, no cap, collisions

SIX_WORDS = ("first", "second", "third")  # each object's Word in six_scene(); entry i holds SIX_VALUES[i]
SIX_ITEMS = ["table", "chair", "lamp"]
SIX_READS = ("ask", "index", "item", "prob", "match", "label", "outcome", "rain", "wind")  # six_scene()'s attributes
LIT = {"name": "lit", "kind": "flag", "levels": ["dark", "bright"], "values": [0, 1], "edges": None}  # declare()d
SETTINGS = [{"name": "Rain", "type": "yes_no", "min": 0.0, "max": 1.0},
            {"name": "Wind", "type": "amount", "min": 0.0, "max": 10.0}]
SIX_VALUES = (  # per entry: Answer, Index, prob, Match p, Label, Outcome, Dials values (Rain p, Wind 0-1)
    (0.25, 1, [0.125, 0.75, 0.125], [0.875, 0.125], [0, 1], [0, 1], [0.75, 0.5]),
    (0.75, 2, [0.0625, 0.25, 0.6875], [0.25, 0.625], [1, 0], [1, 0], [0.25, 0.25]),
    (0.625, 0, [0.5, 0.25, 0.25], [0.375, 0.5], [1, 1], [1, 1], [0.875, 0.75]))


def enabled(sockets, name):
    return next(s for s in sockets if s.name == name and s.enabled)


def six_scene():
    """One tree of all six nodes, used by three objects of 2 points: each node's first word comes from the modifier's
    Word, its other words are typed, and each output is a point attribute (Pick's Item as its length). Rules has one
    flag factor, lit, read from the points' own attribute (False, True); Dials has the settings Rain and Wind."""
    group = bpy.data.node_groups.new("Six", 'GeometryNodeTree')
    for name, socket_type in (("Geometry", 'NodeSocketGeometry'), ("Word", 'NodeSocketString')):
        group.interface.new_socket(name, in_out='INPUT', socket_type=socket_type)
    group.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    gi = group.nodes.new("NodeGroupInput")
    made = [group.nodes.new(idname) for idname in (engine.ASK_ID, engine.PICK_ID, MatchSpec.idname,
                                                    ClassifySpec.idname, RulesSpec.idname, DialsSpec.idname)]
    asker, picker, matcher, classifier, rules, dial = made
    inner.sync_interface(rules, engine.SPECS[RulesSpec.idname].interface(rules, [("lit", "flag")], []))
    inner.sync_interface(dial, engine.SPECS[DialsSpec.idname].interface(dial, [("Rain", "yes_no"), ("Wind", "amount")]))
    for node_, first in zip(made, ("Question", "Prompt", "Question", "Question", "Question", "Prompt")):
        group.links.new(gi.outputs["Word"], node_.inputs[first])
    picker.inputs["Items"].default_value = ", ".join(SIX_ITEMS)
    matcher.inputs["Items"].default_value = classifier.inputs["Items"].default_value = "pan, sofa"
    classifier.inputs["Labels"].default_value = "kitchen; living room"
    lit = group.nodes.new("GeometryNodeInputNamedAttribute")
    lit.data_type = 'BOOLEAN'
    lit.inputs["Name"].default_value = "lit"
    group.links.new(enabled(lit.outputs, "Attribute"), rules.inputs["lit"])
    length = group.nodes.new("FunctionNodeStringLength")
    group.links.new(picker.outputs["Item"], length.inputs["String"])
    geometry = gi.outputs["Geometry"]
    for name, output, data_type in zip(SIX_READS, (
            asker.outputs["Answer"], picker.outputs["Index"], length.outputs["Length"], picker.outputs["Probability"],
            matcher.outputs["Probability"], classifier.outputs["Label"], rules.outputs["Outcome"],
            dial.outputs["Rain"], dial.outputs["Wind"]), ('FLOAT', 'INT', 'INT', 'FLOAT', 'FLOAT', 'INT', 'INT',
                                                          'BOOLEAN', 'FLOAT')):
        geometry = store(group, geometry, output, name, data_type)
    group.links.new(geometry, group.nodes.new("NodeGroupOutput").inputs[0])
    users = {}
    for word in SIX_WORDS:
        users[word] = obj = user_of(group, f"Six {word}", points=2)
        obj.data.attributes.new("lit", 'BOOLEAN', 'POINT').data.foreach_set("value", [False, True])
        set_word(obj, word)
    return made, users


def set_word(obj, word):
    mod = obj.modifiers[0]
    mod[mod.node_group.interface.items_tree["Word"].identifier] = word
    obj.update_tag()


def six_entry(node_, obj, i):
    """The node's entry for the words it has in obj's context (spec.gn()), holding SIX_VALUES[i]."""
    spec = engine.SPECS[node_.bl_idname]
    gn = spec.gn(node_, next(c for c in evaluator.find_uses()[node_.id_data.as_pointer()] if c.obj == obj))
    answer, index, probs, p, labels, outcomes, settings = SIX_VALUES[i]
    names, v, extra = ["only"], {}, {}
    if spec.name == "ask":
        extra = {"answer_type": 'YES_NO', "single": {"Answer": answer, "Yes": answer >= 0.5, "Confidence": 0.5}}
    elif spec.name == "pick":
        names, v = SIX_ITEMS, {"prob": probs}
        extra = {"answer_type": 'ONE_OF', "single": {"Index": index, "Item": SIX_ITEMS[index], "Confidence": 0.5}}
    elif spec.name in ("match", "classify"):
        names, v = ["pan", "sofa"], {"p": p, "c": [0.5, 0.5]} if spec.name == "match" else {"label": labels,
                                                                                                "c": [0.5, 0.5]}
    elif spec.name == "rules":
        names, v = ["dark", "bright"], {"outcome": outcomes, "c": [0.5, 0.5]}
        extra = {"answer_type": 'ONE_OF', "shape": {"factors": [dict(LIT)], "payload": [], "outcomes": ["A", "B"]}}
    else:
        names, v = ["Rain", "Wind"], {"value": settings, "c": [0.5, 0.5]}
        extra = {"shape": {"settings": [dict(s) for s in SETTINGS]}}
    rows = S.Rows(name=list(names), wkey=[""] * len(names), ask=[0] * len(names), src="m" * len(names), v=v)
    return S.Entry(words_key=f"{spec.name} {i}", gn=gn, t=1.0 + i, kind=spec.name, asks=["a"], rows=rows, **extra)


def six_expected(i):
    """What an object of six_scene() reads, point by point (ID = index, lit False then True), when its words find entry
    i, or with None the fallbacks."""
    if i is None:
        return {"ask": [0.0] * 2, "index": [-1] * 2, "item": [0] * 2, "prob": [0.0] * 2, "match": [0.0] * 2,
                "label": [-1] * 2, "outcome": [-1] * 2, "rain": [False] * 2, "wind": [0.0] * 2}
    answer, index, probs, p, labels, outcomes, settings = SIX_VALUES[i]
    return {"ask": [answer] * 2, "index": [index] * 2, "item": [len(SIX_ITEMS[index])] * 2, "prob": probs[:2],
            "match": p, "label": labels, "outcome": outcomes, "rain": [settings[0] >= 0.5] * 2,
            "wind": [settings[1] * 10.0] * 2}


def phase_six():
    """All six nodes in one tree shared by three objects (§8.2): the first two objects' words find their own entry of
    each node, single values at its header row and per-element rows after it; the third's find none, so on the shared
    tree they read the neutral fallbacks. The default a miss shows is a row of the table: pointed at the first entry,
    values only, it gives the third object that entry's values on every node."""
    made, users = six_scene()
    stores = {node_.name: S.NodeStore([six_entry(node_, users[word], i) for i, word in enumerate(SIX_WORDS[:2])])
              for node_ in made}
    for node_ in made:
        inner.build(node_, stores[node_.name])
    got = {word: table_read(obj, SIX_READS) for word, obj in users.items()}
    check("six nodes: each object's words find their own entry of every node, its single values and its per-element "
          "rows", got["first"] == six_expected(0) and got["second"] == six_expected(1), got)
    check("six nodes: words that find no entry, on a shared tree, read every node's neutral fallbacks",
          got["third"] == six_expected(None), got["third"])
    before = [pointers(node_) for node_ in made]
    updated = [inner.update_values(node_, stores[node_.name], None, stores[node_.name].entries[0].words_key)
               for node_ in made]
    got = table_read(users["third"], SIX_READS)
    check("six nodes: the default a miss shows is a row of the table, written values only: the third object reads the "
          "first one's entry on every node", updated == [True] * 6 and [pointers(n) for n in made] == before
          and got == six_expected(0), (updated, got))
    for node_ in made:
        inner.update_values(node_, stores[node_.name])
    return made, users, stores


def phase_append(made, users, stores):
    """The playback path's writes, values only (§8.8): append_entry() adds the third object's entry of every node as
    new rows, found at the next evaluation with no graph edit; the same words again are refused, and so is a Rules
    entry whose factors the graph doesn't read. A Pick entry with a new item name shows its Index at once and its Item
    after the rebuild the new name asks for. set_default() points a miss at an entry, and back at the fallbacks."""
    before, built = [pointers(node_) for node_ in made], [node_.node_tree[inner.BUILT] for node_ in made]
    third = {node_.name: six_entry(node_, users["third"], 2) for node_ in made}
    appended = [inner.append_entry(node_, third[node_.name]) for node_ in made]
    got = table_read(users["third"], SIX_READS)
    check("append_entry: the third object's entry of every node is appended values only and found at once, with the "
          "same nodes and layout", appended == [True] * 6 and [pointers(n) for n in made] == before
          and [n.node_tree[inner.BUILT] for n in made] == built and got == six_expected(2), (appended, got))
    again = [inner.append_entry(node_, third[node_.name]) for node_ in made]
    rules = made[4]
    reversed_lit = dict(LIT, levels=["bright", "dark"], values=[1, 0])  # another layout: the flag's levels swapped
    other = dataclasses.replace(third[rules.name], gn=("other",) + third[rules.name].gn[1:],
                                shape=dict(third[rules.name].shape, factors=[reversed_lit]))
    check("append_entry: words the table holds are refused, and so is a Rules entry whose factor layout the graph "
          "doesn't read", again == [False] * 6 and inner.append_entry(rules, other) is False
          and [pointers(n) for n in made] == before, again)

    picker = made[1]
    set_word(users["third"], "fourth")
    named = dataclasses.replace(six_entry(picker, users["third"], 2), words_key="pick 3",
                                single={"Index": 2, "Item": "a new name", "Confidence": 0.5})
    added = inner.append_entry(picker, named)
    first = table_read(users["third"], ("index", "item"))
    grown = S.NodeStore(stores[picker.name].entries + [third[picker.name], named])
    rebuilt = not inner.update_values(picker, grown)
    if rebuilt:
        inner.build(picker, grown)
    check("append_entry: a Pick entry whose item the graph doesn't list shows its Index at once and the fallback Item; "
          "the new name is a new layout, whose rebuild shows it", added and first == {"index": [2] * 2, "item": [0] * 2}
          and rebuilt and table_read(users["third"], ("item",))["item"] == [10] * 2, (added, first, rebuilt))
    inner.build(picker, S.NodeStore(stores[picker.name].entries + [third[picker.name]]))

    set_word(users["third"], "fifth")  # words with no entry
    before = [pointers(node_) for node_ in made]
    missing = table_read(users["third"], SIX_READS)
    for node_ in made:
        inner.set_default(node_, stores[node_.name].entries[1].words_key, stores[node_.name])
    pointed = table_read(users["third"], SIX_READS)
    for node_ in made:
        inner.set_default(node_, None)
    check("set_default: a miss shows the entry it names on every node, values only, and the fallbacks again for None",
          missing == six_expected(None) and pointed == six_expected(1)
          and table_read(users["third"], SIX_READS) == six_expected(None) and [pointers(n) for n in made] == before,
          (missing, pointed))
    group = made[0].id_data
    for obj in users.values():
        bpy.data.objects.remove(obj)
    bpy.data.node_groups.remove(group)
    tables.collect_garbage(list(engine.iter_nodes()))  # their tables go now: phase_save_reload counts what goes


def ask_tree(name, words):
    """A tree whose Ask reads each of `words` from the modifier and stores its Answer as the point attribute answer."""
    group = bpy.data.node_groups.new(name, 'GeometryNodeTree')
    group.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    for word in words:
        group.interface.new_socket(word, in_out='INPUT', socket_type='NodeSocketString')
    group.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    gi = group.nodes.new("NodeGroupInput")
    asker = group.nodes.new(engine.ASK_ID)
    for word in words:
        group.links.new(gi.outputs[word], asker.inputs[word])
    group.links.new(store(group, gi.outputs["Geometry"], asker.outputs["Answer"], "answer", 'FLOAT'),
                    group.nodes.new("NodeGroupOutput").inputs[0])
    return group, asker


def use(obj, **words):
    mod = obj.modifiers[0]
    for word, value in words.items():
        mod[mod.node_group.interface.items_tree[word].identifier] = value
    obj.update_tag()


def answer_of(obj):
    return table_read(obj, ("answer",))["answer"][0]


@contextlib.contextmanager
def counting_writes(writes):
    """Every table write inside goes into `writes`, as the name of the tables function that wrote."""
    saved = {name: getattr(tables, name) for name in ("write_rows", "write_value", "append_rows")}

    def counted(name):
        def write(*args, **kwargs):
            wrote = saved[name](*args, **kwargs)
            if wrote:  # write_value() compares first, and writes only a new value
                writes.append(name)
            return wrote
        return write

    for name in saved:
        setattr(tables, name, counted(name))
    try:
        yield
    finally:
        for name, function in saved.items():
            setattr(tables, name, function)


def phase_no_cap():
    """No cap on word sets (§8.1): 20 objects share one Ask with different words, and each shows its own stored answer,
    with no error. A new word set is a new block of rows: the pass writes it values only, with the same nodes and
    layout. A second pass writes nothing, so the depsgraph update of a write schedules no endless passes."""
    group, asker = ask_tree("Crowd", ["Question"])
    questions = [f"Is this object number {i}?" for i in range(21)]
    entries = [asked(question, (i + 1) / 32, float(i)) for i, question in enumerate(questions)]
    users = []
    for i in range(20):
        users.append(user_of(group, f"Crowd {i}"))
        use(users[-1], Question=questions[i])
    S.save(asker, S.NodeStore(entries[:20]))
    asked_now = []
    with asking(asker, asked_now):
        engine.run_pass()
    got, status = [answer_of(obj) for obj in users], engine.STATUS[asker.uid]
    check("no cap: 20 objects with different words on one node each show their own stored answer, with no error, and "
          "the table holds 20 entries", got == [(i + 1) / 32 for i in range(20)] and status.icon == 'CHECKMARK'
          and status.contexts == 20 and not asked_now and len(tables.header_rows(asker)) == 20, (got, status))
    before, built = pointers(asker), asker.node_tree[inner.BUILT]
    users.append(user_of(group, "Crowd 20"))
    use(users[-1], Question=questions[20])
    S.save(asker, S.NodeStore(entries))
    with asking(asker, asked_now):
        engine.run_pass()
    check("no cap: a new word set is a new block of rows, written values only: the same nodes and layout, and its "
          "object shows its answer", pointers(asker) == before and asker.node_tree[inner.BUILT] == built
          and answer_of(users[-1]) == 21 / 32 and len(tables.header_rows(asker)) == 21, answer_of(users[-1]))
    writes = []
    with counting_writes(writes), asking(asker, asked_now):
        engine.run_pass()
    check("a second pass writes nothing: no table write and no graph edit", not writes and pointers(asker) == before
          and not asked_now, writes)
    for obj in users:
        bpy.data.objects.remove(obj)
    bpy.data.node_groups.remove(group)
    tables.collect_garbage(list(engine.iter_nodes()))


def phase_collision():
    """Words whose keys collide (§8.2): the Keys 'aZ' and 'b5' after the same Question hash alike. On two objects, the
    entry stored later is left out of the table, so both show the first one's answer, and the node's status names both
    word sets and asks for another Key. With another Key each shows its own answer, and the warm entry whose key a live
    one has stays out, with no error. A live entry wins its key over an earlier warm one."""
    group, asker = ask_tree("Collide", ["Key"])
    asker.inputs["Question"].default_value = "Is it night?"
    gns = {key: ("Is it night?", "", "", "", key) for key in ("aZ", "b5", "b6")}
    users = {key: user_of(group, f"Collide {key}") for key in ("aZ", "b5")}
    for key, obj in users.items():
        use(obj, Key=key)
    S.save(asker, S.NodeStore([asked("Is it night?", p, t, key) for key, p, t in (
        ("aZ", 0.875, 1.0), ("b5", 0.125, 2.0), ("b6", 0.5, 3.0))]))
    with asking(asker, []):
        engine.run_pass()
    status, got = engine.STATUS[asker.uid], (answer_of(users["aZ"]), answer_of(users["b5"]))
    check("collision: 'aZ' and 'b5' share a key (words.shared_keys()), so the entry stored later is left out of the "
          "table, and both objects show the first one's answer", words.shared_keys([gns["aZ"], gns["b5"]]) == [
              (gns["aZ"], gns["b5"])] and got == (0.875, 0.875)
          and set(tables.header_rows(asker)) == {words.key(gns["aZ"]), words.key(gns["b6"])}, got)
    check("collision: the node's status is the error naming both word sets and asking for another Key",
          (status.icon, status.text) == ('CANCEL', engine.COLLIDE.format("Is it night? · aZ", "Is it night? · b5")),
          status)
    use(users["b5"], Key="b6")
    with asking(asker, []):
        engine.run_pass()
    status, got = engine.STATUS[asker.uid], (answer_of(users["aZ"]), answer_of(users["b5"]))
    check("collision: with another Key each object shows its own answer, with no error, and the warm entry whose key "
          "a live one has stays out of the table", status.icon == 'CHECKMARK' and got == (0.875, 0.5)
          and len(tables.header_rows(asker)) == 2, (status, got))
    bpy.data.objects.remove(users.pop("aZ"))
    use(users["b5"], Key="b5")
    with asking(asker, []):
        engine.run_pass()
    status = engine.STATUS[asker.uid]
    check("collision: a live entry wins its key over an earlier warm one, with no error",
          status.icon == 'CHECKMARK' and answer_of(users["b5"]) == 0.125, (status, answer_of(users["b5"])))
    bpy.data.objects.remove(users.pop("b5"))
    bpy.data.node_groups.remove(group)
    tables.collect_garbage(list(engine.iter_nodes()))


def phase_single_value_sample():
    """§8.6 regression guard: a single value sampled at the header row the lookup found is a single value, so Pick's
    Index can drive a single-value socket such as Grid's Vertices X."""
    items = "a, b, c, d, e, f, g, h"
    grid_tree = bpy.data.node_groups.new("Sampled", 'GeometryNodeTree')
    grid_tree.interface.new_socket("Prompt", in_out='INPUT', socket_type='NodeSocketString')
    grid_tree.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    gi, go = grid_tree.nodes.new("NodeGroupInput"), grid_tree.nodes.new("NodeGroupOutput")
    picker = grid_tree.nodes.new(engine.PICK_ID)
    picker.inputs["Items"].default_value = items
    grid_tree.links.new(gi.outputs["Prompt"], picker.inputs["Prompt"])
    grid = grid_tree.nodes.new("GeometryNodeMeshGrid")
    grid.inputs["Vertices Y"].default_value = 2
    link = grid_tree.links.new(picker.outputs["Index"], grid.inputs["Vertices X"])
    grid_tree.links.new(grid.outputs["Mesh"], go.inputs["Geometry"])
    obj = bpy.data.objects.new("Sampled", bpy.data.meshes.new("Sampled"))
    bpy.context.scene.collection.objects.link(obj)
    mod = obj.modifiers.new("Sampled", 'NODES')
    mod.node_group = grid_tree
    mod[grid_tree.interface.items_tree["Prompt"].identifier] = "second"
    inner.build(picker, S.NodeStore([S.Entry(words_key=prompt, gn=(prompt, "", items, "", ""), single={
        "Index": index, "Item": items.split(", ")[index], "Confidence": 0.5}) for prompt, index in (("first", 5),
                                                                                                    ("second", 7))]))
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    evaluated = obj.evaluated_get(depsgraph)
    count = len(evaluated.to_mesh().vertices)
    evaluated.to_mesh_clear()
    shape = getattr(picker.outputs["Index"], "inferred_structure_type", 'SINGLE')  # 5.0+ only
    check("a single value sampled at the found header row drives Grid's Vertices X (§8.6)",
          link.is_valid and count == 7 * 2 and shape == 'SINGLE', (link.is_valid, count, shape))


# ------------------------------------------ table objects and the per-element lookup, on a Pick

POINTS = 6
TABLE_WORDS = {  # object -> its modifier's words, and the shift of its ID (ID = index + Shift)
    "TA": {"Prompt": "something to sit on", "Items": "stool, chair, bench", "Shift": -1},
    "TB": {"Prompt": "something to read", "Items": "book, map, note, card", "Shift": 0},
    "TC": {"Prompt": "something to eat", "Items": "apple, pear", "Shift": 0},
}


def table_entry(key, index, probs, prompt=None, items=None):
    words = TABLE_WORDS.get(key, {})
    gn = (words["Prompt"] if prompt is None else prompt, "", words["Items"] if items is None else items, "", "")
    names = [f"row {r}" for r in range(len(probs))]
    return S.Entry(words_key=key, gn=gn, t=1.0, kind="pick", answer_type='ONE_OF', asks=["a"],
                   shape={"rows": len(probs), "columns": {"prob": "FLOAT"}},
                   rows=S.Rows(name=names, wkey=[""] * len(probs), ask=[0] * len(probs), src="m" * len(probs),
                               v={"prob": list(probs)}),
                   single={"Index": index, "Item": names[index], "Confidence": 0.5})


def table_scene():
    """A tree used by three objects of POINTS points; its Pick reads ID = index + Shift and the tree stores
    Probability, Selection and Index as point attributes."""
    tree_ = bpy.data.node_groups.new("Table", 'GeometryNodeTree')
    for name, socket_type in (("Geometry", 'NodeSocketGeometry'), ("Prompt", 'NodeSocketString'),
                              ("Items", 'NodeSocketString'), ("Shift", 'NodeSocketInt')):
        tree_.interface.new_socket(name, in_out='INPUT', socket_type=socket_type)
    tree_.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    gi, go = tree_.nodes.new("NodeGroupInput"), tree_.nodes.new("NodeGroupOutput")
    picker = tree_.nodes.new(engine.PICK_ID)
    for name in ("Prompt", "Items"):
        tree_.links.new(gi.outputs[name], picker.inputs[name])
    shifted = tree_.nodes.new("FunctionNodeIntegerMath")
    shifted.operation = 'ADD'
    tree_.links.new(tree_.nodes.new("GeometryNodeInputIndex").outputs[0], shifted.inputs[0])
    tree_.links.new(gi.outputs["Shift"], shifted.inputs[1])
    tree_.links.new(shifted.outputs[0], picker.inputs["ID"])
    geometry = gi.outputs["Geometry"]
    for name, output, data_type in (("prob", "Probability", 'FLOAT'), ("sel", "Selection", 'BOOLEAN'),
                                    ("idx", "Index", 'INT')):
        geometry = store(tree_, geometry, picker.outputs[output], name, data_type)
    tree_.links.new(geometry, go.inputs[0])
    idents = {i.name: i.identifier for i in tree_.interface.items_tree if getattr(i, "in_out", "") == 'INPUT'}
    users = {}
    for key in TABLE_WORDS:
        mesh = bpy.data.meshes.new(key)
        mesh.from_pydata([(i, 0, 0) for i in range(POINTS)], [], [])
        users[key] = bpy.data.objects.new(key, mesh)
        bpy.context.scene.collection.objects.link(users[key])
        users[key].modifiers.new("Table", 'NODES').node_group = tree_
    for key, typed in TABLE_WORDS.items():
        set_words(users, idents, key, **typed)
    return tree_, picker, users, idents


def set_words(users, idents, key, **words):
    mod = users[key].modifiers["Table"]
    for name, value in words.items():
        mod[idents[name]] = value
    users[key].update_tag()


def table_read(obj, names=("prob", "sel", "idx")):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        return {name: None if name not in mesh.attributes else
                [round(d.value, 6) if isinstance(d.value, float) else d.value for d in mesh.attributes[name].data]
                for name in names}
    finally:
        evaluated.to_mesh_clear()


def table_expected(entry, shift):
    """What a context reads: entry e's row ID for 0 <= ID < N, else the fallbacks; Selection = (ID == Index) and
    Index >= 0. entry None: an unmatched context's neutral values."""
    probs = entry.rows.v["prob"] if entry else []
    index = entry.single["Index"] if entry else -1
    ids = [i + shift for i in range(POINTS)]
    return {"prob": [round(probs[i], 6) if 0 <= i < len(probs) else 0.0 for i in ids],
            "sel": [i == index >= 0 for i in ids], "idx": [index] * POINTS}


def reads(users, entries):
    return ({key: table_read(obj) for key, obj in users.items()},
            {key: table_expected(entry, TABLE_WORDS[key]["Shift"]) for key, entry in zip(users, entries)})


def phase_table(tree_, picker, users, idents):
    entries = [table_entry("TA", 1, [0.25, 0.625, 0.125]), table_entry("TB", 3, [0.125, 0.25, 0.125, 0.5]),
               table_entry("TC", 0, [0.75, 0.25])]
    node_store, live = S.NodeStore(entries), [e.gn for e in entries]
    inner.build(picker, S.NodeStore(), live)  # never answered, Probability linked
    empty, bare = tables.find(picker), len(picker.node_tree.nodes)
    check("table: before the first answer, the table object holds its meta row alone, which points a miss at no row, "
          "and every context reads neutral values",
          empty is not None and len(empty.data.vertices) == 1 and list(tables.read(picker, tables.DEFAULT)) == [-1]
          and "oa_prob" in empty.data.attributes
          and reads(users, [None] * 3)[0] == {key: table_expected(None, TABLE_WORDS[key]["Shift"]) for key in users},
          reads(users, [None] * 3)[0])
    inner.build(picker, node_store, live)
    table = tables.find(picker)
    columns = {name: [round(v, 6) if isinstance(v, float) else v for v in tables.read(picker, name).tolist()]
               for name in (tables.HEAD, tables.KEY, tables.COUNT, "prob", "index")}
    heads = [False, True, False, False, False, True, False, False, False, False, True, False, False]
    keys = [words.key(entry.gn) for entry in entries]
    check("table: point 0, the meta row, then a block per entry: its header row (oa_head, the key of its words, the "
          "count of rows after it, its single values) and one row per item (oa_prob); named after the node, linked to "
          "no collection", table is not None and table.name == tables.PREFIX + picker.uid
          and table.get(tables.UID) == picker.uid and not table.users_collection and columns == {
              tables.HEAD: heads, tables.KEY: [0, keys[0], 0, 0, 0, keys[1], 0, 0, 0, 0, keys[2], 0, 0],
              tables.COUNT: [0, 3, 0, 0, 0, 4, 0, 0, 0, 0, 2, 0, 0],
              "prob": [0.0, 0.0, 0.25, 0.625, 0.125, 0.0, 0.125, 0.25, 0.125, 0.5, 0.0, 0.75, 0.25],
              "index": [-1, 1, -1, -1, -1, 3, -1, -1, -1, -1, 0, -1, -1]}, columns)
    got, want = reads(users, entries)
    check("per-element: 3 contexts, each finding its entry by the key of its words, read their own rows at row + 1 + "
          "ID, and Selection = (ID == Index) AND (Index >= 0)", got == want, got)
    check("per-element: invalid IDs (-1 on TA's first point; past the row count on TB, where TC's block follows) "
          "give the fallbacks", got["TA"]["prob"][0] == 0.0 and got["TA"]["sel"][0] is False
          and got["TB"]["prob"][4:] == [0.0, 0.0] and got["TC"]["prob"][2:] == [0.0] * 4, got)
    kinds = [n.bl_idname for n in picker.node_tree.nodes]
    hashed = [n for n in picker.node_tree.nodes if n.bl_idname == "FunctionNodeHashValue"]
    statistic = [n for n in picker.node_tree.nodes if n.bl_idname == "GeometryNodeAttributeStatistic"]
    check("hash lookup: one Hash Value (String) of the key text and one Attribute Statistic (Float, Point) over the "
          "table object, no String Compare, and as many nodes as with no entry: nothing per entry",
          len(hashed) == 1 and hashed[0].data_type == 'STRING' and len(statistic) == 1
          and (statistic[0].data_type, statistic[0].domain) == ('FLOAT', 'POINT') and len(kinds) == bare
          and not any(n.bl_idname == "FunctionNodeCompare" and n.data_type == 'STRING'
                      for n in picker.node_tree.nodes), (len(kinds), bare))

    tables.write_rows(picker, {"prob": [0.5] * 3})  # damaged: the rebuild must regenerate it from entries alone
    pointer = table.as_pointer()
    inner.build(picker, node_store, live)
    check("rebuild: every output unchanged, the same table object rewritten from the entries alone",
          reads(users, entries)[0] == got and tables.find(picker).as_pointer() == pointer, reads(users, entries)[0])

    before = pointers(picker)
    entries[1] = table_entry("TB", 2, [0.5, 0.125, 0.25, 0.125])  # new values
    entries[2] = table_entry("TC", 1, [0.125, 0.5, 0.375])  # one row more, the same words
    updated = inner.update_values(picker, node_store, live)
    got, want = reads(users, entries)
    check("values only: new rows, single values and a changed row count, with the same nodes",
          updated is True and pointers(picker) == before and got == want
          and len(tables.find(picker).data.vertices) == 14, (updated, got))
    return node_store, live


def phase_no_match(picker, users, idents, node_store, live):
    entries = node_store.entries
    before = pointers(picker)
    set_words(users, idents, "TA", Prompt="something to sit on, edited")
    edited = ("something to sit on, edited", "", TABLE_WORDS["TA"]["Items"], "", "")
    two = inner.k_default_key(["TA", "TB"])
    updated = [inner.update_values(picker, node_store, [edited, live[1]], two)]
    got = table_read(users["TA"]), table_read(users["TB"])
    check("a miss, two live contexts: TA's words, which find no entry, show neutral values, never another context's "
          "answers", two is None and got == (table_expected(None, -1), table_expected(entries[1], 0)), got)
    one = inner.k_default_key(["TA"])
    updated.append(inner.update_values(picker, node_store, [edited], one))
    got = table_read(users["TA"])
    check("a miss, one live context: TA shows its own last answers, the default the meta row points at", one == "TA"
          and got == table_expected(entries[0], -1) and tables.read(picker, tables.DEFAULT)[0] == 1, got)
    set_words(users, idents, "TA", Prompt="")
    hint = inner.k_default_key(["TA"], blocked=True)
    updated.append(inner.update_values(picker, node_store, [], hint))
    got = table_read(users["TA"])
    check("a miss, a cleared Prompt (a hint): neutral values", hint is None and got == table_expected(None, -1), got)
    check("a miss: every change of the default is a values-only update", updated == [True] * 3
          and pointers(picker) == before, updated)
    set_words(users, idents, "TA", Prompt=TABLE_WORDS["TA"]["Prompt"])
    inner.update_values(picker, node_store, live)


def phase_key_text(picker, users, idents, node_store, live):
    """The key text writes each word's length before it (§8.2), so two word tuples whose words join to the same text,
    ("p\\x1f", "i") and ("p", "\\x1fi"), have different keys and each finds its own entry."""
    tricky = table_entry("TX", 0, [0.375], prompt="p\x1f", items="i")
    other = table_entry("TY", 1, [0.25, 0.625], prompt="p", items="\x1fi")
    set_words(users, idents, "TA", Prompt="p\x1f", Items="i")  # tricky's words
    set_words(users, idents, "TB", Prompt="p", Items="\x1fi")  # other's: joined, the same text as tricky's
    inner.build(picker, S.NodeStore([tricky, other, node_store.entries[2]]), [tricky.gn, other.gn, live[2]])
    got = {key: table_read(obj) for key, obj in users.items()}
    check("key text: word tuples whose words join to the same text have different keys, and each finds its own entry",
          "".join(tricky.gn) == "".join(other.gn) and words.key(tricky.gn) != words.key(other.gn)
          and got == {"TA": table_expected(tricky, -1), "TB": table_expected(other, 0),
                      "TC": table_expected(node_store.entries[2], 0)}, got)
    for key in ("TA", "TB"):
        set_words(users, idents, key, Prompt=TABLE_WORDS[key]["Prompt"], Items=TABLE_WORDS[key]["Items"])
    inner.build(picker, node_store, live)


def phase_write_rows(picker, users, node_store, live):
    types = {"prob": 'FLOAT', "n": 'INT', "ok": 'BOOLEAN'}
    steps = []
    for count in (12, 5, 20):  # grow, shrink (clear_geometry() drops every attribute), grow
        columns = {"prob": [r / 16 for r in range(count)], "n": [3 * r - 1 for r in range(count)],
                   "ok": [r % 3 == 0 for r in range(count)]}
        mesh = tables.write_rows(picker, columns, types).data
        written = {name: [round(d.value, 6) if isinstance(d.value, float) else d.value
                          for d in mesh.attributes[tables.ATTR + name].data] for name in columns}
        steps.append(len(mesh.vertices) == count and written == columns
                     and {name: mesh.attributes[tables.ATTR + name].data_type for name in columns} == types)
    check("write_rows: grow -> shrink -> grow keeps every column, its values and its type", steps == [True] * 3, steps)
    long_name = "pl_" + "x" * 65  # oa_pl_x...: 71 bytes, more than Blender 4.5 takes for an attribute name
    try:
        mesh = tables.write_rows(picker, {long_name: [0.5] * 20}).data
        error = None
    except Exception as caught:  # anything but a ValueError fails the check
        error = caught
    check("write_rows: a column Blender can't create raises a ValueError naming it (4.5), else it is written (5.x)",
          isinstance(error, ValueError) and tables.ATTR + long_name in str(error) if error is not None
          else tables.ATTR + long_name in mesh.attributes, repr(error))
    restored = inner.update_values(picker, node_store, live)
    names = sorted(a.name for a in tables.find(picker).data.attributes if a.name.startswith(tables.ATTR))
    got, want = reads(users, node_store.entries)
    check("values only: a write not from the node's entries leaves no digest, so the next update brings the node's own "
          "rows back; columns it no longer writes are gone, and evaluation sees them", restored and names == [
              "oa_conf", "oa_count", "oa_default", "oa_head", "oa_index", "oa_item", "oa_key", "oa_prob"]
          and got == want, (restored, names, got))


def phase_table_copies(tree_, picker, users, node_store, live):
    def column(obj):
        return [round(d.value, 6) for d in obj.data.attributes["oa_prob"].data]

    mine = tables.find(picker)
    rows = column(mine)
    twin_tree = tree_.copy()  # copy() runs for every node: the copy clears its uid and shares the group
    twin = next(n for n in twin_tree.nodes if n.bl_idname == engine.PICK_ID)
    shared = twin.uid == "" and tables.find(twin) == mine
    engine.ensure_owned(list(engine.iter_nodes()))
    theirs = tables.find(twin)
    check("copies: a copied node gets its own table object, named after it, with the same rows",
          shared and theirs is not None and theirs != mine and theirs.data != mine.data and tables.find(picker) == mine
          and theirs.name == tables.PREFIX + twin.uid and theirs.get(tables.UID) == twin.uid
          and not theirs.users_collection and column(theirs) == rows, theirs and theirs.name)
    tables.write_rows(twin, {"prob": [1.0] * 4})
    check("copies: writing the copy's table leaves the original's alone", column(mine) == rows, column(mine))
    gone = theirs.name
    twin_tree.nodes.remove(twin)
    removed = tables.collect_garbage(list(engine.iter_nodes()))
    check("garbage: deleting the node removes its table object; live tables stay",
          removed == 1 and gone not in bpy.data.objects and tables.find(picker) == mine, removed)

    link = next(link for link in tree_.links if link.from_node == picker and link.from_socket.name == "Probability")
    target = link.to_socket
    tree_.links.remove(link)
    refused = inner.update_values(picker, node_store, live)
    inner.build(picker, node_store, live)
    heads = tables.read(picker, tables.HEAD)
    kept = tables.find(picker) == mine and heads is not None and heads.tolist() == [False, True, True, True] \
        and "oa_prob" not in mine.data.attributes
    removed = tables.collect_garbage(list(engine.iter_nodes()))
    tree_.links.new(picker.outputs["Probability"], target)
    rebuilt = not inner.update_values(picker, node_store, live)
    inner.build(picker, node_store, live)
    got, want = reads(users, node_store.entries)
    check("Pick: with Probability unlinked, its table object holds the header rows alone, and stays; linked again, a "
          "rebuild brings the rows per item back", refused is False and kept and removed == 0 and rebuilt
          and got == want, (refused, kept, removed, rebuilt))
    S.save(picker, node_store)  # phase_save_reload() reopens the file: its entries are the node's own

def phase_save_reload():
    """Last, because it reloads the file (§8.5): the table object, in no collection, is saved through the Object
    Info that reads it and survives Purge Unused Data, with its digest, so the reopened file's first update writes
    nothing; appending its tree brings a table along, renamed .001, and garbage collection, going by reference, keeps
    both."""
    def picker_in(group):
        return next(n for n in group.nodes if n.bl_idname == engine.PICK_ID)

    folder = tempfile.mkdtemp(prefix="openact-lookup-")
    path, twin_path = os.path.join(folder, "tables.blend"), os.path.join(folder, "append.blend")
    try:
        before = {key: table_read(bpy.data.objects[key]) for key in TABLE_WORDS}
        name = tables.find(picker_in(bpy.data.node_groups["Table"])).name
        bpy.ops.wm.save_as_mainfile(filepath=twin_path, copy=True)
        bpy.ops.wm.save_as_mainfile(filepath=path)
        bpy.ops.wm.open_mainfile(filepath=path)
        table = bpy.data.objects.get(name)
        after = {key: table_read(bpy.data.objects[key]) for key in TABLE_WORDS}
        bpy.data.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
        purged = {key: table_read(bpy.data.objects[key]) for key in TABLE_WORDS}
        check("save/reload: the table object, in no collection, is saved with the group that reads it and survives "
              "Purge Unused Data", table is not None and not table.users_collection and after == before
              and bpy.data.objects.get(name) is not None and purged == before, (name, after, purged))
        reopened, writes = picker_in(bpy.data.node_groups["Table"]), []
        with counting_writes(writes):
            updated = inner.update_values(reopened, S.load(reopened))
        check("save/reload: the table keeps its digest, so the reopened file's first update writes nothing",
              updated and not writes and table_read(bpy.data.objects["TA"]) == before["TA"], (updated, writes))

        with bpy.data.libraries.load(twin_path) as (_source, target):
            target.node_groups = ["Table"]
        appended = picker_in(target.node_groups[0])
        engine.ensure_owned(list(engine.iter_nodes()))
        mine, theirs = tables.find(picker_in(bpy.data.node_groups["Table"])), tables.find(appended)
        removed = tables.collect_garbage(list(engine.iter_nodes()))
        check("append: the appended tree brings its own table object along, and garbage going by reference keeps both",
              theirs is not None and theirs != mine and theirs.name.startswith(name) and removed == 0
              and [d.value for d in theirs.data.attributes["oa_prob"].data]
              == [d.value for d in mine.data.attributes["oa_prob"].data], (theirs and theirs.name, removed))
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def phase_layout_version():
    """A newer OpenAct whose build() lays out another graph for the same entries bumps inner.LAYOUT: update_values()
    and append_entry() then refuse the groups built before it, so the first pass after an upgrade or a file load
    rebuilds them. A group OpenAct 1.0 laid out, with its match chain, holds a digest there instead of a layout."""
    asks = S.NodeStore([ask_entry("A", 0.875, 0.75)])
    inner.build(ask, asks)
    inner.LAYOUT += 1
    try:
        bumped = inner.update_values(ask, asks), inner.append_entry(ask, ask_entry("B", 0.25, 0.5))
    finally:
        inner.LAYOUT -= 1
    ask.node_tree[inner.BUILT] = "6f1c0e8b9d2a4c7e5f3b1a0d9c8e7f6a5b4c3d2e"  # a 1.0 group's sha1
    old = inner.update_values(ask, asks), inner.append_entry(ask, ask_entry("B", 0.25, 0.5))
    heads = len(tables.header_rows(ask))
    inner.build(ask, asks)
    check("layout version: a group laid out under another LAYOUT, or by OpenAct 1.0, needs build(): update_values() "
          "and append_entry() refuse it, writing nothing", bumped == old == (False, False) and heads == 1
          and inner.update_values(ask, asks), (bumped, old, heads))


NO_ADDON_ITEMS = "pan, sofa, kettle, lamp"
NO_ADDON = (  # node type, typed words, its entry's rows and single values, {output: (attribute, data type)}
    (MatchSpec.idname, {"Question": "Would you find it in a kitchen?", "Items": NO_ADDON_ITEMS},
     {"p": [0.875, 0.125, 0.75, 0.25], "c": [0.75, 0.75, 0.5, 0.5]}, {},
     {"Probability": ("m_p", 'FLOAT'), "Selection": ("m_sel", 'BOOLEAN'), "Confidence": ("m_c", 'FLOAT')}),
    (ClassifySpec.idname, {"Items": NO_ADDON_ITEMS, "Labels": "kitchen; living room"},
     {"label": [1, 0, 1, 0], "c": [0.625, 0.5, 0.375, 0.25]}, {},
     {"Label": ("c_label", 'INT'), "Confidence": ("c_c", 'FLOAT')}),
    (engine.PICK_ID, {"Prompt": "something to cook with", "Items": NO_ADDON_ITEMS},
     {"prob": [0.125, 0.0625, 0.75, 0.0625]}, {"Index": 2, "Item": "kettle", "Confidence": 0.5},
     {"Probability": ("p_prob", 'FLOAT'), "Selection": ("p_sel", 'BOOLEAN')}),
)
NO_ADDON_READS = {  # what 5 points read with their index as ID: their row, and past the 4 rows the fallbacks
    "m_p": [0.875, 0.125, 0.75, 0.25, 0.0], "m_sel": [True, False, True, False, False],
    "m_c": [0.75, 0.75, 0.5, 0.5, 0.0], "c_label": [1, 0, 1, 0, -1], "c_c": [0.625, 0.5, 0.375, 0.25, 0.0],
    "p_prob": [0.125, 0.0625, 0.75, 0.0625, 0.0], "p_sel": [False, False, True, False, False]}
# What the Blender without OpenAct runs on the saved file: the objects' attributes and their trees' node types
NO_ADDON_READ = """\
import json, os, sys
import bpy
keys, names = json.loads(sys.argv[sys.argv.index("--") + 1])
depsgraph = bpy.context.evaluated_depsgraph_get()
got = {"kinds": sorted({n.bl_idname for key in keys for n in bpy.data.objects[key].modifiers[0].node_group.nodes})}
for key in keys:
    evaluated = bpy.data.objects[key].evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    got[key] = {name: [round(d.value, 6) if isinstance(d.value, float) else d.value for d in mesh.attributes[name].data]
                for name in names}
    evaluated.to_mesh_clear()
print("NOADDON " + json.dumps(got), flush=True)
os._exit(0)
"""


def no_addon_scene(name, float_id=None):
    """An object of 5 points whose tree stores the per-element outputs of a Match, a Classify and a Pick over the same
    4 items, each built from one entry, with its ID unlinked, or linked from a Value node holding float_id: (node,
    its store) of each."""
    tree_ = bpy.data.node_groups.new(name, 'GeometryNodeTree')
    tree_.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    tree_.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    geometry = tree_.nodes.new("NodeGroupInput").outputs["Geometry"]
    value = None
    if float_id is not None:
        value = tree_.nodes.new("ShaderNodeValue")
        value.outputs[0].default_value = float_id
    built = []
    for idname, typed, v, single, outputs in NO_ADDON:
        node_ = tree_.nodes.new(idname)
        for word, text in typed.items():
            node_.inputs[word].default_value = text
        if value is not None:
            tree_.links.new(value.outputs[0], node_.inputs["ID"])
        for output, (attribute, data_type) in outputs.items():  # before build(): Pick reads Probability while linked
            geometry = store(tree_, geometry, node_.outputs[output], attribute, data_type)
        n = len(next(iter(v.values())))
        gn = tuple(typed.get(word, "") for word in engine.SPECS[idname].WORDS)
        node_store = S.NodeStore([S.Entry(words_key=idname, gn=gn, t=1.0, single=single, rows=S.Rows(
            name=NO_ADDON_ITEMS.split(", "), wkey=[""] * n, ask=[0] * n, src="m" * n, v=v))])
        inner.build(node_, node_store)
        built.append((node_, node_store))
    tree_.links.new(geometry, tree_.nodes.new("NodeGroupOutput").inputs[0])
    user_of(tree_, name, points=5)
    return built


def phase_no_addon():
    """G6 with an unlinked ID: without OpenAct its nodes are undefined, and an unlinked input passes its socket's value,
    not the element's index. Match, Classify and Pick with an unlinked ID, saved and opened in a Blender without the
    add-on (a second Blender this suite starts), read what they read with it: each element its own row. A group saved
    before, its ID's value 0, reads row 0 everywhere there until its next pass, which rebuilds it. A float ID beyond
    the Int range, which Blender converts to -2**31, reads the fallbacks in both."""
    keys = ["NoAddon Passed", "NoAddon Old", "NoAddon Float"]
    passed, old = no_addon_scene(keys[0]), no_addon_scene(keys[1])
    no_addon_scene(keys[2], float_id=1e10)
    for node_, _node_store in passed + old:  # as saved before: the ID's value 0, laid out under another LAYOUT
        node_.inputs["ID"].default_value = 0
        node_.node_tree[inner.BUILT] = "an older layout"
    refused = []
    for node_, node_store in passed:  # the next pass: values only, else a rebuild (engine._process())
        refused.append(not inner.update_values(node_, node_store))
        if refused[-1]:
            inner.build(node_, node_store)
    mine = {key: table_read(bpy.data.objects[key], NO_ADDON_READS) for key in keys}
    folder = tempfile.mkdtemp(prefix="openact-noaddon-")
    try:
        path = os.path.join(folder, "noaddon.blend")
        bpy.ops.wm.save_as_mainfile(filepath=path, copy=True)
        result = subprocess.run([bpy.app.binary_path, "-b", "--factory-startup", "-t", "4", path, "--python-expr",
                                 NO_ADDON_READ, "--", json.dumps([keys, list(NO_ADDON_READS)])], capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=300, stdin=subprocess.DEVNULL,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired as error:  # run() killed it
        output = f"the Blender without the add-on timed out: {error}"
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    line = next((text for text in output.splitlines() if text.startswith("NOADDON ")), "")
    theirs = json.loads(line[len("NOADDON "):]) if line else {}
    kinds = theirs.get("kinds", [])
    check("no add-on: with OpenAct, Match, Classify and Pick with an unlinked ID read each element's own row, and the "
          "fallbacks past the rows, whatever the ID's hidden value", mine[keys[0]] == mine[keys[1]] == NO_ADDON_READS,
          mine)
    check("no add-on: saved and opened in a Blender without OpenAct, where they are undefined nodes, they read the same",
          "NodeUndefined" in kinds and not any(kind.startswith("GeometryNodeOpenAct") for kind in kinds)
          and theirs.get(keys[0]) == mine[keys[0]], theirs.get(keys[0], output[-3000:]))
    row_0 = {name: values[:1] * len(values) for name, values in NO_ADDON_READS.items()}
    check("no add-on: a group saved before, its ID's value 0, reads row 0 everywhere there, and its next pass rebuilds "
          "it (update_values() refuses it): the values above", refused == [True] * 3 and theirs.get(keys[1]) == row_0,
          (refused, theirs.get(keys[1])))
    fallbacks = {name: values[-1:] * len(values) for name, values in NO_ADDON_READS.items()}  # the 5th point's
    check("no add-on: an ID linked from a float beyond the Int range (1e10, which converts to -2**31) reads the "
          "fallbacks everywhere, with and without OpenAct", mine[keys[2]] == theirs.get(keys[2]) == fallbacks,
          (mine[keys[2]], theirs.get(keys[2])))
    for key in keys:  # the scene goes, and its tables now: phase_save_reload counts what garbage collection removes
        bpy.data.objects.remove(bpy.data.objects[key])
        bpy.data.node_groups.remove(bpy.data.node_groups[key])
    tables.collect_garbage(list(engine.iter_nodes()))


def phase_factor_id():
    """Only the ID of Match, Classify and Pick holds UNLINKED. A Rules factor may be named ID (§6.5), and build() keeps
    the value typed on it, whatever its kind."""
    group = bpy.data.node_groups.new("FactorID", 'GeometryNodeTree')
    rules = group.nodes.new(RulesSpec.idname)
    kept = {}
    for kind, typed in (("category", 2), ("amount", 0.75), ("flag", True)):
        inner.sync_interface(rules, engine.SPECS[RulesSpec.idname].interface(rules, [("ID", kind)], []))
        rules.inputs["ID"].default_value = typed
        try:
            inner.build(rules, S.NodeStore())
            kept[kind] = rules.inputs["ID"].default_value
        except Exception as error:  # noqa: BLE001 - the check shows it
            kept[kind] = f"{type(error).__name__}: {error}"
    check("a Rules factor named ID keeps the value typed on it through build(), whatever its kind",
          kept == {"category": 2, "amount": 0.75, "flag": True}, kept)
    bpy.data.node_groups.remove(group)
    tables.collect_garbage(list(engine.iter_nodes()))  # its table goes now: phase_save_reload counts what goes


def phase_tables():
    """The table object and the per-element path on synthetic entries, through a Pick."""
    tree_, picker, users, idents = table_scene()
    node_store, live = phase_table(tree_, picker, users, idents)
    phase_no_match(picker, users, idents, node_store, live)
    phase_key_text(picker, users, idents, node_store, live)
    phase_write_rows(picker, users, node_store, live)
    phase_table_copies(tree_, picker, users, node_store, live)


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
    """A BENCH line and, against a budget, the timing rule of §18.2: other work on the machine may slow a run,
    so above the budget is a warning and only above twice the budget a failure."""
    print(f"BENCH {path} {elements} {ms:.1f}", flush=True)
    if budget is None:
        return
    if ms > budget:
        print(f"WARN bench: {path} took {ms:.1f} ms, above its {budget} ms budget", flush=True)
    check(f"bench: {path} within 2x its {budget} ms budget", ms <= 2 * budget, f"{ms:.1f} ms, {elements} elements")


def bench_shape(label, node_, make, count, shape, budget):
    """build() from `count` entries make(i, 0), then update_values() when nothing changed, then alternating with a
    store whose last entry's values differ (make(i, 1)), a table rewrite, then append_entry() of new words, a block of
    rows (§8.8). BENCH elements: the graph's nodes for build(), else the table's rows."""
    store_a = S.NodeStore([make(i, 0) for i in range(count)])
    store_b = S.NodeStore(store_a.entries[:-1] + [make(count - 1, 1)])
    worst = " worst" if shape == "worst" else ""
    ms = timed(lambda: inner.build(node_, store_a), runs=3 if worst else 5)
    bench_line(f"build/{label}/{shape}", len(node_.node_tree.nodes), ms, BUDGET_MS[budget])
    rows = len(tables.find(node_).data.vertices)
    ms = timed(lambda: check_update(node_, store_a))
    bench_line(f"update_values/{label}/{shape}/unchanged", rows, ms, BUDGET_MS["update_values" + worst])
    stores = itertools.cycle((store_b, store_a))
    ms = timed(lambda: check_update(node_, next(stores)), runs=6)
    bench_line(f"update_values/{label}/{shape}/one-entry", rows, ms, BUDGET_MS["update_values" + worst])
    fresh = iter([make(count + i, 0) for i in range(5)])
    ms = timed(lambda: check_append(node_, next(fresh)))
    bench_line(f"append/{label}/{shape}", rows, ms, BUDGET_MS["append" + worst])
    inner.build(node_, S.NodeStore())  # its table goes back to a meta row, which the next shapes' timings don't carry


def check_update(node_, node_store):
    if not inner.update_values(node_, node_store):
        raise AssertionError("update_values() refused the entries build() laid out")


def check_append(node_, entry):
    if not inner.append_entry(node_, entry):
        raise AssertionError("append_entry() refused new words")


BENCH_ITEMS = ["table", "chair", "lamp", "rug", "bookshelf"]


def rows_entry(i, gn, names, v, payload=None, **fields):
    """An entry of one row per name, answered v (and payload columns), last used at i."""
    n = len(names)
    return S.Entry(words_key=str(i), gn=gn, t=float(i), asks=["a"], **fields, rows=S.Rows(
        name=list(names), wkey=[""] * n, ask=[0] * n, src="m" * n, v=v, payload=payload or {}))


def bench_ask(i, offset):
    return S.Entry(words_key=str(i), gn=(f"Question {i}?", f"Context {i}", "", "", ""), single={
        "Answer": (i + offset) % 40 / 40, "Yes": (i + offset) % 2 == 0, "Confidence": (i + 2 * offset) % 50 / 50})


def bench_pick(materials=()):
    """Entries of a Pick of BENCH_ITEMS or, with materials, of Material Slots, whose items are the materials' names;
    each item's probability per row."""
    names = [material.name for material in materials] or BENCH_ITEMS

    def make(i, offset):
        index = (i + offset) % len(names)
        probs = [0.6 if r == index else 0.1 for r in range(len(names))]
        single = {"Index": index, "Item": names[index], "Confidence": (i + offset) % 40 / 40}
        if materials:
            single["Material"] = names[index]
        gn = (f"Prompt {i}", "", "" if materials else ", ".join(names), "", "")
        return rows_entry(i, gn, names, {"prob": probs}, single=single)
    return make


def bench_classify(payload, items):
    """Entries of a Classify of `items` items and 4 labels, each label with `payload` payload columns."""
    names = [f"item {r}" for r in range(items)]

    def make(i, offset):
        labels = [(r + i + offset) % 4 for r in range(items)]
        return rows_entry(i, (f"Question {i}?", "", "//items.csv", "//labels.csv", "", ""), names,
                          {"label": labels, "c": [0.5] * items},
                          {f"p{c}": [(label + c) / 10 for label in labels] for c in range(payload)})
    return make


def bench_rules(bands, custom, mode):
    """Entries of a Rules of 3 Amount factors of `bands` bands (custom edges or equal widths): bands**3 cells."""
    factors = [{"name": f"f{j}", "kind": "amount", "levels": [f"band {b}" for b in range(bands)], "values": None,
                "edges": [b / bands for b in range(bands + 1)] if custom else None} for j in range(3)]
    cells = bands ** 3
    names = [f"cell {c}" for c in range(cells)]

    def make(i, offset):
        gn = (f"Question {i}?", "", "//factors.csv", "//outcomes.csv", "", "")
        if mode == 'ONE_OF':
            outcomes = [(c + i + offset) % 7 for c in range(cells)]
            return rows_entry(i, gn, names, {"outcome": outcomes, "c": [0.5] * cells},
                              {"instance": [o % 3 for o in outcomes], "scale": [o / 7 for o in outcomes]},
                              answer_type=mode, shape={"factors": factors})
        return rows_entry(i, gn, names, {"amount": [(c + i + offset) % 100 / 100 for c in range(cells)],
                                         "c": [0.5] * cells, "ok": [1.0] * cells}, answer_type=mode,
                          shape={"factors": factors})
    return make


def bench_dials(settings):
    """Entries of a Dials of `settings` Amount settings, each with its own range."""
    shape = {"settings": [{"name": f"s{s}", "type": "amount", "min": 0.0, "max": 1.0 + s} for s in range(settings)]}
    names = [f"s{s}" for s in range(settings)]

    def make(i, offset):
        return rows_entry(i, (f"Prompt {i}", "//settings.csv", "", ""), names,
                          {"value": [(s + i + offset) % 10 / 10 for s in range(settings)], "c": [0.5] * settings},
                          shape=shape)
    return make


def plan_4096():
    """What planning a changed 4096-cell Rules context costs in Python (§5.3): each cell's instructions,
    its row words key and its cell key, then the entry's words key, with the pipeline's key functions."""
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
        wkeys.append(pipeline.wkey("rules", 'AMOUNT', ["", question], levels, options))
        pipeline.cell_key("score", state, instructions, list(criteria.items()), True)
    return pipeline.words_key(["", question], [], 'AMOUNT', "shape", wkeys)


def phase_bench(shapes):
    """The six nodes' graphs and tables through the real inner.build(), in a tree a modifier uses, as the demo scenes'
    are: the demo shapes of §18.4, one context each (Ask and Pick two), and the worst ones the limits allow (§7.5,
    §8.7, §9.8), the largest graphs with tables of MANY blocks or 16 blocks of 4096 or 255 rows."""
    group = bpy.data.node_groups.new("Bench", 'GeometryNodeTree')
    group.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    group.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    group.links.new(group.nodes.new("NodeGroupInput").outputs[0], group.nodes.new("NodeGroupOutput").inputs[0])
    user_of(group, "Bench")
    materials = [bpy.data.materials.new(f"Bench {i}") for i in range(lists.MAX_ITEMS)]  # Item's and Material's names

    def made(idname, factors=None, payload=(), outputs=None, mode=None, list_source=None, linked=False):
        node_ = group.nodes.new(idname)
        spec = engine.SPECS[idname]
        if mode is not None:
            node_.answer_type = mode
        if list_source is not None:
            node_.list_source = list_source
        if factors is not None:
            inner.sync_interface(node_, spec.interface(node_, factors, list(payload)))
        elif payload:
            inner.sync_interface(node_, spec.interface(node_, list(payload)))
        elif outputs is not None:
            inner.sync_interface(node_, spec.interface(node_, outputs))
        if linked:  # Probability read, so the table holds a row per item
            probe = group.nodes.new("GeometryNodeStoreNamedAttribute")
            group.links.new(node_.outputs["Probability"], enabled(probe.inputs, "Value"))
        return node_

    three = [(f"f{j}", "amount") for j in range(3)]
    plans = []
    if "demo" in shapes:
        plans += [("ask", made(engine.ASK_ID), bench_ask, 2, "build demo"),
                  ("pick", made(engine.PICK_ID), bench_pick(), 2, "build demo"),
                  ("classify-materials", made(ClassifySpec.idname, payload=[("p0", 'FLOAT'), ("p1", 'FLOAT')]),
                   bench_classify(2, 4), 1, "build demo"),
                  ("rules-biome", made(RulesSpec.idname, three, [("instance", "INT"), ("scale", "FLOAT")],
                                       mode='ONE_OF'), bench_rules(3, False, 'ONE_OF'), 1, "build demo"),
                  ("rules-snow", made(RulesSpec.idname, three, mode='AMOUNT'), bench_rules(3, False, 'AMOUNT'), 1,
                   "build demo interpolated"),
                  ("dials-weather", made(DialsSpec.idname, outputs=[(f"s{s}", "amount") for s in range(5)]),
                   bench_dials(5), 1, "build demo")]
    if "worst" in shapes:
        plans += [("ask", made(engine.ASK_ID), bench_ask, MANY, "build worst"),
                  ("pick-materials", made(engine.PICK_ID, list_source='MATERIALS', linked=True),
                   bench_pick(materials), 16, "build worst"),
                  ("classify", made(ClassifySpec.idname, payload=[(f"p{c}", 'FLOAT') for c in range(16)]),
                   bench_classify(16, 255), 16, "build worst"),
                  ("dials", made(DialsSpec.idname, outputs=[(f"s{s}", "amount") for s in range(DIALS_MAX_SETTINGS)]),
                   bench_dials(DIALS_MAX_SETTINGS), MANY, "build worst"),
                  ("rules", made(RulesSpec.idname, three, mode='AMOUNT'), bench_rules(16, True, 'AMOUNT'), 16,
                   "build worst")]
    for label, node_, make, count, budget in plans:
        bench_shape(label, node_, make, count, "demo" if "demo" in budget else "worst", budget)
    bench_line("plan/rules", 4096, timed(plan_4096, runs=3))  # recorded only: the pass's TIMING covers planning


phase_lookup()
phase_positions()
phase_pass()
phase_pick_pass()
phase_stored_default()
phase_table_only()
phase_carry()
phase_live_set()
phase_last_shown()
phase_statuses()
phase_single_value_sample()
phase_no_cap()
phase_collision()
phase_append(*phase_six())
phase_tables()
phase_layout_version()
phase_no_addon()
phase_factor_id()
if "--bench" in ARGS:
    chosen = os.environ.get("OPENACT_BENCH_SHAPE", "")
    phase_bench({chosen} if chosen in ("demo", "worst") else {"demo", "worst"})
phase_save_reload()  # last: it reloads the file
finish()