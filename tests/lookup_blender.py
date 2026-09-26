"""The inner group built from a node's entries, with no server: the match chain, entry-level values,
values-only updates and rebuilds, and single-value sampling from a table object. With --bench, also the
main-thread budgets of §8.4, as BENCH lines: inner.build() and update_values() for the Ask and Pick
shapes and the contract spike's stand-ins for the demo and worst shapes of the nodes still to come; and,
recorded without a gate, the planning cost of a 4096-cell context (§5.3). As §18.2 checks timings, a
measurement above its budget prints a WARN line and fails only above twice the budget.

Run:  blender -b --factory-startup --python tests/lookup_blender.py [-- --bench]
      OPENACT_BENCH_SHAPE=demo|worst limits --bench to one set of shapes (default: both).
"""

import contextlib
import dataclasses
import itertools
import json
import os
import shutil
import sys
import tempfile
import time

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import ROOT, check, finish, sock, store  # noqa: E402
from openact import engine, evaluator, inner, nodes, ops, pipeline, tables  # noqa: E402  (bpy must be imported first)
from openact import prefs as P  # noqa: E402
from openact import store as S  # noqa: E402  (store() is the harness's attribute writer)
from openact.inner import SocketDef  # noqa: E402
from openact.specs.base import Spec  # noqa: E402
from openact.specs.match import MatchSpec  # noqa: E402
from openact.specs.rules import RulesSpec  # noqa: E402

P._Defaults.auto_start = False  # nothing here may start a model server
nodes.register()
ops.register()
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
    return S.Entry(words_key=key, gn=(WORDS[key]["Question"], "", "", "", ""), kind="ask", answer_type='YES_NO',
                   single={"Answer": answer, "Yes": answer >= 0.5, "Confidence": confidence})


def pick_entry(key, index, confidence):
    words = WORDS[key]
    names = [name.strip() for name in words["Items"].split(",")]
    return S.Entry(words_key=key, gn=(words["Prompt"], "", words["Items"], "", ""), kind="pick", answer_type='ONE_OF',
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
    try:
        inner.build(ask, S.NodeStore(grown.entries + [S.Entry(gn=("Is it broken?", "", "", "", ""))]))  # no values
        raised = False
    except KeyError:
        raised = True
    check("build: after a build that raised, update_values() refuses even the entries built before it",
          raised and inner.update_values(ask, grown) is False, raised)
    inner.build(ask, grown)
    inner.build(pick, picks)
    check("build: a rebuild from the same entries leaves every output unchanged",
          {key: read(key) for key in WORDS} == got, got)


def phase_converted():
    """Entries converted from schema 1 hold v1's 5-word gn (§14.4), so the match chain compares each word at its
    position: a converted text Pick matches only its prompt and Items text, a converted collection Pick only its
    prompt with no Items text. An entry with the prototype's 2-word gn (saved before the fixed word tuples)
    matches nothing, and the entries after it keep their numbers."""
    pick.entries = json.dumps([
        {"key": "a", "gn": [WORDS["A"]["Prompt"], WORDS["A"]["Items"]], "t": 1.0,
         "a": {"index": 1, "item": "chair", "confidence": 0.5, "probabilities": [0.25, 0.75],
               "items": ["table", "chair"], "collection": ""}},
        {"key": "b", "gn": [WORDS["B"]["Prompt"], ""], "t": 1.0,
         "a": {"index": 0, "item": "book", "confidence": 0.5, "probabilities": [0.75, 0.25],
               "items": ["book", "lamp"], "collection": "Shelf"}}])  # v0.1.0, schema 1
    converted = S.load(pick)
    old = S.Entry(words_key="old", gn=(WORDS["C"]["Prompt"], WORDS["C"]["Items"]), kind="pick", answer_type='ONE_OF',
                  single={"Index": 1, "Item": "pear", "Confidence": 0.5})
    inner.build(pick, S.NodeStore(converted.entries + [old, pick_entry("C", 0, 0.5)]))
    got = {key: read(key)["index"] for key in WORDS}
    mod = objects["C"].modifiers["Lookup"]
    for key in ("A", "B"):  # C with that prompt and no Items text: a collection Pick's words
        mod[ids["Prompt"]], mod[ids["Items"]] = WORDS[key]["Prompt"], ""
        objects["C"].update_tag()
        got[f"C with {key}'s prompt"] = read("C")["index"]
    mod[ids["Prompt"]], mod[ids["Items"]] = WORDS["C"]["Prompt"], WORDS["C"]["Items"]
    objects["C"].update_tag()
    check("converted entries: each matches its own words at v1's positions, a 2-word entry matches nothing, and the "
          "entries after it keep their numbers",
          [e.gn for e in converted.entries] == [(WORDS["A"]["Prompt"], "", WORDS["A"]["Items"], "", ""),
                                                (WORDS["B"]["Prompt"], "", "", "", "")]
          and got == {"A": 1, "B": -1, "C": 0, "C with A's prompt": -1, "C with B's prompt": 0}, got)


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
    asks nothing, and a new Key carries every row over. Schema 1 is converted on read, schema 2 is written, the
    live set it passes to save() keeps a live entry that eviction would otherwise remove, and a newer OpenAct's
    JSON is left as it is."""
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
    check("pass: the inner group shows the stored answers, the live ones materialised", engine.STATUS[node_.uid][0]
          == 'CHECKMARK' and inner.update_values(node_, S.load(node_), [job.gn for job in jobs.values()])
          and len(inner.materialised(saved, [job.gn for job in jobs.values()])) == inner.WARM_TOTAL,
          engine.STATUS[node_.uid])

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
          not questions and node_.entries == before and engine.STATUS[node_.uid][0] == 'CHECKMARK', len(questions))
    node_.inputs["Key"].default_value = "left"
    with asking(node_, questions):
        engine.run_pass()
    keyed = {e.gn[0]: e.single["Answer"] for e in S.load(node_).entries if e.gn[-1] == "left"}
    check("pass: a new Key carries every row over from what each context showed, and asks nothing",
          not questions and keyed == {"Old words": 0.25, "New words": 0.875}, (len(questions), keyed))
    node_.inputs["Key"].default_value = ""

    node_.entries = ('[{"key": "k", "gn": ["Old words", ""], "t": 1.0,'
                     ' "a": {"mode": "YES_NO", "value": 1.0, "confidence": 1.0}}]')  # v0.1.0, schema 1
    pipeline.remember(new_cell, {"p": 0.875, "model": "fake"})
    try:
        with asking(node_, questions):
            engine.run_pass()
    finally:
        pipeline.forget([new_cell])
    saved = [(e.gn, e.legacy) for e in S.load(node_).entries]
    check("pass: a schema 1 list is converted on read, its entry kept until a new answer replaces it, and the store "
          "written as schema 2",
          node_.entries.startswith('{"schema":2,') and questions == [jobs["Old words"].rows[0].cell.key]
          and saved == [(("Old words", "", "", "", ""), True), (("New words", "", "", "", ""), False)], saved)
    engine.reask(node_)
    check("Ask Again: the node's answers are forgotten, saved as schema 2 and the group rebuilt",
          node_.entries == '{"schema":2,"entries":[]}' and inner.update_values(node_, S.NodeStore()), node_.entries)

    newer = '{"schema":3,"entries":[{"written by":"a newer OpenAct"}]}'
    node_.entries = newer
    group, asked = pointers(node_), []
    submit, engine._submit = engine._submit, lambda job, _missing: asked.append(job.uid)
    try:
        engine.run_pass()
    finally:
        engine._submit = submit
    check("pass: a newer OpenAct's answers show an error, and are not asked for, written over or rebuilt",
          node_.entries == newer and node_.uid not in asked and pointers(node_) == group
          and engine.STATUS[node_.uid] == ('ERROR', S.newer_message(S.load(node_))), engine.STATUS[node_.uid])
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
          and engine.STATUS[picker.uid][0] == 'CHECKMARK', (engine.STATUS[picker.uid], got))
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
          status[0] == 'ERROR' and "only one answer can show" in status[1] and "Key" in status[1] and not questions,
          status)
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
          and engine.STATUS[picker.uid][0] == 'CHECKMARK', (asked == [kitchen], got, engine.STATUS[picker.uid]))


def phase_converted_default():
    """Every entry converted from v0.1.0 has the words key "", so that key names none of them (§8.2): a single context
    whose words changed shows its own last answer or neutral values while it waits, never another converted entry's;
    and the warm entries are the most recent ones, not the converted ones."""
    group = bpy.data.node_groups.new("Converted", 'GeometryNodeTree')
    for name, socket_type in (("Geometry", 'NodeSocketGeometry'), ("Question", 'NodeSocketString')):
        group.interface.new_socket(name, in_out='INPUT', socket_type=socket_type)
    group.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    gi, go = group.nodes.new("NodeGroupInput"), group.nodes.new("NodeGroupOutput")
    asker = group.nodes.new(engine.ASK_ID)
    group.links.new(gi.outputs["Question"], asker.inputs["Question"])
    group.links.new(store(group, gi.outputs["Geometry"], asker.outputs["Answer"], "answer", 'FLOAT'), go.inputs[0])
    asker.entries = json.dumps([  # v0.1.0 (schema 1): two answers of this node, converted on read
        {"key": "n", "gn": ["Is it night?", ""], "t": 1.0, "a": {"mode": "YES_NO", "value": 0.875, "confidence": 0.75}},
        {"key": "c", "gn": ["Is it cold?", ""], "t": 2.0, "a": {"mode": "YES_NO", "value": 0.125, "confidence": 0.75}}])
    mesh = bpy.data.meshes.new("Converted")
    mesh.from_pydata([(0, 0, 0)], [], [])
    user = bpy.data.objects.new("Converted", mesh)
    bpy.context.scene.collection.objects.link(user)
    user.modifiers.new("Converted", 'NODES').node_group = group
    ident = group.interface.items_tree["Question"].identifier

    def shown(question):
        user.modifiers["Converted"][ident] = question
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
    check("converted entries: a single context whose words changed keeps its own answer or shows neutral values, "
          "never the other converted entry's", before == 0.125 and after in (0.125, 0.0), (before, after))
    live = S.Entry(words_key="live", gn=("Live?", "", "", "", ""))
    old = [S.Entry(words_key="", gn=(f"Converted {i}?", "", "", "", ""), t=float(i), legacy=True) for i in range(3)]
    recent = [S.Entry(words_key=f"recent {i}", gn=(f"Recent {i}?", "", "", "", ""), t=10.0 + i) for i in range(3)]
    warm = [e.gn[0] for e in inner.materialised(old + recent + [live], [live.gn], "")]
    check("converted entries: with the default \"\", the warm entries are the most recent, not the converted ones",
          warm == ["Recent 0?", "Recent 1?", "Recent 2?", "Live?"], warm)
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
    Items CSV (the same words) asks only the row whose wkey changed, the old answers never showing meanwhile, and
    nothing more once its answer arrives; undoing the edit before then asks nothing and shows the stored answers."""
    folder = tempfile.mkdtemp(prefix="openact-carry-")
    path = os.path.join(folder, "items.csv")

    def items(lamp, stamp):  # each edit gets a new stamp, which the fingerprint holds
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(f"answer,when\nchair,a seat\ntable,a flat surface\nlamp,{lamp}\n")
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
    user = user_of(group, "Carry", 3)

    def cells():
        return [row.cell.key for row in
                engine.SPECS[MatchSpec.idname].plan(matcher, evaluator.find_uses()[group.as_pointer()][0]).rows]

    items("gives light", 10 ** 18)
    old, questions = cells(), []
    for key, p in zip(old, (0.875, 0.75, 0.25)):
        pipeline.remember(key, {"p": p, "model": "fake"})
    try:
        with asking(matcher, questions):
            engine.run_pass()
        stored = table_read(user)["prob"]
        pipeline.forget(old)  # a new session, in which the CSV was edited while Blender was closed
        engine.on_load_post()
        items("gives light at night", 2 * 10 ** 18)  # only lamp's wkey changes
        with asking(matcher, questions):
            engine.run_pass()
        edited, meanwhile, lamp = list(questions), table_read(user)["prob"], cells()[2]
        check("carry-over, new session: a text edit asks only the row whose wkey changed, and the old answers never "
              "show meanwhile", stored == [0.875, 0.75, 0.25] and edited == [lamp] and meanwhile == [0.0] * 3,
              (stored, len(edited), meanwhile))
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
    finally:
        pipeline.forget(old + cells())
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
              errored in kept and ("Warm 0?", "", "", "", "") not in kept and engine.STATUS[asker.uid][0] == 'ERROR',
              (errored in kept, len(kept), engine.STATUS[asker.uid]))
    finally:
        pipeline.forget(keys)
        shutil.rmtree(folder, ignore_errors=True)


def phase_converted_last_shown():
    """k_default never names a converted v0.1.0 entry, which has no words key: a context showing one gets new words
    that no model answers, and shows neutral values, never another question's converted answer (§8.2)."""
    group = bpy.data.node_groups.new("Converted", 'GeometryNodeTree')
    group.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    group.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    asker = group.nodes.new(engine.ASK_ID)
    geometry = store(group, group.nodes.new("NodeGroupInput").outputs["Geometry"], asker.outputs["Answer"], "prob",
                     'FLOAT')
    group.links.new(geometry, group.nodes.new("NodeGroupOutput").inputs[0])
    user = user_of(group, "Converted")
    asker.entries = json.dumps([  # v0.1.0, schema 1
        {"key": "a", "gn": ["Is it cold?", ""], "t": 1.0, "a": {"mode": "YES_NO", "value": 0.125, "confidence": 0.75}},
        {"key": "b", "gn": ["Is it night?", ""], "t": 2.0, "a": {"mode": "YES_NO", "value": 0.875, "confidence": 0.75}}])
    got = []
    with asking(asker, []):
        for question in ("Is it night?", "Is it dark?"):
            asker.inputs["Question"].default_value = question
            engine.run_pass()
            got.append(table_read(user)["prob"][0])
    check("converted entries: a context shows its own, and with new words neutral values, never another question's "
          "converted answer", got == [0.875, 0.0], got)


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
    check("status: an empty Pick asks for its prompt first", empty == ('INFO', "Type a prompt"), empty)
    check("status: a node that reads a newer OpenAct's answers as words says why, rather than waiting",
          status[0] == 'ERROR' and "newer OpenAct" in status[1], status)


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


# ------------------------------------------ the materialised set, table objects, the per-element lookup

def phase_materialised():
    """§8.1: live entries first (at most 16, in context order), then pins in pin order, then warm ones while the
    total is below 4; laid out in store order. Ten successive edits of one context stay within 4 entries."""
    def entry(i, t=0.0, pinned=False, frames=()):
        return S.Entry(words_key=f"k{i}", gn=(f"Question {i}?", ""), t=t, pinned=pinned, frames=[list(f) for f in frames],
                       kind="ask", answer_type='YES_NO', single={"Answer": i / 32, "Yes": False, "Confidence": 0.5})

    def keys(entries, live, default=None):
        return [e.words_key for e in inner.materialised(entries, live, default)]

    many = [entry(i, t=i) for i in range(20)]
    got = keys(many, [e.gn for e in reversed(many)])
    check("materialised: at most 16, the first live contexts in context order, laid out in store order",
          got == [f"k{i}" for i in range(4, 20)], got)
    mixed = [entry(0, t=5), entry(1, pinned=True, frames=[(30, 40)]), entry(2, pinned=True, frames=[(10, 20)]),
             entry(3, t=9), entry(4, t=8), entry(5, pinned=True, frames=[(50, 60)])]
    warm = [entry(0, t=5), entry(1, t=9), entry(2, t=7), entry(3, t=8), entry(4, t=6)]
    crowded = [entry(i, t=i) for i in range(10)] + [entry(10 + p, pinned=True, frames=[(100 - p, 200)]) for p in range(12)]
    got = (keys(mixed, [mixed[0].gn]), keys(warm, [warm[0].gn]), keys(warm, [warm[0].gn], "k4"),
           keys(crowded, [e.gn for e in crowded[:10]]))
    check("materialised: live, then pins, then the most recent warm entries (the default first) up to 4 in total; "
          "above 16, the pins that start latest stay out",
          got == (["k0", "k1", "k2", "k5"], ["k0", "k1", "k2", "k3"], ["k0", "k1", "k3", "k4"],
                  [f"k{i}" for i in range(10)] + [f"k{i}" for i in range(16, 22)]), got)

    stored, sizes, rebuilds = [], [], []
    for i in range(10):  # ten successive Question edits of one context
        stored.append(entry(100 + i, t=100 + i))
        live = [stored[-1].gn]
        rebuilt = not inner.update_values(ask, S.NodeStore(stored), live)
        if rebuilt:
            inner.build(ask, S.NodeStore(stored), live)
        rebuilds.append(rebuilt and inner.update_values(ask, S.NodeStore(stored), live))
        sizes.append(sum(n.bl_idname == "FunctionNodeCompare" for n in ask.node_tree.nodes))
    back = keys(stored, [stored[0].gn])
    check("materialised: 10 successive edits keep at most 4 entries in the graph, one rebuild per edit; "
          "switching back materialises the stored entry",
          max(sizes) <= 4 and all(rebuilds) and "k100" in back and len(back) <= 4, (sizes, rebuilds, back))


class PickV1(Spec):
    """Pick as the 5-word specs declare it (§6.2, §8.2), built from inner's primitives only: the recipe for
    specs/pick.py. Context, Overrides and Key are absent from this interface, so the chain joins them as ""."""

    name, label = "pick", "Pick"
    WORDS = words = ("Prompt", "Context", "Items", "Overrides", "Key")
    per_element = (inner.Column("Probability", "prob", 'FLOAT', 0.0, linked_only=True),)
    iface = (SocketDef('INPUT', "Prompt", 'NodeSocketString', kind="word"),
             SocketDef('INPUT', "Items", 'NodeSocketString', kind="word"),
             SocketDef('INPUT', "ID", 'NodeSocketInt', kind="field", default_input='INDEX', hide_value=True),
             SocketDef('OUTPUT', "Index", 'NodeSocketInt'), SocketDef('OUTPUT', "Item", 'NodeSocketString'),
             SocketDef('OUTPUT', "Confidence", 'NodeSocketFloat'), SocketDef('OUTPUT', "Selection", 'NodeSocketBool'),
             SocketDef('OUTPUT', "Probability", 'NodeSocketFloat'))

    def interface(self, node):
        return list(self.iface)

    @staticmethod
    def build(b, gi, go, entries):
        k = b.match_chain(b.words(gi, PickV1.words), [e.gn for e in entries])
        picked = b.lookup(k, 'INT', -1, [e.single["Index"] for e in entries], go.inputs["Index"], 300)
        b.lookup(k, 'STRING', "", [e.single["Item"] for e in entries], go.inputs["Item"], 100)
        b.lookup(k, 'FLOAT', 0.0, [e.single["Confidence"] for e in entries], go.inputs["Confidence"], -100)
        b.selection(gi.outputs["ID"], picked, go.inputs["Selection"])


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
    for key, words in TABLE_WORDS.items():
        mesh = bpy.data.meshes.new(key)
        mesh.from_pydata([(i, 0, 0) for i in range(POINTS)], [], [])
        users[key] = bpy.data.objects.new(key, mesh)
        bpy.context.scene.collection.objects.link(users[key])
        users[key].modifiers.new("Table", 'NODES').node_group = tree_
    for key, words in TABLE_WORDS.items():
        set_words(users, idents, key, **words)
    return tree_, picker, users, idents


def set_words(users, idents, key, **words):
    mod = users[key].modifiers["Table"]
    for name, value in words.items():
        mod[idents[name]] = value
    users[key].update_tag()


def table_read(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        return {name: None if name not in mesh.attributes else
                [round(d.value, 6) if isinstance(d.value, float) else d.value for d in mesh.attributes[name].data]
                for name in ("prob", "sel", "idx")}
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
    inner.build(picker, S.NodeStore(), live)  # never answered, Probability linked: an empty table
    empty = tables.find(picker)
    check("table: before the first answer, an empty table object and neutral values everywhere",
          empty is not None and len(empty.data.vertices) == 0 and "oa_prob" in empty.data.attributes
          and reads(users, [None] * 3)[0] == {key: table_expected(None, TABLE_WORDS[key]["Shift"]) for key in users},
          reads(users, [None] * 3)[0])
    inner.build(picker, node_store, live)
    table = tables.find(picker)
    column = table and [round(d.value, 6) for d in table.data.attributes["oa_prob"].data]
    check("table: one loose vertex per flat row, an oa_prob column, named after the node, linked to no collection",
          table is not None and table.name == tables.PREFIX + picker.uid and table.get(tables.UID) == picker.uid
          and not table.users_collection and column == [0.25, 0.625, 0.125, 0.125, 0.25, 0.125, 0.5, 0.75, 0.25],
          column)
    got, want = reads(users, entries)
    check("per-element: 3 contexts through the joined-key chain each read their own rows at base + ID, and "
          "Selection = (ID == Index) AND (Index >= 0)", got == want, got)
    check("per-element: invalid IDs (-1 on TA's first point; past the row count on TB, where TC's rows follow) "
          "give the fallbacks", got["TA"]["prob"][0] == 0.0 and got["TA"]["sel"][0] is False
          and got["TB"]["prob"][4:] == [0.0, 0.0] and got["TC"]["prob"][2:] == [0.0] * 4, got)
    joins = [n for n in picker.node_tree.nodes if n.bl_idname == "GeometryNodeStringJoin"]
    empties = [n for n in picker.node_tree.nodes if n.bl_idname == "FunctionNodeInputString"]
    check("joined key: one Join Strings over the 5-word tuple, linked in reverse order; the absent Context, "
          "Overrides and Key are constant \"\" nodes, and every stored entry matches",
          len(joins) == 1 and len(joins[0].inputs["Strings"].links) == 5 and len(empties) == 3
          and [got[key]["idx"][0] for key in users] == [1, 3, 0], (len(joins), len(empties)))

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
          and len(tables.find(picker).data.vertices) == 10, (updated, got))
    return node_store, live


def phase_no_match(picker, users, idents, node_store, live):
    entries = node_store.entries
    before = pointers(picker)
    set_words(users, idents, "TA", Prompt="something to sit on, edited")
    edited = ("something to sit on, edited", "", TABLE_WORDS["TA"]["Items"], "", "")
    two = inner.k_default_key(["TA", "TB"])
    updated = [inner.update_values(picker, node_store, [edited, live[1]], two)]
    got = table_read(users["TA"]), table_read(users["TB"])
    check("no match, two live contexts: TA's unmatched words show neutral values, never another context's answers",
          two is None and got == (table_expected(None, -1), table_expected(entries[1], 0)), got)
    one = inner.k_default_key(["TA"])
    updated.append(inner.update_values(picker, node_store, [edited], one))
    got = table_read(users["TA"])
    check("no match, one live context: TA shows its own last answers", one == "TA"
          and got == table_expected(entries[0], -1), got)
    set_words(users, idents, "TA", Prompt="")
    hint = inner.k_default_key(["TA"], blocked=True)
    updated.append(inner.update_values(picker, node_store, [], hint))
    got = table_read(users["TA"])
    check("no match, a cleared Prompt (a hint): neutral values", hint is None and got == table_expected(None, -1), got)
    check("no match: every k_default change is a values-only update", updated == [True] * 3
          and pointers(picker) == before, updated)
    set_words(users, idents, "TA", Prompt=TABLE_WORDS["TA"]["Prompt"])
    inner.update_values(picker, node_store, live)


def phase_per_word(picker, users, idents, node_store, live):
    """A word containing SEP could make two different tuples join to the same key, so the node compares each word."""
    tricky = table_entry("TX", 0, [0.375], prompt="p\x1f", items="i")
    set_words(users, idents, "TA", Prompt="p\x1f", Items="i")  # tricky's words
    set_words(users, idents, "TB", Prompt="p", Items="\x1fi")  # joined, the same key as tricky's
    gns = [tricky.gn, ("p", "", "\x1fi", "", ""), live[2]]
    inner.build(picker, S.NodeStore([tricky, node_store.entries[2]]), gns)
    got = {key: table_read(obj) for key, obj in users.items()}
    check("joined key: a word containing \\x1f switches the node to per-word Compares, which still match",
          "\x1f".join(gns[0]) == "\x1f".join(gns[1])
          and not any(n.bl_idname == "GeometryNodeStringJoin" for n in picker.node_tree.nodes)
          and got == {"TA": table_expected(tricky, -1), "TB": table_expected(None, 0),
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
    check("values only: the node's own rows are back, columns it no longer writes are gone, evaluation sees them",
          restored and names == ["oa_prob"] and got == want, (restored, names, got))


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
    released = tables.find(picker) is None
    removed = tables.collect_garbage(list(engine.iter_nodes()))
    tree_.links.new(picker.outputs["Probability"], target)
    rebuilt = not inner.update_values(picker, node_store, live)
    inner.build(picker, node_store, live)
    got, want = reads(users, node_store.entries)
    check("Pick: with Probability unlinked, build() releases the table object and garbage removes it; linked "
          "again, a rebuild brings a new one", refused is False and released and removed == 1 and rebuilt
          and tables.find(picker) is not None and got == want, (refused, released, removed, rebuilt))


def phase_save_reload():
    """Last, because it reloads the file (§8.5): the table object, in no collection, is saved through the Object
    Info that reads it and survives Purge Unused Data; appending its tree brings a table along, renamed .001, and
    garbage collection, going by reference, keeps both."""
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
    then refuses the groups built before it, so the first pass after an upgrade or a file load rebuilds them."""
    asks = S.NodeStore([ask_entry("A", 0.875, 0.75)])
    inner.build(ask, asks)
    inner.LAYOUT += 1
    try:
        bumped = inner.update_values(ask, asks)
    finally:
        inner.LAYOUT -= 1
    check("layout version: a group laid out under another LAYOUT needs build()",
          bumped is False and inner.update_values(ask, asks), bumped)


def phase_tables():
    """The per-element path on synthetic tables, through a stand-in Pick (PickV1) on a real Pick node."""
    saved = inner.SPECS[engine.PICK_ID]
    inner.SPECS[engine.PICK_ID] = PickV1()
    try:
        tree_, picker, users, idents = table_scene()
        node_store, live = phase_table(tree_, picker, users, idents)
        phase_no_match(picker, users, idents, node_store, live)
        phase_per_word(picker, users, idents, node_store, live)
        phase_write_rows(picker, users, node_store, live)
        phase_table_copies(tree_, picker, users, node_store, live)
    finally:
        inner.SPECS[engine.PICK_ID] = saved


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
    return S.NodeStore([S.Entry(words_key=str(i), gn=(f"Question {i}?", f"Context {i}", "", "", ""), single={
        "Answer": (i + offset) / 40, "Yes": i % 2 == bool(offset), "Confidence": (i + 2 * offset) / 50})
        for i in range(count)])


def pick_store(count, offset=0):
    return S.NodeStore([S.Entry(words_key=str(i), gn=(f"Prompt {i}", "", "table, chair, lamp, rug, bookshelf", "", ""),
                                single={"Index": (i + offset) % 5, "Item": f"item {(i + offset) % 5}",
                                        "Confidence": (i + offset) / 40})
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
phase_converted()
phase_pass()
phase_pick_pass()
phase_converted_default()
phase_table_only()
phase_carry()
phase_live_set()
phase_converted_last_shown()
phase_statuses()
phase_single_value_sample()
phase_materialised()
phase_tables()
phase_layout_version()
if "--bench" in ARGS:
    chosen = os.environ.get("OPENACT_BENCH_SHAPE", "")
    phase_bench({chosen} if chosen in ("demo", "worst") else {"demo", "worst"})
phase_save_reload()  # last: it reloads the file
finish()
