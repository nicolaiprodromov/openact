"""Playback and the words at other frames (§5.5, §7.8) in a headless Blender, against the fake model server that the
add-on starts itself (OPENACT_BACKEND=fake), whose --record log counts every request it gets. The timeline plays while
engine._animation_playing() returns True: each scene.frame_set() then runs the playback path (frame_change_post), and
engine.tick() between frames pumps the answers, runs its work between frames and no pass:
- playback: a Rules node whose Outcomes CSV comes from a keyframed Menu Switch cycling through three Game of Life rule
  sets, two of them answered while the scene was stopped at their frames. Playing frames 1-20 shows each one's table
  and asks and writes nothing; at frame 21 the rule set never answered is asked at once, in one request, while the
  node shows the rules it showed last (its table so far, stale); the answer is written values only as it lands and
  shows from then on; no inner.build() and no pass run during playback; once it stops, the pass stores the answer;
  played again, nothing is asked or written;
- the Trigger: an Ask whose Question holds the frame (Value to String) asks at each new frame, shows its last answer
  (the default a miss shows, §8.2) until the frame's own lands and then that one, while an untriggered Ask beside it
  asks nothing and its table is never written; nothing is stored during playback, and all of it once it stops;
- Rules' cells in use (§5.8): a triggered Rules node whose geometry passes through it asks, while the timeline plays,
  each cell as its points first reach it and no other, with no inner.build(), and the pass after stores them;
- a simulation zone that adds an Ask's Answer to its state each frame, the Ask triggered from frame 3: the answer
  written at frame 3 is seen at frame 4, and the simulation continues without resetting (REDESIGN 2.5);
- stale questions: while the model is busy, a node whose words change every frame keeps only its newest question
  queued, dropping the older ones with a new generation, while a For Each Element zone keeps every element's question,
  those it still reports and each new one; once the model is free, only those are asked;
- storing: the answers given during playback are stored once it stops, evicted as any save evicts (MAX_ENTRIES at 3);
  a save made right after playback, before any pass, holds them too, and so does the file reopened;
- the budget: the frame handler's cost with nothing new and with a new question (TIMING lines); a For Each Element
  zone that brings 40 new word sets at one frame gets what fits in FRAME_BUDGET planned there and the rest at the
  ticks, which ask them all;
- renders never ask (§2.6): frames with new words, played while a render job runs or in a background session that
  renders the saved answers, ask and write nothing;
- the words a background render compares at the frame it renders (§7.8), read through the F-curves with no scene
  evaluation (engine.FrameWords): a keyframed Int modifier input's F-curve is found through the action slot's
  channelbag on 4.5 and 5.1 (Action.fcurves is gone in 5.x), evaluate() gives the keyed values, the words read at
  every frame are the ones the scene has there, and a keyed value beyond the input's hard range is read clamped, as
  the scene clamps it; a render's report (on_render_pre()) compares the words of its frame, not the ones the file
  holds, and asks and stores nothing;
- words that only setting the frame would tell, of a driven input, of an input an NLA strip animates, and under Time
  Stretching: they aren't read through the F-curves, so a render compares them as the file holds them;
- OpenAct's code calls no temp_override (§5.5).

Run:  blender -b --factory-startup --python tests/playback_blender.py
"""

import contextlib
import json
import os
import pathlib
import shutil
import socket
import sys
import tempfile
import time
import traceback

TMP = tempfile.mkdtemp(prefix="openact_playback_")
RECORD = os.path.join(TMP, "record.jsonl")
os.environ["OPENACT_BACKEND"] = "fake"
os.environ["OPENACT_SERVER_PYTHON"] = sys.executable  # the fake backend needs no runtime: Blender's Python runs it
os.environ["OPENACT_STATE_DIR"] = os.path.join(TMP, "state")  # the server log stays out of any shared runtime

import bpy  # noqa: E402
from bpy_extras import anim_utils  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import FAILURES, ROOT, check, drive, finish, sock, store, trigger  # noqa: E402
from openact import engine, inner, nodes, ops, pipeline, server, tables, trust, ui, watch  # noqa: E402
from openact import prefs as P  # noqa: E402
from openact import store as S  # noqa: E402
from openact.specs import SPECS  # noqa: E402

ASK, RULES = SPECS["GeometryNodeOpenActAsk"], SPECS["GeometryNodeOpenActRules"]
RULE_SETS = {"Life": "life_outcomes.csv", "HighLife": "highlife_outcomes.csv", "Day & Night": "daynight_outcomes.csv"}
GOL_QUESTION = "What happens to this cell in the next generation?"
for name in ("life_factors.csv", *RULE_SETS.values()):
    shutil.copy(os.path.join(ROOT, "tests", "fixtures", name), TMP)


def free_port():
    """A free local port, never one the user's session or another job uses."""
    while True:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        if port not in (18765, 18767, 18769):
            return port


PORT = free_port()
P._Defaults.port, P._Defaults.auto_start = PORT, True  # the add-on isn't enabled, so these are its preferences
server.FAKE_ARGS[:] = ["--record", RECORD]
for module in (ops, nodes, ui, trust, watch):
    module.register()
engine.register()  # timers don't run under --python: the test ticks
ASKED, TEXTS = [], {}  # (node uid, cell key) of every missing cell submitted, up or not; cell key -> its question
_submit = engine._submit


def submitted(job, missing):
    for cell in missing:
        ASKED.append((job.uid, cell.key))
        TEXTS[cell.key] = cell.question.get("instructions")
    return _submit(job, missing)


engine._submit = submitted
WRITES, BUILDS, COSTS = [], [], []  # the uid of each node whose table a write changed, of each build; handler seconds


def counted(module, name, into, wrote=bool):
    """Wrap module.name(node, ...): each call that wrote appends the node's uid to `into`."""
    original = getattr(module, name)

    def wrapper(node, *args, **kwargs):
        result = original(node, *args, **kwargs)
        if wrote(result):
            into.append(node.uid)
        return result

    setattr(module, name, wrapper)


for _name in ("write_rows", "append_rows", "write_value"):  # write_value() compares first: a new value only
    counted(tables, _name, WRITES)
counted(inner, "build", BUILDS, wrote=lambda _result: True)
_play_frame = engine._play_frame


def timed_frame(depsgraph):
    """engine._play_frame(), the playback path's work after a frame, timed into COSTS."""
    started = time.perf_counter()
    try:
        _play_frame(depsgraph)
    finally:
        COSTS.append(time.perf_counter() - started)


engine._play_frame = timed_frame


# ------------------------------------------------------------------------------ helpers

def records():
    """The fake server's --record log: one line per request it got."""
    try:
        with open(RECORD, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    except OSError:
        return []


def settle(timeout=90):
    """Tick until nothing is queued or in flight, no pass is due, and neither a node nor the playback path waits for
    the model."""
    return drive(engine, lambda: not pipeline.PENDING and not engine._dirty and not engine.QUESTIONS and not any(
        status.state in ("starting", "waiting", "filling") for status in engine.STATUS.values()), timeout)


@contextlib.contextmanager
def playing():
    """Inside, the timeline plays: engine._animation_playing() returns True, as a window's is_animation_playing does."""
    saved, engine._animation_playing = engine._animation_playing, lambda: True
    try:
        yield
    finally:
        engine._animation_playing = saved


def landed(timeout=30):
    """Tick until the playback path waits for no answer: each one that landed is written."""
    return drive(engine, lambda: not engine.QUESTIONS, timeout)


def asked_by(node, start):
    """The keys of the cells submitted for the node since ASKED[start]."""
    return [key for uid, key in ASKED[start:] if uid == node.uid]


def queued(node):
    """The keys of the node's cells in the queue, which the worker hasn't taken."""
    with pipeline._QUEUED:
        return [cell.key for jobs in pipeline.QUEUE for chunks in jobs for chunk in chunks if chunk.job.uid == node.uid
                for cell in chunk.cells]


@contextlib.contextmanager
def busy():
    """Inside, the model is busy: the worker takes no chunk, as during a render job. Nothing may tick meanwhile, as
    tick() publishes that no render job runs."""
    server.Worker.render_running.set()
    time.sleep(0.3)  # a worker waiting for work has seen it
    try:
        yield
    finally:
        server.Worker.render_running.clear()


def asks(question):
    """Ask's word tuple: (Question, Context, Answers, Overrides, Key)."""
    return (question, "", "", "", "")


def answer_of(node, question):
    """The Answer the node stores for the question, or None."""
    entry = next((e for e in S.load(node).entries if tuple(e.gn) == asks(question)), None)
    return entry.single["Answer"] if entry is not None else None


def enabled(sockets, name):
    return next(s for s in sockets if s.name == name and s.enabled)


def named(sockets, name):
    return next(s for s in sockets if s.name == name)


def numbered(tree, prefix, number, suffix=""):
    """A String socket: prefix, the Int socket `number` through Value to String (INT), and suffix, joined."""
    text = tree.nodes.new("FunctionNodeValueToString")
    text.data_type = 'INT'
    tree.links.new(number, enabled(text.inputs, "Value"))
    parts = []
    for part in (prefix, None, suffix):
        if part is None:
            parts.append(text.outputs["String"])
        elif part:
            constant_text = tree.nodes.new("FunctionNodeInputString")
            constant_text.string = part
            parts.append(constant_text.outputs[0])
    join = tree.nodes.new("GeometryNodeStringJoin")
    for part in reversed(parts):  # Join Strings puts the most recently linked item first
        tree.links.new(part, join.inputs["Strings"])
    return join.outputs[0]


def frame_number(tree):
    """The scene's frame as an Int socket: Scene Time's Frame through Float to Integer."""
    whole = tree.nodes.new("FunctionNodeFloatToInt")
    tree.links.new(tree.nodes.new("GeometryNodeInputSceneTime").outputs["Frame"], whole.inputs[0])
    return whole.outputs[0]


def after(tree, frame):
    """A Bool socket, on after this frame: Scene Time's Frame > frame + 0.5, a Trigger that turns on during playback."""
    compare = tree.nodes.new("FunctionNodeCompare")
    compare.data_type, compare.operation = 'FLOAT', 'GREATER_THAN'
    tree.links.new(tree.nodes.new("GeometryNodeInputSceneTime").outputs["Frame"], compare.inputs[0])
    compare.inputs[1].default_value = frame + 0.5
    return compare.outputs[0]


def timing(what, costs, budget):
    """Print the handler's cost per frame for `what` against its budget (§18.2), and check it within twice that. Returns
    the slowest frame's milliseconds."""
    slowest, mean = max(costs) * 1000, sum(costs) / len(costs) * 1000
    print(f"TIMING playback frame handler, {what}: {mean:.2f} ms per frame, {slowest:.2f} ms at most, over "
          f"{len(costs)} frames (budget {budget * 1000:g} ms)", flush=True)
    if slowest > budget * 1000:
        print(f"WARN playback frame handler, {what}: {slowest:.2f} ms, above its {budget * 1000:g} ms budget",
              flush=True)
    check(f"budget: the frame handler, {what}, within 2x its {budget * 1000:g} ms budget (§18.2)",
          slowest <= 2 * budget * 1000, f"{slowest:.2f} ms")
    return slowest


def reset(first=1, last=250):
    """An empty file between phases, as a render's report reads every node in it."""
    for blocks in (bpy.data.objects, bpy.data.meshes, bpy.data.node_groups, bpy.data.actions):
        for block in list(blocks):
            blocks.remove(block)
    engine.run_pass()
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = first, last
    scene.frame_set(first)
    return scene


def new_tree(name, *inputs):
    """A tree with these (name, socket type) inputs after Geometry: (tree, Group Input, Group Output)."""
    tree = bpy.data.node_groups.new(name, 'GeometryNodeTree')
    tree.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    for input_name, socket_type in inputs:
        tree.interface.new_socket(input_name, in_out='INPUT', socket_type=socket_type)
    tree.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    return tree, tree.nodes.new("NodeGroupInput"), tree.nodes.new("NodeGroupOutput")


def ident(tree, name):
    return next(item.identifier for item in tree.interface.items_tree
                if getattr(item, "in_out", "") == 'INPUT' and item.name == name)


def user(name, tree, mesh=None, **values):
    """An object whose modifier uses the tree, with modifier inputs by name."""
    if mesh is None:
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata([(0, 0, 0)], [], [])
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    mod = obj.modifiers.new("OpenAct", 'NODES')
    mod.node_group = tree
    for word, value in values.items():
        mod[ident(tree, word)] = value
    return obj


def path_of(obj, name):
    """The data path of a modifier input, as keyframes and drivers name it."""
    return f'modifiers["OpenAct"]["{ident(obj.modifiers["OpenAct"].node_group, name)}"]'


def keyframes(obj, name, keys):
    """Key the modifier input at each (frame, value); returns its F-curve."""
    mod, path = obj.modifiers["OpenAct"], path_of(obj, name)
    for frame, value in keys:
        mod[ident(mod.node_group, name)] = value
        obj.keyframe_insert(data_path=path, frame=frame)
    scene = bpy.context.scene
    scene.frame_set(scene.frame_current)  # the input holds the current frame's value again, not the last one keyed
    data = obj.animation_data
    return anim_utils.action_get_channelbag_for_slot(data.action, data.action_slot).fcurves.find(path)


def constant(curve):
    """Each key's value holds until the next key."""
    for point in curve.keyframe_points:
        point.interpolation = 'CONSTANT'
    return curve


def menu_switch(tree, items):
    """A String Menu Switch whose items {name: value} are in this order."""
    menu = tree.nodes.new("GeometryNodeMenuSwitch")
    menu.data_type = 'STRING'
    menu.enum_items.clear()
    for name in items:
        menu.enum_items.new(name)
    for slot, value in zip((s for s in menu.inputs if s.identifier not in ("Menu", "__extend__")), items.values()):
        slot.default_value = value
    return menu


def menu_value(obj, name, item):
    """The modifier's value for a menu item: its identifier, as the modifier's item list gives it."""
    mod = obj.modifiers["OpenAct"]
    return next(entry[-1] for entry in mod.id_properties_ui(ident(mod.node_group, name)).as_dict()["items"]
                if entry[1] == item)


def question_tree(name, count, *extra):
    """A tree whose Ask asks 'Question <Level>?' of `count` questions, Level being the modifier's Int input; extra:
    more (name, socket type) inputs. The geometry passes through the Ask, whose Trigger is on (§6.0). Returns (tree,
    Ask node, Group Input)."""
    tree, gi, go = new_tree(name, ("Level", 'NodeSocketInt'), *extra)
    switch = tree.nodes.new("GeometryNodeIndexSwitch")
    switch.data_type = 'STRING'
    while len(switch.index_switch_items) < count:
        switch.index_switch_items.new()
    for i, slot in enumerate(s for s in switch.inputs if s.identifier not in ("Index", "__extend__")):
        slot.default_value = f"Question {i}?"
    tree.links.new(gi.outputs["Level"], switch.inputs["Index"])
    ask = tree.nodes.new(ASK.idname)
    tree.links.new(switch.outputs[0], ask.inputs["Question"])
    tree.links.new(gi.outputs["Geometry"], go.inputs["Geometry"])
    trigger(ask)
    return tree, ask, gi


def level(obj):
    """The Level the object's modifier input holds now."""
    mod = obj.modifiers["OpenAct"]
    return mod[ident(mod.node_group, "Level")]


def scene_words(scene, obj, frames):
    """{words: frames} as Blender sets the Level input at each frame, from 'Question <Level>?'."""
    found = {}
    for frame in frames:
        scene.frame_set(frame)
        found.setdefault(f"Question {level(obj)}?", []).append(frame)
    scene.frame_set(frames[0])
    return found


def rendered(scene, frame, obj):
    """The Questions a background render of `frame` compares with the saved tables: on_render_pre() without the
    opt-in (§7.8). A render sets its frame without evaluating the original data, so obj's Level keeps the value the
    file holds now."""
    mod = obj.modifiers["OpenAct"]
    key = ident(mod.node_group, "Level")
    held = mod[key]
    scene.frame_current = frame
    mod[key] = held
    engine.RENDERED.clear()
    engine.allow_background_passes(False)
    try:
        engine.on_render_pre(scene)
    finally:
        engine.allow_background_passes(True)
    return sorted(words[0] for _uid, words in engine.RENDERED)


def read(obj, name="out"):
    """The attribute the tree stores, from the evaluated mesh."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        found = mesh.attributes.get(name)
        return [d.value for d in found.data] if found else None
    finally:
        evaluated.to_mesh_clear()


# ------------------------------------------------------------------------------ playback

def life_scene():
    """A Rules node whose Outcomes CSV is the Menu Switch item that the modifier's 'Rule set' menu chooses, as is the
    end of its Context, so each rule set has answers of its own; 18 points, one per cell, store its Outcome as out."""
    tree, gi, go = new_tree("Life", ("Rule set", 'NodeSocketMenu'))
    menu = menu_switch(tree, {name: os.path.join(TMP, csv) for name, csv in RULE_SETS.items()})
    tree.links.new(gi.outputs["Rule set"], menu.inputs["Menu"])
    context, prefix = tree.nodes.new("GeometryNodeStringJoin"), tree.nodes.new("FunctionNodeInputString")
    prefix.string = "Rules from "
    tree.links.new(menu.outputs[0], context.inputs["Strings"])
    tree.links.new(prefix.outputs[0], context.inputs["Strings"])  # the last linked comes first
    rules = tree.nodes.new(RULES.idname)
    rules.inputs["Question"].default_value = GOL_QUESTION
    rules.inputs["Factors"].default_value = os.path.join(TMP, "life_factors.csv")
    tree.links.new(menu.outputs[0], rules.inputs["Outcomes"])
    tree.links.new(context.outputs[0], rules.inputs["Context"])
    mesh = bpy.data.meshes.new("Life")
    mesh.from_pydata([(i * 0.1, 0, 0) for i in range(18)], [], [])  # cell order: state, then neighbours
    mesh.attributes.new("in_state", 'BOOLEAN', 'POINT').data.foreach_set("value", [i >= 9 for i in range(18)])
    mesh.attributes.new("in_neighbours", 'INT', 'POINT').data.foreach_set("value", [i % 9 for i in range(18)])
    obj = user("Life", tree, mesh)
    obj.modifiers["OpenAct"][ident(tree, "Rule set")] = menu_value(obj, "Rule set", "Life")
    trigger(rules)  # the geometry passes through it until wire_life() stores its Outcome
    return tree, rules, obj, gi, go


def wire_life(tree, rules, gi, go):
    """The factor sockets the pass gave the node <- the points' in_ attributes; its Outcome -> out."""
    for name, data_type in (("state", 'BOOLEAN'), ("neighbours", 'INT')):
        attribute = tree.nodes.new("GeometryNodeInputNamedAttribute")
        attribute.data_type = data_type
        attribute.inputs["Name"].default_value = f"in_{name}"
        tree.links.new(next(s for s in attribute.outputs if s.name == "Attribute" and s.enabled), rules.inputs[name])
    tree.links.new(store(tree, gi.outputs["Geometry"], rules.outputs["Outcome"], "out", 'INT'), go.inputs["Geometry"])


def rule_set(entry):
    return next(name for name, csv in RULE_SETS.items() if entry.gn[3] == os.path.join(TMP, csv))


def phase_playback():
    """A Rules node whose rule set a keyframed menu picks, Life at frames 1-10, HighLife at 11-20 and Day & Night at
    21-30: Life and HighLife are answered while the scene is stopped at their frames, Day & Night during playback."""
    scene = reset(1, 30)
    tree, rules, obj, gi, go = life_scene()
    for frame, name in ((1, "Life"), (11, "HighLife"), (21, "Day & Night")):
        keyframes(obj, "Rule set", [(frame, menu_value(obj, "Rule set", name))])
    first = settle()
    wire_life(tree, rules, gi, go)
    engine.schedule(immediate=True)
    settle()
    for frame in (11, 1):  # stopped at HighLife's frames, then at Life's again
        scene.frame_set(frame)
        engine.schedule(immediate=True)
        settle()
    table_of = {rule_set(entry): entry.rows.v["outcome"] for entry in S.load(rules).entries}
    status = engine.STATUS.get(rules.uid)
    check("playback: while the scene is stopped, the pass answers the rule set of the frame shown, Life and then "
          "HighLife, each with a table of its own (the fake answers depend on the state)", first
          and sorted(table_of) == ["HighLife", "Life"] and table_of["Life"] != table_of["HighLife"]
          and read(obj) == table_of["Life"], (sorted(table_of), status and status.text))
    start, requests, writes, builds, passes = len(ASKED), len(records()), len(WRITES), len(BUILDS), engine.passes
    shown, quiet, asked, stale, meanwhile, done = {}, None, 0, None, None, False
    with playing():
        for frame in range(1, 31):
            scene.frame_set(frame)
            if frame == 21:  # no tick yet: its answer can't have landed
                asked, meanwhile = len(asked_by(rules, start)), read(obj)
                stale = next((e for e in engine.TABLES[rules.uid].entries if rule_set(e) == "Day & Night"), None)
                done = landed()
            shown[frame] = read(obj)
            if frame == 20:
                quiet = (len(asked_by(rules, start)), len(records()) - requests, len(WRITES) - writes)
    night = next((e for e in engine.TABLES[rules.uid].entries if rule_set(e) == "Day & Night"), None)
    during = (len(records()) - requests, len(BUILDS) - builds, engine.passes - passes, len(S.load(rules).entries))
    check("playback: frames 1-20 show the stored tables of Life and HighLife, and ask and write nothing: answered "
          "words cost no request", quiet == (0, 0, 0)
          and all(shown[f] == table_of["Life" if f <= 10 else "HighLife"] for f in range(1, 21)), quiet)
    check("playback: at frame 21 the node asks for Day & Night at once, its 18 cells, and shows meanwhile the rules it "
          "showed last, HighLife's: its table so far, stale, written values only", asked == 18 and stale is not None
          and stale.rows.src == "s" * 18 and meanwhile == table_of["HighLife"], (asked, stale and stale.rows.src))
    check("playback: the answer, one request, is written values only as it lands, and frames 21-30 show Day & Night's "
          "own table; no inner.build() and no pass ran during playback, and nothing was stored",
          done and night is not None and night.rows.src == "m" * 18 and night.rows.v["outcome"] not in table_of.values()
          and all(shown[f] == night.rows.v["outcome"] for f in range(21, 31)) and during == (1, 0, 0, 2),
          (done, night and night.rows.src, during))
    finished = settle()
    stored = {rule_set(entry): entry for entry in S.load(rules).entries}
    check("playback: once it stops, the pass stores Day & Night's table, which shows, and asks nothing more", finished
          and sorted(stored) == ["Day & Night", "HighLife", "Life"] and night is not None
          and stored["Day & Night"].rows.v["outcome"] == night.rows.v["outcome"]
          and read(obj) == night.rows.v["outcome"] and len(records()) - requests == 1,
          (sorted(stored), len(records()) - requests))
    requests, writes, costs = len(records()), len(WRITES), len(COSTS)
    again = {}
    with playing():
        for frame in range(1, 31):
            scene.frame_set(frame)
            engine.tick()
            again[frame] = read(obj)
    expect = {f: stored["Life" if f <= 10 else "HighLife" if f <= 20 else "Day & Night"].rows.v["outcome"]
              for f in range(1, 31) if len(stored) == 3}
    check("playback: played again, each rule set shows its stored table, and nothing is asked or written",
          again == expect and len(records()) == requests and len(WRITES) == writes,
          (len(records()) - requests, WRITES[writes:]))
    timing("with nothing new (a Rules node, three rule sets stored)", COSTS[costs:], engine.FRAME_BUDGET)


def phase_trigger():
    """Two Asks whose Question holds the frame, 'Is frame <N> even?' with its Trigger on and 'Is frame <N> odd?' with
    none, both evaluated: their Answers are stored as even and odd."""
    scene = reset(1, 5)
    tree, gi, go = new_tree("Frames")
    even, odd = tree.nodes.new(ASK.idname), tree.nodes.new(ASK.idname)
    tree.links.new(numbered(tree, "Is frame ", frame_number(tree), " even?"), even.inputs["Question"])
    tree.links.new(numbered(tree, "Is frame ", frame_number(tree), " odd?"), odd.inputs["Question"])
    geometry = store(tree, gi.outputs["Geometry"], even.outputs["Answer"], "even", 'FLOAT')
    tree.links.new(store(tree, geometry, odd.outputs["Answer"], "odd", 'FLOAT', 300), go.inputs["Geometry"])
    trigger(even)
    obj = user("Frames", tree)
    settle()  # stopped at frame 1: the pass answers the even question there
    start, requests, writes, builds, passes, costs = (len(ASKED), len(records()), len(WRITES), len(BUILDS),
                                                      engine.passes, len(COSTS))
    keys = tables.read(odd, tables.KEY).tolist()
    before, after_landing, kept = {}, {}, None
    with playing():
        for frame in range(1, 6):
            scene.frame_set(frame)
            before[frame] = read(obj, "even")[0]
            landed()
            after_landing[frame] = read(obj, "even")[0]
        kept = len(S.load(even).entries)
    new_costs = COSTS[costs + 1:]  # frames 2-5, each with a new question
    during = (len(records()) - requests, len(asked_by(even, start)), len(BUILDS) - builds, engine.passes - passes)
    finished = settle()
    answers = {frame: answer_of(even, f"Is frame {frame} even?") for frame in range(1, 6)}
    check("trigger: during playback, an untriggered Ask asks nothing and its table is never written, whatever its "
          "words", not asked_by(odd, start) and odd.uid not in WRITES[writes:]
          and tables.read(odd, tables.KEY).tolist() == keys, (asked_by(odd, start), WRITES[writes:]))
    check("trigger: a triggered Ask whose Question holds the frame asks at each new frame, 4 cells in 4 requests for "
          "frames 2-5 (frame 1's was answered while stopped), with no inner.build() and no pass",
          during == (4, 4, 0, 0),
          during)
    check("trigger: at a new frame it shows its last answer, the default a miss shows on a node with one context, then "
          "the frame's own once it lands", finished and None not in answers.values()
          and all(abs(after_landing[f] - answers[f]) < 1e-6 for f in range(1, 6))
          and all(abs(before[f] - answers[f - 1]) < 1e-6 for f in range(2, 6)), (before, after_landing, answers))
    check("storing: nothing is stored during playback, and once it stops the pass stores the 4 answers given during it",
          kept == 1 and {tuple(e.gn) for e in S.load(even).entries}
          == {asks(f"Is frame {frame} even?") for frame in range(1, 6)}, (kept, len(S.load(even).entries)))
    timing("with a new question at each frame (planned and asked)", new_costs, engine.FRAME_BUDGET)


HEAT = ("factor,kind,level,value,when\nheat,amount,cold,,Cold.\nheat,amount,mild,,Mild.\nheat,amount,warm,,Warm.\n"
        "heat,amount,hot,,Hot.\nlit,flag,dark,0,Dark.\nlit,flag,bright,1,Bright.\n")  # 8 cells: heat * lit


def phase_rules_in_use():
    """A Rules node whose geometry passes through it to the modifier's output, its heat the frame / 10 (Scene Time)
    and its lit flag unlinked (dark), triggered: its points move from cold to hot while the timeline plays."""
    scene = reset(1, 9)
    path = os.path.join(TMP, "heat_factors.csv")
    pathlib.Path(path).write_text(HEAT, encoding="utf-8")
    tree, gi, go = new_tree("Heat")
    rules = tree.nodes.new(RULES.idname)
    rules.inputs["Question"].default_value = "Which one?"
    rules.inputs["Factors"].default_value = path
    rules.inputs["Outcomes"].default_value = "A: the first; B: the second"
    tree.links.new(gi.outputs["Geometry"], rules.inputs["Geometry"])
    tree.links.new(store(tree, rules.outputs["Geometry"], rules.outputs["Outcome"], "out", 'INT'), go.inputs["Geometry"])
    switch = tree.nodes.new("FunctionNodeInputBool")  # its Trigger, on
    switch.boolean = True
    tree.links.new(switch.outputs[0], rules.inputs["Trigger"])
    obj = user("Heat", tree)
    settle()  # the pass gives the node its factor sockets from the Factors CSV
    tenth = tree.nodes.new("ShaderNodeMath")
    tenth.operation = 'DIVIDE'
    tree.links.new(tree.nodes.new("GeometryNodeInputSceneTime").outputs["Frame"], tenth.inputs[0])
    tenth.inputs[1].default_value = 10.0
    tree.links.new(tenth.outputs[0], rules.inputs["heat"])
    obj.update_tag()
    settle()  # stopped at frame 1: the pass lays out the factors, then asks the one cell in use, cold|dark
    names, stopped = S.load(rules).entries[0].rows.name, S.load(rules).entries[0].rows.src  # rows in cell order
    start, builds = len(ASKED), len(BUILDS)
    with playing():
        for frame in range(1, 10):
            scene.frame_set(frame)
            landed()
    during = (len(asked_by(rules, start)), len(BUILDS) - builds)
    settle()
    src = S.load(rules).entries[0].rows.src
    answered = sorted(name for name, letter in zip(names, src) if letter == "m")
    check("rules in use: stopped at frame 1, the node asks only the cell its points read, cold|dark",
          sorted(name for name, letter in zip(names, stopped) if letter == "m") == ["cold|dark"], (names, stopped))
    check("rules in use: during playback it asks each cell as its points first reach it, mild, warm then hot, 3 cells "
          "in all, never a bright one, and builds nothing", during == (3, 0), during)
    check("rules in use: once playback stops, the pass stores the 4 dark cells answered, the bright ones never asked",
          answered == ["cold|dark", "hot|dark", "mild|dark", "warm|dark"], (answered, src))


def phase_simulation():
    """A simulation zone that counts its steps and adds an Ask's Answer to a sum at each step; the Ask's Trigger is on
    after frame 2 (Scene Time)."""
    scene = reset(1, 8)
    tree, gi, go = new_tree("Sim")
    ask = tree.nodes.new(ASK.idname)
    ask.inputs["Question"].default_value = "Is the lamp of the simulation on?"
    tree.links.new(after(tree, 2), ask.inputs["Trigger"])
    begin, end = tree.nodes.new("GeometryNodeSimulationInput"), tree.nodes.new("GeometryNodeSimulationOutput")
    begin.pair_with_output(end)
    end.state_items.new('INT', "Step")
    end.state_items.new('FLOAT', "Sum")
    step = tree.nodes.new("FunctionNodeIntegerMath")
    step.operation, step.inputs[1].default_value = 'ADD', 1
    tree.links.new(named(begin.outputs, "Step"), step.inputs[0])
    tree.links.new(step.outputs[0], named(end.inputs, "Step"))
    add = tree.nodes.new("ShaderNodeMath")
    add.operation = 'ADD'
    tree.links.new(named(begin.outputs, "Sum"), add.inputs[0])
    tree.links.new(ask.outputs["Answer"], add.inputs[1])
    tree.links.new(add.outputs[0], named(end.inputs, "Sum"))
    tree.links.new(gi.outputs["Geometry"], named(begin.inputs, "Geometry"))
    tree.links.new(named(begin.outputs, "Geometry"), named(end.inputs, "Geometry"))
    geometry = store(tree, named(end.outputs, "Geometry"), named(end.outputs, "Step"), "step", 'INT')
    tree.links.new(store(tree, geometry, named(end.outputs, "Sum"), "sum", 'FLOAT', 300), go.inputs["Geometry"])
    obj = user("Sim", tree)
    settle()  # frame 1: the Trigger is off, so nothing is asked
    requests, stopped = len(records()), S.load(ask).entries
    got, done = {}, False
    with playing():
        for frame in range(1, 9):
            scene.frame_set(frame)
            got[frame] = (read(obj, "step")[0], read(obj, "sum")[0])
            if frame == 3:  # its answer is written after this frame's evaluation, which nothing repeats
                done = landed()
    found = engine.TABLES[ask.uid].answered.get(asks("Is the lamp of the simulation on?"))
    answer = found.single["Answer"] if found is not None else None
    check("simulation: triggered after frame 2 inside a simulation's step, the Ask asks during playback at frame 3; "
          "the answer written at frame 3 is seen at frame 4, the sum growing by it at each step after, and the "
          "simulation continues without resetting, its step counting every frame", not stopped and done
          and answer is not None
          and len(records()) - requests == 1 and [got[f][0] for f in range(1, 9)] == list(range(1, 9))
          and all(abs(got[f][1]) < 1e-6 for f in range(1, 4))
          and all(abs(got[f][1] - (f - 3) * answer) < 1e-5 for f in range(4, 9)), (got, answer))
    settle()


def phase_stale():
    """While the model is busy: an Ask whose Question holds the frame, and a For Each Element zone of 3 points whose Ask
    asks 'Is item <i> lit?', item 0 at each frame anew. Both Triggers are on after frame 1."""
    scene = reset(1, 4)
    tree, gi, go = new_tree("Late")
    late = tree.nodes.new(ASK.idname)
    tree.links.new(numbered(tree, "Is it late at frame ", frame_number(tree), "?"), late.inputs["Question"])
    tree.links.new(after(tree, 1), late.inputs["Trigger"])
    tree.links.new(gi.outputs["Geometry"], late.inputs["Geometry"])
    tree.links.new(late.outputs["Geometry"], go.inputs["Geometry"])
    user("Late", tree)
    items, zgi, zgo = new_tree("Items")
    begin = items.nodes.new("GeometryNodeForeachGeometryElementInput")
    end = items.nodes.new("GeometryNodeForeachGeometryElementOutput")
    begin.pair_with_output(end)
    item = items.nodes.new(ASK.idname)
    first = items.nodes.new("FunctionNodeCompare")
    first.data_type, first.operation = 'INT', 'EQUAL'
    items.links.new(begin.outputs["Index"], sock(first.inputs, "A_INT"))
    choose = items.nodes.new("GeometryNodeSwitch")
    choose.input_type = 'STRING'
    items.links.new(first.outputs[0], sock(choose.inputs, "Switch"))
    items.links.new(numbered(items, "Is item ", begin.outputs["Index"], " lit?"), sock(choose.inputs, "False"))
    items.links.new(numbered(items, "Is item 0 lit at frame ", frame_number(items), "?"), sock(choose.inputs, "True"))
    items.links.new(choose.outputs[0], item.inputs["Question"])
    items.links.new(after(items, 1), item.inputs["Trigger"])
    items.links.new(zgi.outputs["Geometry"], begin.inputs["Geometry"])
    items.links.new(begin.outputs["Element"], item.inputs["Geometry"])
    items.links.new(item.outputs["Geometry"], sock(end.inputs, "Generation_0"))
    items.links.new(sock(end.outputs, "Generation_0"), zgo.inputs["Geometry"])
    mesh = bpy.data.meshes.new("Items")
    mesh.from_pydata([(i, 0, 0) for i in range(3)], [], [])
    user("Items", items, mesh)
    settle()  # frame 1: both Triggers are off
    requests, generations = len(records()), (engine.GENERATIONS.get(late.uid, 0), engine.GENERATIONS.get(item.uid, 0))
    budget, engine.FRAME_BUDGET = engine.FRAME_BUDGET, 1.0  # every question planned at its frame: nothing ticks here
    try:
        with busy(), playing():
            for frame in (2, 3, 4):
                scene.frame_set(frame)
            waiting = ([TEXTS.get(key) for key in queued(late)], sorted(TEXTS.get(key) for key in queued(item)),
                       engine.GENERATIONS.get(late.uid, 0) - generations[0],
                       engine.GENERATIONS.get(item.uid, 0) - generations[1])
    finally:
        engine.FRAME_BUDGET = budget
    check("stale: while the model is busy, a node whose words change every frame keeps only its newest question "
          "queued, frame 4's: each new one dropped the older ones with a new generation (2 bumps)",
          waiting[0] == ["Is it late at frame 4?"] and waiting[2] == 2, waiting)
    check("stale: a For Each Element zone keeps every element's question while it reports any of them: items 1 and 2 "
          "and item 0's at frames 2, 3 and 4, and its generation stays", waiting[1] == sorted(
              ["Is item 1 lit?", "Is item 2 lit?"] + [f"Is item 0 lit at frame {f}?" for f in (2, 3, 4)])
          and waiting[3] == 0, waiting)
    with playing():
        done = landed()
    check("stale: once the model is free, only the questions kept are asked, 6 requests", done
          and len(records()) - requests == 6, len(records()) - requests)
    settle()


def phase_budget():
    """A For Each Element zone of 40 points whose Ask asks 'Is point <i> bright?', triggered after frame 1: frame 2
    brings 40 new word sets at once."""
    scene = reset(1, 3)
    tree, gi, go = new_tree("Points")
    begin = tree.nodes.new("GeometryNodeForeachGeometryElementInput")
    end = tree.nodes.new("GeometryNodeForeachGeometryElementOutput")
    begin.pair_with_output(end)
    ask = tree.nodes.new(ASK.idname)
    tree.links.new(numbered(tree, "Is point ", begin.outputs["Index"], " bright?"), ask.inputs["Question"])
    tree.links.new(after(tree, 1), ask.inputs["Trigger"])
    tree.links.new(gi.outputs["Geometry"], begin.inputs["Geometry"])
    tree.links.new(begin.outputs["Element"], ask.inputs["Geometry"])
    tree.links.new(ask.outputs["Geometry"], sock(end.inputs, "Generation_0"))
    tree.links.new(sock(end.outputs, "Generation_0"), go.inputs["Geometry"])
    mesh = bpy.data.meshes.new("Points")
    mesh.from_pydata([(i, 0, 0) for i in range(40)], [], [])
    user("Points", tree, mesh)
    settle()  # frame 1: the Trigger is off
    start, requests, costs = len(ASKED), len(records()), len(COSTS)
    with playing():
        scene.frame_set(2)
        mine = [question for key, question in engine.QUESTIONS.items() if key[0] == ask.uid]
        planned, left = sum(q.job is not None for q in mine), sum(q.job is None for q in mine)
        done = landed(90)
    cost = COSTS[costs]
    print(f"TIMING playback frame handler, 40 new word sets at one frame: {cost * 1000:.2f} ms, {planned} planned "
          f"there, {left} left to the ticks (budget {engine.FRAME_BUDGET * 1000:g} ms)", flush=True)
    # The last question it plans starts within the budget, and may end past it: within twice the budget (§18.2)
    check("budget: of 40 new word sets at one frame, the frame handler plans some and, once its budget is spent, "
          "leaves the rest to the ticks, which plan and ask them all: 40 requests", planned >= 1
          and planned + left == 40 and (left == 0 or cost >= engine.FRAME_BUDGET) and cost <= 2 * engine.FRAME_BUDGET
          and done and len(asked_by(ask, start)) == 40 and len(records()) - requests == 40,
          (planned, left, f"{cost * 1000:.2f} ms", len(records()) - requests))
    settle()


def phase_render():
    """Renders never ask (§2.6): an Ask whose Question holds the frame, triggered, played while a render job runs, and
    in a background session that renders the saved answers only (a farm's blender -b -a)."""
    scene = reset(1, 6)
    tree, gi, go = new_tree("Render")
    ask = tree.nodes.new(ASK.idname)
    tree.links.new(numbered(tree, "Is frame ", frame_number(tree), " rendered?"), ask.inputs["Question"])
    tree.links.new(store(tree, gi.outputs["Geometry"], ask.outputs["Answer"], "answer", 'FLOAT'), go.inputs["Geometry"])
    trigger(ask)
    user("Render", tree)
    settle()  # stopped at frame 1: answered
    start, requests, writes = len(ASKED), len(records()), len(WRITES)
    rendering, engine._rendering = engine._rendering, lambda: True  # stands in for is_job_running('RENDER')
    try:
        with playing():
            for frame in (2, 3):
                scene.frame_set(frame)
                drive(engine, lambda: False, 0.5)  # ticks while the render job runs
    finally:
        engine._rendering = rendering
        server.Worker.render_running.clear()  # tick() published the render job for the decision worker
    job = (asked_by(ask, start), len(records()) - requests, WRITES[writes:], len(engine.QUESTIONS))
    engine.allow_background_passes(False)  # the saved answers only (§7.8)
    try:
        with playing():
            for frame in (4, 5):
                scene.frame_set(frame)
                engine.on_render_pre(scene)
                drive(engine, lambda: False, 0.5)
    finally:
        engine.allow_background_passes(True)
    farm = (asked_by(ask, start), len(records()) - requests, WRITES[writes:], len(engine.QUESTIONS))
    check("render: frames with new words played while a render job runs ask and write nothing, and no tick writes",
          job == ([], 0, [], 0), job)
    check("render: a background session that renders the saved answers asks and writes nothing at the frames it "
          "renders, playing or not", farm == ([], 0, [], 0), farm)
    settle()


def phase_storing():
    """An Ask whose Question holds the frame, triggered: played over frames 2-7 with MAX_ENTRIES at 3, then over frames
    8 and 9 and saved at once, before any pass, and reopened."""
    scene = reset(1, 9)
    tree, gi, go = new_tree("Keep")
    ask = tree.nodes.new(ASK.idname)
    ask.name = "Keep"
    tree.links.new(numbered(tree, "Is frame ", frame_number(tree), " kept?"), ask.inputs["Question"])
    tree.links.new(store(tree, gi.outputs["Geometry"], ask.outputs["Answer"], "answer", 'FLOAT'), go.inputs["Geometry"])
    trigger(ask)
    user("Keep", tree)
    settle()  # stopped at frame 1: the pass answers frame 1's question

    def frames(node):
        return sorted(int(entry.gn[0].split()[2]) for entry in S.load(node).entries)

    kept, S.MAX_ENTRIES = S.MAX_ENTRIES, 3
    try:
        with playing():
            for frame in range(2, 8):
                scene.frame_set(frame)
                landed()
        settle()  # the pass after playback, at frame 7
        evicted = frames(ask)
    finally:
        S.MAX_ENTRIES = kept
    check("storing: once playback stops, the pass stores the answers given during it and evicts as any save does: with "
          "MAX_ENTRIES at 3, the node keeps the frame shown (7, live) and the 3 others used last (4, 5 and 6)",
          evicted == [4, 5, 6, 7], evicted)
    with playing():
        for frame in (8, 9):
            scene.frame_set(frame)
            landed()
    blend = os.path.join(TMP, "keep.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend)  # at once: no pass has run since playback
    saved = frames(ask)
    bpy.ops.wm.open_mainfile(filepath=blend)
    ask, obj = bpy.data.node_groups["Keep"].nodes["Keep"], bpy.data.objects["Keep"]
    reopened, shown = frames(ask), {}
    for frame in (8, 5):
        bpy.context.scene.frame_set(frame)
        shown[frame] = read(obj, "answer")[0]
    answers = {frame: answer_of(ask, f"Is frame {frame} kept?") for frame in (8, 5)}
    check("storing: a save made right after playback, before any pass, holds the answers it gave (frames 8 and 9), and "
          "the file reopened shows them", saved == reopened == [4, 5, 6, 7, 8, 9] and None not in answers.values()
          and all(abs(shown[f] - answers[f]) < 1e-6 for f in shown), (saved, reopened, shown, answers))
    bpy.ops.wm.read_homefile(use_empty=True)  # a new file, created in this session, for the phases after


# ------------------------------------------------------------------------------ words at other frames

def phase_channelbag():
    """A keyframed Int modifier input, read at other frames as a background render reads it (§7.8)."""
    scene = reset(1, 11)
    tree, ask, _gi = question_tree("Levels", 6)
    obj = user("Levels", tree, Level=0)
    curve = keyframes(obj, "Level", [(1, 0), (11, 5)])
    found, why = engine._animation(obj, path_of(obj, "Level"))
    action = obj.animation_data.action
    check("channelbag: a keyframed Int modifier input's F-curve is found through the action slot's channelbag "
          "(Action.fcurves exists only before 5.0), and evaluate() gives the keyed values",
          found is not None and found == curve and why == ""
          and hasattr(action, "fcurves") == (bpy.app.version < (5, 0))
          and (found.evaluate(1), found.evaluate(11)) == (0.0, 5.0), (found, why, hasattr(action, "fcurves")))
    counts = {"frames": 0, "updates": 0}

    def on_frame(*_args):
        counts["frames"] += 1

    def on_update(*_args):
        counts["updates"] += 1

    handlers = ((bpy.app.handlers.frame_change_pre, on_frame), (bpy.app.handlers.depsgraph_update_post, on_update))
    for handler, fn in handlers:
        handler.append(fn)
    try:
        reader = engine.FrameWords(scene)
        fcurves = reader.read_curves()
        after = scene.frame_current
    finally:
        for handler, fn in handlers:
            handler.remove(fn)
    words = {frame: gn[0] for seen in reader.seen.values() for gn, frames in seen.items() for frame in frames}
    truth = {}
    for frame in range(1, 12):  # what Blender sets at each frame
        scene.frame_set(frame)
        truth[frame] = f"Question {level(obj)}?"
    scene.frame_set(1)
    check("channelbag: the F-curves give the words at every frame with no scene evaluation (no frame change, no "
          "depsgraph update, the frame unchanged), and they are the words the scene has there", fcurves
          and counts == {"frames": 0, "updates": 0} and after == 1 and words == truth
          and all(words[f] == f"Question {int(curve.evaluate(f))}?" for f in truth), (reader.reason, counts, words))
    settle()  # the words at frame 1 are answered: a render compares a node that has stored answers
    start, entries = len(records()), ask.entries
    compared = rendered(scene, 11, obj)
    scene.frame_set(1)
    check("render: a background render of frame 11 compares the words there, read through the F-curves (Question "
          "5?), not the ones the file holds (Question 0?), and asks and stores nothing", compared == ["Question 5?"]
          and len(records()) == start and ask.entries == entries, compared)


def phase_ranges():
    """A keyed value beyond the input's hard range, and Time Stretching, which plays animation at other times."""
    scene = reset(1, 10)
    tree, _ask, _gi = question_tree("Ranges", 11)
    obj = user("Ranges", tree, Level=0)
    obj.modifiers["OpenAct"].id_properties_ui(ident(tree, "Level")).update(min=0, max=3)
    keyframes(obj, "Level", [(1, 0), (10, 9)])
    reader = engine.FrameWords(scene)
    fcurves = reader.read_curves()
    words = {gn[0]: frames for seen in reader.seen.values() for gn, frames in seen.items()}
    truth = scene_words(scene, obj, list(range(1, 11)))
    check("ranges: a keyed value beyond the input's hard range (0-3) is read clamped, as the scene sets it",
          fcurves and words == truth and max(truth) == "Question 3?", (words, truth))
    obj.modifiers["OpenAct"].id_properties_ui(ident(tree, "Level")).update(min=-2 ** 31, max=2 ** 31 - 1)
    keyframes(obj, "Level", [(1, 1), (10, 10)])
    settle()  # a render compares a node that has stored answers
    render = scene.render
    render.frame_map_old, render.frame_map_new = 100, 50
    try:
        truth = scene_words(scene, obj, list(range(1, 11)))
        held = f"Question {level(obj)}?"
        reader = engine.FrameWords(scene)
        fcurves = reader.read_curves()
        compared = rendered(scene, 5, obj)
    finally:
        render.frame_map_old = render.frame_map_new = 100
        scene.frame_set(1)
    there = next(words for words, frames in truth.items() if 5 in frames)
    check("ranges: under Time Stretching (100 -> 50) the words aren't read through the F-curves, so a render of "
          "frame 5 compares them as the file holds them, not as the scene has them there", not fcurves
          and reader.reason == "Time Stretching is on" and compared == [held] and there != held,
          (reader.reason, compared, held, there))


def phase_driver():
    """A driven input: only setting the frame tells its value there."""
    scene = reset(1, 29)
    tree, _ask, _gi = question_tree("Driven", 3)
    obj = user("Driven", tree, Level=0)
    driver = obj.driver_add(path_of(obj, "Level")).driver
    driver.type, driver.expression = 'SCRIPTED', "frame // 10"
    scene.frame_set(25)
    there = level(obj)
    scene.frame_set(7)
    settle()
    reader = engine.FrameWords(scene)
    fcurves = reader.read_curves()
    compared = rendered(scene, 25, obj)
    scene.frame_set(7)
    check("driver: a driven input isn't read through the F-curves, so a render of frame 25 compares it as the file "
          "holds it at frame 7 (Question 0?), not as the driver sets it there (Question 2?)", not fcurves
          and reader.reason.endswith("is driven") and there == 2 and compared == ["Question 0?"],
          (reader.reason, there, compared))


def phase_nla():
    """An input animated by a strip on an NLA track: only setting the frame tells its value there."""
    scene = reset(1, 40)
    tree, _ask, _gi = question_tree("Strip", 3)
    obj = user("Strip", tree, Level=0)
    constant(keyframes(obj, "Level", [(1, 0), (11, 1), (21, 2)]))
    data = obj.animation_data
    data.nla_tracks.new().strips.new("Levels", 11, data.action)  # the action plays 10 frames late
    data.action = None
    scene.frame_set(25)
    there = level(obj)  # the action's frame 15
    scene.frame_set(1)
    settle()
    reader = engine.FrameWords(scene)
    fcurves = reader.read_curves()
    compared = rendered(scene, 25, obj)
    scene.frame_set(1)
    check("NLA: an input animated by a strip on an NLA track isn't read through the F-curves, so a render of frame 25 "
          "compares it as the file holds it at frame 1 (Question 0?), not as the strip sets it there (Question 1?)",
          not fcurves and "NLA" in reader.reason and there == 1 and compared == ["Question 0?"],
          (reader.reason, there, compared))


# ------------------------------------------------------------------------------ threads

def phase_no_override():
    """Blender 4.5 and 5.0 (Python 3.11) crash when bpy.ops runs under temp_override while another Python thread runs,
    as OpenAct's decision worker and /health probes do (§5.5)."""
    code = pathlib.Path(ROOT, "openact")
    found = [str(path.relative_to(code)) for path in code.rglob("*.py")
             if "temp_override(" in path.read_text(encoding="utf-8")]
    check("threads: OpenAct's code calls no temp_override, under which bpy.ops crashes Blender 4.5 and 5.0 while "
          "OpenAct's threads run Python", not found, found)


def main():
    print("BLENDER", bpy.app.version_string, "fake model server on port", PORT, flush=True)
    try:
        check("the add-on starts the fake server, which gets ready", server.start() and drive(engine, server.is_ready,
                                                                                            60), server.status())
        phase_playback()
        phase_trigger()
        phase_rules_in_use()
        phase_simulation()
        phase_stale()
        phase_budget()
        phase_render()
        phase_storing()  # it opens a saved file: the phases after it start from a new one
        phase_channelbag()
        phase_ranges()
        phase_driver()
        phase_nla()
        phase_no_override()
    except Exception:
        check("no unexpected error", False, traceback.format_exc())
    finally:
        server.stop()
    log = os.path.join(TMP, "state", "server.log")
    if FAILURES and os.path.isfile(log):
        with open(log, encoding="utf-8", errors="replace") as handle:
            print("--- the fake server's log:\n" + handle.read()[-3000:].rstrip(), flush=True)
    if not FAILURES:
        shutil.rmtree(TMP, ignore_errors=True)
    finish()


main()
