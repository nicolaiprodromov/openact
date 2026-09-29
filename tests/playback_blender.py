"""Playback and the words at other frames (§5.5, §7.8) in a headless Blender, against the fake model server that the
add-on starts itself (OPENACT_BACKEND=fake), whose --record log counts every request it gets:
- playback: a Rules node whose Outcomes CSV comes from a keyframed Menu Switch cycling through three Game of Life rule
  sets, two of them answered while the scene was stopped at their frames. Playing frames 1-30 runs no pass, so it
  sends 0 requests and writes no table: each answered rule set shows its own table, and the one never answered shows
  the object's last rules (k_default, §8.2), never a fallback. Once playback stops, the pass asks for it;
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

import json
import os
import pathlib
import shutil
import socket
import sys
import tempfile
import traceback

TMP = tempfile.mkdtemp(prefix="openact_playback_")
RECORD = os.path.join(TMP, "record.jsonl")
os.environ["OPENACT_BACKEND"] = "fake"
os.environ["OPENACT_SERVER_PYTHON"] = sys.executable  # the fake backend needs no runtime: Blender's Python runs it
os.environ["OPENACT_STATE_DIR"] = os.path.join(TMP, "state")  # the server log stays out of any shared runtime

import bpy  # noqa: E402
from bpy_extras import anim_utils  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import FAILURES, ROOT, check, drive, finish, store, trigger  # noqa: E402
from openact import engine, nodes, ops, pipeline, server, trust, ui, watch  # noqa: E402
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


# ------------------------------------------------------------------------------ helpers

def records():
    """The fake server's --record log: one line per request it got."""
    try:
        with open(RECORD, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    except OSError:
        return []


def settle(timeout=90):
    """Tick until nothing is queued or in flight, no pass is due and no node waits for the model."""
    return drive(engine, lambda: not pipeline.PENDING and not engine._dirty and not any(
        status.state in ("starting", "waiting", "filling") for status in engine.STATUS.values()), timeout)


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
    """v1's playback (§5.5): passes stay paused, so nothing is asked or written while the scene plays. The node shows
    the answers its graph holds; on a node with one context, words it has no answer for show that context's last
    answers (k_default, §8.2) until the pass after playback asks for them."""
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
    tables = {rule_set(entry): entry.rows.v["outcome"] for entry in S.load(rules).entries}
    status = engine.STATUS.get(rules.uid)
    check("playback: while the scene is stopped, the pass answers the rule set of the frame shown, Life and then "
          "HighLife, each with a table of its own (the fake answers depend on the state)", first
          and sorted(tables) == ["HighLife", "Life"] and tables["Life"] != tables["HighLife"]
          and read(obj) == tables["Life"], (sorted(tables), status and status.text))
    count, before = len(records()), rules.entries
    playing, engine._animation_playing = engine._animation_playing, lambda: True
    shown = {}
    try:
        for frame in range(1, 31):
            scene.frame_set(frame)
            engine.tick()
            shown[frame] = read(obj)
    finally:
        engine._animation_playing = playing
    bad = [(frame, got) for frame, got in shown.items() if got != tables["HighLife" if 11 <= frame <= 20 else "Life"]]
    check("playback: playing frames 1-30 runs no pass, so it sends 0 requests and writes no table: Life and HighLife "
          "show their tables, and Day & Night, never answered, the object's last rules (Life), never a fallback (-1)",
          not bad and len(records()) == count and rules.entries == before, (bad[:3], len(records()) - count))
    start = len(records())
    done = settle()
    sent = {record["request"]["state"].split("\n")[0] for record in records()[start:] if record["request"]}
    night = next((entry for entry in S.load(rules).entries if rule_set(entry) == "Day & Night"), None)
    status = engine.STATUS.get(rules.uid)
    check("playback: once it stops, the pass asks for the rule set of the frame shown (30: Day & Night), whose table "
          "then shows", done and night is not None and read(obj) == night.rows.v["outcome"]
          and sent == {f"Rules from {os.path.join(TMP, RULE_SETS['Day & Night'])}"}, (sent, status and status.text))


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
