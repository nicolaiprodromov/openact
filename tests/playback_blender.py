"""Prepare for Playback and entry pinning (§5.5, §8.1, §14.7) in a headless Blender, against the fake model server that
the add-on starts itself (OPENACT_BACKEND=fake), whose --record log counts every request it gets:
- a Rules node whose Outcomes CSV comes from a keyframed Menu Switch cycling through three Game of Life rule sets:
  Prepare for Playback asks the two tables not known yet and pins all three with their frames; then playback sends 0
  requests, writes no table, and every frame shows its rule set's table, never a fallback;
- the F-curve path over a 250-frame range: under 1 s, with no scene evaluation (no frame change, no depsgraph update);
- a keyframed Int modifier input, on 4.5 and 5.1: its F-curve is found through the action slot's channelbag
  (Action.fcurves is gone in 5.x), evaluate() gives the keyed values, and the words read equal the scene's;
- a driven input takes the fallback. Its frame loop, the modal operator's core, is driven here through
  Playback.step_frames(), as a headless Blender has no window to run a modal operator in: cancelled, it sets the
  current frame again, pins nothing and asks nothing, and no pass runs meanwhile. Run from the operator (which steps
  every frame without a modal loop when a script runs it), it pins, and the current frame is set again;
- an NLA-animated input (a strip on an NLA track) takes the fallback, and pins the word tuples the NLA gives;
- 13 distinct word sets are refused with the limit message, and nothing is asked;
- pins survive save and reload (pinned: true, frames); the sidebar lists them with Unpin and Unpin All; openact.unpin
  removes one, then the rest;
- pin order: with 12 pins, objects added until live plus pinned exceeds 16 leave out the pins whose words start latest
  in the frame range (ties by their words), the same ones after a rebuild; a new run is then refused, asking nothing;
- pins above 2 MB are kept, with the size note, and Prepare for Playback reported the projected size first.

Run:  blender -b --factory-startup --python tests/playback_blender.py
"""

import json
import os
import re
import shutil
import socket
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace

TMP = tempfile.mkdtemp(prefix="openact_playback_")
RECORD = os.path.join(TMP, "record.jsonl")
os.environ["OPENACT_BACKEND"] = "fake"
os.environ["OPENACT_SERVER_PYTHON"] = sys.executable  # the fake backend needs no runtime: Blender's Python runs it
os.environ["OPENACT_STATE_DIR"] = os.path.join(TMP, "state")  # the server log stays out of any shared runtime

import bpy  # noqa: E402
from bpy_extras import anim_utils  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import FAILURES, ROOT, check, drive, finish, sock, store  # noqa: E402
from openact import engine, inner, nodes, ops, pipeline, server, trust, ui, watch  # noqa: E402
from openact import prefs as P  # noqa: E402
from openact import store as S  # noqa: E402
from openact.specs import SPECS  # noqa: E402

ASK, RULES = SPECS["GeometryNodeOpenActAsk"], SPECS["GeometryNodeOpenActRules"]
RULE_SETS = {"Life": "life_outcomes.csv", "HighLife": "highlife_outcomes.csv", "Day & Night": "daynight_outcomes.csv"}
GOL_QUESTION = "What happens to this cell in the next generation?"
ICONS = set(bpy.types.UILayout.bl_rna.functions["label"].parameters["icon"].enum_items.keys())
for name in ("life_factors.csv", *RULE_SETS.values()):  # beside the .blend saved below: an untrusted file reads there
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
    """Tick until nothing is queued or in flight, no pass is due, no node waits for the model and no Prepare for
    Playback run waits."""
    return drive(engine, lambda: not pipeline.PENDING and not engine._dirty and engine.PLAYBACK is None and not any(
        status.state in ("starting", "waiting", "filling") for status in engine.STATUS.values()), timeout)


def reset(first=1, last=250):
    """An empty file between phases, as every run covers all of it."""
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
    more (name, socket type) inputs. Returns (tree, Ask node, Group Input)."""
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
    return tree, ask, gi


def pinned(node):
    """The node's pinned entries, by their first frame."""
    return sorted((entry for entry in S.load(node).entries if entry.pinned), key=lambda entry: entry.frames)


def in_graph(node):
    """The joined words of each entry the node's match chain compares with: the materialised entries (§8.2)."""
    return {sock(n.inputs, "B_STR").default_value for n in node.node_tree.nodes
            if n.bl_idname == "FunctionNodeCompare" and n.data_type == 'STRING'}


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


def spans(frames):
    """Sorted frames as [start, end] runs."""
    runs = []
    for frame in frames:
        if runs and frame == runs[-1][1] + 1:
            runs[-1][1] = frame
        else:
            runs.append([frame, frame])
    return runs


def prepare():
    """openact.prepare_playback as a script runs it: (its result, the error it reported or "")."""
    try:
        return bpy.ops.openact.prepare_playback(), ""
    except RuntimeError as error:  # the operator reported an error
        return {'CANCELLED'}, str(error)


class FakeLayout:
    """Stands in for the UILayout a headless Blender has none of, as in status_blender: records labels and buttons,
    and refuses unknown icons, operators and operator properties."""

    __slots__ = ("log", "alert", "active", "enabled", "use_property_split", "operator_context")

    def __init__(self, log):
        self.log = log

    def _child(self, *_args, **_kwargs):
        return FakeLayout(self.log)

    column = row = box = split = _child

    def label(self, text="", icon='NONE'):
        assert icon in ICONS, icon
        self.log.append(text)

    def prop(self, data, name, **_kwargs):
        assert hasattr(data, name), name

    def progress(self, factor=0.0, type='BAR', text=""):  # noqa: A002 - UILayout's own keyword
        self.log.append(text)

    def operator(self, idname, text=None, icon='NONE', depress=False):
        assert icon in ICONS, icon
        module, name = idname.split(".")
        self.log.append(f"[{idname}]")
        return FakeProperties(getattr(getattr(bpy.ops, module), name).get_rna_type(), self.log, idname)


class FakeProperties:
    def __init__(self, rna, log, idname):
        object.__setattr__(self, "names", set(rna.properties.keys()))
        object.__setattr__(self, "log", log)
        object.__setattr__(self, "idname", idname)

    def __setattr__(self, name, value):
        assert name in self.names, (self.idname, name)
        self.log.append((self.idname, name, value))


def drawn(panel, node):
    """What a sidebar panel draws with the node active in its editor."""
    node.id_data.nodes.active = node
    ctx = SimpleNamespace(window_manager=bpy.context.window_manager, preferences=bpy.context.preferences,
                          space_data=SimpleNamespace(tree_type='GeometryNodeTree', edit_tree=node.id_data))
    log = []
    panel.draw(SimpleNamespace(layout=FakeLayout(log)), ctx)
    return log


# ------------------------------------------------------------------------------ a keyframed Menu Switch

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


def phase_menu_switch():
    scene = reset(1, 30)
    tree, rules, obj, gi, go = life_scene()
    first = settle()
    wire_life(tree, rules, gi, go)
    engine.schedule(immediate=True)
    settle()
    shown = read(obj)
    live = next((e for e in S.load(rules).entries if rule_set(e) == "Life"), None)
    check("menu switch: the pass answers the rule set of the current frame, which shows",
          first and live is not None and shown == live.rows.v["outcome"], (shown, engine.STATUS.get(rules.uid)))
    for frame, name in ((1, "Life"), (11, "HighLife"), (21, "Day & Night")):
        keyframes(obj, "Rule set", [(frame, menu_value(obj, "Rule set", name))])
    start = len(records())
    result, error = prepare()
    waiting = engine.PLAYBACK is not None and not pinned(rules)
    done = settle()
    pins = pinned(rules)
    sent = {record["request"]["state"].split("\n")[0] for record in records()[start:] if record["request"]}
    check("menu switch: Prepare for Playback asks the two rule sets not known yet, and pins the three once answered, "
          "each with its frames", result == {'FINISHED'} and not error and waiting and done
          and [(rule_set(e), e.frames) for e in pins] == [("Life", [[1, 10]]), ("HighLife", [[11, 20]]),
                                                          ("Day & Night", [[21, 30]])]
          and sent == {f"Rules from {os.path.join(TMP, RULE_SETS[name])}" for name in ("HighLife", "Day & Night")}
          and engine.PLAYBACK_NOTE == "Pinned 3 word sets for playback", (result, error, [e.frames for e in pins], sent))
    check("menu switch: each rule set has a table of its own (the fake answers depend on the state)",
          len({tuple(e.rows.v["outcome"]) for e in pins}) == 3, [e.rows.v["outcome"] for e in pins])
    status = engine.STATUS[rules.uid]
    check("menu switch: the pass lays out the three pins, and the status lists them in pin order, none inactive",
          len(in_graph(rules)) == 3 and [key for key, _frames, _gn in status.pins] == [e.words_key for e in pins]
          and status.state == "complete", (status.state, status.text, len(in_graph(rules))))
    count, before = len(records()), rules.entries
    playing, engine._animation_playing = engine._animation_playing, lambda: True
    bad = []
    try:
        for frame in range(1, 31):
            scene.frame_set(frame)
            engine.tick()
            want = next(e for e in pins if e.frames[0][0] <= frame <= e.frames[0][1]).rows.v["outcome"]
            got = read(obj)
            if got != want or -1 in got:
                bad.append((frame, got, want))
    finally:
        engine._animation_playing = playing
    check("menu switch: playing frames 1-30 sends 0 requests, writes no table, and every frame shows its rule set's "
          "pinned table, never a fallback (-1)", not bad and len(records()) == count and rules.entries == before,
          (bad[:3], len(records()) - count))
    scene.frame_set(1)
    settle()
    return tree, rules, obj


def phase_timing(rules, obj):
    """The F-curve path over 250 frames (§5.5): well under 1 s, and no scene evaluation."""
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, 250
    names = list(RULE_SETS)
    keyframes(obj, "Rule set", [(frame, menu_value(obj, "Rule set", names[(frame // 10) % 3]))
                                for frame in range(31, 251, 10)])
    counts = {"frames": 0, "updates": 0}

    def on_frame(*_args):
        counts["frames"] += 1

    def on_update(*_args):
        counts["updates"] += 1

    handlers = ((bpy.app.handlers.frame_change_pre, on_frame), (bpy.app.handlers.depsgraph_update_post, on_update))
    for handler, fn in handlers:
        handler.append(fn)
    current, start = scene.frame_current, len(records())
    try:
        started = time.perf_counter()
        run = engine.Playback(scene)
        fcurves = run.read_curves()
        text = run.finish()
        seconds = time.perf_counter() - started
    finally:
        for handler, fn in handlers:
            handler.remove(fn)
    print(f"TIMING Prepare for Playback: F-curve path over 250 frames {seconds:.3f}s (budget 1 s)", flush=True)
    check("timing: over a 250-frame range the F-curve path reads, checks and pins in under 1 s, with no scene "
          "evaluation (no frame change, no depsgraph update, the frame unchanged)",
          fcurves and seconds < 1.0 and counts == {"frames": 0, "updates": 0} and scene.frame_current == current,
          (fcurves, run.fallback, f"{seconds:.3f}s", counts))
    pins = pinned(rules)
    life = [[f, f + 9] for f in range(1, 251, 30)]
    check("timing: its answers are all known, so it pins at once, asking nothing, each rule set with its 8 or 9 "
          "frame runs", engine.PLAYBACK is None and len(records()) == start and len(pins) == 3
          and pins[0].frames == life and text.startswith("3 word sets over frames 1–250: pinning 3 entries adds about"),
          (text, [e.frames for e in pins]))
    settle()


# ------------------------------------------------------------------------------ save, reload, unpin

def phase_reload():
    path = os.path.join(TMP, "pins.blend")
    bpy.ops.wm.save_as_mainfile(filepath=path)
    bpy.ops.wm.open_mainfile(filepath=path)
    rules = next(n for n in bpy.data.node_groups["Life"].nodes if n.bl_idname == RULES.idname)
    saved = json.loads(rules.entries)["entries"]
    kept = sorted((e["frames"] for e in saved if e.get("pinned") is True), key=lambda frames: frames[0])
    check("reload: the pins are saved with pinned: true and their frames, and come back", len(kept) == 3
          and kept[0][0] == [1, 10] and [e.frames for e in pinned(rules)] == kept, [e.get("pinned") for e in saved])
    settle()
    pins = pinned(rules)
    log = []
    ui.draw_pins(FakeLayout(log), rules)  # what the Active Node panel draws below the node's own buttons
    rows = ui.pin_rows(engine.STATUS[rules.uid].pins)
    check("reload: the sidebar lists the pins in pin order, by their frames and the words that differ, each with "
          "Unpin, and Unpin All", "Pinned for playback: 3" in log and ("openact.unpin", "uid", rules.uid) in log
          and all(("openact.unpin", "words_key", e.words_key) in log for e in pins) and len(rows) == 3
          and rows[0][1].startswith("Frames 1–10, 31–40") and "life_outcomes.csv" in rows[0][1]
          and "highlife_outcomes.csv" in rows[1][1], (rows, log))
    log = drawn(ui.NODE_PT_openact, rules)
    check("reload: the OpenAct panel has Prepare for Playback", "[openact.prepare_playback]" in log, log)
    one = next(e for e in pins if rule_set(e) == "HighLife")
    result = bpy.ops.openact.unpin(uid=rules.uid, words_key=one.words_key)
    left = pinned(rules)
    due = rules.uid in engine.DUE
    settle()
    check("unpin: one entry: it stays stored, unpinned and without frames, saved with no eviction; the next pass saves "
          "the node again with its live set", result == {'FINISHED'} and [rule_set(e) for e in left] == [
              "Life", "Day & Night"] and due and rules.uid not in engine.DUE
          and any(e.words_key == one.words_key and not e.frames for e in S.load(rules).entries), [e.frames for e in left])
    result = bpy.ops.openact.unpin(uid=rules.uid)
    settle()
    again = bpy.ops.openact.unpin(uid=rules.uid)
    check("unpin: Unpin All unpins the rest; then there is nothing to unpin", result == {'FINISHED'}
          and again == {'CANCELLED'} and not pinned(rules) and not engine.STATUS[rules.uid].pins
          and len(S.load(rules).entries) == 3 and '"pinned":true' not in rules.entries, (result, again))


# ------------------------------------------------------------------------------ the channelbag

def phase_channelbag():
    scene = reset(1, 11)
    tree, _ask, _gi = question_tree("Levels", 6)
    obj = user("Levels", tree, Level=0)
    curve = keyframes(obj, "Level", [(1, 0), (11, 5)])
    found, why = engine._animation(obj, path_of(obj, "Level"))
    action = obj.animation_data.action
    check("channelbag: a keyframed Int modifier input's F-curve is found through the action slot's channelbag "
          "(Action.fcurves exists only before 5.0), and evaluate() gives the keyed values",
          found is not None and found == curve and why == "" and hasattr(action, "fcurves") == (bpy.app.version < (5, 0))
          and (found.evaluate(1), found.evaluate(11)) == (0.0, 5.0), (found, why, hasattr(action, "fcurves")))
    run = engine.Playback(scene)
    fcurves = run.read_curves()
    words = {frame: gn[0] for seen in run.seen.values() for gn, frames in seen.items() for frame in frames}
    truth = {}
    for frame in range(1, 12):  # what Blender sets at each frame
        scene.frame_set(frame)
        truth[frame] = f"Question {obj.modifiers['OpenAct'][ident(tree, 'Level')]}?"
    scene.frame_set(1)
    check("channelbag: the F-curve path reads the words through evaluate(), with no fallback, and at every frame they "
          "are the words the scene has there", fcurves and words == truth
          and all(words[f] == f"Question {int(curve.evaluate(f))}?" for f in truth), (run.fallback, words, truth))


# ------------------------------------------------------------------------------ the fallback

def phase_driver():
    scene = reset(1, 29)
    tree, ask, _gi = question_tree("Driven", 3)
    obj = user("Driven", tree, Level=0)
    driver = obj.driver_add(path_of(obj, "Level")).driver
    driver.type, driver.expression = 'SCRIPTED', "frame // 10"
    scene.frame_set(7)
    settle()
    start = len(records())
    run = engine.Playback(scene)
    fell = not run.read_curves()
    passes, run_pass = [], engine.run_pass
    engine.run_pass = lambda everything=True: passes.append(everything)
    try:
        steps = run.step_frames()
        stepped = [next(steps) for _ in range(5)]
        inside, holding = scene.frame_current, engine._stepping
        engine.schedule(immediate=True)
        engine.tick()
        steps.close()  # Esc
    finally:
        engine.run_pass = run_pass
    back = scene.frame_current
    settle()
    check("driver: a driven input takes the fallback, which sets frames 1-5 in turn with no pass meanwhile; "
          "cancelled, it sets frame 7 again, pins nothing and asks nothing", fell and "driven" in run.fallback
          and stepped == [1, 2, 3, 4, 5] and inside == 5 and holding and not passes and back == 7
          and not engine._stepping and engine.PLAYBACK is None and not pinned(ask) and len(records()) == start,
          (run.fallback, stepped, inside, passes, back))
    result, error = prepare()
    back = scene.frame_current
    done = settle()
    pins = [(e.gn[0], e.frames) for e in pinned(ask)]
    check("driver: run from the operator, the fallback steps every frame, sets frame 7 again and pins the driven "
          "words", result == {'FINISHED'} and not error and back == 7 and done and pins == [
              ("Question 0?", [[1, 9]]), ("Question 1?", [[10, 19]]), ("Question 2?", [[20, 29]])], (error, back, pins))


def phase_nla():
    scene = reset(1, 40)
    tree, ask, _gi = question_tree("Strip", 3)
    obj = user("Strip", tree, Level=0)
    curve = keyframes(obj, "Level", [(1, 0), (11, 1), (21, 2)])
    for point in curve.keyframe_points:
        point.interpolation = 'CONSTANT'
    data = obj.animation_data
    data.nla_tracks.new().strips.new("Levels", 11, data.action)  # the action plays 10 frames late
    data.action = None
    truth = {}
    for frame in range(1, 41):  # what the NLA sets at each frame
        scene.frame_set(frame)
        truth.setdefault(f"Question {obj.modifiers['OpenAct'][ident(tree, 'Level')]}?", []).append(frame)
    scene.frame_set(1)
    settle()
    run = engine.Playback(scene)
    fell = not run.read_curves()
    for _frame in run.step_frames():
        pass
    back = scene.frame_current
    run.finish()
    done = settle()
    pins = [(e.gn[0], e.frames) for e in pinned(ask)]
    check("NLA: an input animated by a strip on an NLA track takes the fallback, and pins the word tuples the NLA "
          "gives (the strip starts at frame 11), each with its frames", fell and "NLA" in run.fallback and back == 1
          and done and pins == [(words, spans(frames)) for words, frames in truth.items()]
          and pins == [("Question 0?", [[1, 20]]), ("Question 1?", [[21, 30]]), ("Question 2?", [[31, 40]])],
          (run.fallback, pins, truth))


# ------------------------------------------------------------------------------ limits

def phase_limit():
    scene = reset(1, 13)
    tree, ask, _gi = question_tree("Thirteen", 13)
    obj = user("Thirteen", tree, Level=0)
    keyframes(obj, "Level", [(frame, frame - 1) for frame in range(1, 14)])
    settle()
    start, entries = len(records()), ask.entries
    run = engine.Playback(scene)
    try:
        run.read_curves()
        run.finish()
        refused = ""
    except ValueError as error:
        refused = str(error)
    result, error = prepare()
    settle()
    message = engine.PIN_LIMIT.format(ask.name, 13, 1, 13, engine.PIN_MAX)
    check("limit: 13 distinct word sets are refused with the limit message, and nothing is asked or pinned",
          refused == message == "'Ask': 13 distinct word sets over frames 1–13; the limit is 12" and message in error
          and result == {'CANCELLED'} and len(records()) == start and not pipeline.PENDING
          and engine.PLAYBACK is None and ask.entries == entries, (refused, error, len(records()) - start))


def phase_pin_order():
    """Two animated objects, 6 levels each: 12 pins whose first frames pair up, so ties are broken by the words."""
    reset(1, 6)
    tree, ask, gi = question_tree("Crowd", 6, ("Place", 'NodeSocketString'))
    tree.links.new(gi.outputs["Place"], ask.inputs["Context"])
    for place in ("attic", "barn"):
        keyframes(user(place, tree, Level=0, Place=place), "Level", [(frame, frame - 1) for frame in range(1, 7)])
    settle()
    result, error = prepare()
    settle()
    pins = pinned(ask)
    check("pin order: 12 word sets are pinned (live plus pinned is 12)", result == {'FINISHED'} and len(pins) == 12
          and not any("inactive" in p.message for p in engine.STATUS[ask.uid].problems), (error, len(pins)))

    def left_out():
        inside = in_graph(ask)
        return [(e.gn[0], e.gn[1]) for e in pins if inner.SEP.join(e.gn) not in inside]

    for k in range(5):
        user(f"crowd {k}", tree, Level=0, Place=f"crowd {k}")
    engine.schedule(immediate=True)  # as the depsgraph update of a new object does in the UI
    settle()
    one, status = left_out(), engine.STATUS[ask.uid]
    check("pin order: 7 live word sets and 12 pins leave out the pin that starts latest, of the two starting at frame "
          "6 the one whose words sort last; the status says so", one == [("Question 5?", "barn")]
          and status.state == "warning" and "1 pinned entry is inactive: this tree is now used by more objects" in
          status.text and len(in_graph(ask)) == 16, (one, status.text))
    user("crowd 5", tree, Level=0, Place="crowd 5")
    engine.schedule(immediate=True)
    settle()
    two = left_out()
    ask.node_tree.pop(inner.BUILT, None)  # a full rebuild
    engine.schedule(immediate=True)
    settle()
    check("pin order: one more object leaves out both pins that start at frame 6, the same ones after a rebuild; all "
          "12 stay stored", two == left_out() == [("Question 5?", "attic"), ("Question 5?", "barn")]
          and len(pinned(ask)) == 12 and "2 pinned entries are inactive" in engine.STATUS[ask.uid].text,
          (two, left_out(), engine.STATUS[ask.uid].text))
    start = len(records())
    result, error = prepare()
    settle()
    check("pin order: a new run is refused now that live plus pinned would pass 16, asking nothing",
          result == {'CANCELLED'} and "live plus pinned is limited to 16" in error and len(records()) == start
          and len(pinned(ask)) == 12, error)


# ------------------------------------------------------------------------------ sizes

SETS = ("grows; stays; shrinks", "rises; holds; falls", "blooms; rests; wilts", "wakes; waits; sleeps",
        "heats; keeps; cools", "opens; idles; closes", "speeds up; cruises; slows down")


def answer(cells):
    """Made-up answers, as pipeline.parse() gives them, with confidences of a real length."""
    for cell in cells:
        count = len(cell.question["criteria"])
        pipeline.remember(cell.key, {"probabilities": [0.61803398875] + [0.38196601125 / (count - 1)] * (count - 1),
                                     "model": "fake"})


def phase_size():
    """Seven 4096-cell Rules tables (§14.7), answered straight into MEMORY: a real fill would take minutes."""
    scene = reset(1, 7)
    path = os.path.join(TMP, "factors_4096.csv")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("factor,kind,level,value,when\n")
        for factor in ("warmth", "damp", "height"):
            handle.writelines(f"{factor},category,{factor} {v},{v},The {factor} is at step {v} of 16.\n"
                              for v in range(16))
    tree, gi, go = new_tree("Plants", ("Set", 'NodeSocketMenu'))
    menu = menu_switch(tree, {f"Set {i}": words for i, words in enumerate(SETS)})
    tree.links.new(gi.outputs["Set"], menu.inputs["Menu"])
    rules = tree.nodes.new(RULES.idname)
    rules.inputs["Question"].default_value = "What happens to the plant here?"
    rules.inputs["Factors"].default_value = path
    tree.links.new(menu.outputs[0], rules.inputs["Outcomes"])
    tree.links.new(gi.outputs["Geometry"], go.inputs["Geometry"])
    obj = user("Plants", tree)
    keyframes(obj, "Set", [(frame, menu_value(obj, "Set", f"Set {frame - 1}")) for frame in range(1, 8)])
    asked, submit = {}, engine._submit
    engine._submit = lambda job, missing: asked.update((cell.key, cell) for cell in missing) or True
    try:
        engine.run_pass()
        answer(asked.values())
        engine.run_pass()
        live = len(S.load(rules).entries) == 1
        run = engine.Playback(scene)
        run.read_curves()
        text = run.finish()
        first = not pinned(rules) and engine.PLAYBACK is not None
        answer(asked.values())
        done = drive(engine, lambda: engine.PLAYBACK is None and not engine._dirty, 120)
    finally:
        engine._submit = submit
    size, pins = len(rules.entries.encode("utf-8")), pinned(rules)
    projected = float(re.search(r"adds about ([\d.]+) MB", text).group(1)) if "adds about" in text else 0.0
    notes = [p.message for p in engine.STATUS[rules.uid].problems if p.severity == "info"]
    check("size: Prepare for Playback reports the projected size first, before anything is pinned (about the size "
          "the pins then take)", live and first and text.startswith(
              "7 word sets over frames 1–7: pinning 7 entries adds about") and projected > 2.0
          and abs(projected * S.MB - size) < 0.25 * size, (text, size))
    check("size: pins above 2 MB are all kept, with the size note in the status",
          done and len(pins) == 7 and size > S.SOFT_CAP and notes == [
              f"entries use {S._mb(size)} MB: live and pinned contexts are kept; pin fewer word sets to shrink the file"],
          (done, len(pins), size, notes))


def main():
    print("BLENDER", bpy.app.version_string, "fake model server on port", PORT, flush=True)
    try:
        check("the add-on starts the fake server, which gets ready", server.start() and drive(engine, server.is_ready,
                                                                                            60), server.status())
        _tree, rules, obj = phase_menu_switch()
        phase_timing(rules, obj)
        phase_reload()
        phase_channelbag()
        phase_driver()
        phase_nla()
        phase_limit()
        phase_pin_order()
        phase_size()
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
