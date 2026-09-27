"""Prepare for Playback and entry pinning (§5.5, §8.1, §14.7) in a headless Blender, against the fake model server that
the add-on starts itself (OPENACT_BACKEND=fake), whose --record log counts every request it gets:
- a Rules node whose Outcomes CSV comes from a keyframed Menu Switch cycling through three Game of Life rule sets:
  Prepare for Playback asks the two tables not known yet and pins all three with their frames; then playback sends 0
  requests, writes no table, and every frame shows its rule set's table, never a fallback;
- the F-curve path over a 250-frame range: under 1 s, with no scene evaluation (no frame change, no depsgraph update);
  with 64 contexts of which 4 are animated, each static context is read once;
- a keyframed Int modifier input, on 4.5 and 5.1: its F-curve is found through the action slot's channelbag
  (Action.fcurves is gone in 5.x), evaluate() gives the keyed values, and the words read equal the scene's; a keyed
  value beyond the property's hard range is clamped as the scene clamps it; under Time Stretching the fallback runs;
- a driven input takes the fallback. Its frame loop, cancelled, sets the current frame again, pins nothing and asks
  nothing, and no pass runs meanwhile, nor can another run start. Run from the operator (which steps every frame
  without a modal loop when a script runs it), it pins, sets the current frame again and flags the modifier's bakes;
  Ask Again keeps its pins, and a run that changes no pin clears the flag;
- the modal operator's own functions, called on a stand-in instance with a stand-in window manager and workspace, as a
  headless Blender runs no modal loop: Esc mid-way, a run to the end, an internal error mid-way and cancel() each
  leave no timer, progress or status text behind, and set the current frame again;
- an NLA-animated input (a strip on an NLA track) takes the fallback, and pins the word tuples the NLA gives;
- 13 distinct word sets are refused with the limit message, which the sidebar keeps, and nothing is asked;
- pins survive save and reload (pinned: true, frames); the sidebar lists them with Unpin and Unpin All; openact.unpin
  removes one, then the rest, and without a node unpins nothing;
- undo, for real (a first ed.undo_push() gives a headless Blender its undo stack): pins that a pass writes after their
  operator's step have a step of their own, which only its undo removes, and the sidebar then claims no pin; undoing
  the operator's step while its run waits leaves the run dormant, and a redo wakes it; Cancel and Unpin stop a waiting
  run; a question that failed is asked again by the next run;
- pin order: with 12 pins, objects added until live plus pinned exceeds 16 leave out the pins whose words start latest
  in the frame range (ties by their words), the same ones after a rebuild; a new run is then refused, asking nothing;
- pins above 2 MB are kept, with the size note, and Prepare for Playback reported the size of the pins first.

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
import types
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
    return drive(engine, lambda: not pipeline.PENDING and not engine._dirty and not engine.pinning() and not any(
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


def prepare(undo=False):
    """openact.prepare_playback as a script runs it, with undo as a button runs it: (its result, the error it reported
    or "")."""
    try:
        return bpy.ops.openact.prepare_playback('EXEC_DEFAULT', undo), ""
    except RuntimeError as error:  # the operator reported an error
        return {'CANCELLED'}, str(error)


def constant(curve):
    """Each key's value holds until the next key."""
    for point in curve.keyframe_points:
        point.interpolation = 'CONSTANT'
    return curve


TIMER, ESC = SimpleNamespace(type='TIMER', value='NOTHING'), SimpleNamespace(type='ESC', value='PRESS')


def modal_operator(scene):
    """The Prepare for Playback operator's own functions on a stand-in instance, with a stand-in window manager and
    workspace that record their calls, as a headless Blender runs no modal loop: (instance, context, calls)."""
    calls = []

    def record(name):
        return lambda *args, **_kwargs: calls.append((name, *args)) or name

    op = SimpleNamespace(reports=[])
    op.report = lambda kind, text: op.reports.append((next(iter(kind)), text))
    for name in ("invoke", "modal", "cancel", "end", "finish"):
        setattr(op, name, types.MethodType(getattr(ops.OPENACT_OT_prepare_playback, name), op))
    wm = SimpleNamespace(**{name: record(name) for name in ("progress_begin", "progress_update", "progress_end",
                                                            "event_timer_add", "event_timer_remove",
                                                            "modal_handler_add")})
    context = SimpleNamespace(scene=scene, window=None, window_manager=wm,
                              workspace=SimpleNamespace(status_text_set=record("status_text_set")))
    return op, context, calls


def cleaned(calls):
    """True once the stand-in window manager and workspace saw the timer, the progress and the status text go."""
    names = [call[0] for call in calls]
    return "event_timer_remove" in names and "progress_end" in names and calls[-1] == ("status_text_set", None)


def said(log, text):
    """True when a drawn panel shows the text, which wrap() may have split over several labels."""
    return text in " ".join(line for line in log if isinstance(line, str))


def undo(redo=False):
    """ed.undo() or ed.redo(), then the passes it schedules: False when there was no step to go to."""
    try:
        done = (bpy.ops.ed.redo if redo else bpy.ops.ed.undo)() == {'FINISHED'}
    except RuntimeError:  # its poll(): no step that way
        done = False
    settle()
    return done


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
    waiting = engine.pinning() == 3 and not pinned(rules)
    log = drawn(ui.NODE_PT_openact, rules)
    done = settle()
    pins = pinned(rules)
    sent = {record["request"]["state"].split("\n")[0] for record in records()[start:] if record["request"]}
    check("menu switch: Prepare for Playback asks the two rule sets not known yet, and pins the three once answered, "
          "each with its frames; meanwhile the sidebar says it waits, with Cancel", result == {'FINISHED'}
          and not error and waiting and done and said(log, "Pinning 3 word sets once the model has answered them")
          and "[openact.cancel_playback]" in log
          and [(rule_set(e), e.frames) for e in pins] == [("Life", [[1, 10]]), ("HighLife", [[11, 20]]),
                                                          ("Day & Night", [[21, 30]])]
          and sent == {f"Rules from {os.path.join(TMP, RULE_SETS[name])}" for name in ("HighLife", "Day & Night")},
          (result, error, [e.frames for e in pins], sent, log))
    check("menu switch: each rule set has a table of its own (the fake answers depend on the state)",
          len({tuple(e.rows.v["outcome"]) for e in pins}) == 3, [e.rows.v["outcome"] for e in pins])
    status, log = engine.STATUS[rules.uid], drawn(ui.NODE_PT_openact, rules)
    check("menu switch: the pass lays out the three pins, and the status lists them in pin order, none inactive; the "
          "sidebar counts them from the entries, and a tree without bakes has nothing to re-bake",
          len(in_graph(rules)) == 3 and [e.words_key for e in status.pins] == [e.words_key for e in pins]
          and status.state == "complete" and said(log, "3 word sets pinned for playback") and not engine.REBAKES
          and not engine.PLAYBACK_NOTE, (status.state, status.text, len(in_graph(rules)), log))
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
          and pins[0].frames == life and text.startswith("3 word sets over frames 1–250: the pinned entries take about"),
          (text, [e.frames for e in pins]))
    settle()


def phase_static():
    """The F-curve path reads a context whose words read nothing animated once (§5.5): 16 objects on a tree with 4 Asks,
    one object animated, over 250 frames."""
    scene = reset(1, 250)
    tree, gi, go = new_tree("Many", ("Level", 'NodeSocketInt'))
    switch = tree.nodes.new("GeometryNodeIndexSwitch")
    switch.data_type = 'STRING'
    for i, slot in enumerate(s for s in switch.inputs if s.identifier not in ("Index", "__extend__")):
        slot.default_value = f"Question {i}?"
    tree.links.new(gi.outputs["Level"], switch.inputs["Index"])
    for k in range(4):
        ask = tree.nodes.new(ASK.idname)
        ask.inputs["Context"].default_value = f"Room {k}"
        tree.links.new(switch.outputs[0], ask.inputs["Question"])
    tree.links.new(gi.outputs["Geometry"], go.inputs["Geometry"])
    objects = [user(f"Many {i}", tree, Level=0) for i in range(16)]
    keyframes(objects[0], "Level", [(1, 0), (125, 1), (250, 0)])
    settle()
    reads, read = [], engine._read
    engine._read = lambda *args: reads.append(args) or read(*args)
    try:
        started = time.perf_counter()
        run = engine.Playback(scene)
        fcurves = run.read_curves()
        seconds = time.perf_counter() - started
    finally:
        engine._read = read
    print(f"TIMING Prepare for Playback: F-curve path over 250 frames, 64 contexts, 4 animated: {seconds:.3f}s "
          f"(budget 1 s)", flush=True)
    live = len(run.live)
    check("static contexts: over 250 frames the F-curve path reads each of the 60 contexts whose words read nothing "
          "animated once, and the 4 animated ones at every frame (the first again once its F-curve is found), in "
          "under 1 s", fcurves and live == 64 and len(reads) - live == 64 + 1 + 249 * 4 and seconds < 1.0,
          (len(reads) - live, f"{seconds:.3f}s"))


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
    check("reload: the OpenAct panel has Prepare for Playback, and counts the pins", "[openact.prepare_playback]" in log
          and "3 word sets pinned for playback" in log, log)
    result = bpy.ops.openact.unpin()
    check("unpin: without a node, nothing is unpinned (Unpin All is per node)", result == {'CANCELLED'}
          and len(pinned(rules)) == 3, result)
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
    again, log = bpy.ops.openact.unpin(uid=rules.uid), drawn(ui.NODE_PT_openact, rules)
    check("unpin: Unpin All unpins the rest; then there is nothing to unpin, and the sidebar claims no pin",
          result == {'FINISHED'} and again == {'CANCELLED'} and not pinned(rules) and not engine.STATUS[rules.uid].pins
          and len(S.load(rules).entries) == 3 and '"pinned":true' not in rules.entries
          and not any("pinned for playback" in str(line) for line in log), (result, again, log))


# ------------------------------------------------------------------------------ the channelbag

def phase_channelbag():
    scene = reset(1, 11)
    tree, ask, _gi = question_tree("Levels", 6)
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
    op, context, calls = modal_operator(scene)
    result = op.invoke(context, None)
    done = settle()
    check("modal: on the F-curve path invoke finishes at once, with no timer, progress or status text",
          result == {'FINISHED'} and not calls and done and len(pinned(ask)) == 6, (result, calls, op.reports))


def scene_words(scene, obj, frames):
    """{words: frames} as Blender sets the Level input at each frame, from 'Question <Level>?'."""
    found = {}
    for frame in frames:
        scene.frame_set(frame)
        level = obj.modifiers["OpenAct"][ident(obj.modifiers["OpenAct"].node_group, "Level")]
        found.setdefault(f"Question {level}?", []).append(frame)
    scene.frame_set(frames[0])
    return found


def phase_ranges():
    """A keyed value beyond the property's hard range, and Time Stretching, which plays animation at other times."""
    scene = reset(1, 10)
    tree, _ask, _gi = question_tree("Ranges", 11)
    obj = user("Ranges", tree, Level=0)
    obj.modifiers["OpenAct"].id_properties_ui(ident(tree, "Level")).update(min=0, max=3)
    keyframes(obj, "Level", [(1, 0), (10, 9)])
    run = engine.Playback(scene)
    fcurves = run.read_curves()
    words = {gn[0]: frames for seen in run.seen.values() for gn, frames in seen.items()}
    truth = scene_words(scene, obj, list(range(1, 11)))
    check("ranges: a keyed value beyond the input's hard range (0-3) is read clamped, as the scene sets it",
          fcurves and words == truth and max(truth) == "Question 3?", (words, truth))
    obj.modifiers["OpenAct"].id_properties_ui(ident(tree, "Level")).update(min=-2 ** 31, max=2 ** 31 - 1)
    keyframes(obj, "Level", [(1, 1), (10, 10)])
    render = scene.render
    render.frame_map_old, render.frame_map_new = 100, 50
    try:
        truth = scene_words(scene, obj, list(range(1, 11)))
        run = engine.Playback(scene)
        fell = not run.read_curves()
        for _frame in run.step_frames():
            pass
        words = {gn[0]: frames for seen in run.seen.values() for gn, frames in seen.items()}
    finally:
        render.frame_map_old = render.frame_map_new = 100
    check("ranges: under Time Stretching (100 -> 50) the words come from setting each frame, as the scene has them "
          "there", fell and run.fallback == "Time Stretching is on" and words == truth and "Question 2?" in truth,
          (run.fallback, words, truth))


# ------------------------------------------------------------------------------ the fallback

def phase_driver():
    scene = reset(1, 29)
    tree, ask, _gi = question_tree("Driven", 3)
    tree.nodes.new("GeometryNodeBake")  # the modifier has a bake: new pins change the rules it may have baked
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
        inside, holding, other = scene.frame_current, engine._stepping, bpy.ops.openact.prepare_playback.poll()
        engine.schedule(immediate=True)
        engine.tick()
        steps.close()  # Esc
    finally:
        engine.run_pass = run_pass
    back = scene.frame_current
    settle()
    check("driver: a driven input takes the fallback, which sets frames 1-5 in turn with no pass meanwhile, nor can "
          "another run start; cancelled, it sets frame 7 again, pins nothing and asks nothing", fell
          and "driven" in run.fallback and stepped == [1, 2, 3, 4, 5] and inside == 5 and holding and not passes
          and not other and back == 7 and not engine._stepping and engine.PLAYBACK is None and not pinned(ask)
          and len(records()) == start and bpy.ops.openact.prepare_playback.poll()
          and 'REGISTER' not in ops.OPENACT_OT_prepare_playback.bl_options, (run.fallback, stepped, inside, passes, back))
    result, error = prepare()
    back = scene.frame_current
    done = settle()
    pins = [(e.gn[0], e.frames) for e in pinned(ask)]
    check("driver: run from the operator, the fallback steps every frame, sets frame 7 again and pins the driven "
          "words", result == {'FINISHED'} and not error and back == 7 and done and pins == [
              ("Question 0?", [[1, 9]]), ("Question 1?", [[10, 19]]), ("Question 2?", [[20, 29]])], (error, back, pins))
    flag = "Re-bake 'Driven > OpenAct' if baked: rules changed for frames 1–29"
    status = engine.STATUS[ask.uid]
    check("re-bake: the pins change the rules of a modifier with a bake: its status and the sidebar say which to "
          "re-bake, and for which frames", engine.REBAKES.get(ask.uid) == flag and engine.PLAYBACK_NOTE == flag
          and status.state == "warning" and flag in status.text and said(drawn(ui.NODE_PT_openact, ask), flag),
          (engine.REBAKES, engine.PLAYBACK_NOTE, status.text))
    start = len(records())
    result = bpy.ops.openact.reask(uid=ask.uid)
    asking = [(e.gn[0], e.frames, e.rows.src) for e in pinned(ask)]
    done = settle()
    pins = [(e.gn[0], e.frames, e.rows.src) for e in pinned(ask)]
    check("Ask Again keeps the pins: the live words' pin waits for its new answer, and gets it", result == {'FINISHED'}
          and done and asking == [("Question 0?", [[1, 9]], "p"), ("Question 1?", [[10, 19]], "m"),
                                  ("Question 2?", [[20, 29]], "m")]
          and pins == [(words, frames, "m") for words, frames, _src in asking] and len(records()) == start + 1,
          (asking, pins, len(records()) - start))
    result, error = prepare()
    settle()
    check("re-bake: a run that changes no pin clears the flag", result == {'FINISHED'} and not engine.REBAKES
          and "Re-bake" not in engine.STATUS[ask.uid].text and len(pinned(ask)) == 3, (error, engine.REBAKES))
    return scene


def phase_modal(scene):
    """The modal operator's own functions on stand-ins, over the driven scene of phase_driver (29 frames, at frame 7)."""
    op, context, calls = modal_operator(scene)
    started = op.invoke(context, None)
    steps = [op.modal(context, TIMER) for _ in range(2)]
    frame, stopped = scene.frame_current, op.modal(context, ESC)
    check("modal: invoke starts the fallback with progress, a status text and a timer; Esc after two frames sets "
          "frame 7 again, leaves no timer, progress or status text, and says nothing was pinned",
          started == {'RUNNING_MODAL'} and ("progress_begin", 0, 29) in calls and ("modal_handler_add", op) in calls
          and steps == [{'RUNNING_MODAL'}] * 2 and frame == 2 and stopped == {'CANCELLED'}
          and scene.frame_current == 7 and not engine._stepping and cleaned(calls)
          and op.reports == [('INFO', "Prepare for Playback cancelled: nothing was pinned")], (calls, op.reports))
    op, context, calls = modal_operator(scene)
    op.invoke(context, None)
    results = [op.modal(context, TIMER) for _ in range(30)]
    check("modal: run to the end, a timer event per frame, then finish(): it reports, sets frame 7 again and leaves "
          "nothing behind", results == [{'RUNNING_MODAL'}] * 29 + [{'FINISHED'}] and scene.frame_current == 7
          and cleaned(calls) and ("progress_update", 29) in calls and op.reports[-1][0] == 'INFO'
          and op.reports[-1][1].startswith("3 word sets over frames 1–29"), (results[-2:], op.reports))
    op, context, calls = modal_operator(scene)
    op.invoke(context, None)
    read, count = engine._read, [0]

    def failing(*args):
        count[0] += 1
        if count[0] == 3:
            raise RuntimeError("simulated internal error")
        return read(*args)

    engine._read = failing
    try:
        results = [op.modal(context, TIMER) for _ in range(3)]
    except RuntimeError as error:  # escaped modal()
        results = [f"escaped: {error}"]
    finally:
        engine._read = read
    check("modal: an internal error at the third frame stops it: CANCELLED with the error, frame 7 again, no timer, "
          "progress or status text left behind", results == [{'RUNNING_MODAL'}] * 2 + [{'CANCELLED'}]
          and scene.frame_current == 7 and not engine._stepping and cleaned(calls)
          and op.reports == [('ERROR', "Prepare for Playback stopped: simulated internal error")], (results, calls))
    op, context, calls = modal_operator(scene)
    op.invoke(context, None)
    op.modal(context, TIMER)
    op.cancel(context)
    check("modal: cancel(), when Blender ends the loop itself, sets frame 7 again and leaves nothing behind",
          scene.frame_current == 7 and not engine._stepping and cleaned(calls), calls)
    settle()


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
    check("limit: 13 distinct word sets are refused with the limit message, which the sidebar keeps, and nothing is "
          "asked or pinned", refused == message == "'Ask': 13 distinct word sets over frames 1–13; the limit is 12"
          and message in error and result == {'CANCELLED'} and len(records()) == start and not pipeline.PENDING
          and engine.PLAYBACK is None and ask.entries == entries and engine.PLAYBACK_NOTE == message
          and said(drawn(ui.NODE_PT_openact, ask), message), (refused, error, len(records()) - start))


# ------------------------------------------------------------------------------ undo, Cancel, Unpin and retry

def phase_undo():
    """Undo for real: a first ed.undo_push() gives a headless Blender its undo stack, and the operators run with undo,
    as their buttons run them. Question <n> occurs at frames 10n+1 to 10n+10; each run widens the range."""
    reset(1, 20)
    tree, ask, _gi = question_tree("Undo", 7)
    ask.inputs["Context"].default_value = "A room where things are undone."  # questions MEMORY holds no answer to
    obj = user("Undo", tree, Level=0)
    constant(keyframes(obj, "Level", [(10 * n + 1, n) for n in range(7)]))
    settle()
    name = ask.name

    def node():  # an undo rebuilds every data-block: every reference is looked up again after one
        return bpy.data.node_groups["Undo"].nodes[name]

    def words():
        return [(e.gn[0], e.frames) for e in pinned(node())]

    bpy.ops.ed.undo_push(message="OpenAct test")
    pushes, push = [], engine._undo_push
    engine._undo_push = lambda message: pushes.append(message) or push(message)
    try:
        result, _error = prepare(undo=True)
        waiting = engine.pinning() == 2 and not pinned(node())
        settle()
    finally:
        engine._undo_push = push
    first = words()
    unpinned = bpy.ops.openact.unpin('EXEC_DEFAULT', True, uid=node().uid, words_key=pinned(node())[1].words_key)
    settle()
    steps = [undo()]  # the Unpin
    kept = words()
    steps.append(undo())  # the pins' own step
    gone, listed, log = words(), engine.STATUS[node().uid].pins, drawn(ui.NODE_PT_openact, node())
    steps.append(undo(redo=True))
    check("undo: pins that a pass writes after their operator's step have a step of their own: undoing a later Unpin "
          "keeps them, undoing that step removes them and the sidebar claims none, and a redo brings them back",
          result == {'FINISHED'} and waiting and pushes == ["Pin for Playback"] and steps == [True] * 3
          and first == [("Question 0?", [[1, 10]]), ("Question 1?", [[11, 20]])] and unpinned == {'FINISHED'}
          and kept == first and not gone and not listed and not said(log, "pinned for playback") and words() == first,
          (result, waiting, pushes, steps, first, unpinned, kept, gone, listed, words(), log))

    bpy.context.scene.frame_end = 40  # Questions 2 and 3 are new: the run waits for them
    result, _error = prepare(undo=True)
    live = engine.pinning()
    try:
        steps = [bpy.ops.ed.undo() == {'FINISHED'}]  # its step, while it waits: no pass may pin before
    except RuntimeError:
        steps = [False]
    dormant, log = engine.pinning(), drawn(ui.NODE_PT_openact, node())
    settle()  # the answers come in, and the dormant run pins nothing
    asleep = words()
    steps.append(undo(redo=True))
    check("undo: undoing Prepare for Playback while its run waits leaves the run dormant: the sidebar shows no run "
          "waiting and nothing is pinned when the answers come in; a redo wakes it, and it pins",
          result == {'FINISHED'} and live == 4 and steps == [True] * 2 and dormant == 0 and not said(log, "Pinning")
          and asleep == first and [w for w, _frames in words()] == [f"Question {n}?" for n in range(4)],
          (live, steps, dormant, asleep, words()))

    bpy.context.scene.frame_end = 50  # Question 4 is new
    prepare()
    waiting = engine.pinning()
    result = bpy.ops.openact.cancel_playback()
    settle()
    check("cancel: Cancel stops the waiting run: its answers come in, and nothing more is pinned", waiting == 5
          and result == {'FINISHED'} and len(words()) == 4 and not engine.pinning()
          and engine.PLAYBACK_NOTE == "Prepare for Playback cancelled: nothing more was pinned", (waiting, words()))

    bpy.context.scene.frame_end = 60  # Question 5 is new
    prepare()
    waiting = engine.pinning()
    result = bpy.ops.openact.unpin(uid=node().uid, words_key=pinned(node())[3].words_key)
    settle()
    check("cancel: Unpin cancels the waiting run, which would pin over it: the other pins stay as they were",
          waiting == 6 and result == {'FINISHED'} and [w for w, _frames in words()] == [
              f"Question {n}?" for n in range(3)] and engine.PLAYBACK_NOTE.startswith(
              "Prepare for Playback cancelled by Unpin"), (waiting, words(), engine.PLAYBACK_NOTE))

    bpy.context.scene.frame_end = 70  # Question 6 is new, and the model refuses it once
    http = server.http

    def refusing(method, path, body=None, port=None, timeout=10.0):
        if path == "/v1/systemone" and "Question 6?" in json.dumps(body):
            return 500, {"error": "simulated model failure"}
        return http(method, path, body, port, timeout)

    server.http = refusing
    try:
        prepare()
        settle()
    finally:
        server.http = http
    stopped, note = words(), engine.PLAYBACK_NOTE
    result, error = prepare()
    settle()
    check("retry: a question the model refused stops the run, naming the node and frame, with nothing pinned; the "
          "next run asks it again and pins", note == ("Prepare for Playback stopped: 'Ask' at frame 61: The model "
                                                      "refused: simulated model failure. Nothing was pinned: run it "
                                                      "again to retry") and len(stopped) == 3
          and result == {'FINISHED'} and [w for w, _frames in words()] == [f"Question {n}?" for n in range(7)],
          (note, stopped, error, words()))


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
        first = not pinned(rules) and engine.pinning() == 7
        answer(asked.values())
        done = drive(engine, lambda: not engine.pinning() and not engine._dirty, 120)
    finally:
        engine._submit = submit
    size, pins = len(rules.entries.encode("utf-8")), pinned(rules)
    projected = float(re.search(r"take about ([\d.]+) MB", text).group(1)) if "take about" in text else 0.0
    notes = [p.message for p in engine.STATUS[rules.uid].problems if p.severity == "info"]
    check("size: Prepare for Playback reports the size of the pins first, before anything is pinned (about the size "
          "they then take)", live and first and text.startswith(
              "7 word sets over frames 1–7: the pinned entries take about") and projected > 2.0
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
        phase_ranges()
        phase_static()
        phase_modal(phase_driver())
        phase_nla()
        phase_limit()
        phase_undo()
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
