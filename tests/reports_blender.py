"""The pass reads each node's words from the Warning channel (§5.3, §5.7) and asks only for the ones reported with its
Trigger on (§6.0), in a headless Blender, against the fake model server that the add-on starts itself
(OPENACT_BACKEND=fake). Every node expected to ask gets a Trigger (_harness.trigger()):
- words built by nodes the static evaluator can't read are planned from what the evaluation reported, and answered:
  Scene Time's frame through Value to String at two frames (switching back asks nothing), a count from a Repeat zone,
  a step of a simulation;
- a For Each Element zone's word sets each become an entry, and the sidebar tells them apart;
- a Pick of a collection's children reads the collection statically while its Prompt comes from the report;
- a user group used three times in one modifier: each reported word tuple goes to the path whose static words equal
  it, one no path's words equal to the first path, and the use whose words merged into another's plans nothing;
- words built from another node's answer wait for it, as in 1.0: an Ask whose Question joins a Pick's Item plans
  nothing while the Pick is unanswered (Geometry Nodes builds its words from the Pick's fallback);
- contexts no evaluation reported are planned statically and never asked, their status says why: a tree no modifier
  uses (Link a Trigger), a node whose outputs are unused, even with its Trigger on (Not evaluated); Ask Again asks
  the first once; a node whose words only Geometry Nodes computes shows a hint that they are read when it
  evaluates, no error;
- the Trigger: unlinked, a node asks nothing and says so; linked to True it asks; turned off, new words ask nothing
  and the stored answer keeps showing, while a cell asked before still fills; Blender's conversion decides what is on
  (a Float -1 off and 0.3 on, an Int, a Vector, a Color); Ask Again asks once without a Trigger; the sidebar shows
  the Trigger at the last evaluation and the word sets reported;
- a modifier that doesn't evaluate reports nothing, whatever list it kept: an object disabled in viewports, a modifier
  disabled in the viewport (so their new words are not asked); an object hidden with the eye still evaluates;
- a copy sharing its original's group: their reports are ignored until each group reports its own uid;
- engine.plan_words() plans one reported word set through the pass's plan cache, with the node's Overrides CSV
  applied, or raises its Hint;
- no pass loop: once everything is answered, ticking with an evaluation after each tick runs only the 2 s safety
  passes, which plan nothing again and bump no generation;
- a file saved by 1.0, without Trigger sockets: it keeps its answers and asks nothing, new words neither;
- the trust rule (§16.1): in an untrusted file, reported words that Geometry Nodes' Import Text read from outside the
  file's folder are refused as a blocked read until the file is allowed.

Run:  blender -b --factory-startup --python tests/reports_blender.py
"""

import os
import shutil
import socket
import sys
import tempfile
import time
import traceback

TMP = tempfile.mkdtemp(prefix="openact_reports_")
os.environ["OPENACT_BACKEND"] = "fake"
os.environ["OPENACT_SERVER_PYTHON"] = sys.executable  # the fake backend needs no runtime: Blender's Python runs it
os.environ["OPENACT_STATE_DIR"] = os.path.join(TMP, "state")  # the server log stays out of any shared runtime

import bpy  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import FAILURES, check, drive, finish, sock, store, trigger  # noqa: E402
from openact import channel, engine, evaluator, inner, nodes, pipeline, server, trust, ui, words  # noqa: E402
from openact import prefs as P  # noqa: E402
from openact import store as S  # noqa: E402

ASK, PICK = engine.ASK_ID, engine.PICK_ID
SPEC = engine.SPECS[ASK]


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


PORT = free_port()
P._Defaults.port, P._Defaults.auto_start = PORT, True  # the add-on isn't enabled, so these are its preferences
nodes.register()
engine.register()
ASKED = []  # every missing cell the passes submit, whether or not the server is up
_submit = engine._submit
engine._submit = lambda job, missing: (ASKED.extend(missing), _submit(job, missing))[1]


# ------------------------------------------------------------------------------ helpers

def scene(name, points=1):
    """A tree used by one object of `points` loose vertices: (tree, Group Input, Group Output, object, modifier)."""
    tree = bpy.data.node_groups.new(name, 'GeometryNodeTree')
    tree.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    tree.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([(i, 0, 0) for i in range(points)], [], [])
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    mod = obj.modifiers.new(name, 'NODES')
    mod.node_group = tree
    return tree, tree.nodes.new("NodeGroupInput"), tree.nodes.new("NodeGroupOutput"), obj, mod


def along(tree, start, chain, end):
    """Pass the geometry from `start` through the Geometry sockets of each node of `chain` into `end`, so the modifier
    evaluates each of them."""
    for node in chain:
        tree.links.new(start, node.inputs["Geometry"])
        start = node.outputs["Geometry"]
    tree.links.new(start, end)


def kept(tree, gi, go, node, output, name, data_type):
    """The geometry through the node, then its output stored on it as the point attribute `name`, into the Group
    Output: the modifier evaluates both."""
    tree.links.new(gi.outputs["Geometry"], node.inputs["Geometry"])
    tree.links.new(store(tree, node.outputs["Geometry"], node.outputs[output], name, data_type), go.inputs["Geometry"])


def listed(mod):
    """The first word of each report the modifier lists, whichever node's."""
    return [report.words[0] for report in map(words.parse, (w.message for w in mod.node_warnings)) if report]


def enabled(sockets, name):
    return next(s for s in sockets if s.name == name and s.enabled)


def named(sockets, name):
    return next(s for s in sockets if s.name == name)


def numbered(tree, prefix, number):
    """A String socket: prefix + the Int `number` socket, through Value to String (INT) and Join Strings."""
    text = tree.nodes.new("FunctionNodeValueToString")
    text.data_type = 'INT'
    head = tree.nodes.new("FunctionNodeInputString")
    head.string = prefix
    join = tree.nodes.new("GeometryNodeStringJoin")
    tree.links.new(number, enabled(text.inputs, "Value"))
    for item in (text.outputs["String"], head.outputs[0]):  # Join Strings puts the most recently linked item first
        tree.links.new(item, join.inputs["Strings"])
    return join.outputs[0]


def frame_number(tree):
    """The scene's frame as an Int socket: Scene Time's Frame through Float to Integer."""
    whole = tree.nodes.new("FunctionNodeFloatToInt")
    tree.links.new(tree.nodes.new("GeometryNodeInputSceneTime").outputs["Frame"], whole.inputs[0])
    return whole.outputs[0]


def attribute(obj, name):
    """The values of the evaluated object's point attribute `name`."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        found = mesh.attributes.get(name)
        return [d.value for d in found.data] if found else None
    finally:
        evaluated.to_mesh_clear()


def asks(*question):
    """Ask's word tuple: (Question, Context, Answers, Overrides, Key)."""
    return tuple(question) + ("",) * (5 - len(question))


def stored(node):
    return {tuple(entry.gn) for entry in S.load(node).entries}


def answered(node, gns, timeout=30):
    """Tick until the node stores an entry for each of these word tuples, and nothing is left to ask."""
    return drive(engine, lambda: set(gns) <= stored(node) and not pipeline.PENDING and not engine._dirty, timeout)


def static(node, ctx):
    """What the static evaluator reads as the node's words in ctx: the words, or the error's class and text."""
    try:
        return engine.SPECS[node.bl_idname].gn(node, ctx)
    except evaluator.Unresolvable as error:
        return f"{type(error).__name__}: {error}"


def uses(node):
    """(label, words) of each word set the node's last pass planned a Job for, as the sidebar lists them."""
    return [(label, job.gn) for label, job, _entry in engine.answers_for(node)]


def reported(node, obj, mod):
    """The word tuples the pass read from the modifier's list for the node: {words: triggered}."""
    return engine.REPORTS.get(node.uid, {}).get((obj.as_pointer(), mod.name))


# ------------------------------------------------------------------------------ phases

def phase_start():
    check("start: the add-on starts the fake server, which gets ready",
          server.start() and drive(engine, server.is_ready, 30), server.status())


def phase_zones():
    """A count from a Repeat zone and a step of a simulation, which only Geometry Nodes computes."""
    tree, gi, go, obj, mod = scene("Zones")
    begin, end = tree.nodes.new("GeometryNodeRepeatInput"), tree.nodes.new("GeometryNodeRepeatOutput")
    begin.pair_with_output(end)
    end.repeat_items.new('INT', "Count")
    begin.inputs["Iterations"].default_value = 3
    add = tree.nodes.new("FunctionNodeIntegerMath")
    add.operation, add.inputs[1].default_value = 'ADD', 1
    tree.links.new(named(begin.outputs, "Count"), add.inputs[0])
    tree.links.new(add.outputs[0], named(end.inputs, "Count"))
    counted = tree.nodes.new(ASK)
    tree.links.new(numbered(tree, "Is this count odd: ", named(end.outputs, "Count")), counted.inputs["Question"])
    sim_in, sim_out = tree.nodes.new("GeometryNodeSimulationInput"), tree.nodes.new("GeometryNodeSimulationOutput")
    sim_in.pair_with_output(sim_out)
    sim_out.state_items.new('INT', "Step")
    step = tree.nodes.new("FunctionNodeIntegerMath")
    step.operation, step.inputs[1].default_value = 'ADD', 1
    tree.links.new(named(sim_in.outputs, "Step"), step.inputs[0])
    tree.links.new(step.outputs[0], named(sim_out.inputs, "Step"))
    tree.links.new(gi.outputs["Geometry"], named(sim_in.inputs, "Geometry"))
    tree.links.new(named(sim_in.outputs, "Geometry"), named(sim_out.inputs, "Geometry"))
    simulated = tree.nodes.new(ASK)
    tree.links.new(numbered(tree, "Is this step odd: ", named(sim_out.outputs, "Step")), simulated.inputs["Question"])
    geometry = store(tree, named(sim_out.outputs, "Geometry"), named(sim_out.outputs, "Step"), "step", 'INT')
    along(tree, geometry, [counted, simulated], go.inputs["Geometry"])
    trigger(counted)
    trigger(simulated)
    ctx = evaluator.Ctx(obj, mod, (), tree)
    for frame in (1, 2, 3):  # a simulation steps one frame at a time from the start frame
        bpy.context.scene.frame_set(frame)
    steps = attribute(obj, "step")
    words = [asks("Is this count odd: 3"), asks(f"Is this step odd: {steps[0]}")]
    done = answered(counted, words[:1]) and answered(simulated, words[1:])
    got = (static(counted, ctx), static(simulated, ctx))
    check("zones: words the static evaluator can't read (a Repeat zone's count, a simulation's step) are planned from "
          "what the evaluation reported and answered, the simulation's at the step Geometry Nodes computed",
          done and all(text.startswith("Unreadable: ") for text in got) and steps == [3]
          and (uses(counted), uses(simulated)) == ([("Zones > Zones", words[0])], [("Zones > Zones", words[1])])
          and stored(counted) == {words[0]} and stored(simulated) == {words[1]},
          (done, got, steps, uses(counted), uses(simulated)))


def phase_frame():
    """Scene Time's frame through Value to String, at two frames: each frame's words are planned and answered, and the
    inner group shows each answer at its frame; going back to the first frame asks nothing."""
    tree, gi, go, obj, mod = scene("Frame")
    ask = tree.nodes.new(ASK)
    tree.links.new(numbered(tree, "Is the scene at frame ", frame_number(tree)), ask.inputs["Question"])
    kept(tree, gi, go, ask, "Answer", "answer", 'FLOAT')
    trigger(ask)
    ctx = evaluator.Ctx(obj, mod, (), tree)
    got = {}
    for frame in (7, 12):
        bpy.context.scene.frame_set(frame)
        words = asks(f"Is the scene at frame {frame}")
        done = answered(ask, [words])
        entry = next((e for e in S.load(ask).entries if tuple(e.gn) == words), None)
        shown = attribute(obj, "answer")
        got[frame] = (done, reported(ask, obj, mod), uses(ask),
                      entry is not None and shown is not None and abs(shown[0] - entry.single["Answer"]) < 1e-6)
    check("frame: the words are planned from the report at each frame (Scene Time's frame, which the static evaluator "
          "can't read), answered, and the inner group shows each frame's own answer",
          static(ask, ctx).startswith("Unreadable: ") and all(
              value == (True, {asks(f"Is the scene at frame {frame}"): True},
                        [("Frame > Frame", asks(f"Is the scene at frame {frame}"))], True)
              for frame, value in got.items()), (static(ask, ctx), got))
    before = len(ASKED)
    bpy.context.scene.frame_set(7)
    drive(engine, lambda: False, 1.0)  # the safety pass is 2 s away: the frame change schedules this one
    check("frame: back at the first frame, a pass plans its words (a frame change schedules one), its stored answer "
          "shows and nothing is asked",
          len(ASKED) == before and uses(ask) == [("Frame > Frame", asks("Is the scene at frame 7"))]
          and stored(ask) == {asks("Is the scene at frame 7"), asks("Is the scene at frame 12")},
          (len(ASKED) - before, uses(ask)))
    bpy.context.scene.frame_set(12)
    return ask, obj, mod


def phase_zone():
    """A For Each Element zone over 3 points reports one word tuple per element: each is planned and stored."""
    tree, gi, go, obj, mod = scene("Each", points=3)
    begin = tree.nodes.new("GeometryNodeForeachGeometryElementInput")
    end = tree.nodes.new("GeometryNodeForeachGeometryElementOutput")
    begin.pair_with_output(end)
    ask = tree.nodes.new(ASK)
    tree.links.new(numbered(tree, "Is item ", begin.outputs["Index"]), ask.inputs["Question"])
    ask.inputs["Context"].default_value = "A row of three items."
    tree.links.new(gi.outputs["Geometry"], begin.inputs["Geometry"])
    along(tree, begin.outputs["Element"], [ask], sock(end.inputs, "Generation_0"))
    tree.links.new(sock(end.outputs, "Generation_0"), go.inputs["Geometry"])
    trigger(ask)
    words = [asks(f"Is item {i}", "A row of three items.") for i in range(3)]
    done = answered(ask, words)
    check("zone: each element's word tuple is its own word set, planned and stored as an entry, and the sidebar "
          "labels each with its first word", done and stored(ask) == set(words) and uses(ask) == [
              (f"Each > Each: Is item {i}", words[i]) for i in range(3)], (done, stored(ask), uses(ask)))


def phase_collection():
    """A Pick of a collection's children, its Prompt from Scene Time: the collection is read statically, the Prompt from
    the report."""
    shelf = bpy.data.collections.new("Shelf")
    for name in ("Vase", "Lamp", "Book"):
        shelf.objects.link(bpy.data.objects.new(name, None))
    tree, gi, go, obj, mod = scene("Shelf")
    pick = tree.nodes.new(PICK)
    pick.list_source = 'COLLECTION'
    pick.inputs["Collection"].default_value = shelf
    tree.links.new(numbered(tree, "something to read at frame ", frame_number(tree)), pick.inputs["Prompt"])
    kept(tree, gi, go, pick, "Index", "index", 'INT')
    trigger(pick)
    words = ("something to read at frame 12", "", "", "", "")
    done = answered(pick, [words])
    job = engine.answers_for(pick)[0][1] if engine.answers_for(pick) else None
    ctx = evaluator.Ctx(obj, mod, (), tree)
    check("collection: the Prompt comes from the report (the static evaluator can't read it), the items from the "
          "collection, read statically in natural order, and the pick is answered",
          done and static(pick, ctx).startswith("Unreadable: ") and job is not None and job.gn == words
          and [row.name for row in job.rows] == ["Book", "Lamp", "Vase"] and stored(pick) == {words},
          (done, job and job.gn, job and [row.name for row in job.rows]))


def phase_paths():
    """A user group holding an Ask, used three times in one modifier: twice with typed words, once with words from
    Scene Time, which no path's static words equal."""
    user = bpy.data.node_groups.new("Asker", 'GeometryNodeTree')
    user.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    user.interface.new_socket("Words", in_out='INPUT', socket_type='NodeSocketString')
    user.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    ugi, ugo = user.nodes.new("NodeGroupInput"), user.nodes.new("NodeGroupOutput")
    asker = user.nodes.new(ASK)
    user.links.new(ugi.outputs["Words"], asker.inputs["Question"])
    along(user, ugi.outputs["Geometry"], [asker], ugo.inputs["Geometry"])
    trigger(asker)  # inside the group: every use is triggered
    tree, gi, go, obj, mod = scene("Paths")
    groups = []
    for text in ("Is it the left one?", "Is it the right one?", "Is it the left one?", None):
        group = tree.nodes.new("GeometryNodeGroup")
        group.node_tree = user
        if text is None:
            tree.links.new(numbered(tree, "Is it frame ", frame_number(tree)), group.inputs["Words"])
        else:
            group.inputs["Words"].default_value = text
        groups.append(group)
    along(tree, gi.outputs["Geometry"], groups, go.inputs["Geometry"])
    left, right, framed = asks("Is it the left one?"), asks("Is it the right one?"), asks("Is it frame 12")
    done = answered(asker, [left, right, framed])
    contexts = evaluator.find_uses()[user.as_pointer()]
    sets = [([group.name for group in ctx.path], words)
            for ctx, words in engine._sets(SPEC, asker, contexts, engine.REPORTS.get(asker.uid))]
    check("paths: a modifier's report goes to the group path whose static words equal each tuple, one no path's static "
          "words equal (Scene Time's) to the first path; the use whose words merged into the first's plans nothing",
          done and sets == [([groups[0].name], framed), ([groups[0].name], left), ([groups[1].name], right)]
          and sorted(words for _label, words in uses(asker)) == sorted([left, right, framed]), (done, sets))


def phase_chain():
    """An Ask whose Question joins a Pick's Item: until the Pick is answered, Geometry Nodes builds the Question from
    its fallback (""), and the static evaluator finds the Ask waiting for the Pick (§5.3)."""
    tree, gi, go, obj, mod = scene("Chain")
    pick = tree.nodes.new(PICK)
    pick.inputs["Prompt"].default_value = "a fruit for the bowl"
    pick.inputs["Items"].default_value = "apple, pear, plum"
    ask = tree.nodes.new(ASK)
    join = tree.nodes.new("GeometryNodeStringJoin")
    head, tail = tree.nodes.new("FunctionNodeInputString"), tree.nodes.new("FunctionNodeInputString")
    head.string, tail.string = "Is the ", " sweet?"
    for item in (tail.outputs[0], pick.outputs["Item"], head.outputs[0]):  # the most recently linked item comes first
        tree.links.new(item, join.inputs["Strings"])
    tree.links.new(join.outputs[0], ask.inputs["Question"])
    along(tree, gi.outputs["Geometry"], [ask], go.inputs["Geometry"])
    trigger(pick)
    trigger(ask)
    before = len(ASKED)
    server.Worker.render_running.set()  # the worker starts no chunk, as during a render job: passes run, ticks don't
    try:
        time.sleep(0.3)
        engine.run_pass()
        status = engine.STATUS.get(ask.uid)
        waiting = (reported(ask, obj, mod), status and status.text, uses(ask),
                   [cell.question["instructions"] for cell in ASKED[before:] if cell.question["type"] == "noul"])
    finally:
        server.Worker.render_running.clear()
    done = drive(engine, lambda: bool(S.load(ask).entries) and not pipeline.PENDING, 30)
    item = S.load(pick).entries[0].single["Item"] if S.load(pick).entries else None
    check("chain: while the Pick has no answer, the Ask's reported Question holds the Pick's fallback, which is "
          "neither planned nor asked: the Ask waits for the Pick, as in 1.0; once the Pick is answered, the Ask's "
          "question with its item is planned and answered",
          waiting == ({asks("Is the  sweet?"): True}, "Waiting for another OpenAct node...", [], []) and done
          and stored(ask) == {asks(f"Is the {item} sweet?")}, (waiting, done, item, stored(ask)))


def edited(seconds=0.5):
    """A pass now, as Blender's event loop schedules one after an edit (evaluating the change, whose depsgraph update
    schedules it; headless, nothing evaluates), then `seconds` of ticks."""
    engine.run_pass()
    drive(engine, lambda: False, seconds)


def asked_about(start, text):
    """The cells the passes submitted since ASKED[start] whose question is `text`."""
    return [cell for cell in ASKED[start:] if cell.question.get("instructions") == text]


def line(node):
    """The node's status line as its body draws it: (icon, text)."""
    status = engine.STATUS.get(node.uid)
    return (status.icon, status.text) if status else None


def phase_static():
    """Contexts no evaluation reported: a tree no modifier uses, a node whose outputs are unused."""
    lone = bpy.data.node_groups.new("Lone", 'GeometryNodeTree')
    lone.use_fake_user = True
    alone = lone.nodes.new(ASK)
    alone.inputs["Question"].default_value = "Is anyone there?"
    tree, gi, go, obj, mod = scene("Unused")
    tree.links.new(gi.outputs["Geometry"], go.inputs["Geometry"])
    unused = tree.nodes.new(ASK)
    unused.inputs["Question"].default_value = "Am I used?"
    switch = tree.nodes.new("FunctionNodeInputBool")  # a Trigger on, but nothing evaluates the node
    switch.boolean = True
    tree.links.new(switch.outputs[0], unused.inputs["Trigger"])
    later = tree.nodes.new(ASK)  # words only Geometry Nodes computes, never evaluated
    tree.links.new(numbered(tree, "Is it frame ", frame_number(tree)), later.inputs["Question"])
    before = len(ASKED)
    drive(engine, lambda: False, 1.5)
    status = engine.STATUS.get(later.uid)
    check("static: a tree no modifier uses and a node whose outputs are unused report nothing, so their words are read "
          "statically and planned, and never asked, even with a Trigger on: the status says to link a Trigger, or that "
          "nothing evaluates the node",
          alone.uid not in engine.REPORTS and unused.uid not in engine.REPORTS
          and uses(alone) == [("(not used by a modifier)", asks("Is anyone there?"))]
          and uses(unused) == [("Unused > Unused", asks("Am I used?"))]
          and not asked_about(before, "Is anyone there?") and not asked_about(before, "Am I used?")
          and line(alone) == ('RADIOBUT_OFF', "Link a Trigger to ask: 1 answer missing")
          and line(unused) == ('RADIOBUT_OFF', "Not evaluated: 1 answer missing")
          and ui.activity_rows(engine.STATUS[unused.uid]) == [ui.TRIGGER_ROWS["unused"]],
          (uses(alone), uses(unused), line(alone), line(unused)))
    check("static: an unused node whose words only Geometry Nodes computes shows a hint that they are read when the "
          "node evaluates, not an error", status is not None and (status.icon, status.text) == (
              'INFO', engine.READ_LATER.format("Question")) and not status.problems and not uses(later),
          status and (status.icon, status.text, status.problems))
    engine.reask(alone)
    done = answered(alone, [asks("Is anyone there?")])
    check("static: Ask Again asks a statically planned word set once, with no Trigger and no evaluation",
          done and len(asked_about(before, "Is anyone there?")) == 1 and alone.uid not in engine.ASK_ONCE,
          (done, len(asked_about(before, "Is anyone there?"))))


def phase_trigger():
    """The Trigger decides asking (§6.0): an Ask on the geometry path, its Answer stored, first with no Trigger."""
    tree, gi, go, obj, mod = scene("Trigger")
    ask = tree.nodes.new(ASK)
    ask.inputs["Question"].default_value = "Is the lamp on?"
    kept(tree, gi, go, ask, "Answer", "answer", 'FLOAT')
    lamp, before = asks("Is the lamp on?"), len(ASKED)
    drive(engine, lambda: False, 1.5)
    status = engine.STATUS.get(ask.uid)
    check("trigger: a node whose Trigger is unlinked asks and stores nothing; its status says to link one, as "
          "information, and the sidebar that its Trigger is unlinked and 1 word set was reported",
          not asked_about(before, "Is the lamp on?") and not S.load(ask).entries
          and line(ask) == ('RADIOBUT_OFF', "Link a Trigger to ask: 1 answer missing") and not status.problems
          and ui.activity_rows(status) == [ui.TRIGGER_ROWS["unlinked"], ('BLANK1', "1 word set reported")],
          (line(ask), status and ui.activity_rows(status)))
    switch = trigger(ask)
    done = answered(ask, [lamp])
    answer = S.load(ask).entries[0].single["Answer"] if done else None
    check("trigger: linked to a Boolean set to True, the node asks and stores the answer, which shows; the sidebar "
          "says the Trigger was on", done and line(ask)[0] == 'CHECKMARK'
          and abs(attribute(obj, "answer")[0] - answer) < 1e-6
          and ui.activity_rows(engine.STATUS[ask.uid])[0] == ui.TRIGGER_ROWS["on"], (done, line(ask)))
    switch.boolean = False
    ask.inputs["Question"].default_value = "Is the lamp broken?"
    before = len(ASKED)
    edited(1.0)
    check("trigger: turned off, new words ask nothing: the status says the Trigger is off and how many answers are "
          "missing, and the stored answer keeps showing",
          not asked_about(before, "Is the lamp broken?") and stored(ask) == {lamp}
          and line(ask) == ('RADIOBUT_OFF', "Trigger off: 1 answer missing")
          and abs(attribute(obj, "answer")[0] - answer) < 1e-6
          and ui.activity_rows(engine.STATUS[ask.uid])[0] == ui.TRIGGER_ROWS["off"], (line(ask), stored(ask)))
    switch.boolean = True
    ask.inputs["Question"].default_value = "Is the lamp new?"
    server.Worker.render_running.set()  # the worker starts no chunk: the cell stays queued
    try:
        time.sleep(0.3)
        engine.run_pass()
        queued = [cell.key in pipeline.PENDING for cell in asked_about(before, "Is the lamp new?")]
        switch.boolean = False
        engine.run_pass()
        filling = line(ask)[0], engine.STATUS[ask.uid].pending
    finally:
        server.Worker.render_running.clear()
    done = answered(ask, [asks("Is the lamp new?")])
    check("trigger: a cell asked before the Trigger went off still fills: the node shows filling meanwhile, and the "
          "answer is stored", queued == [True] and filling == ('SORTTIME', 1) and done, (queued, filling, done))
    cases = (("ShaderNodeValue", "Float", lambda node, value: setattr(node.outputs[0], "default_value", value),
              ((-1.0, False), (0.3, True))),
             ("FunctionNodeInputInt", "Int", lambda node, value: setattr(node, "integer", value),
              ((0, False), (2, True))),
             ("FunctionNodeInputVector", "Vector", lambda node, value: setattr(node, "vector", value),
              (((0.0, 0.0, 0.0), False), ((0.0, -0.5, 0.0), True))),
             ("FunctionNodeInputColor", "Color", lambda node, value: setattr(node, "value", value),
              (((0.0, 0.0, 0.0, 1.0), False), ((0.2, 0.1, 0.0, 0.0), True))))
    got = []
    for idname, kind, put, values in cases:
        source = tree.nodes.new(idname)
        tree.links.new(source.outputs[0], ask.inputs["Trigger"])
        for value, on in values:
            put(source, value)
            question = f"Is the lamp on with a {kind} Trigger of {value}?"
            ask.inputs["Question"].default_value = question
            before = len(ASKED)
            done = answered(ask, [asks(question)]) if on else edited()
            got.append((kind, value, on, bool(asked_about(before, question)), bool(done), line(ask)[0]))
    check("trigger: Blender's conversion into the Bool socket decides what is on: a Float -1 off and 0.3 on, an Int 0 "
          "off and 2 on, a zero Vector off and one with a non-zero component on, a black Color off and a lighter one "
          "on whatever its alpha (off: the status says the Trigger is off)",
          all(on == asked == done and icon == ('CHECKMARK' if on else 'RADIOBUT_OFF')
              for _kind, _value, on, asked, done, icon in got), got)
    tree.links.remove(ask.inputs["Trigger"].links[0])
    ask.inputs["Question"].default_value = "Is the lamp on?"
    edited()
    before = len(ASKED)
    engine.reask(ask)
    done = answered(ask, [lamp])
    drive(engine, lambda: False, 2.5)  # later passes, the safety pass's included
    check("Ask Again: with the Trigger unlinked, the node asks its current words once: answered and stored, and later "
          "passes ask nothing more", done and len(asked_about(before, "Is the lamp on?")) == 1
          and len(ASKED) == before + 1 and ask.uid not in engine.ASK_ONCE and line(ask)[0] == 'CHECKMARK',
          (done, len(ASKED) - before, line(ask)))


def phase_hidden():
    """A modifier that doesn't evaluate keeps the list of its last evaluation, which the pass ignores."""
    got = {}
    for how in ("disabled in viewports", "modifier disabled", "hidden with the eye"):
        tree, gi, go, obj, mod = scene(how)
        ask = tree.nodes.new(ASK)
        ask.inputs["Question"].default_value = "Is it the first question?"
        along(tree, gi.outputs["Geometry"], [ask], go.inputs["Geometry"])
        trigger(ask)
        first = answered(ask, [asks("Is it the first question?")]) and reported(ask, obj, mod)
        if how == "disabled in viewports":
            obj.hide_viewport = True
        elif how == "modifier disabled":
            mod.show_viewport = False
        else:
            obj.hide_set(True)
        ask.inputs["Question"].default_value = "Is it the second question?"
        edited(0.0)
        done = answered(ask, [asks("Is it the second question?")], timeout=2.5)
        got[how] = (first == {asks("Is it the first question?"): True}, done, listed(mod), reported(ask, obj, mod),
                    uses(ask), line(ask))
    second = asks("Is it the second question?")
    check("hidden: an object disabled in viewports and a modifier disabled in the viewport keep their last list, "
          "which the pass ignores: it plans the words it reads statically instead, which nothing asks",
          all(got[how] == (True, False, ["Is it the first question?"], None, [(f"{how} > {how}", second)],
                           ('RADIOBUT_OFF', "Not evaluated: 1 answer missing"))
              for how in ("disabled in viewports", "modifier disabled")), got)
    check("hidden: an object hidden with the eye still evaluates, so its report holds the new words, which its Trigger "
          "asks", got["hidden with the eye"][:2] == (True, True) and got["hidden with the eye"][2:5] == (
              ["Is it the second question?"], {second: True},
              [("hidden with the eye > hidden with the eye", second)]), got["hidden with the eye"])


def phase_copies():
    """A copy shares its original's group until the pass gives it its own, whose uid node holds the original's uid until
    the pass rewrites it: the shared uid's reports belong to neither."""
    tree, gi, go, obj, mod = scene("Copies")
    original, copy = tree.nodes.new(ASK), tree.nodes.new(ASK)
    original.inputs["Question"].default_value = "Am I the original?"
    copy.inputs["Question"].default_value = "Am I the copy?"
    along(tree, gi.outputs["Geometry"], [original], go.inputs["Geometry"])
    switch = trigger(original)
    answered(original, [asks("Am I the original?")])
    copy.node_tree, copy.uid, copy.entries = original.node_tree, "", original.entries  # as duplicated in the editor
    tree.links.new(original.outputs["Geometry"], copy.inputs["Geometry"])
    tree.links.new(copy.outputs["Geometry"], go.inputs["Geometry"])
    tree.links.new(switch.outputs[0], copy.inputs["Trigger"])
    contexts = evaluator.find_uses()[tree.as_pointer()]
    shared = channel.read(list(engine.iter_nodes()), contexts, channel.evaluated())
    raw = sorted(listed(mod))
    engine.run_pass()  # gives the copy its uid and group, whose uid node the pass rewrites
    first = original.uid in engine.REPORTS
    done = answered(copy, [asks("Am I the copy?")])
    check("copies: while the copy shares the original's group, both report the original's uid, which the reader leaves "
          "out, also in the pass that gives the copy its own group; after that each group reports its own uid",
          raw == ["Am I the copy?", "Am I the original?"] and original.uid not in shared and not first and done
          and reported(original, obj, mod) == {asks("Am I the original?"): True}
          and reported(copy, obj, mod) == {asks("Am I the copy?"): True}, (raw, shared, first, done))


def phase_plan_words():
    """engine.plan_words(), for the playback path."""
    path = os.path.join(TMP, "frame_overrides.csv")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write("question,answer,lock\nIs it frame 12,yes,1\n")
    tree, gi, go, obj, mod = scene("Helper")
    ask = tree.nodes.new(ASK)
    tree.links.new(numbered(tree, "Is it frame ", frame_number(tree)), ask.inputs["Question"])
    ask.inputs["Overrides"].default_value = path
    along(tree, gi.outputs["Geometry"], [ask], go.inputs["Geometry"])
    trigger(ask)
    words = asks("Is it frame 12", "", "", path)
    done = answered(ask, [words])
    ctx = evaluator.find_uses()[tree.as_pointer()][0]
    planned = engine.PLANS.get((ask.uid, ctx.key(), words), (None, None))[1]
    plans = engine._plans_made
    job = engine.plan_words(ask, ctx, words)
    plans = engine._plans_made - plans
    try:
        engine.plan_words(ask, ctx, asks("", "", "", path))
        empty = "planned"
    except evaluator.Hint as hint:
        empty = str(hint)
    check("plan_words(): a reported word set's Job comes from the pass's plan cache (the same Job, nothing planned "
          "again), with the node's Overrides CSV applied (its row is a lock); empty words raise the Hint",
          done and job is planned and not plans and job.gn == words
          and [row.source for row in job.rows] == ["lock"] and empty == "Type a question",
          (done, job is planned, plans, job.gn, [row.source for row in job.rows], empty))


def phase_loop(ask):
    """Everything answered: ticks, each followed by an evaluation as Blender's event loop does after a timer."""
    settled = drive(engine, lambda: not pipeline.PENDING and not engine._dirty, 30)
    passes, run_pass = [], engine.run_pass
    plans, generations, asked = engine._plans_made, dict(engine.GENERATIONS), len(ASKED)

    def counted(everything=True):
        passes.append(everything)  # the safety pass visits every node; a scheduled one passes False
        return run_pass(everything)

    engine.run_pass = counted
    try:
        end = time.time() + 5.0
        while time.time() < end:
            engine.tick()
            bpy.context.evaluated_depsgraph_get()
            time.sleep(0.05)
    finally:
        engine.run_pass = run_pass
    check("no pass loop: with everything answered, 5 s of ticks each followed by an evaluation run only the 2 s safety "
          "passes, which plan nothing again, ask nothing and bump no generation",
          settled and 2 <= len(passes) <= 3 and all(passes) and engine._plans_made == plans
          and engine.GENERATIONS == generations and len(ASKED) == asked and ask.uid in engine.REPORTS,
          (settled, passes, engine._plans_made - plans, len(ASKED) - asked))


def phase_saved():
    """A file saved by OpenAct 1.0: its node has stored answers and none of the Geometry and Trigger sockets."""
    bpy.ops.wm.read_homefile(use_empty=True)  # File > New
    tree, gi, go, obj, mod = scene("Old")
    ask = tree.nodes.new(ASK)
    ask.name = "Old"
    ask.inputs["Question"].default_value = "Was it saved by 1.0?"
    kept(tree, gi, go, ask, "Answer", "answer", 'FLOAT')
    switch = trigger(ask)
    answered(ask, [asks("Was it saved by 1.0?")])
    answer = S.load(ask).entries[0].single["Answer"]
    tree.nodes.remove(switch)
    shared = {(d.in_out, d.name) for d in engine.SPECS[ASK].interface(ask)[:3]}  # Geometry in, Trigger, Geometry out
    inner.sync_interface(ask, [d for d in engine.SPECS[ASK].interface(ask) if (d.in_out, d.name) not in shared])
    inner.build(ask, S.load(ask))
    kept_store = next(n for n in tree.nodes if n.bl_idname == "GeometryNodeStoreNamedAttribute")
    tree.links.new(gi.outputs["Geometry"], kept_store.inputs["Geometry"])  # its Geometry sockets went with the layout
    blend = os.path.join(TMP, "old.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend)
    bpy.ops.wm.open_mainfile(filepath=blend)
    ask, obj = bpy.data.node_groups["Old"].nodes["Old"], bpy.data.objects["Old"]
    entries, before = {tuple(e.gn): e.single for e in S.load(ask).entries}, len(ASKED)
    lacked = "Trigger" not in ask.inputs
    drive(engine, lambda: False, 2.5)
    opened = (lacked, "Trigger" in ask.inputs and not ask.inputs["Trigger"].is_linked, line(ask)[0],
              abs(attribute(obj, "answer")[0] - answer) < 1e-6)
    ask.inputs["Question"].default_value = "Was it saved by 1.0, really?"
    edited(1.0)
    check("1.0 file: opened, a node saved without Trigger sockets gains them, its Trigger unlinked, and shows its "
          "stored answer; it asks nothing, new words neither: its status says to link a Trigger, and the stored "
          "answer keeps showing", opened == (True, True, 'CHECKMARK', True) and len(ASKED) == before
          and {tuple(e.gn): e.single for e in S.load(ask).entries} == entries
          and line(ask) == ('RADIOBUT_OFF', "Link a Trigger to ask: 1 answer missing")
          and abs(attribute(obj, "answer")[0] - answer) < 1e-6,
          (opened, len(ASKED) - before, line(ask), attribute(obj, "answer"), answer))


def phase_trust():
    """An untrusted file (§16.1) whose Import Text node reads a file outside its folder into a word: Geometry Nodes
    reads it, and the evaluation reports it."""
    trust.register()
    bpy.ops.wm.read_homefile(use_empty=True)  # File > New: the earlier phases stay out of the saved file
    proj, outside = os.path.join(TMP, "proj"), os.path.join(TMP, "outside")
    os.makedirs(proj)
    os.makedirs(outside)
    secret = os.path.join(outside, "secret.txt")
    with open(secret, "w", encoding="utf-8") as handle:
        handle.write("Is the secret out?")
    tree, gi, go, obj, mod = scene("Secret")
    ask = tree.nodes.new(ASK)
    ask.name = "Secret"
    text = tree.nodes.new("GeometryNodeImportText")
    text.inputs["Path"].default_value = secret
    tree.links.new(text.outputs[0], ask.inputs["Question"])
    along(tree, gi.outputs["Geometry"], [ask], go.inputs["Geometry"])
    trigger(ask)
    blend = os.path.join(proj, "secret.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend)
    bpy.ops.wm.open_mainfile(filepath=blend)  # opened: not trusted
    ask, mod = bpy.data.node_groups["Secret"].nodes["Secret"], bpy.data.objects["Secret"].modifiers["Secret"]
    before = len(ASKED)
    drive(engine, lambda: False, 2.5)
    status = engine.STATUS.get(ask.uid)
    sent = [cell for cell in ASKED[before:] if "secret" in cell.question["instructions"]]
    check("trust: in an untrusted file, words an Import Text node reads from outside the file's folder are reported, "
          "but refused as a blocked read: nothing is planned, asked or stored",
          not trust.is_trusted() and listed(mod) == ["Is the secret out?"] and status is not None
          and status.icon == 'CANCEL' and status.text.startswith(f"{trust.BLOCKED} {secret}:") and not sent
          and not S.load(ask).entries, (listed(mod), status and status.text, len(sent)))
    trust.allow()  # Allow This File
    try:
        done = answered(ask, [asks("Is the secret out?")])
    finally:
        P._Defaults.trusted_files.clear()
    check("trust: allowed, the reported words are planned, asked and stored", done, stored(ask))


def main():
    print("BLENDER", bpy.app.version_string, "fake model server on port", PORT, flush=True)
    try:
        phase_start()
        phase_zones()
        ask, _obj, _mod = phase_frame()
        phase_zone()
        phase_collection()
        phase_paths()
        phase_chain()
        phase_static()
        phase_trigger()
        phase_hidden()
        phase_copies()
        phase_plan_words()
        phase_loop(ask)
        phase_saved()
        phase_trust()
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
