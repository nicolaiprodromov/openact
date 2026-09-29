"""The Warning channel (§5.7) in a headless Blender. No model needed.

Every inner group reports "OpenAct <uid> <0|1> <key text>" through one INFO Warning node, which the modifier lists
after each evaluation that evaluates the node:
- on all six nodes the message equals words.message() of the node's uid, its Trigger and the words the static
  evaluator reads (spec.gn()), so the key text Geometry Nodes builds is Python's, Unicode (é, 雪, an astral emoji)
  and colons included, and it parses back with words.parse(); each group holds one Warning, whose Show output gates
  every output, and the String node 'OpenAct uid'; a word whose socket a dropdown removed reads "", before the
  rebuild and after;
- the Trigger: unlinked 0, a Boolean 1, and Blender's implicit conversion from a Float (-1 gives 0, 0.3 gives 1), an
  Int, a Vector and a Color;
- a word the static evaluator can't read (Scene Time's frame through Value to String) is reported at each frame;
- words and a Trigger linked from other nodes are reported whichever output is used, the Geometry or the Answer; a
  node whose outputs are unused reports nothing, nor does a muted one; a user group used three times, twice with
  the same words, gives two messages; a For Each Element zone gives one message per element;
- a copy reports its own uid: update_values() rewrites the one ensure_owned() copied, values only, and the pass
  does it for a copy it finds; rebuilds keep exactly one Warning node.

Run:  blender -b --factory-startup --python tests/channel_blender.py
"""

import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import check, finish, sock, store  # noqa: E402
from openact import engine, evaluator, inner, nodes, words  # noqa: E402  (bpy must be imported first)
from openact import prefs as P  # noqa: E402
from openact import store as S  # noqa: E402

P._Defaults.auto_start = False  # nothing here may start a model server
nodes.register()
ASK = engine.ASK_ID
TEXT = "{label} {name}: café 雪 \U0001f327 at 12:30, 3:ab"  # every word of the six nodes: Unicode, colons, a length


def scene(name, points=1):
    """A tree used by one object of `points` loose vertices, which enter as Geometry and leave as Geometry."""
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


def enabled(sockets, name):
    return next(s for s in sockets if s.name == name and s.enabled)


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


def messages(obj, mod):
    """(type, message) of each warning the modifier lists once the object is evaluated."""
    obj.update_tag()
    bpy.context.view_layer.update()
    bpy.context.evaluated_depsgraph_get()
    return [(warning.type, warning.message) for warning in mod.node_warnings]


def reports(obj, mod, node=None):
    """The Reports among the modifier's messages (words.parse() accepts them), of the node's uid when given."""
    found = [words.parse(text) for _kind, text in messages(obj, mod)]
    return [report for report in found if report is not None and (node is None or report.uid == node.uid)]


def gated(socket, warning):
    """True when the Group Output socket is fed by a Switch whose condition is the Warning's Show output."""
    switch = socket.links[0].from_node if socket.is_linked else None
    condition = sock(switch.inputs, "Switch").links if switch and switch.bl_idname == "GeometryNodeSwitch" else ()
    return bool(condition) and condition[0].from_node == warning


def channel(node):
    """(Warning nodes, the first one's type, every output read through a Switch on its Show, the uid node holding the
    node's uid)."""
    group = node.node_tree
    found = [n for n in group.nodes if n.bl_idname == "GeometryNodeWarning"]
    out = next(n for n in group.nodes if n.bl_idname == "NodeGroupOutput")
    ident = group.nodes.get(inner.UID)
    return (len(found), found[0].warning_type if found else None,
            bool(found) and all(gated(s, found[0]) for s in out.inputs if s.type in inner.GATES),
            ident is not None and ident.bl_idname == "FunctionNodeInputString" and ident.string == node.uid)


def phase_six():
    tree, gi, go, obj, mod = scene("Six")
    made = [tree.nodes.new(idname) for idname in engine.SPECS]
    collected = tree.nodes.new(engine.PICK_ID)
    for node in (*made, collected):
        for name in engine.SPECS[node.bl_idname].words(node):
            node.inputs[name].default_value = TEXT.format(label=node.bl_label, name=name)
    along(tree, gi.outputs["Geometry"], (*made, collected), go.inputs["Geometry"])
    ctx = evaluator.Ctx(obj, mod, (), tree)
    listed = messages(obj, mod)
    expected = sorted(words.message(node.uid, False, engine.SPECS[node.bl_idname].gn(node, ctx))
                      for node in (*made, collected))
    check("six nodes: each message is words.message() of the node's uid, Trigger 0 and the words the static evaluator "
          "reads, so the key text Geometry Nodes builds is Python's, Unicode (é, 雪, an astral emoji) and colons "
          "included", sorted(text for _kind, text in listed) == expected, (listed, expected))
    got = {report.uid: report for report in reports(obj, mod)}
    check("six nodes: each message parses back to the node's uid, its Trigger off and its words (spec.gn())",
          len(got) == 7 and all(got.get(node.uid) == words.Report(node.uid, False, engine.SPECS[node.bl_idname].gn(
              node, ctx)) for node in (*made, collected)), got)
    check("six nodes: every message is an info marker", listed and all(kind == 'INFO' for kind, _text in listed),
          listed)
    faces = {node.bl_label: channel(node) for node in made}
    check("six nodes: each group holds one INFO Warning, whose Show output gates every output, and the String node "
          "'OpenAct uid' holding the node's uid", all(face == (1, 'INFO', True, True) for face in faces.values()),
          faces)

    collected.list_source = 'COLLECTION'  # the dropdown removes Items at once; the group is rebuilt by the next pass
    before = [report.words for report in reports(obj, mod, collected)]
    inner.build(collected, S.load(collected))
    after = [report.words for report in reports(obj, mod, collected)]
    gn = engine.SPECS[engine.PICK_ID].gn(collected, ctx)
    check("dropdown: a word whose socket the node lost (Items under Collection) reads \"\", as spec.gn() reads it, "
          "before the rebuild and after", gn[2] == "" and before == after == [gn], (before, after, gn))


def phase_trigger():
    tree, gi, go, obj, mod = scene("Trigger")
    ask = tree.nodes.new(ASK)
    ask.inputs["Question"].default_value = "Is it raining?"
    along(tree, gi.outputs["Geometry"], [ask], go.inputs["Geometry"])

    def state():
        return [report.triggered for report in reports(obj, mod, ask)]

    got = [("unlinked", state(), [False])]
    boolean = tree.nodes.new("FunctionNodeInputBool")
    tree.links.new(boolean.outputs[0], ask.inputs["Trigger"])
    for value in (True, False):
        boolean.boolean = value
        got.append((f"Boolean {value}", state(), [value]))
    value = tree.nodes.new("ShaderNodeValue")
    tree.links.new(value.outputs[0], ask.inputs["Trigger"])
    for number, on in ((-1.0, False), (0.0, False), (0.3, True)):
        value.outputs[0].default_value = number
        got.append((f"Float {number}", state(), [on]))
    integer = tree.nodes.new("FunctionNodeInputInt")
    tree.links.new(integer.outputs[0], ask.inputs["Trigger"])
    for number, on in ((-2, False), (0, False), (3, True)):
        integer.integer = number
        got.append((f"Int {number}", state(), [on]))
    vector = tree.nodes.new("FunctionNodeInputVector")
    tree.links.new(vector.outputs[0], ask.inputs["Trigger"])
    for xyz, on in (((0.0, 0.0, 0.0), False), ((0.0, -0.5, 0.0), True)):
        vector.vector = xyz
        got.append((f"Vector {xyz}", state(), [on]))
    color = tree.nodes.new("FunctionNodeInputColor")
    tree.links.new(color.outputs[0], ask.inputs["Trigger"])
    for rgba, on in (((0.0, 0.0, 0.0, 1.0), False), ((0.2, 0.1, 0.0, 0.0), True)):
        color.value = rgba
        got.append((f"Color {rgba}", state(), [on]))
    check("Trigger: unlinked reports 0; linked, Blender's conversion into the Bool socket decides: a Boolean as it is, "
          "a Float or Int above 0 (-1 and 0 give 0), a Vector with a non-zero component, a Color lighter than black "
          "whatever its alpha", all(seen == wanted for _case, seen, wanted in got), got)


def phase_frame():
    tree, gi, go, obj, mod = scene("Frame")
    ask = tree.nodes.new(ASK)
    whole = tree.nodes.new("FunctionNodeFloatToInt")
    tree.links.new(tree.nodes.new("GeometryNodeInputSceneTime").outputs["Frame"], whole.inputs[0])
    tree.links.new(numbered(tree, "frame ", whole.outputs[0]), ask.inputs["Question"])
    along(tree, gi.outputs["Geometry"], [ask], go.inputs["Geometry"])
    try:
        read = repr(engine.SPECS[ASK].gn(ask, evaluator.Ctx(obj, mod, (), tree)))
    except evaluator.Unresolvable as error:
        read = f"Unresolvable: {error}"
    got = []
    for frame in (7, 12):
        bpy.context.scene.frame_set(frame)
        got.append([report.words[0] for report in reports(obj, mod, ask)])
    bpy.context.scene.frame_set(1)
    check("frame: words the static evaluator can't read (Scene Time's frame through Value to String) are reported, "
          "at the current frame", read.startswith("Unresolvable") and got == [["frame 7"], ["frame 12"]], (read, got))


def phase_used():
    tree, gi, go, obj, mod = scene("Used")
    passing, answering, unused = (tree.nodes.new(ASK) for _ in range(3))
    for node, text in ((passing, "Do I pass the geometry on?"), (answering, "Is my Answer read?"),
                       (unused, "Am I unused?")):  # linked from another node: only a gated Warning evaluates it
        words_node = tree.nodes.new("FunctionNodeInputString")
        words_node.string = text
        tree.links.new(words_node.outputs[0], node.inputs["Question"])
    on = tree.nodes.new("FunctionNodeInputBool")
    on.boolean = True
    tree.links.new(on.outputs[0], answering.inputs["Trigger"])
    along(tree, store(tree, gi.outputs["Geometry"], answering.outputs["Answer"], "answer", 'FLOAT'), [passing],
          go.inputs["Geometry"])
    got = {report.uid: (report.words[0], report.triggered) for report in reports(obj, mod)}
    check("used: words and a Trigger linked from other nodes are reported, whether the node's Geometry or its Answer "
          "is used; a node whose outputs are unused is not evaluated, so it reports nothing",
          got == {passing.uid: ("Do I pass the geometry on?", False), answering.uid: ("Is my Answer read?", True)},
          got)
    passing.mute = True
    got = {report.uid for report in reports(obj, mod)}
    points = len(obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices)
    check("used: a muted node reports nothing, and still passes its geometry on",
          got == {answering.uid} and points == 1, (got, points))


def phase_groups():
    user = bpy.data.node_groups.new("Asker", 'GeometryNodeTree')
    user.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    user.interface.new_socket("Words", in_out='INPUT', socket_type='NodeSocketString')
    user.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    ugi, ugo = user.nodes.new("NodeGroupInput"), user.nodes.new("NodeGroupOutput")
    asker = user.nodes.new(ASK)
    user.links.new(ugi.outputs["Words"], asker.inputs["Question"])
    along(user, ugi.outputs["Geometry"], [asker], ugo.inputs["Geometry"])
    tree, gi, go, obj, mod = scene("Groups")
    groups = []
    for text in ("first words", "second words", "first words"):
        group = tree.nodes.new("GeometryNodeGroup")
        group.node_tree = user
        group.inputs["Words"].default_value = text
        groups.append(group)
    along(tree, gi.outputs["Geometry"], groups, go.inputs["Geometry"])
    got = sorted(report.words for report in reports(obj, mod, asker))
    check("groups: a user group holding the node, used three times, reports the words of each use; the two identical "
          "messages merge", got == [("first words", "", "", "", ""), ("second words", "", "", "", "")], got)


def phase_zone():
    tree, gi, go, obj, mod = scene("Zone", points=3)
    begin = tree.nodes.new("GeometryNodeForeachGeometryElementInput")
    end = tree.nodes.new("GeometryNodeForeachGeometryElementOutput")
    begin.pair_with_output(end)
    ask = tree.nodes.new(ASK)
    tree.links.new(numbered(tree, "item ", begin.outputs["Index"]), ask.inputs["Question"])
    tree.links.new(gi.outputs["Geometry"], begin.inputs["Geometry"])
    along(tree, begin.outputs["Element"], [ask], sock(end.inputs, "Generation_0"))
    tree.links.new(sock(end.outputs, "Generation_0"), go.inputs["Geometry"])
    got = sorted(report.words[0] for report in reports(obj, mod, ask))
    check("zone: a For Each Element zone over 3 points reports one message per element",
          got == ["item 0", "item 1", "item 2"], got)


def phase_copies():
    tree, gi, go, obj, mod = scene("Copies")
    original, first, second = (tree.nodes.new(ASK) for _ in range(3))
    for node, text in ((original, "Is it the original?"), (first, "Is it the first copy?"),
                       (second, "Is it the second copy?")):
        node.inputs["Question"].default_value = text
    for twin in (first, second):  # duplicated in the editor: copy() clears the uid, and the group is shared
        twin.node_tree, twin.uid, twin.entries = original.node_tree, "", original.entries
    along(tree, gi.outputs["Geometry"], [original, first, second], go.inputs["Geometry"])

    def uids():
        return {report.words[0]: report.uid for report in reports(obj, mod)}

    got = uids()
    check("copies: before the pass, the copies share the original's group, so they report its uid",
          got == dict.fromkeys(("Is it the original?", "Is it the first copy?", "Is it the second copy?"),
                               original.uid), got)
    engine.ensure_owned([(tree, original), (tree, first)])  # the pass does it for every node; here the first copy
    copied = first.node_tree.nodes[inner.UID].string
    before = [n.as_pointer() for n in first.node_tree.nodes]
    updated = inner.update_values(first, S.load(first))
    check("copies: ensure_owned() gives the copy its own uid and group, whose uid node still holds the original's; "
          "update_values() rewrites it, values only (the uid is no part of the layout)",
          first.uid not in ("", original.uid) and copied == original.uid and updated
          and [n.as_pointer() for n in first.node_tree.nodes] == before and channel(first)[3],
          (first.uid, copied, updated))
    engine.run_pass()
    got = uids()
    check("copies: after the pass, which found the second copy, each node reports its own uid",
          second.uid not in ("", original.uid, first.uid) and got == {
              "Is it the original?": original.uid, "Is it the first copy?": first.uid,
              "Is it the second copy?": second.uid}, got)

    inner.build(original, S.load(original))
    inner.build(original, S.load(original))
    engine.run_pass()
    check("rebuild: after two rebuilds and a pass the group holds exactly one Warning node, and the node reports "
          "one message", channel(original) == (1, 'INFO', True, True) and len(reports(obj, mod, original)) == 1,
          (channel(original), reports(obj, mod, original)))


phase_six()
phase_trigger()
phase_frame()
phase_used()
phase_groups()
phase_zone()
phase_copies()
finish()