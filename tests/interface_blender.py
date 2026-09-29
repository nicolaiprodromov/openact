"""Node interfaces, in a headless Blender. No model needed.

- inner.sync_interface(): interface edits reach the custom group node and its links survive; sockets
  are matched by (in_out, name, socket_type) and ordered with interface.move(), never cleared; the
  structure type of each input per version; a socket declared hidden is hidden when it is created, and
  stays visible once revealed.
- The verified facts the nodes rely on: default_input='INDEX' with hide_value gives [0, 1, 2, 3]; Index
  Switch MATERIAL works; a string field (5.1+) linked into a SINGLE word stays valid, so the evaluator is
  the guard.
- copy() plus ensure_owned() gives every copy its own inner group, also when the whole tree is duplicated.
- Every node's first sockets (§6.0): Geometry and Trigger inputs, a Geometry output, through dropdowns and
  dynamic sockets; the Trigger a SINGLE Bool with no checkbox; the geometry passes through all six nodes. A
  node saved by 1.0, without them, gains them on the next pass, its other sockets and links kept.

Run:  blender -b --factory-startup --python tests/interface_blender.py
"""

import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import check, finish, sock, store  # noqa: E402
from openact import compat, engine, evaluator, inner, nodes  # noqa: E402  (bpy must be imported first)
from openact import prefs as P  # noqa: E402
from openact import store as S  # noqa: E402  (store() is the harness's attribute writer)
from openact.inner import SocketDef  # noqa: E402
from openact.specs.base import SHARED  # noqa: E402
from openact.specs.classify import ClassifySpec  # noqa: E402
from openact.specs.dials import DialsSpec  # noqa: E402
from openact.specs.rules import RulesSpec  # noqa: E402

P._Defaults.auto_start = False  # nothing here may start a model server
nodes.register()
FIELDS = bpy.app.version >= (5, 0, 0)
STRING_FIELDS = bpy.app.version >= (5, 1, 0)  # before 5.1 a per-element condition can't drive a String Switch
RULES, CLASSIFY, DIALS = RulesSpec.idname, ClassifySpec.idname, DialsSpec.idname
FIRST = [("Geometry", 'GEOMETRY'), ("Trigger", 'BOOLEAN'), ("Geometry", 'GEOMETRY')]  # every node's first sockets
SHARED_KEYS = {(in_out, name, socket_type) for in_out, name, socket_type, *_rest in SHARED}

tree = bpy.data.node_groups.new("Interface", 'GeometryNodeTree')
tree.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
tree.interface.new_socket("Question", in_out='INPUT', socket_type='NodeSocketString')
tree.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
N, L = tree.nodes, tree.links
g_in, g_out = N.new("NodeGroupInput"), N.new("NodeGroupOutput")
ask = N.new(engine.ASK_ID)
question_link = L.new(g_in.outputs["Question"], ask.inputs["Question"])
geometry = store(tree, g_in.outputs["Geometry"], ask.outputs["Answer"], "answer", 'FLOAT')
id_node = N.new(engine.ASK_ID)  # its interface becomes an ID input and one Int output
geometry = store(tree, geometry, id_node.outputs["Answer"], "placeholder", 'FLOAT')
L.new(geometry, g_out.inputs[0])
mesh = bpy.data.meshes.new("Points")
mesh.from_pydata([(i, 0, 0) for i in range(4)], [], [])
host = bpy.data.objects.new("Host", mesh)
bpy.context.scene.collection.objects.link(host)
mod = host.modifiers.new("Interface", 'NODES')
mod.node_group = tree
mod[tree.interface.items_tree["Question"].identifier] = "Is it night?"
S.save(ask, S.NodeStore([S.Entry(words_key="night", gn=("Is it night?", "", "", "", ""), kind="ask",
                                 answer_type='YES_NO', single={"Answer": 0.75, "Yes": True, "Confidence": 0.5})]))
inner.build(ask, S.load(ask))


def read(obj, name):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    evaluated = obj.evaluated_get(depsgraph)
    result = evaluated.to_mesh()
    try:
        attribute = result.attributes.get(name)
        return None if attribute is None else [round(item.value, 6) if isinstance(item.value, float) else item.value
                                               for item in attribute.data]
    finally:
        evaluated.to_mesh_clear()


def names(sockets):
    return [s.name for s in sockets if s.enabled]


def items(node):
    return {(i.in_out, i.name): i for i in node.node_tree.interface.items_tree if i.item_type == 'SOCKET'}


def phase_sync():
    spec = engine.SPECS[engine.ASK_ID].interface(ask)  # shared, Question, Context, outputs, Answers, Overrides, Key
    shared, by = spec[:3], {d.name: d for d in spec[3:]}  # Geometry in, Trigger, Geometry out (§6.0)
    kept = [(s.is_output, s.name, s.identifier) for s in (*ask.inputs, *ask.outputs)]
    tag = SocketDef('INPUT', "Tag", 'NodeSocketString', "Compared, never sent to the model", kind="word", hidden=True)
    extra = SocketDef('OUTPUT', "Extra", 'NodeSocketInt', "An extra output")
    wanted = [*shared, by["Context"], by["Question"], by["Answers"], by["Answers CSV"], by["Overrides"],
              by["Overrides CSV"], tag, by["Key"], by["Confidence"], by["Answer"], by["Yes"], extra]  # Context first
    inner.sync_interface(ask, wanted)
    check("sync: new sockets reach the node, in the wanted order",
          names(ask.inputs) == ["Geometry", "Trigger", "Context", "Question", "Answers", "Answers CSV", "Overrides",
                                "Overrides CSV", "Tag", "Key"]
          and names(ask.outputs) == ["Geometry", "Confidence", "Answer", "Yes", "Extra"],
          (names(ask.inputs), names(ask.outputs)))
    check("sync: matched sockets keep their identifiers (the interface is never cleared)",
          all(sock(ask.outputs if output else ask.inputs, identifier).name == name
              for output, name, identifier in kept), kept)
    check("sync: links to the node survive", question_link.is_valid and question_link.to_socket == ask.inputs["Question"]
          and all(link.is_valid for link in L), [(link.to_socket.name, link.is_valid) for link in L if not link.is_valid])
    check("sync: the node still evaluates through its links", read(host, "answer") == [0.75] * 4, read(host, "answer"))
    check("sync: tooltips come from the descriptions",
          items(ask)[('INPUT', "Tag")].description == tag.description
          and items(ask)[('OUTPUT', "Answer")].description == by["Answer"].description)
    check("sync: a socket declared hidden is hidden when it is created, like every node's Key",
          ask.inputs["Tag"].hide and ask.inputs["Key"].hide and not ask.inputs["Question"].hide)

    ask.inputs["Tag"].hide = False  # the artist reveals it with Ctrl H
    inner.sync_interface(ask, wanted)
    inner.build(ask, S.load(ask))
    check("sync: a revealed socket stays visible through later syncs and rebuilds", not ask.inputs["Tag"].hide)

    grid = N.new("GeometryNodeMeshGrid")
    L.new(ask.outputs["Extra"], grid.inputs["Vertices X"])
    retyped = SocketDef('OUTPUT', "Extra", 'NodeSocketFloat', "An extra output")
    identifier = ask.outputs["Extra"].identifier
    inner.sync_interface(ask, wanted[:-1] + [retyped])
    check("sync: a retyped socket is replaced, dropping its links",
          ask.outputs["Extra"].identifier != identifier and ask.outputs["Extra"].type == 'VALUE'
          and not grid.inputs["Vertices X"].is_linked, ask.outputs["Extra"].identifier)
    N.remove(grid)

    inner.sync_interface(ask, spec)
    check("sync: obsolete sockets are removed and the order restored",
          names(ask.inputs) == ["Geometry", "Trigger", "Question", "Context", "Answers", "Answers CSV", "Overrides",
                                "Overrides CSV", "Key"]
          and names(ask.outputs) == ["Geometry", "Answer", "Yes", "Confidence"],
          (names(ask.inputs), names(ask.outputs)))
    check("sync: kept sockets and links are untouched by the removals",
          all(sock(ask.outputs if output else ask.inputs, identifier).name == name for output, name, identifier in kept)
          and question_link.is_valid and read(host, "answer") == [0.75] * 4)


def phase_structure_types():
    word, field = SocketDef('INPUT', "Word", 'NodeSocketString', kind="word"), SocketDef(
        'INPUT', "Factor", 'NodeSocketFloat', kind="field")
    value, collection = SocketDef('INPUT', "Threshold", 'NodeSocketFloat'), SocketDef(
        'INPUT', "Collection", 'NodeSocketCollection', kind="word")
    single = SocketDef('INPUT', "Trigger", 'NodeSocketBool', kind="single", hide_value=True)
    stand = N.new(engine.ASK_ID)
    inner.sync_interface(stand, [word, field, value, collection, single])
    got = {name: item.structure_type for (_in_out, name), item in items(stand).items()}
    check("structure types: SINGLE for String words and single-value inputs (the Trigger) on every version, FIELD "
          "for fields only on 5.0+, AUTO otherwise",
          got == {"Word": 'SINGLE', "Factor": 'FIELD' if FIELDS else 'AUTO', "Threshold": 'AUTO', "Collection": 'AUTO',
                  "Trigger": 'SINGLE'}, got)
    check("structure types: sync never asks for FIELD on a String socket",
          compat.structure_type('NodeSocketString', "field") == 'AUTO')
    string = stand.node_tree.interface.new_socket("Probe", in_out='INPUT', socket_type='NodeSocketString')
    try:
        string.structure_type = 'FIELD'
        refused = False
    except TypeError:
        refused = True
    check("structure types: Blender refuses FIELD on a String input", refused, string.structure_type)

    # A string field: a String Switch whose condition is one per element (5.1+; on 4.5 and 5.0 that condition link
    # is red, strings being no fields there).
    per_element = N.new("GeometryNodeSwitch")
    per_element.input_type = 'STRING'
    condition = L.new(N.new("GeometryNodeInputIndex").outputs[0], sock(per_element.inputs, "Switch"))
    word_link = L.new(per_element.outputs[0], ask.inputs["Context"])
    bpy.context.view_layer.update()
    try:
        caught = f"no error, reads {evaluator.eval_input(ask.inputs['Context'], evaluator.Ctx(host, mod, (), tree))!r}"
        refused = False
    except evaluator.Unresolvable as error:
        caught, refused = str(error), True
    refusal = "fields can't be words" if STRING_FIELDS else "An input has an invalid (red) link"
    check("structure types: a per-element String Switch linked into a SINGLE word stays a valid link, and the "
          "evaluator refuses the word: a string field on 5.1+, a red condition link on 4.5 and 5.0",
          word_link.is_valid and condition.is_valid == STRING_FIELDS and refused and caught.endswith(refusal),
          (word_link.is_valid, condition.is_valid, caught))
    L.remove(word_link)


def phase_index_default():
    ident = SocketDef('INPUT', "ID", 'NodeSocketInt', kind="field", default_input='INDEX', hide_value=True)
    inner.sync_interface(id_node, [ident, SocketDef('OUTPUT', "Answer", 'NodeSocketFloat')])  # kept, with its link
    group = id_node.node_tree
    group.nodes.clear()
    group.links.new(group.nodes.new("NodeGroupInput").outputs["ID"], group.nodes.new("NodeGroupOutput").inputs["Answer"])
    item = items(id_node)[('INPUT', "ID")]
    check("ID: sync sets default_input INDEX and hide_value", (item.default_input, item.hide_value) == ('INDEX', True))
    check("ID: unlinked on the node, it is the element index [0, 1, 2, 3]", read(host, "placeholder") == [0, 1, 2, 3],
          read(host, "placeholder"))


def phase_material_switch():
    first, second = bpy.data.materials.new("First"), bpy.data.materials.new("Second")
    group = bpy.data.node_groups.new("Materials", 'GeometryNodeTree')
    group.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    out = group.nodes.new("NodeGroupOutput")
    grid = group.nodes.new("GeometryNodeMeshGrid")
    painted = group.nodes.new("GeometryNodeSetMaterial")
    group.links.new(grid.outputs["Mesh"], painted.inputs["Geometry"])
    index = group.nodes.new("FunctionNodeInputInt")
    index.integer = 2
    inner._Builder(group).lookup(index.outputs[0], 'MATERIAL', None, [first, second], painted.inputs["Material"], 0)
    chosen = group.nodes.new("GeometryNodeMaterialSelection")
    chosen.inputs["Material"].default_value = second
    stored = store(group, painted.outputs["Geometry"], chosen.outputs[0], "second", 'BOOLEAN')
    stored.node.domain = 'FACE'
    group.links.new(stored, out.inputs["Geometry"])
    obj = bpy.data.objects.new("Painted", bpy.data.meshes.new("Painted"))
    bpy.context.scene.collection.objects.link(obj)
    obj.modifiers.new("Materials", 'NODES').node_group = group
    faces = read(obj, "second")
    check("Index Switch MATERIAL: item 2 of [None, First, Second] paints Second", bool(faces) and all(faces), faces)


def phase_copies():
    original = (ask.uid, ask.node_tree.name, ask.node_tree.as_pointer())
    twin_tree = tree.copy()  # like making the modifier's node group single-user: copy() runs for every node
    third_tree = twin_tree.copy()
    twins = [next(n for n in t.nodes if n.bl_idname == engine.ASK_ID and n.name == ask.name)
             for t in (twin_tree, third_tree)]
    in_tree = N.new(engine.ASK_ID)  # a node duplicated in the editor: it shares the group, and copy() clears uid
    in_tree.node_tree, in_tree.uid, in_tree.entries = ask.node_tree, "", ask.entries
    shared = all(twin.uid == "" and twin.node_tree == ask.node_tree for twin in twins)
    engine.ensure_owned(list(engine.iter_nodes()))
    copies = twins + [in_tree]
    groups = [copy.node_tree for copy in copies]
    check("copies: copy() cleared the uids and the copies shared the original's group", shared)
    check("copies: the original keeps its uid and its group",
          (ask.uid, ask.node_tree.name, ask.node_tree.as_pointer()) == original, (ask.uid, ask.node_tree.name))
    check("copies: every copy has its own uid and its own group, named after it",
          len({copy.uid for copy in copies} | {ask.uid}) == 4 and all(copy.uid for copy in copies)
          and len({group.as_pointer() for group in groups} | {original[2]}) == 4
          and all(copy.node_tree.name == inner.inner_name(copy) for copy in copies),
          [(copy.uid, copy.node_tree.name) for copy in copies])
    check("copies: each copy starts with the same answers and graph",
          len(S.load(ask).entries) == 1 and all(copy.entries == ask.entries
                                                and len(copy.node_tree.nodes) == len(ask.node_tree.nodes)
                                                for copy in copies))
    engine.ensure_owned(list(engine.iter_nodes()))
    check("copies: a second pass changes nothing",
          [copy.node_tree for copy in copies] == groups and ask.node_tree.as_pointer() == original[2])


def first(node):
    """(name, type) of the node's first two inputs and its first output."""
    return [(s.name, s.type) for s in (node.inputs[0], node.inputs[1], node.outputs[0])]


def passes(node):
    """True when the node's inner group feeds its Geometry output from its Geometry input, through the Warning's
    gate (§5.7)."""
    return any(name == "Geometry" and source.node.bl_idname == "NodeGroupInput" and source.name == "Geometry"
               for name, source in inner._feeds(node.node_tree))


def passing_tree(name):
    """A tree whose modifier geometry, a copy of the host's 4 points, enters as Geometry and leaves as Geometry."""
    group = bpy.data.node_groups.new(name, 'GeometryNodeTree')
    group.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    group.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    obj = bpy.data.objects.new(name, mesh.copy())
    bpy.context.scene.collection.objects.link(obj)
    obj.modifiers.new(name, 'NODES').node_group = group
    return group, group.nodes.new("NodeGroupInput"), group.nodes.new("NodeGroupOutput"), obj


def phase_shared():
    """§6.0: every node starts with a Geometry and a Trigger input and a Geometry output, whatever its dropdowns and
    dynamic sockets, and its inner group passes the geometry through unchanged."""
    group, gi, go, obj = passing_tree("Shared")
    geometry = store(group, gi.outputs["Geometry"], group.nodes.new("GeometryNodeInputIndex").outputs[0], "tag", 'INT')
    made = {}
    for idname in engine.SPECS:  # the six nodes in a row, each passing on what the one before passed
        node = made[idname] = group.nodes.new(idname)
        group.links.new(geometry, node.inputs["Geometry"])
        geometry = node.outputs["Geometry"]
    group.links.new(geometry, go.inputs["Geometry"])
    check("shared: Geometry and Trigger are the first inputs, Geometry the first output, on all six nodes",
          all(first(node) == FIRST for node in made.values()), {node.bl_label: first(node) for node in made.values()})
    trigger = {node.bl_label: items(node)[('INPUT', "Trigger")] for node in made.values()}
    check("shared: the Trigger is a single value on every version, with no checkbox, and off while unlinked",
          all((i.structure_type, i.hide_value, i.default_value) == ('SINGLE', True, False) for i in trigger.values())
          and not any(node.inputs["Trigger"].default_value for node in made.values()),
          {label: (i.structure_type, i.hide_value, i.default_value) for label, i in trigger.items()})
    check("shared: their tooltips",
          all(items(node)[(in_out, name)].description == text for node in made.values()
              for in_out, name, _type, text, *_rest in SHARED))
    check("shared: every inner group links its Geometry input to its Geometry output",
          all(passes(node) for node in made.values()), [node.bl_label for node in made.values() if not passes(node)])
    check("shared: the six nodes pass the geometry through unchanged, attributes included",
          read(obj, "tag") == [0, 1, 2, 3], read(obj, "tag"))

    rules, classify, dials = made[RULES], made[CLASSIFY], made[DIALS]
    rules.answer_type = 'YES_NO'  # dropdowns swap sockets at once (nodes._interface_changed)
    made[engine.PICK_ID].list_source = 'COLLECTION'  # Instance: another Geometry output
    inner.sync_interface(rules, engine.SPECS[RULES].interface(rules, [("heat", "amount"), ("wet", "flag")], []))
    inner.sync_interface(classify, engine.SPECS[CLASSIFY].interface(classify, [("roughness", 'FLOAT')]))
    inner.sync_interface(dials, engine.SPECS[DIALS].interface(dials, [("Rain", "yes_no"), ("Wind", "amount")]))
    check("shared: they stay first through dropdowns, factors, payload and settings",
          all(first(node) == FIRST for node in made.values()), {node.bl_label: first(node) for node in made.values()})
    got = ([d.name for d in engine.SPECS[RULES].interface(rules) if d.kind == "field"],
           engine.SPECS[CLASSIFY].payload_outputs(classify), engine.SPECS[DIALS].outputs(dials))
    check("shared: the Trigger is no flag factor, and the Geometry output no payload or setting",
          got == (["heat", "wet"], [("roughness", 'FLOAT')], [("Rain", "yes_no"), ("Wind", "amount")]), got)
    check("shared: the geometry still passes through", read(obj, "tag") == [0, 1, 2, 3], read(obj, "tag"))


def phase_saved_before():
    """A node saved by OpenAct 1.0 lacks the shared sockets: the next pass adds them in their places and rebuilds the
    group, which passes the geometry through; every other socket keeps its identifier and its links, the factor,
    payload and setting sockets included; a second pass changes nothing."""
    group, gi, go, _obj = passing_tree("Before")
    olds = [group.nodes.new(idname) for idname in (engine.ASK_ID, RULES, CLASSIFY, DIALS)]
    asker, rules, classify, dials = olds
    for node, wanted in ((asker, engine.SPECS[engine.ASK_ID].interface(asker)),
                         (rules, engine.SPECS[RULES].interface(rules, [("heat", "amount")], [])),
                         (classify, engine.SPECS[CLASSIFY].interface(classify, [("roughness", 'FLOAT')])),
                         (dials, engine.SPECS[DIALS].interface(dials, [("Rain", "yes_no")]))):
        inner.sync_interface(node, [d for d in wanted if (d.in_out, d.name, d.socket_type) not in SHARED_KEYS])
        inner.build(node, S.load(node))
    words, heat = group.nodes.new("FunctionNodeInputString"), group.nodes.new("ShaderNodeValue")
    words.string = "Is it night?"
    group.links.new(words.outputs[0], asker.inputs["Question"])
    group.links.new(heat.outputs[0], rules.inputs["heat"])
    geometry = gi.outputs["Geometry"]
    for node, output, data_type in ((asker, "Answer", 'FLOAT'), (rules, "Outcome", 'INT'),
                                    (classify, "roughness", 'FLOAT'), (dials, "Rain", 'BOOLEAN')):
        geometry = store(group, geometry, node.outputs[output], output, data_type)
    group.links.new(geometry, go.inputs["Geometry"])

    def faces():
        return {node.name: [(s.is_output, s.name, s.identifier) for s in (*node.inputs, *node.outputs)]
                for node in olds}

    def linked():
        return sorted((link.from_node.name, link.from_socket.identifier, link.to_node.name, link.to_socket.identifier,
                       link.is_valid) for link in group.links if link.from_node in olds or link.to_node in olds)

    before, links = faces(), linked()
    check("saved before: the 1.0 nodes lack the shared sockets, and their groups pass nothing",
          all(first(node) != FIRST and not passes(node) for node in olds), {n.name: first(n) for n in olds})
    engine.run_pass()
    check("saved before: the next pass gives each node Geometry and Trigger first, and Geometry first among outputs",
          all(first(node) == FIRST for node in olds), {node.name: first(node) for node in olds})
    after = {name: [face for face in sockets if (face[0], face[1]) not in (
        (False, "Geometry"), (False, "Trigger"), (True, "Geometry"))] for name, sockets in faces().items()}
    check("saved before: every other socket keeps its identifier and its order, factor, payload and setting included",
          after == before, (before, after))
    check("saved before: every link to and from the nodes survives", linked() == links, (links, linked()))
    check("saved before: the pass rebuilt each group, which now passes the geometry through",
          all(passes(node) for node in olds), [node.name for node in olds if not passes(node)])
    settled = faces()
    engine.run_pass()
    check("saved before: a second pass changes nothing", faces() == settled and linked() == links)


phase_sync()
phase_structure_types()
phase_index_default()
phase_material_switch()
phase_copies()
phase_shared()
phase_saved_before()
finish()
