"""Node interfaces, in a headless Blender. No model needed.

- inner.sync_interface(): interface edits reach the custom group node and its links survive; sockets
  are matched by (in_out, name, socket_type) and ordered with interface.move(), never cleared; the
  structure type of each input per version; a socket declared hidden is hidden when it is created, and
  stays visible once revealed.
- The verified facts the nodes rely on: default_input='INDEX' with hide_value gives [0, 1, 2, 3]; Index
  Switch MATERIAL works; a string field linked into a SINGLE word stays valid, so the evaluator is the
  guard.
- copy() plus ensure_owned() gives every copy its own inner group, also when the whole tree is duplicated.

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

P._Defaults.auto_start = False  # nothing here may start a model server
nodes.register()
FIELDS = bpy.app.version >= (5, 0, 0)

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
    spec = engine.SPECS[engine.ASK_ID].interface(ask)  # Question, Context, outputs, Answers, Overrides (+CSV), Key
    by = {d.name: d for d in spec}
    kept = {s.name: s.identifier for s in (*ask.inputs, *ask.outputs)}
    tag = SocketDef('INPUT', "Tag", 'NodeSocketString', "Compared, never sent to the model", kind="word", hidden=True)
    extra = SocketDef('OUTPUT', "Extra", 'NodeSocketInt', "An extra output")
    wanted = [by["Context"], by["Question"], by["Answers"], by["Answers CSV"], by["Overrides"], by["Overrides CSV"], tag,
              by["Key"], by["Confidence"], by["Answer"], by["Yes"], extra]  # Context first, Confidence first
    inputs = ("Question", "Context", "Answers", "Answers CSV", "Overrides", "Overrides CSV", "Key")
    inner.sync_interface(ask, wanted)
    check("sync: new sockets reach the node, in the wanted order",
          names(ask.inputs) == ["Context", "Question", "Answers", "Answers CSV", "Overrides", "Overrides CSV", "Tag",
                                "Key"]
          and names(ask.outputs) == ["Confidence", "Answer", "Yes", "Extra"], (names(ask.inputs), names(ask.outputs)))
    check("sync: matched sockets keep their identifiers (the interface is never cleared)",
          all(sock(ask.inputs if name in inputs else ask.outputs, identifier).name == name
              for name, identifier in kept.items()), kept)
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
          names(ask.inputs) == ["Question", "Context", "Answers", "Answers CSV", "Overrides", "Overrides CSV", "Key"]
          and names(ask.outputs) == ["Answer", "Yes", "Confidence"], (names(ask.inputs), names(ask.outputs)))
    check("sync: kept sockets and links are untouched by the removals",
          all(sock((*ask.inputs, *ask.outputs), identifier).name == name for name, identifier in kept.items())
          and question_link.is_valid and read(host, "answer") == [0.75] * 4)


def phase_structure_types():
    word, field = SocketDef('INPUT', "Word", 'NodeSocketString', kind="word"), SocketDef(
        'INPUT', "Factor", 'NodeSocketFloat', kind="field")
    value, collection = SocketDef('INPUT', "Threshold", 'NodeSocketFloat'), SocketDef(
        'INPUT', "Collection", 'NodeSocketCollection', kind="word")
    stand = N.new(engine.ASK_ID)
    inner.sync_interface(stand, [word, field, value, collection])
    got = {name: item.structure_type for (_in_out, name), item in items(stand).items()}
    check("structure types: SINGLE for String words, FIELD for fields only on 5.0+, AUTO otherwise",
          got == {"Word": 'SINGLE', "Factor": 'FIELD' if FIELDS else 'AUTO', "Threshold": 'AUTO', "Collection": 'AUTO'},
          got)
    check("structure types: sync never asks for FIELD on a String socket",
          compat.structure_type('NodeSocketString', "field") == 'AUTO')
    string = stand.node_tree.interface.new_socket("Probe", in_out='INPUT', socket_type='NodeSocketString')
    try:
        string.structure_type = 'FIELD'
        refused = False
    except TypeError:
        refused = True
    check("structure types: Blender refuses FIELD on a String input", refused, string.structure_type)

    # A string field: a String Switch whose condition is one per element (5.0+; on 4.5 that condition link is
    # already red, strings being no fields there).
    per_element = N.new("GeometryNodeSwitch")
    per_element.input_type = 'STRING'
    L.new(N.new("GeometryNodeInputIndex").outputs[0], sock(per_element.inputs, "Switch"))
    word_link = L.new(per_element.outputs[0], ask.inputs["Context"])
    bpy.context.view_layer.update()
    try:
        caught = f"no error, reads {evaluator.eval_input(ask.inputs['Context'], evaluator.Ctx(host, mod, (), tree))!r}"
        refused = False
    except evaluator.Unresolvable as error:
        caught, refused = str(error), True
    check("structure types: a string field linked into a SINGLE word stays valid; the evaluator refuses it",
          word_link.is_valid and refused and (caught.endswith("fields can't be words") or not FIELDS),
          (word_link.is_valid, caught))
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


phase_sync()
phase_structure_types()
phase_index_default()
phase_material_switch()
phase_copies()
finish()
