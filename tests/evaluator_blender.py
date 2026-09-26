"""Checks that the add-on's input reader computes the same strings as Geometry Nodes.

No model needed. Each case wires a string into a Compare node, sets B to the reader's result,
and lets Geometry Nodes say whether they are equal.

Run:  blender -b --factory-startup --python tests/evaluator_blender.py
"""

import os
import sys
import tempfile

import bpy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openact import engine  # noqa: E402  (bpy must be imported first)

FAILURES = []
TMP = tempfile.mkdtemp(prefix="openact_eval_")


def sock(sockets, identifier):
    return next(s for s in sockets if s.identifier == identifier)


def value_input(node):
    return next(s for s in node.inputs if s.name == "Value" and s.enabled)


tree = bpy.data.node_groups.new("Eval", 'GeometryNodeTree')
tree.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
tree.interface.new_socket("Word", in_out='INPUT', socket_type='NodeSocketString')
tree.interface.new_socket("Flag", in_out='INPUT', socket_type='NodeSocketBool')
tree.interface.new_socket("Pick", in_out='INPUT', socket_type='NodeSocketInt')
tree.interface.new_socket("Amount", in_out='INPUT', socket_type='NodeSocketFloat')
tree.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
N, L = tree.nodes, tree.links
g_in, g_out = N.new("NodeGroupInput"), N.new("NodeGroupOutput")
mesh = bpy.data.meshes.new("Host")
mesh.from_pydata([(0, 0, 0)], [], [])
host = bpy.data.objects.new("Host", mesh)
bpy.context.scene.collection.objects.link(host)
mod = host.modifiers.new("Eval", 'NODES')
mod.node_group = tree
ids = {i.name: i.identifier for i in tree.interface.items_tree if getattr(i, "in_out", "") == 'INPUT'}
mod[ids["Word"]] = "héllo wörld"
mod[ids["Flag"]] = True
mod[ids["Pick"]] = 2
mod[ids["Amount"]] = 0.1 + 0.2
ctx = engine.Ctx(host, mod, (), tree)


def string(text):
    node = N.new("FunctionNodeInputString")
    node.string = text
    return node.outputs[0]


def join(parts, delimiter="-"):
    node = N.new("GeometryNodeStringJoin")
    node.inputs["Delimiter"].default_value = delimiter
    for part in parts:  # the last one linked comes first in the result
        L.new(part, node.inputs["Strings"])
    return node


def group_passthrough(inner_source=None):
    """A group whose output string is its input with ' +g' appended."""
    sub = bpy.data.node_groups.new("Sub", 'GeometryNodeTree')
    sub.interface.new_socket("S", in_out='INPUT', socket_type='NodeSocketString')
    sub.interface.new_socket("S", in_out='OUTPUT', socket_type='NodeSocketString')
    si, so = sub.nodes.new("NodeGroupInput"), sub.nodes.new("NodeGroupOutput")
    j = sub.nodes.new("GeometryNodeStringJoin")
    j.inputs["Delimiter"].default_value = " +"
    tail = sub.nodes.new("FunctionNodeInputString")
    tail.string = "g"
    sub.links.new(tail.outputs[0], j.inputs["Strings"])
    sub.links.new(si.outputs[0], j.inputs["Strings"])
    sub.links.new(j.outputs[0], so.inputs[0])
    return sub


cases = []  # (name, socket carrying the string)

cases.append(("join order + delimiter", join([string("a"), string("b"), string("c")]).outputs[0]))
j = join([string("x"), string("y"), string("z")])
j.inputs["Strings"].links[1].is_muted = True
cases.append(("join with a muted link", j.outputs[0]))
j = join([string("m1"), string("m2")])
j.mute = True
cases.append(("muted join passes its first link", j.outputs[0]))
v = N.new("ShaderNodeValue")
v.outputs[0].default_value = 3.75
v.mute = True
vs = N.new("FunctionNodeValueToString")
L.new(v.outputs[0], vs.inputs["Value"])
vs.inputs["Decimals"].default_value = 2
cases.append(("muted Value gives 0", vs.outputs[0]))
vs2 = N.new("FunctionNodeValueToString")
L.new(g_in.outputs["Amount"], vs2.inputs["Value"])
vs2.inputs["Decimals"].default_value = 6
cases.append(("modifier float to string", vs2.outputs[0]))
rep = N.new("FunctionNodeReplaceString")
L.new(g_in.outputs["Word"], rep.inputs["String"])
rep.inputs["Find"].default_value = "ö"
rep.inputs["Replace"].default_value = "oe"
sl = N.new("FunctionNodeSliceString")
L.new(rep.outputs[0], sl.inputs["String"])
sl.inputs["Position"].default_value = 1
sl.inputs["Length"].default_value = 7
cases.append(("replace + unicode slice from modifier", sl.outputs[0]))
sw = N.new("GeometryNodeSwitch")
sw.input_type = 'STRING'
L.new(g_in.outputs["Flag"], sock(sw.inputs, "Switch"))
L.new(string("off"), sock(sw.inputs, "False"))
L.new(string("on"), sock(sw.inputs, "True"))
cases.append(("switch from modifier bool", sw.outputs[0]))
isw = N.new("GeometryNodeIndexSwitch")
isw.data_type = 'STRING'
isw.index_switch_items.new()
slots = [s for s in isw.inputs if s.identifier not in ("Index", "__extend__")]
for slot, text in zip(slots, ("zero", "one", "two")):
    slot.default_value = text
L.new(g_in.outputs["Pick"], sock(isw.inputs, "Index"))
cases.append(("index switch from modifier int", isw.outputs[0]))
reroute = N.new("NodeReroute")
L.new(g_in.outputs["Word"], reroute.inputs[0])
reroute2 = N.new("NodeReroute")
L.new(reroute.outputs[0], reroute2.inputs[0])
cases.append(("reroute chain", reroute2.outputs[0]))
sub = group_passthrough()
outer = bpy.data.node_groups.new("Outer", 'GeometryNodeTree')  # nested two levels
outer.interface.new_socket("S", in_out='INPUT', socket_type='NodeSocketString')
outer.interface.new_socket("S", in_out='OUTPUT', socket_type='NodeSocketString')
oi, oo = outer.nodes.new("NodeGroupInput"), outer.nodes.new("NodeGroupOutput")
og = outer.nodes.new("GeometryNodeGroup")
og.node_tree = sub
outer.links.new(oi.outputs[0], og.inputs[0])
outer.links.new(og.outputs[0], oo.inputs[0])
top = N.new("GeometryNodeGroup")
top.node_tree = outer
L.new(g_in.outputs["Word"], top.inputs[0])
cases.append(("nested groups (2 levels) from modifier", top.outputs[0]))
muted_group = N.new("GeometryNodeGroup")
muted_group.node_tree = sub
L.new(string("through"), muted_group.inputs[0])
muted_group.mute = True
cases.append(("muted group passes its input", muted_group.outputs[0]))
if hasattr(bpy.types, "GeometryNodeImportText"):
    for label, data in (("crlf", b"table\r\nlamp\r\n"), ("bom", b"\xef\xbb\xbfhi"), ("invalid", b"ok\xffok"),
                        ("lone cr", b"a\rb")):
        path = os.path.join(TMP, label.replace(" ", "_") + ".txt")
        with open(path, "wb") as handle:
            handle.write(data)
        it = N.new("GeometryNodeImportText")
        it.inputs["Path"].default_value = path
        cases.append((f"import text ({label})", it.outputs[0]))

# evaluate our side, wire the comparisons
geometry = g_in.outputs["Geometry"]
expected = []
for index, (name, source) in enumerate(cases):
    compare = N.new("FunctionNodeCompare")
    compare.data_type = 'STRING'
    compare.operation = 'EQUAL'
    L.new(source, sock(compare.inputs, "A_STR"))
    try:
        ours = engine.eval_input(sock(compare.inputs, "A_STR"), ctx)
    except Exception as error:  # noqa: BLE001
        ours = f"<{type(error).__name__}: {error}>"
    sock(compare.inputs, "B_STR").default_value = ours if isinstance(ours, str) else str(ours)
    expected.append((name, ours))
    store = N.new("GeometryNodeStoreNamedAttribute")
    store.data_type = 'BOOLEAN'
    store.domain = 'POINT'
    store.inputs["Name"].default_value = f"case{index}"
    L.new(geometry, store.inputs["Geometry"])
    L.new(compare.outputs[0], value_input(store))
    geometry = store.outputs[0]
L.new(geometry, g_out.inputs[0])

depsgraph = bpy.context.evaluated_depsgraph_get()
depsgraph.update()
evaluated = host.evaluated_get(depsgraph)
result = evaluated.to_mesh()
for index, (name, ours) in enumerate(expected):
    same = result.attributes[f"case{index}"].data[0].value
    print(("PASS " if same else "FAIL ") + f"{name}: reader gives {ours!r}", flush=True)
    if not same:
        FAILURES.append(name)
evaluated.to_mesh_clear()

# a red (invalid) link must be reported, not silently read
bad = N.new("GeometryNodeStringJoin")
L.new(N.new("ShaderNodeValue").outputs[0], bad.inputs["Strings"])  # float into string: invalid
try:
    engine.eval_input(bad.inputs["Strings"], ctx)
    print("FAIL invalid link is reported: no error", flush=True)
    FAILURES.append("invalid link")
except engine.Unresolvable as error:
    print(f"PASS invalid link is reported: {error}", flush=True)

print(f"BLENDER {bpy.app.version_string} RESULT {'OK' if not FAILURES else 'FAILED: ' + ', '.join(FAILURES)}", flush=True)
sys.stdout.flush()
os._exit(0 if not FAILURES else 1)
