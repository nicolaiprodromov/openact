"""Checks that the add-on's input reader computes the same values as Geometry Nodes.

No model needed. Each case wires a value into a Compare node, sets B to the reader's result,
and lets Geometry Nodes say whether they are equal. Then the per-pass memo must give the same
values, errors must name the socket and nodes they come through, and a pass must not write to
linked data.

Run:  blender -b --factory-startup --python tests/evaluator_blender.py
"""

import os
import sys
import tempfile

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import check, finish, sock, store  # noqa: E402
from openact import engine, evaluator, inner, nodes  # noqa: E402  (bpy must be imported first)
from openact import prefs as P  # noqa: E402
from openact import store as S  # noqa: E402  (store() is the harness's attribute writer)

TMP = tempfile.mkdtemp(prefix="openact_eval_")
P._Defaults.auto_start = False  # the passes below must never start a model server
nodes.register()  # OpenAct nodes: their outputs as words, and the linked-library check


tree = bpy.data.node_groups.new("Eval", 'GeometryNodeTree')
tree.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
tree.interface.new_socket("Word", in_out='INPUT', socket_type='NodeSocketString')
tree.interface.new_socket("Flag", in_out='INPUT', socket_type='NodeSocketBool')
tree.interface.new_socket("Pick", in_out='INPUT', socket_type='NodeSocketInt')
tree.interface.new_socket("Amount", in_out='INPUT', socket_type='NodeSocketFloat')
for menu in ("Mode", "Level", "Scale", "Nested", "Edited", "Fresh"):
    tree.interface.new_socket(menu, in_out='INPUT', socket_type='NodeSocketMenu')
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
ctx = evaluator.Ctx(host, mod, (), tree)


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


def menu_slots(node):
    return [s for s in node.inputs if s.identifier not in ("Menu", "__extend__")]


def menu_switch(data_type, values, tree_nodes=N):
    """A Menu Switch whose items A, B, C... (identifiers 0, 1, 2...) hold the values."""
    node = tree_nodes.new("GeometryNodeMenuSwitch")
    node.data_type = data_type
    for name in "CDEF"[:len(values) - 2]:
        node.enum_items.new(name)
    for slot, value in zip(menu_slots(node), values):
        slot.default_value = value
    return node


def read(socket, context=ctx):
    try:
        return evaluator.eval_input(socket, context)
    except Exception as error:  # noqa: BLE001
        return f"<{type(error).__name__}: {error}>"


cases = []  # (name, socket carrying the value[, Compare data type, STRING when left out])

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
for label, data in (("crlf", b"table\r\nlamp\r\n"), ("bom", b"\xef\xbb\xbfhi"), ("invalid", b"ok\xffok"),
                    ("lone cr", b"a\rb")):
    path = os.path.join(TMP, label.replace(" ", "_") + ".txt")
    with open(path, "wb") as handle:
        handle.write(data)
    it = N.new("GeometryNodeImportText")
    it.inputs["Path"].default_value = path
    cases.append((f"import text ({label})", it.outputs[0]))

# Menu Switch: a value typed on the socket is the item's name; a modifier value is the item's identifier
for data_type, values, item in (('STRING', ("apple", "banana", "cherry"), 'B'), ('INT', (10, 20, 30), 'C'),
                                ('FLOAT', (0.25, 0.5, 0.75), 'A')):
    ms = menu_switch(data_type, values)
    ms.inputs["Menu"].default_value = item
    cases.append((f"menu switch ({data_type}): socket value by name", ms.outputs[0], data_type))
for menu, data_type, values, choice in (("Mode", 'STRING', ("one", "two", "three"), 2),
                                        ("Level", 'INT', (7, 8, 9), 1), ("Scale", 'FLOAT', (1.5, 2.5, 3.5), 2)):
    ms = menu_switch(data_type, values)
    L.new(g_in.outputs[menu], ms.inputs["Menu"])
    mod[ids[menu]] = choice
    cases.append((f"menu switch ({data_type}): modifier int through id_properties_ui", ms.outputs[0], data_type))
edited = menu_switch('INT', (10, 20, 30, 40))
L.new(g_in.outputs["Edited"], edited.inputs["Menu"])
mod[ids["Edited"]] = 2  # C
by_name = menu_switch('STRING', ("a", "b", "c", "d"))
by_name.inputs["Menu"].default_value = 'D'
for node in (edited, by_name):  # delete A, then move B to the end: C, D, B
    node.enum_items.remove(node.enum_items[0])
    node.enum_items.move(0, 2)
cases.append(("menu switch: modifier int after deleted and reordered items", edited.outputs[0], 'INT'))
cases.append(("menu switch: socket name after deleted and reordered items", by_name.outputs[0]))
per_item = [o for ms in (edited, by_name) for o in ms.outputs if o.identifier != "Output"]
check("menu switch: a Boolean output per item on 5.0+ only", bool(per_item) == (bpy.app.version >= (5, 0, 0)),
      [o.identifier for o in per_item])
for flag in per_item:  # true for the chosen item only
    chosen = N.new("GeometryNodeSwitch")
    chosen.input_type = 'STRING'
    L.new(flag, sock(chosen.inputs, "Switch"))
    sock(chosen.inputs, "False").default_value = "not chosen"
    sock(chosen.inputs, "True").default_value = "chosen"
    cases.append((f"menu switch: per-item output '{flag.name}' of {flag.node.name} as a Switch condition",
                  chosen.outputs[0]))
menu_sub = bpy.data.node_groups.new("MenuSub", 'GeometryNodeTree')
menu_sub.interface.new_socket("Mode", in_out='INPUT', socket_type='NodeSocketMenu')
menu_sub.interface.new_socket("S", in_out='OUTPUT', socket_type='NodeSocketString')
mi, mo = menu_sub.nodes.new("NodeGroupInput"), menu_sub.nodes.new("NodeGroupOutput")
nested_switch = menu_switch('STRING', ("nested a", "nested b", "nested c"), menu_sub.nodes)
menu_sub.links.new(mi.outputs[0], nested_switch.inputs["Menu"])
menu_sub.links.new(nested_switch.outputs[0], mo.inputs[0])
typed = N.new("GeometryNodeGroup")
typed.node_tree = menu_sub
typed.inputs["Mode"].default_value = 'C'
cases.append(("menu inside a nested group: typed on the group node", typed.outputs[0]))
exposed = N.new("GeometryNodeGroup")
exposed.node_tree = menu_sub
L.new(g_in.outputs["Nested"], exposed.inputs["Mode"])
mod[ids["Nested"]] = 1  # B
cases.append(("menu inside a nested group: from the modifier", exposed.outputs[0]))
fresh = menu_switch('STRING', ("x", "y", "z"))
L.new(g_in.outputs["Fresh"], fresh.inputs["Menu"])
next(i for i in tree.interface.items_tree if getattr(i, "name", "") == "Fresh").default_value = 'B'  # mod keeps -1

special = N.new("FunctionNodeInputSpecialCharacters")
cases.append(("special characters: line break", special.outputs["Line Break"]))
cases.append(("special characters: tab", special.outputs["Tab"]))
lines = join([string("table"), string("lamp")], delimiter="")
L.new(special.outputs["Line Break"], lines.inputs["Delimiter"])
cases.append(("special characters: multi-line items with Join Strings", lines.outputs[0]))
for label, source in (("a unicode word from the modifier", g_in.outputs["Word"]),
                      ("code points", string("\U0001f600e\u0301 \u200d"))):
    length = N.new("FunctionNodeStringLength")
    L.new(source, length.inputs["String"])
    cases.append((f"string length: {label}", length.outputs[0], 'INT'))
by_length = N.new("GeometryNodeIndexSwitch")
by_length.data_type = 'STRING'
by_length.index_switch_items.new()
for slot, text in zip([s for s in by_length.inputs if s.identifier not in ("Index", "__extend__")],
                      ("zero", "one", "two")):
    slot.default_value = text
length = N.new("FunctionNodeStringLength")
L.new(string("ab"), length.inputs["String"])
L.new(length.outputs[0], sock(by_length.inputs, "Index"))
cases.append(("string length as an index switch index", by_length.outputs[0]))

answered = N.new(engine.ASK_ID)  # an OpenAct single-value output used as a word
answered.inputs["Question"].default_value = "Is it night?"
job = engine.SPECS[engine.ASK_ID].plan(answered, ctx)
saved = S.NodeStore([S.Entry(words_key=job.words_key, gn=job.gn, kind="ask", answer_type='YES_NO',
                             single={"Answer": 0.8125, "Yes": True, "Confidence": 0.625})])
S.save(answered, saved)
inner.build(answered, saved)
cases.append(("OpenAct Ask answer used as a word", answered.outputs["Answer"], 'FLOAT'))

field_ask = N.new(engine.ASK_ID)  # its Answer made a field, as a per-element output would be
field_inner = field_ask.node_tree
field_inner.links.new(field_inner.nodes.new("GeometryNodeInputIndex").outputs[0],
                      next(n for n in field_inner.nodes if n.bl_idname == "NodeGroupOutput").inputs["Answer"])

# evaluate our side, wire the comparisons
COMPARE = {'STRING': ("A_STR", "B_STR"), 'INT': ("A_INT", "B_INT"), 'FLOAT': ("A", "B")}
geometry = g_in.outputs["Geometry"]
expected, words = [], []
for index, (name, source, *kind) in enumerate(cases):
    data_type = kind[0] if kind else 'STRING'
    compare = N.new("FunctionNodeCompare")
    compare.data_type = data_type
    compare.operation = 'EQUAL'
    word = sock(compare.inputs, COMPARE[data_type][0])
    L.new(source, word)
    ours = read(word)
    theirs = sock(compare.inputs, COMPARE[data_type][1])
    typed_ok = data_type == 'STRING' or isinstance(ours, (int, float))  # an error text can't go into B
    if data_type == 'STRING':
        theirs.default_value = ours if isinstance(ours, str) else str(ours)
    elif typed_ok:
        theirs.default_value = ours
    if data_type == 'FLOAT':
        sock(compare.inputs, "Epsilon").default_value = 0.0
    expected.append((name, ours, typed_ok))
    words.append(word)
    geometry = store(tree, geometry, compare.outputs[0], f"case{index}", 'BOOLEAN')
fresh_empty = N.new("FunctionNodeCompare")  # Geometry Nodes' own output for a menu at -1
fresh_empty.data_type = 'STRING'
fresh_empty.operation = 'EQUAL'
L.new(fresh.outputs[0], sock(fresh_empty.inputs, "A_STR"))
geometry = store(tree, geometry, fresh_empty.outputs[0], "fresh_empty", 'BOOLEAN')
L.new(geometry, g_out.inputs[0])

depsgraph = bpy.context.evaluated_depsgraph_get()
depsgraph.update()
evaluated = host.evaluated_get(depsgraph)
result = evaluated.to_mesh()
for index, (name, ours, typed_ok) in enumerate(expected):
    check(name, typed_ok and result.attributes[f"case{index}"].data[0].value, f"reader gives {ours!r}")
check("menu at -1 (fresh modifier): Geometry Nodes selects no item, even with an interface default",
      mod[ids["Fresh"]] == -1 and result.attributes["fresh_empty"].data[0].value, mod[ids["Fresh"]])
evaluated.to_mesh_clear()

# the per-pass memo: identical values, each output read once per context
evaluator.MEMO = {}
memo_values = [read(word) for word in words]
check("memo: every case gives the same value", memo_values == [ours for _, ours, _ in expected],
      [name for (name, ours, _), got in zip(expected, memo_values) if got != ours])
check("memo: outputs are remembered", len(evaluator.MEMO) > len(words), len(evaluator.MEMO))
memo_word = string("before")
memo_reader = join([])
L.new(memo_word, memo_reader.inputs["Delimiter"])
first = read(memo_reader.inputs["Delimiter"])
memo_word.node.string = "after"
second = read(memo_reader.inputs["Delimiter"])
evaluator.MEMO = None
third = read(memo_reader.inputs["Delimiter"])
check("memo: an output is read once per pass", (first, second, third) == ("before", "before", "after"),
      (first, second, third))
host2 = bpy.data.objects.new("Host 2", mesh)  # the same tree, other modifier values
bpy.context.scene.collection.objects.link(host2)
mod2 = host2.modifiers.new("Eval", 'NODES')
mod2.node_group = tree
mod2[ids["Word"]] = "second"
evaluator.MEMO = {}
per_context = [read(words[[name for name, _, _ in expected].index("reroute chain")], c)
               for c in (ctx, evaluator.Ctx(host2, mod2, (), tree))]
evaluator.MEMO = None
check("memo: contexts keep their own values", per_context == ["héllo wörld", "second"], per_context)
chain = [string("deep")]
for _ in range(40):  # each Reroute takes two of the 64 levels MAX_DEPTH allows
    hop = N.new("NodeReroute")
    L.new(chain[-1], hop.inputs[0])
    chain.append(hop.outputs[0])
near, far = join([]), join([])
L.new(chain[20], near.inputs["Delimiter"])
L.new(chain[40], far.inputs["Delimiter"])
alone = read(far.inputs["Delimiter"])
evaluator.MEMO = {}
got = (read(near.inputs["Delimiter"]), read(far.inputs["Delimiter"]))
evaluator.MEMO = None
check("memo: a nearer read doesn't carry a deeper one past MAX_DEPTH",
      got == ("deep", alone) and "too deep" in alone, (got, alone))
source = string("old")
readers = []
for hops in (0, 2, 1):  # each reader through its own Reroutes
    socket = source
    for _ in range(hops):
        hop = N.new("NodeReroute")
        L.new(socket, hop.inputs[0])
        socket = hop.outputs[0]
    reader = join([])
    L.new(socket, reader.inputs["Delimiter"])
    readers.append(reader.inputs["Delimiter"])
evaluator.MEMO = {}
got = [read(readers[0])]
source.node.string = "new"  # only a read that misses the memo sees an edit
got.append(read(readers[1]))
source.node.string = "newer"
got.append(read(readers[2]))
evaluator.MEMO = None
check("memo: the deepest read's entry serves the shallower reads after it", got == ["old", "new", "new"], got)

# a red (invalid) link must be reported, not silently read
bad = N.new("GeometryNodeStringJoin")
L.new(N.new("ShaderNodeValue").outputs[0], bad.inputs["Strings"])  # float into string: invalid
try:
    evaluator.eval_input(bad.inputs["Strings"], ctx)
    check("invalid link is reported", False, "no error")
except evaluator.Unresolvable as error:
    check("invalid link is reported", True, error)


def reader_error(source):
    """What the reader reports for a socket linked from source."""
    compare = N.new("FunctionNodeCompare")
    compare.data_type = {'INT': 'INT', 'VALUE': 'FLOAT'}.get(source.type, 'STRING')
    word = sock(compare.inputs, COMPARE[compare.data_type][0])
    L.new(source, word)
    try:
        return f"no error, reads {evaluator.eval_input(word, ctx)!r}"
    except evaluator.Unresolvable as error:
        return str(error)


# breadcrumbs: the socket that was read, then up to two nodes before the one that failed. A field that
# turns into a string gives a red link, so fields travel on Float sockets here.
ask = N.new(engine.ASK_ID)
missing = N.new("GeometryNodeImportText")
missing.inputs["Path"].default_value = os.path.join(TMP, "missing.txt")
context = join([missing.outputs[0]])
L.new(context.outputs[0], ask.inputs["Context"])
got = read(ask.inputs["Context"])
check("breadcrumb: from an OpenAct node's word", got == f"<Unresolvable: Context ← {context.name} ← {missing.name}: "
      f"Can't read the text file '{missing.inputs['Path'].default_value}'>", got)
pick = N.new(engine.PICK_ID)
L.new(missing.outputs[0], pick.inputs["Prompt"])
L.new(pick.outputs["Item"], ask.inputs["Question"])
got = read(ask.inputs["Question"])
check("breadcrumb: a chained OpenAct node carries the socket it read", got == f"<Unresolvable: Question ← "
      f"{pick.name}.Prompt ← {missing.name}: Can't read the text file '{missing.inputs['Path'].default_value}'>", got)
attribute = N.new("GeometryNodeInputNamedAttribute")
chosen = N.new("GeometryNodeSwitch")
chosen.input_type = 'FLOAT'
sock(chosen.inputs, "Switch").default_value = True
through = N.new("NodeReroute")
L.new(attribute.outputs["Attribute"], through.inputs[0])
L.new(through.outputs[0], sock(chosen.inputs, "True"))
got = reader_error(chosen.outputs[0])
check("breadcrumb: a field, reroutes left out", got == f"A ← {chosen.name} ← {attribute.name}: fields can't be words",
      got)
first_item = N.new("GeometryNodeIndexSwitch")
first_item.data_type = 'FLOAT'
L.new(chosen.outputs[0], [s for s in first_item.inputs if s.identifier not in ("Index", "__extend__")][0])
got = reader_error(first_item.outputs[0])
check("breadcrumb: at most 3 hops", got == f"A ← … ← {chosen.name} ← {attribute.name}: fields can't be words", got)
math = N.new("ShaderNodeMath")
math_text = N.new("FunctionNodeValueToString")
L.new(math.outputs[0], math_text.inputs["Value"])
got = reader_error(math_text.outputs[0])
check("breadcrumb: an unsupported node", got == f"A ← {math_text.name} ← {math.name}: Math isn't supported as an "
      "OpenAct input", got)
got = reader_error(field_ask.outputs["Answer"])
check("an OpenAct field output is rejected", got == f"A ← {field_ask.name}: '{field_ask.name}' output 'Answer' is a "
      "field; words must be single values", got)
unset = N.new("GeometryNodeMenuSwitch")
unset.data_type = 'STRING'
unset.enum_items.clear()  # leaves the Menu socket on no item
unset.enum_items.new("Q")
got = reader_error(unset.outputs[0])
check("menu: an unknown name names the menu", got == f"A ← {unset.name}: '{unset.name}' has no item ''", got)
got = reader_error(fresh.outputs[0])
check("menu: -1 (fresh modifier) names the modifier's menu",
      got == f"A ← {fresh.name} ← {g_in.name}: choose an item for 'Fresh' in the modifier", got)
mod[ids["Edited"]] = 0  # A, deleted above
got = reader_error(edited.outputs[0])
mod[ids["Edited"]] = 2
check("menu: a deleted item's identifier names the modifier's menu",
      got == f"A ← {edited.name} ← {g_in.name}: choose an item for 'Edited' in the modifier", got)


def linked_library():
    """A pass reads linked OpenAct nodes but writes nothing into them."""
    library_tree = bpy.data.node_groups.new("Library Tree", 'GeometryNodeTree')
    node = library_tree.nodes.new(engine.ASK_ID)
    node.inputs["Question"].default_value = "Is it linked?"
    saved = S.NodeStore([S.Entry(words_key="saved", gn=("Is it linked?", ""), kind="ask", answer_type='YES_NO',
                                 single={"Answer": 1.0, "Yes": True, "Confidence": 1.0})])
    S.save(node, saved)
    inner.build(node, saved)
    path = os.path.join(TMP, "library.blend")
    bpy.data.libraries.write(path, {library_tree})
    group = node.node_tree
    bpy.data.node_groups.remove(library_tree)
    bpy.data.node_groups.remove(group)
    loaded = []
    for link in (False, True):  # appended, then linked: two nodes that arrive with one uid
        with bpy.data.libraries.load(path, link=link) as (_source, target):
            target.node_groups = ["Library Tree"]
        loaded.append(target.node_groups[0])
    appended, linked = loaded
    user = bpy.data.objects.new("Library User", bpy.data.meshes.new("Library User"))
    bpy.context.scene.collection.objects.link(user)
    user_mod = user.modifiers.new("Linked", 'NODES')
    user_mod.node_group = linked
    node = next(n for n in linked.nodes if n.bl_idname == engine.ASK_ID)
    local_node = next(n for n in appended.nodes if n.bl_idname == engine.ASK_ID)
    shared = local_node.uid == node.uid
    linked_inner = node.node_tree
    before = (node.uid, node.entries, linked_inner, len(bpy.data.node_groups), len(linked_inner.nodes))
    engine.run_pass()
    after = (node.uid, node.entries, node.node_tree, len(bpy.data.node_groups), len(linked_inner.nodes))
    check("linked: the tree and its hidden group are linked",
          linked.library is not None and linked_inner is not None and linked_inner.library is not None)
    status = engine.STATUS.get(node.uid)
    check("linked: the node says where its answers come from",
          status == ('LINKED', f"Linked from {linked.library.name}: answers come from that file"), status)
    check("linked: a pass writes nothing", after == before, (before, after))
    local_status = engine.STATUS.get(local_node.uid)
    check("append then link: the local copy gets its own uid and status",
          shared and local_node.uid != node.uid and local_status is not None and local_status[0] != 'LINKED',
          (shared, local_node.uid, node.uid, local_status))
    check("append then link: each uid finds its own node, so Ask Again can't clear the wrong one",
          engine.find_node(node.uid)[1] == node and engine.find_node(local_node.uid)[1] == local_node)
    check("append then link: the local copy's hidden group follows its new uid",
          local_node.node_tree.library is None and local_node.node_tree.name == inner.inner_name(local_node),
          local_node.node_tree.name)
    check("linked: the pass's memo is gone after it", evaluator.MEMO is None, evaluator.MEMO)
    engine.reask(node)
    check("linked: Ask Again writes nothing", node.entries == before[1] and len(linked_inner.nodes) == before[4])
    got = read(node.inputs["Question"], evaluator.Ctx(user, user_mod, (), linked))
    check("linked: its words are still read", got == "Is it linked?", got)
    local = linked.make_local()  # the tree alone: its node still points at the linked hidden group
    node = next(n for n in local.nodes if n.bl_idname == engine.ASK_ID)
    engine.run_pass()
    check("made local: the node gets a local hidden group, the linked one is untouched",
          node.node_tree is not linked_inner and node.node_tree.library is None
          and node.node_tree.name == inner.inner_name(node) and len(linked_inner.nodes) == before[4],
          (node.node_tree.name, node.node_tree.library, local.library))
    bpy.data.objects.remove(user)  # its tree has no Group Output, which later scene updates would report


def reappended_asset():
    """An asset deleted and appended again keeps its node's uid, but its hidden group arrives as '....001'
    because the old group, now unused, still holds the name. Passes must settle anyway."""
    asset = bpy.data.node_groups.new("Asset", 'GeometryNodeTree')
    asset.nodes.new(engine.ASK_ID).inputs["Question"].default_value = "Is it an asset?"
    path = os.path.join(TMP, "asset.blend")
    bpy.data.libraries.write(path, {asset})
    bpy.data.node_groups.remove(asset)  # its hidden group stays behind, unused
    with bpy.data.libraries.load(path) as (_source, target):
        target.node_groups = ["Asset"]
    node = next(n for n in target.node_groups[0].nodes if n.bl_idname == engine.ASK_ID)
    arrived = node.node_tree.name
    bpy.context.view_layer.update()
    bpy.app.handlers.depsgraph_update_post.append(engine.on_depsgraph_update)  # what schedules passes live
    try:
        engine.run_pass()
        bpy.context.view_layer.update()
        before = (sorted(group.name for group in bpy.data.node_groups), node.uid, node.node_tree.name)
        engine.run_pass()
        bpy.context.view_layer.update()  # a pass that wrote anything schedules another one here
        after = (sorted(group.name for group in bpy.data.node_groups), node.uid, node.node_tree.name)
        scheduled = engine._dirty
    finally:
        bpy.app.handlers.depsgraph_update_post.remove(engine.on_depsgraph_update)
    check("re-appended asset: two passes leave the tree unchanged and schedule no further pass",
          arrived.endswith(".001") and after == before and not scheduled,
          (arrived, before[1:], after[1:], scheduled))


linked_library()
reappended_asset()
finish()
