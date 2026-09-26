"""List sources, the Import CSV hand-off, answer sets, the trust rule and the CSV watcher, with no model.

A group node with the list-source and CSV sockets stands in for the OpenAct nodes (their specs arrive with
PR-09). Checks that a collection's entries come in Collection Info's instance order, that an empty material slot
keeps its index, that the Attributes source reads original data, that trace_csv_source() follows Reroutes, nested
groups and selectors to an Import CSV node and names what is in the way otherwise, that its Delimiter is used, that
an untrusted file reads nothing outside its folder until Allow This File, and that the watcher sees edits.

Run:  blender -b --factory-startup --python tests/sources_blender.py
"""

import builtins
import os
import shutil
import sys
import tempfile

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import check, finish, sock  # noqa: E402
from openact import csvfmt, engine, evaluator, geometry, inner, lists, nodes, ops, sources, trust, watch  # noqa: E402
from openact import prefs as P  # noqa: E402
from openact.inner import SocketDef  # noqa: E402

TMP = tempfile.mkdtemp(prefix="openact_sources_")
FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
PROPS = os.path.join(FIXTURES, "props.csv")
FAMILIES = os.path.join(FIXTURES, "material_families.csv")
P._Defaults.auto_start = False  # nothing here may start a model server
nodes.register()
trust.register()
watch.register()
bpy.utils.register_class(ops.OPENACT_OT_allow_file)


def write(name, text, folder=TMP):
    path = os.path.join(folder, name)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    return path


def bump(path):
    """A new stamp even when the edit fell within the file system's clock tick."""
    status = os.stat(path)
    os.utime(path, ns=(status.st_atime_ns, status.st_mtime_ns + 2_000_000_000))


def outcome(call):
    try:
        return call()
    except Exception as error:  # noqa: BLE001
        return f"<{type(error).__name__}: {error}>"


def plug(source, target):
    for link in list(target.links):
        target.id_data.links.remove(link)
    if source is not None:
        target.id_data.links.new(source, target)


def host_object(name, tree):
    obj = bpy.data.objects.new(name, bpy.data.meshes.new(name))
    bpy.context.scene.collection.objects.link(obj)
    mod = obj.modifiers.new(name, 'NODES')
    mod.node_group = tree
    return obj, mod


def geometry_tree(name):
    tree = bpy.data.node_groups.new(name, 'GeometryNodeTree')
    tree.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    tree.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    g_in, g_out = tree.nodes.new("NodeGroupInput"), tree.nodes.new("NodeGroupOutput")
    tree.links.new(g_in.outputs[0], g_out.inputs[0])
    return tree, g_in, g_out


# ---------------------------------------------------------------- the stand-in node
DEFS = ([SocketDef('OUTPUT', "Geometry", 'NodeSocketGeometry')] + sources.list_inputs('TEXT')
        + sources.list_inputs('COLLECTION') + sources.list_inputs('MATERIALS')
        + sources.csv_inputs("Answers", "Yes/No whens, Amount levels or One-of options"))
STAND = bpy.data.node_groups.new("Stand-in", 'GeometryNodeTree')
STAND.nodes.new("NodeGroupOutput")


def stand_in(tree):
    node = tree.nodes.new("GeometryNodeGroup")
    node.node_tree = STAND
    inner.sync_interface(node, DEFS)
    return node


tree, g_in, g_out = geometry_tree("Sources")
tree.interface.new_socket("CSV Path", in_out='INPUT', socket_type='NodeSocketString')
N, L = tree.nodes, tree.links
host, mod = host_object("Host", tree)
host2, mod2 = host_object("Host 2", tree)
ids = {i.name: i.identifier for i in tree.interface.items_tree if getattr(i, "in_out", "") == 'INPUT'}
ctx, ctx2 = evaluator.Ctx(host, mod, (), tree), evaluator.Ctx(host2, mod2, (), tree)
stand = stand_in(tree)
names = [s.name for s in stand.inputs]
check("companions: every CSV input has its '<Input> CSV' Geometry socket directly below it",
      names[names.index("Items") + 1] == "Items CSV" and names[names.index("Answers") + 1] == "Answers CSV"
      and stand.inputs["Items CSV"].type == 'GEOMETRY', names)
defs = sources.csv_inputs("Overrides", "Overrides CSV path", hidden=True)
check("companions: declared hidden with their input, kinds word and csv",
      [(d.name, d.socket_type, d.kind, d.hidden) for d in defs] == [
          ("Overrides", 'NodeSocketString', "word", True), ("Overrides CSV", 'NodeSocketGeometry', "csv", True)])
check("list sockets: one set per list source",
      [[d.name for d in sources.list_inputs(s)] for s in lists.SOURCES] == [
          ["Items", "Items CSV"], ["Collection"], ["Object"], ["Object"]])

# ---------------------------------------------------------- Collection, natsort order
props = bpy.data.collections.new("Props")
bpy.context.scene.collection.children.link(props)
shape = bpy.data.meshes.new("Shape")
shape.from_pydata([(0, 0, 0)], [], [])
for name in ("chair 10", "Chair 2", "lamp", "Table.001", "table", "bench_3", "\u00c4mber", "chair 1"):
    props.objects.link(bpy.data.objects.new(name, shape))
crates = bpy.data.collections.new("crates")
props.children.link(crates)
crates.objects.link(bpy.data.objects.new("crate", shape))
stand.inputs["Collection"].default_value = props
got = lists.resolve_items(stand, ctx, source='COLLECTION')
order_tree = bpy.data.node_groups.new("Order", 'GeometryNodeTree')
order_tree.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
info = order_tree.nodes.new("GeometryNodeCollectionInfo")
info.inputs["Collection"].default_value = props
sock(info.inputs, "Separate Children").default_value = True
sock(info.inputs, "Reset Children").default_value = True
order_tree.links.new(info.outputs[0], order_tree.nodes.new("NodeGroupOutput").inputs[0])
order, _order_mod = host_object("Order", order_tree)
depsgraph = bpy.context.evaluated_depsgraph_get()
depsgraph.update()
instances = bpy.types.GeometrySet.from_evaluated_object(order.evaluated_get(depsgraph))
references = instances.instance_references()
points = instances.instances_pointcloud()
truth = [references[d.value].name for d in points.attributes[".reference_index"].data]
del points, instances
check("collection: an item's index is its Collection Info (Separate Children) instance index",
      list(got.names) == truth and len(truth) == 9, (got.names, truth))
check("collection: the list keeps the collection for the Instance output, and a fingerprint",
      got.datablock == props and got.source == 'COLLECTION' and got.fingerprint[1] == got.names)
stand.inputs["Collection"].default_value = None
check("collection: none chosen is a hint", outcome(lambda: lists.resolve_items(stand, ctx, source='COLLECTION'))
      == "<Hint: Choose a collection>")

# ------------------------------------------------------- Materials with an empty slot
palette_mesh = bpy.data.meshes.new("Palette")
palette = bpy.data.objects.new("Palette", palette_mesh)
red, blue = bpy.data.materials.new("Red"), bpy.data.materials.new("Blue")
for material in (red, None, blue):
    palette_mesh.materials.append(material)
stand.inputs["Object"].default_value = palette
got = lists.resolve_items(stand, ctx, source='MATERIALS')
check("materials: slot order, the empty slot keeps its index",
      got.names == ("Red", "", "Blue") and [i.offered for i in got.items] == [True, False, True], got.items)
check("materials: the empty slot is never offered", [(i, item.name) for i, item in got.options()] == [
    (0, "Red"), (2, "Blue")])
empty = bpy.data.objects.new("Bare", bpy.data.meshes.new("Bare"))
stand.inputs["Object"].default_value = empty
check("materials: an object without materials is a hint",
      outcome(lambda: lists.resolve_items(stand, ctx, source='MATERIALS')) == "<Hint: 'Bare' has no materials>")

# ------------------------------------------------------- Attributes, original data only
wet_tree, wet_in, wet_out = geometry_tree("Wet")
wet_mesh = bpy.data.meshes.new("Wet")
wet_mesh.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [], [(0, 1, 2)])
for name, domain in (("wetness", 'POINT'), ("stored_by_tree", 'POINT'), ("linked_name", 'POINT'),
                     ("zone 10", 'FACE'), ("zone 2", 'FACE')):
    wet_mesh.attributes.new(name, 'FLOAT', domain)
wet, wet_mod = host_object("Wet", wet_tree)
wet.data = wet_mesh
geo = wet_in.outputs[0]
pick = wet_tree.nodes.new(engine.PICK_ID)  # an OpenAct answer as a Name: must neither be excluded nor evaluated
name_node = wet_tree.nodes.new("FunctionNodeInputString")
name_node.string = "linked_name"
for name in ("stored_by_tree", "evaluated_only", name_node.outputs[0], pick.outputs["Item"]):
    store = wet_tree.nodes.new("GeometryNodeStoreNamedAttribute")
    wet_tree.links.new(geo, store.inputs["Geometry"])
    if isinstance(name, str):
        store.inputs["Name"].default_value = name
    else:
        wet_tree.links.new(name, store.inputs["Name"])
    geo = store.outputs["Geometry"]
wet_tree.links.new(geo, wet_out.inputs[0])
wet_stand = stand_in(wet_tree)
wet_ctx = evaluator.Ctx(wet, wet_mod, (), wet_tree)
got = outcome(lambda: lists.resolve_items(wet_stand, wet_ctx, source='ATTRIBUTES'))
depsgraph = bpy.context.evaluated_depsgraph_get()
depsgraph.update()
evaluated = wet.evaluated_get(depsgraph).data
check("attributes: the evaluated mesh has the tree's own attribute", "evaluated_only" in evaluated.attributes)
names = list(got.names) if isinstance(got, lists.ItemList) else got
check("attributes: names from the original data, natsorted",
      isinstance(got, lists.ItemList) and "wetness" in names and "evaluated_only" not in names
      and names.index("zone 2") < names.index("zone 10") and names == sorted(names, key=lists.natsort.sort_key),
      names)
check("attributes: no '.' names, nor names this tree writes (typed or linked)",
      isinstance(got, lists.ItemList) and not any(n.startswith(".") for n in names)
      and "stored_by_tree" not in names and "linked_name" not in names, names)
check("attributes: an empty Object means the modifier's object", isinstance(got, lists.ItemList) and got.datablock == wet)
check("attributes: attribute_names() skips exclude", "wetness" not in geometry.attribute_names(wet, {"wetness"}))
values = geometry.read_attributes(wet, ["evaluated_only", "missing"])
check("attributes: read_attributes() reads evaluated values (agents hook)",
      list(values) == ["evaluated_only"] and len(values["evaluated_only"]) == 3, values)

# ----------------------------------------------------------------- Text or CSV items
stand.inputs["Items"].default_value = "chair: to sit on; table, lamp"
got = lists.resolve_items(stand, ctx, source='TEXT')
check("text: inline items with descriptions", [(i.name, i.when) for i in got.items] == [
    ("chair", "to sit on"), ("table, lamp", "")] and got.raw == "chair: to sit on; table, lamp", got.items)
stand.inputs["Items"].default_value = ""
check("text: empty is a hint", outcome(lambda: lists.resolve_items(stand, ctx, source='TEXT'))
      == "<Hint: Type items separated by commas, or a .csv path>")
stand.inputs["Items"].default_value = "//props.csv"
check("paths: '//' in an unsaved file is a hint", outcome(lambda: lists.resolve_items(stand, ctx, source='TEXT'))
      == "<Hint: Save the .blend: '//' paths are relative to it>")
stand.inputs["Items"].default_value = PROPS
got = lists.resolve_items(stand, ctx, source='TEXT')
check("text: a typed CSV path, payload per row",
      len(got.items) == 10 and got.items[8].name == "anvil" and got.items[8].payload == {"instance": 8, "scale": 0.6}
      and got.doc.num_cols == {"instance": "INT", "scale": "FLOAT"}, got.items[8])
check("watch: reading a CSV watches it for the node",
      stand.name in watch.WATCH.files.get(csvfmt.identity(PROPS), watch._File("", None)).owners)

# ----------------------------------------------------------- Import CSV hand-off
stand.inputs["Items"].default_value = ""
imp = N.new("GeometryNodeImportCSV")
imp.inputs["Path"].default_value = PROPS
plug(imp.outputs[0], stand.inputs["Items CSV"])
src = lists.csv_source(stand, "Items", ctx)
check("hand-off: a direct link reads the Import CSV node's Path and Delimiter",
      src == csvfmt.CsvSource(PROPS, ",", (imp.name,)), src)
got = lists.resolve_items(stand, ctx, source='TEXT')
check("hand-off: its file gives the items", got.names[8] == "anvil" and got.csv == src, got.names)
stand.inputs["Items"].default_value = "//typed.csv"
src = lists.csv_source(stand, "Items", ctx)
got = lists.resolve_items(stand, ctx, source='TEXT')
warning = "Items: using the linked Import CSV; the typed path is ignored"
check("hand-off: the link wins over a typed path, with a warning",
      src.path == PROPS and src.warning == warning and got.problems[-1].message == warning, src)
stand.inputs["Items CSV"].links[0].is_muted = True
check("hand-off: a muted link counts as unlinked", outcome(lambda: lists.csv_source(stand, "Items", ctx))
      == csvfmt.CsvSource("//typed.csv"))
stand.inputs["Items"].default_value = ""
hops = []
socket = imp.outputs[0]
for _ in range(2):
    reroute = N.new("NodeReroute")
    L.new(socket, reroute.inputs[0])
    socket = reroute.outputs[0]
plug(socket, stand.inputs["Items CSV"])
check("hand-off: through Reroutes", outcome(lambda: lists.csv_source(stand, "Items", ctx))
      == csvfmt.CsvSource(PROPS, ",", (imp.name,)))

source_group = bpy.data.node_groups.new("CSV Source", 'GeometryNodeTree')  # a path exposed on the modifier
source_group.interface.new_socket("Path", in_out='INPUT', socket_type='NodeSocketString')
source_group.interface.new_socket("Points", in_out='OUTPUT', socket_type='NodeSocketGeometry')
s_in, s_out = source_group.nodes.new("NodeGroupInput"), source_group.nodes.new("NodeGroupOutput")
nested_import = source_group.nodes.new("GeometryNodeImportCSV")
source_group.links.new(s_in.outputs[0], nested_import.inputs["Path"])
source_group.links.new(nested_import.outputs[0], s_out.inputs[0])
group = N.new("GeometryNodeGroup")
group.node_tree = source_group
L.new(g_in.outputs["CSV Path"], group.inputs["Path"])
plug(group.outputs[0], stand.inputs["Items CSV"])
mod[ids["CSV Path"]] = PROPS
mod2[ids["CSV Path"]] = FAMILIES
got = [outcome(lambda c=c: lists.csv_source(stand, "Items", c)) for c in (ctx, ctx2)]
check("hand-off: out of a nested group, with the Path from the modifier, per context", got == [
    csvfmt.CsvSource(PROPS, ",", (group.name, nested_import.name)),
    csvfmt.CsvSource(FAMILIES, ",", (group.name, nested_import.name))], got)

user_group = bpy.data.node_groups.new("CSV User", 'GeometryNodeTree')  # the node inside a group, fed from outside
user_group.interface.new_socket("Table", in_out='INPUT', socket_type='NodeSocketGeometry')
u_in = user_group.nodes.new("NodeGroupInput")
user_stand = stand_in(user_group)
user_group.links.new(u_in.outputs[0], user_stand.inputs["Items CSV"])
user = N.new("GeometryNodeGroup")
user.node_tree = user_group
outer_import = N.new("GeometryNodeImportCSV")
outer_import.inputs["Path"].default_value = FAMILIES
L.new(outer_import.outputs[0], user.inputs["Table"])
got = outcome(lambda: lists.csv_source(user_stand, "Items", ctx.push(user)))
check("hand-off: into a nested group through its Group Input",
      got == csvfmt.CsvSource(FAMILIES, ",", (user.name, outer_import.name)), got)

menu = N.new("GeometryNodeMenuSwitch")
menu.data_type = 'GEOMETRY'
choices = [s for s in menu.inputs if s.identifier not in ("Menu", "__extend__")]
plug(imp.outputs[0], choices[0])
plug(outer_import.outputs[0], choices[1])
menu.inputs["Menu"].default_value = 'B'
plug(menu.outputs[0], stand.inputs["Answers CSV"])
got = outcome(lambda: sources.answer_set(stand, ctx, "Answers", "one_of", spec="classify"))
check("hand-off: a Geometry Menu Switch chooses the Import CSV node",
      isinstance(got, csvfmt.AnswerSet) and [o.name for o in got.options] == ["metal", "wood", "stone", "fabric"]
      and got.source.via == (menu.name, outer_import.name) and got.options[0].payload == {"roughness": 0.35,
                                                                                          "metallic": 1}, got)
index_switch = N.new("GeometryNodeIndexSwitch")
index_switch.data_type = 'GEOMETRY'
items = [s for s in index_switch.inputs if s.identifier not in ("Index", "__extend__")]
plug(outer_import.outputs[0], items[0])
plug(imp.outputs[0], items[1])
sock(index_switch.inputs, "Index").default_value = 1
switch = N.new("GeometryNodeSwitch")
switch.input_type = 'GEOMETRY'
sock(switch.inputs, "Switch").default_value = True
plug(index_switch.outputs[0], sock(switch.inputs, "True"))
plug(switch.outputs[0], stand.inputs["Items CSV"])
got = outcome(lambda: lists.csv_source(stand, "Items", ctx))
check("hand-off: Switch and Index Switch of type Geometry",
      got == csvfmt.CsvSource(PROPS, ",", (switch.name, index_switch.name, imp.name)), got)

grid = N.new("GeometryNodeMeshGrid")
plug(grid.outputs[0], stand.inputs["Items CSV"])
check("hand-off: another source says to connect an Import CSV node",
      outcome(lambda: lists.csv_source(stand, "Items", ctx))
      == f"<Unresolvable: Items CSV: connect an Import CSV node (found '{grid.name}')>")
transform = N.new("GeometryNodeTransform")
L.new(imp.outputs[0], transform.inputs[0])
plug(transform.outputs[0], stand.inputs["Items CSV"])
check("hand-off: a modifying node in between says to connect it directly",
      outcome(lambda: lists.csv_source(stand, "Items", ctx))
      == f"<Unresolvable: Items CSV: connect the Import CSV node directly ('{transform.name}' is in between)>")
blank = N.new("GeometryNodeImportCSV")
plug(blank.outputs[0], stand.inputs["Items CSV"])
check("hand-off: an empty Path is a hint", outcome(lambda: lists.csv_source(stand, "Items", ctx))
      == f"<Hint: Items CSV: type a Path on '{blank.name}'>")

semi = write("semi.csv", 'answer;when;"size, cm"\npan;cooking, frying;30\nsofa;sitting;200\n')
blank.inputs["Path"].default_value = semi
blank.inputs["Delimiter"].default_value = ";"
got = outcome(lambda: lists.resolve_items(stand, ctx, source='TEXT'))
check("hand-off: the Import CSV node's ';' Delimiter is used, not sniffed",
      isinstance(got, lists.ItemList) and [(i.name, i.when, i.payload) for i in got.items] == [
          ("pan", "cooking, frying", {"size, cm": 30}), ("sofa", "sitting", {"size, cm": 200})]
      and got.csv.delimiter == ";", got)
check("hand-off: sniffing would have split that header at the comma",
      csvfmt.read_csv(semi).header == ('answer;when;"size', 'cm"'), csvfmt.read_csv(semi).header)

# ---------------------------------------------------------------------- answer sets
plug(None, stand.inputs["Answers CSV"])
stand.inputs["Answers"].default_value = "yes: it is cold, wet or windy; no: warm and calm"
got = sources.answer_set(stand, ctx, "Answers", "yes_no", spec="ask")
check("answers: inline Yes/No whens become the noul criteria", sources.question("yes_no", "Is it stormy?", got) == {
    "type": "noul", "instructions": "Is it stormy?",
    "criteria": {"true": "it is cold, wet or windy", "false": "warm and calm"}})
stand.inputs["Answers"].default_value = ""
got = sources.answer_set(stand, ctx, "Answers", "amount", spec="ask")
check("answers: empty Amount means the five default levels", [o.name for o in got.options] == list(
    sources.AMOUNT_LEVELS))
check("answers: an empty One-of set is a hint", outcome(
    lambda: sources.answer_set(stand, ctx, "Answers", "one_of", spec="classify", empty_hint="Add labels"))
      == "<Hint: Add labels>")
snow = write("snow.csv", "answer,when\nnone,bare ground\ndusting,a thin white layer\npatchy,snow in patches\n"
                         "thick,most ground covered\nburied,deep snow everywhere\n")
stand.inputs["Answers"].default_value = snow
got = sources.answer_set(stand, ctx, "Answers", "amount", spec="ask")
check("answers: a CSV of 5 custom levels maps score / 4",
      len(got.options) == 5 and got.doc is not None and sources.amount_value(3.0, len(got.options)) == 0.75)
broken = write("broken.csv", "answer\nyes\nmaybe\n")
stand.inputs["Answers"].default_value = broken
got = outcome(lambda: sources.answer_set(stand, ctx, "Answers", "yes_no", spec="ask"))
check("answers: a file with errors names the file", got == "<FileError: broken.csv: A Yes/No answer set needs rows "
      "'yes' and 'no'>", got)
stand.inputs["Answers"].default_value = "only one"
got = sources.answer_set(stand, ctx, "Answers", "one_of", spec="classify")
check("answers: a single option is decided by code", sources.decided(got))

# ------------------------------------------------------------------------ the watcher
w = watch.Watcher()
seen = []
w.listeners.append(seen.extend)
text = "answer,when,scale\npan,cooking,1\nsofa,sitting,2\n"
path = write("watched.csv", text)
w.use("node", csvfmt.load(path))
check("watch: nothing changed, nothing reported", w.poll(force=True) == [])
write("watched.csv", text.replace(",1\n", ",5\n"))
bump(path)
check("watch: not polled again within 0.5 s", w.poll() == [])
changes = w.poll(force=True)
check("watch: a numeric-only edit", [(c.kind, c.owners) for c in changes] == [("numbers", frozenset({"node"}))],
      changes)
write("watched.csv", text.replace("cooking", "frying").replace(",1\n", ",5\n"))
bump(path)
check("watch: a text edit", [c.kind for c in w.poll(force=True)] == ["text"])
bump(path)
check("watch: saved without a real change, nothing", w.poll(force=True) == [])
os.remove(path)
changes = w.poll(force=True)
check("watch: a missing file", [c.kind for c in changes] == ["missing"] and changes[0].message.startswith(
    "Can't find"), changes)
write("watched.csv", "answer,when,scale\npan,\"frying\n")
changes = w.poll(force=True)
good = w.files[csvfmt.identity(path)].reads[("answer_set", None)]
check("watch: a broken file is an error, and the last good read stays",
      [c.kind for c in changes] == ["error"] and len(good.rows) == 2 and good.rows[0][1] == "frying", changes)
write("watched.csv", text.replace("cooking", "frying").replace(",1\n", ",5\n"))
bump(path)
check("watch: fixed with the last good content, restored", [c.kind for c in w.poll(force=True)] == ["restored"])
check("watch: dirty owners and listeners", w.take_dirty() == {"node": "restored"} and w.take_dirty() == {}
      and [c.kind for c in seen] == ["numbers", "text", "missing", "error", "restored"], seen)
many = watch.Watcher()
paths = [write(f"many_{i}.csv", "answer\na\n") for i in range(40)]
for i, p in enumerate(paths):
    many.use(f"n{i}", csvfmt.load(p))
write("many_35.csv", "answer\nb\n")
bump(paths[35])
first, second = many.poll(force=True), many.poll(force=True)
check("watch: 32 files per poll, round-robin", first == [] and [c.owners for c in second] == [frozenset({"n35"})],
      (first, second))
many.retain({"n1"})
check("watch: retain() drops the files nobody reads", list(many.files) == [csvfmt.identity(paths[1])])

# ------------------------------------------------------------------------ the trust rule
proj, outside = os.path.join(TMP, "proj"), os.path.join(TMP, "outside")
os.makedirs(proj)
os.makedirs(outside)
write("near.csv", "answer\nnear item\n", proj)
far = write("far.csv", "answer\nsecret item\n", outside)
far_text = write("far.txt", "secret words", outside)
trust_tree = bpy.data.node_groups.new("Trust", 'GeometryNodeTree')
trust_tree.use_fake_user = True  # saved without a modifier using it
reader = stand_in(trust_tree)
reader.name = "Reader"
text_node = trust_tree.nodes.new("GeometryNodeImportText")
text_node.inputs["Path"].default_value = far_text
trust_tree.links.new(text_node.outputs[0], reader.inputs["Answers"])
text_name = text_node.name  # the node itself is gone once the file is opened again
blend = os.path.join(proj, "scene.blend")
check("trust: a file created in this session is trusted", trust.is_trusted())
bpy.ops.wm.save_as_mainfile(filepath=blend)
bpy.ops.wm.open_mainfile(filepath=blend)
reader = bpy.data.node_groups["Trust"].nodes["Reader"]
ctx = evaluator.Ctx(None, None, (), reader.id_data)
check("trust: an opened file without Auto Run is not trusted", not trust.is_trusted() and trust.TRUST["opened"])
opened = []
real_open = builtins.open


def spy(file, *args, **kwargs):
    opened.append(os.path.normcase(os.path.abspath(file)) if isinstance(file, (str, bytes)) else file)
    return real_open(file, *args, **kwargs)


builtins.open = spy
try:
    reader.inputs["Items"].default_value = far
    blocked_csv = outcome(lambda: lists.resolve_items(reader, ctx, source='TEXT'))
    blocked_text = outcome(lambda: evaluator.eval_input(reader.inputs["Answers"], ctx))
finally:
    builtins.open = real_open
message = f"Blocked reading {far}: outside this file's folder. Allow this file (or trust its folder) to read it"
check("trust: a CSV outside the file's folder is blocked", blocked_csv == f"<Unresolvable: {message}>", blocked_csv)
check("trust: Import Text outside the folder is blocked",
      blocked_text == f"<Unresolvable: Answers ← {text_name}: Blocked reading {far_text}: outside this file's "
      "folder. Allow this file (or trust its folder) to read it>", blocked_text)
check("trust: blocked files are never opened",
      not {os.path.normcase(far), os.path.normcase(far_text)} & set(opened), opened)
check("trust: nor cached", csvfmt.identity(far) not in {key[0] for key in csvfmt._DOCS})
reader.inputs["Items"].default_value = "//near.csv"
got = outcome(lambda: lists.resolve_items(reader, ctx, source='TEXT'))
check("trust: a CSV next to the file is read", isinstance(got, lists.ItemList) and got.names == ("near item",), got)
reader.inputs["Items"].default_value = os.path.join(proj, "..", "outside", "far.csv")
check("trust: '..' does not escape", outcome(lambda: lists.resolve_items(reader, ctx, source='TEXT')).startswith(
    "<Unresolvable: Blocked reading"))
P._Defaults.trusted_folders.append(TMP)
check("trust: a trusted parent folder trusts the file", trust.is_trusted())
P._Defaults.trusted_folders.clear()
check("trust: Allow This File", bpy.ops.openact.allow_file() == {'FINISHED'} and trust.is_trusted()
      and P._Defaults.trusted_files == [os.path.normpath(blend)], P._Defaults.trusted_files)
reader.inputs["Items"].default_value = far
got = outcome(lambda: lists.resolve_items(reader, ctx, source='TEXT'))
check("trust: after Allow This File the CSV is read", isinstance(got, lists.ItemList)
      and got.names == ("secret item",), got)
got = outcome(lambda: evaluator.eval_input(reader.inputs["Answers"], ctx))
check("trust: and Import Text too", got == "secret words", got)
check("trust: allowing twice adds nothing", not trust.allow(blend) and len(P._Defaults.trusted_files) == 1)
for cls in P.classes:  # the preferences register, with the trusted lists as collections
    bpy.utils.register_class(cls)
check("prefs: trusted_folders and trusted_files are path collections",
      all(P.OpenActPreferences.bl_rna.properties[name].type == 'COLLECTION'
          for name in ("trusted_folders", "trusted_files")))
for cls in reversed(P.classes):
    bpy.utils.unregister_class(cls)

shutil.rmtree(TMP, ignore_errors=True)
finish()
