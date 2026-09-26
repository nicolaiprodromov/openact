"""End-to-end test of OpenAct in a headless Blender, with the real local model.

Phases (pass after "--"):
  build  <out.blend> [port] [gpu]  build a scene, let the add-on start the model and answer,
                                   check the evaluated geometry, save the file
  noaddon <file.blend>             reopen WITHOUT the add-on: stored answers must still evaluate
  reopen  <file.blend>             reopen WITH the add-on and no server: no new questions needed

Run:  blender -b --factory-startup --python tests/e2e_blender.py -- build C:/tmp/openact_e2e.blend
"""

import os
import sys
import time

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import check, drive, enable_addon, finish, store  # noqa: E402

SCENES = {
    "A": ("A snowy mountain village at night, warm windows glowing.", "something to sit on"),
    "B": ("A sunny tropical beach at noon, people swimming.", "something to put books on"),
}
ITEMS = "table, lamp, armchair, bookshelf, rug"
PROPS = {"Armchair": 3, "Bookshelf": 6, "Floor Lamp": 4, "Rug": 7, "Table": 5}  # name -> vertex count


def build_scene():
    scene = bpy.context.scene
    props = bpy.data.collections.new("Props")
    scene.collection.children.link(props)
    for name, count in PROPS.items():
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata([(i * 0.1, 0.0, 0.0) for i in range(count)], [], [])
        props.objects.link(bpy.data.objects.new(name, mesh))
    props.hide_render = True

    # nested group: Ask (Amount) reading its context from the parent modifier
    nested = bpy.data.node_groups.new("OpenActNested", 'GeometryNodeTree')
    nested.interface.new_socket("Ctx", in_out='INPUT', socket_type='NodeSocketString')
    nested.interface.new_socket("Cold", in_out='OUTPUT', socket_type='NodeSocketFloat')
    n_in = nested.nodes.new("NodeGroupInput")
    n_out = nested.nodes.new("NodeGroupOutput")
    cold = nested.nodes.new("GeometryNodeOpenActAsk")
    cold.mode = 'AMOUNT'
    cold.inputs["Question"].default_value = "How cold is it?"
    nested.links.new(n_in.outputs["Ctx"], cold.inputs["Context"])
    nested.links.new(cold.outputs["Answer"], n_out.inputs["Cold"])

    tree = bpy.data.node_groups.new("OpenActTest", 'GeometryNodeTree')
    tree.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    tree.interface.new_socket("Scene", in_out='INPUT', socket_type='NodeSocketString')
    tree.interface.new_socket("Pick Prompt", in_out='INPUT', socket_type='NodeSocketString')
    tree.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    g_in = tree.nodes.new("NodeGroupInput")
    g_out = tree.nodes.new("NodeGroupOutput")

    night = tree.nodes.new("GeometryNodeOpenActAsk")
    night.inputs["Question"].default_value = "Is it night?"
    tree.links.new(g_in.outputs["Scene"], night.inputs["Context"])

    pick = tree.nodes.new("GeometryNodeOpenActPick")
    tree.links.new(g_in.outputs["Pick Prompt"], pick.inputs["Prompt"])
    pick.inputs["Items"].default_value = ITEMS

    sit = tree.nodes.new("GeometryNodeOpenActAsk")  # chained: Pick.Item -> Ask.Context
    sit.inputs["Question"].default_value = "Can a person sit on this?"
    tree.links.new(pick.outputs["Item"], sit.inputs["Context"])

    group = tree.nodes.new("GeometryNodeGroup")
    group.node_tree = nested
    tree.links.new(g_in.outputs["Scene"], group.inputs["Ctx"])

    lamp = tree.nodes.new("GeometryNodeOpenActPick")  # collection mode + Instance output
    lamp.inputs["Prompt"].default_value = "something that gives light"
    lamp.inputs["Collection"].default_value = props

    geometry = g_in.outputs["Geometry"]
    geometry = store(tree, geometry, night.outputs["Answer"], "night", 'FLOAT', 200)
    geometry = store(tree, geometry, night.outputs["Yes"], "night_yes", 'BOOLEAN', 400)
    geometry = store(tree, geometry, pick.outputs["Index"], "pick", 'INT', 600)
    geometry = store(tree, geometry, sit.outputs["Answer"], "sit", 'FLOAT', 800)
    geometry = store(tree, geometry, group.outputs["Cold"], "cold", 'FLOAT', 1000)
    geometry = store(tree, geometry, lamp.outputs["Index"], "lamp_index", 'INT', 1200)
    realize = tree.nodes.new("GeometryNodeRealizeInstances")
    tree.links.new(lamp.outputs["Instance"], realize.inputs["Geometry"])
    join = tree.nodes.new("GeometryNodeJoinGeometry")
    tree.links.new(realize.outputs[0], join.inputs[0])
    tree.links.new(geometry, join.inputs[0])
    tree.links.new(join.outputs[0], g_out.inputs["Geometry"])

    ids = {item.name: item.identifier for item in tree.interface.items_tree
           if getattr(item, "in_out", "") == 'INPUT'}
    for key, (description, prompt) in SCENES.items():
        mesh = bpy.data.meshes.new(key)
        mesh.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)], [], [])
        obj = bpy.data.objects.new(key, mesh)
        scene.collection.objects.link(obj)
        mod = obj.modifiers.new("OpenAct", 'NODES')
        mod.node_group = tree
        mod[ids["Scene"]] = description
        mod[ids["Pick Prompt"]] = prompt
        obj.update_tag()


def read(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        values = {"verts": len(mesh.vertices)}
        for name in ("night", "night_yes", "pick", "sit", "cold", "lamp_index"):
            attribute = mesh.attributes.get(name)
            values[name] = attribute.data[0].value if attribute else None
        return values
    finally:
        evaluated.to_mesh_clear()


def verify(tag):
    a, b = read(bpy.data.objects["A"]), read(bpy.data.objects["B"])
    print(f"{tag} A:", a, flush=True)
    print(f"{tag} B:", b, flush=True)
    lamp_verts = PROPS["Floor Lamp"]
    check(f"{tag}: A night is high", a["night"] is not None and a["night"] > 0.9, a["night"])
    check(f"{tag}: B night is low", b["night"] is not None and b["night"] < 0.1, b["night"])
    check(f"{tag}: Yes output follows Answer", a["night_yes"] is True and b["night_yes"] is False)
    check(f"{tag}: A picks armchair (2)", a["pick"] == 2, a["pick"])
    check(f"{tag}: B picks bookshelf (3)", b["pick"] == 3, b["pick"])
    check(f"{tag}: chained Ask - armchair is sittable", a["sit"] is not None and a["sit"] > 0.8, a["sit"])
    check(f"{tag}: chained Ask - bookshelf is not", b["sit"] is not None and b["sit"] < 0.5, b["sit"])
    check(f"{tag}: nested Ask - snowy night colder than beach",
          a["cold"] is not None and b["cold"] is not None and a["cold"] > 0.6 > b["cold"], (a["cold"], b["cold"]))
    check(f"{tag}: collection Pick index = Floor Lamp (2)", a["lamp_index"] == 2 and b["lamp_index"] == 2,
          (a["lamp_index"], b["lamp_index"]))
    check(f"{tag}: Instance output is the Floor Lamp mesh", a["verts"] == 4 + lamp_verts == b["verts"],
          (a["verts"], b["verts"]))


def all_answered(engine):
    """Nodes of the scene's trees must be answered; nodes in unused copies may show a hint instead
    (an unused tree has empty default inputs, e.g. no prompt)."""
    nodes = list(engine.iter_nodes())
    if not nodes or engine.PENDING:
        return False
    for tree, node in nodes:
        icon = engine.STATUS.get(node.uid, ('', ''))[0]
        allowed = ('CHECKMARK',) if tree.name in ("OpenActTest", "OpenActNested") else ('CHECKMARK', 'INFO')
        if icon not in allowed:
            return False
    return True


def dump_status(engine, server):
    print("server:", server.status(), flush=True)
    for tree, node in engine.iter_nodes():
        print(f"  {tree.name}/{node.name}: {engine.STATUS.get(node.uid)}", flush=True)


def phase_build(path, port, gpu):
    engine, server, P = enable_addon()
    prefs = P.get_prefs()
    prefs.port, prefs.gpu = port, gpu
    check("runtime detected", P.runtime_installed(P.runtime_root(prefs)), P.runtime_root(prefs))
    build_scene()
    started = time.time()
    ready_at = None

    def done():
        nonlocal ready_at
        if ready_at is None and server.is_ready():
            ready_at = time.time() - started
        return all_answered(engine)

    finished = drive(engine, done, timeout=300)
    total = time.time() - started
    dump_status(engine, server)
    check("all nodes answered", finished, f"server ready after {ready_at}s, all answered after {total:.1f}s")
    if ready_at is not None:
        print(f"TIMING model start {ready_at:.1f}s, answers {total - ready_at:.1f}s after ready", flush=True)
    verify("live")

    # the same node tree, a new prompt on one object: only that object changes
    tree = bpy.data.node_groups["OpenActTest"]
    ident = next(i.identifier for i in tree.interface.items_tree if getattr(i, "name", "") == "Pick Prompt")
    obj = bpy.data.objects["A"]
    obj.modifiers["OpenAct"][ident] = "something to lie on the floor"
    obj.update_tag()
    engine.schedule(immediate=True)
    t0 = time.time()
    drive(engine, lambda: all_answered(engine) and read(obj)["pick"] == 4, timeout=60)
    check("edit: new prompt re-asks and updates (rug = 4)", read(obj)["pick"] == 4,
          f"{read(obj)['pick']} after {time.time() - t0:.2f}s")
    check("edit: other object unchanged", read(bpy.data.objects["B"])["pick"] == 3)
    obj.modifiers["OpenAct"][ident] = SCENES["A"][1]  # switch back: answer is remembered, no model call
    obj.update_tag()
    engine.schedule(immediate=True)
    t0 = time.time()
    drive(engine, lambda: read(obj)["pick"] == 2, timeout=10)
    check("edit: switching back is instant", read(obj)["pick"] == 2, f"{time.time() - t0:.2f}s")
    requests = server.STATE.health.get("requests")
    print("server requests so far:", requests, flush=True)

    # duplicate the whole tree (like making a node group single-user): the original keeps its hidden
    # group and answers, the copy gets its own group with the same answers
    def night_node(t):
        return next(n for n in t.nodes if n.bl_idname == "GeometryNodeOpenActAsk" and
                    n.inputs["Question"].default_value == "Is it night?")

    night = night_node(tree)
    uid_before, inner_before = night.uid, night.node_tree.name
    copy_tree = tree.copy()
    copy_tree.name = "OpenActTest Copy"
    copy_tree.use_fake_user = True  # keep it in the saved file so reopening checks it too
    engine.run_pass()
    twin = night_node(copy_tree)
    check("copy: original keeps its uid and hidden group",
          night.uid == uid_before and night.node_tree is not None and night.node_tree.name == inner_before,
          (night.uid, getattr(night.node_tree, "name", None)))
    check("copy: duplicated node owns its hidden group",
          twin.node_tree is not None and twin.node_tree != night.node_tree and twin.uid not in ("", night.uid),
          (twin.uid, getattr(twin.node_tree, "name", None)))
    check("copy: the copy starts with the same answers", engine.get_entries(twin) == engine.get_entries(night))
    throwaway = tree.copy()
    engine.run_pass()
    bpy.data.node_groups.remove(throwaway)
    engine.run_pass()
    check("copy: deleting a copy leaves the original intact",
          night.node_tree is not None and night.node_tree.name == inner_before)
    verify("after copies")
    check("copy: the unused copy settles", drive(engine, lambda: all_answered(engine), timeout=60))

    bpy.ops.wm.save_as_mainfile(filepath=path)
    print("saved", path, flush=True)
    server.stop()
    check("server stopped", not server.is_ready())


def inner_groups_used():
    """Every OpenAct node in the file (original and copy) must point at its own saved hidden group."""
    nodes = [n for t in bpy.data.node_groups if not t.name.startswith(".OpenAct ") for n in t.nodes
             if n.bl_idname in ("GeometryNodeOpenActAsk", "GeometryNodeOpenActPick", "NodeUndefined")]
    inners = [t for t in bpy.data.node_groups if t.name.startswith(".OpenAct ")]
    return len(nodes), sum(1 for t in inners if t.users > 0)


def phase_noaddon(path):
    bpy.ops.wm.open_mainfile(filepath=path)
    kinds = {n.bl_idname for n in bpy.data.node_groups["OpenActTest"].nodes}
    check("without the add-on the nodes are undefined", "NodeUndefined" in kinds, kinds)
    count, used = inner_groups_used()
    check("every saved OpenAct node has its own hidden group", count == 9 and used == 9, (count, used))
    verify("no add-on")


def phase_reopen(path):
    engine, server, P = enable_addon()
    prefs = P.get_prefs()
    prefs.port, prefs.auto_start = 18799, False  # nothing listens there: no model available
    bpy.ops.wm.open_mainfile(filepath=path)
    engine.schedule(immediate=True)
    drive(engine, lambda: all_answered(engine), timeout=15)
    dump_status(engine, server)
    check("reopen: every node answered from the file", all_answered(engine))
    check("reopen: no questions were sent", not engine.PENDING)
    count, used = inner_groups_used()
    check("reopen: every OpenAct node has its own hidden group", count == 9 and used == 9, (count, used))
    verify("reopen")


def main():
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    phase = args[0] if args else "build"
    path = args[1] if len(args) > 1 else os.path.join(os.environ.get("TEMP", "/tmp"), "openact_e2e.blend")
    print("BLENDER", bpy.app.version_string, "phase", phase, flush=True)
    if phase == "build":
        phase_build(path, int(args[2]) if len(args) > 2 else 18767, int(args[3]) if len(args) > 3 else 0)
    elif phase == "noaddon":
        phase_noaddon(path)
    elif phase == "reopen":
        phase_reopen(path)
    finish()


main()
