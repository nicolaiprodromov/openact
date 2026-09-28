"""Builds the six demo scenes of the promo video (design §18.4) headlessly: one .blend each, next to its CSV files.

  blender -b --factory-startup --python demos/build_demos.py -- [<out folder>] [--model [PORT GPU]]
  blender -b --factory-startup --python demos/build_demos.py -- [<out folder>] --noaddon

The out folder defaults to demos/. The build copies each demo's CSV files from demos/ into it and saves the demo there
as <name>.blend. With --model the add-on starts the local model on PORT and GPU, by default on the add-on's own port
18765 and GPU 0: give the port and GPU of a server no other Blender uses, since the checks count its requests. Every
table fills, the demo's acceptance checks run, and the answers are saved in the file; without --model nothing is
asked. The build also writes evaluated.json: what every demo object evaluates to, attribute by attribute. --noaddon
opens the six saved demos in a Blender without OpenAct and checks that each evaluates to exactly that (G6). Lines as in
tests/_harness.py: PASS or FAIL per check, TIMING, and one RESULT line, with exit code 1 when anything failed.
"""

import collections
import hashlib
import itertools
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
import traceback

import bmesh
import bpy
from bpy_extras import anim_utils
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tests"))
from _harness import check, drive, enable_addon, finish, sock  # noqa: E402

USAGE = ("usage: blender -b --factory-startup --python demos/build_demos.py -- [<out folder>] "
         "[--model [PORT GPU] | --noaddon]")
ASK, PICK, DIALS = "GeometryNodeOpenActAsk", "GeometryNodeOpenActPick", "GeometryNodeOpenActDials"
MATCH, CLASSIFY, RULES = "GeometryNodeOpenActMatch", "GeometryNodeOpenActClassify", "GeometryNodeOpenActRules"


def arguments(args):
    """(out folder, model, noaddon) from the arguments after "--". model: None without --model, () for the add-on's
    default port and GPU, else (port, GPU). Raises ValueError for an argument it can't use."""
    rest, out, model, noaddon = list(args), HERE, None, False
    if rest and not rest[0].startswith("--"):
        out = os.path.abspath(rest.pop(0))
    while rest:
        option = rest.pop(0)
        if option == "--noaddon" and not noaddon:
            noaddon = True
        elif option == "--model" and model is None:
            values = []
            while rest and not rest[0].startswith("--"):
                values.append(rest.pop(0))
            if values and not (len(values) == 2 and all(value.isdigit() for value in values)
                               and 1024 <= int(values[0]) <= 65535 and int(values[1]) <= 15):
                raise ValueError(f"--model takes a PORT (1024-65535) and a GPU (0-15), or neither: got {values}")
            model = tuple(int(value) for value in values)
        else:
            raise ValueError(f"can't use {option!r}")
    if noaddon and model is not None:
        raise ValueError("--noaddon runs without the add-on, so without --model")
    return out, model, noaddon


try:
    OUT, MODEL, NOADDON = arguments(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
except ValueError as error:
    print(USAGE, flush=True)
    check("arguments", False, error)
    finish()
EVALUATED = os.path.join(OUT, "evaluated.json")
engine = None  # the add-on's engine, once it is enabled; --noaddon sees the nodes as a Blender without OpenAct does
if not NOADDON:
    try:
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # so the server's GPU index is the one nvidia-smi shows
        engine, server, P = enable_addon()
        engine.allow_background_passes(True)  # headless authoring: fill and save the tables (7.8)
        from openact import natsort, pipeline  # noqa: E402
        from openact import store as S  # noqa: E402
        from openact.specs.base import complete  # noqa: E402
    except Exception:
        traceback.print_exc()
        check("the add-on enables", False)
        finish()


# ------------------------------------------------------------------------------------------------ building blocks

def node(tree, idname, x, y, **settings):
    """A new node at (x, y), with these properties set."""
    made = tree.nodes.new(idname)
    made.location = (x, y)
    for name, value in settings.items():
        setattr(made, name, value)
    return made


def enabled(sockets, name):
    return next(s for s in sockets if s.name == name and s.enabled)


def feed(tree, socket, value):
    """Link an output into the socket, or give the socket a value."""
    if isinstance(value, bpy.types.NodeSocket):
        tree.links.new(value, socket)
    else:
        socket.default_value = value


def operation(tree, idname, op, x, y, *values):
    """The result of a math node (Integer Math, Math, Vector Math, Boolean Math) of these inputs."""
    made = node(tree, idname, x, y, operation=op)
    for socket, value in zip(made.inputs, values):
        feed(tree, socket, value)
    return made.outputs[0]


def integer(tree, op, x, y, *values):
    return operation(tree, "FunctionNodeIntegerMath", op, x, y, *values)


def number(tree, op, x, y, *values):
    return operation(tree, "ShaderNodeMath", op, x, y, *values)


def vector(tree, x, y, xyz):
    """A Combine XYZ of three numbers or sockets."""
    made = node(tree, "ShaderNodeCombineXYZ", x, y)
    for socket, value in zip(made.inputs, xyz):
        feed(tree, socket, value)
    return made.outputs[0]


def attribute(tree, name, data_type, x, y):
    made = node(tree, "GeometryNodeInputNamedAttribute", x, y, data_type=data_type)
    made.inputs["Name"].default_value = name
    return enabled(made.outputs, "Attribute")


def stored(tree, geometry, value, name, data_type, x, y, domain='POINT'):
    """The geometry with the value stored as a named attribute, which the checks and evaluated.json read."""
    made = node(tree, "GeometryNodeStoreNamedAttribute", x, y, data_type=data_type, domain=domain)
    made.inputs["Name"].default_value = name
    tree.links.new(geometry, made.inputs["Geometry"])
    feed(tree, enabled(made.inputs, "Value"), value)
    return made.outputs["Geometry"]


def unit(tree, value, low, high, x, y):
    """The value mapped from low-high onto 0-1, clamped (Map Range)."""
    made = node(tree, "ShaderNodeMapRange", x, y, clamp=True)
    feed(tree, made.inputs["Value"], value)
    made.inputs["From Min"].default_value, made.inputs["From Max"].default_value = low, high
    return made.outputs["Result"]


def randomly(tree, data_type, seed, x, y, **inputs):
    """A Random Value of this type, with these inputs (Min, Max, Probability)."""
    made = node(tree, "FunctionNodeRandomValue", x, y, data_type=data_type)
    made.inputs["Seed"].default_value = seed
    for name, value in inputs.items():
        enabled(made.inputs, name).default_value = value
    return enabled(made.outputs, "Value")


def primitive(tree, idname, x, y, **inputs):
    """A mesh primitive node's Mesh; an input name has _ for its spaces."""
    made = node(tree, idname, x, y)
    for name, value in inputs.items():
        made.inputs[name.replace("_", " ")].default_value = value
    return made.outputs["Mesh"]


def dropdown(tree, gi, name, choices, x, y):
    """The output of a String Menu Switch over choices ({item: text}), driven by the tree's menu input `name`."""
    made = node(tree, "GeometryNodeMenuSwitch", x, y, data_type='STRING')
    made.enum_items.clear()
    for item in choices:
        made.enum_items.new(item)
    for slot, text in zip([s for s in made.inputs if s.identifier not in ("Menu", "__extend__")], choices.values()):
        slot.default_value = text
    tree.links.new(gi.outputs[name], made.inputs["Menu"])
    return made.outputs[0]


def placed(tree, points, instance, x, y, selection=None, index=None, rotation=None, scale=None):
    """Instance on Points; with index, Pick Instance picks each point's child at that index."""
    made = node(tree, "GeometryNodeInstanceOnPoints", x, y)
    tree.links.new(points, made.inputs["Points"])
    tree.links.new(instance, made.inputs["Instance"])
    made.inputs["Pick Instance"].default_value = index is not None
    for name, value in (("Selection", selection), ("Instance Index", index), ("Rotation", rotation), ("Scale", scale)):
        if value is not None:
            feed(tree, made.inputs[name], value)
    return made.outputs["Instances"]


def switch(tree, condition, false, true, x, y):
    """A geometry Switch; None leaves that side empty."""
    made = node(tree, "GeometryNodeSwitch", x, y, input_type='GEOMETRY')
    feed(tree, made.inputs["Switch"], condition)
    for name, value in (("False", false), ("True", true)):
        if value is not None:
            tree.links.new(value, made.inputs[name])
    return made.outputs[0]


def chosen(tree, index, items, x, y):
    """An Index Switch over these vectors."""
    made = node(tree, "GeometryNodeIndexSwitch", x, y, data_type='VECTOR')
    while len(made.index_switch_items) < len(items):
        made.index_switch_items.new()
    for slot, value in zip([s for s in made.inputs if s.identifier not in ("Index", "__extend__")], items):
        slot.default_value = value
    feed(tree, made.inputs["Index"], index)
    return made.outputs[0]


def children(tree, collection, x, y):
    """Collection Info with Separate and Reset Children: an instance per child, in the order of the nodes' lists."""
    made = node(tree, "GeometryNodeCollectionInfo", x, y)
    made.inputs["Collection"].default_value = collection
    made.inputs["Separate Children"].default_value = made.inputs["Reset Children"].default_value = True
    return made.outputs[0]


def joined(tree, x, y, *geometries):
    made = node(tree, "GeometryNodeJoinGeometry", x, y)
    for geometry in reversed(geometries):  # Join Geometry puts the last linked first
        tree.links.new(geometry, made.inputs[0])
    return made.outputs[0]


def painted(tree, geometry, mat, x, y):
    made = node(tree, "GeometryNodeSetMaterial", x, y)
    tree.links.new(geometry, made.inputs["Geometry"])
    made.inputs["Material"].default_value = mat
    return made.outputs["Geometry"]


def new_tree(name, *inputs):
    """A Geometry Nodes tree with Geometry in and out, and these (name, socket type) inputs: (tree, Group Input,
    Group Output)."""
    tree = bpy.data.node_groups.new(name, 'GeometryNodeTree')
    tree.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    for input_name, socket_type in inputs:
        tree.interface.new_socket(input_name, in_out='INPUT', socket_type=socket_type)
    tree.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    return tree, node(tree, "NodeGroupInput", -1900, 0), node(tree, "NodeGroupOutput", 1900, 0)


def ident(tree, name):
    return next(item.identifier for item in tree.interface.items_tree
                if getattr(item, "in_out", "") == 'INPUT' and item.name == name)


def user(name, tree, mesh=None, location=(0.0, 0.0, 0.0), **inputs):
    """An object whose modifier "OpenAct" uses the tree, with these modifier inputs (an input name has _ for its
    spaces)."""
    obj = bpy.data.objects.new(name, mesh or bpy.data.meshes.new(name))
    obj.location = location
    bpy.context.scene.collection.objects.link(obj)
    modifier = obj.modifiers.new("OpenAct", 'NODES')
    modifier.node_group = tree
    for word, value in inputs.items():
        modifier[ident(tree, word.replace("_", " "))] = value
    return obj


def modify(obj, name, value):
    """Set a modifier input, as the artist does. With the add-on, the pass that edit brings is scheduled without its
    0.25 s debounce: §3.2 times new answers from after it."""
    modifier = obj.modifiers["OpenAct"]
    modifier[ident(modifier.node_group, name)] = value
    obj.update_tag()
    if engine is not None:
        engine.schedule(immediate=True)


def menu_item(obj, name, item):
    """The modifier's value for an item of its menu input: the item's identifier, from the modifier's item list."""
    modifier = obj.modifiers["OpenAct"]
    items = modifier.id_properties_ui(ident(modifier.node_group, name)).as_dict()["items"]
    return next(entry[-1] for entry in items if entry[1] == item)


def choose(obj, name, item):
    """Pick an item of the object's menu input, as in the modifier's dropdown."""
    modify(obj, name, menu_item(obj, name, item))


def material(name, color, emission=0.0, reads=(), ramp=""):
    """A Principled material of this colour. reads: (BSDF input, attribute) pairs it takes from the geometry; ramp: an
    attribute whose 0-1 blends the colour into white."""
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    if mat.node_tree is None:  # Blender 4.5; 5.x gives every material its nodes
        mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = next((n for n in nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"), None) or nodes.new(
        "ShaderNodeBsdfPrincipled")
    output = next((n for n in nodes if n.bl_idname == "ShaderNodeOutputMaterial"), None) or nodes.new(
        "ShaderNodeOutputMaterial")
    links.new(bsdf.outputs[0], output.inputs["Surface"])
    bsdf.inputs["Base Color"].default_value = color
    if emission:
        bsdf.inputs["Emission Color"].default_value = color
        bsdf.inputs["Emission Strength"].default_value = emission
    for socket, attribute_name in reads:
        reader = nodes.new("ShaderNodeAttribute")
        reader.attribute_name = attribute_name
        links.new(reader.outputs["Fac"], bsdf.inputs[socket])
    if ramp:
        reader, blend = nodes.new("ShaderNodeAttribute"), nodes.new("ShaderNodeValToRGB")
        reader.attribute_name = ramp
        blend.color_ramp.elements[0].color, blend.color_ramp.elements[1].color = color, (0.95, 0.95, 1.0, 1.0)
        links.new(reader.outputs["Fac"], blend.inputs["Fac"])
        links.new(blend.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def fog_material():
    """A volume material: the Volume Cube's density grid is the fog."""
    mat = bpy.data.materials.new("Fog")
    if mat.node_tree is None:
        mat.use_nodes = True
    nodes = mat.node_tree.nodes
    for surface in [n for n in nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"]:
        nodes.remove(surface)
    output = next((n for n in nodes if n.bl_idname == "ShaderNodeOutputMaterial"), None) or nodes.new(
        "ShaderNodeOutputMaterial")
    mat.node_tree.links.new(nodes.new("ShaderNodeVolumePrincipled").outputs[0], output.inputs["Volume"])
    return mat


def solid(name, parts, color):
    """A mesh of primitives in one material of this colour; parts: (box, ball, rod or cone, radius, height, height of
    its base)."""
    bm = bmesh.new()
    for kind, radius, height, base in parts:
        if kind == "box":
            verts = bmesh.ops.create_cube(bm, size=1.0)["verts"]
        elif kind == "ball":
            verts = bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=8, radius=0.5)["verts"]
        else:
            verts = bmesh.ops.create_cone(bm, cap_ends=True, segments=16, radius1=radius,
                                          radius2=0.0 if kind == "cone" else radius, depth=height)["verts"]
        if kind in ("box", "ball"):
            bmesh.ops.scale(bm, vec=(2 * radius, 2 * radius, height), verts=verts)
        bmesh.ops.translate(bm, vec=(0.0, 0.0, base + height / 2), verts=verts)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    mesh.materials.append(material(name, color))
    return mesh


def plane(name, size):
    half = size / 2
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([(-half, -half, 0.0), (half, -half, 0.0), (half, half, 0.0), (-half, half, 0.0)], [], [(0, 1, 2, 3)])
    return mesh


def hidden_collection(name, meshes):
    """A collection of an object per (name, mesh), excluded from the view layer: Collection Info instances its
    children, and the nodes list them."""
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    for object_name, mesh in meshes:
        collection.objects.link(bpy.data.objects.new(object_name, mesh))
    bpy.context.view_layer.layer_collection.children[name].exclude = True
    return collection


# ------------------------------------------------------------------------------------------ what objects evaluate to

WIDTH = {"FLOAT_VECTOR": 3, "FLOAT_COLOR": 4, "BYTE_COLOR": 4, "FLOAT2": 2, "INT32_2D": 2, "QUATERNION": 4,
         "FLOAT4X4": 16}
FIELD = {"FLOAT_VECTOR": "vector", "FLOAT_COLOR": "color", "BYTE_COLOR": "color", "FLOAT2": "vector"}


def attribute_values(item):
    """An attribute's values, flat."""
    if item.data_type == 'STRING':
        return [datum.value for datum in item.data]
    found = [0] * (len(item.data) * WIDTH.get(item.data_type, 1))
    item.data.foreach_get(FIELD.get(item.data_type, "value"), found)
    return found


def evaluated(obj):
    """The object's evaluated GeometrySet, which must stay referenced while its parts are read."""
    return bpy.types.GeometrySet.from_evaluated_object(obj.evaluated_get(bpy.context.evaluated_depsgraph_get()))


def values(obj, name):
    """A named attribute of the object's evaluated mesh, one value per element."""
    geometry = evaluated(obj)
    return attribute_values(geometry.mesh.attributes[name])


def instances(obj, names):
    """(what it instances, {attribute: value}) of each instance the object evaluates to."""
    geometry = evaluated(obj)
    points, references = geometry.instances_pointcloud(), geometry.instance_references()
    if points is None:
        return []
    columns = [attribute_values(points.attributes[name]) for name in (".reference_index", *names)]
    return [(references[row[0]].name, dict(zip(names, row[1:]))) for row in zip(*columns)]


def digest(obj):
    """What the object evaluates to: the size of each part of its geometry, a SHA-1 of each attribute's values
    (without Blender's own "." ones, but for which child an instance is), and what its instances instance. An
    instance's position is in its instance_transform: the point cloud of the instances leaves position unset (zeros on
    4.5, whatever memory held on 5.1)."""
    geometry = evaluated(obj)
    parts = {"mesh": geometry.mesh, "pointcloud": geometry.pointcloud, "curves": geometry.curves,
             "instances": geometry.instances_pointcloud()}
    found = {"references": [ref.name if isinstance(ref, bpy.types.ID) else "geometry"
                            for ref in geometry.instance_references()]}
    for part, data in parts.items():
        if data is None:
            continue
        found[part] = ([len(data.vertices), len(data.edges), len(data.polygons)] if part == "mesh" else
                       [len(data.points), len(data.curves)] if part == "curves" else [len(data.points)])
        for item in data.attributes:
            if part == "instances" and item.name == "position":
                continue
            if not item.name.startswith(".") or item.name == ".reference_index":
                found[f"{part} {item.domain} {item.name}"] = hashlib.sha1(
                    repr(attribute_values(item)).encode()).hexdigest()
    return found


def record(frames, menu=None):
    """digest() of each object with a Geometry Nodes modifier at these frames, set in order, once for each item of the
    demo's menu input, picked as by hand (menu: (object, input, items)): {item: {frame: {object: digest}}}, with the
    item "" when there's no menu."""
    scene = bpy.context.scene
    users = [obj for obj in scene.objects if any(modifier.type == 'NODES' for modifier in obj.modifiers)]
    found = {}
    for item in menu[2] if menu else ("",):
        if item:
            choose(bpy.data.objects[menu[0]], menu[1], item)
        found[item] = {}
        for frame in frames:
            scene.frame_set(frame)
            found[item][frame] = {obj.name: digest(obj) for obj in users}
    return found


# ------------------------------------------------------------------------------------------------ the passes

def down():
    """Why the model can't answer, or "" while it can: its server reports an error, or the runtime isn't installed."""
    state, message = server.status()
    return (message or state) if state in ("error", "not_installed") else ""


def wait(done, timeout):
    """Run the passes until done(), which it returns, or until the model can't answer (down()), or timeout."""
    drive(engine, lambda: done() or bool(down()), timeout)
    return done()


def answering(name, done):
    """done; else a FAIL that says why the demo's tables didn't fill, as its checks need them and are skipped."""
    if not done:
        check(f"{name}: the model answers", False, down() or "the tables didn't fill in time")
    return done


def answered():
    """Every OpenAct node shows whole tables for all its contexts and nothing is asked: each status is complete."""
    nodes = [n for _tree, n in engine.iter_nodes()]
    if not nodes or pipeline.PENDING:
        return False
    for n in nodes:
        status = engine.STATUS.get(n.uid)
        if status is None or status.state != "complete" or not status.uses:
            return False
        if any(entry is None or not complete(entry) for _label, _job, entry in status.uses):
            return False
    return True


def fill(what, timeout=180):
    """Run the passes until answered(), timed: True once it is. It stops as soon as the model can't answer. Without
    --model one pass plans, and nothing is asked."""
    engine.schedule(immediate=True)
    if MODEL is None:
        engine.run_pass()
        return False
    started = time.time()
    done = wait(answered, timeout)
    print(f"TIMING {what}: {time.time() - started:.1f}s", flush=True)
    if not done:
        print(f"  model server: {server.status()}", flush=True)
        for tree, n in engine.iter_nodes():
            status = engine.STATUS.get(n.uid)
            print(f"  {tree.name}/{n.name}: {status and (status.state, status.text)}", flush=True)
    return done


def requests():
    """How many requests the model server has answered (its /health): the "0 requests" checks compare it."""
    try:
        code, health = server.http("GET", "/health", timeout=5.0)
    except OSError:
        return None
    return health.get("requests") if code == 200 else None


def sent_since(before):
    """Requests answered since `before` (requests()), or None when the server didn't say."""
    after = requests()
    return None if before is None or after is None else after - before


def whole(name):
    """Every table of every OpenAct node is complete, with no pending or stale rows: a "0 requests" check can't pass
    on an empty table."""
    found = {}
    for _tree, n in engine.iter_nodes():
        entries = S.load(n).entries
        found[n.name] = (len(entries), sum(not complete(entry) for entry in entries))
    check(f"{name}: every table is complete (no pending or stale rows)",
          bool(found) and all(count and not partial for count, partial in found.values()),
          "; ".join(f"{key}: {count} tables, {partial} partial" for key, (count, partial) in found.items()))


# ------------------------------------------------------------------------------------------------ life (the hero)

GOL_QUESTION = "What happens to this cell in the next generation?"
GOL_CONTEXT = "A cellular automaton on a square grid; every cell has eight neighbours."
SIDE = 64  # cells per side of the toroidal grid
RULE_SETS = {  # the Rule set dropdown: each item's Outcomes CSV, and its birth and survival neighbour counts
    "Conway": ("life_outcomes.csv", {3}, {2, 3}),
    "HighLife": ("highlife_outcomes.csv", {3, 6}, {2, 3}),
    "Seeds": ("seeds_outcomes.csv", {2}, set()),
    "Day & Night": ("daynight_outcomes.csv", {3, 6, 7, 8}, {3, 4, 6, 7, 8}),
}
LOCKED = 9 + 5  # the cell life_overrides.csv locks, alive with five neighbours: it dies under every rule set
KEYS = {1: "Conway", 31: "HighLife", 61: "Seeds", 91: "Day & Night"}  # the saved dropdown's keyframes
LAST = 120
OFFSETS = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
NEIGHBOURS = [[(i // SIDE + dr) % SIDE * SIDE + (i % SIDE + dc) % SIDE for dr, dc in OFFSETS]
              for i in range(SIDE * SIDE)]


def outcome(alive, count, birth, survive):
    if alive:
        return "Survives" if count in survive else "Dies"
    return "Birth" if count in birth else "Stays empty"


def generation(cells, birth, survive):
    """One step of the Python reference, on the same toroidal grid."""
    return [int(sum(cells[k] for k in NEIGHBOURS[i]) in (survive if alive else birth)) for i, alive in enumerate(cells)]


def reference(start, first, last, rule_set):
    """The Python reference for frames 1 to last, each frame stepped with the rules of rule_set(frame). At frame 1 the
    zone shows the start pattern or steps it once: first, what it shows there, tells which."""
    stepped = generation(start, *RULE_SETS[rule_set(1)][1:])
    cells = stepped if first == stepped else start
    frames = [cells]
    for frame in range(2, last + 1):
        cells = generation(cells, *RULE_SETS[rule_set(frame)][1:])
        frames.append(cells)
    return frames


def generations(obj, last, playing=False):
    """alive at frames 1 to last, set in order, and the seconds each frame took to evaluate. playing: as during
    playback, where the engine ticks every frame and runs no pass."""
    scene, frames, seconds = bpy.context.scene, [], []
    for frame in range(1, last + 1):
        started = time.perf_counter()
        scene.frame_set(frame)
        seconds.append(time.perf_counter() - started)
        if playing:
            engine.tick()
        frames.append(values(obj, "alive"))
    return frames, seconds


def life_zone(tree, rules, gi, go):
    """In a Simulation Zone: state = alive > 0, neighbours = 8 Sample Index offsets on the toroidal grid (Integer Math
    FLOORED_MODULO on row and column, §6.5), and the Rules payload output 'alive' stored back. A cube shows each live
    cell."""
    sim_in, sim_out = node(tree, "GeometryNodeSimulationInput", -1600, -300), node(
        tree, "GeometryNodeSimulationOutput", 800, -300)
    sim_in.pair_with_output(sim_out)
    tree.links.new(gi.outputs["Geometry"], sim_in.inputs["Geometry"])
    cells = sim_in.outputs["Geometry"]
    alive = attribute(tree, "alive", 'INT', -1400, -500)
    state = node(tree, "FunctionNodeCompare", -600, -150, data_type='INT', operation='GREATER_THAN')
    tree.links.new(alive, sock(state.inputs, "A_INT"))
    tree.links.new(state.outputs[0], rules.inputs["state"])
    index = node(tree, "GeometryNodeInputIndex", -1600, -800).outputs[0]
    row = integer(tree, 'DIVIDE_FLOOR', -1400, -800, index, SIDE)
    column = integer(tree, 'FLOORED_MODULO', -1400, -950, index, SIDE)

    def wrapped(value, offset, y):
        return value if not offset else integer(tree, 'FLOORED_MODULO', -1000, y, integer(
            tree, 'ADD', -1200, y, value, offset), SIDE)

    total = None
    for k, (dr, dc) in enumerate(OFFSETS):
        y = -800 - 220 * k
        sample = node(tree, "GeometryNodeSampleIndex", -500, y, data_type='INT', domain='POINT')
        tree.links.new(cells, sample.inputs["Geometry"])
        tree.links.new(alive, enabled(sample.inputs, "Value"))
        tree.links.new(integer(tree, 'MULTIPLY_ADD', -750, y, wrapped(row, dr, y), SIDE, wrapped(column, dc, y - 100)),
                       sample.inputs["Index"])
        count = enabled(sample.outputs, "Value")
        total = count if total is None else integer(tree, 'ADD', -300, y, total, count)
    tree.links.new(total, rules.inputs["neighbours"])
    tree.links.new(stored(tree, cells, rules.outputs["alive"], "alive", 'INT', 500, -300), sim_out.inputs["Geometry"])
    grid = sim_out.outputs["Geometry"]
    live = node(tree, "FunctionNodeCompare", 1100, -600, data_type='INT', operation='GREATER_THAN')
    tree.links.new(attribute(tree, "alive", 'INT', 900, -650), sock(live.inputs, "A_INT"))
    cube = primitive(tree, "GeometryNodeMeshCube", 1100, -800, Size=(0.08, 0.08, 0.08))
    tree.links.new(joined(tree, 1600, -300, grid, placed(tree, grid, cube, 1300, -600, selection=live.outputs[0])),
                   go.inputs["Geometry"])


def build_life():
    tree, gi, go = new_tree("Life", ("Rule set", 'NodeSocketMenu'))
    rules = node(tree, RULES, -300, 300)
    for word, text in (("Question", GOL_QUESTION), ("Context", GOL_CONTEXT), ("Factors", "//life_factors.csv"),
                       ("Overrides", "//life_overrides.csv")):
        rules.inputs[word].default_value = text
    tree.links.new(dropdown(tree, gi, "Rule set", {name: "//" + csv for name, (csv, _birth, _survive)
                                                   in RULE_SETS.items()}, -1600, 300), rules.inputs["Outcomes"])
    rng = random.Random(25)
    start = [int(rng.random() < 0.3) for _ in range(SIDE * SIDE)]
    mesh = bpy.data.meshes.new("Life")
    mesh.from_pydata([(i % SIDE * 0.1, i // SIDE * 0.1, 0.0) for i in range(SIDE * SIDE)], [], [])
    mesh.attributes.new("alive", 'INT', 'POINT').data.foreach_set("value", start)
    obj = user("Life", tree, mesh)
    choose(obj, "Rule set", "Conway")
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, LAST
    engine.run_pass()  # plans: the factor inputs and the payload output 'alive' appear
    life_zone(tree, rules, gi, go)
    done = fill("life Conway table, the model starting", 600)
    if MODEL is None or not answering("life", done):
        return

    for name, (csv, birth, survive) in RULE_SETS.items():
        if name != "Conway":  # shown already
            choose(obj, "Rule set", name)
            done = fill(f"life {name} table")
        entry = next((e for e in S.load(rules).entries if e.gn[3] == "//" + csv), None)
        names = entry.shape.get("outcomes", []) if entry else []
        table = [names[o] if 0 <= o < len(names) else "?" for o in entry.rows.v["outcome"]] if entry else []
        want = [outcome(alive, count, birth, survive) for alive in (0, 1) for count in range(9)]
        wrong = [(cell, got, right) for cell, (got, right) in enumerate(zip(table, want)) if got != right]
        exact = len(table) - len(wrong) if len(table) == len(want) else 0
        model = [c for src, c in zip(entry.rows.src, entry.rows.v["c"]) if src == "m"] if entry else []
        unsure = sum(c < P.get_prefs().unsure_below for c in model)
        check(f"life: {name}: 18/18 exact and 0 unsure, the locked cell a lock", done and exact == 18 and unsure == 0
              and entry.rows.src[LOCKED] == "l",
              f"{exact}/18 exact, {unsure} unsure, lowest confidence {min(model, default=0.0):.2f}; wrong {wrong}")

    before = requests()
    choose(obj, "Rule set", "Conway")
    started = time.time()
    done = wait(lambda: answered() and engine.STATUS[rules.uid].uses[0][2].gn[3] == "//life_outcomes.csv", 30)
    sent = sent_since(before)
    check("life: switching back to Conway is instant, with 0 requests", done and sent == 0,
          f"{time.time() - started:.2f}s, {sent} requests")

    frames, seconds = generations(obj, 32)
    want = reference(start, frames[0], 32, lambda frame: "Conway")
    wrong = [frame for frame, (got, right) in enumerate(zip(frames, want), start=1) if got != right]
    check("life: 32 frames in the Simulation Zone equal the Python B3/S23 reference (toroidal)", not wrong,
          f"wrong at frames {wrong[:5]}; {1000 * sum(seconds[1:]) / len(seconds[1:]):.1f} ms per frame")

    # The continuous shot through the four rule sets (video beat 2): the dropdown keyframed, then Prepare for Playback
    # pins their tables (§5.5)
    path = f'modifiers["OpenAct"]["{ident(tree, "Rule set")}"]'
    for frame, name in KEYS.items():
        choose(obj, "Rule set", name)
        obj.keyframe_insert(data_path=path, frame=frame)
    animation = obj.animation_data
    curve = anim_utils.action_get_channelbag_for_slot(animation.action, animation.action_slot).fcurves.find(path)
    for point in curve.keyframe_points:
        point.interpolation = 'CONSTANT'
    scene.frame_set(1)
    before = requests()
    result = bpy.ops.openact.prepare_playback('EXEC_DEFAULT')
    wait(lambda: not pipeline.PENDING and not engine.pinning() and not engine._dirty, 60)
    pins = sorted(entry.gn[3] for entry in S.load(rules).entries if entry.pinned)
    sent = sent_since(before)
    check("life: Prepare for Playback pins the four rule sets, with 0 requests", result == {'FINISHED'} and sent == 0
          and pins == sorted("//" + csv for csv, _birth, _survive in RULE_SETS.values()), (result, sent, pins))

    table, before = rules.entries, requests()
    obj.update_tag()  # a change that isn't time drops the zone's cached frames: each frame is computed as it plays
    playing, engine._animation_playing = engine._animation_playing, lambda: True
    try:
        frames, seconds = generations(obj, LAST, playing=True)
    finally:
        engine._animation_playing = playing
    want = reference(start, frames[0], LAST, lambda frame: KEYS[max(key for key in KEYS if key <= frame)])
    wrong = [frame for frame, (got, right) in enumerate(zip(frames, want), start=1) if got != right]
    sent = sent_since(before)
    check(f"life: playing frames 1-{LAST} sends 0 requests and writes no table, and every frame equals the Python "
          "reference as the dropdown switches rule sets", not wrong and sent == 0 and rules.entries == table,
          f"{sent} requests; wrong at frames {wrong[:5]}")
    fps = scene.render.fps / scene.render.fps_base
    frame_ms = 1000 * sum(seconds[1:]) / len(seconds[1:])
    print(f"TIMING life playback: {frame_ms:.1f} ms per frame, the slowest {1000 * max(seconds[1:]):.1f} ms (a frame "
          f"lasts {1000 / fps:.1f} ms at {fps:g} fps)", flush=True)
    if frame_ms > 1000 / fps:
        print(f"WARN life playback takes {frame_ms:.1f} ms per frame, above {1000 / fps:.1f} ms", flush=True)
    check("life: playback keeps up with the frame rate (fails above twice the frame time)", frame_ms <= 2000 / fps,
          f"{frame_ms:.1f} ms per frame")

    # Saved without the keys and the pins, so the dropdown stays where the artist switches it (the start of video beat
    # 2): by hand, each rule set shows its stored table and plays exactly, with no model
    action = obj.animation_data.action
    obj.animation_data_clear()
    bpy.data.actions.remove(action)
    engine.unpin(rules.uid)
    before, wrong, moved = requests(), {}, []
    for name in RULE_SETS:
        choose(obj, "Rule set", name)
        wait(lambda: answered() and engine.STATUS[rules.uid].uses[0][2].gn[3] == "//" + RULE_SETS[name][0], 30)
        frames, _seconds = generations(obj, 32)
        want = reference(start, frames[0], 32, lambda frame: name)
        wrong[name] = [frame for frame, (got, right) in enumerate(zip(frames, want), start=1) if got != right]
        if obj.modifiers["OpenAct"][ident(tree, "Rule set")] != menu_item(obj, "Rule set", name):
            moved.append(name)
    sent = sent_since(before)
    choose(obj, "Rule set", "Conway")
    scene.frame_set(1)
    done = fill("life saved with a free dropdown", 30)
    check("life: saved with no keys and no pins, the dropdown switched by hand stays put, and each rule set plays "
          "exactly (32 frames), with 0 requests", done and not any(wrong.values()) and not moved and sent == 0
          and obj.animation_data is None and not any(entry.pinned for entry in S.load(rules).entries),
          f"{sent} requests; wrong frames {wrong}; moved back {moved}")


# ------------------------------------------------------------------------------------------------ biome

PLANTS = (  # biome_outcomes.csv's rows, whose instance column counts in this order: name, parts, colour
    ("Pine", (("rod", 0.06, 0.3, 0.0), ("cone", 0.4, 1.4, 0.3)), (0.1, 0.3, 0.12, 1.0)),
    ("Birch", (("rod", 0.05, 1.0, 0.0), ("ball", 0.3, 0.8, 0.8)), (0.6, 0.75, 0.3, 1.0)),
    ("Oak", (("rod", 0.1, 0.5, 0.0), ("ball", 0.55, 0.9, 0.4)), (0.2, 0.45, 0.15, 1.0)),
    ("Palm", (("rod", 0.06, 1.5, 0.0), ("ball", 0.5, 0.2, 1.4)), (0.3, 0.6, 0.2, 1.0)),
    ("Cactus", (("rod", 0.15, 0.9, 0.0),), (0.3, 0.55, 0.3, 1.0)),
    ("Olive", (("rod", 0.08, 0.4, 0.0), ("ball", 0.35, 0.5, 0.35)), (0.45, 0.5, 0.35, 1.0)),
    ("Heather", (("ball", 0.3, 0.25, 0.0),), (0.55, 0.3, 0.55, 1.0)),
)
LAND, SPAN, PEAK = 48, 24.0, 6.0  # terrain vertices per side, metres per side, metres at its top
TEMPERATURE, RAIN = (-10.0, 35.0), (0.0, 3000.0)  # the terrain's °C and mm of rain a year: Map Range makes them 0-1
OUTCOMES = {"Descriptions": "biome_outcomes.csv", "Bare names": "biome_outcomes_bare.csv"}  # the A/B dropdown


def land(x, y):
    """(height, temperature, rain) at (x, y): a mountain to the north-east and hills to the north-west; warm in the
    south, cold in the north and colder uphill; dry in the west, wet in the east. So every plant has ground, the
    hills' cold and dry tops too."""
    height = PEAK * math.exp(-((x - 5) ** 2 + (y - 5) ** 2) / 30) + 2.5 * math.exp(-((x + 6) ** 2 + (y - 7) ** 2) / 20)
    return height, 33.0 - 1.1 * (y + SPAN / 2) - 5.0 * height, 100.0 + 2800.0 * (x + SPAN / 2) / SPAN


def build_biome():
    plants = hidden_collection("Plants", [(f"{i} {name}", solid(name, parts, color))  # sorted as the payload counts
                                          for i, (name, parts, color) in enumerate(PLANTS)])
    tree, gi, go = new_tree("Biome", ("Plant descriptions", 'NodeSocketMenu'))
    position = node(tree, "ShaderNodeSeparateXYZ", -1600, -400)
    tree.links.new(node(tree, "GeometryNodeInputPosition", -1800, -400).outputs[0], position.inputs[0])
    factors = {"temperature": unit(tree, attribute(tree, "temperature", 'FLOAT', -1700, 0), *TEMPERATURE, -1400, 0),
               "moisture": unit(tree, attribute(tree, "rain", 'FLOAT', -1700, -200), *RAIN, -1400, -200),
               "altitude": unit(tree, position.outputs["Z"], 0.0, PEAK, -1400, -400)}
    kind, snow = node(tree, RULES, -1000, 400), node(tree, RULES, -1000, -500, answer_type='AMOUNT')
    kind.inputs["Question"].default_value = "Which plant grows here?"
    snow.inputs["Question"].default_value = "How much snow covers the ground?"
    for rules in (kind, snow):
        rules.inputs["Factors"].default_value = "//biome_factors.csv"
    tree.links.new(dropdown(tree, gi, "Plant descriptions", {item: "//" + csv for item, csv in OUTCOMES.items()},
                            -1400, 700), kind.inputs["Outcomes"])
    kind.inputs["Dither"].default_value = 0.3
    coords, temperature, rain = [], [], []
    for j in range(LAND):
        for i in range(LAND):
            x, y = SPAN * (i / (LAND - 1) - 0.5), SPAN * (j / (LAND - 1) - 0.5)
            height, degrees, millimetres = land(x, y)
            coords.append((x, y, height))
            temperature.append(degrees)
            rain.append(millimetres)
    mesh = bpy.data.meshes.new("Terrain")
    mesh.from_pydata(coords, [], [(j * LAND + i, j * LAND + i + 1, (j + 1) * LAND + i + 1, (j + 1) * LAND + i)
                                  for j in range(LAND - 1) for i in range(LAND - 1)])
    mesh.attributes.new("temperature", 'FLOAT', 'POINT').data.foreach_set("value", temperature)
    mesh.attributes.new("rain", 'FLOAT', 'POINT').data.foreach_set("value", rain)
    terrain = user("Terrain", tree, mesh)
    choose(terrain, "Plant descriptions", "Descriptions")
    engine.run_pass()  # plans: the factor inputs, and the payload outputs 'instance' and 'scale', appear
    for rules in (kind, snow):
        for name, value in factors.items():
            tree.links.new(value, rules.inputs[name])
    ground = stored(tree, gi.outputs["Geometry"], snow.outputs["Amount"], "snow", 'FLOAT', -500, 500)
    ground = painted(tree, ground, material("Terrain", (0.3, 0.38, 0.18, 1.0), ramp="snow"), -300, 500)
    spots = node(tree, "GeometryNodeMeshToPoints", -800, -900)
    tree.links.new(gi.outputs["Geometry"], spots.inputs["Mesh"])
    moved = node(tree, "GeometryNodeSetPosition", -600, -900)
    tree.links.new(spots.outputs[0], moved.inputs["Geometry"])
    feed(tree, moved.inputs["Offset"], randomly(tree, 'FLOAT_VECTOR', 1, -800, -1100, Min=(-0.2, -0.2, 0.0),
                                                Max=(0.2, 0.2, 0.0)))
    turn = vector(tree, -400, -1300, (0.0, 0.0, randomly(tree, 'FLOAT', 2, -600, -1300, Min=0.0, Max=math.tau)))
    trees = placed(tree, moved.outputs[0], children(tree, plants, -400, -700), -100, -900,
                   selection=randomly(tree, 'BOOLEAN', 3, -400, -1100, Probability=0.35),
                   index=kind.outputs["instance"], rotation=turn, scale=kind.outputs["scale"])
    tree.links.new(joined(tree, 400, 0, ground, trees), go.inputs["Geometry"])
    done = fill("biome tables")
    if MODEL is None or not answering("biome", done):
        return

    def table(item):
        """(mean confidence, the outcomes used) of the plants' table with that item's Outcomes CSV."""
        entry = next((e for e in S.load(kind).entries if e.gn[3] == "//" + OUTCOMES[item]), None)
        confidences = entry.rows.v["c"] if entry else []
        return (sum(confidences) / len(confidences) if confidences else 0.0), set(entry.rows.v["outcome"] if entry
                                                                                  else ())

    mean, used = table("Descriptions")
    check("biome: mean confidence at least 0.55, with all 7 plants used", mean >= 0.55
          and used == set(range(len(PLANTS))), f"mean {mean:.3f}, used {sorted(used)}")
    grown = collections.Counter(name for name, _values in instances(terrain, ()))
    print(f"  plants on the terrain: {grown.most_common()}", flush=True)

    # The A/B: the same plants by bare name, a table of its own, and back to the descriptions from the file
    choose(terrain, "Plant descriptions", "Bare names")
    done = fill("biome bare names table")
    bare, bare_used = table("Bare names")
    check("biome: bare plant names score below the descriptions (the A/B)", done and bare < mean,
          f"bare names: mean {bare:.3f}, {len(bare_used)} plants used; descriptions: mean {mean:.3f}")
    before = requests()
    choose(terrain, "Plant descriptions", "Descriptions")
    done = fill("biome back to the descriptions", 30)
    sent = sent_since(before)
    check("biome: back to the descriptions is instant, with 0 requests", done and sent == 0, f"{sent} requests")

    # A transect from hot lowland to cold summit, with moderate rain, through the same tree
    steps = [k / 32 for k in range(33)]
    line = bpy.data.meshes.new("Transect")
    line.from_pydata([(20.0 * s, -20.0, PEAK * s) for s in steps], [], [])
    line.attributes.new("temperature", 'FLOAT', 'POINT').data.foreach_set(
        "value", [TEMPERATURE[1] - s * (TEMPERATURE[1] - TEMPERATURE[0]) for s in steps])
    line.attributes.new("rain", 'FLOAT', 'POINT').data.foreach_set("value", [sum(RAIN) / 2] * len(steps))
    transect = user("Transect", tree, line)
    choose(transect, "Plant descriptions", "Descriptions")
    fill("biome transect", 60)
    amounts = values(transect, "snow")
    falls = [(k, round(a, 4), round(b, 4)) for k, (a, b) in enumerate(zip(amounts, amounts[1:])) if b < a - 1e-6]
    check("biome: snow rises monotonically along a transect from hot lowland to cold summit",
          len(amounts) == len(steps) and not falls and amounts[-1] > amounts[0],
          f"{' '.join(f'{a:.2f}' for a in amounts)}; falls {falls[:3]}")
    bpy.data.objects.remove(transect)
    bpy.data.meshes.remove(line)
    fill("biome without the transect", 60)


# ------------------------------------------------------------------------------------------------ kitchen (clutter)

PROPS = (  # the clutter demo's 40 props: name, shape, radius, height
    ("pan", "rod", 0.25, 0.06), ("kettle", "ball", 0.15, 0.3), ("toaster", "box", 0.15, 0.2),
    ("fridge", "box", 0.4, 1.8), ("microwave", "box", 0.3, 0.3), ("cutting board", "box", 0.2, 0.03),
    ("ladle", "rod", 0.04, 0.35), ("colander", "ball", 0.15, 0.15), ("teapot", "ball", 0.12, 0.2),
    ("rolling pin", "rod", 0.03, 0.45), ("sofa", "box", 0.9, 0.8), ("armchair", "box", 0.45, 0.9),
    ("coffee table", "box", 0.5, 0.45), ("television", "box", 0.6, 0.7), ("floor lamp", "rod", 0.1, 1.7),
    ("rug", "box", 1.0, 0.02), ("bed", "box", 1.0, 0.6), ("pillow", "ball", 0.3, 0.15), ("wardrobe", "box", 0.6, 2.0),
    ("nightstand", "box", 0.25, 0.5), ("alarm clock", "box", 0.06, 0.1), ("bathtub", "box", 0.8, 0.6),
    ("toilet", "ball", 0.25, 0.45), ("towel", "box", 0.3, 0.05), ("toothbrush", "rod", 0.01, 0.18),
    ("shampoo", "rod", 0.04, 0.2), ("desk", "box", 0.7, 0.75), ("office chair", "box", 0.3, 1.0),
    ("bookshelf", "box", 0.5, 1.8), ("printer", "box", 0.25, 0.25), ("stapler", "box", 0.08, 0.05),
    ("hammer", "rod", 0.03, 0.3), ("drill", "box", 0.1, 0.25), ("wrench", "rod", 0.02, 0.25),
    ("anvil", "box", 0.25, 0.3), ("toolbox", "box", 0.25, 0.25), ("car tire", "rod", 0.35, 0.2),
    ("feather", "rod", 0.01, 0.2), ("umbrella", "cone", 0.5, 0.9), ("houseplant", "cone", 0.2, 0.6),
)
ROOMS = ("kitchen: cooking, food and dishes; living room: sitting and relaxing; bedroom: sleeping and dressing; "
         "bathroom: washing and grooming; office: desk work and paper; garage: tools and repairs")
ANCHORS = ((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (20.0, 0.0, 0.0), (0.0, 12.0, 0.0), (10.0, 12.0, 0.0),
           (20.0, 12.0, 0.0))  # a place per room, in ROOMS order


def build_kitchen():
    props = hidden_collection("Props", [(name, solid(name, [(shape, radius, height, 0.0)], (0.6, 0.55, 0.5, 1.0)))
                                        for name, shape, radius, height in PROPS])
    tree, gi, go = new_tree("Clutter", ("Kitchen Only", 'NodeSocketBool'))
    kitchen = node(tree, MATCH, -1200, 600, list_source='COLLECTION')
    weight = node(tree, MATCH, -1200, 100, list_source='COLLECTION', answer_type='AMOUNT')
    rooms = node(tree, CLASSIFY, -1200, -400, list_source='COLLECTION')
    kitchen.inputs["Question"].default_value = "Would you find it in a kitchen?"
    weight.inputs["Question"].default_value = "How heavy is it?"
    rooms.inputs["Labels"].default_value = ROOMS
    for n in (kitchen, weight, rooms):
        n.inputs["Collection"].default_value = props
    things = children(tree, props, -1200, 1000)
    for x, (value, name, data_type) in enumerate(((kitchen.outputs["Probability"], "kitchen", 'FLOAT'),
                                                  (weight.outputs["Probability"], "weight", 'FLOAT'),
                                                  (rooms.outputs["Label"], "room", 'INT'))):
        things = stored(tree, things, value, name, data_type, -800 + 200 * x, 1000, 'INSTANCE')
    index = node(tree, "GeometryNodeInputIndex", -900, -900).outputs[0]
    for n in (kitchen, weight, rooms):  # linked: without OpenAct, an unlinked ID reads 0, not the element's index
        tree.links.new(index, n.inputs["ID"])
    spread = vector(tree, -400, -900, (number(tree, 'MULTIPLY', -600, -800, integer(
        tree, 'FLOORED_MODULO', -800, -800, index, 5), 1.4), number(tree, 'MULTIPLY', -600, -1000, integer(
            tree, 'DIVIDE_FLOOR', -800, -1000, index, 5), 1.4), 0.0))
    where = operation(tree, "ShaderNodeVectorMath", 'ADD', -200, -700,
                      chosen(tree, rooms.outputs["Label"], ANCHORS, -400, -600), spread)
    moved = node(tree, "GeometryNodeTranslateInstances", 0, 1000)
    tree.links.new(things, moved.inputs["Instances"])
    tree.links.new(where, moved.inputs["Translation"])
    scaled = node(tree, "GeometryNodeScaleInstances", 200, 1000)
    tree.links.new(moved.outputs[0], scaled.inputs["Instances"])
    tree.links.new(number(tree, 'ADD', 0, 800, weight.outputs["Probability"], 0.5), scaled.inputs["Scale"])
    kept = node(tree, "GeometryNodeDeleteGeometry", 500, 800, domain='INSTANCE')
    tree.links.new(scaled.outputs[0], kept.inputs["Geometry"])
    tree.links.new(operation(tree, "FunctionNodeBooleanMath", 'NOT', 300, 700, kitchen.outputs["Selection"]),
                   kept.inputs["Selection"])
    tree.links.new(switch(tree, gi.outputs["Kitchen Only"], scaled.outputs[0], kept.outputs[0], 800, 1000),
                   go.inputs["Geometry"])
    house = user("House", tree, Kitchen_Only=False)
    kitchen_only = user("Kitchen", tree, location=(40.0, 0.0, 0.0), Kitchen_Only=True)
    done = fill("kitchen tables")
    if MODEL is None or not answering("kitchen", done):
        return

    got = dict(instances(house, ("kitchen", "weight", "room")))
    labels = [label.split(":")[0] for label in ROOMS.split("; ")]
    for name, value in sorted(got.items()):
        room = labels[value["room"]] if 0 <= value["room"] < len(labels) else value["room"]
        print(f"  {name}: kitchen {value['kitchen']:.2f}, weight {value['weight']:.2f}, {room}", flush=True)
    pan, sofa = got.get("pan", {}), got.get("sofa", {})
    check("kitchen: pan = yes and sofa = no", done and pan.get("kitchen", 0.0) >= 0.5 > sofa.get("kitchen", 1.0),
          (pan, sofa))
    anvil, feather = got.get("anvil", {}), got.get("feather", {})
    check("kitchen: an anvil is heavier than a feather", done and anvil.get("weight", 0.0) > feather.get("weight", 1.0),
          (anvil, feather))
    check("kitchen: the pan's room is the kitchen", done and pan.get("room") == 0, pan)
    selected = sorted(name for name, value in got.items() if value["kitchen"] >= 0.5)
    check("kitchen: the Kitchen object keeps exactly the props that Match selects",
          done and sorted(name for name, _value in instances(kitchen_only, ())) == selected, selected)


# ------------------------------------------------------------------------------------------------ materials

SWATCHES = (("Steel", (0.62, 0.62, 0.66, 1.0)), ("Oak Planks", (0.55, 0.36, 0.2, 1.0)),
            ("Marble", (0.9, 0.9, 0.87, 1.0)), ("Silk", (0.8, 0.25, 0.35, 1.0)), ("Copper", (0.8, 0.45, 0.3, 1.0)),
            ("Brick", (0.6, 0.26, 0.2, 1.0)), ("Walnut", (0.33, 0.21, 0.12, 1.0)), ("Velvet", (0.32, 0.1, 0.42, 1.0)))
CUBE = ((0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5))  # a unit cube's faces


def build_materials():
    mesh = bpy.data.meshes.new("Swatches")
    corners = [(dx, dy, dz) for dz in (0, 1) for dy in (0, 1) for dx in (0, 1)]
    mesh.from_pydata([(1.5 * k + dx, dy, dz) for k in range(len(SWATCHES)) for dx, dy, dz in corners], [],
                     [tuple(8 * k + c for c in face) for k in range(len(SWATCHES)) for face in CUBE])
    mesh.polygons.foreach_set("material_index", [k for k in range(len(SWATCHES)) for _face in CUBE])
    for name, color in SWATCHES:  # the family's numbers from the CSV, through the faces' attributes
        mesh.materials.append(material(name, color, reads=(("Roughness", "roughness"), ("Metallic", "metallic"))))
    tree, gi, go = new_tree("Materials")
    swatches = user("Swatches", tree, mesh)
    families = node(tree, CLASSIFY, -800, 300, list_source='MATERIALS')
    metal = node(tree, MATCH, -800, -300, list_source='MATERIALS')
    metal.inputs["Question"].default_value = "Is this material metal?"
    index = node(tree, "GeometryNodeInputMaterialIndex", -1200, 0).outputs[0]
    for n in (families, metal):
        n.inputs["Object"].default_value = swatches
        tree.links.new(index, n.inputs["ID"])
    reader = node(tree, "GeometryNodeImportCSV", -1200, 400)
    reader.inputs["Path"].default_value = "//material_families.csv"
    tree.links.new(reader.outputs[0], families.inputs["Labels CSV"])
    engine.run_pass()  # plans: the Labels CSV's payload outputs 'roughness' and 'metallic' appear
    faces = gi.outputs["Geometry"]
    for x, (value, name, data_type) in enumerate(((families.outputs["Label"], "family", 'INT'),
                                                  (families.outputs["roughness"], "roughness", 'FLOAT'),
                                                  (families.outputs["metallic"], "metallic", 'FLOAT'),
                                                  (metal.outputs["Probability"], "metal", 'FLOAT'))):
        faces = stored(tree, faces, value, name, data_type, -400 + 200 * x, 0, 'FACE')
    tree.links.new(faces, go.inputs["Geometry"])
    done = fill("materials tables")
    if MODEL is None or not answering("materials", done):
        return

    check("materials: the Labels CSV comes through the Import CSV node, with its payload outputs",
          "roughness" in families.outputs and "metallic" in families.outputs, [s.name for s in families.outputs])
    family, metal_yes = values(swatches, "family"), values(swatches, "metal")
    labels = next(iter(S.load(families).entries), None)
    labels = labels.shape.get("labels", []) if labels else []
    for k, (name, _color) in enumerate(SWATCHES):  # six faces each
        label = labels[family[6 * k]] if 0 <= family[6 * k] < len(labels) else family[6 * k]
        print(f"  {name}: {label}, metal {metal_yes[6 * k]:.2f}", flush=True)

    # The edits go to a scratch copy of the demo and its CSV: the demo, and the CSV it ships with, never change
    shipped = os.path.join(OUT, "material_families.csv")
    with open(shipped, "rb") as handle:
        original = handle.read()
    bpy.ops.wm.save_mainfile()
    real, scratch = bpy.data.filepath, tempfile.mkdtemp(prefix="openact_materials_")
    try:
        shutil.copyfile(real, os.path.join(scratch, "materials.blend"))
        path = shutil.copyfile(shipped, os.path.join(scratch, "material_families.csv"))
        bpy.ops.wm.open_mainfile(filepath=os.path.join(scratch, "materials.blend"))
        swatches = bpy.data.objects["Swatches"]
        fill("materials scratch copy", 60)
        text = original.decode("utf-8")
        lines = text.splitlines(keepends=True)
        head, roughness, tail = lines[1 + family[0]].rsplit(",", 2)  # the family of the Steel faces, the first six
        old, new = float(roughness), round(float(roughness) + 0.1, 2)
        lines[1 + family[0]] = f"{head},{new},{tail}"

        def saved(content, want):
            """Save the CSV as the artist does: (whether the Steel faces show roughness want, seconds, requests)."""
            before, stamp = requests(), os.stat(path).st_mtime_ns + 10 ** 9
            with open(path, "w", encoding="utf-8", newline="") as handle:
                handle.write(content)
            os.utime(path, ns=(stamp, stamp))  # a later stamp, whatever the file system's clock resolution
            started = time.time()
            shown = wait(lambda: abs(values(swatches, "roughness")[0] - want) < 1e-6, 15)
            return shown, time.time() - started, sent_since(before)

        shown, seconds, sent = saved("".join(lines), new)
        check("materials: a roughness edit in the CSV reaches the faces with 0 requests", shown and sent == 0,
              f"{old} -> {new} after {seconds:.2f}s, {sent} requests")
        shown, seconds, sent = saved(text, old)
        check("materials: edited back, the faces follow, again with 0 requests", shown and sent == 0,
              f"after {seconds:.2f}s, {sent} requests")
    finally:
        bpy.ops.wm.open_mainfile(filepath=real)
        shutil.rmtree(scratch, ignore_errors=True)
    with open(shipped, "rb") as handle:
        check("materials: the CSV beside the demo is unchanged", handle.read() == original, shipped)


# ------------------------------------------------------------------------------------------------ weather

STORM, CALM = "A stormy night in a fishing village", "A calm summer morning"
SUN = ((-12.0, 0.0, 2.0), (0.0, -4.0, 14.0), (12.0, 0.0, 2.0), (0.0, 0.0, -6.0))  # dawn, day, dusk, night
DIALED = (("Rain", "rain", 'BOOLEAN'), ("Wind speed", "wind_speed", 'FLOAT'), ("Time of day", "time_of_day", 'INT'),
          ("Street lights", "street_lights", 'BOOLEAN'), ("Fog", "fog", 'FLOAT'), ("Confidence", "confidence", 'FLOAT'))


def dialed(obj):
    """What the dials show, as stored on the ground."""
    found = {}
    for _setting, name, _type in DIALED:
        value = values(obj, name)[0]
        found[name] = round(value, 4) if isinstance(value, float) else value
    return found


def build_weather():
    tree, gi, go = new_tree("Weather", ("Prompt", 'NodeSocketString'))
    dials = node(tree, DIALS, -1400, 400)
    tree.links.new(gi.outputs["Prompt"], dials.inputs["Prompt"])
    dials.inputs["Settings"].default_value = "//weather_dials.csv"
    village = user("Village", tree, plane("Village", 20.0), Prompt=STORM)
    engine.run_pass()  # plans: an output per setting appears
    out = dials.outputs
    ground = gi.outputs["Geometry"]
    for x, (setting, name, data_type) in enumerate(DIALED):
        ground = stored(tree, ground, out[setting], name, data_type, -1000 + 180 * x, 900)
    sway = vector(tree, -900, 100, (number(tree, 'MULTIPLY', -1100, 100, out["Wind speed"], 0.015), 0.0, 0.0))
    trees = placed(tree, primitive(tree, "GeometryNodeMeshLine", -900, 500, Count=6, Start_Location=(-7.5, 5.0, 1.25),
                                   Offset=(3.0, 0.0, 0.0)),
                   primitive(tree, "GeometryNodeMeshCone", -900, 300, Vertices=12, Radius_Bottom=0.6, Depth=2.5),
                   -600, 400, rotation=sway)
    lifted = node(tree, "GeometryNodeSetPosition", -700, -200)
    tree.links.new(primitive(tree, "GeometryNodeMeshGrid", -900, -200, Size_X=18.0, Size_Y=18.0, Vertices_X=14,
                             Vertices_Y=14), lifted.inputs["Geometry"])
    feed(tree, lifted.inputs["Offset"], vector(tree, -900, -400, (0.0, 0.0, randomly(tree, 'FLOAT', 5, -1100, -400,
                                                                                     Min=1.0, Max=8.0))))
    drop = painted(tree, primitive(tree, "GeometryNodeMeshCylinder", -700, -500, Vertices=6, Radius=0.02, Depth=0.35),
                   material("Rain", (0.5, 0.6, 0.8, 1.0)), -500, -500)
    rain = switch(tree, out["Rain"], None, placed(tree, lifted.outputs[0], drop, -400, -200), -200, -200)
    sky = node(tree, "GeometryNodePoints", -700, -800)
    sky.inputs["Count"].default_value = 1
    tree.links.new(chosen(tree, out["Time of day"], SUN, -900, -800), sky.inputs["Position"])
    sun = placed(tree, sky.outputs[0], painted(tree, primitive(tree, "GeometryNodeMeshUVSphere", -700, -1000,
                                                               Radius=1.0),
                                               material("Sun", (1.0, 0.85, 0.5, 1.0), emission=5.0), -500, -1000),
                 -400, -800)
    bulb = painted(tree, primitive(tree, "GeometryNodeMeshIcoSphere", -700, -1300, Radius=0.25, Subdivisions=2),
                   material("Bulb", (1.0, 0.8, 0.4, 1.0), emission=20.0), -500, -1300)
    posts = primitive(tree, "GeometryNodeMeshLine", -700, -1200, Count=4, Start_Location=(-6.0, -3.0, 3.0),
                      Offset=(4.0, 0.0, 0.0))
    lights = switch(tree, out["Street lights"], None, placed(tree, posts, bulb, -400, -1200), -200, -1200)
    cube = node(tree, "GeometryNodeVolumeCube", -700, -1600)
    tree.links.new(out["Fog"], cube.inputs["Density"])
    cube.inputs["Min"].default_value, cube.inputs["Max"].default_value = (-10.0, -10.0, 0.0), (10.0, 10.0, 6.0)
    for axis in "XYZ":
        cube.inputs[f"Resolution {axis}"].default_value = 16
    fog = painted(tree, cube.outputs[0], fog_material(), -400, -1600)
    tree.links.new(joined(tree, 400, 0, ground, trees, rain, sun, lights, fog), go.inputs["Geometry"])
    done = fill("weather dials")
    if MODEL is None or not answering("weather", done):
        return

    storm = dialed(village)
    print(f"  storm: {storm}", flush=True)

    def shown(prompt):
        return answered() and engine.STATUS[dials.uid].uses[0][2].gn[0] == prompt

    before = requests()
    modify(village, "Prompt", CALM)
    started = time.time()
    done = wait(lambda: shown(CALM), 60)
    seconds, sent = time.time() - started, sent_since(before)
    print(f"  calm: {dialed(village)}", flush=True)
    print(f"TIMING weather new prompt: {seconds:.2f}s (budget 0.6 s)", flush=True)
    if done and seconds > 0.6:
        print(f"WARN weather: the new prompt took {seconds:.2f}s, above 0.6 s", flush=True)
    check("weather: a new prompt re-answers the five dials in one request", done and sent == 1, f"{sent} requests")
    check("weather: a new prompt is answered within 0.6 s (fails above twice that)", done and seconds <= 1.2,
          f"{seconds:.2f}s")
    before = requests()
    modify(village, "Prompt", STORM)
    started = time.time()
    done = wait(lambda: shown(STORM) and dialed(village) == storm, 10)
    sent = sent_since(before)
    check("weather: switching back is instant (cached), with 0 requests", done and sent == 0,
          f"{time.time() - started:.2f}s, {sent} requests")


# ------------------------------------------------------------------------------------------------ Ask and Pick

FURNITURE = (("Armchair", "box", 0.45, 0.9), ("Bookshelf", "box", 0.5, 1.8), ("Floor Lamp", "rod", 0.1, 1.7),
             ("Rug", "box", 1.0, 0.02), ("Table", "box", 0.6, 0.75))
SCENES = {  # the ask_pick scene: object -> its Scene and Pick Prompt
    "A": ("A snowy mountain village at night, warm windows glowing.", "something to sit on"),
    "B": ("A sunny tropical beach at noon, people swimming.", "something to put books on"),
}


def build_ask_pick():
    props = hidden_collection("Furniture", [(name, solid(name, [(shape, radius, height, 0.0)], (0.55, 0.4, 0.3, 1.0)))
                                            for name, shape, radius, height in FURNITURE])
    tree, gi, go = new_tree("Ask and Pick", ("Scene", 'NodeSocketString'), ("Pick Prompt", 'NodeSocketString'))
    night = node(tree, ASK, -1000, 600)
    night.inputs["Question"].default_value = "Is it night?"
    tree.links.new(gi.outputs["Scene"], night.inputs["Context"])
    pick = node(tree, PICK, -1000, 0, list_source='COLLECTION')
    pick.inputs["Collection"].default_value = props
    tree.links.new(gi.outputs["Pick Prompt"], pick.inputs["Prompt"])
    base = gi.outputs["Geometry"]
    for x, (value, name, data_type) in enumerate(((night.outputs["Answer"], "night", 'FLOAT'),
                                                  (night.outputs["Yes"], "night_yes", 'BOOLEAN'),
                                                  (pick.outputs["Index"], "pick", 'INT'))):
        base = stored(tree, base, value, name, data_type, -600 + 200 * x, 700)
    index = node(tree, "GeometryNodeInputIndex", -900, -600).outputs[0]
    tree.links.new(index, pick.inputs["ID"])  # linked: without OpenAct, an unlinked ID reads 0, not the element's index
    shelf = node(tree, "GeometryNodeTranslateInstances", -400, -300)
    tree.links.new(children(tree, props, -800, -300), shelf.inputs["Instances"])
    tree.links.new(vector(tree, -600, -600, (number(tree, 'MULTIPLY_ADD', -800, -700, index, 1.8, -3.6), 3.0, 0.0)),
                   shelf.inputs["Translation"])
    fitness = stored(tree, shelf.outputs[0], pick.outputs["Probability"], "fitness", 'FLOAT', -200, -300, 'INSTANCE')
    heat = node(tree, "GeometryNodeScaleInstances", 0, -300)  # the heat map: each child as big as it fits
    tree.links.new(fitness, heat.inputs["Instances"])
    tree.links.new(number(tree, 'ADD', -200, -500, pick.outputs["Probability"], 0.4), heat.inputs["Scale"])
    sky = node(tree, "GeometryNodePoints", -400, -900)
    sky.inputs["Position"].default_value = (0.0, 6.0, 5.0)
    moon = painted(tree, primitive(tree, "GeometryNodeMeshIcoSphere", -600, -1100, Radius=0.6, Subdivisions=2),
                   material("Moon", (0.8, 0.85, 0.95, 1.0), emission=1.0), -400, -1100)
    sun = painted(tree, primitive(tree, "GeometryNodeMeshUVSphere", -600, -1300, Radius=1.0),
                  material("Sun", (1.0, 0.85, 0.4, 1.0), emission=5.0), -400, -1300)
    celestial = switch(tree, night.outputs["Yes"], placed(tree, sky.outputs[0], sun, -200, -1200),
                       placed(tree, sky.outputs[0], moon, -200, -1000), 0, -1000)
    tree.links.new(joined(tree, 400, 0, base, heat.outputs[0], celestial), go.inputs["Geometry"])
    users = {key: user(key, tree, plane(key, 1.0), location=(14.0 * i, 0.0, 0.0), Scene=scene, Pick_Prompt=prompt)
             for i, (key, (scene, prompt)) in enumerate(SCENES.items())}
    done = fill("ask_pick answers")
    if MODEL is None or not answering("ask_pick", done):
        return

    names = sorted((name for name, *_shape in FURNITURE), key=natsort.sort_key)  # Collection Info's order
    got = {key: (values(obj, "night")[0], values(obj, "pick")[0]) for key, obj in users.items()}
    picked = {key: names[index] if 0 <= index < len(names) else index for key, (_night, index) in got.items()}
    check("ask_pick: A's night > 0.9", done and got["A"][0] > 0.9, round(got["A"][0], 3))
    check("ask_pick: B's night < 0.1", done and got["B"][0] < 0.1, round(got["B"][0], 3))
    check("ask_pick: A picks the armchair", done and picked["A"] == "Armchair", picked["A"])
    check("ask_pick: B picks the bookshelf", done and picked["B"] == "Bookshelf", picked["B"])


# ------------------------------------------------------------------------------------------------ main

DEMOS = (  # name, builder, the CSV files beside it, the frames evaluated.json records, the menu it records them for
    ("life", build_life, ("life_factors.csv", "life_outcomes.csv", "highlife_outcomes.csv", "seeds_outcomes.csv",
                          "daynight_outcomes.csv", "life_overrides.csv"), range(1, 33), ("Life", "Rule set", RULE_SETS)),
    ("biome", build_biome, ("biome_factors.csv", *OUTCOMES.values()), (1,), ("Terrain", "Plant descriptions", OUTCOMES)),
    ("kitchen", build_kitchen, (), (1,), None),
    ("materials", build_materials, ("material_families.csv",), (1,), None),
    ("weather", build_weather, ("weather_dials.csv",), (1,), None),
    ("ask_pick", build_ask_pick, (), (1,), None),
)


def new_file(name, csvs):
    """An empty file saved as <out>/<name>.blend, beside copies of the demo's CSV files, which its '//' paths name."""
    bpy.ops.wm.read_homefile(use_empty=True)
    for csv in csvs:
        source, target = os.path.join(HERE, csv), os.path.join(OUT, csv)
        if os.path.normcase(source) != os.path.normcase(target):
            shutil.copyfile(source, target)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, name + ".blend"))


def stage():
    """A camera framed on what the demo evaluates to at frame 1, from the front right and above, a sun, and Blender's
    default grey world: the empty file has none of them, and the saved demo must render as it is (18.4)."""
    scene = bpy.context.scene
    scene.frame_set(1)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    corners = [instance.matrix_world @ Vector(corner) for instance in depsgraph.object_instances
               if instance.object.type not in ('CAMERA', 'LIGHT') for corner in instance.object.bound_box]
    lo = Vector([min(c[k] for c in corners) for k in range(3)]) if corners else Vector((-1.0, -1.0, -1.0))
    hi = Vector([max(c[k] for c in corners) for k in range(3)]) if corners else Vector((1.0, 1.0, 1.0))
    camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    scene.collection.objects.link(camera)
    scene.camera = camera
    toward = Vector((0.9, -1.5, 1.0)).normalized()  # from the subject to the camera
    camera.rotation_euler = (-toward).to_track_quat('-Z', 'Y').to_euler()
    box = [value for corner in itertools.product(*zip(lo, hi)) for value in corner]
    fitted, _scale = camera.camera_fit_coords(bpy.context.evaluated_depsgraph_get(), box)  # evaluates the new camera
    center, fitted = (lo + hi) / 2, Vector(fitted)
    camera.location = fitted + toward * 0.15 * (fitted - center).length  # a margin around the subject
    camera.data.clip_end = max(camera.data.clip_end, 4 * (camera.location - center).length)
    light = bpy.data.objects.new("Light", bpy.data.lights.new("Light", 'SUN'))
    light.data.energy = 3.0
    light.rotation_euler = (math.radians(40), math.radians(15), math.radians(30))
    scene.collection.objects.link(light)
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
        scene.world.color = (0.05, 0.05, 0.05)


def build():
    prefs = P.get_prefs()
    prefs.auto_start = MODEL is not None
    if MODEL:
        prefs.port, prefs.gpu = MODEL
    bpy.context.preferences.filepaths.save_version = 0  # no .blend1 copies beside the demos
    os.makedirs(OUT, exist_ok=True)
    print(f"BLENDER {bpy.app.version_string}: demos in {OUT}, "
          + (f"the model on port {prefs.port}, GPU {prefs.gpu}" if MODEL is not None else "no model"), flush=True)
    if MODEL is not None:
        check("runtime detected", P.runtime_installed(P.runtime_root(prefs)), P.runtime_root(prefs))
    results, started = {}, time.time()
    try:
        for name, builder, csvs, frames, menu in DEMOS:
            begun = time.time()
            try:
                new_file(name, csvs)
                builder()
                if MODEL is not None:
                    whole(name)
                stage()
                bpy.ops.wm.save_mainfile()
                results[name] = record(frames, menu)
            except Exception:
                traceback.print_exc()
                check(f"{name}: built without an error", False)
            print(f"TIMING {name}: {time.time() - begun:.1f}s", flush=True)
    finally:
        if MODEL is not None:
            server.stop()
    with open(EVALUATED, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=1)
    print(f"TIMING all demos: {time.time() - started:.1f}s", flush=True)


def noaddon():
    """Open each of the six demos the build saved, in a Blender without OpenAct: each evaluates to what it did with it
    (G6), also with every item of its menu picked by hand. A demo the build didn't record or save fails."""
    print(f"BLENDER {bpy.app.version_string}: demos in {OUT}, without the add-on", flush=True)
    try:
        with open(EVALUATED, encoding="utf-8") as handle:
            expected = json.load(handle)
    except (OSError, ValueError) as error:
        print(f"  no record of the build: {error}", flush=True)
        expected = {}
    for name, _builder, _csvs, _frames, menu in DEMOS:
        path = os.path.join(OUT, name + ".blend")
        if name not in expected or not os.path.isfile(path):
            check(f"noaddon: {name}: recorded by the build, and saved", False, f"{EVALUATED}, {path}")
            continue
        try:
            bpy.ops.wm.open_mainfile(filepath=path)
            kinds = [n.bl_idname for tree in bpy.data.node_groups for n in tree.nodes]
            check(f"noaddon: {name}: the OpenAct nodes are undefined nodes", "NodeUndefined" in kinds
                  and not any(kind.startswith("GeometryNodeOpenAct") for kind in kinds),
                  f"{kinds.count('NodeUndefined')} undefined")
            scene, differ, count = bpy.context.scene, [], 0
            for item, frames in expected[name].items():
                if item:
                    choose(bpy.data.objects[menu[0]], menu[1], item)
                for frame, objects in frames.items():
                    scene.frame_set(int(frame))
                    count += 1
                    for obj_name, want in objects.items():
                        obj = bpy.data.objects.get(obj_name)
                        got = digest(obj) if obj is not None else {"missing": obj_name}
                        if got != want:
                            differ.append((item, frame, obj_name, sorted(key for key in got.keys() | want.keys()
                                                                         if got.get(key) != want.get(key))))
            check(f"noaddon: {name}: evaluates to the same attributes as with the add-on", count > 0 and not differ,
                  f"{count} frames; differences {differ[:3]}")
        except Exception:
            traceback.print_exc()
            check(f"noaddon: {name}: opens and evaluates", False)


try:
    if NOADDON:
        noaddon()
    else:
        build()
except Exception:
    traceback.print_exc()
    check("no exception", False)
finish()
