"""The upgrade from jev_nodes v0.1.0 (§12) in a headless Blender, with no model: the untouched prototype, enabled from
a copy of its package, builds and answers a v0.1.0 file through a fake Jev server; then OpenAct opens and upgrades it.

- The file: the prototype e2e's scene (its tests/e2e_blender.py build: Ask Yes/No, a text Pick, an Ask chained to the
  Pick's Item, an Amount Ask in a nested group, a collection Pick whose Instance is realised, the prompt edit and back,
  a copy of the tree that no modifier uses), plus a Pick whose Items text has ';' and ': ', a Pick with both Items
  text and a Collection, the Pick's Item joined into another Ask's Context by a Join Strings it was linked to first,
  and node settings: a frame, a label, a colour, a width, a collapsed and muted node.
- Both add-ons enabled: OpenAct takes jev_nodes' preferences once, registers no legacy node types, and says to disable
  Jev Nodes.
- OpenAct alone, the v0.1.0 file loaded: the old nodes are defined, frozen and hidden from the Add menu, evaluate as
  saved and are never asked; an INFO report after loading and the sidebar banner point to the upgrade.
- openact.upgrade_legacy, run as a button runs it (registered, with its undo step): its report; outputs identical to
  v0.1.0's; with a fake model server ready, 0 requests (its --record log) and no cell planned; each node's uid,
  answers (with v1 words keys, "asked with older settings"), inner group (the same data-block, renamed, owned),
  links (a multi-input socket's order too) and settings kept; Ask's mode as its answer type; Pick's list source by
  v0.1.0's precedence (a set Collection wins), and its legacy item parsing. Then a new word is asked: the count works.
- Undo (the undo system started in background mode): the old nodes, their inner groups, answers and outputs are back,
  and nothing is asked.
- A problem: a text Pick whose Instance output is linked stays frozen and is named in the report, with nothing left
  behind for it; a node that lost its inner group gets a new one, built from its answers.
- jev_nodes enabled after OpenAct takes the legacy node types over, and OpenAct still disables cleanly.

Run:  blender -b --factory-startup --python tests/migrate_blender.py
"""

import contextlib
import hashlib
import http.server
import io
import json
import os
import shutil
import socket
import sys
import tempfile
import threading
import traceback

TMP = tempfile.mkdtemp(prefix="openact_migrate_")
os.environ["LOCALAPPDATA" if sys.platform == "win32" else "XDG_DATA_HOME"] = os.path.join(TMP, "local")  # no runtime
os.environ["OPENACT_BACKEND"] = "fake"  # openact_server.py --backend fake: no torch, GPU or runtime
os.environ["OPENACT_SERVER_PYTHON"] = sys.executable
os.environ.setdefault("OPENACT_STATE_DIR", os.path.join(TMP, "state"))

import addon_utils  # noqa: E402
import bpy  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import FAILURES, check, drive, finish, store  # noqa: E402

PROTOTYPE = r"X:\interzis\CHUMP_KIT\jev_nodes"  # read only: its package is copied into TMP and enabled from there
V010 = os.path.join(TMP, "v010.blend")
RECORD = os.path.join(TMP, "record.jsonl")
SCENES = {
    "A": ("A snowy mountain village at night, warm windows glowing.", "something to sit on"),
    "B": ("A sunny tropical beach at noon, people swimming.", "something to put books on"),
}
ITEMS = "table, lamp, armchair, bookshelf, rug"
LEGACY_ITEMS = "Chair: comfy; Stool, Bench: long"  # v0.1.0 split on commas only and kept ": " in the names
PROPS = {"Armchair": 3, "Bookshelf": 6, "Floor Lamp": 4, "Rug": 7, "Table": 5}  # name -> vertex count
ATTRIBUTES = ("night", "night_yes", "pick", "pick_conf", "sit", "cold", "lamp_index", "rest", "cup", "joined", "muted")
USED = ("JevTest", "JevNested")  # the trees the modifiers use; "JevTest Copy" has no modifier
LEGACY_TYPES = ("GeometryNodeJevAsk", "GeometryNodeJevPick")
OLD_NODES, OLD_TREES = 17, 3  # 8 in JevTest and its copy, 1 in JevNested


class Stop(Exception):
    """A check failed that the rest of the suite depends on."""


def free_port():
    """A free local port, never 18765 or 18769: the user's own OpenAct session and another job use those."""
    while True:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        if port not in (18765, 18769):
            return port


# ---------------------------------------------------------------------------------- the fake Jev server

def weight(*words):
    """0-1, fixed for these words: the fake model's answers are the same in every run."""
    return int(hashlib.sha256(json.dumps(words).encode()).hexdigest()[:8], 16) / 16 ** 8


def answer(state, question):
    """A v0.1.0 question's answer, in the form jev_nodes' server gives it."""
    kind, text = question["type"], question["instructions"]
    if kind == "noul":
        return {"type": "noul", "noul": weight(state, text)}
    if kind == "score":
        levels = [weight(state, text, i) + 0.01 for i in range(len(question["criteria"]))]
        probabilities = {str(i): level / sum(levels) for i, level in enumerate(levels)}
        return {"type": "score", "score": sum(i * p for i, p in enumerate(probabilities.values())),
                "confidence": weight(text, state), "probabilities": probabilities}
    weights = {key: weight(state, text, name) + 0.01 for key, name in question["criteria"].items()}
    probabilities = {key: w / sum(weights.values()) for key, w in weights.items()}
    ranked = sorted(probabilities.values(), reverse=True)
    return {"type": "choice", "choice": max(probabilities, key=probabilities.get),
            "confidence": ranked[0] - ranked[1], "probabilities": probabilities}


class JevServer(http.server.BaseHTTPRequestHandler):
    """/health, /v1/systemone and /shutdown of the model server that jev_nodes v0.1.0 talks to."""

    asked = []  # every /v1/systemone request

    def do_GET(self):
        self.reply({"server": "jev-nodes-semif", "version": "0.1.0", "status": "ready", "source": "fake",
                    "device": "cpu"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if self.path != "/v1/systemone":
            self.reply({})
            return
        JevServer.asked.append(body)
        self.reply({"answers": {name: answer(body["state"], question)
                                for name, question in body["questions"].items()}})

    def reply(self, data):
        payload = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args):
        pass


def fake_runtime(root, model_dir):
    """Just enough for jev_nodes' runtime_installed(): the venv's Python and a one-shard model."""
    python = os.path.join(root, "venv", "Scripts" if sys.platform == "win32" else "bin",
                          "python.exe" if sys.platform == "win32" else "python")
    os.makedirs(os.path.dirname(python), exist_ok=True)
    open(python, "wb").close()
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "model.safetensors.index.json"), "w", encoding="utf-8") as handle:
        json.dump({"weight_map": {"w": "model-1.safetensors"}}, handle)
    with open(os.path.join(model_dir, "model-1.safetensors"), "wb") as handle:
        handle.write(b"x")


# ------------------------------------------------------------------------------------------ helpers

@contextlib.contextmanager
def captured():
    """What Python prints meanwhile, still printed: Blender prints each operator report there ("Info: ...")."""
    real, seen = sys.stdout, []

    class Tee(io.TextIOBase):
        def write(self, text):
            seen.append(text)
            return real.write(text)

        def flush(self):
            real.flush()

    sys.stdout = Tee()
    out = type("Out", (), {"text": ""})()
    try:
        yield out
    finally:
        sys.stdout = real
        out.text = "".join(seen)


def read(name):
    """The first vertex's attributes of the evaluated object, and its vertex count (the realised Instance's too)."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    evaluated = bpy.data.objects[name].evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        values = {"verts": len(mesh.vertices)}
        for attribute in ATTRIBUTES:
            found = mesh.attributes.get(attribute)
            values[attribute] = found.data[0].value if found else None
        return values
    finally:
        evaluated.to_mesh_clear()


def outputs():
    return {name: read(name) for name in SCENES}


def requests():
    """The requests the fake model server got: its --record log's lines."""
    if not os.path.exists(RECORD):
        return 0
    with open(RECORD, encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def legacy_nodes():
    return [(tree, node) for tree in bpy.data.node_groups if not tree.name.startswith((".Jev ", ".OpenAct "))
            for node in tree.nodes if node.bl_idname in LEGACY_TYPES]


def links_of(node):
    """The node's links: (its socket's name, the other node's name, the other socket's identifier, muted)."""
    links = node.id_data.links
    return ({(link.to_socket.name, link.from_node.name, link.from_socket.identifier, link.is_muted) for link in links
             if link.to_node == node},
            {(link.from_socket.name, link.to_node.name, link.to_socket.identifier, link.is_muted) for link in links
             if link.from_node == node})


def settings_of(node):
    return (node.label, node.width, node.mute, node.hide, node.use_custom_color, tuple(node.color),
            node.parent.name if node.parent else "", tuple(node.location))


def join_order():
    """The joined Ask's Join Strings: its links' source nodes in the order they are joined."""
    join = bpy.data.node_groups["JevTest"].nodes["Join Strings"]
    links = sorted(join.inputs["Strings"].links, key=lambda link: -link.multi_input_sort_id)
    return [link.from_node.name for link in links]


class Stand:
    """Stands in for a node: store.load() reads only its entries string."""

    def __init__(self, entries):
        self.entries = entries

# ------------------------------------------------------------------------------ the v0.1.0 file

def build_scene():
    """The prototype e2e's scene, with the nodes of the upgrade's other cases."""
    scene = bpy.context.scene
    props = bpy.data.collections.new("Props")
    scene.collection.children.link(props)
    for name, count in PROPS.items():
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata([(i * 0.1, 0.0, 0.0) for i in range(count)], [], [])
        props.objects.link(bpy.data.objects.new(name, mesh))
    props.hide_render = True

    nested = bpy.data.node_groups.new("JevNested", 'GeometryNodeTree')
    nested.interface.new_socket("Ctx", in_out='INPUT', socket_type='NodeSocketString')
    nested.interface.new_socket("Cold", in_out='OUTPUT', socket_type='NodeSocketFloat')
    n_in, n_out = nested.nodes.new("NodeGroupInput"), nested.nodes.new("NodeGroupOutput")
    cold = nested.nodes.new("GeometryNodeJevAsk")
    cold.mode = 'AMOUNT'
    cold.inputs["Question"].default_value = "How cold is it?"
    nested.links.new(n_in.outputs["Ctx"], cold.inputs["Context"])
    nested.links.new(cold.outputs["Answer"], n_out.inputs["Cold"])

    tree = bpy.data.node_groups.new("JevTest", 'GeometryNodeTree')
    tree.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    tree.interface.new_socket("Scene", in_out='INPUT', socket_type='NodeSocketString')
    tree.interface.new_socket("Pick Prompt", in_out='INPUT', socket_type='NodeSocketString')
    tree.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    g_in, g_out = tree.nodes.new("NodeGroupInput"), tree.nodes.new("NodeGroupOutput")
    night = tree.nodes.new("GeometryNodeJevAsk")  # "Ask Jev"
    night.inputs["Question"].default_value = "Is it night?"
    tree.links.new(g_in.outputs["Scene"], night.inputs["Context"])
    pick = tree.nodes.new("GeometryNodeJevPick")  # "Jev Pick"
    tree.links.new(g_in.outputs["Pick Prompt"], pick.inputs["Prompt"])
    pick.inputs["Items"].default_value = ITEMS
    sit = tree.nodes.new("GeometryNodeJevAsk")  # "Ask Jev.001", chained: Pick.Item -> Ask.Context
    sit.inputs["Question"].default_value = "Can a person sit on this?"
    tree.links.new(pick.outputs["Item"], sit.inputs["Context"])
    group = tree.nodes.new("GeometryNodeGroup")
    group.node_tree = nested
    tree.links.new(g_in.outputs["Scene"], group.inputs["Ctx"])
    lamp = tree.nodes.new("GeometryNodeJevPick")  # "Jev Pick.001": collection mode + Instance output
    lamp.inputs["Prompt"].default_value = "something that gives light"
    lamp.inputs["Collection"].default_value = props

    frame = tree.nodes.new("NodeFrame")
    frame.label, frame.location = "Old items", (-400, -900)
    rest = tree.nodes.new("GeometryNodeJevPick")
    rest.name, rest.label, rest.width = "Rest", "Rest spot", 320
    rest.inputs["Prompt"].default_value = "somewhere to rest"
    rest.inputs["Items"].default_value = LEGACY_ITEMS
    rest.parent = frame
    rest.location = (40, -60)
    rest.use_custom_color, rest.color = True, (0.25, 0.5, 0.75)
    cup = tree.nodes.new("GeometryNodeJevPick")  # v0.1.0 took the collection and ignored the Items text
    cup.name = "Cup"
    cup.inputs["Prompt"].default_value = "something to put a cup on"
    cup.inputs["Items"].default_value = "tray, saucer"
    cup.inputs["Collection"].default_value = props
    words = tree.nodes.new("FunctionNodeInputString")
    words.string = "Think about:"
    join = tree.nodes.new("GeometryNodeStringJoin")
    join.inputs["Delimiter"].default_value = " "
    tree.links.new(pick.outputs["Item"], join.inputs["Strings"])  # linked first: joined last
    tree.links.new(words.outputs[0], join.inputs["Strings"])
    joined = tree.nodes.new("GeometryNodeJevAsk")
    joined.name = "Joined"
    joined.inputs["Question"].default_value = "Is it furniture?"
    tree.links.new(join.outputs[0], joined.inputs["Context"])
    muted = tree.nodes.new("GeometryNodeJevAsk")
    muted.name = "Muted"
    muted.inputs["Question"].default_value = "Is it raining?"
    muted.mute = muted.hide = True

    geometry = g_in.outputs["Geometry"]
    for x, (output, name, data_type) in enumerate((
            (night.outputs["Answer"], "night", 'FLOAT'), (night.outputs["Yes"], "night_yes", 'BOOLEAN'),
            (pick.outputs["Index"], "pick", 'INT'), (pick.outputs["Confidence"], "pick_conf", 'FLOAT'),
            (sit.outputs["Answer"], "sit", 'FLOAT'), (group.outputs["Cold"], "cold", 'FLOAT'),
            (lamp.outputs["Index"], "lamp_index", 'INT'), (rest.outputs["Index"], "rest", 'INT'),
            (cup.outputs["Index"], "cup", 'INT'), (joined.outputs["Answer"], "joined", 'FLOAT'),
            (muted.outputs["Answer"], "muted", 'FLOAT'))):
        geometry = store(tree, geometry, output, name, data_type, 200 * (x + 1))
    realize = tree.nodes.new("GeometryNodeRealizeInstances")
    tree.links.new(lamp.outputs["Instance"], realize.inputs["Geometry"])
    join_geometry = tree.nodes.new("GeometryNodeJoinGeometry")
    tree.links.new(realize.outputs[0], join_geometry.inputs[0])
    tree.links.new(geometry, join_geometry.inputs[0])  # linked last: its points come first
    tree.links.new(join_geometry.outputs[0], g_out.inputs["Geometry"])

    ids = {item.name: item.identifier for item in tree.interface.items_tree if getattr(item, "in_out", "") == 'INPUT'}
    for key, (description, prompt) in SCENES.items():
        mesh = bpy.data.meshes.new(key)
        mesh.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)], [], [])
        obj = bpy.data.objects.new(key, mesh)
        scene.collection.objects.link(obj)
        mod = obj.modifiers.new("Jev", 'NODES')
        mod.node_group = tree
        mod[ids["Scene"]] = description
        mod[ids["Pick Prompt"]] = prompt
        obj.update_tag()
    return tree, ids


def phase_build():
    """Build, answer and save the v0.1.0 file with the prototype. Returns the outputs it evaluated."""
    copy = os.path.join(TMP, "prototype")
    shutil.copytree(os.path.join(PROTOTYPE, "jev_nodes"), os.path.join(copy, "jev_nodes"),
                    ignore=shutil.ignore_patterns("__pycache__"))
    sys.path.insert(0, copy)
    enabled = addon_utils.enable("jev_nodes", default_set=True, handle_error=print)
    check("build: the untouched prototype enables from a copy of its package", enabled is not None)
    if enabled is None:
        raise Stop
    from jev_nodes import engine as jev
    from jev_nodes import prefs as jev_prefs
    from jev_nodes import server as jev_server

    listener = http.server.ThreadingHTTPServer(("127.0.0.1", free_port()), JevServer)
    threading.Thread(target=listener.serve_forever, daemon=True).start()
    prefs = jev_prefs.get_prefs()
    prefs.port, prefs.runtime_dir = listener.server_address[1], os.path.join(TMP, "jev runtime")
    fake_runtime(prefs.runtime_dir, jev_prefs.model_dir(prefs.runtime_dir))
    jev_server.refresh()

    def settled():
        nodes = list(jev.iter_jev_nodes())
        return bool(nodes) and not jev.PENDING and all(
            jev.STATUS.get(node.uid, ("",))[0] in (('CHECKMARK',) if tree.name in USED else ('CHECKMARK', 'INFO'))
            for tree, node in nodes)

    tree, ids = build_scene()
    answered = drive(jev, settled, timeout=120)
    obj = bpy.data.objects["A"]  # the e2e's edit, then back to the first prompt, answered from the node's entries
    for prompt in ("something to lie on the floor", SCENES["A"][1]):
        obj.modifiers["Jev"][ids["Pick Prompt"]] = prompt
        obj.update_tag()
        jev.schedule(immediate=True)
        answered = answered and drive(jev, settled, timeout=60)
    copy_tree = tree.copy()  # a copy that no modifier uses, kept in the file
    copy_tree.name, copy_tree.use_fake_user = "JevTest Copy", True
    jev.run_pass()
    answered = answered and drive(jev, settled, timeout=60)
    check("build: the prototype answered every node through the fake Jev server", answered,
          [(t.name, n.name, jev.STATUS.get(n.uid)) for t, n in jev.iter_jev_nodes()])
    values = outputs()
    print("v0.1.0 outputs:", values, flush=True)
    check("build: the picked collection child's Instance is realised, and the answers differ between objects",
          values["A"]["verts"] == 4 + list(PROPS.values())[values["A"]["lamp_index"]]
          and values["A"]["night"] != values["B"]["night"], values)
    groups = [t.name for t in bpy.data.node_groups if t.name.startswith(".Jev ")]
    check(f"build: each of the {OLD_NODES} old nodes has its own .Jev inner group",
          len(legacy_nodes()) == OLD_NODES == len({n.node_tree.name for _t, n in legacy_nodes()}) == len(groups),
          (len(legacy_nodes()), len(groups)))
    bpy.ops.wm.save_as_mainfile(filepath=V010)
    listener.shutdown()
    return values


# ------------------------------------------------------------------------------ both add-ons enabled

def phase_both_enabled():
    """jev_nodes enabled with its own settings, then OpenAct enabled beside it."""
    old = bpy.context.preferences.addons["jev_nodes"].preferences
    custom = {"runtime_dir": os.path.join(TMP, "custom runtime"), "port": free_port(), "device": 'cuda', "gpu": 1,
              "auto_start": False, "debias": False}
    for name, value in custom.items():
        setattr(old, name, value)
    enabled = addon_utils.enable("openact", default_set=True, handle_error=print)
    check("both enabled: OpenAct enables", enabled is not None)
    from openact import ui
    from openact import prefs as P

    got = {name: getattr(P.get_prefs(), name) for name in custom}
    check("both enabled: OpenAct takes jev_nodes' runtime folder, port, device, GPU, auto-start and order balancing",
          got == custom, got)
    registered = bpy.types.Node.bl_rna_get_subclass_py("GeometryNodeJevAsk")
    check("both enabled: OpenAct registers no legacy node types, so jev_nodes keeps its own",
          registered is not None and registered.__module__.startswith("jev_nodes"), registered)
    banners = [banner for banner in ui.live_banners(bpy.context) if banner[0] == "jev_nodes"]
    check("both enabled: the sidebar says to disable Jev Nodes",
          banners == [("jev_nodes", 'ERROR', "Disable Jev Nodes: OpenAct replaces it", ())], banners)

    addon_utils.disable("openact", default_set=False)  # its preferences stay, as across a restart
    old.port = free_port()
    addon_utils.enable("openact", default_set=False, handle_error=print)
    check("the import runs once: registering OpenAct again keeps its own settings",
          P.get_prefs().port == custom["port"], P.get_prefs().port)

    addon_utils.disable("openact", default_set=False)
    addon_utils.disable("jev_nodes", default_set=True)  # as the banner says
    addon_utils.enable("openact", default_set=False, handle_error=print)
    registered = bpy.types.Node.bl_rna_get_subclass_py("GeometryNodeJevAsk")
    check("OpenAct alone: the legacy node types are its frozen aliases",
          registered is not None and registered.__module__ == "openact.nodes", registered)
    return custom


# ------------------------------------------------------------------------------ the upgrade

def phase_upgrade(v010, custom):
    from openact import engine, inner, migrate, nodes, ops, pipeline, server, ui
    from openact import prefs as P
    from openact import store as S

    prefs = P.get_prefs()
    prefs.port, prefs.auto_start, prefs.device, prefs.gpu, prefs.runtime_dir = free_port(), True, 'auto', 0, ""
    server.FAKE_ARGS[:] = ["--record", RECORD]
    server.start()
    check("a fake model server is ready: a missing cell would be sent at once",
          drive(engine, lambda: server.status()[0] == "ready", timeout=60), server.status())
    asked = []
    submit = engine._submit
    engine._submit = lambda job, missing: asked.extend(missing) or submit(job, missing)  # every missing cell

    def settled():
        """Every node of the used trees shows its answer, and nothing is queued or in flight."""
        found = list(engine.iter_nodes())
        return bool(found) and not pipeline.PENDING and all(
            getattr(engine.STATUS.get(node.uid), "icon", "") in (
                ('CHECKMARK',) if tree.name in USED else ('CHECKMARK', 'INFO')) for tree, node in found)

    with captured() as out:
        bpy.ops.wm.open_mainfile(filepath=V010)
    old = legacy_nodes()
    tree = bpy.data.node_groups["JevTest"]
    check("loaded: the old nodes are OpenAct's defined, frozen aliases",
          len(old) == OLD_NODES and all(type(node) in nodes.legacy_classes for _t, node in old),
          sorted({type(node).__name__ for _t, node in old}))
    check("loaded: an INFO report points to the upgrade",
          f"Info: This file has {OLD_NODES} old Jev nodes: upgrade them in the OpenAct sidebar" in out.text,
          out.text[-300:])
    check("loaded: the sidebar banner offers the upgrade",
          ("legacy", 'INFO', f"This file has {OLD_NODES} old Jev nodes: Upgrade", ("openact.upgrade_legacy",))
          in ui.live_banners(bpy.context), migrate.banner())
    try:
        tree.nodes.new("GeometryNodeJevAsk")
        added = True
    except RuntimeError:
        added = False
    check("loaded: an old node can't be added, so it is never in the Add menu",
          not added and not nodes.GeometryNodeJevAsk.poll(tree) and not nodes.GeometryNodeJevPick.poll(tree))
    labels = []
    layout = type("Layout", (), {"label": lambda self, text="", icon='NONE': labels.append((text, icon))})()
    nodes._LegacyNode.draw_buttons(old[0][1], bpy.context, layout)  # as Blender calls it, with a stand-in layout
    check("loaded: an old node's status line says to upgrade it", labels == [(nodes.LEGACY_STATUS, 'INFO')], labels)
    check("loaded: the old nodes evaluate their saved inner groups as v0.1.0 did", outputs() == v010, outputs())
    saved = {f"{node.name}@{t.name}": node.entries for t, node in old}
    engine.schedule(immediate=True)
    drive(engine, lambda: False, timeout=1.0)  # passes run, with nothing to plan
    check("loaded: frozen: nothing is asked, planned or written, and no status is kept for them",
          not asked and requests() == 0 and not engine.STATUS
          and {f"{node.name}@{t.name}": node.entries for t, node in legacy_nodes()} == saved, (len(asked), requests()))

    before = {f"{node.name}@{t.name}": {
        "type": node.bl_idname, "uid": node.uid, "group": node.node_tree.as_pointer(), "links": links_of(node),
        "settings": settings_of(node), "mode": getattr(node, "mode", None), "entries": node.entries,
        "label": "Ask" if node.bl_idname == "GeometryNodeJevAsk" else "Pick"} for t, node in old}
    order = join_order()
    wm = bpy.context.window_manager
    bpy.ops.ed.undo_push(message="Before the upgrade")  # in background mode, this starts the undo system
    registered = len(wm.operators)
    with captured() as out:
        result = bpy.ops.openact.upgrade_legacy('EXEC_DEFAULT', True)  # as a button runs it: registered, undoable
    check("upgrade: one registered operator run, with its undo step",
          result == {'FINISHED'} and len(wm.operators) == registered + 1
          and wm.operators[-1].bl_idname == "OPENACT_OT_upgrade_legacy"
          and ops.OPENACT_OT_upgrade_legacy.bl_options == {'REGISTER', 'UNDO'}, (result, len(wm.operators)))
    check("upgrade: the report counts the nodes and node groups upgraded, and the problems",
          f"Info: Upgraded {OLD_NODES} nodes in {OLD_TREES} node groups; 0 problems" in out.text, out.text[-300:])
    check("upgrade: no old node is left, and so no banner", not legacy_nodes() and migrate.banner() is None)

    new = {key: bpy.data.node_groups[key.split("@", 1)[1]].nodes.get(key.split("@", 1)[0]) for key in before}
    kinds = {key: getattr(node, "bl_idname", None) for key, node in new.items()}
    check("upgrade: each old node's place holds its OpenAct node, under its name",
          all(kinds[key] == engine.LEGACY_TYPES[record["type"]] for key, record in before.items()), kinds)
    if not all(new.values()):
        raise Stop
    check("upgrade: every uid is kept", all(node.uid == before[key]["uid"] for key, node in new.items()))
    check("upgrade: every inner group is the old data-block, renamed .OpenAct <node> <uid>, owned by its node alone",
          all(node.node_tree.as_pointer() == before[key]["group"] and node.node_tree.name == inner.inner_name(node)
              == f".OpenAct {before[key]['label']} {node.uid}" and node.node_tree.users == 1
              for key, node in new.items())
          and not [t.name for t in bpy.data.node_groups if t.name.startswith(".Jev ")],
          [(node.node_tree.name, node.node_tree.users) for node in new.values()])
    check("upgrade: every link is kept, on the sockets of the same names",
          all(links_of(node) == before[key]["links"] for key, node in new.items()),
          [key for key, node in new.items() if links_of(node) != before[key]["links"]])
    check("upgrade: a multi-input socket joins its links in the same order", join_order() == order,
          (order, join_order()))
    check("upgrade: frame, location, label, width, colour, mute and collapse are kept",
          all(settings_of(node) == before[key]["settings"] for key, node in new.items()),
          [(key, settings_of(node)) for key, node in new.items() if settings_of(node) != before[key]["settings"]])
    asks = {key: node for key, node in new.items() if node.bl_idname == engine.ASK_ID}
    picks = {key: node for key, node in new.items() if node.bl_idname == engine.PICK_ID}
    check("upgrade: Ask's mode is its answer type",
          all(node.answer_type == before[key]["mode"] for key, node in asks.items())
          and asks["Ask Jev@JevNested"].answer_type == 'AMOUNT')
    sources = {key: node.list_source for key, node in picks.items()}
    check("upgrade: a Pick whose Collection is set picks from it, even with Items text (v0.1.0's precedence); the "
          "others from their text, all keeping v0.1.0's item parsing",
          sources == {f"{name}@{tree_name}": source for tree_name in ("JevTest", "JevTest Copy") for name, source in
                      (("Jev Pick", 'TEXT'), ("Jev Pick.001", 'COLLECTION'), ("Rest", 'TEXT'), ("Cup", 'COLLECTION'))}
          and all(node.items_parse_v0 for node in picks.values()), sources)
    check("upgrade: Balance Option Order starts from the preference, jev_nodes' imported setting",
          all(node.balance_order == custom["debias"] for node in new.values()))
    check("upgrade: the answers are kept, as v1 stores them: v1 words keys, asked with jev_nodes 0.1.0",
          all(S.load(node).entries == S.load(Stand(before[key]["entries"])).entries and S.load(node).entries
              and all(entry.legacy and entry.words_key and entry.asks == [S.V010] for entry in S.load(node).entries)
              for key, node in new.items()), [key for key, node in new.items() if not S.load(node).entries])

    check("after the upgrade: every node shows its answer", drive(engine, settled, timeout=30),
          [(t.name, n.name, getattr(engine.STATUS.get(n.uid), "text", None)) for t, n in engine.iter_nodes()])
    check("after the upgrade: 0 requests reach the ready model server, and no cell is planned to be asked",
          requests() == 0 and not asked, (requests(), len(asked)))
    check("after the upgrade: the outputs are identical to v0.1.0's", outputs() == v010, (outputs(), v010))
    planned = [row.name for row in engine.STATUS[new["Rest@JevTest"].uid].uses[0][1].rows]
    check("after the upgrade: Items with ';' and ': ' keep their v0.1.0 names",
          planned == ["Chair: comfy; Stool", "Bench: long"], planned)
    shown = [(node, entry) for node in new.values() for _label, _job, entry in engine.STATUS[node.uid].uses if entry]
    check("after the upgrade: the answers shown are the converted ones, labelled as asked with older settings",
          shown and all(entry.legacy and engine.older_settings(node, entry) for node, entry in shown), len(shown))

    new["Ask Jev@JevTest"].inputs["Question"].default_value = "Is it dark?"
    drive(engine, lambda: requests() > 0 and settled(), timeout=30)
    check("new words are asked: the request count works", requests() > 0 and asked, (requests(), len(asked)))
    return before


# ------------------------------------------------------------------------------ undo

def phase_undo(v010, before):
    from openact import engine, pipeline

    sent, answers = requests(), len(pipeline.MEMORY)
    bpy.ops.ed.undo()
    old = {f"{node.name}@{t.name}": node for t, node in legacy_nodes()}
    check("undo: the old nodes are back, with their uids, answers and .Jev inner groups",
          set(old) == set(before) and all(
              (node.bl_idname, node.uid, node.entries, node.node_tree.name)
              == (before[key]["type"], before[key]["uid"], before[key]["entries"],
                  f".Jev {before[key]['label']} {before[key]['uid']}") for key, node in old.items()), sorted(old))
    check("undo: no OpenAct node and no .OpenAct inner group is left",
          not list(engine.iter_nodes()) and not [t for t in bpy.data.node_groups if t.name.startswith(".OpenAct ")])
    check("undo: the outputs are v0.1.0's again", outputs() == v010, outputs())
    drive(engine, lambda: False, timeout=1.0)
    check("undo: the passes after it keep no status and ask nothing",
          not engine.STATUS and not pipeline.PENDING and (requests(), len(pipeline.MEMORY)) == (sent, answers),
          (len(engine.STATUS), requests(), sent))


def phase_problem(v010, before):
    """In the file as undo left it: a text Pick whose Instance output is linked stays frozen, and is named; a node that
    lost its inner group gets a new one, built from its answers."""
    from openact import engine, inner, migrate

    tree = bpy.data.node_groups["JevTest"]
    rest = tree.nodes["Rest"]
    realize = tree.nodes.new("GeometryNodeRealizeInstances")
    tree.links.new(rest.outputs["Instance"], realize.inputs["Geometry"])
    entries, group = rest.entries, rest.node_tree.name
    bpy.data.node_groups["JevTest Copy"].nodes["Muted"].node_tree = None
    with captured() as out:
        result = bpy.ops.openact.upgrade_legacy()
    check("a problem: the report names the node left as it was, and why",
          result == {'FINISHED'} and f"Warning: Upgraded {OLD_NODES - 1} nodes in {OLD_TREES} node groups; 1 problem: "
          "'Rest' in JevTest: its Instance output is linked" in out.text, out.text[-400:])
    old = [(t.name, n.name) for t, n in legacy_nodes()]
    check("a problem: that node stays frozen, with its answers, inner group and links, and the banner counts it",
          old == [("JevTest", "Rest")] and (rest.entries, rest.node_tree.name) == (entries, group)
          and realize.inputs["Geometry"].is_linked and migrate.banner()[0] == "This file has 1 old Jev node: Upgrade",
          old)
    check("a problem: no new node or inner group is left behind for it",
          len([t for t in bpy.data.node_groups if t.name.startswith(".OpenAct ")]) == OLD_NODES - 1
          and len(list(engine.iter_nodes())) == OLD_NODES - 1)
    muted = bpy.data.node_groups["JevTest Copy"].nodes["Muted"]
    check("a node without its inner group gets a new one, named for its kept uid",
          muted.uid == before["Muted@JevTest Copy"]["uid"] and muted.node_tree.name == inner.inner_name(muted),
          (muted.uid, before["Muted@JevTest Copy"]["uid"], muted.node_tree.name))
    check("a problem: the outputs stay v0.1.0's", outputs() == v010, outputs())


def phase_jev_later():
    """jev_nodes enabled after OpenAct: Blender hands it the legacy node types, and OpenAct still disables cleanly."""
    addon_utils.enable("jev_nodes", default_set=True, handle_error=print)  # its default runtime is not installed here
    registered = bpy.types.Node.bl_rna_get_subclass_py("GeometryNodeJevAsk")
    errors = []
    addon_utils.disable("openact", default_set=True, handle_error=errors.append)
    check("jev_nodes enabled after OpenAct takes its node types back, and OpenAct still disables cleanly",
          registered is not None and registered.__module__.startswith("jev_nodes") and not errors, (registered, errors))


def main():
    print("BLENDER", bpy.app.version_string, "migrate", flush=True)
    try:
        found = os.path.isdir(os.path.join(PROTOTYPE, "jev_nodes"))
        check(f"the jev_nodes prototype is at {PROTOTYPE}", found)
        if not found:
            raise Stop
        v010 = phase_build()
        custom = phase_both_enabled()
        before = phase_upgrade(v010, custom)
        phase_undo(v010, before)
        phase_problem(v010, before)
        phase_jev_later()
    except Stop:  # a check failed that the rest depends on
        pass
    except Exception:
        check("no unexpected error", False, traceback.format_exc())
    finally:
        if "openact.server" in sys.modules:
            sys.modules["openact.server"].shutdown_owned()  # the fake model server this test started
    if not FAILURES:
        shutil.rmtree(TMP, ignore_errors=True)
    finish()


main()
