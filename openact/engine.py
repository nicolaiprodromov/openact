"""Keeps the OpenAct nodes answered.

Geometry Nodes cannot run Python while it evaluates (Blender has no script node), so an OpenAct
node is answered outside evaluation:

1. After a change, each OpenAct node's inputs are read by the static evaluator (evaluator.py).
2. The local model is asked in the background (see server.py).
3. The answer is written into the node's hidden node group as a small lookup table keyed by the
   exact input strings. Geometry Nodes picks the matching answer by itself, so one node group
   can be used on several objects with different prompts, answers switch instantly back and
   forth, and they are saved with the file (Blender 4.5+ even evaluates them without the add-on).
"""

from __future__ import annotations

import hashlib
import json
import time
import traceback
import uuid

import bpy
from bpy.app.handlers import persistent

from . import evaluator, natsort, server
from . import prefs as P
from .evaluator import Ctx, Hint, Pending, Unresolvable, as_text, eval_input, find_uses, is_field

ASK_ID = "GeometryNodeOpenActAsk"
PICK_ID = "GeometryNodeOpenActPick"
OPENACT_TYPES = (ASK_ID, PICK_ID)
INNER_PREFIX = ".OpenAct "
MAX_ENTRIES = 16  # answers remembered per node
MAX_ITEMS = 255
DEBOUNCE = 0.25
AMOUNT_LEVELS = ["Not at all", "Slightly", "Moderately", "Very", "Extremely"]
NO_CONTEXT = "No additional context."
PICK_STATE = "Choose one of the listed items."

MEMORY = {}   # request key -> parsed answer (this session)
PENDING = {}  # request key -> set of node uids waiting for it
FAILED = {}   # request key -> error message (not retried until inputs change or Re-ask)
STATUS = {}   # node uid -> (icon, text)

_dirty = True
_last_change = 0.0
_last_refresh = 0.0
_last_pass = 0.0
_has_nodes = False
_group_count = -1  # number of node groups at the last pass that found no OpenAct nodes
SAFETY_PASS = 2.0  # edits to trees no modifier uses don't send depsgraph updates


# ------------------------------------------------------------------------ node specs


def _sock(sockets, identifier):
    for sock in sockets:
        if sock.identifier == identifier:
            return sock
    raise KeyError(identifier)


def _openact_output(node, out, ctx, depth):
    """Evaluator handler: the answer an OpenAct node shows for the words it gets in this context."""
    if is_field(out):
        raise Unresolvable(f"'{node.name}' output '{out.name}' is a field; words must be single values")
    spec = SPECS[node.bl_idname]
    try:
        req = spec.request(node, ctx)
    except Hint:
        req = None
    if req is None:
        return spec.fallback(out.name)
    answer = MEMORY.get(req.key)
    if answer is None:
        answer = next((e["a"] for e in get_entries(node) if e.get("key") == req.key), None)
    if answer is None:
        raise Pending()
    return spec.output(answer, out.name)


evaluator.HANDLERS.update(dict.fromkeys(OPENACT_TYPES, _openact_output))


class Req:
    __slots__ = ("gn", "key", "payload", "meta")

    def __init__(self, gn, key, payload, meta=None):
        self.gn, self.key, self.payload, self.meta = tuple(gn), key, payload, meta or {}


def _key(*parts):
    return hashlib.sha1(json.dumps(parts, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def parse_items(text):
    """Items are separated by commas, or by new lines when there are any (e.g. Import Text)."""
    if not text:
        return []
    parts = text.splitlines() if ("\n" in text or "\r" in text) else text.split(",")
    return [part.strip() for part in parts if part.strip()]


def collection_items(collection):
    """Children in the order Collection Info (Separate Children) instances them."""
    names = [obj.name for obj in collection.objects] + [child.name for child in collection.children]
    return sorted(names, key=natsort.sort_key)


class AskSpec:
    label = "Ask"
    keys = ("Question", "Context")
    empty_hint = "Type a question"
    iface = (
        ('INPUT', "Question", "NodeSocketString",
         "The question, e.g. 'Is this scene set at night?' (Amount mode: 'How weathered should it look?')"),
        ('INPUT', "Context", "NodeSocketString",
         "Optional facts the question is about, e.g. a scene description"),
        ('OUTPUT', "Answer", "NodeSocketFloat",
         "Yes / No: probability of yes (0-1). Amount: how much (0-1). 0 until answered"),
        ('OUTPUT', "Yes", "NodeSocketBool", "True when Answer is 0.5 or more"),
        ('OUTPUT', "Confidence", "NodeSocketFloat",
         "How clearly the top answer wins over the runner-up (0 = unsure, 1 = certain)"),
    )

    @staticmethod
    def request(node, ctx):
        q_raw = as_text(eval_input(node.inputs["Question"], ctx))
        c_raw = as_text(eval_input(node.inputs["Context"], ctx))
        question, context = q_raw.strip(), c_raw.strip()
        if not question:
            raise Hint(AskSpec.empty_hint)
        debias = bool(P.get_prefs().debias)
        if node.mode == 'AMOUNT':
            typed = {"type": "score", "instructions": question, "criteria": AMOUNT_LEVELS}
        else:
            typed = {"type": "noul", "instructions": question}
        payload = {
            "model": server.MODEL_NAME,
            "state": context or NO_CONTEXT,
            "questions": {"answer": typed},
            "options": {"debias": debias},
        }
        return Req((q_raw, c_raw), _key("ask", node.mode, question, context, debias), payload)

    @staticmethod
    def parse(req, answers):
        answer = answers["answer"]
        if answer["type"] == "score":
            value = min(max(float(answer["score"]) / (len(AMOUNT_LEVELS) - 1), 0.0), 1.0)
            return {"mode": 'AMOUNT', "value": value, "confidence": float(answer.get("confidence", 0.0)),
                    "levels": [float(answer["probabilities"].get(str(i), 0.0)) for i in range(len(AMOUNT_LEVELS))]}
        p = float(answer["noul"])
        return {"mode": 'YES_NO', "value": p, "confidence": abs(2.0 * p - 1.0)}

    @staticmethod
    def output(answer, name):
        if name == "Answer":
            return answer["value"]
        if name == "Yes":
            return answer["value"] >= 0.5
        if name == "Confidence":
            return answer["confidence"]
        raise Unresolvable(f"unknown output {name}")

    @staticmethod
    def fallback(name):
        return {"Answer": 0.0, "Yes": False, "Confidence": 0.0}.get(name)

    @staticmethod
    def summary(answer):
        if answer["mode"] == 'AMOUNT':
            return f"Amount {answer['value']:.2f}"
        p = answer["value"]
        return f"{'Yes' if p >= 0.5 else 'No'}  ({p * 100:.0f}% yes)"

    @staticmethod
    def build(b, gi, go, entries):
        index = b.match_chain([gi.outputs["Question"], gi.outputs["Context"]], [e["gn"] for e in entries])
        answers = [e["a"] for e in entries]
        b.lookup(index, 'FLOAT', 0.0, [a["value"] for a in answers], go.inputs["Answer"], 200)
        b.lookup(index, 'BOOLEAN', False, [a["value"] >= 0.5 for a in answers], go.inputs["Yes"], 0)
        b.lookup(index, 'FLOAT', 0.0, [a["confidence"] for a in answers], go.inputs["Confidence"], -200)


class PickSpec:
    label = "Pick"
    keys = ("Prompt", "Items")
    empty_hint = "Type a prompt and items"
    iface = (
        ('INPUT', "Prompt", "NodeSocketString", "What to pick, e.g. 'something to sit on'"),
        ('INPUT', "Items", "NodeSocketString",
         "Items separated by commas (or one per line, e.g. from Import Text). Ignored when a Collection is set"),
        ('INPUT', "Collection", "NodeSocketCollection",
         "Pick one of this collection's children instead, in the same order as Collection Info"),
        ('OUTPUT', "Index", "NodeSocketInt",
         "Position of the picked item, -1 until answered. Matches Collection Info / Pick Instance order"),
        ('OUTPUT', "Item", "NodeSocketString", "Name of the picked item"),
        ('OUTPUT', "Confidence", "NodeSocketFloat", "How clearly the pick wins over the runner-up (0-1)"),
        ('OUTPUT', "Instance", "NodeSocketGeometry",
         "The picked collection child as an instance (empty when picking from text)"),
    )

    @staticmethod
    def request(node, ctx):
        prompt_raw = as_text(eval_input(node.inputs["Prompt"], ctx))
        items_raw = as_text(eval_input(node.inputs["Items"], ctx))
        collection = eval_input(node.inputs["Collection"], ctx)
        if not isinstance(collection, bpy.types.Collection):
            collection = None
        prompt = prompt_raw.strip()
        items = collection_items(collection) if collection is not None else parse_items(items_raw)
        if not prompt:
            raise Hint("Type a prompt")
        if not items:
            raise Hint("The collection is empty" if collection is not None else
                       "Type items separated by commas, or set a collection")
        if len(items) > MAX_ITEMS:
            raise Unresolvable(f"Too many items ({len(items)}); the limit is {MAX_ITEMS}")
        debias = bool(P.get_prefs().debias)
        payload = {
            "model": server.MODEL_NAME,
            "state": PICK_STATE,
            "questions": {"pick": {
                "type": "choice",
                "instructions": f"Which item best fits: {prompt.rstrip('?').strip()}?",
                "criteria": {str(i): name for i, name in enumerate(items)},
            }},
            "options": {"debias": debias},
        }
        meta = {"items": items, "collection": collection.name if collection else ""}
        return Req((prompt_raw, items_raw), _key("pick", prompt, items, debias), payload, meta)

    @staticmethod
    def parse(req, answers):
        answer = answers["pick"]
        items = req.meta["items"]
        index = int(answer["choice"])
        probabilities = [float(answer["probabilities"].get(str(i), 0.0)) for i in range(len(items))]
        return {"index": index, "item": items[index], "confidence": float(answer.get("confidence", 0.0)),
                "probabilities": probabilities, "items": items, "collection": req.meta["collection"]}

    @staticmethod
    def output(answer, name):
        if name == "Index":
            return answer["index"]
        if name == "Item":
            return answer["item"]
        if name == "Confidence":
            return answer["confidence"]
        raise Unresolvable("the Instance output is geometry, not text")

    @staticmethod
    def fallback(name):
        return {"Index": -1, "Item": "", "Confidence": 0.0}.get(name)

    @staticmethod
    def summary(answer):
        p = answer["probabilities"][answer["index"]]
        return f"{answer['item']}  ({p * 100:.0f}%)"

    @staticmethod
    def build(b, gi, go, entries):
        index = b.match_chain([gi.outputs["Prompt"], gi.outputs["Items"]], [e["gn"] for e in entries])
        answers = [e["a"] for e in entries]
        picked = b.lookup(index, 'INT', -1, [a["index"] for a in answers], go.inputs["Index"], 300)
        b.lookup(index, 'STRING', "", [a["item"] for a in answers], go.inputs["Item"], 100)
        b.lookup(index, 'FLOAT', 0.0, [a["confidence"] for a in answers], go.inputs["Confidence"], -100)
        info = b.new("GeometryNodeCollectionInfo", 150, -350)
        b.link(gi.outputs["Collection"], _sock(info.inputs, "Collection"))
        _sock(info.inputs, "Separate Children").default_value = True
        _sock(info.inputs, "Reset Children").default_value = True
        position = b.new("GeometryNodeInputIndex", 150, -560)
        other = b.new("FunctionNodeCompare", 350, -520)
        other.data_type = 'INT'
        other.operation = 'NOT_EQUAL'
        b.link(position.outputs[0], _sock(other.inputs, "A_INT"))
        b.link(picked, _sock(other.inputs, "B_INT"))
        delete = b.new("GeometryNodeDeleteGeometry", 550, -350)
        delete.domain = 'INSTANCE'
        b.link(info.outputs[0], _sock(delete.inputs, "Geometry"))
        b.link(other.outputs[0], _sock(delete.inputs, "Selection"))
        b.link(delete.outputs[0], go.inputs["Instance"])


SPECS = {ASK_ID: AskSpec, PICK_ID: PickSpec}


# ------------------------------------------------------------------------ inner trees


class _Builder:
    def __init__(self, tree):
        self.tree = tree

    def new(self, idname, x, y):
        node = self.tree.nodes.new(idname)
        node.location = (x, y)
        return node

    def link(self, a, b):
        return self.tree.links.new(a, b)

    def match_chain(self, sockets, keys):
        """Integer socket: 1 + index of the entry whose key strings equal the inputs, else 0."""
        previous = None
        for i, key in enumerate(keys, start=1):
            y = 400 - i * 110
            conditions = []
            for j, (sock, value) in enumerate(zip(sockets, key)):
                compare = self.new("FunctionNodeCompare", -600 + j * 40, y - j * 40)
                compare.data_type = 'STRING'
                compare.operation = 'EQUAL'
                compare.hide = True
                self.link(sock, _sock(compare.inputs, "A_STR"))
                _sock(compare.inputs, "B_STR").default_value = value
                conditions.append(compare.outputs[0])
            condition = conditions[0]
            for other in conditions[1:]:
                both = self.new("FunctionNodeBooleanMath", -380, y)
                both.operation = 'AND'
                both.hide = True
                self.link(condition, both.inputs[0])
                self.link(other, both.inputs[1])
                condition = both.outputs[0]
            switch = self.new("GeometryNodeSwitch", -200, y)
            switch.input_type = 'INT'
            switch.hide = True
            self.link(condition, _sock(switch.inputs, "Switch"))
            if previous is None:
                _sock(switch.inputs, "False").default_value = 0
            else:
                self.link(previous, _sock(switch.inputs, "False"))
            _sock(switch.inputs, "True").default_value = i
            previous = switch.outputs[0]
        return previous

    def lookup(self, index, data_type, fallback, values, target, y):
        node = self.new("GeometryNodeIndexSwitch", 150, y)
        node.data_type = data_type
        items = node.index_switch_items
        items.clear()
        for _ in range(len(values) + 1):
            items.new()
        slots = [s for s in node.inputs if s.identifier not in ("Index", "__extend__")]
        slots[0].default_value = fallback
        for slot, value in zip(slots[1:], values):
            slot.default_value = value
        if index is not None:
            self.link(index, _sock(node.inputs, "Index"))
        self.link(node.outputs[0], target)
        return node.outputs[0]


def new_uid():
    return uuid.uuid4().hex[:12]


def inner_name(node):
    return f"{INNER_PREFIX}{SPECS[node.bl_idname].label} {node.uid}"


def create_inner(node):
    spec = SPECS[node.bl_idname]
    tree = bpy.data.node_groups.new(inner_name(node), 'GeometryNodeTree')
    for in_out, name, socket_type, description in spec.iface:
        item = tree.interface.new_socket(name, in_out=in_out, socket_type=socket_type)
        item.description = description
    node.node_tree = tree
    build_inner(node)
    return tree


def build_inner(node):
    """Rebuild the hidden node group from the node's stored answers."""
    tree = node.node_tree
    if tree is None:
        create_inner(node)
        return
    tree.nodes.clear()
    builder = _Builder(tree)
    group_in = builder.new("NodeGroupInput", -900, 0)
    group_out = builder.new("NodeGroupOutput", 800, 0)
    SPECS[node.bl_idname].build(builder, group_in, group_out, get_entries(node))


def get_entries(node):
    try:
        data = json.loads(node.entries or "[]")
    except ValueError:
        return []
    return data if isinstance(data, list) else []


def set_entries(node, entries):
    node.entries = json.dumps(entries, ensure_ascii=False)


def ensure_owned(nodes):
    """Give every OpenAct node a unique uid and its own hidden group.

    Copies (duplicated nodes or trees, appended data) arrive sharing the original's group; the
    rightful owner is the node whose uid matches the group's name, everyone else gets a copy.
    Linked data is read-only: linked nodes are left alone and keep their uids, so a local node that
    shares one (appended from the library that is also linked) gets a new uid. A local node whose
    group is linked (a tree made local on its own) gets a local copy."""
    seen = {node.uid for tree, node in nodes if tree.library is not None}
    nodes = [(tree, node) for tree, node in nodes if tree.library is None]
    owners = {}
    for _tree, node in nodes:
        inner = node.node_tree
        if node.uid and inner is not None and inner.name == inner_name(node):
            owners.setdefault(inner.as_pointer(), node)
    for _tree, node in nodes:
        fresh = not node.uid or node.uid in seen
        if fresh:
            node.uid = new_uid()
        seen.add(node.uid)
        inner = node.node_tree
        if inner is None:
            create_inner(node)
        elif inner.library is not None or owners.setdefault(inner.as_pointer(), node) is not node:
            own = inner.copy()
            own.name = inner_name(node)
            node.node_tree = own
        elif fresh:  # frees the old uid's group name. Only once: a taken name would be retried every pass
            inner.name = inner_name(node)


def iter_nodes():
    for tree in bpy.data.node_groups:
        if tree.bl_idname != 'GeometryNodeTree' or tree.name.startswith(INNER_PREFIX):
            continue
        for node in tree.nodes:
            if node.bl_idname in OPENACT_TYPES:
                yield tree, node


def find_node(uid):
    for tree, node in iter_nodes():
        if node.uid == uid:
            return tree, node
    return None, None


# ----------------------------------------------------------------------------- passes


def schedule(immediate=False):
    global _dirty, _last_change
    _dirty = True
    _last_change = 0.0 if immediate else time.monotonic()


def run_pass():
    """Bring every OpenAct node up to date. Main thread only."""
    global _dirty, _has_nodes, _last_pass, _group_count
    _dirty = False
    _last_pass = time.monotonic()
    nodes = list(iter_nodes())
    _has_nodes = bool(nodes)
    _group_count = -1 if nodes else len(bpy.data.node_groups)
    if not nodes:
        return
    ensure_owned(nodes)
    uses = find_uses()
    evaluator.MEMO = {}  # nodes that read the same upstream words in a context share one read
    try:
        for tree, node in nodes:
            try:
                _process(tree, node, uses.get(tree.as_pointer()))
            except Exception as error:
                traceback.print_exc()
                STATUS[node.uid] = ('ERROR', f"Internal error: {error}")
    finally:
        evaluator.MEMO = None  # keyed by pointers: after the pass, nodes and data may be freed or edited
    redraw()


def _process(tree, node, contexts):
    if tree.library is not None:  # read-only: the library file holds this node's answers
        STATUS[node.uid] = ('LINKED', f"Linked from {tree.library.name}: answers come from that file")
        return
    spec = SPECS[node.bl_idname]
    contexts = contexts or [Ctx(None, None, (), tree)]
    wanted = {}
    problems, hints = [], []
    upstream = False
    for ctx in contexts:
        try:
            req = spec.request(node, ctx)
        except Pending:
            upstream = True
            continue
        except Hint as hint:
            hints.append(str(hint))
            continue
        except Unresolvable as error:
            problems.append(str(error))
            continue
        other = wanted.get(req.gn)
        if other is not None and other.key != req.key:
            problems.append("Same prompt and items used with different collections; only one answer can show")
            continue
        wanted[req.gn] = req

    entries = get_entries(node)
    changed = False
    asks = []
    for gn, req in wanted.items():
        found = next((e for e in entries if tuple(e.get("gn", ())) == gn), None)
        if found is not None and found.get("key") == req.key:
            continue
        if found is not None:  # same strings, different meaning (collection or settings changed)
            entries.remove(found)
            changed = True
        answer = MEMORY.get(req.key)
        if answer is not None:
            entries.append({"key": req.key, "gn": list(gn), "a": answer, "t": time.time()})
            changed = True
        elif req.key not in FAILED:
            asks.append(req)
    if changed:
        if len(entries) > MAX_ENTRIES:
            keep = [e for e in entries if tuple(e["gn"]) in wanted]
            rest = sorted((e for e in entries if tuple(e["gn"]) not in wanted), key=lambda e: e.get("t", 0.0))
            entries = keep + rest[-max(MAX_ENTRIES - len(keep), 0):] if len(keep) < MAX_ENTRIES else keep
        set_entries(node, entries)
        build_inner(node)
    for req in asks:
        _submit(node, req)
    STATUS[node.uid] = _status(spec, wanted, entries, problems, hints, upstream)


def _submit(node, req):
    waiting = PENDING.get(req.key)
    if waiting is not None:
        waiting.add(node.uid)
        return
    state, _message = server.status()
    if state != "ready":
        if state in ("stopped", "unknown") and P.get_prefs().auto_start:
            server.start()
        return  # asked again by the pass that runs when the server is ready
    PENDING[req.key] = {node.uid}
    spec = SPECS[node.bl_idname]
    server.ask(req.payload, lambda kind, data: _on_answer(req, spec, kind, data))


def _on_answer(req, spec, kind, data):
    PENDING.pop(req.key, None)
    if kind == "ok":
        try:
            MEMORY[req.key] = spec.parse(req, data["answers"])
        except Exception as error:
            FAILED[req.key] = f"Unexpected answer from the model: {error}"
    elif kind == "error":
        FAILED[req.key] = f"The model refused: {data}"
    schedule(immediate=True)


def _status(spec, wanted, entries, problems, hints, upstream):
    if problems:
        return ('ERROR', problems[0])
    failed = [FAILED[req.key] for req in wanted.values() if req.key in FAILED]
    if failed:
        return ('ERROR', failed[0])
    keys = {e.get("key") for e in entries}
    missing = [req for req in wanted.values() if req.key not in keys]
    if missing or upstream:
        state, message = server.status()
        if state == "not_installed":
            return ('ERROR', "Model not installed: open the OpenAct tab in the sidebar")
        if state == "installing":
            return ('SORTTIME', "Installing the model...")
        if state == "error":
            return ('ERROR', message or "Model server error")
        if state == "starting":
            return ('SORTTIME', "Starting the model...")
        if state in ("stopped", "unknown") and not P.get_prefs().auto_start:
            return ('PAUSE', "Model server is stopped (OpenAct tab)")
        if upstream and not missing:
            return ('SORTTIME', "Waiting for another OpenAct node...")
        return ('SORTTIME', "Asking...")
    if not wanted:
        return ('INFO', hints[0] if hints else spec.empty_hint)
    first = next(iter(wanted.values()))
    entry = next(e for e in entries if e.get("key") == first.key)
    text = spec.summary(entry["a"])
    if len(wanted) > 1:
        text += f"   +{len(wanted) - 1} more"
    return ('CHECKMARK', text)


def answers_for(node):
    """(context label, answer) for the node's current uses, for the sidebar."""
    tree = node.id_data
    contexts = find_uses().get(tree.as_pointer()) or [Ctx(None, None, (), tree)]
    spec = SPECS[node.bl_idname]
    entries = {e.get("key"): e["a"] for e in get_entries(node)}
    rows = []
    for ctx in contexts:
        try:
            req = spec.request(node, ctx)
        except (Hint, Pending, Unresolvable):
            continue
        rows.append((ctx.label(), entries.get(req.key) or MEMORY.get(req.key)))
    return rows


def reask(node):
    """Forget this node's answers and ask again."""
    if node.id_data.library is not None:  # linked data is read-only
        return
    for entry in get_entries(node):
        MEMORY.pop(entry.get("key"), None)
    FAILED.clear()
    set_entries(node, [])
    build_inner(node)
    schedule(immediate=True)


def redraw():
    wm = getattr(bpy.context, "window_manager", None)
    if wm is None:
        return
    for window in wm.windows:
        for area in window.screen.areas:
            if area.type in ('NODE_EDITOR', 'PREFERENCES'):
                area.tag_redraw()


# ------------------------------------------------------------------ handlers & timer


def _playing():
    """Don't touch node trees during playback or while a render job runs."""
    if bpy.app.is_job_running('RENDER'):
        return True
    wm = getattr(bpy.context, "window_manager", None)
    return bool(wm) and any(w.screen and w.screen.is_animation_playing for w in wm.windows)


def tick():
    global _last_refresh
    try:
        if server.pump():
            redraw()
        now = time.monotonic()
        if now - _last_refresh > 5.0 and (_has_nodes or server.STATE.status in ("unknown", "ready", "starting")):
            _last_refresh = now
            server.refresh()
        if not _playing():
            if _dirty and now - _last_change >= DEBOUNCE:
                run_pass()
            elif _has_nodes and now - _last_pass >= SAFETY_PASS:
                run_pass()
    except Exception:
        traceback.print_exc()
    return 0.1


@persistent
def on_depsgraph_update(scene, depsgraph):
    global _dirty, _last_change
    # With no OpenAct nodes in the file, only new node groups (append, link) can bring some in;
    # adding or duplicating an OpenAct node, loading, undo and redo schedule a pass on their own.
    if not _has_nodes and len(bpy.data.node_groups) == _group_count:
        return
    _dirty = True
    _last_change = time.monotonic()


@persistent
def on_load_post(*_args):
    STATUS.clear()
    FAILED.clear()
    schedule(immediate=True)


@persistent
def on_undo_redo(*_args):
    schedule(immediate=True)


def on_server_status():
    schedule(immediate=True)
    redraw()


def _handlers():
    h = bpy.app.handlers
    return ((h.depsgraph_update_post, on_depsgraph_update), (h.load_post, on_load_post),
            (h.undo_post, on_undo_redo), (h.redo_post, on_undo_redo))


def register():
    for handlers, fn in _handlers():
        if fn not in handlers:
            handlers.append(fn)
    server.add_status_listener(on_server_status)
    if not bpy.app.timers.is_registered(tick):
        bpy.app.timers.register(tick, first_interval=0.5, persistent=True)


def unregister():
    if bpy.app.timers.is_registered(tick):
        bpy.app.timers.unregister(tick)
    for handlers, fn in _handlers():
        if fn in handlers:
            handlers.remove(fn)
