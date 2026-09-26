"""The hidden node group inside every OpenAct node, `.OpenAct <Node> <uid>`.

It is derived data: build() regenerates it from the node's stored entries (store.py) at any time.
Geometry Nodes selects the entry whose words equal the node's live inputs (the match chain) and
reads that entry's values from Index Switches, so one tree answers every context that uses it, with
no Python at evaluation. The interface is never cleared: sync_interface() diffs it, so the links
from the outer tree survive every rebuild.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass

import bpy

from . import compat
from . import store as _store  # build() and update_values() take `store`, the name §13.3 fixes

INNER_PREFIX = ".OpenAct "
SPECS = {}  # node bl_idname -> spec (label, iface, build); the engine adds the OpenAct nodes
BUILT = "openact_built"  # ID property of the group: which entries build() laid out, for update_values()


@dataclass(frozen=True)
class SocketDef:
    """A socket of a node's interface. sync_interface() matches sockets by (in_out, name, socket_type)."""

    in_out: str  # 'INPUT' or 'OUTPUT'
    name: str
    socket_type: str  # e.g. 'NodeSocketString'
    description: str = ""  # the tooltip
    kind: str = ""  # inputs: "word" (read by the evaluator), "field" (per element), "" (a Geometry Nodes value)
    hidden: bool = False  # hidden on the node when the socket is created; Ctrl H reveals it
    default_input: str = 'VALUE'  # 'INDEX': unlinked, the input is the element's index
    hide_value: bool = False


def sock(sockets, identifier):
    for socket in sockets:
        if socket.identifier == identifier:
            return socket
    raise KeyError(identifier)


def _items(switch):
    """The item inputs of an Index Switch, in order."""
    return [s for s in switch.inputs if s.identifier not in ("Index", "__extend__")]


class _Builder:
    """Adds nodes and links to an inner group. finish() writes every location in one batched call,
    because each single write costs up to a few milliseconds once the tree is large."""

    def __init__(self, tree):
        self.tree = tree
        self.locations = {}  # node pointer -> (x, y)

    def new(self, idname, x, y):
        node = self.tree.nodes.new(idname)
        self.locations[node.as_pointer()] = (x, y)
        return node

    def link(self, a, b):
        return self.tree.links.new(a, b)

    def match_chain(self, sockets, keys):
        """Integer socket: 1 + index of the entry whose key strings equal the inputs, else 0. A key of
        another length (converted from schema 1) never matches: its words sit at other positions."""
        previous = None
        for i, key in enumerate(keys, start=1):
            if len(key) != len(sockets):
                continue
            y = 400 - i * 110
            conditions = []
            for j, (socket, value) in enumerate(zip(sockets, key)):
                compare = self.new("FunctionNodeCompare", -600 + j * 40, y - j * 40)
                compare.data_type = 'STRING'
                compare.operation = 'EQUAL'
                compare.hide = True
                self.link(socket, sock(compare.inputs, "A_STR"))
                sock(compare.inputs, "B_STR").default_value = value
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
            self.link(condition, sock(switch.inputs, "Switch"))
            if previous is None:
                sock(switch.inputs, "False").default_value = 0
            else:
                self.link(previous, sock(switch.inputs, "False"))
            sock(switch.inputs, "True").default_value = i
            previous = switch.outputs[0]
        return previous

    def lookup(self, index, data_type, fallback, values, target, y):
        """An entry-level value: an Index Switch on the match chain, items [fallback, value 1 ...]."""
        node = self.new("GeometryNodeIndexSwitch", 150, y)
        node.data_type = data_type
        items = node.index_switch_items
        items.clear()
        for _ in range(len(values) + 1):
            items.new()
        slots = _items(node)
        slots[0].default_value = fallback
        for slot, value in zip(slots[1:], values):
            slot.default_value = value
        if index is not None:
            self.link(index, sock(node.inputs, "Index"))
        self.link(node.outputs[0], target)
        return node.outputs[0]

    def finish(self):
        nodes = self.tree.nodes
        nodes.foreach_set("location", [v for node in nodes for v in self.locations[node.as_pointer()]])


def inner_name(node):
    return f"{INNER_PREFIX}{SPECS[node.bl_idname].label} {node.uid}"


def create_inner(node, store=None):
    """Give the node a new inner group with its spec's interface, built from its stored entries."""
    tree = bpy.data.node_groups.new(inner_name(node), 'GeometryNodeTree')
    node.node_tree = tree
    sync_interface(node, SPECS[node.bl_idname].iface)
    build(node, _store.load(node) if store is None else store)
    return tree


def build(node, store):
    """Rebuild the inner group's nodes from the store; the interface is kept."""
    tree = node.node_tree
    if tree is None:
        create_inner(node, store)
        return
    tree.pop(BUILT, None)  # a build that raises leaves a partial graph, which update_values() must not write
    tree.nodes.clear()
    builder = _Builder(tree)
    group_in = builder.new("NodeGroupInput", -900, 0)
    group_out = builder.new("NodeGroupOutput", 800, 0)
    SPECS[node.bl_idname].build(builder, group_in, group_out, store.entries)
    builder.finish()
    tree[BUILT] = _built(node, store.entries)


def update_values(node, store):
    """Write the changed entry-level values without rebuilding: every Index Switch that feeds an output
    holds the fallback, then each entry's `single` value for that output. Returns False, writing
    nothing, when build() laid out other entries than the store's, so build() is needed."""
    tree = node.node_tree
    entries = store.entries
    if tree is None or tree.get(BUILT) != _built(node, entries):
        return False
    feeds = [(link.to_socket.name, _items(link.from_node)) for link in tree.links
             if link.to_node.bl_idname == "NodeGroupOutput" and link.from_node.bl_idname == "GeometryNodeIndexSwitch"]
    if any(len(slots) != len(entries) + 1 for _name, slots in feeds):
        return False
    for name, slots in feeds:
        for slot, entry in zip(slots[1:], entries):
            value = entry.single[name]
            if not _same(slot.default_value, value):  # every write costs a tree update
                slot.default_value = value
    return True


def _built(node, entries):
    """The node type and every entry's words, in order: what the graph build() lays out depends on."""
    text = json.dumps([node.bl_idname, [entry.gn for entry in entries]], ensure_ascii=False)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _same(current, value):
    if isinstance(current, float):  # a float socket holds the value rounded to 32 bits
        return current == struct.unpack("f", struct.pack("f", value))[0]
    return current == value


def sync_interface(node, wanted):
    """Make the inner group's interface `wanted`, in order, without clearing it. Sockets are matched by
    (in_out, name, socket_type), so their identifiers and the links to them survive; the others are
    removed. A socket declared hidden is hidden on the node when it is created, and never again, so
    one the artist revealed with Ctrl H stays visible."""
    interface = node.node_tree.interface
    defs = {(d.in_out, d.name, d.socket_type): d for d in wanted}
    items = {}
    for item in [i for i in interface.items_tree if i.item_type == 'SOCKET']:
        key = (item.in_out, item.name, item.socket_type)
        if key in defs and key not in items:
            items[key] = item
        else:
            interface.remove(item)
    created = []
    for key, d in defs.items():
        item = items.get(key)
        if item is None:
            item = items[key] = interface.new_socket(d.name, in_out=d.in_out, socket_type=d.socket_type)
            created.append((d, item.identifier))
        _set(item, "description", d.description)
        if d.in_out == 'INPUT':
            _set(item, "structure_type", compat.structure_type(d.socket_type, d.kind))
            _set(item, "hide_value", d.hide_value)
            if hasattr(item, "default_input"):  # not on every socket type
                _set(item, "default_input", d.default_input)
    for position, key in enumerate(sorted(defs, key=lambda k: k[0] != 'OUTPUT')):  # outputs first, then inputs
        if items[key].position != position:
            interface.move(items[key], position)  # always upwards: the items above are in place
    for d, identifier in created:
        if d.hidden:
            sock(node.inputs if d.in_out == 'INPUT' else node.outputs, identifier).hide = True


def _set(item, name, value):
    if getattr(item, name) != value:  # every write updates the interface and the nodes using it
        setattr(item, name, value)
