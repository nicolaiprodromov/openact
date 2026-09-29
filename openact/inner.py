"""The hidden node group inside every OpenAct node, `.OpenAct <Node> <uid>`.

It is derived data: build() regenerates it, and the node's table object (tables.py), from the node's
stored entries (store.py) at any time. Only the materialised entries (§8.1) are laid out. Geometry
Nodes selects the entry whose words equal the node's live inputs (the joined-key match chain, §8.2),
reads that entry's single values from Index Switches, and its per-element rows from the table object
at base + ID (§8.3), so one tree answers every context that uses it, with no Python at evaluation.
Every group passes its Geometry input to its Geometry output unchanged (§6.0), and reports its uid,
Trigger and live words through one Warning node (§5.7). The interface is never cleared:
sync_interface() diffs it, so the links from the outer tree survive every rebuild.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import struct
from dataclasses import dataclass

import bpy

from . import compat, tables, words
from . import store as _store  # build() and update_values() take `store`, the name §13.3 fixes
from .specs import SPECS

INNER_PREFIX = ".OpenAct "
BUILT = "openact_built"  # ID property of the group: what build() laid out, for update_values()
LAYOUT = 3  # bump when build() lays out another graph for the same entries: groups saved before then are rebuilt
SEP = "\x1f"  # joins a word tuple into one key (§8.2)
UNLINKED = -2**31 + 1  # an unlinked ID's hidden value, read as the index (_Builder.ident()); no float converts to it
MATERIALISED_MAX = 16  # entries in the graph and table object, pinned included (§8.1)
WARM_TOTAL = 4  # warm entries are materialised only while the total is below this
ROW_BUDGET = 16 * 4096  # flat rows warm entries may bring the table object to
K_DEFAULT = "OpenAct k_default"  # the chain's innermost Switch: its False value is what a no-match shows
BASE, COUNT = "OpenAct base", "OpenAct count"  # Index Switches: each entry's first flat row, its row count
GEOMETRY = "Geometry"  # the input and output every node passes its geometry through (§6.0)
TRIGGER = "Trigger"  # the input whose state the Warning reports (§5.7)
UID = "OpenAct uid"  # the String node holding the uid the Warning reports: update_values() rewrites it for a copy
GATES = {'VALUE': 'FLOAT', 'INT': 'INT', 'BOOLEAN': 'BOOLEAN', 'STRING': 'STRING', 'GEOMETRY': 'GEOMETRY',
         'MATERIAL': 'MATERIAL'}  # output socket type -> the input_type of the Switch it is read through (gate())


@dataclass(frozen=True)
class Column:
    """A per-element output, read from the node's table object at base + ID (§8.3). A spec lists its columns
    in the optional `per_element` attribute (a tuple, or a callable taking the node); the default is none."""

    output: str  # the group output it feeds, e.g. "Probability"; "" writes the table only (Rules reads it)
    name: str  # the entries' rows.v column, stored as the attribute "oa_<name>"
    data_type: str  # 'FLOAT', 'INT' or 'BOOLEAN'
    fallback: object  # §6.0: 0.0, -1 or False, for invalid IDs and unanswered contexts
    linked_only: bool = False  # built only while the output is linked, else the node holds no table (Pick)
    payload: bool = False  # read from rows.payload[output], a baked CSV column (§14.2), instead of rows.v[name]


@dataclass(frozen=True)
class SocketDef:
    """A socket of a node's interface. sync_interface() matches sockets by (in_out, name, socket_type)."""

    in_out: str  # 'INPUT' or 'OUTPUT'
    name: str
    socket_type: str  # e.g. 'NodeSocketString'
    description: str = ""  # the tooltip
    kind: str = ""  # inputs: "word" (read by the evaluator), "field" (per element), "single" (one value), "" (any)
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


def _enabled(sockets, name):
    return next(s for s in sockets if s.name == name and s.enabled)


class _Builder:
    """Adds nodes and links to an inner group. finish() writes every location in one batched call,
    because each single write costs up to a few milliseconds once the tree is large."""

    def __init__(self, tree, default=0, per_word=False):
        self.tree = tree
        self.locations = {}  # node pointer -> (x, y)
        self.default = default  # k_default: what the match chain yields when no entry matches (§8.2)
        self.per_word = per_word  # a word contains SEP: compare word by word
        self.key = None  # the key text of the live words (key_text()), which the Warning reports
        self.k = None  # the match chain's output, which the per-element lookup reads
        self.after_elements = []  # a spec's callables (valid, {output: value}), run once per_element() built them
        self.node = None  # the OpenAct node, for a spec that samples its own table object (entry_table, below)
        self.table = None  # the node's table object, when it has one (build() sets it)
        self.index = None  # what per_element() reads at: None for the ID input; Rules sets its cell index
        self._info = None  # the group's one Object Info, table_geometry()
        self._ident = None  # the ID as the lookups read it, ident()

    def new(self, idname, x, y):
        node = self.tree.nodes.new(idname)
        self.locations[node.as_pointer()] = (x, y)
        return node

    def link(self, a, b):
        return self.tree.links.new(a, b)

    def pass_through(self, gi, go):
        """Every node's Geometry input to its Geometry output, unchanged (§6.0). Found by name and type, as a CSV
        companion is Geometry too; an interface without them (saved before them, until the pass adds them) passes
        nothing."""
        found = [next((s for s in sockets if s.name == GEOMETRY and s.type == 'GEOMETRY'), None)
                 for sockets in (gi.outputs, go.inputs)]
        if None not in found:
            self.link(*found)

    def words(self, gi, names):
        """The Group Input sockets of the node's word tuple, in order, None where the interface has no such
        String socket (e.g. Items under a Collection source): match_chain() joins those as "" (§8.2)."""
        return [next((s for s in gi.outputs if s.name == name and s.type == 'STRING'), None) for name in names]

    def key_text(self, sockets):
        """The key text of the live words, words.key_text() in Geometry Nodes (§5.7): per word, its length in code
        points (String Length, then Value to String INT), ":" and the word, all joined in one Join Strings. A word
        the node has no socket for (None) is "", the constant "0:", as the match chain treats it (§8.2); a word
        socket a dropdown removes before the next rebuild leaves its length 0 and ":", so it reads "" too. Kept as
        self.key."""
        key = self.new("GeometryNodeStringJoin", -250, 700)
        key.hide = True
        parts = []
        for j, socket in enumerate(sockets):
            y = 700 + 60 * len(sockets) - 60 * j
            if socket is None:
                empty = self.new("FunctionNodeInputString", -400, y)
                empty.string, empty.hide = words.key_text([""]), True
                parts.append(empty.outputs[0])
                continue
            length = self.new("FunctionNodeStringLength", -700, y)
            number = self.new("FunctionNodeValueToString", -550, y)
            colon = self.new("FunctionNodeInputString", -400, y)
            number.data_type, colon.string = 'INT', ":"
            length.hide = number.hide = colon.hide = True
            self.link(socket, length.inputs["String"])
            self.link(length.outputs["Length"], _enabled(number.inputs, "Value"))
            parts += [number.outputs["String"], colon.outputs[0], socket]
        for part in reversed(parts):  # Join Strings puts the most recently linked item first
            self.link(part, key.inputs["Strings"])
        self.key = key.outputs[0]
        return self.key

    def warning(self, gi, uid):
        """The Warning channel (§5.7): one INFO Warning whose message is words.message(uid, trigger, words),
        "OpenAct <uid> <0|1> <key text>", which the modifier lists after each evaluation that evaluates the node
        (gate()). The uid is the String node UID; "1" comes from a String Switch on the Trigger, so Blender's
        implicit conversion into its Bool socket decides what is on."""
        mark, ident = self.new("FunctionNodeInputString", -250, 1000), self.new("FunctionNodeInputString", -250, 950)
        mark.string, ident.name, ident.string = words.MARK, UID, uid
        switch = self.new("GeometryNodeSwitch", -250, 900)
        switch.input_type = 'STRING'
        sock(switch.inputs, "False").default_value, sock(switch.inputs, "True").default_value = "0", "1"
        trigger = next((s for s in gi.outputs if s.name == TRIGGER and s.type == 'BOOLEAN'), None)
        if trigger is not None:  # else it reads "0", as an unlinked Trigger does
            self.link(trigger, sock(switch.inputs, "Switch"))
        message = self.new("GeometryNodeStringJoin", 0, 900)
        message.inputs["Delimiter"].default_value = " "
        for item in reversed((mark.outputs[0], ident.outputs[0], switch.outputs[0], self.key)):
            self.link(item, message.inputs["Strings"])
        warning = self.new("GeometryNodeWarning", 200, 900)
        warning.warning_type = 'INFO'
        warning.inputs["Show"].default_value = True
        self.link(message.outputs[0], warning.inputs["Message"])
        return warning

    def gate(self, go, warning):
        """Every output read through a Switch on the Warning's Show output, which is always on (§5.7). Geometry Nodes
        evaluates a Warning inside a group only with values the group computes for a requested output: a word or
        Trigger linked from another node outside would never reach an unread Warning. So any output an evaluation
        asks for evaluates the Warning first, with everything its message needs; an unused node reports nothing."""
        show = warning.outputs["Show"]
        for j, socket in enumerate(s for s in go.inputs if s.type in GATES):
            switch = self.new("GeometryNodeSwitch", 650, 400 - 40 * j)
            switch.input_type, switch.hide = GATES[socket.type], True
            self.link(show, sock(switch.inputs, "Switch"))
            if socket.is_linked:  # an unlinked output gives the Switch's default, the type's zero, as before
                self.link(socket.links[0].from_socket, sock(switch.inputs, "True"))
            self.link(switch.outputs[0], socket)

    def match_chain(self, sockets, keys):
        """Integer socket k: 1 + index of the entry whose words equal the live words, else k_default; None
        when no entry can match. `sockets` has one item per word position, None for an absent socket (§8.2).
        A key of another length (saved before the fixed word tuples) never matches: its words sit at other
        positions."""
        keys = [tuple(key) for key in keys]
        self.k = None
        if not any(len(key) == len(sockets) for key in keys):
            return None
        sockets = [self._empty(j) if s is None else s for j, s in enumerate(sockets)]
        if self.per_word:
            joined = None
        else:
            join = self.new("GeometryNodeStringJoin", -760, 400)
            join.hide = True
            join.inputs["Delimiter"].default_value = SEP
            for socket in reversed(sockets):  # Join Strings puts the most recently linked item first
                self.link(socket, join.inputs["Strings"])
            joined = join.outputs[0]
        for i, key in enumerate(keys, start=1):
            if len(key) != len(sockets):
                continue
            y = 400 - i * 110
            pairs = list(zip(sockets, key)) if joined is None else [(joined, SEP.join(key))]
            conditions = []
            for j, (socket, value) in enumerate(pairs):
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
            if self.k is None:
                switch.name = K_DEFAULT  # update_values() rewrites its False value
                sock(switch.inputs, "False").default_value = self.default
            else:
                self.link(self.k, sock(switch.inputs, "False"))
            sock(switch.inputs, "True").default_value = i
            self.k = switch.outputs[0]
        return self.k

    def _empty(self, j):
        node = self.new("FunctionNodeInputString", -960, 400 - 40 * j)  # holds ""
        node.hide = True
        return node.outputs[0]

    def _compare(self, operation, a, b, x, y):
        """An INT Compare of a socket with a socket or a number."""
        compare = self.new("FunctionNodeCompare", x, y)
        compare.data_type = 'INT'
        compare.operation = operation
        for identifier, value in (("A_INT", a), ("B_INT", b)):
            if isinstance(value, int):
                sock(compare.inputs, identifier).default_value = value
            elif value is not None:  # None: linked by the caller
                self.link(value, sock(compare.inputs, identifier))
        return compare.outputs[0]

    def _and(self, a, b, x, y):
        both = self.new("FunctionNodeBooleanMath", x, y)
        both.operation = 'AND'
        self.link(a, both.inputs[0])
        self.link(b, both.inputs[1])
        return both.outputs[0]

    def ident(self, gi):
        """The ID as the lookups read it (§8.3), laid out once; None when the node has none. The ID is the input whose
        default_input is 'INDEX', whatever its name: a Rules factor may be called ID. Unlinked, it is the element's
        index. OpenAct's default_input gives it, but a file opened without OpenAct evaluates the undefined node, whose
        unlinked input passes its socket's value instead: that value is UNLINKED, which reads as the index here (G6).
        -2**31 would not do: NaN and every float beyond the Int range convert to it."""
        if self._ident is None:
            item = next((i for i in self.tree.interface.items_tree if i.item_type == 'SOCKET' and i.in_out == 'INPUT'
                         and getattr(i, "default_input", 'VALUE') == 'INDEX'), None)
            if item is None:
                return None
            socket = sock(gi.outputs, item.identifier)
            _set(sock(self.node.inputs, item.identifier), "default_value", UNLINKED)
            switch = self.new("GeometryNodeSwitch", -600, -800)
            switch.input_type = 'INT'
            self.link(self._compare('EQUAL', socket, UNLINKED, -750, -800), sock(switch.inputs, "Switch"))
            self.link(socket, sock(switch.inputs, "False"))
            self.link(self.new("GeometryNodeInputIndex", -750, -950).outputs[0], sock(switch.inputs, "True"))
            self._ident = switch.outputs[0]
        return self._ident

    def per_element(self, gi, go, entries, columns):
        """Per-element outputs (§8.3): each column sampled from the table object at flat = base + ID behind
        valid = 0 <= ID < count, else its fallback. base and count are Index Switches on k, so a no-match
        (k = 0) and an invalid ID both give the fallbacks. Without an ID input, the element index is used;
        a spec may set self.index to read elsewhere (Rules: its cell index, -1 when invalid). Columns
        without an output are only written to the table: the spec reads them itself."""
        columns = [column for column in columns if column.output]
        if not columns:
            return
        ident = self.index
        if ident is None:
            ident = self.ident(gi)
        if ident is None:
            ident = self.new("GeometryNodeInputIndex", -900, -700).outputs[0]
        counts = [row_count(entry) for entry in entries]
        bases = list(itertools.accumulate([0] + counts))[:-1]
        flat = self.new("FunctionNodeIntegerMath", 300, -600)
        flat.operation = 'ADD'
        self.lookup(self.k, 'INT', 0, bases, flat.inputs[0], -600).node.name = BASE
        self.link(ident, flat.inputs[1])
        below = self._compare('LESS_THAN', ident, None, 300, -750)
        self.lookup(self.k, 'INT', 0, counts, sock(below.node.inputs, "B_INT"), -750).node.name = COUNT
        valid = self._and(self._compare('GREATER_EQUAL', ident, 0, 300, -900), below, 450, -800)
        geometry = self.table_geometry()
        values = {}
        for j, column in enumerate(columns):
            y = -1050 - 150 * j
            value = self.sample(geometry, column.name, column.data_type, flat.outputs[0], 450, y)
            switch = self.new("GeometryNodeSwitch", 750, y)
            switch.input_type = column.data_type
            self.link(valid, sock(switch.inputs, "Switch"))
            sock(switch.inputs, "False").default_value = column.fallback
            self.link(value, sock(switch.inputs, "True"))
            self.link(switch.outputs[0], go.inputs[column.output])
            values[column.output] = switch.outputs[0]
        for then in self.after_elements:  # outputs derived from the columns, e.g. Match's Selection (§6.3)
            then(valid, values)

    # -- table primitives, shared by per_element() and the specs that sample the table themselves (Rules)

    def table_geometry(self):
        """The table object's geometry, from the group's one Object Info, named tables.NODE."""
        if self._info is None:
            self._info = self.new("GeometryNodeObjectInfo", 300, -1050)
            self._info.name = tables.NODE
            self._info.inputs["Object"].default_value = self.table
        return self._info.outputs["Geometry"]

    def sample(self, geometry, name, data_type, index, x, y, attribute=None):
        """Column "oa_<name>" of the table object at the `index` socket (§8.3): Named Attribute, Sample Index
        (POINT). attribute: a Named Attribute output to share between lookups of the same column."""
        if attribute is None:
            attribute = self.attribute(name, data_type, x, y)
        sampled = self.new("GeometryNodeSampleIndex", x + 150, y)
        sampled.data_type = data_type
        self.link(geometry, sampled.inputs["Geometry"])
        self.link(attribute, _enabled(sampled.inputs, "Value"))
        self.link(index, sampled.inputs["Index"])
        return _enabled(sampled.outputs, "Value")

    def attribute(self, name, data_type, x, y):
        node = self.new("GeometryNodeInputNamedAttribute", x, y)
        node.data_type = data_type
        node.inputs["Name"].default_value = tables.ATTR + name
        return _enabled(node.outputs, "Attribute")

    def selection(self, ident, picked, target):
        """Pick's Selection = (ID == Index) AND (Index >= 0) (§8.3): `picked` is the entry-level Index."""
        answered = self._compare('GREATER_EQUAL', picked, 0, 300, -1500)
        both = self._and(self._compare('EQUAL', ident, picked, 300, -1400), answered, 450, -1450)
        self.link(both, target)
        return both

    def lookup(self, index, data_type, fallback, values, target, y):
        """An entry-level value: an Index Switch on the match chain, items [fallback, value 1 ...]. A MATERIAL
        value may be a material's name (entries store it so, §8.6)."""
        node = self.new("GeometryNodeIndexSwitch", 150, y)
        node.data_type = data_type
        items = node.index_switch_items
        items.clear()
        for _ in range(len(values) + 1):
            items.new()
        slots = _items(node)
        slots[0].default_value = fallback
        for slot, value in zip(slots[1:], _values(data_type, values)):
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
    sync_interface(node, SPECS[node.bl_idname].interface(node))
    build(node, _store.load(node) if store is None else store)
    return tree


def build(node, store, live=None, default=None):
    """Rebuild the inner group's nodes, and rewrite its table object, from the store; the interface is kept, the
    geometry passed through (§6.0), the Warning channel laid out (§5.7) and the ID's hidden value set
    (_Builder.ident()).

    live: the gn tuples of the node's live contexts, in context order (§8.1). None means unknown: every
    stored entry is laid out, in store order; only the pass knows the live set. default: the words_key of
    the entry a no-match shows (k_default_key(), §8.2)."""
    tree = node.node_tree
    if tree is None:
        create_inner(node, store)
        return
    tree.pop(BUILT, None)  # a build that raises leaves a partial graph, which update_values() must not write
    entries = materialised(store.entries, live, default)
    columns = _columns(node)
    own = _entry_table(node)
    table = tables.ensure_table(node) if columns or own else None  # before clear(), which drops the reference
    tree.nodes.clear()
    split = per_word(itertools.chain((entry.gn for entry in entries), live or ()))
    builder = _Builder(tree, _k(entries, default), split)
    builder.node, builder.table = node, table
    group_in = builder.new("NodeGroupInput", -900, 0)
    group_out = builder.new("NodeGroupOutput", 800, 0)
    builder.pass_through(group_in, group_out)
    spec = SPECS[node.bl_idname]
    builder.key_text(builder.words(group_in, spec.words(node)))  # before the spec's build(), which may read it
    warning = builder.warning(group_in, node.uid)
    if table is not None:  # the table's only user, and how find() sees it, even while nothing samples it yet
        builder.table_geometry()
    spec.build(builder, group_in, group_out, entries)
    if columns:
        builder.per_element(group_in, group_out, entries, columns)
    builder.gate(group_out, warning)  # last: every output, whatever built it
    builder.finish()
    if columns:
        tables.write_rows(node, _table_rows(node, entries, columns), {c.name: c.data_type for c in columns})
    elif own:
        tables.write_rows(node, *own(node, entries))
    tree[BUILT] = _built(node, entries, columns, split)


def update_values(node, store, live=None, default=None, rows=True):
    """Values only (§8.8): write the changed entry-level Index Switch values (each output's single values,
    each entry's base and row count), k_default and the uid the Warning reports (§5.7), and with `rows` the
    table object's per-element rows. Returns False, writing nothing, when build() laid out another
    materialised set or shape, so build() is needed. live and default as for build(); the pass passes the
    same ones to both."""
    tree = node.node_tree
    if tree is None:
        return False
    entries = materialised(store.entries, live, default)
    columns = _columns(node)
    own = _entry_table(node)
    split = per_word(itertools.chain((entry.gn for entry in entries), live or ()))
    if tree.get(BUILT) != _built(node, entries, columns, split) or ((columns or own) and tables.find(node) is None):
        return False
    own_rows = own(node, entries) if own else None
    if own and own_rows is None:  # the spec wants other outputs
        return False
    feeds = [(name, source.node.data_type, _items(source.node)) for name, source in _feeds(tree)
             if source.node.bl_idname == "GeometryNodeIndexSwitch"]
    if any(len(slots) != len(entries) + 1 for _name, _type, slots in feeds):
        return False
    ident = tree.nodes.get(UID)
    if ident is not None and ident.string != node.uid:  # a copy's group, which engine.ensure_owned() copied
        ident.string = node.uid
    for name, data_type, slots in feeds:  # an entry without the value (e.g. no Material) holds the fallback
        _write(slots[1:], _values(data_type, [entry.single.get(name, slots[0].default_value) for entry in entries]))
    counts = [row_count(entry) for entry in entries]
    for name, values in ((BASE, list(itertools.accumulate([0] + counts))[:-1]), (COUNT, counts)):
        switch = tree.nodes.get(name)
        if switch is not None:
            _write(_items(switch)[1:], values)
    end = tree.nodes.get(K_DEFAULT)
    if end is not None:
        _write([sock(end.inputs, "False")], [_k(entries, default)])
    if rows and columns:
        tables.write_rows(node, _table_rows(node, entries, columns), {c.name: c.data_type for c in columns})
    elif rows and own_rows:
        tables.write_rows(node, *own_rows)
    return True


def _feeds(tree):
    """(output name, the socket that computes it) of each output the group computes, found behind its gate, the
    Switch on the Warning's Show (_Builder.gate())."""
    found = []
    for link in tree.links:
        if link.to_node.bl_idname == "NodeGroupOutput" and link.from_node.bl_idname == "GeometryNodeSwitch":
            fed = sock(link.from_node.inputs, "True").links
            if fed:
                found.append((link.to_socket.name, fed[0].from_socket))
    return found


def _values(data_type, values):
    """Stored single values as the Index Switch items take them: a material by name, None when missing (§8.6)."""
    if data_type != 'MATERIAL':
        return values
    return [bpy.data.materials.get(v) if isinstance(v, str) else v for v in values]


def _write(slots, values):
    for slot, value in zip(slots, values):
        if not _same(slot.default_value, value):  # every write costs a tree update
            slot.default_value = value


def materialised(entries, live=None, default=None):
    """The entries the graph and table object hold (§8.1), in store order: every entry whose gn is live (the
    first MATERIALISED_MAX in context order), then pinned ones in pin order, then warm ones, the default
    first and then the most recently used, while the total is below WARM_TOTAL and their rows fit ROW_BUDGET.
    live=None (unknown) gives every stored entry. Pass live as a list: a set has no context order."""
    if live is None:
        return list(entries)
    rank = {tuple(gn): i for i, gn in reversed(list(enumerate(live)))}  # a gn's first context wins
    chosen = sorted((e for e in entries if e.gn in rank), key=lambda e: rank[e.gn])[:MATERIALISED_MAX]
    pins = sorted((e for e in entries if e.pinned and e.gn not in rank), key=_pin_order)
    chosen += pins[:MATERIALISED_MAX - len(chosen)]
    total = sum(row_count(e) for e in chosen)
    for entry in sorted((e for e in entries if not e.pinned and e.gn not in rank),
                        key=lambda e: (not default or e.words_key != default, -e.t)):  # "" names no entry (_k())
        if len(chosen) >= WARM_TOTAL:
            break
        if total + row_count(entry) <= ROW_BUDGET:
            chosen.append(entry)
            total += row_count(entry)
    keep = {id(entry) for entry in chosen}
    return [entry for entry in entries if id(entry) in keep]


def _pin_order(entry):
    """By the first frame at which the entry's words occur, ties broken by the words (§5.5)."""
    return min((span[0] for span in entry.frames if span), default=math.inf), entry.gn


def k_default_key(last_shown, blocked=False):
    """The words_key a no-match shows (§8.2): the last-shown entry when exactly one context is live and the
    node is in neither a hint nor an error state (blocked); else None, so k_default = 0 and every no-match
    shows the neutral fallbacks. last_shown: LAST_SHOWN of each live context, None where it never showed one."""
    return last_shown[0] if len(last_shown) == 1 and not blocked else None


def per_word(gns):
    """True when a word contains SEP: its joined key could equal another tuple's, so the node compares word
    by word (§8.2)."""
    return any(SEP in word for gn in gns for word in gn)


def row_count(entry):
    return len(entry.rows.name)


def flat_rows(entries, columns):
    """The table object's columns (§8.1): entry e's rows at base_e + r, entries in materialised order. An
    entry without a column holds that column's fallback."""
    flat = {}
    for column in columns:
        if column.name in flat:
            continue
        values = flat[column.name] = []
        for entry in entries:
            own = entry.rows.payload.get(column.output) if column.payload else entry.rows.v.get(column.name)
            values.extend(own if own is not None else [column.fallback] * row_count(entry))
    return flat


def _columns(node):
    """The spec's per-element columns whose output the node has; a linked_only one only while it is linked."""
    wanted = getattr(SPECS[node.bl_idname], "per_element", ())
    if callable(wanted):
        wanted = wanted(node)
    outputs = {socket.name: socket for socket in node.outputs}
    return [c for c in wanted if not c.output  # table only: the spec reads it itself
            or (c.output in outputs and (outputs[c.output].is_linked or not c.linked_only))]


def _table_rows(node, entries, columns):
    """The table object's columns: flat_rows(), or the spec's own table_rows(node, entries, columns) (Rules)."""
    own = getattr(SPECS[node.bl_idname], "table_rows", None)
    return flat_rows(entries, columns) if own is None else own(node, entries, columns)


def _k(entries, default):
    """k of the entry with words_key `default`, 0 when there is none: k_default (§8.2). The words key "" names
    no entry."""
    return next((k for k, entry in enumerate(entries, start=1) if default and entry.words_key == default), 0)


def _built(node, entries, columns, split):
    """What the graph build() lays out depends on: LAYOUT, the node type, the materialised entries' words in
    order, the per-element columns, per-word compares, the interface (an absent word is joined as ""), and
    what the spec bakes from the entries besides, its optional layout(node, entries) (Rules: the factors)."""
    face = sorted((i.in_out, i.name, i.socket_type) for i in node.node_tree.interface.items_tree
                  if i.item_type == 'SOCKET')
    layout = getattr(SPECS[node.bl_idname], "layout", None)
    text = json.dumps([LAYOUT, node.bl_idname, [entry.gn for entry in entries],
                       [(c.output, c.name, c.data_type) for c in columns], split, face,
                       layout and layout(node, entries)], ensure_ascii=False, sort_keys=True)
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


def lacks(node, wanted) -> bool:
    """True when the inner group's interface lacks a socket of `wanted`, as a file saved before the node had it does:
    sync_interface() adds it, and keeps the others with their links."""
    have = {(i.in_out, i.name, i.socket_type) for i in node.node_tree.interface.items_tree if i.item_type == 'SOCKET'}
    return any((d.in_out, d.name, d.socket_type) not in have for d in wanted)


def _set(item, name, value):
    if getattr(item, name) != value:  # every write updates the interface and the nodes using it
        setattr(item, name, value)


# ------------------------------------------------------------ one-row-per-entry tables (Dials, §8.4)
# A spec with entry_table(node, entries) -> (columns, types), or None when build() must run, samples its own table
# object at k - 1 in its build(), through builder.node and builder.table. build() and update_values() write the rows.


def _entry_table(node):
    return getattr(SPECS[node.bl_idname], "entry_table", None)
