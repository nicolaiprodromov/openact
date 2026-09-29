"""The hidden node group inside every OpenAct node, `.OpenAct <Node> <uid>`, and the rows of its table object.

It is derived data: build() regenerates it, and the node's table object (tables.py), from the node's
stored entries (store.py) at any time. The table object holds every stored entry (§8.1): a meta row,
then one block per entry, its header row and its element rows (§8.5). Geometry Nodes finds the entry
of the node's live words by their hash, Hash Value of the key text, among the header rows (Attribute
Statistic, §8.2), reads its single values at that row (§8.6) and its per-element rows after it
(§8.3), so one tree answers every context that uses it, with no Python at evaluation. A new entry
is new rows, never a graph edit: the graph depends only on the layout (_built(), §8.8). Every group
passes its Geometry input to its Geometry output (§6.0), Rules' stamping each element's cells on it
(§5.8), and reports its uid, Trigger and live words through one Warning node (§5.7). The interface is
never cleared: sync_interface() diffs it, so the links from the outer tree survive every rebuild.
"""

from __future__ import annotations

import array
import hashlib
import json
from dataclasses import dataclass

import bpy

from . import compat, tables, words
from . import store as _store  # build() and update_values() take `store`, the name §13.3 fixes
from .specs import SPECS

INNER_PREFIX = ".OpenAct "
BUILT = "openact_built"  # ID property of the group: the layout build() laid out (_built()), for update_values()
LAYOUT = 5  # bump when build() lays out another graph for the same entries: groups saved before then are rebuilt
UNLINKED = -2**31 + 1  # an unlinked ID's hidden value, read as the index (_Builder.ident()); no float converts to it
GEOMETRY = "Geometry"  # the input and output every node passes its geometry through (§6.0)
TRIGGER = "Trigger"  # the input whose state the Warning reports (§5.7)
UID = "OpenAct uid"  # the String node holding the uid the Warning reports: update_values() rewrites it for a copy
CELL = ".openact_cell_"  # + the uid: the hidden point attribute Rules stamps each element's cells in (§5.8)
GATES = {'VALUE': 'FLOAT', 'INT': 'INT', 'BOOLEAN': 'BOOLEAN', 'STRING': 'STRING', 'GEOMETRY': 'GEOMETRY',
         'MATERIAL': 'MATERIAL'}  # output socket type -> the input_type of the Switch it is read through (gate())
_PACK = {'FLOAT': 'd', 'INT': 'q', 'BOOLEAN': 'B'}  # array codes the rows are digested with (_digest())
_CAST = {'FLOAT': float, 'INT': int, 'BOOLEAN': bool}


@dataclass(frozen=True)
class Column:
    """A column of the node's table object (§8.5). A spec lists its per-element columns, read at row + 1 + ID (§8.3),
    in the optional `per_element` attribute, and its single values, read at the entry's header row (§8.6), in the
    optional `single_values` attribute: each a tuple, or a callable taking the node; the default is none."""

    output: str  # the group output it feeds, e.g. "Probability"; "" writes the table only (the spec reads it)
    name: str  # stored as the attribute "oa_<name>"; per element, the entries' rows.v column of that name
    data_type: str  # 'FLOAT', 'INT' or 'BOOLEAN'
    fallback: object  # §6.0: 0.0, -1 or False, for invalid IDs, unanswered contexts and the rows that don't use it
    linked_only: bool = False  # per element, built only while the output is linked (Pick's Probability)
    payload: bool = False  # read from rows.payload[output], a baked CSV column (§14.2), instead of rows.v[name]


RESERVED = (Column("", tables.DEFAULT, 'INT', -1), Column("", tables.HEAD, 'BOOLEAN', False),
            Column("", tables.KEY, 'INT', 0), Column("", tables.COUNT, 'INT', 0))  # every table's own columns (§8.5)
LEAD = 1  # the row after the meta row (point 0), where a spec's lead rows start: Rules' category maps (§9.2)


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

    def __init__(self, tree):
        self.tree = tree
        self.locations = {}  # node pointer -> (x, y)
        self.key = None  # the key text of the live words (key_text()): the Warning reports it, find() hashes it
        self.row = None  # the header row of the entry the live words find, -1 for none (find())
        self.found = None  # row >= 0: the single values read the entry, else their fallbacks
        self.after_elements = []  # a spec's callables (valid, {output: value}), run once per_element() built them
        self.node = None  # the OpenAct node (build() sets it), for a spec that reads its own sockets or outputs
        self.table = None  # the node's table object (build() sets it)
        self.index = None  # what per_element() reads at: None for the ID input; Rules sets its cell index
        self.through = None  # (Group Input Geometry, Group Output Geometry) once pass_through() linked them
        self.uid = None  # the String node UID's output, which warning() lays out
        self._info = None  # the group's one Object Info, table_geometry()
        self._ident = None  # the ID as the lookups read it, ident()
        self._first = None  # row + 1, the entry's first element row, first()
        self._count = None  # its element rows, count()

    def new(self, idname, x, y):
        node = self.tree.nodes.new(idname)
        self.locations[node.as_pointer()] = (x, y)
        return node

    def link(self, a, b):
        return self.tree.links.new(a, b)

    def pass_through(self, gi, go):
        """Every node's Geometry input to its Geometry output (§6.0), kept as self.through for stamp(). Found by name
        and type, as a CSV companion is Geometry too; an interface without them (saved before them, until the pass
        adds them) passes nothing."""
        found = [next((s for s in sockets if s.name == GEOMETRY and s.type == 'GEOMETRY'), None)
                 for sockets in (gi.outputs, go.inputs)]
        if None not in found:
            self.link(*found)
            self.through = tuple(found)

    def stamp(self, value):
        """Store `value`, an Int socket or a number, on each point of the geometry passing through, as the hidden
        attribute CELL + uid (§5.8): Rules' cells of each element, which Python reads from the evaluated objects after
        the evaluation (engine.in_use()). The name joins CELL and the String node UID, so a copy's group stamps its
        own uid once update_values() rewrites it there. Nothing is stamped without Geometry sockets."""
        if self.through is None:
            return
        source, target = self.through
        prefix = self.new("FunctionNodeInputString", 200, 1150)
        prefix.string, prefix.hide = CELL, True
        name = self.new("GeometryNodeStringJoin", 350, 1100)
        name.hide = True
        for item in (self.uid, prefix.outputs[0]):  # Join Strings puts the most recently linked item first
            self.link(item, name.inputs["Strings"])
        stored = self.new("GeometryNodeStoreNamedAttribute", 500, 1100)
        stored.data_type, stored.domain = 'INT', 'POINT'
        self.link(source, stored.inputs["Geometry"])
        self.link(name.outputs[0], stored.inputs["Name"])
        if isinstance(value, int):
            _enabled(stored.inputs, "Value").default_value = value
        else:
            self.link(value, _enabled(stored.inputs, "Value"))
        self.link(stored.outputs["Geometry"], target)  # in place of the pass-through link: an input takes one

    def words(self, gi, names):
        """The Group Input sockets of the node's word tuple, in order, None where the interface has no such
        String socket (e.g. Items under a Collection source): key_text() reads those as "" (§8.2)."""
        return [next((s for s in gi.outputs if s.name == name and s.type == 'STRING'), None) for name in names]

    def key_text(self, sockets):
        """The key text of the live words, words.key_text() in Geometry Nodes (§5.7): per word, its length in code
        points (String Length, then Value to String INT), ":" and the word, all joined in one Join Strings. A word
        the node has no socket for (None) is "", the constant "0:", as spec.gn() reads it (§8.2); a word socket a
        dropdown removes before the next rebuild leaves its length 0 and ":", so it reads "" too. Kept as
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
        (gate()). The uid is the String node UID, kept as self.uid for stamp(); "1" comes from a String Switch on
        the Trigger, so Blender's implicit conversion into its Bool socket decides what is on."""
        mark, ident = self.new("FunctionNodeInputString", -250, 1000), self.new("FunctionNodeInputString", -250, 950)
        mark.string, ident.name, ident.string = words.MARK, UID, uid
        self.uid = ident.outputs[0]
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

    def find(self):
        """The header row of the entry whose words are the live words (§8.2). The key, Hash Value (String) of the key
        text, selects the header rows that hold it (oa_head AND oa_key == key), and Attribute Statistic's Max of
        index + 1 over them is 1 + that row, 0 when there is none: then the row is the default at point 0
        (oa_default, -1 for none). Sets self.row, and self.found = row >= 0. The table object has no row for new words
        until their entry is written, and a written entry is rows only: the graph never changes for it."""
        geometry = self.table_geometry()
        hashed = self.new("FunctionNodeHashValue", -250, 300)
        hashed.data_type = 'STRING'
        self.link(self.key, _enabled(hashed.inputs, "Value"))
        same = self._compare('EQUAL', self.attribute(tables.KEY, 'INT', -250, 200), hashed.outputs["Hash"], -50, 250)
        chosen = self._and(self.attribute(tables.HEAD, 'BOOLEAN', -250, 100), same, 100, 200)
        plus = self.new("ShaderNodeMath", -50, 50)  # index + 1: exact in a float below 2**24 rows
        plus.operation = 'ADD'
        self.link(self.new("GeometryNodeInputIndex", -250, 0).outputs[0], plus.inputs[0])
        plus.inputs[1].default_value = 1.0
        statistic = self.new("GeometryNodeAttributeStatistic", 250, 200)
        statistic.data_type, statistic.domain = 'FLOAT', 'POINT'
        self.link(geometry, statistic.inputs["Geometry"])
        self.link(chosen, statistic.inputs["Selection"])
        self.link(plus.outputs[0], _enabled(statistic.inputs, "Attribute"))
        top = self.new("FunctionNodeFloatToInt", 400, 200)
        top.rounding_mode = 'ROUND'
        self.link(_enabled(statistic.outputs, "Max"), top.inputs[0])
        hit = self.new("FunctionNodeIntegerMath", 550, 250)
        hit.operation = 'SUBTRACT'
        self.link(top.outputs[0], hit.inputs[0])
        hit.inputs[1].default_value = 1
        row = self.new("GeometryNodeSwitch", 700, 200)
        row.input_type = 'INT'
        self.link(self._compare('GREATER_THAN', top.outputs[0], 0, 550, 150), sock(row.inputs, "Switch"))
        self.link(self.sample(geometry, tables.DEFAULT, 'INT', 0, 400, 50), sock(row.inputs, "False"))
        self.link(hit.outputs[0], sock(row.inputs, "True"))
        self.row = row.outputs[0]
        self.found = self._compare('GREATER_EQUAL', self.row, 0, 850, 200)

    def first(self):
        """row + 1: the entry's first element row (§8.3), laid out once."""
        if self._first is None:
            node = self.new("FunctionNodeIntegerMath", 850, 50)
            node.operation = 'ADD'
            self.link(self.row, node.inputs[0])
            node.inputs[1].default_value = 1
            self._first = node.outputs[0]
        return self._first

    def count(self):
        """The entry's element rows, oa_count at its header row (§8.3), laid out once. It reads 0 when no entry is
        found: Sample Index gives 0 outside the rows, as at row -1."""
        if self._count is None:
            self._count = self.sample(self.table_geometry(), tables.COUNT, 'INT', self.row, 700, -50)
        return self._count

    def single(self, column, x, y):
        """An entry-level value (§8.6): its column sampled at the header row, behind Switch(found, value, fallback)."""
        value = self.sample(self.table_geometry(), column.name, column.data_type, self.row, x, y)
        switch = self.new("GeometryNodeSwitch", x + 300, y)
        switch.input_type = column.data_type
        self.link(self.found, sock(switch.inputs, "Switch"))
        sock(switch.inputs, "False").default_value = column.fallback
        self.link(value, sock(switch.inputs, "True"))
        return switch.outputs[0]

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

    def per_element(self, gi, go, columns):
        """Per-element outputs (§8.3): each column sampled from the table object at flat = row + 1 + ID behind
        valid = 0 <= ID < count, else its fallback. When no entry is found, count reads 0 (count()), so a miss
        and an invalid ID both give the fallbacks. Without an ID input, the element index is used; a spec may
        set self.index to read elsewhere (Rules: its cell index, -1 when invalid). Columns without an output
        are only written to the table: the spec reads them itself."""
        columns = [column for column in columns if column.output]
        if not columns:
            return
        ident = self.index
        if ident is None:
            ident = self.ident(gi)
        if ident is None:
            ident = self.new("GeometryNodeInputIndex", -900, -700).outputs[0]
        flat = self.new("FunctionNodeIntegerMath", 300, -600)
        flat.operation = 'ADD'
        self.link(self.first(), flat.inputs[0])
        self.link(ident, flat.inputs[1])
        below = self._compare('LESS_THAN', ident, self.count(), 300, -750)
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
        """Column "oa_<name>" of the table object at `index`, a socket or a row number (§8.3): Named Attribute, Sample
        Index (POINT). attribute: a Named Attribute output to share between lookups of the same column."""
        if attribute is None:
            attribute = self.attribute(name, data_type, x, y)
        sampled = self.new("GeometryNodeSampleIndex", x + 150, y)
        sampled.data_type = data_type
        self.link(geometry, sampled.inputs["Geometry"])
        self.link(attribute, _enabled(sampled.inputs, "Value"))
        if isinstance(index, int):
            sampled.inputs["Index"].default_value = index
        else:
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
        """An Index Switch on the `index` socket, items [fallback, value 1 ...]: a constant the graph bakes (Rules'
        band widths and centres, §9.2) or one of Pick's names (§8.6). A MATERIAL value may be a material's name
        (entries store it so, §8.6)."""
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
    """Rebuild the inner group's nodes, and rewrite its table object, from the store: every entry but the refused
    ones (_written(), §8.2). The interface is kept, the geometry passed through (§6.0), the Warning channel laid out
    (§5.7) and the ID's hidden value set (_Builder.ident()).

    live: the gn tuples of the node's live contexts (§8.1), whose entries win a key they share with others; None
    means unknown. default: the words_key of the entry a miss shows (k_default_key(), §8.2)."""
    tree = node.node_tree
    if tree is None:
        create_inner(node, store)
        return
    tree.pop(BUILT, None)  # a build that raises leaves a partial graph, which update_values() must not write
    table = tables.ensure_table(node)  # before clear(), which drops the reference
    rows = _rows(node, store.entries, live, default)
    tree.nodes.clear()
    builder = _Builder(tree)
    builder.node, builder.table = node, table
    group_in = builder.new("NodeGroupInput", -900, 0)
    group_out = builder.new("NodeGroupOutput", 800, 0)
    builder.pass_through(group_in, group_out)
    spec = SPECS[node.bl_idname]
    builder.key_text(builder.words(group_in, spec.words(node)))  # before find(), which hashes it
    warning = builder.warning(group_in, node.uid)
    builder.find()
    spec.build(builder, group_in, group_out, rows.entries)
    builder.per_element(group_in, group_out, rows.columns)
    builder.gate(group_out, warning)  # last: every output, whatever built it
    builder.finish()
    tables.write_rows(node, rows.flat, rows.types, rows.digest)
    tree[BUILT] = rows.built


def update_values(node, store, live=None, default=None):
    """Values only (§8.8): rewrite the table object when its rows changed (their digest), else write the default at
    point 0 alone when it changed, and rewrite the uid the Warning reports (§5.7). So a pass that changes nothing
    writes nothing, and its table write re-evaluates no modifier. Returns False, writing nothing, when build() laid
    out another layout (_built()), so build() is needed. live and default as for build(); the pass passes the same
    ones to both."""
    tree = node.node_tree
    if tree is None or tables.find(node) is None:
        return False
    rows = _rows(node, store.entries, live, default)
    if tree.get(BUILT) != rows.built:
        return False
    ident = tree.nodes.get(UID)
    if ident is not None and ident.string != node.uid:  # a copy's group, which engine.ensure_owned() copied
        ident.string = node.uid
    if tables.digest(node) != rows.digest:
        tables.write_rows(node, rows.flat, rows.types, rows.digest)
    else:
        tables.write_value(node, tables.DEFAULT, 0, rows.default)
    return True


def append_entry(node, entry) -> bool:
    """Append one entry's block to the node's table object, values only (§8.8): its header row and element rows after
    every other row, so nothing else moves, and the graph finds it at the next evaluation. For the playback path,
    which never calls build(). Returns False, writing nothing, when that takes build() or a pass: no layout built,
    another one due (the interface or the columns changed), an entry the graph can't read (spec.fits(): Rules'
    reference factors), or a key the table already holds, the entry's own words or words it collides with (§8.2).
    A Pick entry whose item the graph doesn't list shows the fallback Item and Material until a pass rebuilds it."""
    tree = node.node_tree
    if tree is None or BUILT not in tree or tables.find(node) is None:
        return False
    try:
        layout = json.loads(tree[BUILT])[-1]
    except (ValueError, TypeError, IndexError):  # laid out by an older OpenAct (LAYOUT): the next pass rebuilds it
        return False
    columns, singles = _columns(node), _singles(node)
    fits = getattr(SPECS[node.bl_idname], "fits", None)
    if tree[BUILT] != _built(node, columns, singles, layout) or (fits is not None and not fits(node, layout, entry)):
        return False
    if words.key(entry.gn) in tables.header_rows(node):
        return False
    block = {column.name: [] for column in (*RESERVED, *singles, *columns)}
    _blocks(block, [entry], _headers(node, [entry], singles, layout), singles, columns)
    tables.append_rows(node, block)
    return True


def built_for(node, layout) -> bool:
    """True when the group was laid out for the node as it is and for the spec layout `layout` (_built(), §8.8), so
    what its evaluations compute follows them: Rules' stamp numbers the cells of that factor layout (§5.8)."""
    tree = node.node_tree
    return tree is not None and tree.get(BUILT) == _built(node, _columns(node), _singles(node), layout)


def set_default(node, words_key, store=None):
    """Write the default a miss shows (§8.2), values only: point 0's oa_default, the header row of the entry with this
    words key, -1 for None or an entry the table doesn't hold. For the playback path, which never calls build().
    store: the entries the table was written from (the pass's, with the table so far shown in a hidden entry's
    place, §5.4); by default the node's stored ones."""
    row = -1
    if words_key:
        entries = (_store.load(node) if store is None else store).entries
        entry = next((entry for entry in entries if entry.words_key == words_key), None)
        if entry is not None:
            row = tables.header_rows(node).get(words.key(entry.gn), -1)
    tables.write_value(node, tables.DEFAULT, 0, row)


def collisions(entries, live):
    """(kept, other) of each live word tuple whose entry the lookup can't find (§8.2): its key equals that of a word
    tuple before it, an entry of the table or, for words with no entry yet, a live one. The lookup finds the kept
    one's entry for both, so the pass shows each pair as an error."""
    live = list(live or ())
    wanted = set(live)
    return [(kept, other) for kept, other in words.shared_keys(_order(entries, live) + live) if other in wanted]


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


def k_default_key(last_shown, blocked=False):
    """The words_key a miss shows (§8.2): the last-shown entry when exactly one context is live and the node is in
    neither a hint nor an error state (blocked); else None, so every miss shows the neutral fallbacks. last_shown:
    LAST_SHOWN of each live context, None where it never showed one."""
    return last_shown[0] if len(last_shown) == 1 and not blocked else None


def row_count(entry):
    return len(entry.rows.name)


@dataclass
class _Rows:
    """What build() and update_values() write into the table object (§8.5), and the layout the graph reads it with."""

    flat: dict  # column -> its value on every row
    types: dict  # column -> 'FLOAT', 'INT' or 'BOOLEAN'
    default: int  # the header row a miss shows, -1 for none (§8.2), on point 0
    digest: str  # of every value but that one, which set_default() writes alone
    built: str  # the layout (_built())
    entries: list  # the entries written, in store order
    columns: list  # the per-element columns


def _rows(node, entries, live=None, default=None):
    """The table object's rows (§8.5): point 0, the meta row; then the spec's lead rows (Rules' category maps, §9.2);
    then one block per written entry, in store order (_blocks()). A column a row doesn't use holds its fallback."""
    spec = SPECS[node.bl_idname]
    written = _written(entries, live)
    columns, singles = _columns(node), _singles(node)
    layout = spec.layout(node, written) if hasattr(spec, "layout") else None
    everything = (*RESERVED, *singles, *columns)
    flat = {column.name: [column.fallback] for column in everything}  # the meta row, its default set below
    lead = spec.lead_rows(node, written) if hasattr(spec, "lead_rows") else {}
    count = len(next(iter(lead.values()), ()))
    for column in everything:
        flat[column.name].extend(lead.get(column.name) or [column.fallback] * count)
    heads = _blocks(flat, written, _headers(node, written, singles, layout), singles, columns)
    types = {column.name: column.data_type for column in everything}
    digest = _digest(flat, types)
    found = next((row for row, entry in zip(heads, written) if default and entry.words_key == default), -1)
    flat[tables.DEFAULT][0] = found  # "" names no entry
    return _Rows(flat, types, found, digest, _built(node, columns, singles, layout), written, columns)


def _blocks(flat, entries, headers, singles, columns):
    """Append each entry's block to the columns in flat (§8.5): its header row (oa_head, its key, the count of element
    rows after it, its single values), then its element rows, one per table row while the node has per-element
    columns. Returns each header row."""
    counts = [row_count(entry) if columns else 0 for entry in entries]
    heads, at = [], len(flat[tables.HEAD])
    for count in counts:
        heads.append(at)
        at += 1 + count
    per_entry = [(tables.DEFAULT, [-1] * len(entries), -1), (tables.HEAD, [True] * len(entries), False),
                 (tables.KEY, [words.key(entry.gn) for entry in entries], 0), (tables.COUNT, counts, 0)]
    for name, values, rest in per_entry + [(c.name, headers[c.name], c.fallback) for c in singles]:
        target = flat[name]
        if not columns:  # header rows only (Ask, Dials)
            target.extend(values)
            continue
        for value, count in zip(values, counts):
            target.append(value)
            target.extend([rest] * count)
    for column in columns:
        target, rest = flat[column.name], column.fallback
        for entry, count in zip(entries, counts):
            own = entry.rows.payload.get(column.output) if column.payload else entry.rows.v.get(column.name)
            target.append(rest)
            target.extend(own if own is not None else [rest] * count)
    return heads


def _headers(node, entries, singles, layout):
    """{column: its value on each entry's header row} (§8.6): the spec's own headers(node, entries, layout), by default
    each entry's single value of the column's output, else the column's fallback."""
    own = getattr(SPECS[node.bl_idname], "headers", None)
    if own is not None:
        return own(node, entries, layout)
    return {column.name: [entry.single.get(column.output, column.fallback) for entry in entries] for column in singles}


def _written(entries, live=None):
    """The entries the table holds, in store order (§8.2): all but those whose key an earlier one has
    (words.shared_keys()). Live ones come first, so a warm entry never takes a live one's place."""
    refused = {other for _kept, other in words.shared_keys(_order(entries, live))}
    return [entry for entry in entries if entry.gn not in refused]


def _order(entries, live):
    """The stored entries' word tuples in the order they claim a key: the live ones, then all in store order."""
    stored = {entry.gn for entry in entries}
    return [gn for gn in live or () if gn in stored] + [entry.gn for entry in entries]


def _digest(flat, types):
    """A digest of the rows, column by column (tables.DIGEST)."""
    digest = hashlib.sha1()
    for name in sorted(flat):
        values, kind = flat[name], types[name]
        try:
            packed = array.array(_PACK[kind], values)
        except TypeError:  # e.g. floats such as 1.0 in an INT column, as tables.write_rows() casts them
            packed = array.array(_PACK[kind], map(_CAST[kind], values))
        digest.update(f"{name} {kind} {len(values)}\n".encode())
        digest.update(packed.tobytes())
    return digest.hexdigest()


def _columns(node):
    """The spec's per-element columns whose output the node has; a linked_only one only while it is linked. A name
    listed twice is one column."""
    wanted = getattr(SPECS[node.bl_idname], "per_element", ())
    if callable(wanted):
        wanted = wanted(node)
    outputs = {socket.name: socket for socket in node.outputs}
    chosen = [c for c in wanted if not c.output  # table only: the spec reads it itself
              or (c.output in outputs and (outputs[c.output].is_linked or not c.linked_only))]
    return [c for i, c in enumerate(chosen) if all(c.name != other.name for other in chosen[:i])]


def _singles(node):
    """The spec's single-value columns (§8.6), read at the entry's header row."""
    wanted = getattr(SPECS[node.bl_idname], "single_values", ())
    return list(wanted(node) if callable(wanted) else wanted)


def _built(node, columns, singles, layout):
    """The layout the graph reads the table object with (§8.8), as JSON text: LAYOUT, the node type, the interface
    (an absent word is read as ""), the per-element and single-value columns, and what the spec bakes besides, its
    optional layout(node, entries) (Rules: the reference factors; Dials: the outputs; Pick: the names its Item and
    Material choose from). Entries change it only through the last; append_entry() reads it back."""
    face = sorted((i.in_out, i.name, i.socket_type) for i in node.node_tree.interface.items_tree
                  if i.item_type == 'SOCKET')
    return json.dumps([LAYOUT, node.bl_idname, face, [(c.output, c.name, c.data_type) for c in columns],
                       [(c.output, c.name, c.data_type) for c in singles], layout], ensure_ascii=False, sort_keys=True)


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
