"""Reads the words an OpenAct node is given, without running Geometry Nodes.

Geometry Nodes cannot run Python while it evaluates, so a node's inputs are computed from the node
tree itself: values typed on sockets; Group Input values from the modifier (through nested groups)
or the tree's interface defaults; the String, Integer, Boolean, Value, Object and Collection inputs;
Join Strings, Value to String, Replace String, Slice String, String Length, Special Characters;
Switch, Index Switch, Menu Switch; Import Text, Reroute, muted nodes and links; and the single-value
outputs of OpenAct nodes, through the handler the engine registers.

Anything that needs geometry or fields raises Unresolvable, with a breadcrumb from the socket that
was read to the node that failed: "Context ← Join Strings ← Named Attribute: fields can't be words".
The one geometry link followed is a CSV socket's, by trace_csv_source(), and only to read the Path
and Delimiter of the Import CSV node at its end. Import CSV and Import Text Paths resolve as Geometry
Nodes resolves them (import_path()), and Import Text obeys the file trust rule (trust.py).

For Prepare for Playback (§5.5), the values animation can change (modifier inputs, socket defaults, the
Integer and Boolean nodes) are recorded in READS, and OVERRIDES replaces them, e.g. by their F-curves'
values at another frame.
"""

from __future__ import annotations

import os

import bpy

from . import trust
from .csvfmt import CsvSource

MAX_CONTEXTS = 64
MAX_DEPTH = 64
MEMO = None  # (output socket pointer, Ctx.key()) -> (value, depth read at) during engine.run_pass(); else None
READS = None  # while Prepare for Playback reads words: (ID pointer, data path) -> (struct, name, custom, value) read
OVERRIDES = None  # while Prepare for Playback reads words: (ID pointer, data path) -> the value read instead


class Unresolvable(Exception):
    """An input comes from something that cannot be computed outside Geometry Nodes."""

    def __init__(self, message):
        super().__init__(message)
        self.socket = ""  # the socket that was read
        self.hops = []  # node names, from that socket to the node that failed

    def __str__(self):
        message = super().__str__()
        names = [self.socket, *self.hops] if self.socket else self.hops
        if not names:
            return message
        if len(names) > 3:
            names = [names[0], "…", *names[-2:]]
        return f"{' ← '.join(names)}: {message}"


class FileError(Unresolvable):
    """A CSV with errors: the node keeps its last good table. problems holds every csvfmt.Problem of the file."""

    def __init__(self, message, problems=()):
        super().__init__(message)
        self.problems = tuple(problems)


class Pending(Exception):
    """An input depends on another OpenAct node that has no answer yet."""


class Hint(Exception):
    """Nothing to ask yet; tells the user what is missing."""


# --------------------------------------------------------------------------- contexts


class Ctx:
    """One way a node tree is reached: object modifier -> top tree -> group nodes -> tree."""

    __slots__ = ("obj", "mod", "path", "root")

    def __init__(self, obj, mod, path, root):
        self.obj, self.mod, self.path, self.root = obj, mod, path, root

    def push(self, group_node):
        return Ctx(self.obj, self.mod, self.path + (group_node,), self.root)

    def pop(self):
        return Ctx(self.obj, self.mod, self.path[:-1], self.root)

    def key(self):
        """Tells contexts apart in MEMO."""
        obj = self.obj.as_pointer() if self.obj is not None else 0
        mod = self.mod.name if self.mod is not None else ""
        return obj, mod, tuple(node.as_pointer() for node in self.path)

    def label(self):
        if self.obj is None:
            return "(not used by a modifier)"
        return f"{self.obj.name} > {self.mod.name}"


def find_uses():
    """Map node tree pointer -> list of Ctx for every modifier that (indirectly) uses it."""
    uses = {}

    def walk(tree, ctx, stack):
        key = tree.as_pointer()
        if len(stack) > 32 or key in stack:
            return
        found = uses.setdefault(key, [])
        if len(found) < MAX_CONTEXTS:
            found.append(ctx)
        for node in tree.nodes:
            if node.bl_idname == "GeometryNodeGroup" and node.node_tree is not None and not node.mute:
                walk(node.node_tree, ctx.push(node), stack | {key})

    for obj in bpy.data.objects:
        for mod in obj.modifiers:
            if mod.type == 'NODES' and mod.node_group is not None:
                walk(mod.node_group, Ctx(obj, mod, (), mod.node_group), frozenset())
    return uses


# --------------------------------------------------------------------- static evaluator


def _available(sock):
    if getattr(sock, "is_unavailable", False):
        return False
    return getattr(sock, "enabled", True)


def _in(node, name):
    for sock in node.inputs:
        if sock.name == name and _available(sock):
            return sock
    raise Unresolvable(f"'{node.name}' has no input '{name}'")


def as_text(value):
    """A word's value as a string: an ID gives its name, None gives ""."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bpy.types.ID):
        return value.name
    return str(value)


def _interface_item(tree, identifier):
    for item in tree.interface.items_tree:
        if getattr(item, "item_type", "") == 'SOCKET' and item.identifier == identifier:
            return item
    return None


def _implicit(sock):
    """True for a group node input that Geometry Nodes gives an implicit field while unlinked (its interface item's
    default_input), e.g. an OpenAct node's ID, which the node passes on while muted: the element's index, never the
    socket's hidden value (§6.0). No String input has one, and words are Strings: they skip the search."""
    if sock.type == 'STRING':
        return False
    tree = getattr(sock.node, "node_tree", None)
    return tree is not None and getattr(_interface_item(tree, sock.identifier), "default_input", 'VALUE') != 'VALUE'


def is_field(sock):
    """True when Blender's field inference gives the socket one value per element."""
    inferred = getattr(sock, "inferred_structure_type", None)  # 5.0+, where display_shape doesn't tell
    return inferred == 'FIELD' if inferred is not None else sock.display_shape == 'DIAMOND'


def _animated(struct, name, custom=False):
    """The property `name` of struct (a custom property with custom), which animation may change: recorded in READS,
    and replaced by OVERRIDES, under the owning ID and the property's data path."""
    value = struct[name] if custom else getattr(struct, name)
    if READS is None and OVERRIDES is None:
        return value
    owner = struct.id_data
    path = f'{struct.path_from_id()}["{name}"]' if custom else struct.path_from_id(name)
    key = (owner.as_pointer(), path)
    if READS is not None:
        READS.setdefault(key, (struct, name, custom, value))
    return value if OVERRIDES is None else OVERRIDES.get(key, value)


def eval_input(sock, ctx, depth=0):
    """Value of an input socket in the given context."""
    if depth > MAX_DEPTH:
        raise Unresolvable("the connection chain is too deep")
    try:
        links = [link for link in sock.links if not link.is_muted]
        if any(not link.is_valid for link in links):
            raise Unresolvable("An input has an invalid (red) link")
        multi = getattr(sock, "is_multi_input", False)
        if not links:
            if multi or not hasattr(sock, "default_value"):
                return [] if multi else None
            if _implicit(sock):
                raise Unresolvable("fields can't be words")
            return _animated(sock, "default_value")
        if multi:  # Join Strings joins the most recently connected link first
            ordered = sorted(links, key=lambda link: -link.multi_input_sort_id)
            return [eval_output(link.from_node, link.from_socket, ctx, depth + 1) for link in ordered]
        return eval_output(links[0].from_node, links[0].from_socket, ctx, depth + 1)
    except Unresolvable as error:
        if depth == 0:  # the breadcrumb starts at the socket that was asked for
            error.socket = sock.name
        raise


def eval_output(node, out, ctx, depth):
    key = None
    if MEMO is not None:
        key = (out.as_pointer(), ctx.key())
        hit = MEMO.get(key)
        if hit is not None and depth <= hit[1]:  # a deeper read might pass MAX_DEPTH, so it reads again
            return hit[0]
    try:
        value = _eval_output(node, out, ctx, depth)
    except Unresolvable as error:
        if node.bl_idname != "NodeReroute":
            # An OpenAct node reads its own words from depth 0: that socket belongs to its hop.
            error.hops.insert(0, f"{node.name}.{error.socket}" if error.socket else node.name)
            error.socket = ""
        raise
    if key is not None:
        MEMO[key] = (value, depth)
    return value


def _eval_output(node, out, ctx, depth):
    if node.mute:
        for internal in node.internal_links:
            if internal.to_socket == out:
                value = eval_input(internal.from_socket, ctx, depth + 1)
                if isinstance(value, list):  # a muted Join Strings passes its first link on
                    value = value[0] if value else ""
                return value
        return _muted_value(out)
    kind = node.bl_idname
    if kind == "NodeReroute":
        return eval_input(node.inputs[0], ctx, depth + 1)
    if kind == "NodeGroupInput":
        return _group_input(out.identifier, ctx, depth)
    if kind == "GeometryNodeGroup":
        return _group_output(node, out, ctx, depth)
    handler = HANDLERS.get(kind)
    if handler is None:
        if is_field(out):
            raise Unresolvable("fields can't be words")
        raise Unresolvable(f"{node.bl_label} isn't supported as an OpenAct input")
    return handler(node, out, ctx, depth + 1)


def _muted_value(out):
    """What a muted node without a pass-through outputs: the type's zero value."""
    return {'STRING': "", 'VALUE': 0.0, 'INT': 0, 'BOOLEAN': False}.get(out.type)


def _group_input(identifier, ctx, depth):
    if ctx.path:
        parent = ctx.path[-1]
        for sock in parent.inputs:
            if sock.identifier == identifier:
                return eval_input(sock, ctx.pop(), depth + 1)
        raise Unresolvable("a group input is missing on the group node")
    item = _interface_item(ctx.root, identifier)
    if ctx.mod is not None:
        if ctx.mod.get(identifier + "_use_attribute"):
            raise Unresolvable("a modifier input is set to use an attribute")
        try:
            value = _animated(ctx.mod, identifier, custom=True)
        except KeyError:
            pass
        else:
            if getattr(item, "socket_type", "") != 'NodeSocketMenu':
                return value
            # A modifier stores a menu as the item's identifier, which stays fixed when items are
            # deleted or reordered, so it is never a position; the modifier's item list names it.
            for entry in ctx.mod.id_properties_ui(identifier).as_dict().get("items", ()):
                if entry[-1] == value:
                    return entry[1]
            # Geometry Nodes selects no item for any other value, -1 on a fresh modifier included.
            raise Unresolvable(f"choose an item for '{item.name}' in the modifier")
    return getattr(item, "default_value", None) if item is not None else None


def _group_output(node, out, ctx, depth):
    sub = node.node_tree
    if sub is None:
        return getattr(out, "default_value", None)
    output = next((n for n in sub.nodes if n.bl_idname == "NodeGroupOutput" and n.is_active_output), None)
    if output is None:
        return getattr(out, "default_value", None)
    for sock in output.inputs:
        if sock.identifier == out.identifier:
            return eval_input(sock, ctx.push(node), depth + 1)
    return getattr(out, "default_value", None)


def _h_join(node, out, ctx, depth):
    delimiter = as_text(eval_input(_in(node, "Delimiter"), ctx, depth))
    parts = eval_input(_in(node, "Strings"), ctx, depth)
    if not isinstance(parts, list):
        parts = [parts]
    return delimiter.join(as_text(part) for part in parts)


def _h_value_to_string(node, out, ctx, depth):
    value = eval_input(_in(node, "Value"), ctx, depth)
    if getattr(node, "data_type", "FLOAT") == 'INT':
        return str(int(value))
    decimals = int(eval_input(_in(node, "Decimals"), ctx, depth) or 0)
    return f"{float(value):.{max(decimals, 0)}f}"


def _h_replace(node, out, ctx, depth):
    text = as_text(eval_input(_in(node, "String"), ctx, depth))
    find = as_text(eval_input(_in(node, "Find"), ctx, depth))
    replace = as_text(eval_input(_in(node, "Replace"), ctx, depth))
    return text.replace(find, replace) if find else text


def _h_slice(node, out, ctx, depth):
    text = as_text(eval_input(_in(node, "String"), ctx, depth))
    position = max(int(eval_input(_in(node, "Position"), ctx, depth) or 0), 0)
    length = max(int(eval_input(_in(node, "Length"), ctx, depth) or 0), 0)
    return text[position:position + length]


def _h_string_length(node, out, ctx, depth):
    return len(as_text(eval_input(_in(node, "String"), ctx, depth)))  # code points, as Geometry Nodes counts


def _h_special_characters(node, out, ctx, depth):
    return {"Line Break": "\n", "Tab": "\t"}[out.identifier]


def _h_switch(node, out, ctx, depth):
    condition = bool(eval_input(_in(node, "Switch"), ctx, depth))
    return eval_input(_in(node, "True" if condition else "False"), ctx, depth)


def _h_index_switch(node, out, ctx, depth):
    index = int(eval_input(_in(node, "Index"), ctx, depth) or 0)
    items = [s for s in node.inputs if s.identifier not in ("Index", "__extend__") and _available(s)]
    if 0 <= index < len(items):
        return eval_input(items[index], ctx, depth)
    return getattr(out, "default_value", None)


def _h_menu_switch(node, out, ctx, depth):
    name = eval_input(_in(node, "Menu"), ctx, depth)  # a menu value is an item name
    items = [s for s in node.inputs if s.identifier not in ("Menu", "__extend__")]  # in enum_items order
    for item, sock in zip(node.enum_items, items):
        if item.name == name:
            if out.identifier != "Output":  # 5.0+: a Boolean output per item, with its input's identifier
                return out.identifier == sock.identifier
            return eval_input(sock, ctx, depth)
    raise Unresolvable(f"'{node.name}' has no item '{as_text(name)}'")


def import_path(path: str, node) -> str:
    """The file an Import CSV or Import Text node reads for this Path, as Geometry Nodes resolves it: '//' is relative
    to the file that holds the node's tree (the .blend, or a linked library), and any other relative path to Blender's
    working folder. Raises Hint for '//' in an unsaved .blend, where Geometry Nodes reads nothing."""
    library = node.id_data.library
    if path.startswith("//") and library is None and not bpy.data.filepath:
        raise Hint("Save the .blend: '//' paths are relative to it")
    return os.path.abspath(bpy.path.abspath(path, library=library))


def _h_import_text(node, out, ctx, depth):
    path = as_text(eval_input(_in(node, "Path"), ctx, depth))
    if not path:
        return ""
    full = import_path(path, node)
    message = trust.blocked(full)  # before open(): blocked text is never read
    if message:
        raise Unresolvable(message)
    try:
        with open(full, "rb") as handle:
            data = handle.read()
    except OSError:
        raise Unresolvable(f"Can't read the text file '{path}'") from None
    # Same as Blender's Import Text (measured in 4.5 and 5.1): CRLF becomes LF, a lone CR and a
    # BOM are kept, and invalid UTF-8 or a NUL byte gives an empty string.
    if b"\x00" in data:
        return ""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return ""
    return text.replace("\r\n", "\n")


HANDLERS = {  # node bl_idname -> handler(node, out, ctx, depth); the engine adds the OpenAct nodes
    "FunctionNodeInputString": lambda node, out, ctx, depth: node.string,
    "GeometryNodeInputCollection": lambda node, out, ctx, depth: node.collection,
    "GeometryNodeInputObject": lambda node, out, ctx, depth: node.object,
    "ShaderNodeValue": lambda node, out, ctx, depth: _animated(node.outputs[0], "default_value"),
    "FunctionNodeInputInt": lambda node, out, ctx, depth: _animated(node, "integer"),
    "FunctionNodeInputBool": lambda node, out, ctx, depth: _animated(node, "boolean"),
    "GeometryNodeStringJoin": _h_join,
    "FunctionNodeValueToString": _h_value_to_string,
    "FunctionNodeReplaceString": _h_replace,
    "FunctionNodeSliceString": _h_slice,
    "FunctionNodeStringLength": _h_string_length,
    "FunctionNodeInputSpecialCharacters": _h_special_characters,
    "GeometryNodeSwitch": _h_switch,
    "GeometryNodeIndexSwitch": _h_index_switch,
    "GeometryNodeMenuSwitch": _h_menu_switch,
    "GeometryNodeImportText": _h_import_text,
}


# ------------------------------------------------------------------ Import CSV hand-off

IMPORT_CSV = "GeometryNodeImportCSV"


def _by_id(node, identifier):
    return next(s for s in node.inputs if s.identifier == identifier)


def _read(sock, ctx, node, label):
    """eval_input() for a socket the trace needs, with the CSV socket and the node in the breadcrumb."""
    try:
        return eval_input(sock, ctx)
    except Unresolvable as error:
        error.hops.insert(0, f"{node.name}.{error.socket}" if error.socket else node.name)
        error.socket = label
        raise


def trace_csv_source(node, csv_socket, ctx) -> CsvSource:
    """The Import CSV node plugged into a node's '<Input> CSV' socket, with its Path and Delimiter in this context.

    Follows geometry links upstream through Reroutes, muted nodes, nested group boundaries, and Switch, Index Switch
    and Menu Switch nodes whose selector is read statically. The Import CSV geometry itself is never used, and the
    path is the file that node reads (import_path()). Raises Unresolvable when the trace ends anywhere but at an Import
    CSV node, and Hint when that node has no Path.
    """
    label = csv_socket.name
    via = []
    sock = csv_socket
    for _ in range(MAX_DEPTH):
        links = [link for link in sock.links if not link.is_muted]
        if not links:
            raise Unresolvable(f"{label}: connect an Import CSV node")
        if not links[0].is_valid:
            raise Unresolvable(f"{label}: the link is invalid (red)")
        up, out = links[0].from_node, links[0].from_socket
        kind = up.bl_idname
        if up.mute:  # passes its first geometry input on, or nothing
            sock = next((i.from_socket for i in up.internal_links if i.to_socket == out), None)
            if sock is None:
                raise Unresolvable(f"{label}: connect an Import CSV node ('{up.name}' is muted)")
            continue
        if kind == IMPORT_CSV:
            path = as_text(_read(_by_id(up, "Path"), ctx, up, label)).strip()
            if not path:
                raise Hint(f"{label}: type a Path on '{up.name}'")
            delimiter = as_text(_read(_by_id(up, "Delimiter"), ctx, up, label))
            return CsvSource(import_path(path, up), delimiter, tuple(via + [up.name]), written=path)
        if kind == "NodeReroute":
            sock = up.inputs[0]
        elif kind == "NodeGroupInput":
            if not ctx.path:
                raise Unresolvable(f"{label}: connect an Import CSV node (found the modifier's '{out.name}' input)")
            parent = ctx.path[-1]
            sock = next((s for s in parent.inputs if s.identifier == out.identifier), None)
            if sock is None:
                raise Unresolvable(f"{label}: a group input is missing on '{parent.name}'")
            ctx = ctx.pop()
            via.append(parent.name)
        elif kind == "GeometryNodeGroup":
            sub = up.node_tree
            output = sub and next((n for n in sub.nodes if n.bl_idname == "NodeGroupOutput" and n.is_active_output),
                                  None)
            sock = output and next((s for s in output.inputs if s.identifier == out.identifier), None)
            if sock is None:
                raise Unresolvable(f"{label}: connect an Import CSV node ('{up.name}' has no such output)")
            ctx = ctx.push(up)
            via.append(up.name)
        elif kind == "GeometryNodeSwitch":
            condition = bool(_read(_in(up, "Switch"), ctx, up, label))
            sock = _in(up, "True" if condition else "False")
            via.append(up.name)
        elif kind == "GeometryNodeIndexSwitch":
            index = int(_read(_in(up, "Index"), ctx, up, label) or 0)
            items = [s for s in up.inputs if s.identifier not in ("Index", "__extend__") and _available(s)]
            if not 0 <= index < len(items):
                raise Unresolvable(f"{label}: connect an Import CSV node ('{up.name}' selects nothing)")
            sock = items[index]
            via.append(up.name)
        elif kind == "GeometryNodeMenuSwitch":
            name = _read(_in(up, "Menu"), ctx, up, label)
            items = [s for s in up.inputs if s.identifier not in ("Menu", "__extend__")]
            sock = next((s for item, s in zip(up.enum_items, items) if item.name == name), None)
            if sock is None:
                raise Unresolvable(f"{label}: '{up.name}' has no item '{as_text(name)}'")
            via.append(up.name)
        elif any(s.type == 'GEOMETRY' for s in up.inputs):  # it changes the points it passes on
            raise Unresolvable(f"{label}: connect the Import CSV node directly ('{up.name}' is in between)")
        else:
            raise Unresolvable(f"{label}: connect an Import CSV node (found '{up.name}')")
    raise Unresolvable(f"{label}: the connection chain is too deep")
