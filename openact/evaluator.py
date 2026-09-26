"""Reads the words an OpenAct node is given, without running Geometry Nodes.

Geometry Nodes cannot run Python while it evaluates, so a node's inputs are computed from the node
tree itself: values typed on sockets; Group Input values from the modifier (through nested groups)
or the tree's interface defaults; the String, Integer, Boolean, Value, Object and Collection inputs;
Join Strings, Value to String, Replace String, Slice String, String Length, Special Characters;
Switch, Index Switch, Menu Switch; Import Text, Reroute, muted nodes and links; and the single-value
outputs of OpenAct nodes, through the handler the engine registers.

Anything that needs geometry or fields raises Unresolvable, with a breadcrumb from the socket that
was read to the node that failed: "Context ← Join Strings ← Named Attribute: fields can't be words".
"""

from __future__ import annotations

import bpy

MAX_CONTEXTS = 64
MAX_DEPTH = 64
MEMO = None  # (output socket pointer, Ctx.key()) -> value while engine.run_pass() runs; None otherwise


class Unresolvable(Exception):
    """An input comes from something that cannot be computed outside Geometry Nodes."""

    def __init__(self, message):
        super().__init__(message)
        self.hops = []  # socket and node names, from the socket that was read to the node that failed

    def __str__(self):
        message = super().__str__()
        if not self.hops:
            return message
        hops = self.hops if len(self.hops) <= 3 else [self.hops[0], "…", *self.hops[-2:]]
        return f"{' ← '.join(hops)}: {message}"


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


def _text(value):
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


def is_field(sock):
    """True when Blender's field inference gives the socket one value per element."""
    inferred = getattr(sock, "inferred_structure_type", None)  # 5.0+, where display_shape doesn't tell
    return inferred == 'FIELD' if inferred is not None else sock.display_shape == 'DIAMOND'


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
            return [] if multi else getattr(sock, "default_value", None)
        if multi:  # Join Strings joins the most recently connected link first
            ordered = sorted(links, key=lambda link: -link.multi_input_sort_id)
            return [eval_output(link.from_node, link.from_socket, ctx, depth + 1) for link in ordered]
        return eval_output(links[0].from_node, links[0].from_socket, ctx, depth + 1)
    except Unresolvable as error:
        if depth == 0:  # the breadcrumb starts at the socket that was asked for
            error.hops.insert(0, sock.name)
        raise


def eval_output(node, out, ctx, depth):
    key = None
    if MEMO is not None:
        key = (out.as_pointer(), ctx.key())
        if key in MEMO:
            return MEMO[key]
    try:
        value = _eval_output(node, out, ctx, depth)
    except Unresolvable as error:
        if node.bl_idname != "NodeReroute":
            error.hops.insert(0, node.name)
        raise
    if key is not None:
        MEMO[key] = value
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
            value = ctx.mod[identifier]
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
    delimiter = _text(eval_input(_in(node, "Delimiter"), ctx, depth))
    parts = eval_input(_in(node, "Strings"), ctx, depth)
    if not isinstance(parts, list):
        parts = [parts]
    return delimiter.join(_text(part) for part in parts)


def _h_value_to_string(node, out, ctx, depth):
    value = eval_input(_in(node, "Value"), ctx, depth)
    if getattr(node, "data_type", "FLOAT") == 'INT':
        return str(int(value))
    decimals = int(eval_input(_in(node, "Decimals"), ctx, depth) or 0)
    return f"{float(value):.{max(decimals, 0)}f}"


def _h_replace(node, out, ctx, depth):
    text = _text(eval_input(_in(node, "String"), ctx, depth))
    find = _text(eval_input(_in(node, "Find"), ctx, depth))
    replace = _text(eval_input(_in(node, "Replace"), ctx, depth))
    return text.replace(find, replace) if find else text


def _h_slice(node, out, ctx, depth):
    text = _text(eval_input(_in(node, "String"), ctx, depth))
    position = max(int(eval_input(_in(node, "Position"), ctx, depth) or 0), 0)
    length = max(int(eval_input(_in(node, "Length"), ctx, depth) or 0), 0)
    return text[position:position + length]


def _h_string_length(node, out, ctx, depth):
    return len(_text(eval_input(_in(node, "String"), ctx, depth)))  # code points, as Geometry Nodes counts


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
            return eval_input(sock, ctx, depth)
    raise Unresolvable(f"'{node.name}' has no item '{_text(name)}'")


def _h_import_text(node, out, ctx, depth):
    path = _text(eval_input(_in(node, "Path"), ctx, depth))
    if not path:
        return ""
    try:
        with open(bpy.path.abspath(path), "rb") as handle:
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
    "ShaderNodeValue": lambda node, out, ctx, depth: node.outputs[0].default_value,
    "FunctionNodeInputInt": lambda node, out, ctx, depth: node.integer,
    "FunctionNodeInputBool": lambda node, out, ctx, depth: node.boolean,
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