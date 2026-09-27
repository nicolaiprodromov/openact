"""Word lists: the list sources, inline answer sets and items, and where a node's CSV comes from.

resolve_items() gives the entries of Pick, Match and Classify for the node's list source (design 6.0). csv_source()
picks a CSV-capable input's route: an Import CSV node linked to the '<Input> CSV' companion wins over a typed path
(7.7). load_csv() resolves the path, applies the trust rule, parses the file through the shared cache and watches it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace

# csvfmt holds the inline format because the Settings CSV's answers column uses it too, and csvfmt may import only
# the standard library.
from . import csvfmt, store
from .csvfmt import CsvSource, parse_inline

try:
    import bpy
except ImportError:  # the pure suites import this module for parse_inline() and is_csv_path()
    bpy = None
if bpy is not None:
    from . import evaluator, geometry, natsort, trust
    from .evaluator import Ctx, FileError, Hint, Pending, Unresolvable, as_text, eval_input
    from .watch import WATCH

SOURCES = ("TEXT", "COLLECTION", "MATERIALS", "ATTRIBUTES")
MAX_ITEMS = 255
DECISION_PREFIXES = ("GeometryNodeOpenAct", "GeometryNodeJev")  # OpenAct nodes and their legacy aliases
STORE = "GeometryNodeStoreNamedAttribute"


@dataclass(frozen=True)
class Item:
    name: str
    when: str = ""
    payload: dict = field(default_factory=dict)  # the row's numeric Items CSV columns, where the node bakes them
    offered: bool = True  # False for an empty material slot: it keeps its ID but is never an option


@dataclass(frozen=True)
class ItemList:
    source: str  # one of SOURCES
    items: tuple[Item, ...]  # in ID order
    raw: str = ""  # the Items word as given ("" for the other sources): its position in gn
    doc: csvfmt.CsvDoc | None = None  # the Items CSV, when the list is one
    csv: CsvSource | None = None
    datablock: object = None  # the Collection or Object, for the Instance and Material outputs
    fingerprint: tuple = ()  # changes whenever the entries may have: the plan cache's list-source part (5.3)
    problems: tuple = ()  # csvfmt.Problem warnings and info; the list is usable

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.items)

    def options(self) -> list[tuple[int, Item]]:
        """(ID, item) of the entries that can be offered, in ID order."""
        return [(i, item) for i, item in enumerate(self.items) if item.offered]


def is_csv_path(text: str) -> bool:
    """True when the stripped text is a single line ending in ".csv", in any case. Anything else is inline text."""
    text = text.strip()
    return "\n" not in text and "\r" not in text and text.lower().endswith(".csv")


# ------------------------------------------------------------------------------ paths


def resolve_path(text: str, node) -> str:
    """The absolute, normalised path of a path word: an Import node's Path (evaluator.import_path()) with '//' implied,
    so any relative path is relative to the .blend, or to the library file of a linked tree. Raises Hint for a
    relative path in an unsaved file."""
    text = text.strip()
    return evaluator.import_path(text if text.startswith("//") or os.path.isabs(text) else "//" + text, node)


def find_input(node, name):
    """The node's available input with this name, or None (a list-source socket absent under the dropdown)."""
    return next((s for s in node.inputs if s.name == name and evaluator._available(s)), None)


def _linked(sock):
    return any(not link.is_muted for link in sock.links)


def csv_source(node, input_name: str, ctx) -> CsvSource | None:
    """Where the CSV of a CSV-capable input comes from in this context, or None for inline text or nothing.

    An Import CSV node linked to the '<input_name> CSV' companion wins, and a typed path is then ignored with a
    warning (CsvSource.warning). Otherwise the String input counts when it holds a CSV path (is_csv_path())."""
    companion = find_input(node, f"{input_name} CSV")
    typed = find_input(node, input_name)
    if companion is not None and _linked(companion):
        source = evaluator.trace_csv_source(node, companion, ctx)
        if typed is not None and _ignored_text(typed, ctx):
            source = replace(source, warning=f"{input_name}: using the linked Import CSV; the typed path is ignored")
        return source
    if typed is None:
        return None
    text = as_text(eval_input(typed, ctx))
    return CsvSource(text.strip()) if is_csv_path(text) else None


def _ignored_text(sock, ctx):
    try:
        return bool(as_text(eval_input(sock, ctx)).strip())
    except (Hint, Pending, Unresolvable):
        return True  # linked to something: ignored all the same


def load_csv(node, source: CsvSource, schema: str = "answer_set") -> csvfmt.CsvDoc:
    """The CsvDoc of a source: resolved (resolve_path()), confined by the trust rule, parsed once per file change and
    shared (csvfmt.load()), and watched for edits on the node's behalf (WATCH). Raises Hint, or Unresolvable for a
    blocked read, which is never opened. Problems inside the file are the doc's."""
    path = resolve_path(source.path, node)
    message = trust.blocked(path)
    if message:
        raise Unresolvable(message)
    doc = csvfmt.load(path, source.delimiter, schema)
    WATCH.use(getattr(node, "uid", "") or node.name, doc, source.delimiter)
    return doc


def file_error(doc, error) -> FileError:
    """The node's error for a file with errors (csvfmt.CsvError), named after the file."""
    name, message = os.path.basename(doc.path), str(error)
    if not (message.startswith(name) or doc.path in message):
        message = f"{name}: {message}"
    return FileError(message, error.problems)


# ------------------------------------------------------------------------ list sources


def resolve_items(node, ctx, source: str | None = None, spec: str | None = None) -> ItemList:
    """The entries of a Pick, Match or Classify node in this context, in ID order (design 6.0).

    source: one of SOURCES, by default the node's list_source. spec: the Spec name the Items CSV is parsed for
    (csvfmt.parse_answer_set(): only Pick keeps payload), by default from the node's bl_idname. Raises Hint,
    Pending or Unresolvable (FileError for a CSV with errors)."""
    source = source or getattr(node, "list_source", "TEXT")
    spec = _spec_name(node) if spec is None else spec
    if source == "TEXT":
        found = _text_items(node, ctx, spec)
    elif source == "COLLECTION":
        found = _collection_items(node, ctx)
    elif source == "MATERIALS":
        found = _material_items(node, ctx)
    elif source == "ATTRIBUTES":
        found = _attribute_items(node, ctx)
    else:
        raise ValueError(f"unknown list source {source!r}")
    if len(found.items) > MAX_ITEMS:
        raise Unresolvable(f"Too many items ({len(found.items)}); the limit is {MAX_ITEMS}")
    return found


def _spec_name(node):
    name = node.bl_idname.removeprefix("GeometryNodeOpenAct").lower()
    return name if name and name in csvfmt.NODES else ""


def _word(node, name, ctx):
    sock = find_input(node, name)
    return eval_input(sock, ctx) if sock is not None else None


def _text_items(node, ctx, spec):
    from . import sources  # sources imports this module

    raw = as_text(_word(node, "Items", ctx))
    hint = "Type items separated by commas, or a .csv path"
    companion = find_input(node, "Items CSV")
    if getattr(node, "items_parse_v0", False) and not (companion is not None and _linked(companion)):
        # An upgraded Pick parses the Items texts of its stored answers as v0.1.0 did, which keeps their names and
        # words keys (§12.3); another text is a change, and v1 parsing takes over for good.
        if any(entry.gn[2:3] == (raw,) for entry in store.load(node).entries):
            items = tuple(Item(name) for name, _when in parse_inline(raw, mode="v0"))
            if not items:
                raise Hint(hint)
            return ItemList("TEXT", items, raw, fingerprint=("TEXT", raw, "v0"))
        try:
            node.items_parse_v0 = False
        except AttributeError:  # read-only now: linked data, or while Blender draws
            pass
    answers = sources.answer_set(node, ctx, "Items", "items", spec, hint)
    doc, source = answers.doc, answers.source
    items = tuple(Item(o.name, o.when, dict(o.payload)) for o in answers.options)
    fingerprint = ("TEXT", raw) + ((doc.path, source.delimiter, doc.stamp) if doc is not None else ())
    return ItemList("TEXT", items, raw, doc, source, None, fingerprint, answers.problems)


def _collection_items(node, ctx):
    collection = _word(node, "Collection", ctx)
    if not isinstance(collection, bpy.types.Collection):
        raise Hint("Choose a collection")
    # Collection Info (Separate Children) instances child objects and child collections in this order.
    names = [obj.name for obj in collection.objects] + [child.name for child in collection.children]
    names.sort(key=natsort.sort_key)
    if not names:
        raise Hint(f"The collection '{collection.name}' is empty")
    return ItemList("COLLECTION", tuple(Item(name) for name in names), datablock=collection,
                    fingerprint=("COLLECTION", tuple(names)))


def _material_items(node, ctx):
    obj = _word(node, "Object", ctx)
    if not isinstance(obj, bpy.types.Object):
        raise Hint("Choose an object")
    names = [slot.material.name if slot.material is not None else "" for slot in obj.material_slots]
    if not any(names):
        raise Hint(f"'{obj.name}' has no materials")
    # ID = material index, so an empty slot keeps its place, but it is never offered.
    return ItemList("MATERIALS", tuple(Item(name, offered=bool(name)) for name in names), datablock=obj,
                    fingerprint=("MATERIALS", tuple(names)))


def _attribute_items(node, ctx):
    obj = _word(node, "Object", ctx)
    if obj is None:  # empty means the modifier's own object
        obj = ctx.obj
    if not isinstance(obj, bpy.types.Object):
        raise Hint("Choose an object")
    names = geometry.attribute_names(obj, exclude=written_names(ctx))
    if not names:
        raise Hint(f"'{obj.name}' has no attributes to offer")
    return ItemList("ATTRIBUTES", tuple(Item(name) for name in names), datablock=obj,
                    fingerprint=("ATTRIBUTES", tuple(names)))


def written_names(ctx) -> set[str]:
    """Names that the modifier's tree, nested groups included, writes with Store Named Attribute: a static scan.

    A Name that may come from an OpenAct node is left out: leaving its answer off the list would change the answer.
    """
    names = set()

    def walk(tree, context, trail):
        for node in tree.nodes:
            if node.mute:
                continue
            if node.bl_idname == STORE:
                name = _static_name(next(s for s in node.inputs if s.identifier == "Name"), context)
                if name:
                    names.add(name)
            elif node.bl_idname == "GeometryNodeGroup" and node.node_tree is not None:
                key = node.node_tree.as_pointer()
                if key not in trail and len(trail) < 32:
                    walk(node.node_tree, context.push(node), trail | {key})

    walk(ctx.root, Ctx(ctx.obj, ctx.mod, (), ctx.root), frozenset({ctx.root.as_pointer()}))
    return names


def _static_name(sock, ctx):
    if not _linked(sock):
        return sock.default_value
    if _from_decision(sock, ctx):
        return ""
    try:
        return as_text(eval_input(sock, ctx))
    except (Hint, Pending, Unresolvable):
        return ""


def _from_decision(sock, ctx):
    """True when the socket's value may come from an OpenAct node, upstream through any node or group boundary."""
    stack, seen = [(sock, ctx)], set()
    while stack:
        sock, ctx = stack.pop()
        key = (sock.as_pointer(), ctx.key())
        if key in seen:
            continue
        seen.add(key)
        if len(seen) > 4096:
            return True  # too far to tell
        for link in sock.links:
            if link.is_muted:
                continue
            up, out = link.from_node, link.from_socket
            if up.bl_idname.startswith(DECISION_PREFIXES):
                return True
            if up.bl_idname == "NodeGroupInput":
                if ctx.path:
                    stack += [(s, ctx.pop()) for s in ctx.path[-1].inputs if s.identifier == out.identifier]
            elif up.bl_idname == "GeometryNodeGroup" and up.node_tree is not None:
                for n in up.node_tree.nodes:
                    if n.bl_idname == "NodeGroupOutput" and n.is_active_output:
                        stack += [(s, ctx.push(up)) for s in n.inputs if s.identifier == out.identifier]
            else:
                stack += [(s, ctx) for s in up.inputs]
    return False
