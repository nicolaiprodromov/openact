"""Hidden table objects, `.OpenAct Table <uid>` (§8.5): every stored entry of a node, as rows Geometry Nodes reads.

A mesh of loose vertices, one per row (vertex i = row i), with one POINT attribute `oa_<column>` per column. Point 0
is the meta row, whose DEFAULT column holds the header row a miss shows; then come one block per entry, its header row
(HEAD True, its KEY and the COUNT of element rows that follow, and its single values) and its element rows, in the
layout inner.py writes. It is linked to no collection: its only user is the Object Info node of the node's inner
group, so it is saved, appended and linked with that group, survives Purge Unused Data while the group uses it, and
needs no re-linking on any pass. inner.build() regenerates it from the node's entries at any time.
"""

from __future__ import annotations

import bpy
import numpy as np

PREFIX = ".OpenAct Table "
UID = "openact_uid"  # marks a table object for collect_garbage(), which goes by reference, never by name
NODE = "OpenAct Table"  # the Object Info node of the inner group that holds the table object
ATTR = "oa_"
DIGEST = "openact_digest"  # ID property of the mesh: what write_rows() last wrote, so an unchanged table isn't written
DEFAULT = "default"  # the meta row's column: the header row a miss shows, -1 for none (§8.2)
HEAD, KEY, COUNT = "head", "key", "count"  # a header row's columns: True, its entry's key, the element rows after it
_CAST = {'FLOAT': float, 'INT': int, 'BOOLEAN': bool}
_BUFFERS = {'FLOAT': np.float32, 'INT': np.intc, 'BOOLEAN': np.bool_}  # np.int32 takes a slow path on Windows


def find(node):
    """The table object the node's inner group reads, or None."""
    tree = node.node_tree
    info = tree.nodes.get(NODE) if tree is not None else None
    if info is None or info.bl_idname != "GeometryNodeObjectInfo":
        return None
    return info.inputs["Object"].default_value


def ensure_table(node):
    """The node's table object: the one its inner group reads, else a new empty one. inner.build() calls it
    before clearing the group, while the Object Info node still holds the reference."""
    obj = find(node)
    if obj is None or obj.library is not None:  # linked data is read-only
        obj = bpy.data.objects.new(PREFIX, bpy.data.meshes.new(PREFIX))
        _label(obj, node.uid)
    return obj


def write_rows(node, flat_columns, types=None, digest=""):
    """Write the rows into the node's table object: {column: values}, every column one value per row, stored as the
    POINT attribute "oa_<column>". types maps a column to 'FLOAT', 'INT' or 'BOOLEAN'; without it an existing
    attribute keeps its type and a new one is inferred. Attributes of columns no longer written go. digest: what the
    rows hold, which digest() returns until the next write ("" for unknown). Returns the object, or None when the
    node's group reads no table."""
    obj = find(node)
    if obj is None:
        return None
    mesh = obj.data
    counts = {len(values) for values in flat_columns.values()}
    if len(counts) > 1:
        raise ValueError("every table column needs one value per flat row")
    count = counts.pop() if counts else 0
    kinds = {ATTR + name: _kind(mesh, name, values, types) for name, values in flat_columns.items()}
    if count < len(mesh.vertices):
        mesh.clear_geometry()  # removes every custom attribute too; they are re-created below
    if count > len(mesh.vertices):
        mesh.vertices.add(count - len(mesh.vertices))  # keeps and resizes the existing attributes
    for name in [a.name for a in mesh.attributes if a.name.startswith(ATTR)]:
        attribute = mesh.attributes[name]
        if kinds.get(name) != attribute.data_type or attribute.domain != 'POINT':
            mesh.attributes.remove(attribute)
    for name, values in flat_columns.items():
        kind = kinds[ATTR + name]
        attribute = mesh.attributes.get(ATTR + name) or mesh.attributes.new(ATTR + name, kind, 'POINT')
        if attribute is None:  # Blender 4.5 refuses names of 68 bytes or more
            raise ValueError(f"the table column '{ATTR + name}' can't be created")
        try:
            attribute.data.foreach_set("value", values)
        except TypeError:  # e.g. floats such as 1.0 in an INT column
            attribute.data.foreach_set("value", [_CAST[kind](v) for v in values])
    mesh[DIGEST] = digest
    mesh.update()
    return obj


def digest(node):
    """The digest the node's table rows were last written with (write_rows()), "" when unknown or there is no table."""
    obj = find(node)
    return obj.data.get(DIGEST, "") if obj is not None else ""


def append_rows(node, flat_columns):
    """Add rows at the end of the node's table object, values only (§8.8): {column: values}, every existing column
    given, as many values each. The rows before keep their values, so nothing else moves. The digest becomes unknown,
    so the next update_values() digests the rows afresh. Returns the object, or None when the group reads no table."""
    obj = find(node)
    if obj is None:
        return None
    mesh = obj.data
    counts = {len(values) for values in flat_columns.values()}
    if len(counts) > 1:
        raise ValueError("every table column needs one value per row")
    count = counts.pop() if counts else 0
    at = len(mesh.vertices)
    mesh.vertices.add(count)  # keeps the existing attributes' values; the new rows hold zeros until written
    for name, values in flat_columns.items():
        attribute = mesh.attributes[ATTR + name]
        buffer = np.empty(at + count, dtype=_BUFFERS[attribute.data_type])
        attribute.data.foreach_get("value", buffer)  # foreach_set() writes a whole column
        buffer[at:] = values
        attribute.data.foreach_set("value", buffer)
    mesh[DIGEST] = ""
    mesh.update()
    return obj


def header_rows(node):
    """{key: row} of the header rows of the node's table object (§8.5). A key is on one header row, the pass refusing
    any entry whose key an earlier one has."""
    heads, keys = read(node, HEAD), read(node, KEY)
    if heads is None or keys is None:
        return {}
    rows = np.flatnonzero(heads)
    return dict(zip(keys[rows].tolist(), rows.tolist()))


def read(node, name):
    """Column `name` of the node's table object as an array, one value per row, or None when it has no such column."""
    obj = find(node)
    attribute = obj.data.attributes.get(ATTR + name) if obj is not None else None
    if attribute is None or attribute.data_type not in _BUFFERS:
        return None
    buffer = np.empty(len(attribute.data), dtype=_BUFFERS[attribute.data_type])
    attribute.data.foreach_get("value", buffer)
    return buffer


def write_value(node, name, row, value):
    """Set one value of column `name` in the node's table object, values only, when it differs. True when written."""
    obj = find(node)
    attribute = obj.data.attributes.get(ATTR + name) if obj is not None else None
    if attribute is None or row >= len(attribute.data) or attribute.data[row].value == value:
        return False
    attribute.data[row].value = value
    obj.data.update()
    return True


def copy_for(node):
    """Give a node whose inner group was just copied (engine.ensure_owned) its own table object, as
    `.OpenAct Table <its uid>`: the copied group still reads the original's. Returns the copy, or None."""
    obj = find(node)
    if obj is None:
        return None
    own = obj.copy()
    own.data = obj.data.copy()
    _label(own, node.uid)
    node.node_tree.nodes[NODE].inputs["Object"].default_value = own
    return own


def collect_garbage(nodes):
    """Remove every local table object that no inner group of `nodes` (all OpenAct nodes, as engine.iter_nodes()
    yields them) reads, with its mesh. By reference, never by name: appending renames duplicates to .001, so a
    name could point at a live table. Returns how many were removed."""
    used = {obj.as_pointer() for obj in (find(node) for _tree, node in nodes) if obj is not None}
    orphans = [obj for obj in bpy.data.objects if obj.library is None and UID in obj and obj.as_pointer() not in used]
    for obj in orphans:
        mesh = obj.data
        bpy.data.objects.remove(obj)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    return len(orphans)


def _label(obj, uid):
    obj.name = obj.data.name = PREFIX + uid
    obj[UID] = uid


def _kind(mesh, name, values, types):
    if types and name in types:
        return types[name]
    existing = mesh.attributes.get(ATTR + name)
    if existing is not None:
        return existing.data_type
    if values and all(type(v) is bool for v in values):
        return 'BOOLEAN'
    if values and all(type(v) is int for v in values):
        return 'INT'
    return 'FLOAT'
