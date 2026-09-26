"""Hidden table objects, `.OpenAct Table <uid>` (§8.5): the per-element rows of a node's materialised entries.

A mesh of loose vertices, one per flat row (vertex i = flat row i), with one POINT attribute `oa_<column>` per
column. It is linked to no collection: its only user is the Object Info node of the node's inner group, so it is
saved, appended and linked with that group, survives Purge Unused Data while the group uses it, and needs no
re-linking on any pass. inner.build() regenerates it from the node's entries at any time.
"""

from __future__ import annotations

import bpy

PREFIX = ".OpenAct Table "
UID = "openact_uid"  # marks a table object for collect_garbage(), which goes by reference, never by name
NODE = "OpenAct Table"  # the Object Info node of the inner group that holds the table object
ATTR = "oa_"
_CAST = {'FLOAT': float, 'INT': int, 'BOOLEAN': bool}


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


def write_rows(node, flat_columns, types=None):
    """Write the flat rows into the node's table object: {column: values}, every column one value per flat row,
    stored as the POINT attribute "oa_<column>". types maps a column to 'FLOAT', 'INT' or 'BOOLEAN'; without it an
    existing attribute keeps its type and a new one is inferred. Attributes of columns no longer written go.
    Returns the object, or None when the node's group reads no table."""
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
        try:
            attribute.data.foreach_set("value", values)
        except TypeError:  # e.g. floats such as 1.0 in an INT column
            attribute.data.foreach_set("value", [_CAST[kind](v) for v in values])
    mesh.update()
    return obj


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
