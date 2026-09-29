"""Attribute names and values of objects.

attribute_names() feeds the Attributes list source. It reads the object's original data, never the evaluated
geometry, which holds the modifier stack's output and so this node's own answer (design 6.0). int_values() reads the
evaluated geometry for the geometry channel: Rules' stamp of the cells in use (design 5.8). read_attributes() is the
agents hook (design 5.6); OpenAct itself doesn't use it.
"""

from __future__ import annotations

import bpy
import numpy as np

from . import natsort


def attribute_names(obj, exclude=()) -> list[str]:
    """Attribute names of the object's original data, natsorted, without "." names and without exclude."""
    obj = getattr(obj, "original", obj)
    attributes = getattr(getattr(obj, "data", None), "attributes", None)  # meshes, curves, point clouds
    if attributes is None:
        return []
    skip = set(exclude)
    names = {a.name for a in attributes if not a.name.startswith(".") and a.name not in skip}
    return sorted(names, key=natsort.sort_key)


def int_values(obj, name, depsgraph):
    """Every value of the Int attribute `name` on the evaluated object's mesh, point cloud and curves, in one np.intc
    array, or None when none of them holds it (design 5.8). foreach_get into an np.intc buffer is the fast path: an
    np.int32 one takes a slow path on Windows, 10x slower at 100k points. The GeometrySet stays referenced while its
    components are read."""
    try:
        geometry = bpy.types.GeometrySet.from_evaluated_object(obj.evaluated_get(depsgraph))
    except RuntimeError:  # not evaluated in this depsgraph
        return None
    found = []
    for component in (geometry.mesh, geometry.pointcloud, geometry.curves):
        attributes = getattr(component, "attributes", None)
        attribute = attributes.get(name) if attributes is not None else None
        if attribute is not None and attribute.data_type == 'INT':
            values = np.empty(len(attribute.data), dtype=np.intc)
            attribute.data.foreach_get("value", values)
            found.append(values)
    del geometry
    return np.concatenate(found) if found else None


def read_attributes(obj, names, depsgraph=None) -> dict[str, list]:
    """Per-element values of the named attributes of the evaluated object, from its mesh, point cloud or curves.

    The GeometrySet stays referenced while its components are read: a component taken from a temporary set would
    point at freed data."""
    depsgraph = depsgraph or bpy.context.evaluated_depsgraph_get()
    geometry = bpy.types.GeometrySet.from_evaluated_object(obj.evaluated_get(depsgraph))
    found = {}
    for component in (geometry.mesh, geometry.pointcloud, geometry.curves):
        attributes = getattr(component, "attributes", None)
        if attributes is None:
            continue
        for name in names:
            attribute = attributes.get(name)
            if attribute is not None and name not in found:
                found[name] = [_value(item) for item in attribute.data]
    del geometry
    return found


def _value(item):
    for name in ("value", "vector", "color"):
        value = getattr(item, name, None)
        if value is not None:
            return value if isinstance(value, (bool, int, float, str)) else tuple(value)
    return None
