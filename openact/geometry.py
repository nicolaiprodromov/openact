"""Attribute names and values of objects.

attribute_names() feeds the Attributes list source. It reads the object's original data, never the evaluated
geometry, which holds the modifier stack's output and so this node's own answer (design 6.0). read_attributes() is
the agents hook (design 5.6); v1 itself doesn't use it.
"""

from __future__ import annotations

import bpy

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
