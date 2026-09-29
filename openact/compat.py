"""Differences between the Blender versions OpenAct supports (4.5 LTS, 5.0, 5.1), in one place."""

from __future__ import annotations

import bpy

# Blender 4.5 accepts only AUTO and SINGLE as an interface input's structure type; 5.0 added FIELD,
# for Int, Float and Bool sockets but never String.
FIELD_INPUTS = bpy.app.version >= (5, 0, 0)


def structure_type(socket_type: str, kind: str) -> str:
    """The structure type of a node input: SINGLE for String words and for single-value inputs (kind
    "single", the Trigger) on every version, FIELD for per-element inputs where the version accepts it,
    AUTO otherwise."""
    string = socket_type == 'NodeSocketString'
    if (kind == "word" and string) or kind == "single":
        return 'SINGLE'
    if kind == "field" and FIELD_INPUTS and not string:
        return 'FIELD'
    return 'AUTO'
