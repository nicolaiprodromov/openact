"""OpenAct: decision nodes for Geometry Nodes, run by a local model.

Six nodes: Ask, Pick and Dials judge the whole situation, Match and Classify each item of a list, and Rules each
measured situation. The model runs on this computer: SemIf's open "System One" method on Qwen3.5-4B, served by a
small Jev-compatible server in its own Python environment.
"""

bl_info = {  # legacy add-on installs (the extension manifest is used on Blender 4.2+)
    "name": "OpenAct",
    "author": "Nic",
    "version": (1, 0, 0),
    "blender": (4, 5, 0),
    "location": "Geometry Node Editor > Add > OpenAct, Sidebar > OpenAct",
    "description": "Decision nodes for Geometry Nodes, run by a local model",
    "category": "Node",
}

import bpy

from . import engine, nodes, ops, prefs, server, trust, ui, watch

_modules = (ops, nodes, ui, trust, watch)


def register():
    # The manifest stops extension installs on older versions; legacy folder installs
    # (addon_utils.enable) only meet this check.
    if bpy.app.version < (4, 5, 0):
        raise RuntimeError("OpenAct needs Blender 4.5 or newer")
    for cls in prefs.classes:
        bpy.utils.register_class(cls)
    for module in _modules:
        module.register()
    engine.register()
    server.refresh()


def unregister():
    engine.unregister()
    server.shutdown_owned()
    for module in reversed(_modules):
        module.unregister()
    for cls in reversed(prefs.classes):
        bpy.utils.unregister_class(cls)
