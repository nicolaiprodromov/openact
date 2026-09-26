"""Sidebar panel (Geometry Node editor > N > OpenAct) and the preferences panel."""

from __future__ import annotations

import bpy

from . import engine, server
from . import prefs as P

_ICONS = {
    "ready": 'CHECKMARK',
    "starting": 'SORTTIME',
    "installing": 'IMPORT',
    "stopped": 'PAUSE',
    "not_installed": 'ERROR',
    "error": 'ERROR',
    "unknown": 'QUESTION',
}
_TITLES = {
    "ready": "Model ready",
    "starting": "Starting",
    "installing": "Installing",
    "stopped": "Model stopped",
    "not_installed": "Model not installed",
    "error": "Error",
    "unknown": "Checking...",
}


def _wrap(layout, text, width=46, icon='NONE'):
    col = layout.column(align=True)
    line, first = "", True
    for word in text.split():
        if line and len(line) + len(word) + 1 > width:
            col.label(text=line, icon=icon if first else 'BLANK1')
            line, first = "", False
        line = f"{line} {word}".strip()
    if line:
        col.label(text=line, icon=icon if first else 'BLANK1')


def draw_server(layout):
    state, message = server.status()
    box = layout.box()
    box.label(text=_TITLES.get(state, state), icon=_ICONS.get(state, 'QUESTION'))
    if message:
        _wrap(box, message, icon='BLANK1')
    row = box.row(align=True)
    installed = state not in ("not_installed", "error") or P.runtime_installed(P.runtime_root())
    if state == "installing":
        row.operator("openact.open_log", icon='TEXT')
        return
    if not installed:
        row.operator("openact.install_runtime", icon='IMPORT')
    elif state in ("ready", "starting"):
        row.operator("openact.server_stop", icon='PAUSE')
    else:
        row.operator("openact.server_start", icon='PLAY')
    row.operator("openact.open_log", text="", icon='TEXT')


def draw_preferences(prefs, context, layout):
    draw_server(layout)
    col = layout.column()
    col.use_property_split = True
    col.prop(prefs, "auto_start")
    col.prop(prefs, "device")
    sub = col.column()
    sub.active = prefs.device != 'cpu'
    sub.prop(prefs, "gpu")
    col.prop(prefs, "debias")
    col.prop(prefs, "port")
    col.prop(prefs, "runtime_dir")
    row = col.row()
    row.label(text=P.runtime_root(prefs), icon='FILE_FOLDER')
    row.operator("openact.open_runtime_folder", text="", icon='FILEBROWSER')
    note = layout.box().column(align=True)
    note.label(text=f"Model: {P.MODEL_LABEL} ({P.MODEL_REVISION[:10]}), runs on this computer", icon='INFO')
    note.label(text="SemIf (formerly OpenJev), MIT - github.com/TheoLeeCJ/SemIf-OpenJev", icon='BLANK1')
    note.label(text="Qwen3.5-4B by Alibaba Qwen, Apache-2.0. Not affiliated with TypeSafe or Jev", icon='BLANK1')


class NODE_PT_openact(bpy.types.Panel):
    bl_space_type = 'NODE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "OpenAct"
    bl_label = "OpenAct"

    @classmethod
    def poll(cls, context):
        return getattr(context.space_data, "tree_type", None) == 'GeometryNodeTree'

    def draw(self, context):
        layout = self.layout
        draw_server(layout)
        tree = context.space_data.edit_tree
        node = tree.nodes.active if tree is not None else None
        if node is not None and node.bl_idname in engine.OPENACT_TYPES:
            layout.label(text=node.name, icon='NODE')
            node.draw_buttons_ext(context, layout)
        else:
            col = layout.column(align=True)
            col.label(text="Add nodes with Shift A > OpenAct", icon='INFO')
            col.label(text="Ask: a question in, a number out", icon='BLANK1')
            col.label(text="Pick: a prompt and a list in, one item out", icon='BLANK1')


classes = (NODE_PT_openact,)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
