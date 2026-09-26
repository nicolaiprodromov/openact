"""Operators: install, start and stop the model; re-ask a node; open files; trust a .blend's file reads."""

from __future__ import annotations

import os
import subprocess
import sys

import bpy

from . import engine, server, trust
from . import prefs as P


def _open_path(path):
    if sys.platform == "win32":
        os.startfile(path)  # noqa: S606 - opening a local file/folder for the user
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


class OPENACT_OT_install_runtime(bpy.types.Operator):
    bl_idname = "openact.install_runtime"
    bl_label = "Install Model"
    bl_description = ("Download the local model runtime: Python, PyTorch, SemIf and the Qwen3.5-4B model "
                      "(about 12 GB, one time)")

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(
            self, event, title="Install the local model?",
            message=f"Downloads about 12 GB into {P.runtime_root()}", confirm_text="Install",
        )

    def execute(self, context):
        if not bpy.app.online_access:
            self.report({'ERROR'}, "Online access is disabled. Enable Preferences > System > Network > "
                                   "Allow Online Access to download the model")
            return {'CANCELLED'}
        if not server.install():
            self.report({'WARNING'}, "An install is already running")
            return {'CANCELLED'}
        self.report({'INFO'}, "Installing the model in the background")
        return {'FINISHED'}


class OPENACT_OT_server_start(bpy.types.Operator):
    bl_idname = "openact.server_start"
    bl_label = "Start Model"
    bl_description = "Start the local model server (loads the model onto the GPU, about 9 GB of VRAM)"

    def execute(self, context):
        if not server.start():
            _state, message = server.status()
            self.report({'ERROR'}, message or "Could not start the model")
            return {'CANCELLED'}
        return {'FINISHED'}


class OPENACT_OT_server_stop(bpy.types.Operator):
    bl_idname = "openact.server_stop"
    bl_label = "Stop Model"
    bl_description = "Stop the local model server and free its GPU memory. Answers already given stay"

    def execute(self, context):
        server.stop()
        return {'FINISHED'}


class OPENACT_OT_reask(bpy.types.Operator):
    bl_idname = "openact.reask"
    bl_label = "Ask Again"
    bl_description = "Forget this node's answers and ask the model again"
    bl_options = {'REGISTER', 'UNDO'}

    uid: bpy.props.StringProperty(options={'HIDDEN'})

    def execute(self, context):
        _tree, node = engine.find_node(self.uid)
        if node is None:
            return {'CANCELLED'}
        try:
            engine.reask(node)
        except ValueError as error:  # a newer OpenAct's answers
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        return {'FINISHED'}


class OPENACT_OT_open_log(bpy.types.Operator):
    bl_idname = "openact.open_log"
    bl_label = "Open Log"
    bl_description = "Open the log of the model server or of the install"

    def execute(self, context):
        root = P.runtime_root()
        for path in (server.STATE.log_path, os.path.join(server.state_dir(root), "server.log"),
                     os.path.join(root, "install.log")):
            if path and os.path.isfile(path):
                _open_path(path)
                return {'FINISHED'}
        self.report({'WARNING'}, "No log yet")
        return {'CANCELLED'}


class OPENACT_OT_open_runtime_folder(bpy.types.Operator):
    bl_idname = "openact.open_runtime_folder"
    bl_label = "Open Runtime Folder"
    bl_description = "Show the folder that holds the model runtime (delete it to uninstall the model)"

    def execute(self, context):
        root = P.runtime_root()
        if not os.path.isdir(root):
            self.report({'WARNING'}, "The runtime folder does not exist yet")
            return {'CANCELLED'}
        _open_path(root)
        return {'FINISHED'}


class OPENACT_OT_allow_file(bpy.types.Operator):
    bl_idname = "openact.allow_file"
    bl_label = "Allow This File"
    bl_description = ("Let this .blend file read CSV and text files outside its own folder. Only this file is "
                      "trusted, not the rest of its folder")

    @classmethod
    def poll(cls, context):
        return bool(bpy.data.filepath)

    def execute(self, context):
        trust.allow(bpy.data.filepath)
        engine.schedule(immediate=True)  # the reads that were blocked
        return {'FINISHED'}


classes = (OPENACT_OT_install_runtime, OPENACT_OT_server_start, OPENACT_OT_server_stop, OPENACT_OT_reask,
           OPENACT_OT_open_log, OPENACT_OT_open_runtime_folder, OPENACT_OT_allow_file)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
