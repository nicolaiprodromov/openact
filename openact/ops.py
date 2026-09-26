"""Operators: install or locate the runtime, start and stop the model; re-ask a node; write its Overrides CSV; open
files; trust a .blend's file reads."""

from __future__ import annotations

import csv
import os
import subprocess
import sys

import bpy

from . import csvfmt, engine, lists, server, store, trust
from . import prefs as P
from .evaluator import Hint, Pending, Unresolvable
from .specs.base import SOURCES, UNSURE_BELOW

PERMISSION = "Close the file in the other program (e.g. Excel) and try again."
EXPORTED = ("answer", "lock", "model_answer", "confidence", "unsure", "source")  # after the key columns (§7.6)


class OverridesError(Exception):
    """Why the Overrides CSV can't be written: the operator reports it."""


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
        if not server.install():
            state, message = server.status()
            if state == "installing":
                self.report({'WARNING'}, "An install is already running")
            else:
                self.report({'ERROR'}, message)  # refused, e.g. online access is off
            return {'CANCELLED'}
        self.report({'INFO'}, "Installing the model in the background")
        return {'FINISHED'}


class OPENACT_OT_locate_runtime(bpy.types.Operator):
    bl_idname = "openact.locate_runtime"
    bl_label = "Locate Existing Runtime..."
    bl_description = ("Use a model runtime that is already installed, such as the one Jev Nodes installed, "
                      "instead of downloading it again")

    directory: bpy.props.StringProperty(subtype='DIR_PATH')
    filter_folder: bpy.props.BoolProperty(default=True, options={'HIDDEN'})

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        root = os.path.normpath(bpy.path.abspath(self.directory))
        if not P.runtime_installed(root):
            self.report({'ERROR'}, f"No installed runtime in {root}: choose the folder that holds 'venv' and 'hf'")
            return {'CANCELLED'}
        prefs = P.get_prefs()
        prefs.runtime_dir = root
        server.probe_runtime(prefs)
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


# ----------------------------------------------------------------------------- the Overrides CSV (§7.6)

def export_context(node, index=0):
    """(ctx, Job, entry or None) of the node's live context `index` (0: the first), as the pass plans it."""
    uses = engine._jobs(node)
    if not uses:
        raise OverridesError("Nothing to export: the node has no answered context")
    if not 0 <= index < len(uses):
        raise OverridesError(f"The node has {len(uses)} context{'s' if len(uses) != 1 else ''}, not {index + 1}")
    ctx, job = uses[index]
    entries = store.load(node).entries
    entry = engine.SPECS[node.bl_idname].resolve(job, entries, engine.LAST_SHOWN.get((node.uid, ctx.key()), ""))[0]
    return ctx, job, entry


def overrides_target(node, ctx):
    """(absolute path, delimiter) of the Overrides CSV in ctx; (None, None) when the socket is empty and unlinked, so
    the export proposes a file and fills the socket with it."""
    source = lists.csv_source(node, "Overrides", ctx)
    if source is not None:
        return lists.resolve_path(source.path, node), source.delimiter
    socket = lists.find_input(node, "Overrides")
    if socket is None:
        raise OverridesError("This node has no Overrides input")
    if socket.is_linked or socket.default_value.strip():
        raise OverridesError("Overrides must name a .csv file to write (set it on the node or the modifier)")
    return None, None


def proposed_path(node):
    """//<tree>_<node>_overrides.csv, beside the .blend (§7.6)."""
    return f"//{bpy.path.clean_name(node.id_data.name)}_{bpy.path.clean_name(node.name)}_overrides.csv"


def merged_rows(spec, job, entry, path, delimiter):
    """(header, rows) of the Overrides CSV: one row per lockable unit of the table, in table order. Locked rows of
    the existing file are kept verbatim, user columns included; other rows keep their user columns."""
    keys = spec.override_keys(job)
    header, old = [*keys, *EXPORTED], {}
    if os.path.exists(path):
        doc = csvfmt.read_csv(path, delimiter)
        errors = [p.message for p in doc.problems if p.severity == csvfmt.ERROR]
        if errors:
            raise OverridesError(f"{os.path.basename(path)}: {errors[0]}. Fix it, or write another file")
        known = {column.casefold() for column in header}
        header += [column for column in doc.header if column.casefold() not in known]
        for row in doc.rows:
            cells = {column.casefold(): cell for column, cell in zip(doc.header, row)}
            old.setdefault(csvfmt.KEY_SEP.join(cells.get(k.casefold(), "").casefold() for k in keys), cells)
    kinds = {src: source for source, src in SOURCES.items()}
    if entry is not None and len(entry.rows.name) == len(job.rows):
        table = [(kinds.get(entry.rows.src[i], "model"), {column: v[i] for column, v in entry.rows.v.items()})
                 for i in range(len(job.rows))]
    else:  # not answered yet: lock and code rows have their values, the others are pending
        table = [(row.source, dict(row.values)) for row in job.rows]

    def locked(cells):
        return cells.get("lock", "").casefold() in csvfmt.LOCKED

    def line(cells):
        return [cells.get(column.casefold(), "") for column in header]

    rows, seen = [], set()
    for unit, indices in spec.override_units(job):
        key = csvfmt.KEY_SEP.join(cell.strip().casefold() for cell in unit)
        seen.add(key)
        cells = dict(old.get(key, {}))
        if not locked(cells):
            answer, c, source = spec.export_unit(job, indices, table)
            cells.update({k.casefold(): cell for k, cell in zip(keys, unit)})
            cells.update(answer=answer, lock="0", model_answer=answer, confidence="" if c is None else f"{c:.2f}",
                         unsure="" if c is None else str(int(c < UNSURE_BELOW)), source=source)
        rows.append(line(cells))
    rows += [line(cells) for key, cells in old.items() if key not in seen and locked(cells)]
    return header, rows


def write_csv(path, header, rows, delimiter=","):
    """Write <path>.tmp, then replace path with it, so a reader never sees half a file."""
    temporary = path + ".tmp"
    try:
        with open(temporary, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter=delimiter, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)
        os.replace(temporary, path)
    except OSError as error:
        if os.path.exists(temporary):
            os.remove(temporary)
        if isinstance(error, PermissionError):
            raise OverridesError(PERMISSION) from None
        raise OverridesError(f"Can't write {path}: {error.strerror or error}") from None


def _delimiter(path):
    """The existing file's delimiter, sniffed as csvfmt.read_csv() does; ',' for a new file."""
    try:
        with open(path, encoding="utf-8-sig", errors="replace") as handle:
            first = next((line for line in handle if line.strip()), "")
    except OSError:
        return ","
    return next((sign for sign in (",", ";", "\t") if sign in first), ",")


def write_overrides(node, index=0, path=None):
    """Export the table of the node's live context `index` to its Overrides CSV (§7.6); returns the path written.
    path: the file chosen for an empty, unlinked Overrides socket, which then takes it. Raises OverridesError."""
    spec = engine.SPECS[node.bl_idname]
    try:
        ctx, job, entry = export_context(node, index)
        target, delimiter = overrides_target(node, ctx)
    except (Hint, Pending, Unresolvable) as error:
        raise OverridesError(str(error)) from None
    fill = target is None
    if fill:
        if not path:
            raise OverridesError("Choose a file for the Overrides CSV")
        target = os.path.normpath(bpy.path.abspath(path))
        if not target.lower().endswith(".csv"):
            target += ".csv"
    blocked = trust.blocked(target)
    if blocked:
        raise OverridesError(blocked.replace("reading", "writing", 1))
    delimiter = delimiter or _delimiter(target)
    header, rows = merged_rows(spec, job, entry, target, delimiter)
    write_csv(target, header, rows, delimiter)
    if fill:
        try:
            shown = bpy.path.relpath(target) if bpy.data.filepath else target
        except ValueError:  # another drive
            shown = target
        node.inputs["Overrides"].default_value = shown
    engine.schedule(immediate=True, node=node)
    return target


class OPENACT_OT_write_overrides(bpy.types.Operator):
    bl_idname = "openact.write_overrides"
    bl_label = "Write Overrides CSV"
    bl_description = ("Write this node's answers to its Overrides CSV, to lock rows in (lock = 1). Rows you locked, "
                      "and your own columns, are kept")
    bl_options = {'REGISTER', 'UNDO'}

    uid: bpy.props.StringProperty(options={'HIDDEN'})
    context_index: bpy.props.IntProperty(name="Context", default=0, min=0, options={'HIDDEN'},
                                         description="Which of the node's live contexts to export (0 = the first)")
    filepath: bpy.props.StringProperty(subtype='FILE_PATH', options={'SKIP_SAVE'})
    filter_glob: bpy.props.StringProperty(default="*.csv", options={'HIDDEN'})

    def invoke(self, context, event):
        _tree, node = engine.find_node(self.uid)
        if node is None:
            return {'CANCELLED'}
        try:
            ctx, _job, _entry = export_context(node, self.context_index)
            target, _delimiter = overrides_target(node, ctx)
        except (OverridesError, Hint, Pending, Unresolvable) as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        if target is None and not self.filepath:  # an empty socket: propose a file beside the .blend
            self.filepath = bpy.path.abspath(proposed_path(node)) if bpy.data.filepath else proposed_path(node)[2:]
            context.window_manager.fileselect_add(self)
            return {'RUNNING_MODAL'}
        return self.execute(context)

    def execute(self, context):
        _tree, node = engine.find_node(self.uid)
        if node is None or node.id_data.library is not None:
            return {'CANCELLED'}
        try:
            path = write_overrides(node, self.context_index, self.filepath or None)
        except OverridesError as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        self.report({'INFO'}, f"Wrote {path}")
        return {'FINISHED'}


classes = (OPENACT_OT_install_runtime, OPENACT_OT_locate_runtime, OPENACT_OT_server_start, OPENACT_OT_server_stop,
           OPENACT_OT_reask, OPENACT_OT_open_log, OPENACT_OT_open_runtime_folder, OPENACT_OT_allow_file,
           OPENACT_OT_write_overrides)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
