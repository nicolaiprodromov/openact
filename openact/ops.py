"""Operators: install or locate the runtime, start and stop the model; re-ask a node; write its Overrides CSV; show
what a row asks; open files; trust a .blend's file reads; upgrade the old Jev nodes of a v0.1.0 file."""

from __future__ import annotations

import contextlib
import csv
import os
import subprocess
import sys

import bpy

from . import csvfmt, engine, lists, migrate, server, store, trust, ui
from . import prefs as P
from .evaluator import Ctx, Hint, Pending, Unresolvable, find_uses
from .specs.base import UNSURE_BELOW, stored_rows, table_of

PERMISSION = "Close the file in the other program (e.g. Excel) and try again."
NEVER_SENT = {"lock": "Locked by the Overrides CSV: never sent to the model",
              "code": "Decided by code (one option applies): never sent to the model",
              "error": "No outcome applies to this cell: never sent to the model"}
NEVER_OFFERED = "An empty material slot: never offered or sent"  # a code row without a name


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
        engine.schedule(immediate=True)  # trust is in every plan's fingerprint: the pass plans blocked reads again
        return {'FINISHED'}


class OPENACT_OT_trust_folder(bpy.types.Operator):
    bl_idname = "openact.trust_folder"
    bl_label = "Trust This Folder"
    bl_description = ("Let every .blend file in this file's folder and its subfolders, now and in the future, read CSV "
                      "and text files outside its own folder")

    @classmethod
    def poll(cls, context):
        return bool(bpy.data.filepath)

    def execute(self, context):
        trust.allow(bpy.data.filepath, folder=True)
        engine.schedule(immediate=True)  # as Allow This File: the pass plans blocked reads again
        return {'FINISHED'}


class OPENACT_OT_open_file(bpy.types.Operator):
    bl_idname = "openact.open_file"
    bl_label = "Open File"
    bl_description = "Open this CSV file in the program your system opens CSV files with"

    filepath: bpy.props.StringProperty(subtype='FILE_PATH', options={'HIDDEN', 'SKIP_SAVE'})

    def execute(self, context):
        path = os.path.normpath(bpy.path.abspath(self.filepath))
        if not lists.is_csv_path(path):  # the system would run anything else it can open
            self.report({'ERROR'}, f"Only CSV files open from here, not {path}")
            return {'CANCELLED'}
        if not os.path.isfile(path):
            self.report({'ERROR'}, f"Can't find {path}")
            return {'CANCELLED'}
        _open_path(path)
        return {'FINISHED'}


# ------------------------------------------------------------------------------ sidebar inspection (§17.2)

def row_question(node, context=0, row=0) -> dict:
    """What row `row` of the node's context `context` asks the model, from the plan its last pass cached: its source
    ("lock", "code" and "error" rows are never sent), the state, question type, instructions, options in the order
    sent and debias, as sent or as they would be sent now, the cell key, the ask key it was answered with ("" when not
    answered), and whether that ask key is older than the one it would be answered with now. Asks nothing. Raises
    ValueError when there is no such row."""
    uses = engine.answers_for(node)
    if not 0 <= context < len(uses):
        raise ValueError("This context has no table yet")
    _label, job, entry = uses[context]
    if not 0 <= row < len(job.rows):
        raise ValueError(f"The table has no row {row + 1}")
    planned, source = job.rows[row], table_of(job, entry)[row][0]
    ask = ""
    if source == "model" and entry is not None:  # the entry row table_of() read: this one, or the one with its wkey
        ask = (entry.asks[entry.rows.ask[row]] if entry.words_key == job.words_key
               else next(old.ask for old in stored_rows(entry) if old.wkey == planned.wkey))
    now, cell = engine.ask_key_now(engine.SPECS[node.bl_idname], job.debias), planned.cell
    return {"name": planned.name, "source": source, "debias": job.debias, "ask": ask,
            "older": bool(ask and now and ask != now), "state": cell and cell.state,
            "type": cell and cell.question["type"], "instructions": cell and cell.question["instructions"],
            "options": cell and cell.question.get("criteria"), "key": cell and cell.key}


def question_text(found) -> str:
    """row_question() as the Show Question popup writes it: every text exactly as sent, or as it would be sent now for
    a row not answered yet or answered with older settings."""
    lines = [f"Row: {found['name'] or ui.EMPTY_SLOT}"]
    if found["state"] is None:
        reason = NEVER_SENT.get(found["source"], "Never sent to the model") if found["name"] else NEVER_OFFERED
        return "\n".join([*lines, reason])
    lines += [f"State: {found['state']}", f"Question ({found['type']}): {found['instructions']}"]
    options = found["options"]
    if options is None:  # a Yes/No question without whens: the server offers these
        lines.append("Options: Yes. / No.")
    else:
        pairs = options.items() if isinstance(options, dict) else enumerate(options)
        lines += ["Options, in the order sent:", *(f"  {key}: {text}" for key, text in pairs)]
    lines.append("Balance Option Order: " + ("on (asked in both option orders)" if found["debias"] else "off"))
    if found["older"]:  # the templates, Balance Option Order or backend changed since it was answered
        lines.append(f"Answered with older settings (templates, Balance Option Order or backend; ask key "
                     f"{found['ask'][:12]}): this is what would be sent now. Ask Again re-asks it")
    elif found["source"] == "error":  # its cell failed, and no pass sends it again
        lines.append("Asking it failed (see the status): this is what was sent. Ask Again asks it again")
    else:
        lines.append(f"Answered with ask key {found['ask'][:12]}" if found["ask"] else
                     "Not answered yet: this is what would be sent")
    return "\n".join(lines)


class OPENACT_OT_show_question(bpy.types.Operator):
    bl_idname = "openact.show_question"
    bl_label = "Show Question"
    bl_description = "Show exactly what this row asks the model: the state, the question and its options. Asks nothing"
    bl_options = {'INTERNAL'}

    uid: bpy.props.StringProperty(options={'HIDDEN'})
    context_index: bpy.props.IntProperty(options={'HIDDEN'})
    row: bpy.props.IntProperty(options={'HIDDEN'})
    copy: bpy.props.BoolProperty(options={'HIDDEN', 'SKIP_SAVE'})
    text: bpy.props.StringProperty(options={'HIDDEN', 'SKIP_SAVE'})

    def question(self):
        _tree, node = engine.find_node(self.uid)
        if node is None:
            raise ValueError("The node is gone")
        return question_text(row_question(node, self.context_index, self.row))

    def invoke(self, context, event):
        try:
            self.text = self.question()
        except ValueError as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        return context.window_manager.invoke_popup(self, width=600)

    def draw(self, context):
        col = self.layout.column(align=True)
        for line in self.text.splitlines():
            ui.wrap(col, line, width=96)
        row = self.layout.row()
        row.operator_context = 'EXEC_DEFAULT'
        op = row.operator(self.bl_idname, text="Copy", icon='COPYDOWN')
        op.uid, op.context_index, op.row, op.copy = self.uid, self.context_index, self.row, True

    def execute(self, context):
        try:
            text = self.question()
        except ValueError as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        if self.copy:
            context.window_manager.clipboard = text
        self.report({'INFO'}, "Copied the question" if self.copy else text.replace("\n", " | "))
        return {'FINISHED'}


class OPENACT_OT_table_view(bpy.types.Operator):
    bl_idname = "openact.table_view"
    bl_label = "Show Table"
    bl_description = "Show this context's table, or another page of it"
    bl_options = {'INTERNAL'}  # no undo step: which table shows is UI state

    context_index: bpy.props.IntProperty(min=0, options={'HIDDEN'})
    page: bpy.props.IntProperty(min=0, options={'HIDDEN'})

    def execute(self, context):
        wm = context.window_manager
        wm.openact_context, wm.openact_page = self.context_index, self.page
        return {'FINISHED'}


# ----------------------------------------------------------------------------- the Overrides CSV (§7.6)

def export_uses(node):
    """(ctx, Job) of each context of the node that plans, as the pass plans it, then of each context whose Overrides
    CSV alone can't be applied (missing, with errors or blocked), with its Job without that file, so the export can
    create it, or refuse it with its own error. The first ones are the contexts the sidebar lists, in its order."""
    spec, tree = engine.SPECS[node.bl_idname], node.id_data
    uses, unlocked = [], []
    for ctx in find_uses().get(tree.as_pointer()) or [Ctx(None, None, (), tree)]:
        try:
            uses.append((ctx, engine._plan(spec, node, ctx)))
        except (Hint, Unresolvable):  # perhaps its Overrides CSV alone
            try:
                unlocked.append((ctx, spec.plan(node, ctx)))
            except (Hint, Pending, Unresolvable):
                continue
        except Pending:
            continue
    return uses + unlocked


def export_context(node, index=0):
    """(uses, ctx, Job): the node's contexts the export can write (export_uses()), and its context `index` (0: the
    first), whose Overrides CSV it writes. Raises OverridesError."""
    uses = export_uses(node)
    if not uses:
        raise OverridesError("Nothing to export: the node has no answered context")
    if not 0 <= index < len(uses):
        raise OverridesError(f"The node has {len(uses)} context{'s' if len(uses) != 1 else ''}, not {index + 1}")
    return (uses, *uses[index])


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


def _identity(path):
    """csvfmt.identity() of a path the trust rule lets this file read, else None: a blocked path is never touched."""
    return None if trust.blocked(path) else csvfmt.identity(path)


def input_files():
    """csvfmt.identity() -> (node name, input) of each file an OpenAct node reads as other than its Overrides CSV:
    the export never writes into one."""
    files, uses = {}, find_uses()
    for tree, node in engine.iter_nodes():
        inputs = [name for name in engine.SPECS[node.bl_idname].words(node)
                  if name != "Overrides" and lists.find_input(node, f"{name} CSV") is not None]
        for ctx in uses.get(tree.as_pointer()) or [Ctx(None, None, (), tree)]:
            for name in inputs:
                try:
                    source = lists.csv_source(node, name, ctx)
                    found = source and _identity(lists.resolve_path(source.path, node))
                except (Hint, Pending, Unresolvable):
                    continue
                if found:
                    files.setdefault(found, (node.name, name))
    return files


def _writes_to(node, ctx, ident):
    """True when ctx's Overrides CSV is the file with this csvfmt.identity(); ident None: an empty, unlinked socket."""
    try:
        path = overrides_target(node, ctx)[0]
    except (OverridesError, Hint, Pending, Unresolvable):
        return False
    return path is None if ident is None else path is not None and _identity(path) == ident


def _mergeable(doc, keys):
    """Raise OverridesError unless the existing file can be merged into: an Overrides CSV with these key columns, a
    'lock' column and no errors, or an empty file."""
    name = os.path.basename(doc.path)
    errors = [p.message for p in doc.problems if p.severity == csvfmt.ERROR]
    if errors:
        raise OverridesError(f"{name}: {errors[0]}. Fix it, or write another file")
    if not doc.header:  # nothing to keep
        return
    folded = {column.casefold() for column in doc.header}
    if not any(key.casefold() in folded for key in keys):
        raise OverridesError(f"{name} is not an Overrides CSV for this node (no '{keys[0]}' column): choose another "
                             "file")
    # An input file can have the key column (an Items CSV's 'item', a Settings CSV's 'question') although no open node
    # reads it (input_files()); none has 'lock', which an Overrides file needs to lock anything.
    if "lock" not in folded:
        raise OverridesError(f"{name} is not an Overrides CSV (no 'lock' column): choose another file")
    try:
        csvfmt.parse_overrides(doc, keys)
    except csvfmt.CsvError as error:  # a factor's column is missing, or a key is locked twice
        raise OverridesError(f"{name}: {error}. Fix it, or write another file") from None


def merged_rows(spec, answered, doc=None):
    """(header, rows) of the Overrides CSV. answered: (Job, entry or None) of each context that uses the file, in
    context order; doc: the file as it is, or None. One row per distinct key of their lockable units, in table order
    and with its own context's answer, then every other row of the file as it was. A unit takes the file's row with
    its key, a locked one first: kept verbatim when locked, else with its user columns. Raises OverridesError for a
    file it can't merge into (_mergeable())."""
    keys = spec.override_keys(answered[0][0])
    header, old = [*keys, *csvfmt.OVERRIDE_COLUMNS], []
    if doc is not None:
        _mergeable(doc, keys)
        known = {column.casefold() for column in header}
        header += [column for column in doc.header if column.casefold() not in known]
        old = [{column.casefold(): cell for column, cell in zip(doc.header, row)} for row in doc.rows]

    def locked(cells):
        return cells.get("lock", "").casefold() in csvfmt.LOCKED

    def line(cells):
        return [cells.get(column.casefold(), "") for column in header]

    taken = {}  # key -> the file's row its unit takes: the locked one, as parse_overrides() reads it, else the first
    for i, cells in enumerate(old):
        key = csvfmt.lock_key(cells.get(k.casefold(), "") for k in keys)
        if key not in taken or (locked(cells) and not locked(old[taken[key]])):
            taken[key] = i
    rows, seen, left = [], set(), set(range(len(old)))
    for job, entry in answered:
        table = table_of(job, entry)
        for unit, indices in spec.override_units(job):
            key = csvfmt.lock_key(unit)
            if key in seen:  # a material in two slots, a question in two contexts: one row locks every use
                continue
            seen.add(key)
            at = taken.get(key)
            left.discard(at)
            cells = dict(old[at]) if at is not None else {}
            if not locked(cells):
                answer, c, source = spec.export_unit(job, indices, table)
                cells.update({k.casefold(): cell for k, cell in zip(keys, unit)})
                cells.update(answer=answer, lock="0", model_answer=answer, confidence="" if c is None else f"{c:.2f}",
                             unsure="" if c is None else str(int(c < UNSURE_BELOW)), source=source)
            rows.append(line(cells))
    return header, rows + [line(old[i]) for i in sorted(left)]


def write_csv(path, header, rows, delimiter=",", bom=False):
    """Write <path>.tmp, then replace path with it, so a reader never sees half a file and a failed write leaves path
    as it was. bom: start with a UTF-8 BOM, as the file did."""
    if os.path.exists(path) and not os.access(path, os.W_OK):
        raise OverridesError(f"Can't write {path}: the file is read-only")
    temporary, created = path + ".tmp", False
    try:
        if os.path.lexists(temporary):  # left by an export that stopped half-way; a link is removed, never followed
            os.remove(temporary)
        with open(temporary, "x", encoding="utf-8-sig" if bom else "utf-8", newline="") as handle:
            created = True
            writer = csv.writer(handle, delimiter=delimiter, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)
    except (OSError, csv.Error, TypeError) as error:  # TypeError: an Import CSV Delimiter that isn't one character
        if created:
            with contextlib.suppress(OSError):
                os.remove(temporary)
        raise OverridesError(f"Can't write {path}: {getattr(error, 'strerror', None) or error}") from None
    try:
        os.replace(temporary, path)
    except OSError as error:
        with contextlib.suppress(OSError):
            os.remove(temporary)
        if isinstance(error, PermissionError):  # not read-only (checked above): open in another program
            raise OverridesError(PERMISSION) from None
        raise OverridesError(f"Can't write {path}: {error.strerror or error}") from None


def write_overrides(node, index=0, path=None):
    """Write the node's answers to the Overrides CSV of its live context `index` (§7.6), with the rows of every context
    that uses that file; returns the path written. path: the file chosen for an empty, unlinked Overrides socket,
    which then takes it. Raises OverridesError, having written nothing."""
    spec = engine.SPECS[node.bl_idname]
    try:
        uses, ctx, job = export_context(node, index)
        target, delimiter = overrides_target(node, ctx)
        fill = target is None
        if fill:
            if not path:
                raise OverridesError("Choose a file for the Overrides CSV")
            target = lists.resolve_path(path, node)
            if not target.lower().endswith(".csv"):
                target += ".csv"
    except (Hint, Pending, Unresolvable) as error:
        raise OverridesError(str(error)) from None
    blocked = trust.blocked(target, writing=True)
    if blocked:
        raise OverridesError(blocked)
    ident = csvfmt.identity(target)
    used = input_files().get(ident)
    if used:
        raise OverridesError(f"{os.path.basename(target)} is the {used[1]} CSV of '{used[0]}': choose another file")
    keys, entries = spec.override_keys(job), store.load(node).entries
    answered = [(j, spec.resolve(j, entries, engine.LAST_SHOWN.get((node.uid, c.key()), ""))[0])
                for c, j in uses if spec.override_keys(j) == keys and _writes_to(node, c, None if fill else ident)]
    doc = csvfmt.read_csv(target, delimiter) if os.path.exists(target) else None
    header, rows = merged_rows(spec, answered, doc)
    write_csv(target, header, rows, doc.delimiter if doc else delimiter or ",", doc is not None and doc.bom)
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
    bl_description = ("Write this node's answers to its Overrides CSV, to lock rows in (lock = 1). The file's other "
                      "rows, the rows you locked and your own columns are kept")
    bl_options = {'REGISTER', 'UNDO'}

    uid: bpy.props.StringProperty(options={'HIDDEN'})
    context_index: bpy.props.IntProperty(name="Context", default=0, min=0, options={'HIDDEN'},
                                         description="Which of the node's live contexts to export (0 = the first)")
    filepath: bpy.props.StringProperty(subtype='FILE_PATH', options={'SKIP_SAVE'})
    filter_glob: bpy.props.StringProperty(default="*.csv", options={'HIDDEN'})
    check_existing: bpy.props.BoolProperty(default=True, options={'HIDDEN'})  # the file dialog flags an existing file

    def invoke(self, context, event):
        _tree, node = engine.find_node(self.uid)
        if node is None:
            return {'CANCELLED'}
        try:
            _uses, ctx, _job = export_context(node, self.context_index)
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


# ---------------------------------------------------------------------------- the upgrade from v0.1.0 (§12.3)

class OPENACT_OT_upgrade_legacy(bpy.types.Operator):
    bl_idname = "openact.upgrade_legacy"
    bl_label = "Upgrade"
    bl_description = ("Replace this file's old Jev nodes with OpenAct nodes, keeping their words, links and answers. "
                      "One undo step")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        upgraded, trees, problems = migrate.upgrade_all()
        if not (upgraded or problems):
            self.report({'INFO'}, "This file has no old Jev nodes")
            return {'CANCELLED'}
        self.report({'WARNING'} if problems else {'INFO'}, migrate.report(upgraded, trees, problems))
        return {'FINISHED'} if upgraded else {'CANCELLED'}


class OPENACT_OT_legacy_report(bpy.types.Operator):
    """Say that the file just opened has old Jev nodes, and where to upgrade them"""

    bl_idname = "openact.legacy_report"
    bl_label = "Old Jev Nodes"
    bl_options = {'INTERNAL'}

    def execute(self, context):
        text = migrate.notice()
        if not text:
            return {'CANCELLED'}
        self.report({'INFO'}, text)
        return {'FINISHED'}


classes = (OPENACT_OT_install_runtime, OPENACT_OT_locate_runtime, OPENACT_OT_server_start, OPENACT_OT_server_stop,
           OPENACT_OT_reask, OPENACT_OT_open_log, OPENACT_OT_open_runtime_folder, OPENACT_OT_allow_file,
           OPENACT_OT_trust_folder, OPENACT_OT_open_file, OPENACT_OT_show_question, OPENACT_OT_table_view,
           OPENACT_OT_write_overrides, OPENACT_OT_upgrade_legacy, OPENACT_OT_legacy_report)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
