"""The sidebar (Geometry Node editor > N > OpenAct) and the preferences panel (§17.2): banners, the Model box, the
active node's status, contexts, table and pins, and the files it reads. Inspection only: every change goes through a
file, a socket or an operator.

status_rows(), table_rows(), table_page(), pin_rows() and banners() are plain data, so the tests check them without
drawing.
Drawing plans and writes nothing: Blender refuses changes to nodes and data while it draws.
"""

from __future__ import annotations

import functools
import os
import time

import bpy

from . import csvfmt, engine, evaluator, inner, lists, migrate, pipeline, server, trust
from . import prefs as P
from .specs.base import table_of, unsure_below
from .watch import WATCH

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
PAGE_ROWS = 50  # table rows per page
FILE_PROBLEMS = 50  # problems listed per file
ROW_ICONS = {"lock": 'LOCKED', "code": 'CHECKBOX_HLT', "error": 'CANCEL'}
PROBLEM_ICONS = {csvfmt.ERROR: 'CANCEL', csvfmt.WARNING: 'ERROR', csvfmt.INFO: 'INFO'}
SERVER_BANNERS = ("install", "runtime", "port")  # shown in the Model box, above its buttons
IMPORT_CSV_NOTE = ("Import CSV '{}' reads the same file; Blender won't re-read it after edits in this session. Use "
                   "this node's payload outputs, or restart Blender")
LEGACY = []  # callables giving the old Jev nodes banner as (text, operator idname), or None (§12.3)
EMPTY_SLOT = "(empty slot)"  # a row without a name: an empty material slot, never offered


def wrap(layout, text, width=46, icon='NONE'):
    col = layout.column(align=True)
    line, first = "", True
    for word in text.split():
        if line and len(line) + len(word) + 1 > width:
            col.label(text=line, icon=icon if first else 'BLANK1')
            line, first = "", False
        line = f"{line} {word}".strip()
    if line:
        col.label(text=line, icon=icon if first else 'BLANK1')


# ---------------------------------------------------------------------------------------------- data

def status_rows(status) -> list[tuple[str, str]]:
    """(icon, text) lines of a node's status record (§17.1): the status line (its ETA included), the row counts, then
    every other problem, errors first."""
    if status is None:
        return [('SORTTIME', "Waiting...")]
    rows = [(status.icon, status.text)]
    if status.rows:
        counts = ", ".join(f"{n} {name}" for name, n in (
            ("answered", status.answered), ("pending", status.pending), ("stale", status.stale),
            ("unsure", status.unsure), ("locked", status.locked), ("by code", status.code),
            ("in error", status.errors)) if n)
        total = f"{status.rows} row{'s' if status.rows != 1 else ''}"
        rows.append(('BLANK1', f"{total}: {counts}" if counts else total))
    return rows + [(PROBLEM_ICONS.get(problem.severity, 'INFO'), problem.message) for problem in status.problems
                   if problem.message != status.text]  # an error's line is its first problem


def table_rows(spec, job, entry=None) -> list[tuple]:
    """(name, answer, confidence or None, source) of each row of a context's table: from the entry it shows once it is
    answered, else from its plan, whose pending rows have no answer yet. An empty material slot has neither."""
    rows = []
    for i, (source, values) in enumerate(table_of(job, entry)):
        name = job.rows[i].name
        if not name:  # never offered, so nothing was decided
            rows.append((EMPTY_SLOT, "", None, source))
            continue
        if not values or source == "error":
            answer = ""
        elif "prob" in values:  # Pick: how well each item fits
            answer = f"{values['prob'] * 100:.0f}%"
        else:
            answer = spec.answer_text(job, i, values)  # as the Overrides CSV writes it
        rows.append((name, answer, values.get("c"), source))
    return rows


def table_page(rows, page=0):
    """One page of a table view (§17.2): (lines, page, pages), page clamped to the table. rows: table_rows(); lines:
    (name, answer, confidence, icon, active) of up to PAGE_ROWS rows. The icon tells a lock, a code row, a row in error
    and an unsure model row (confidence below unsure_below()); stale and pending rows are not active (greyed out)."""
    pages = max(1, -(-len(rows) // PAGE_ROWS))
    page = min(max(page, 0), pages - 1)
    lines, below = [], unsure_below()
    for name, answer, confidence, source in rows[page * PAGE_ROWS:(page + 1) * PAGE_ROWS]:
        unsure = source == "model" and confidence is not None and confidence < below
        icon = ROW_ICONS.get(source, 'QUESTION' if unsure else 'BLANK1')
        lines.append((name, answer, confidence, icon, source not in ("stale", "pending")))
    return lines, page, pages


def pin_rows(pins) -> list[tuple[str, str]]:
    """(words key, text) of each pinned entry (NodeStatus.pins, in pin order): the frames its words occur at, and the
    words that tell it from the node's other pins, a CSV path by its file name."""
    varying = [j for j in range(len(pins[0].gn) if pins else 0) if len({entry.gn[j] for entry in pins}) > 1]
    rows = []
    for entry in pins:
        words = [os.path.basename(word) if lists.is_csv_path(word) else word
                 for j, word in enumerate(entry.gn) if word and (j in varying or not varying)]
        rows.append((entry.words_key, f"Frames {engine.frames_text(entry.frames)}: {' · '.join(words)}"))
    return rows


def banners(state, message, probe="", blocked=0, addons=(), notes=(), legacy=(), over=()) -> list[tuple]:
    """The sidebar's banners (§17.2), most urgent first: (kind, icon, text, operator idnames of its buttons). Plain
    data: live_banners() passes the live state. state, message: server.status(); probe: the runtime probe's failure;
    blocked: how many reads of this untrusted file were blocked; addons: the enabled add-on modules; notes: Import CSV
    notes; legacy: the old Jev nodes banners as (text, operator idname); over: the over-cap errors."""
    found = []
    if state == "not_installed":
        found.append(("install", 'ERROR', "Model not installed: Install Model downloads it once (about 12 GB); Locate "
                      "Existing Runtime uses one already on this computer",
                      ("openact.install_runtime", "openact.locate_runtime")))
    elif probe:
        found.append(("runtime", 'ERROR', probe, ("openact.locate_runtime",)))
    elif message == server.CHECKING:
        found.append(("runtime", 'SORTTIME', message, ("openact.locate_runtime",)))
    elif state == "error" and (message == server.LEGACY_SERVER_MESSAGE or message.startswith("Port ")):
        found.append(("port", 'ERROR', message, ()))
    if blocked:
        found.append(("trust", 'ERROR', f"This file is not trusted, so {blocked} read{'s' if blocked > 1 else ''} "
                      f"outside its folder {'were' if blocked > 1 else 'was'} blocked",
                      ("openact.allow_file", "openact.trust_folder")))
    if any(map(P.is_jev, addons)):
        found.append(("jev_nodes", 'ERROR', "Disable Jev Nodes: OpenAct replaces it", ()))
    found += [("legacy", 'INFO', text, (idname,)) for text, idname in legacy]
    found += [("over_cap", 'CANCEL', text, ()) for text in over]
    found += [("import_csv", 'INFO', text, ()) for text in notes]
    return found


def runtime_line(prefs=None) -> str:
    """§12.2's line while the runtime is the Jev Nodes one, used in place, else "": its folder, and the base Python its
    venv uses, which can live outside it. No size: walking 13 GB of files while drawing is too slow."""
    root = P.runtime_root(prefs)
    legacy = os.path.normpath(os.path.join(os.path.dirname(P.default_runtime_root()), "JevNodes"))
    if os.path.normcase(root) != os.path.normcase(legacy):
        return ""
    home = P.venv_home(root)
    return f"Using the Jev Nodes runtime at {root}" + (f"; Python from {home}" if home else "")


def import_csv_notes() -> dict:
    """§7.1: csvfmt.identity() -> note, for each file an OpenAct node reads that one of the artist's own Import CSV
    nodes reads too. Blender keeps its first read of a path for the session, so after an edit that node shows old
    rows. Worked out once per pass, when drawn."""
    return _import_csv_notes(engine.passes)


@functools.lru_cache(maxsize=1)  # keyed by the pass
def _import_csv_notes(_pass):
    notes, uses = {}, None
    if not WATCH.files:
        return notes
    for tree in bpy.data.node_groups:
        if tree.bl_idname != 'GeometryNodeTree' or tree.name.startswith(inner.INNER_PREFIX):
            continue
        for node in tree.nodes:
            if node.bl_idname != evaluator.IMPORT_CSV or node.mute or all(
                    link.is_muted or link.to_node.bl_idname in engine.OPENACT_TYPES
                    for out in node.outputs for link in out.links):
                continue  # unused, or only plugged into OpenAct nodes, which read its Path and never evaluate it
            uses = evaluator.find_uses() if uses is None else uses
            path = next(s for s in node.inputs if s.identifier == "Path")
            for ctx in uses.get(tree.as_pointer()) or [evaluator.Ctx(None, None, (), tree)]:
                try:
                    text = evaluator.as_text(evaluator.eval_input(path, ctx))
                    ident = csvfmt.identity(lists.resolve_path(text, node))
                except Exception:  # noqa: BLE001 - a Path that can't be read here gets no note
                    continue
                if ident in WATCH.files:
                    notes.setdefault(ident, IMPORT_CSV_NOTE.format(node.name))
    return notes


def live_banners(context) -> list[tuple]:
    """banners() for this session: the server and the runtime probe, this file's trust, the add-ons, the nodes."""
    state, message = server.status()
    records = list(engine.STATUS.values())
    # from BLOCKED on: a node that reads a blocked node's answer repeats its read behind a breadcrumb
    blocked = {problem.message[problem.message.index(trust.BLOCKED):] for status in records
               for problem in status.problems if trust.BLOCKED in problem.message}
    if blocked and trust.is_trusted():  # allowed since the pass; asked only then, as it resolves paths
        blocked = set()
    over = []
    if any(status.contexts > inner.MATERIALISED_MAX for status in records):
        over = [f"{node.name}: {engine.OVER_CAP.format(status.contexts, inner.MATERIALISED_MAX)}"
                for _tree, node in engine.iter_nodes()
                if (status := engine.STATUS.get(node.uid)) and status.contexts > inner.MATERIALISED_MAX]
    notes = [f"{os.path.basename(WATCH.files[ident].path)}: {note}" for ident, note in import_csv_notes().items()
             if ident in WATCH.files]
    legacy = [found for hook in LEGACY if (found := hook()) is not None]
    return banners(state, message, server.STATE.probes.get(P.runtime_root()) or "", len(blocked),
                   context.preferences.addons.keys(), notes, legacy, over)


# ------------------------------------------------------------------------------------------ drawing

def draw_banner(layout, banner):
    _kind, icon, text, actions = banner
    col = layout.column(align=True)
    col.alert = icon in ('ERROR', 'CANCEL')
    wrap(col, text, icon=icon)
    if actions:
        row = layout.row(align=True)
        for idname in actions:
            row.operator(idname)


def draw_server(layout, banner=None):
    """The Model box: the server's status, or the banner about it, its details, and Install / Start / Stop."""
    state, message = server.status()
    box = layout.box()
    if banner is not None:
        draw_banner(box, banner)
    else:
        box.label(text=_TITLES.get(state, state), icon=_ICONS.get(state, 'QUESTION'))
        if message:
            wrap(box, message, icon='BLANK1')
    if server.STATE.install_warning and server.STATE.install_warning not in message:  # "Runtime installed. <it>"
        wrap(box, server.STATE.install_warning, icon='ERROR')
    shared = state in ("ready", "starting") and server.attached()
    if shared:
        box.label(text=server.SHARED, icon='LINKED')
    line = runtime_line()
    if line:
        row = box.row()
        wrap(row.column(), line, icon='FILE_FOLDER')
        row.operator("openact.open_runtime_folder", text="", icon='FILEBROWSER')
    health = server.STATE.health
    if state == "ready" and health:
        col = box.column(align=True)
        wrap(col, " · ".join(str(health[key]) for key in ("backend", "device", "model_id") if health.get(key)),
             icon='BLANK1')
        if health.get("load_seconds") is not None:
            col.label(text=f"Loaded in {health['load_seconds']:.1f} s", icon='BLANK1')
        per_class, flight = pipeline.queue_counts()
        col.label(text="Queue: {}/{}/{} chunks (P0/P1/P2)".format(*per_class), icon='BLANK1')
        col.label(text=f"In flight: {flight} cells", icon='BLANK1')
        throughput = [f"{health['ms_per_row']:.1f} ms per row"] if health.get("ms_per_row") is not None else []
        if health.get("questions"):
            throughput.append(f"cache hits {health.get('cache_hits', 0) / health['questions']:.0%}")
        if throughput:
            col.label(text=" · ".join(throughput), icon='BLANK1')
        note = engine.slow_note()
        if note:
            wrap(col, note, icon='ERROR')
    row = box.row(align=True)
    if state == "installing":
        row.operator("openact.open_log", icon='TEXT')
        return
    if banner is None or banner[0] != "install":  # that banner has the Install button
        installed = state not in ("not_installed", "error") or P.runtime_installed(P.runtime_root())
        if not installed:
            row.operator("openact.install_runtime", icon='IMPORT')
        elif shared:  # another Blender's server: Stop only stops using it (§10.5)
            row.operator("openact.server_stop", text="Detach", icon='UNLINKED')
        elif state in ("ready", "starting"):
            row.operator("openact.server_stop", icon='PAUSE')
        else:
            row.operator("openact.server_start", icon='PLAY')
    row.operator("openact.open_log", text="", icon='TEXT')
    row.operator("openact.copy_diagnostics", text="", icon='COPYDOWN')


def draw_preferences(prefs, context, layout):
    state, message = server.status()
    found = banners(state, message, server.STATE.probes.get(P.runtime_root(prefs)) or "")
    draw_server(layout, found[0] if found else None)
    col = layout.column()
    col.use_property_split = True
    col.prop(prefs, "auto_start")
    col.prop(prefs, "device")
    sub = col.column()
    sub.active = prefs.device != 'cpu'
    sub.prop(prefs, "gpu")
    col.prop(prefs, "debias")
    col.prop(prefs, "unsure_below")
    col.prop(prefs, "port")
    col.prop(prefs, "runtime_dir")
    root = P.runtime_root(prefs)
    row = col.row()
    row.label(text=root, icon='FILE_FOLDER')
    row.operator("openact.open_runtime_folder", text="", icon='FILEBROWSER')
    folder = os.path.abspath(server.state_dir(root))  # where the token file is (OPENACT_STATE_DIR, else the runtime)
    if not trust.inside(os.path.normcase(folder), os.path.normcase(os.path.abspath(os.path.expanduser("~")))):
        wrap(col, f"The folder that holds the model server's token, {folder}, is outside your user folder: its "
                  "permissions decide who else can read the token", icon='ERROR')
    col.prop(prefs, "log_requests")
    note = layout.box().column(align=True)
    note.label(text=f"Model: {P.MODEL_LABEL} ({P.MODEL_REVISION[:10]}), runs on this computer", icon='INFO')
    note.label(text="SemIf (formerly OpenJev), MIT - github.com/TheoLeeCJ/SemIf-OpenJev", icon='BLANK1')
    note.label(text="Qwen3.5-4B by Alibaba Qwen, Apache-2.0. Not affiliated with TypeSafe or Jev", icon='BLANK1')


def draw_table(layout, context, node, uses, index):
    """The table of context `index` (§17.2), PAGE_ROWS rows a page: name, answer, confidence bar, source icon, and
    Show Question."""
    label, job, entry = uses[index]
    lines, page, pages = table_page(table_rows(engine.SPECS[node.bl_idname], job, entry),
                                    context.window_manager.openact_page)
    box = layout.box()
    head = box.row(align=True)
    head.label(text=f"Table: {label}" if len(uses) > 1 else "Table", icon='SPREADSHEET')
    if pages > 1:
        for target, icon in ((page - 1, 'TRIA_LEFT'), (page + 1, 'TRIA_RIGHT')):
            button = head.row(align=True)
            button.enabled = 0 <= target < pages
            op = button.operator("openact.table_view", text="", icon=icon)
            op.context_index, op.page = index, min(max(target, 0), pages - 1)
            if icon == 'TRIA_LEFT':
                head.label(text=f"{page + 1} / {pages}")
    for i, (name, answer, confidence, icon, active) in enumerate(lines):
        row = box.row(align=True)
        row.active = active
        split = row.split(factor=0.45, align=True)
        split.label(text=name, icon=icon)
        right = split.row(align=True)
        right.label(text=answer)
        if confidence is not None:
            right.progress(factor=min(max(confidence, 0.0), 1.0), type='BAR', text=f"{confidence:.2f}")
        op = right.operator("openact.show_question", text="", icon='VIEWZOOM')
        op.uid, op.context_index, op.row = node.uid, index, page * PAGE_ROWS + i


def draw_pins(layout, node):
    """The node's pinned entries (§5.5), each with Unpin, and Unpin All."""
    status = engine.STATUS.get(node.uid)
    rows = pin_rows(status.pins) if status is not None else []
    if not rows:
        return
    box = layout.box()
    head = box.row(align=True)
    head.label(text=f"Pinned for playback: {len(rows)}", icon='PINNED')
    head.operator("openact.unpin", text="Unpin All", icon='UNPINNED').uid = node.uid
    for key, text in rows:
        row = box.row(align=True)
        row.label(text=text, icon='BLANK1')
        op = row.operator("openact.unpin", text="", icon='UNPINNED')
        op.uid, op.words_key = node.uid, key


def draw_files(layout, node):
    """The files the node reads (§7.10): New CSV from Template where one of the node's inputs has a starter file, then
    each file's path, rows, text and payload columns, when read, the Import CSV note, and up to FILE_PROBLEMS problems,
    with Open File."""
    from .ops import starter_files  # ops imports this module

    if starter_files(node):
        row = layout.row()
        row.enabled = node.id_data.library is None  # a linked node's inputs are read-only
        row.operator("openact.new_csv_from_template", icon='FILE_NEW').uid = node.uid
    files = sorted(((ident, watched) for ident, watched in WATCH.files.items() if node.uid in watched.owners),
                   key=lambda item: item[1].path.casefold())
    if not files:
        layout.label(text="This node reads no CSV file", icon='INFO')
        return
    notes = import_csv_notes()
    for ident, watched in files:
        box = layout.box()
        head = box.row(align=True)
        head.label(text=os.path.basename(watched.path), icon='CANCEL' if watched.problem else 'FILE')
        head.operator("openact.open_file", text="", icon='FILEBROWSER').filepath = watched.path
        col = box.column(align=True)
        wrap(col, watched.path, icon='BLANK1')
        doc = watched.doc
        problems = doc.problems if doc is not None else ()
        if watched.problem and all(problem.message != watched.problem for problem in problems):  # e.g. missing
            wrap(col, watched.problem, icon='CANCEL')
        if doc is not None and doc.stamp != (0, 0):
            when = time.strftime("%H:%M:%S", time.localtime(watched.read))
            col.label(text=f"{len(doc.rows)} data row{'s' if len(doc.rows) != 1 else ''} · read {when}", icon='BLANK1')
            if doc.text_cols:
                wrap(col, "Text columns: " + ", ".join(doc.text_cols), icon='BLANK1')
            if doc.num_cols:
                wrap(col, "Payload columns: " + ", ".join(f"{name} ({kind})" for name, kind in doc.num_cols.items()),
                     icon='BLANK1')
        if ident in notes:
            wrap(col, notes[ident], icon='INFO')
        for problem in problems[:FILE_PROBLEMS]:
            text = problem.message
            if problem.line and not text.startswith("Line "):
                text = f"Line {problem.line}: {text}"
            wrap(col, text, icon=PROBLEM_ICONS.get(problem.severity, 'INFO'))
        if len(problems) > FILE_PROBLEMS:
            col.label(text=f"... {len(problems) - FILE_PROBLEMS} more problems", icon='BLANK1')


# ------------------------------------------------------------------------------------------- panels

def _active(context):
    """The active node of the edited tree when it is an OpenAct node."""
    tree = getattr(context.space_data, "edit_tree", None)
    node = tree.nodes.active if tree is not None else None
    return node if node is not None and node.bl_idname in engine.OPENACT_TYPES else None


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
        found = live_banners(context)
        for banner in found:
            if banner[0] not in SERVER_BANNERS:
                draw_banner(layout.box(), banner)
        draw_server(layout, next((banner for banner in found if banner[0] in SERVER_BANNERS), None))
        if engine.STATUS:  # the file has OpenAct nodes
            col = layout.column(align=True)
            col.operator("openact.prepare_playback", icon='PLAY')
            waiting, held = engine.pinning(), engine.held()
            if waiting:
                sets = f"{waiting} word set{'s' if waiting != 1 else ''}"
                wrap(col, f"{sets} answered: {held}" if held else f"Pinning {sets} once the model has answered them",
                     icon='SORTTIME')
                col.operator("openact.cancel_playback", icon='CANCEL')
            if engine.PLAYBACK_NOTE:
                wrap(col, engine.PLAYBACK_NOTE, icon='INFO')
            for flag in dict.fromkeys(filter(None, (engine.rebake(uid, status.pins)
                                                    for uid, status in engine.STATUS.items()))):
                wrap(col, flag, icon='ERROR')
            pinned = sum(len(status.pins) for status in engine.STATUS.values())  # as the entries hold them
            if pinned:
                col.label(text=f"{pinned} word set{'s' if pinned != 1 else ''} pinned for playback", icon='PINNED')
        if _active(context) is None:
            col = layout.column(align=True)
            col.label(text="Add nodes with Shift A > OpenAct", icon='INFO')
            col.label(text="Whole situation: Ask, Pick, Dials", icon='BLANK1')
            col.label(text="Each item: Match, Classify", icon='BLANK1')
            col.label(text="Each situation: Rules", icon='BLANK1')


class NODE_PT_openact_node(bpy.types.Panel):
    bl_space_type = 'NODE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "OpenAct"
    bl_label = "Active Node"

    @classmethod
    def poll(cls, context):
        return NODE_PT_openact.poll(context) and _active(context) is not None

    def draw(self, context):
        node = _active(context)
        self.layout.label(text=node.name, icon='NODE')
        node.draw_buttons_ext(context, self.layout)
        draw_pins(self.layout, node)


class NODE_PT_openact_files(bpy.types.Panel):
    bl_space_type = 'NODE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "OpenAct"
    bl_label = "Files"

    @classmethod
    def poll(cls, context):
        return NODE_PT_openact.poll(context) and _active(context) is not None

    def draw(self, context):
        draw_files(self.layout, _active(context))


classes = (NODE_PT_openact, NODE_PT_openact_node, NODE_PT_openact_files)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    # UI state, not file data: which context's table the sidebar shows, and its page
    bpy.types.WindowManager.openact_context = bpy.props.IntProperty(name="Context", min=0)
    bpy.types.WindowManager.openact_page = bpy.props.IntProperty(name="Page", min=0)
    LEGACY.append(migrate.banner)


def unregister():
    LEGACY.remove(migrate.banner)
    del bpy.types.WindowManager.openact_page
    del bpy.types.WindowManager.openact_context
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
