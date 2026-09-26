"""Keeps the OpenAct nodes answered.

Geometry Nodes cannot run Python while it evaluates (Blender has no script node), so an OpenAct
node is answered outside evaluation:

1. After a change, each OpenAct node's words are read by the static evaluator (evaluator.py) and
   planned into a Job per context (specs/, pipeline.py). A Job is planned again only when the
   fingerprint of what it was planned from changes.
2. Its rows are answered by the validity rule: the node's stored answers while the words mean the
   same, then this session's answers, then the local model, asked in the background (server.py).
3. The answers are written into the node's hidden node group as a small lookup table keyed by the
   exact input strings. Geometry Nodes picks the matching answer by itself, so one node group
   can be used on several objects with different prompts, answers switch instantly back and
   forth, and they are saved with the file (Blender 4.5+ even evaluates them without the add-on).
"""

from __future__ import annotations

import time
import traceback
import uuid

import bpy
from bpy.app.handlers import persistent

from . import evaluator, inner, pipeline, server, store
from . import prefs as P
from . import tables
from .evaluator import Ctx, Hint, Pending, Unresolvable, find_uses, is_field
from .inner import INNER_PREFIX
from .specs import SPECS
from .specs.ask import AskSpec
from .specs.base import orphaned
from .specs.pick import PickSpec
from .watch import WATCH

ASK_ID, PICK_ID = AskSpec.idname, PickSpec.idname
OPENACT_TYPES = tuple(SPECS)
DEBOUNCE = 0.25
SAFETY_PASS = 2.0  # edits to trees no modifier uses send no depsgraph update; it re-plans only changed fingerprints
CLASH = ("'{}' and '{}' share every word but not their list (a collection, object or CSV): only one answer can "
         "show. Give each a different Key (Ctrl H shows it; expose it on the modifier)")

STATUS = {}  # node uid -> (icon, text)
PLANS = {}  # (node uid, ctx key) -> (fingerprint, Job, or the (Hint or Unresolvable, message) planning gave)
LAST_SHOWN = {}  # (node uid, ctx key) -> words key of the entry that context showed last
LIVE = {}  # node uid -> words keys of the entries its contexts showed at its last pass
DIRTY = set()  # uids of the nodes the next pass visits, when not all of them are dirty

_dirty = True  # a pass is due
_dirty_all = True  # ... and it visits every node
_last_change = 0.0
_last_refresh = 0.0
_last_pass = 0.0
_has_nodes = False
_group_count = -1  # number of node groups at the last pass that found no OpenAct nodes
_plans_made = 0  # a pass that plans changes what nodes reading another's answers as words may read


# ----------------------------------------------------------------------------- plans


def _openact_output(node, out, ctx, depth):
    """Evaluator handler: the answer an OpenAct node shows for the words it gets in this context."""
    if is_field(out):
        raise Unresolvable(f"'{node.name}' output '{out.name}' is a field; words must be single values")
    spec = SPECS[node.bl_idname]
    try:
        job = _plan(spec, node, ctx)
    except Hint:
        return spec.fallback(out.name)
    entry, _missing, _failed = spec.resolve(job, store.load(node).entries, LAST_SHOWN.get((node.uid, ctx.key()), ""))
    if entry is None:
        raise Pending()
    try:
        return spec.output(entry, out.name)
    except KeyError:
        raise Unresolvable(f"'{node.name}' output '{out.name}' can't be a word") from None


evaluator.HANDLERS.update(dict.fromkeys(OPENACT_TYPES, _openact_output))


def _plan(spec, node, ctx):
    """The node's Job in ctx, planned again only when its fingerprint changed (§5.3). Raises Hint, Pending or
    Unresolvable."""
    global _plans_made
    key = (node.uid, ctx.key())
    fingerprint = spec.fingerprint(node, ctx)
    cached = PLANS.get(key)
    if cached is None or cached[0] != fingerprint:
        try:
            planned = spec.plan(node, ctx)
        except (Hint, Unresolvable) as error:  # kept as a message: the evaluator adds its hops to a raised error
            planned = (type(error), str(error))
        cached = PLANS[key] = (fingerprint, planned)
        _plans_made += 1
    if isinstance(cached[1], tuple):
        kind, message = cached[1]
        raise kind(message)
    return cached[1]


def _jobs(node):
    """(ctx, Job) for each use of the node that has a question."""
    tree = node.id_data
    spec = SPECS[node.bl_idname]
    jobs = []
    for ctx in find_uses().get(tree.as_pointer()) or [Ctx(None, None, (), tree)]:
        try:
            jobs.append((ctx, _plan(spec, node, ctx)))
        except (Hint, Pending, Unresolvable):
            continue
    return jobs


# ------------------------------------------------------------------------ inner trees


def new_uid():
    return uuid.uuid4().hex[:12]


def ensure_owned(nodes):
    """Give every OpenAct node a unique uid and its own hidden group.

    Copies (duplicated nodes or trees, appended data) arrive sharing the original's group; the
    rightful owner is the node whose uid matches the group's name, everyone else gets a copy.
    Linked data is read-only: linked nodes are left alone and keep their uids, so a local node that
    shares one (appended from the library that is also linked) gets a new uid. A local node whose
    group is linked (a tree made local on its own) gets a local copy."""
    seen = {node.uid for tree, node in nodes if tree.library is not None}
    nodes = [(tree, node) for tree, node in nodes if tree.library is None]
    owners = {}
    for _tree, node in nodes:
        group = node.node_tree
        if node.uid and group is not None and group.name == inner.inner_name(node):
            owners.setdefault(group.as_pointer(), node)
    for _tree, node in nodes:
        fresh = not node.uid or node.uid in seen
        if fresh:
            node.uid = new_uid()
        seen.add(node.uid)
        group = node.node_tree
        if group is None:
            inner.create_inner(node)
        elif group.library is not None or owners.setdefault(group.as_pointer(), node) is not node:
            own = group.copy()
            own.name = inner.inner_name(node)
            node.node_tree = own
            tables.copy_for(node)  # the copied group still reads the original's table object
        elif fresh:  # frees the old uid's group name. Only once: a taken name would be retried every pass
            group.name = inner.inner_name(node)


def iter_nodes():
    for tree in bpy.data.node_groups:
        if tree.bl_idname != 'GeometryNodeTree' or tree.name.startswith(INNER_PREFIX):
            continue
        for node in tree.nodes:
            if node.bl_idname in OPENACT_TYPES:
                yield tree, node


def find_node(uid):
    for tree, node in iter_nodes():
        if node.uid == uid:
            return tree, node
    return None, None


# ----------------------------------------------------------------------------- passes


def schedule(immediate=False, node=None):
    """Ask for a pass over every node, or only over `node` when its own sockets or settings changed."""
    global _dirty, _dirty_all, _last_change
    _dirty = True
    if node is not None and node.uid:
        DIRTY.add(node.uid)
    else:
        _dirty_all = True
    _last_change = 0.0 if immediate else time.monotonic()


def run_pass(everything=True):
    """Bring the OpenAct nodes up to date: every node, or with everything=False only the dirty ones. Main thread
    only."""
    global _dirty, _dirty_all, _has_nodes, _last_pass, _group_count
    visit_all = everything or _dirty_all
    dirty = set(DIRTY)
    DIRTY.clear()
    for uid in WATCH.take_dirty():  # a CSV they read changed on disk: plan them again (§7.8)
        for key in [key for key in PLANS if key[0] == uid]:
            del PLANS[key]
        dirty.add(uid)
    _dirty = _dirty_all = False
    _last_pass = time.monotonic()
    nodes = list(iter_nodes())
    _has_nodes = bool(nodes)
    _group_count = -1 if nodes else len(bpy.data.node_groups)
    if not nodes:
        PLANS.clear()
        LAST_SHOWN.clear()
        LIVE.clear()
        WATCH.retain(())
        tables.collect_garbage(nodes)
        return
    ensure_owned(nodes)
    tables.collect_garbage(nodes)  # after ensure_owned(), which gives copies their own table objects
    uses = find_uses()
    plans_made, stored = _plans_made, False
    evaluator.MEMO = {}  # nodes that read the same upstream words in a context share one read
    try:
        for tree, node in nodes:
            if not (visit_all or node.uid in dirty):
                continue
            try:
                stored |= _process(tree, node, uses.get(tree.as_pointer()))
            except Exception as error:
                traceback.print_exc()
                STATUS[node.uid] = ('ERROR', f"Internal error: {error}")
    finally:
        evaluator.MEMO = None  # keyed by pointers: after the pass, nodes and data may be freed or edited
    if stored or _plans_made != plans_made:
        schedule(immediate=True)  # a node that reads another's answers as words may read new ones
    contexts = {node.uid: {ctx.key() for ctx in uses.get(tree.as_pointer()) or [Ctx(None, None, (), tree)]}
                for tree, node in nodes}
    for cache in (PLANS, LAST_SHOWN):  # contexts that disappeared, and nodes that did
        for key in [key for key in cache if key[1] not in contexts.get(key[0], ())]:
            del cache[key]
    for uid in [uid for uid in LIVE if uid not in contexts]:
        del LIVE[uid]
    WATCH.retain(contexts)  # stop watching the files of deleted nodes
    redraw()


def _process(tree, node, contexts):
    """Plan and answer the node's contexts, store new answers and rebuild its group. True if it stored any."""
    if tree.library is not None:  # read-only: the library file holds this node's answers
        STATUS[node.uid] = ('LINKED', f"Linked from {tree.library.name}: answers come from that file")
        return False
    node_store = store.load(node)
    if node_store.schema > store.SCHEMA:  # save() would refuse any answer, and the group holds the newer ones
        STATUS[node.uid] = ('ERROR', store.newer_message(node_store))
        return False
    spec = SPECS[node.bl_idname]
    entries = node_store.entries
    contexts = contexts or [Ctx(None, None, (), tree)]
    jobs, shown, live = {}, [], []  # words -> (Job, ctx), one per distinct words; the entries shown; live gns
    problems, warnings, hints, failed = [], [], [], []
    upstream = waiting = changed = False
    for ctx in contexts:
        key = (node.uid, ctx.key())
        try:
            job = _plan(spec, node, ctx)
        except Pending:
            upstream = True
            continue
        except Hint as hint:
            hints.append(str(hint))
            continue
        except Unresolvable as error:  # e.g. a CSV with errors: its words stay live, so the last good table shows
            problems.append(str(error))
            _live(live, spec, node, ctx)
            continue
        warnings += job.problems
        if job.gn not in live:
            live.append(job.gn)
        other = jobs.setdefault(job.gn, (job, ctx))
        if other[0] is not job:
            if other[0].words_key != job.words_key:
                problems.append(CLASH.format(other[1].label(), ctx.label()))
            continue
        found = next((entry for entry in entries if entry.gn == job.gn), None)
        if found is not None:
            LAST_SHOWN.setdefault(key, found.words_key)  # after a load, a context shows its stored entry
        entry, missing, errors = spec.resolve(job, entries, LAST_SHOWN.get(key, ""))
        if entry is not None:
            LAST_SHOWN[key] = entry.words_key
            shown.append(entry)
            if entry is not found:
                if found is not None:
                    entries.remove(found)
                entries.append(entry)
                changed = True
            continue
        # Words that mean something else now never show the old answer. A converted v0.1.0 answer has no words
        # key to judge it by, so it stays until the new answer replaces it.
        if found is not None and found.words_key != job.words_key and not (
                found.legacy and found.answer_type == job.answer_type):
            entries.remove(found)
            changed = True
        failed += errors
        if missing:
            waiting = True
            _submit(job, missing)
    rows = changed  # entries were added, replaced or dropped: the table object's rows change too
    showing = {entry.words_key for entry in shown}
    left = LIVE.get(node.uid, showing) - showing  # entries no context shows any more
    LIVE[node.uid] = showing
    if changed or left:  # warm order and eviction go by last use (§8.1): stamp what was just in use
        now = time.time()
        for entry in entries:
            if entry.words_key in showing or entry.words_key in left:
                entry.t = now
        changed = True
    if changed:
        try:
            store.save(node, node_store, frozenset(jobs))  # live contexts are never evicted
        except ValueError as error:  # a damaged or too large entry: the node keeps what it has
            problems.append(str(error))
            changed = rows = False
            node_store = store.load(node)
    warnings += orphaned([job for job, _ctx in jobs.values()])  # PR-21: locks no context's rows use (§7.6)
    last = [LAST_SHOWN.get((node.uid, ctx.key())) for ctx in contexts]
    default = inner.k_default_key(last, blocked=bool(problems or hints or failed))
    if not inner.update_values(node, node_store, live, default, rows=rows):  # values only when it can (§8.8)
        inner.build(node, node_store, live, default)
    STATUS[node.uid] = _status(spec, jobs, shown, problems, hints, failed, upstream, waiting, warnings)
    return changed


def _live(live, spec, node, ctx):
    """Add the words ctx has now to the live gns, when they can be read."""
    try:
        gn = spec.gn(node, ctx)
    except (Hint, Pending, Unresolvable):
        return
    if gn not in live:
        live.append(gn)


def _submit(job, missing):
    """Ask for the missing cells once the model is ready, starting it if the preferences allow."""
    state, _message = server.status()
    if state == "ready":
        pipeline.submit(job, missing, job.priority, on_rows=lambda _job: schedule(immediate=True))
    elif state in ("stopped", "unknown") and P.get_prefs().auto_start:
        server.start()  # asked by the pass that runs when the server is ready


def _status(spec, jobs, shown, problems, hints, failed, upstream, waiting, warnings=()):
    if problems:
        return ('ERROR', problems[0])
    if failed:
        return ('ERROR', failed[0])
    if waiting or upstream:
        state, message = server.status()
        if state == "not_installed":
            return ('ERROR', "Model not installed: open the OpenAct tab in the sidebar")
        if state == "installing":
            return ('SORTTIME', "Installing the model...")
        if state == "error":
            return ('ERROR', message or "Model server error")
        if state == "starting":
            return ('SORTTIME', "Starting the model...")
        if state in ("stopped", "unknown") and not P.get_prefs().auto_start:
            return ('PAUSE', "Model server is stopped (OpenAct tab)")
        if upstream and not waiting:
            return ('SORTTIME', "Waiting for another OpenAct node...")
        return ('SORTTIME', "Asking...")
    if not jobs:
        return ('INFO', hints[0] if hints else spec.empty_hint)
    text = spec.summary(shown[0])
    if len(jobs) > 1:
        text += f"   +{len(jobs) - 1} more"
    if warnings:  # a usable list or answer set that has a warning (§17.1)
        return ('ERROR', f"{warnings[0]}   ({text})")
    return ('CHECKMARK', text)


def answers_for(node):
    """(context label, entry or None) for the node's current uses, for the sidebar."""
    spec = SPECS[node.bl_idname]
    entries = store.load(node).entries
    return [(ctx.label(), spec.resolve(job, entries, LAST_SHOWN.get((node.uid, ctx.key()), ""))[0])
            for ctx, job in _jobs(node)]


def older_settings(node, entry):
    """True when a row of the entry was asked with other templates, order balancing or backend than the node would
    use now. Unknown, so False, until the model server has reported its backend."""
    model = pipeline.provenance(server.STATE.health)
    current = SPECS[node.bl_idname].ask_key(node.balance_order, model)
    return bool(model) and any(ask and ask != current for ask in entry.asks)  # "": code rows, no model asked


def reask(node):
    """Ask the node's questions again with its current settings: the stored answers of its current words, and this
    session's answers and failures of their cells, are forgotten. Raises ValueError, changing nothing, if a newer
    OpenAct saved the answers."""
    if node.id_data.library is not None:  # linked data is read-only
        return
    jobs = [job for _ctx, job in _jobs(node)]
    node_store = store.load(node)
    words = {job.gn for job in jobs}
    node_store.entries = [entry for entry in node_store.entries if entry.gn not in words]
    store.save(node, node_store)  # no live set: nothing is evicted (§13.3)
    pipeline.forget(row.cell.key for job in jobs for row in job.rows if row.cell is not None)
    for key in [key for key in LAST_SHOWN if key[0] == node.uid]:
        del LAST_SHOWN[key]
    inner.build(node, node_store)
    schedule(immediate=True)


def redraw():
    wm = getattr(bpy.context, "window_manager", None)
    if wm is None:
        return
    for window in wm.windows:
        for area in window.screen.areas:
            if area.type in ('NODE_EDITOR', 'PREFERENCES'):
                area.tag_redraw()


# ------------------------------------------------------------------ handlers & timer


def _playing():
    """Don't touch node trees during playback or while a render job runs."""
    if bpy.app.is_job_running('RENDER'):
        return True
    wm = getattr(bpy.context, "window_manager", None)
    return bool(wm) and any(w.screen and w.screen.is_animation_playing for w in wm.windows)


def tick():
    global _last_refresh, _dirty
    try:
        if server.pump():
            redraw()
        now = time.monotonic()
        if now - _last_refresh > 5.0 and (_has_nodes or server.STATE.status in ("unknown", "ready", "starting")):
            _last_refresh = now
            server.refresh()
        if not _playing():
            if WATCH.poll(now):  # a watched CSV changed: the pass visits the nodes that read it (take_dirty())
                _dirty = True
            if _dirty and now - _last_change >= DEBOUNCE:
                run_pass(everything=False)
            elif _has_nodes and now - _last_pass >= SAFETY_PASS:
                run_pass()
    except Exception:
        traceback.print_exc()
    return 0.1


@persistent
def on_depsgraph_update(scene, depsgraph):
    # With no OpenAct nodes in the file, only new node groups (append, link) can bring some in;
    # adding or duplicating an OpenAct node, loading, undo and redo schedule a pass on their own.
    if not _has_nodes and len(bpy.data.node_groups) == _group_count:
        return
    schedule()


@persistent
def on_load_post(*_args):
    STATUS.clear()
    pipeline.FAILED.clear()
    PLANS.clear()
    LAST_SHOWN.clear()
    LIVE.clear()
    schedule(immediate=True)


@persistent
def on_undo_redo(*_args):
    schedule(immediate=True)


def on_server_status():
    schedule(immediate=True)
    redraw()


def _handlers():
    h = bpy.app.handlers
    return ((h.depsgraph_update_post, on_depsgraph_update), (h.load_post, on_load_post),
            (h.undo_post, on_undo_redo), (h.redo_post, on_undo_redo))


def register():
    for handlers, fn in _handlers():
        if fn not in handlers:
            handlers.append(fn)
    server.add_status_listener(on_server_status)
    if not bpy.app.timers.is_registered(tick):
        bpy.app.timers.register(tick, first_interval=0.5, persistent=True)


def unregister():
    if bpy.app.timers.is_registered(tick):
        bpy.app.timers.unregister(tick)
    for handlers, fn in _handlers():
        if fn in handlers:
            handlers.remove(fn)
