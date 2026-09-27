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

import dataclasses
import math
import time
import traceback
import uuid
from dataclasses import dataclass, field

import bpy
from bpy.app.handlers import persistent
from bpy_extras import anim_utils

from . import csvfmt, evaluator, inner, pipeline, server, store, trust
from . import prefs as P
from . import tables
from .csvfmt import ERROR, INFO, WARNING, Problem
from .evaluator import Ctx, Hint, Pending, Unresolvable, find_uses, is_field
from .inner import INNER_PREFIX
from .specs import SPECS
from .specs.ask import AskSpec
from .specs.base import UNSURE_BELOW, orphaned
from .specs.pick import PickSpec
from .watch import WATCH

ASK_ID, PICK_ID = AskSpec.idname, PickSpec.idname
OPENACT_TYPES = tuple(SPECS)
DEBOUNCE = 0.25
SAFETY_PASS = 2.0  # edits to trees no modifier uses send no depsgraph update; it re-plans only changed fingerprints
CLASH = ("'{}' and '{}' share every word but not their list (a collection, object or CSV): only one answer can "
         "show. Give each a different Key (Ctrl H shows it; expose it on the modifier)")
OVER_CAP = "{} objects use this tree with different words; the limit is {}: split the tree"
ICONS = {"hint": 'INFO', "paused": 'PAUSE', "starting": 'SORTTIME', "waiting": 'SORTTIME', "filling": 'SORTTIME',
         "complete": 'CHECKMARK', "warning": 'ERROR', "error": 'CANCEL', "linked": 'LINKED'}
INTERACTIVE = ("ask", "pick", "dials")  # asked at P0 whatever their size, as is any job of at most P0_CELLS (§5.5)
P0_CELLS = 8
PARTIAL = "openact_partial"  # ID property of the inner group: digest of the unstored partial tables its table holds

STATUS = {}  # node uid -> NodeStatus
PLANS = {}  # (node uid, ctx key) -> (fingerprint, Job, or the (Hint or Unresolvable, message) planning gave)
LAST_SHOWN = {}  # (node uid, ctx key) -> words key of the entry that context showed last
LIVE = {}  # node uid -> words keys of the entries its contexts showed at its last pass
CONTEXTS = {}  # node uid -> the ctx keys of its contexts at the last pass: tells the pass when one goes away
DIRTY = set()  # uids of the nodes the next pass visits, when not all of them are dirty
GENERATIONS = {}  # node uid -> its generation, bumped when its words change: its older queued chunks are dropped
DUE = set()  # uids saved without their live set since their last pass (pins, Unpin): that pass saves with it (§13.3)
NOTES = {}  # node uid -> what its last save evicted or kept above the soft cap (NodeStore.note), for its status

_dirty = True  # a pass is due
_dirty_all = True  # ... and it visits every node
_last_change = 0.0
_last_refresh = 0.0
_last_pass = 0.0
_has_nodes = False
_group_count = -1  # number of node groups at the last pass that found no OpenAct nodes
_plans_made = 0  # a pass that plans changes what nodes reading another's answers as words may read
_pass_plans = None  # the PLANS keys the running pass planned, contexts only an evaluator read reaches included
passes = 0  # passes run this session: what the sidebar works out once per pass is keyed by it


@dataclass
class NodeStatus:
    """What a node shows (§17.1): its body the icon and text, the sidebar all of it."""

    state: str  # a key of ICONS
    text: str  # one line, e.g. "Filling 12/50 · ~2 s"
    rows: int = 0
    answered: int = 0
    pending: int = 0
    stale: int = 0
    unsure: int = 0
    locked: int = 0
    code: int = 0
    errors: int = 0
    eta_s: float | None = None  # remaining scored rows × ms_per_row
    contexts: int = 0  # distinct live word sets; above inner.MATERIALISED_MAX the node shows the over-cap error (§8.1)
    problems: list = field(default_factory=list)  # csvfmt.Problem records: the errors, then the warnings
    uses: list = field(default_factory=list)  # (context label, Job, entry shown or None) of each planned context
    asking: frozenset = frozenset()  # keys of the cells the pass asks: sent, or sent once the starting model is ready
    pins: list = field(default_factory=list)  # the pinned entries, in pin order (§5.5)

    @property
    def icon(self) -> str:
        return ICONS[self.state]


def _problem(severity, text):
    return Problem(severity, "", 0, "", text)


def _error(text):
    return NodeStatus("error", text, problems=[_problem(ERROR, text)])


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
    node_store = store.load(node)
    if node_store.schema > store.SCHEMA:  # its answers can't be read, so they never arrive
        raise Unresolvable(f"'{node.name}': {store.newer_message(node_store)}")
    entry, _missing, _failed = spec.resolve(job, node_store.entries, LAST_SHOWN.get((node.uid, ctx.key()), ""))
    if entry is None:
        raise Pending()
    try:
        return spec.output(entry, out.name)
    except KeyError:
        raise Unresolvable(f"'{node.name}' output '{out.name}' can't be a word") from None


evaluator.HANDLERS.update(dict.fromkeys(OPENACT_TYPES, _openact_output))


def _plan(spec, node, ctx):
    """The node's Job in ctx with its Overrides CSV applied, planned again only when its fingerprint changed (§5.3).
    Raises Hint, Pending or Unresolvable."""
    global _plans_made
    if evaluator.READS is not None or evaluator.OVERRIDES is not None:  # Prepare for Playback reads other frames
        return spec.locked(node, ctx, spec.plan(node, ctx))  # the plan cache and generations are the live words'
    key = (node.uid, ctx.key())
    if _pass_plans is not None:
        _pass_plans.add(key)
    fingerprint = spec.fingerprint(node, ctx)
    cached = PLANS.get(key)
    if cached is None or cached[0] != fingerprint:
        if cached is not None:  # new words: the chunks still queued for the node ask about the old ones
            _bump(node.uid)
        try:
            planned = spec.locked(node, ctx, spec.plan(node, ctx))
        except (Hint, Unresolvable) as error:  # kept as a message: the evaluator adds its hops to a raised error
            planned = (type(error), str(error))
        cached = PLANS[key] = (fingerprint, planned)
        _plans_made += 1
    if isinstance(cached[1], tuple):
        kind, message = cached[1]
        raise kind(message)
    return cached[1]


def _bump(uid):
    """A new generation for the node, whose queued chunks of older ones are dropped (§5.5)."""
    GENERATIONS[uid] = GENERATIONS.get(uid, 0) + 1
    pipeline.cancel(uid, below=GENERATIONS[uid])


def _jobs(node):
    """(ctx, Job) for each use of the node that has a question."""
    tree = node.id_data
    uses = find_uses().get(tree.as_pointer()) or [Ctx(None, None, (), tree)]
    return [(ctx, job) for ctx, job in _scan(SPECS[node.bl_idname], node, uses) if not isinstance(job, Exception)]


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


def iter_nodes(types=OPENACT_TYPES):
    """(tree, node) of each node of these types outside inner groups: the OpenAct nodes, which the pass visits, or the
    old Jev nodes (migrate.LEGACY_TYPES), which stay frozen until the upgrade (§12.3)."""
    for tree in bpy.data.node_groups:
        if tree.bl_idname != 'GeometryNodeTree' or tree.name.startswith(INNER_PREFIX):
            continue
        for node in tree.nodes:
            if node.bl_idname in types:
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
    global _dirty, _dirty_all, _has_nodes, _last_pass, _group_count, _pass_plans, passes
    visit_all = everything or _dirty_all
    dirty = set(DIRTY)
    DIRTY.clear()
    for uid, kind in WATCH.take_dirty().items():  # a CSV they read changed on disk: plan them again (§7.8)
        for key in [key for key in PLANS if key[0] == uid]:
            del PLANS[key]
        if kind == "text":  # new words: the chunks queued for the old ones go
            _bump(uid)
        dirty.add(uid)
    _dirty = _dirty_all = False
    _last_pass = time.monotonic()
    passes += 1
    nodes = list(iter_nodes())
    _has_nodes = bool(nodes)
    _group_count = -1 if nodes else len(bpy.data.node_groups)
    if not nodes:
        for uid in LIVE:  # the nodes are gone: their queued chunks too
            pipeline.cancel(uid, below=math.inf)
        STATUS.clear()
        PLANS.clear()
        LAST_SHOWN.clear()
        LIVE.clear()
        CONTEXTS.clear()
        NOTES.clear()
        DUE.clear()
        REBAKES.clear()
        WATCH.retain(())
        tables.collect_garbage(nodes)
        return
    ensure_owned(nodes)
    tables.collect_garbage(nodes)  # after ensure_owned(), which gives copies their own table objects
    pinned = _pin_pending()  # a Prepare for Playback run whose last answer came: its pins are written now
    dirty |= pinned
    uses = find_uses()
    plans_made, stored = _plans_made, False
    evaluator.MEMO = {}  # nodes that read the same upstream words in a context share one read
    csvfmt.IDENTITIES = {}  # every fingerprint resolves its files and asks for the trust state: once per pass (§3.2)
    planned = _pass_plans = set()
    try:
        trust.PASS = trust.is_trusted()
        for tree, node in nodes:
            if not (visit_all or node.uid in dirty):
                continue
            try:
                stored |= _process(tree, node, uses.get(tree.as_pointer()))
            except Exception as error:
                traceback.print_exc()
                STATUS[node.uid] = _error(f"Internal error: {error}")
    finally:
        evaluator.MEMO = None  # keyed by pointers: after the pass, nodes and data may be freed or edited
        csvfmt.IDENTITIES = trust.PASS = None  # files, the .blend and the trusted lists change between passes
        _pass_plans = None
    if stored or _plans_made != plans_made:
        schedule(immediate=True)  # a node that reads another's answers as words may read new ones
    contexts = {node.uid: {ctx.key() for ctx in uses.get(tree.as_pointer()) or [Ctx(None, None, (), tree)]}
                for tree, node in nodes}
    # Nothing shows the words of a context that has gone (its object deleted, its modifier removed or re-pointed, a
    # group node on its path muted): its node's queued chunks go, and the next pass queues the other contexts' again
    lost = [uid for uid, keys in CONTEXTS.items() if uid in contexts and not keys <= contexts[uid]]
    CONTEXTS.clear()
    CONTEXTS.update(contexts)
    # Contexts that disappeared, and nodes that did, but never one this pass planned: an evaluator read also reaches
    # contexts find_uses() lacks (through a group node in a tree no modifier uses), planned again every pass if dropped
    for cache in (PLANS, LAST_SHOWN):
        for key in [key for key in cache if key not in planned and key[1] not in contexts.get(key[0], ())]:
            del cache[key]
    for uid in lost:
        _bump(uid)
        schedule(immediate=True)
    for uid in [uid for uid in LIVE if uid not in contexts]:  # deleted nodes: their queued chunks go (undo asks again)
        pipeline.cancel(uid, below=math.inf)
    # nodes that disappeared: undo and deleting a node group call no node's free()
    for cache in (LIVE, STATUS, NOTES, REBAKES):
        for uid in [uid for uid in cache if uid not in contexts]:
            del cache[uid]
    WATCH.retain(contexts)  # stop watching the files of deleted nodes
    if pinned:  # written after their operator's undo step: in a step of their own, which only its undo removes
        _undo_push("Pin for Playback")
    redraw()


def _process(tree, node, contexts):
    """Plan and answer the node's contexts, store new answers, rebuild its group and set its status. True if it stored
    any."""
    if tree.library is not None:  # read-only: the library file holds this node's answers
        STATUS[node.uid] = NodeStatus("linked", f"Linked from {tree.library.name}: answers come from that file")
        return False
    node_store = store.load(node)
    if node_store.schema > store.SCHEMA:  # save() would refuse any answer, and the group holds the newer ones
        STATUS[node.uid] = _error(store.newer_message(node_store))
        return False
    spec = SPECS[node.bl_idname]
    entries = node_store.entries
    contexts = contexts or [Ctx(None, None, (), tree)]
    jobs, shown, live = {}, [], []  # words -> (Job, ctx), one per distinct words; the entries shown; live gns
    # Stored entries whose words mean something else now stay stored for their rows, but leave the graph (§5.4):
    hidden = {}  # their (gn, words key) -> the partial table shown in their place, or None while nothing is known
    instead = {}  # (uid, ctx key) -> the words key of the partial table that context shows in a hidden entry's place
    problems, warnings, failed = [], [], []
    uses, resolved, asked = [], {}, set()  # (label, Job) per planned context; words -> entry shown; cells missing
    sent = set()  # the missing cells the model is asked: sent now, or once the model that is starting is ready
    upstream = waiting = changed = False
    planned = _scan(spec, node, contexts)
    # §6.0: the sockets follow the live jobs (Rules, Classify, Dials). Jobs that disagree answer nothing, so the last
    # good table stays. A context without a job (Pending; Unresolvable: a CSV with errors, too many items...; or a Hint
    # whose words still name a stored entry: an Import CSV node without a Path, an empty collection) still shows its
    # last table, whose shape is unknown. While there is one, agree() keeps the sockets, and jobs that don't fit them
    # wait (None) until that context plans again or goes away; jobs that fit them are answered
    gns = {entry.gn for entry in entries}
    hinted = {ctx.key(): gn for ctx, job in planned if isinstance(job, Hint) and (gn := _gn(spec, node, ctx)) in gns}
    hints = [str(job) for ctx, job in planned if ctx.key() in hinted]  # first: they are the hints that hold the node
    unknown = bool(hinted) or any(isinstance(job, (Pending, Unresolvable)) for _ctx, job in planned)
    conflict = spec.agree(node, [(ctx, job) for ctx, job in planned if not isinstance(job, Exception)], unknown,
                          entries) if hasattr(spec, "agree") else ""
    held = conflict is None
    if conflict:
        problems.append(conflict)
    for ctx, job in planned:
        key = (node.uid, ctx.key())
        if isinstance(job, Pending):  # its words wait for another node's answer: the entry it showed last stays live
            upstream = True
            last = next((entry for entry in entries if entry.words_key == LAST_SHOWN.get(key)), None)
            if last is not None and last.gn not in live:
                live.append(last.gn)
            continue
        if isinstance(job, Hint):
            gn = hinted.get(ctx.key())
            if gn is None:
                hints.append(str(job))
            elif gn not in live:  # the stored entry its words name still shows: kept, as for an error
                live.append(gn)
            continue
        if isinstance(job, Unresolvable):  # e.g. a CSV with errors: its words stay live, so the last good table shows
            problems.append(str(job))
            _live(live, spec, node, ctx)
            continue
        uses.append((ctx.label(), job))
        warnings += job.problems
        problems += job.errors  # rows in error in a table that still shows, e.g. cells no outcome applies to
        if job.gn not in live:
            live.append(job.gn)
        if conflict or held:  # nothing is answered: the sidebar shows the stored entry resolve() keeps, if any
            found = next((entry for entry in entries if entry.gn == job.gn), None)
            kept = spec.resolve(job, entries, LAST_SHOWN.get(key, ""), partial=True)[0] if found is not None else None
            if kept is not None and kept.words_key == found.words_key:  # it, its table so far or a refreshed copy
                resolved[job.gn] = found
            continue
        other = jobs.setdefault(job.gn, (job, ctx))
        if other[0] is not job and other[0].words_key != job.words_key:
            problems.append(CLASH.format(other[1].label(), ctx.label()))
            continue
        found = next((entry for entry in entries if entry.gn == job.gn), None)
        if found is not None:  # after a load, a context shows its stored entry
            LAST_SHOWN.setdefault(key, found.words_key)
        if other[0] is not job:  # the same words and meaning as an earlier context: the graph shows it that entry
            if any(entry is found for entry in shown):
                LAST_SHOWN[key] = found.words_key
            continue
        # While cells are missing, the table so far shows: stale-while-refilling (§5.4)
        entry, missing, errors = spec.resolve(job, entries, LAST_SHOWN.get(key, ""), partial=True)
        if entry is not None and found is not None and entry is not found:
            entry.pinned, entry.frames = found.pinned, found.frames  # a pin holds its words, whatever they mean
            if dataclasses.replace(entry, t=found.t) == found:
                entry = found  # the partial table as stored already
        if entry is not None:  # the table shown: the sidebar and the counts describe it, not a hidden entry
            shown.append(entry)
            resolved[job.gn] = entry
        # Words that mean something else now never show the old answer, but its rows carry over until the new table is
        # whole, also after a reload: the old entry stays stored, and the partial table shows in its place unstored.
        # LAST_SHOWN keeps naming the stored entry its stale rows come from.
        if found is not None and found.words_key != job.words_key and (entry is None or missing or errors):
            hidden[(found.gn, found.words_key)] = entry
            if entry is not None:
                instead[key] = entry.words_key
        elif entry is not None:
            LAST_SHOWN[key] = entry.words_key
            if entry is not found:
                if found is not None:  # in place: the graph keeps its entry order, so values only (§8.8)
                    entries[entries.index(found)] = entry
                else:
                    entries.append(entry)
                changed = True
        failed += errors
        if missing:
            waiting = True
            asked.update(cell.key for cell in missing)
            if _submit(dataclasses.replace(job, priority=_priority(spec, node, missing),
                                           generation=GENERATIONS.get(node.uid, 0)), missing):
                sent.update(cell.key for cell in missing)
    if len(live) > inner.MATERIALISED_MAX:  # the first contexts stay materialised, the others show neutral values
        problems.append(OVER_CAP.format(len(live), inner.MATERIALISED_MAX))
    due = node.uid in DUE  # saved without its live set since the last pass: the limits apply now (§13.3)
    rows = changed or due  # entries were added or replaced (here, or before that save): the table object's rows too
    showing = {entry.words_key for entry in shown}
    left = LIVE.get(node.uid, showing) - showing  # entries no context shows any more
    LIVE[node.uid] = showing
    if changed or left:  # warm order and eviction go by last use (§8.1): stamp what was just in use
        now = time.time()
        for entry in entries:
            if entry.words_key in showing:  # one step after the ones just left: the newest is a shown entry
                entry.t = math.nextafter(now, math.inf)
            elif entry.words_key in left:
                entry.t = now
        changed = True
    if changed or due:
        DUE.discard(node.uid)
        try:
            store.save(node, node_store, frozenset(live))  # live contexts are never evicted
            NOTES[node.uid] = node_store.note
        except ValueError as error:  # a damaged or too large entry: the node keeps what it has
            problems.append(str(error))
            changed = rows = False
            node_store = store.load(node)
    warnings += orphaned([job for job, _ctx in jobs.values()])  # locks no context's rows use (§7.6)
    keys = [(node.uid, ctx.key()) for ctx in contexts]
    last = [instead.get(key, LAST_SHOWN.get(key)) for key in keys]  # a no-match shows what the graph showed last
    default = inner.k_default_key(last, blocked=bool(problems or hints or failed))
    graph = node_store
    if hidden:  # a hidden entry leaves the graph, or leaves its place to the partial table shown instead
        graph = store.NodeStore([entry for entry in (hidden.get((stored.gn, stored.words_key), stored)
                                                     for stored in node_store.entries) if entry is not None])
    # An unstored partial table has the gn, so the place in the graph, of the entry it shows instead of. The table
    # object's rows are rewritten when the partial tables shown change, and once after (also in a reopened file): only
    # then, as a rewrite re-evaluates every modifier that reads the table, whose depsgraph update schedules a pass.
    partials = [(entry.words_key, entry.rows.v, entry.rows.payload) for entry in hidden.values() if entry is not None]
    partial = pipeline.digest(*partials) if partials else ""
    written = node.node_tree.get(PARTIAL, "") if node.node_tree is not None else ""
    if not inner.update_values(node, graph, live, default, rows=rows or partial != written):  # values only (§8.8)
        inner.build(node, graph, live, default)
    if partial != written:  # the table object holds the rows of the graph just laid out
        node.node_tree[PARTIAL] = partial
    pins = sorted((entry for entry in node_store.entries if entry.pinned), key=inner._pin_order)
    # live entries win the materialised places (§8.1): the pins that start latest in the frame range are left out
    inactive = sum(entry.pinned for entry in graph.entries)
    if inactive:
        inactive -= sum(entry.pinned for entry in inner.materialised(graph.entries, live, default))
    if inactive:
        warnings.append(INACTIVE.format(inactive, "ies are" if inactive > 1 else "y is", inner.MATERIALISED_MAX))
    if flag := rebake(node.uid, pins):  # while it holds the pins that changed the rules, until the next run
        warnings.append(flag)
    warnings += spec.node_warnings(node)  # known once built, e.g. a renamed output that lost its link
    faults, warnings = list(dict.fromkeys(problems + failed)), list(dict.fromkeys(warnings))  # contexts share words
    state, text, eta = _status(spec, node.uid, jobs, shown, faults, hints, upstream, waiting, warnings)
    STATUS[node.uid] = NodeStatus(
        state, text, eta_s=eta, contexts=len(live), uses=[(label, job, resolved.get(job.gn)) for label, job in uses],
        problems=[_problem(ERROR, fault) for fault in faults] + [_problem(WARNING, w) for w in warnings]
        + [_problem(INFO, note) for note in (NOTES.get(node.uid),) if note],
        asking=frozenset(sent), pins=pins, **_counts(jobs, resolved, asked))
    return changed


def _live(live, spec, node, ctx):
    """Add the words ctx has now to the live gns, when they can be read."""
    gn = _gn(spec, node, ctx)
    if gn is not None and gn not in live:
        live.append(gn)


def _gn(spec, node, ctx):
    """The words ctx has now, or None when they can't be read."""
    try:
        return spec.gn(node, ctx)
    except (Hint, Pending, Unresolvable):
        return None


def _scan(spec, node, contexts):
    """(ctx, its Job, or the Hint, Pending or Unresolvable that planning raised) of each context: a visit plans each
    context once, for agree() and for answering."""
    planned = []
    for ctx in contexts:
        try:
            planned.append((ctx, _plan(spec, node, ctx)))
        except (Hint, Pending, Unresolvable) as error:
            planned.append((ctx, error))
    return planned


def _submit(job, missing):
    """Ask for the missing cells once the model is ready, starting it if the preferences allow. True when they are
    asked: sent now, or sent by the pass that runs once the model is ready, as a start is under way, queued or begun
    here."""
    state, _message = server.status()
    if state == "ready":
        pipeline.submit(job, missing, job.priority, on_rows=lambda _job: schedule(immediate=True))
        return True
    if state in ("stopped", "unknown", "installing") and P.get_prefs().auto_start:
        return server.start()  # asked by the pass that runs when the server is ready; an install starts it as it ends
    return state == "starting" or server.STATE.start_queued


def _priority(spec, node, missing):
    """The job's class (§5.5): P0 for Ask, Pick, Dials and any job of at most P0_CELLS missing cells, P1 for the table
    of the node active in a node editor, P2 for every other table."""
    if spec.name in INTERACTIVE or len(missing) <= P0_CELLS:
        return 0
    wm = getattr(bpy.context, "window_manager", None)
    for window in wm.windows if wm else ():
        for area in window.screen.areas if window.screen else ():
            tree = getattr(area.spaces.active, "edit_tree", None) if area.type == 'NODE_EDITOR' else None
            if tree is not None and tree.nodes.active == node:
                return 1
    return 2


def _status(spec, uid, jobs, shown, faults, hints, upstream, waiting, warnings):
    """(state, text, ETA in seconds or None) of the node's status line (§17.1)."""
    if faults:
        return "error", faults[0], None
    if waiting or upstream:
        state, message = server.status()
        if state == "not_installed":
            return "error", "Model not installed: open the OpenAct tab in the sidebar", None
        if state == "installing":
            return "starting", "Installing the model...", None
        if state == "error":
            return "error", message or "Model server error", None
        if state == "starting":
            return "starting", "Starting the model...", None
        if message == server.CHECKING:  # a start waits for the runtime probe
            return "starting", message, None
        if state in ("stopped", "unknown") and not P.get_prefs().auto_start:
            return "paused", "Model server is stopped (OpenAct tab)", None
        if upstream and not waiting:
            return "waiting", "Waiting for another OpenAct node...", None
        return ("filling", *_filling(uid))
    if not jobs:
        return "hint", hints[0] if hints else spec.empty_hint, None
    text = spec.summary(shown[0])
    locked = shown[0].rows.src.count("l")
    if locked:
        text += f" · {locked} locked"
    if len(jobs) > 1:
        text += f"   +{len(jobs) - 1} more"
    if warnings:  # a usable table whose lists, answer sets or locks have warnings
        return "warning", f"{text} · {len(warnings)} warning{'s' if len(warnings) > 1 else ''}: {warnings[0]}", None
    return "complete", text, None


def _filling(uid):
    """The filling line and its ETA: n/N once the decision worker reports the node's progress, else "Asking..."."""
    found = pipeline.progress(uid)
    if not found or not found[1]:
        return "Asking...", None
    done, total, ms_per_row = found  # ms_per_row: per remaining cell, /health's rate and the §5.4 default folded in
    eta = (total - done) * ms_per_row / 1000
    left = f"~{max(math.ceil(eta), 1)} s" if eta < 60 else f"~{math.ceil(eta / 60)} min"
    return f"Filling {done}/{total} · {left}", eta


def _counts(jobs, resolved, asked):
    """The row counts of the node's tables (§17.1): by the source of each row of the entry a table shows, else of its
    plan's rows, whose cells are asked (pending), failed (errors) or answered this session."""
    counts = dict.fromkeys(("rows", "answered", "pending", "stale", "unsure", "locked", "code", "errors"), 0)
    letters = {"m": "answered", "p": "pending", "s": "stale", "l": "locked", "c": "code", "e": "errors"}
    for gn, (job, _ctx) in jobs.items():
        counts["rows"] += len(job.rows)
        entry = resolved.get(gn)
        if entry is not None:
            for letter in entry.rows.src:
                counts[letters.get(letter, "answered")] += 1
            confidences = entry.rows.v.get("c") or ()
            counts["unsure"] += sum(src == "m" and c < UNSURE_BELOW for src, c in zip(entry.rows.src, confidences))
            continue
        for row in job.rows:
            if row.cell is None:
                name = {"lock": "locked", "error": "errors"}.get(row.source, "code")
            elif row.cell.key in asked:
                name = "pending"
            else:
                name = "errors" if row.cell.key in pipeline.FAILED else "answered"
            counts[name] += 1
    return counts


def answers_for(node):
    """(context label, Job, entry shown or None) of each context the node's last pass planned, for the sidebar. Drawing
    reads them and plans nothing: planning may rebuild the node, which Blender refuses while drawing. Nothing for a
    newer OpenAct's answers, which can't be read: the node's status says why."""
    status = STATUS.get(node.uid)
    return status.uses if status is not None else []


def ask_key_now(spec, debias):
    """The ask key a row would be answered with now: templates, Balance Option Order and backend. "" until the model
    server has reported its backend."""
    model = pipeline.provenance(server.STATE.health)
    return spec.ask_key(debias, model) if model else ""


def older_settings(node, entry):
    """True when a row of the entry was asked with other templates, order balancing or backend than the node would
    use now. Unknown, so False, until the model server has reported its backend."""
    current = ask_key_now(SPECS[node.bl_idname], node.balance_order)
    return bool(current) and any(ask and ask != current for ask in entry.asks)  # "": code rows, no model asked


def reask(node):
    """Ask the node's questions again with its current settings: the stored answers of its current words, and this
    session's answers and failures of their cells, are forgotten. Raises ValueError, changing nothing, if a newer
    OpenAct saved the answers."""
    if node.id_data.library is not None:  # linked data is read-only
        return
    jobs = [job for _ctx, job in _jobs(node)]
    node_store = store.load(node)
    spec, words = SPECS[node.bl_idname], {job.gn: job for job in jobs}
    kept = []
    for entry in node_store.entries:
        job = words.get(entry.gn)
        if job is None:
            kept.append(entry)
        elif entry.pinned:  # a pin holds its words (§14.2): they wait for their new answers, none known yet
            asking = spec.entry(job, spec.stale(job, list(job.rows), None))
            asking.pinned, asking.frames = True, entry.frames
            kept.append(asking)
    node_store.entries = kept
    store.save(node, node_store)  # no live set: nothing is evicted (§13.3)
    pipeline.forget(row.cell.key for job in jobs for row in job.rows if row.cell is not None)
    for key in [key for key in LAST_SHOWN if key[0] == node.uid]:
        del LAST_SHOWN[key]
    # as the next pass lays it out: the live set, and no default now that nothing was shown
    inner.build(node, node_store, list(dict.fromkeys(job.gn for job in jobs)), None)
    schedule(immediate=True)


def redraw():
    wm = getattr(bpy.context, "window_manager", None)
    if wm is None:
        return
    for window in wm.windows:
        for area in window.screen.areas:
            if area.type in ('NODE_EDITOR', 'PREFERENCES'):
                area.tag_redraw()


# ------------------------------------------------------------ Prepare for Playback (§5.5)

PIN_MAX = 12  # pinned entries per node (§8.1)
PIN_LIMIT = "'{}': {} distinct word sets over frames {}–{}; the limit is {}"
PIN_CAP = "'{}': {} word sets live now and {} more over frames {}–{}; live plus pinned is limited to {}: split the tree"
INACTIVE = ("{} pinned entr{} inactive: this tree is now used by more objects (live + pinned > {}). Re-run Prepare "
            "for Playback, or split the tree")
REBAKE = "Re-bake {} if baked: rules changed for frames {}"
LAND_MODES = ('OBJECT', 'POSE', 'PAINT_WEIGHT', 'PAINT_VERTEX')  # whole-file undo steps: a push there holds pins
RUN_KEY = "_openact_playback"  # the scene's custom property naming its last run; "_" hides it in the sidebar
PLAYBACK = None  # the Run waiting for its answers, to pin them, or None
PLAYBACK_NOTE = ""  # why the last run stopped or was refused, or what it couldn't save, for the sidebar
REBAKES = {}  # node uid -> (which of its modifiers to re-bake, the words keys of the pins that changed their rules)
_stepping = False  # the fallback holds the scene at other frames: no pass runs


@dataclass
class Run:
    """A Prepare for Playback run waiting for its answers."""

    id: str  # also its scene's RUN_KEY, written inside the operator's undo step: undoing that leaves the run dormant
    sets: dict  # node uid -> [(Job, frames)] of the word sets to pin
    held: str = ""  # once every answer is in, why its pins still wait (_hold()), as the last pass found


class Playback:
    """One Prepare for Playback run: every distinct word tuple each context of each OpenAct node sees over the scene's
    frame range, read through the F-curves of the values the words read, with no scene evaluation (read_curves()), or
    by setting each frame (step_frames(), the fallback); then answered and pinned (finish())."""

    def __init__(self, scene):
        self.scene, self.first, self.last = scene, scene.frame_start, scene.frame_end
        uses = find_uses()
        self.uses = [(node, SPECS[node.bl_idname], uses.get(tree.as_pointer()) or [Ctx(None, None, (), tree)])
                     for tree, node in iter_nodes() if tree.library is None and node.uid]  # linked nodes stay as saved
        self.contexts = [(node, spec, ctx) for node, spec, contexts in self.uses for ctx in contexts]
        self.live = {(node.uid, ctx.key()): words for node, spec, ctx in self.contexts
                     if (words := _read(node, spec, ctx)[0]) is not None}  # the words each context has now
        self.seen = {}  # (uid, ctx key) -> {words: the frames the context has them at}
        self.origin = {}  # (uid, words) -> (the first frame, the context) they occur at, where they are planned
        self.values = {}  # frame -> the values the words read there (evaluator.OVERRIDES)
        self.problems = {}  # why some words can't be read -> the first frame
        self.fallback = ""  # why the frames must be set, found by read_curves()

    def _record(self, frame, node, ctx, words, problem):
        if words is not None:
            self.seen.setdefault((node.uid, ctx.key()), {}).setdefault(words, []).append(frame)
            self.origin.setdefault((node.uid, words), (frame, ctx))
        if problem:
            self.problems.setdefault(problem, frame)

    def read_curves(self):
        """The F-curve path, purely in Python: each context's words are read at the first frame, and at a later one
        only when they read a value an F-curve animates, which then takes that F-curve's value there (evaluate()
        applies its modifiers). False as soon as a value needs the fallback: self.fallback says why."""
        render = self.scene.render
        stretched = render.frame_map_old != render.frame_map_new  # Time Stretching plays animation at other times
        curves, checked = {}, set()  # read key -> (its F-curve, its value now, its hard range); the keys looked at
        last = {}  # (uid, ctx key) -> (words, the keys they read, why they can't be read) at the last frame read
        try:
            for frame in range(self.first, self.last + 1):
                values = evaluator.OVERRIDES = {key: _as(value, curve.evaluate(frame), limits)
                                                for key, (curve, value, limits) in curves.items()}
                for node, spec, ctx in self.contexts:
                    at = (node.uid, ctx.key())
                    if at in last and last[at][1].isdisjoint(curves):  # it reads nothing animated: its words stay
                        self._record(frame, node, ctx, last[at][0], last[at][2])
                        continue
                    while True:  # a value read only at this frame (down another Switch branch) may be animated too
                        words, reads, problem = _read(node, spec, ctx)
                        new = reads.keys() - checked
                        checked |= new
                        for key in new:
                            struct, name, custom, value = reads[key]
                            curve, why = _animation(struct.id_data, key[1])
                            if curve is not None and stretched:
                                why = "Time Stretching is on"
                            elif curve is not None and isinstance(value, str):  # a menu socket: its F-curve holds ids
                                why = f"a menu socket of '{struct.id_data.name}' is keyframed"
                            if why:
                                self.fallback = why
                                return False
                            if curve is not None:
                                limits = None if isinstance(value, bool) else _range(struct, name, custom)
                                curves[key] = (curve, value, limits)
                                values[key] = _as(value, curve.evaluate(frame), limits)
                        if new.isdisjoint(curves):
                            break
                    last[at] = (words, set(reads), problem)
                    self._record(frame, node, ctx, words, problem)
                self.values[frame] = values
        finally:
            evaluator.OVERRIDES = None
        return True

    def step_frames(self):
        """The fallback: set each frame in turn, evaluating the scene, and read the words; yields each frame once read.
        Closing the generator cancels. Either way the frame current before is set again."""
        global _stepping
        self.seen, self.origin, self.values, self.problems = {}, {}, {}, {}
        scene = self.scene
        current, subframe = scene.frame_current, scene.frame_subframe
        _stepping = True  # a pass now would take this frame's words for the live ones
        try:
            for frame in range(self.first, self.last + 1):
                scene.frame_set(frame)
                values = self.values[frame] = {}
                for node, spec, ctx in self.contexts:
                    words, reads, problem = _read(node, spec, ctx)
                    values.update((key, read[3]) for key, read in reads.items())
                    self._record(frame, node, ctx, words, problem)
                yield frame
        finally:
            scene.frame_set(current, subframe=subframe)
            _stepping = False

    def finish(self):
        """Check the limits, then ask for the word sets of the contexts whose words over the range aren't only the
        ones they have now, and pin them once all are answered, replacing every earlier pin. A new run is how a
        question that failed is asked again. Returns the report, the size of the pins first; raises ValueError, having
        asked nothing, when a node would pass a limit."""
        global PLAYBACK, PLAYBACK_NOTE
        chosen = {}  # uid -> {words: the frames they occur at}
        for node, _spec, contexts in self.uses:
            sets = {}
            for ctx in contexts:
                key = (node.uid, ctx.key())
                seen = self.seen.get(key, {})
                if set(seen) <= {self.live.get(key)}:  # its live words all along: the graph holds them anyway
                    continue
                for gn, frames in seen.items():
                    sets.setdefault(gn, set()).update(frames)
            live = {gn for (uid, _key), gn in self.live.items() if uid == node.uid}
            refused = ""
            if len(sets) > PIN_MAX:
                refused = PIN_LIMIT.format(node.name, len(sets), self.first, self.last, PIN_MAX)
            elif sets and len(live | sets.keys()) > inner.MATERIALISED_MAX:
                refused = PIN_CAP.format(node.name, len(live), len(sets.keys() - live), self.first, self.last,
                                         inner.MATERIALISED_MAX)
            if refused:
                PLAYBACK_NOTE = refused
                raise ValueError(refused)
            chosen[node.uid] = sets
        REBAKES.clear()  # the last run's: this one flags its own
        pending, size = {}, 0
        try:
            for node, spec, _contexts in self.uses:
                entries = None
                for gn, frames in chosen[node.uid].items():
                    frame, ctx = self.origin[(node.uid, gn)]
                    evaluator.OVERRIDES = self.values[frame]  # the values its words read at that frame
                    try:
                        job = spec.locked(node, ctx, spec.plan(node, ctx))
                    except Hint:
                        continue  # nothing to ask at those frames
                    except (Pending, Unresolvable) as error:
                        self.problems.setdefault(_unread(node, error), frame)
                        continue
                    entries = store.load(node).entries if entries is None else entries
                    size += _size(spec, job, entries)
                    for row in job.rows:
                        if row.cell is not None:
                            pipeline.FAILED.pop(row.cell.key, None)
                    pending.setdefault(node.uid, []).append((job, _spans(sorted(frames))))
        finally:
            evaluator.OVERRIDES = None
        count = sum(len(sets) for sets in pending.values())
        text = (f"{count} word set{'s' if count != 1 else ''} over frames {self.first}–{self.last}: the pinned "
                f"entr{'ies take' if count != 1 else 'y takes'} about {store._mb(size)} MB in the file" if count else
                f"No OpenAct words change over frames {self.first}–{self.last}: nothing to pin")
        run = new_uid()
        self.scene[RUN_KEY] = run  # inside the operator's undo step: undoing that step leaves the run dormant
        PLAYBACK, PLAYBACK_NOTE = Run(run, pending), ""
        if _pin_pending(guard=False) and not count:  # pins it writes now are in the operator's own undo step
            text += ", and the earlier pins are removed"
        if PLAYBACK is not None:
            text += ". They are pinned once the model has answered them"
        notes = [note for note in (PLAYBACK_NOTE, *(flag for flag, _keys in REBAKES.values())) if note]
        if notes:  # why nothing was pinned, or what it couldn't save, then what to re-bake
            text += f". {'; '.join(notes)}"
        if self.problems:
            problem, frame = next(iter(self.problems.items()))
            text += f". Words that can't be read are left out, e.g. at frame {frame}: {problem}"
        schedule(immediate=True)
        return text


def _pin_pending(guard=True):
    """Ask for what the waiting run still misses; once every word set is answered, pin them all and unpin every other
    entry, saved without the live set: nothing is evicted, and each node's next pass applies the limits (§13.3). With
    guard, the pins wait until an undo step can hold them without cutting the user's history (_hold()). The
    modifiers with bakes whose rules that changes are flagged (REBAKES). Returns the uids of the nodes whose pins
    changed."""
    global PLAYBACK, PLAYBACK_NOTE
    run = _waiting()
    if run is None:
        return set()
    ready, waiting = {}, False
    for uid, sets in run.sets.items():
        _tree, node = find_node(uid)
        if node is None:  # deleted meanwhile
            continue
        spec, node_store = SPECS[node.bl_idname], store.load(node)
        ready[uid] = (node_store, [])
        for job, frames in sets:
            entry, missing, failed = spec.resolve(job, node_store.entries)
            stop = f"Prepare for Playback stopped: '{node.name}' at frame {frames[0][0]}"
            if entry is None and not missing and not failed:  # Dials: other settings than its contexts agreed on
                PLAYBACK, PLAYBACK_NOTE = None, (f"{stop}: its settings there aren't the ones its outputs show. Keep "
                                                 f"the setting names and types the same over the frame range. Nothing "
                                                 f"was pinned")
                return set()
            if failed:  # not asked again in this run: a new run asks again
                PLAYBACK, PLAYBACK_NOTE = None, f"{stop}: {failed[0]}. Nothing was pinned: run it again to retry"
                return set()
            if entry is None:
                waiting = True
                _submit(dataclasses.replace(job, priority=_priority(spec, node, missing),
                                            generation=GENERATIONS.get(uid, 0)), missing)
            else:
                ready[uid][1].append((entry, frames))
    run.held = "" if waiting or not guard else _hold()
    if waiting or run.held:
        return set()
    PLAYBACK = None
    changed, errors, uses = set(), [], find_uses()
    for tree, node in iter_nodes():
        if tree.library is not None:
            continue
        node_store, pins = ready.get(node.uid) or (store.load(node), [])
        entries = node_store.entries
        before = {entry.words_key: entry.frames for entry in entries if entry.pinned}
        if not pins and not before:
            continue
        for entry in entries:
            entry.pinned, entry.frames = False, []
        for entry, frames in pins:
            entry.pinned, entry.frames = True, frames
            at = next((i for i, old in enumerate(entries) if old.gn == entry.gn), None)
            if at is None:
                entries.append(entry)
            else:  # in place: the entry of those words, or a table carrying its rows
                entries[at] = entry
        try:
            store.save(node, node_store)  # no live set: nothing is evicted (§13.3)
        except ValueError as error:  # a newer OpenAct's answers, or a table too large to store
            errors.append(f"'{node.name}': {error}")
            continue
        DUE.add(node.uid)
        changed.add(node.uid)
        after = {entry.words_key: frames for entry, frames in pins}
        frames = {frame for key in before.keys() ^ after.keys() for start, end in before.get(key, after.get(key))
                  for frame in range(start, end + 1)}  # where the pins that came or went decide the rules now
        # RNA can't tell a baked simulation or Bake node from an unbaked one, so every modifier with bakes is named
        baked = sorted({f"'{ctx.obj.name} > {ctx.mod.name}'" for ctx in uses.get(tree.as_pointer(), ())
                        if ctx.mod is not None and len(ctx.mod.bakes)})
        if frames and baked:
            REBAKES[node.uid] = (REBAKE.format(", ".join(baked), frames_text(_spans(sorted(frames)))), frozenset(after))
    PLAYBACK_NOTE = "; ".join(errors)
    return changed


def _waiting():
    """The run waiting for its answers, or None: none waits, or undoing its operator step left it dormant (a redo
    wakes it)."""
    run = PLAYBACK
    return run if run is not None and any(scene.get(RUN_KEY) == run.id for scene in bpy.data.scenes) else None


def rebake(uid, pins):
    """The node's re-bake flag while it holds the pins (entries) the flag was raised for, else None: an undo can take
    them away, and a redo bring them back."""
    flag = REBAKES.get(uid)
    return flag[0] if flag is not None and flag[1] == {entry.words_key for entry in pins} else None


def pinning():
    """How many word sets the waiting run pins once they are answered; 0 when none waits."""
    run = _waiting()
    return sum(len(sets) for sets in run.sets.values()) if run is not None else 0


def held():
    """Why the waiting run's pins still wait once every answer is in, for the sidebar; "" while answers are missing, or
    when none waits."""
    run = _waiting()
    return run.held if run is not None else ""


def cancel_playback(note="Prepare for Playback cancelled: nothing more was pinned"):
    """Stop the run waiting for its answers: nothing more is pinned. False when none waits."""
    global PLAYBACK, PLAYBACK_NOTE
    live = _waiting() is not None
    PLAYBACK = None  # a dormant one too: after another undo step, no redo can wake it
    if live:
        PLAYBACK_NOTE = note
    return live


def unpin(uid, words_key=""):
    """Unpin one of the node's entries (by words key), or with "" all its pins. They stay stored, saved without the
    live set, so nothing is evicted now: the node's next pass applies the limits (§13.3). A run waiting for its answers
    is cancelled, as it would replace these pins. Returns how many entries were unpinned."""
    tree, node = find_node(uid) if uid else (None, None)
    if node is None or tree.library is not None:
        return 0
    node_store = store.load(node)
    found = [entry for entry in node_store.entries if entry.pinned and words_key in ("", entry.words_key)]
    for entry in found:
        entry.pinned, entry.frames = False, []
    if found:
        store.save(node, node_store)  # no live set: nothing is evicted (§13.3)
        DUE.add(uid)
        cancel_playback("Prepare for Playback cancelled by Unpin: nothing more was pinned")
        schedule(immediate=True)
    return len(found)


def frames_text(spans):
    """[start, end] spans as the sidebar and the notes write them: "1–10, 21, 31–40"."""
    return ", ".join(str(start) if start == end else f"{start}–{end}" for start, end in spans)


def _window():
    """The window an undo step is pushed from: the context's, else the first one (a timer has no window)."""
    wm = getattr(bpy.context, "window_manager", None)
    return bpy.context.window or (wm.windows[0] if wm and wm.windows else None)


def _undo_push(message):
    """An undo step for what the pass wrote on its own, when there is a window to push it from."""
    window = _window()
    if window is None:
        return
    with bpy.context.temp_override(window=window):
        if bpy.ops.ed.undo_push.poll():
            bpy.ops.ed.undo_push(message=message)


def _modal_undo():
    """True while a window runs a modal operator that pushes an undo step when it ends (a transform, a stroke)."""
    wm = getattr(bpy.context, "window_manager", None)
    return wm is not None and any('UNDO' in op.bl_options for window in wm.windows for op in window.modal_operators)


def _hold():
    """Why pins a pass writes must wait, or "" when an undo step can hold them without cutting the user's history: a
    modal operator is mid-way (its own step would take them), the mode's undo steps hold only its own data (Edit Mode,
    Sculpt Mode, Texture Paint...), or there are steps to redo (a push would delete them)."""
    window = _window()
    if window is None:  # no undo step to push, and none to cut
        return ""
    if _modal_undo():
        return "they are pinned once the running tool finishes"
    with bpy.context.temp_override(window=window):
        if bpy.context.mode not in LAND_MODES:
            obj = bpy.context.active_object
            mode = obj.bl_rna.properties["mode"].enum_items[obj.mode].name if obj is not None else bpy.context.mode
            return f"they are pinned once you leave {mode}"
        if bpy.ops.ed.redo.poll():
            return "they are pinned once nothing is left to redo (redo it, or make any change)"
    return ""


def _read(node, spec, ctx):
    """(words, or None when there are none to ask about; the values they read, as evaluator.READS records them; why
    they can't be read, or "") of one context now."""
    evaluator.READS = reads = {}
    try:
        return spec.gn(node, ctx), reads, ""
    except Hint:
        return None, reads, ""
    except (Pending, Unresolvable) as error:
        return None, reads, _unread(node, error)
    finally:
        evaluator.READS = None


def _animation(owner, path):
    """(F-curve, why) of the property at `path` of the ID `owner`: (None, "") when nothing animates it, its F-curve
    when only the active action's F-curves do, else why only setting the frames tells its values."""
    data = getattr(owner, "animation_data", None)
    if data is None:
        return None, ""
    if data.drivers.find(path) is not None:
        return None, f"{path} of '{owner.name}' is driven"
    if len(data.nla_tracks):
        return None, f"'{owner.name}' has NLA tracks"
    if data.use_tweak_mode:
        return None, f"'{owner.name}' has an action in tweak mode"
    if data.action_blend_type != 'REPLACE' or data.action_influence < 1.0:
        return None, f"'{owner.name}' blends its action"
    # Action.fcurves is gone in 5.x: the F-curves are found through the action's slot for this ID
    bag = anim_utils.action_get_channelbag_for_slot(data.action, data.action_slot)
    curve = bag.fcurves.find(path) if bag is not None else None
    return (curve if curve is not None and not curve.mute else None), ""


def _as(value, number, limits):
    """An F-curve's number as Blender writes it into a property that holds `value`: a bool from 0.5 up; an int
    truncated (an integer property's F-curve rounds already), then clamped to limits, the hard range; a float
    clamped."""
    if isinstance(value, bool):
        return number >= 0.5
    low, high = limits
    return min(max(int(number) if isinstance(value, int) else number, low), high)


def _range(struct, name, custom):
    """The hard range Blender clamps an animated number to: a custom property's own (id_properties_ui()), else the RNA
    property's."""
    if custom:
        ui = struct.id_properties_ui(name).as_dict()
        return ui.get("min", -math.inf), ui.get("max", math.inf)
    prop = struct.bl_rna.properties[name]
    return prop.hard_min, prop.hard_max


def _spans(frames):
    """Sorted frames as the [start, end] runs of consecutive frames an entry's `frames` holds (§14.2)."""
    spans = []
    for frame in frames:
        if spans and frame == spans[-1][1] + 1:
            spans[-1][1] = frame
        else:
            spans.append([frame, frame])
    return spans


def _size(spec, job, entries):
    """The bytes the job's entry takes in the file (§14.7): the stored one's, else a stand-in's whose numbers have a
    typical length."""
    entry = next((entry for entry in entries if entry.words_key == job.words_key), None)
    if entry is None:
        rows = [dataclasses.replace(row, source="model", ask="", values={
            column: 0.123456789012345 if kind == "FLOAT" else 1 for column, kind in job.shape.columns})
            for row in job.rows]
        entry = spec.entry(job, rows)
    return len(store._dump(entry).encode("utf-8"))


def _unread(node, error):
    """Why a node's words or plan can't be read at some frame, for the report."""
    if isinstance(error, Unresolvable):
        return f"'{node.name}': {error}"
    return f"'{node.name}' waits for another OpenAct node's answer"


# ------------------------------------------------------------------ handlers & timer


def _rendering():
    return bpy.app.is_job_running('RENDER')


def _animation_playing():
    wm = getattr(bpy.context, "window_manager", None)
    return bool(wm) and any(w.screen and w.screen.is_animation_playing for w in wm.windows)


def _publish(event, value):
    if value:
        event.set()
    else:
        event.clear()


def tick():
    global _last_refresh, _dirty
    try:
        if server.pump():
            redraw()
        rendering, playing = _rendering(), _animation_playing()
        _publish(server.Worker.render_running, rendering)  # for the worker, which never calls bpy (§5.5)
        _publish(server.Worker.playing, playing)
        now = time.monotonic()
        if now - _last_refresh > 5.0 and (_has_nodes or server.STATE.status in ("unknown", "ready", "starting")):
            _last_refresh = now
            server.refresh()
        # node trees are left alone during playback and render jobs, and while Prepare for Playback sets other frames
        if not (rendering or playing or _stepping):
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
    global PLAYBACK, PLAYBACK_NOTE
    pipeline.cancel_all()  # asked for the file just closed; the chunk in flight still lands in MEMORY
    STATUS.clear()
    pipeline.FAILED.clear()
    PLANS.clear()
    LAST_SHOWN.clear()
    LIVE.clear()
    CONTEXTS.clear()
    NOTES.clear()
    DUE.clear()
    REBAKES.clear()
    PLAYBACK, PLAYBACK_NOTE = None, ""  # the closed file's run
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
