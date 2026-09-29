"""Keeps the OpenAct nodes answered.

Geometry Nodes cannot run Python while it evaluates (Blender has no script node), so an OpenAct
node is answered outside evaluation:

1. After a change, each OpenAct node's words are read from what the evaluation reported through
   the Warning channel (channel.py), or by the static evaluator (evaluator.py) for a context no
   evaluation reported, and planned into a Job per word set (specs/, pipeline.py). A Job is
   planned again only when the fingerprint of what it was planned from changes.
2. Its rows are answered by the validity rule: the node's stored answers while the words mean the
   same, then this session's answers, then the local model, asked in the background (server.py).
3. The answers are written into the node's table object, which its hidden node group reads, keyed
   by the hash of the words. Geometry Nodes finds the matching answer by itself, so one node group
   can be used on several objects with different prompts, answers switch instantly back and
   forth, and they are saved with the file (Blender 4.5+ even evaluates them without the add-on).
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
import os
import time
import traceback
import uuid
from collections import Counter
from dataclasses import dataclass, field

import bpy
from bpy.app.handlers import persistent
from bpy_extras import anim_utils

from . import channel, csvfmt, evaluator, inner, lists, pipeline, server, store, trust
from . import prefs as P
from . import tables
from .csvfmt import ERROR, INFO, WARNING, Problem
from .evaluator import Ctx, Hint, Pending, Unreadable, Unresolvable, find_uses, is_field
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
COLLIDE = ("The words '{}' and '{}' hash to the same key: only the first one's answer can show. Give one a different "
           "Key (Ctrl H shows it; expose it on the modifier)")  # the lookup finds an entry by its key (§8.2)
STALL = 60.0  # seconds with cells asked of a ready model and no chunk answered: the stalled-fill alert (§17.6)
STALLED = "No answers for {:g} s: see the log"
SLOW_MS, SLOW_FOR = 3 * pipeline.MS_PER_ROW, 60.0  # /health's ms_per_row above this for so long: the slow note (§17.3)
SLOW = "Running slower than expected: is the GPU busy (rendering, another app)?"
ICONS = {"hint": 'INFO', "paused": 'PAUSE', "starting": 'SORTTIME', "waiting": 'SORTTIME', "filling": 'SORTTIME',
         "complete": 'CHECKMARK', "warning": 'ERROR', "error": 'CANCEL', "linked": 'LINKED'}
INTERACTIVE = ("ask", "pick", "dials")  # asked at P0 whatever their size, as is any job of at most P0_CELLS (§5.5)
P0_CELLS = 8
DIFFERS = "CSV {} differs from the saved table ({})"  # a file edited since a table read from it was stored (R23)
RENDERING = "; rendering the saved answers"  # ... printed where passes leave the saved tables as they are (§7.8)
READ_LATER = "Its {} is read when the node evaluates"  # a word only Geometry Nodes computes, of an unreported context

STATUS = {}  # node uid -> NodeStatus
PLANS = {}  # _key() of a word set -> (fingerprint, Job, or the (Hint or Unresolvable, message) planning gave)
LAST_SHOWN = {}  # (node uid, ctx key) -> words key of the entry that context showed last, while it has one word set
LIVE = {}  # node uid -> words keys of the entries its contexts showed at its last pass
CONTEXTS = {}  # node uid -> the ctx keys of its contexts at the last pass: tells the pass when one goes away
REPORTS = {}  # node uid -> {(object pointer, modifier name): {words: triggered}}: the pass's channel.read() (§5.7)
SETS = {}  # node uid -> the _key()s of the word sets its last visit planned: tells a visit when a context's changed
DIRTY = set()  # uids of the nodes the next pass visits, when not all of them are dirty
GENERATIONS = {}  # node uid -> its generation, bumped when its words change: its older queued chunks are dropped
NOTES = {}  # node uid -> what its last save evicted or kept above the soft cap (NodeStore.note), for its status
DRIFT = {}  # node uid -> {path: note} of the CSVs that differed from its saved tables when the file loaded (R23)
REPORTED = set()  # CSV paths whose "differs" line was printed since the file loaded (§7.8)
RENDERED = set()  # (node uid, words) a background render compared since the file loaded

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
_quiet_since = time.monotonic()  # the stall clock: since a chunk was last answered, or cells were first asked
_slow_since = None  # since when /health's ms_per_row has been above SLOW_MS, cells asked all along
_opened = True  # no pass has run since the file loaded: the next one compares its files with the saved tables


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
    contexts: int = 0  # distinct live word sets (§8.1)
    problems: list = field(default_factory=list)  # csvfmt.Problem records: the errors, then the warnings
    uses: list = field(default_factory=list)  # (context label, Job, entry shown or None) of each planned context
    asking: frozenset = frozenset()  # keys of the cells the pass asks: sent, or sent once the starting model is ready

    @property
    def icon(self) -> str:
        return ICONS[self.state]


def _problem(severity, text):
    return Problem(severity, "", 0, "", text)


def _error(text):
    return NodeStatus("error", text, problems=[_problem(ERROR, text)])


# ----------------------------------------------------------------------------- plans


def _openact_output(node, out, ctx, depth):
    """Evaluator handler: the answer an OpenAct node shows for the words the static evaluator reads in this context."""
    if is_field(out):
        raise Unreadable(f"'{node.name}' output '{out.name}' is a field; words must be single values")
    spec = SPECS[node.bl_idname]
    outer, lists.HASHES = lists.HASHES, None  # its files are its own entries': not the reading node's
    try:
        job = _plan(spec, node, ctx)
    except Hint:
        return spec.fallback(out.name)
    finally:
        lists.HASHES = outer
    node_store = store.load(node)
    if node_store.schema > store.SCHEMA:  # its answers can't be read, so they never arrive
        raise Unresolvable(f"'{node.name}': {store.newer_message(node_store)}")
    entry, _missing, _failed = spec.resolve(job, node_store.entries, LAST_SHOWN.get((node.uid, ctx.key()), ""))
    if entry is None:
        raise Pending()
    try:
        return spec.output(entry, out.name)
    except KeyError:
        raise Unreadable(f"'{node.name}' output '{out.name}' can't be a word") from None


evaluator.HANDLERS.update(dict.fromkeys(OPENACT_TYPES, _openact_output))


def _plan(spec, node, ctx, words=None):
    """The node's Job in ctx with its Overrides CSV applied, planned again only when its fingerprint changed (§5.3).
    words: a word tuple an evaluation reported for the node (§5.7), which its String word sockets read instead of
    the static evaluator's; None plans its static words. Raises Hint, Pending or Unresolvable."""
    global _plans_made
    if words is not None:
        _checked(spec, node, ctx)
    with reading(spec, node, words):
        if evaluator.READS is not None or evaluator.OVERRIDES is not None:  # words read at another frame (FrameWords)
            return spec.locked(node, ctx, spec.plan(node, ctx))  # the plan cache and generations are the live words'
        key = _key(node.uid, ctx, words)
        if _pass_plans is not None:
            _pass_plans.add(key)
        fingerprint = spec.fingerprint(node, ctx)
        cached = PLANS.get(key)
        if cached is None or cached[0] != fingerprint:
            if cached is not None:  # new words: the chunks still queued for the node ask about the old ones
                _bump(node.uid)
            try:
                planned = _planned(spec, node, ctx)
            except (Hint, Unresolvable) as error:  # kept as a message: the evaluator adds its hops to a raised error
                planned = (type(error), str(error))
            cached = PLANS[key] = (fingerprint, planned)
            _plans_made += 1
    if isinstance(cached[1], tuple):
        kind, message = cached[1]
        raise kind(message)
    return cached[1]


def _key(uid, ctx, words=None):
    """The plan cache's key of a word set: its node and context, and the words when an evaluation reported them."""
    return (uid, ctx.key()) if words is None else (uid, ctx.key(), words)


@contextlib.contextmanager
def reading(spec, node, words):
    """Within it, the node's String word sockets read the reported word tuple `words` (evaluator.WORDS, §5.7), in
    spec.words() order, whatever is linked to them; None: the static evaluator reads them."""
    found = None
    if words is not None:
        found = {}
        for name, word in zip(spec.words(node), words):
            socket = next((s for s in node.inputs if s.name == name and s.type == 'STRING'), None)
            if socket is not None:  # e.g. Items under a Collection source: reported as ""
                found[(node.as_pointer(), socket.identifier)] = word
    outer, evaluator.WORDS = evaluator.WORDS, found
    try:
        yield
    finally:
        evaluator.WORDS = outer


def _checked(spec, node, ctx):
    """Check the words an evaluation reported for the node in ctx before they are planned (§5.7). The static evaluator
    reads them too, as 1.0 did: when they wait for another OpenAct node's answer it raises Pending, and when it meets
    an error (another node's blocked or broken file, a missing text file, an invalid link) it raises that
    Unresolvable, as Geometry Nodes then built the reported words from fallbacks, which a pass never asks about (§5.3,
    chaining). Words only Geometry Nodes computes (Unreadable) are its report's. In a file that isn't trusted, raises
    Unresolvable when an Import Text node may have read them from outside its folder (_outside(), §16.1)."""
    if not trust.is_trusted() and (blocked := _outside(spec, node, ctx)):
        raise Unresolvable(blocked)
    try:
        with reading(spec, node, None):
            spec.gn(node, ctx)
    except (Hint, Unreadable):  # only Geometry Nodes computes them: its report is all there is
        pass


def _outside(spec, node, ctx):
    """In a file that isn't trusted, why the node's reported words in ctx may hold a file outside its folder, else "":
    the blocked read of an Import Text node that feeds its String words (trust.blocked()), or one whose Path the static
    evaluator can't read. Geometry Nodes reads such a file itself, but OpenAct never stores its text (§16.1). Text never
    travels through geometry (no attribute holds a string), so Geometry sockets are not followed."""
    stack = [(socket, ctx) for socket in node.inputs if socket.name in spec.words(node) and socket.type == 'STRING']
    seen = set()
    while stack:
        socket, at = stack.pop()
        if (socket.as_pointer(), at.key()) in seen:
            continue
        seen.add((socket.as_pointer(), at.key()))
        if len(seen) > 4096:
            return f"{trust.BLOCKED} the words of '{node.name}': too many nodes feed them to check their files"
        for link in socket.links:
            if link.is_muted or socket.type == 'GEOMETRY':
                continue
            up, out = link.from_node, link.from_socket
            if up.bl_idname == "GeometryNodeImportText" and not up.mute:
                try:
                    path = evaluator.as_text(evaluator.eval_input(evaluator._in(up, "Path"), at))
                    message = path and trust.blocked(evaluator.import_path(path, up))
                except (Hint, Pending, Unresolvable):
                    message = f"{trust.BLOCKED} the text file of '{up.name}': its Path can't be checked"
                if message:
                    return message
            elif up.bl_idname == "NodeGroupInput":
                if at.path:
                    stack += [(s, at.pop()) for s in at.path[-1].inputs if s.identifier == out.identifier]
            elif up.bl_idname == "GeometryNodeGroup" and up.node_tree is not None:
                stack += [(s, at.push(up)) for n in up.node_tree.nodes if n.bl_idname == "NodeGroupOutput"
                          and n.is_active_output for s in n.inputs if s.identifier == out.identifier]
            else:
                stack += [(s, at) for s in up.inputs]
    return ""


def plan_words(node, ctx, words):
    """The Job of a word tuple an evaluation reported for the node in ctx (§5.7), with the node's Overrides CSV
    applied: for the playback path, through the pass's plan cache, so a word set is planned once for both, and a
    changed fingerprint bumps the node's generation as in a pass (GENERATIONS; submit at GENERATIONS[uid]). Raises
    Hint, Pending or Unresolvable. Main thread, outside evaluation."""
    return _plan(SPECS[node.bl_idname], node, ctx, tuple(words))


def _planned(spec, node, ctx):
    """spec.plan() with the node's Overrides CSV applied, holding the hashes of the CSVs it read, which its entries
    store (§14.2). Raises Hint, Pending or Unresolvable."""
    outer, lists.HASHES = lists.HASHES, {}
    try:
        job = spec.locked(node, ctx, spec.plan(node, ctx))
        job.csv = lists.HASHES
        return job
    finally:
        lists.HASHES = outer


def _bump(uid):
    """A new generation for the node, whose queued chunks of older ones are dropped (§5.5)."""
    GENERATIONS[uid] = GENERATIONS.get(uid, 0) + 1
    pipeline.cancel(uid, below=GENERATIONS[uid])


def _jobs(node):
    """(ctx, Job) for each word set of the node that has a question, as a pass would plan them now: the words its
    evaluations report (§5.7), else its static words."""
    spec, tree = SPECS[node.bl_idname], node.id_data
    contexts = find_uses().get(tree.as_pointer()) or [Ctx(None, None, (), tree)]
    found = channel.read(list(iter_nodes()), contexts, channel.evaluated()).get(node.uid)
    return [(ctx, job) for ctx, _words, job in _scan(spec, node, _sets(spec, node, contexts, found))
            if not isinstance(job, Exception)]


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
    """(tree, node) of each node of these types outside inner groups: by default the OpenAct nodes, which the pass
    visits."""
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


def _env_opt_in(environ=os.environ):
    """OPENACT_BACKGROUND_PASSES as set before Blender starts: 1, true or yes, spaces ignored (Windows cmd's
    `set OPENACT_BACKGROUND_PASSES=1 && blender ...` keeps the one before &&)."""
    return environ.get("OPENACT_BACKGROUND_PASSES", "").strip().lower() in ("1", "true", "yes")


_background_passes = _env_opt_in()  # headless authoring runs passes (§7.8)


def allow_background_passes(flag=True):
    """Let passes run in background mode (blender -b) for headless authoring: scripts that set words, fill the tables
    and save. Unless this or OPENACT_BACKGROUND_PASSES=1 (read at import) opts in, a background session renders the
    saved answers only (§7.8). Call it after enabling the add-on."""
    global _background_passes
    _background_passes = bool(flag)
    schedule(immediate=True)


def _saved_only():
    """True in background mode unless headless authoring opted in: passes and the operators that plan and write the
    tables leave the saved answers and payload as they are (§7.8). A render job holds them too: tick() starts no pass
    then, and those operators' polls refuse."""
    return bpy.app.background and not _background_passes


def run_pass(everything=True):
    """Bring the OpenAct nodes up to date: every node, or with everything=False only the dirty ones. While
    _saved_only(), it only reports the CSVs that differ from the saved tables. Main thread only, and never during a
    render job (tick())."""
    global _dirty, _dirty_all, _has_nodes, _last_pass, _group_count, _pass_plans, passes, _opened
    if _saved_only():  # a farm renders exactly the saved answers and payload (§7.8)
        _dirty, _last_pass = False, time.monotonic()  # what else is due (DIRTY, WATCH) waits for a pass that writes
        _report()
        return
    # The evaluation the reports describe (§5.7): it evaluates any change since, whose depsgraph update then schedules
    # nothing more, as this pass covers it
    depsgraph = channel.evaluated()
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
        REPORTS.clear()
        SETS.clear()
        NOTES.clear()
        DRIFT.clear()
        WATCH.retain(())
        tables.collect_garbage(nodes)
        return
    uses = find_uses()
    trees = dict.fromkeys(tree.as_pointer() for tree, _node in nodes)  # the trees that hold OpenAct nodes
    REPORTS.clear()  # before ensure_owned(): read() tells a copy's reports from its original's by the groups they share
    REPORTS.update(channel.read(nodes, [ctx for key in trees for ctx in uses.get(key, ())], depsgraph))
    ensure_owned(nodes)
    tables.collect_garbage(nodes)  # after ensure_owned(), which gives copies their own table objects
    plans_made, stored = _plans_made, False
    evaluator.MEMO = {}  # nodes that read the same upstream words in a context share one read
    csvfmt.IDENTITIES = {}  # every fingerprint resolves its files and asks for the trust state: once per pass (§3.2)
    planned = _pass_plans = set()
    visited = set()
    try:
        trust.PASS = trust.is_trusted()
        for tree, node in nodes:
            if not (visit_all or node.uid in dirty):
                continue
            visited.add(node.uid)
            try:
                stored |= _process(tree, node, uses.get(tree.as_pointer()))
            except Exception as error:
                traceback.print_exc()
                STATUS[node.uid] = _error(f"Internal error: {error}")
    finally:
        evaluator.MEMO = None  # keyed by pointers: after the pass, nodes and data may be freed or edited
        csvfmt.IDENTITIES = trust.PASS = None  # files, the .blend and the trusted lists change between passes
        _pass_plans = None
    _opened = False  # later edits are this session's own: no note
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
    # contexts find_uses() lacks (through a group node in a tree no modifier uses), planned again every pass if dropped.
    # A reported word set a visited node no longer plans goes too: its context's words changed
    for cache in (PLANS, LAST_SHOWN):
        for key in [key for key in cache if key not in planned and (
                key[1] not in contexts.get(key[0], ()) or len(key) > 2 and key[0] in visited)]:
            del cache[key]
    for uid in lost:
        _bump(uid)
        schedule(immediate=True)
    for uid in [uid for uid in LIVE if uid not in contexts]:  # deleted nodes: their queued chunks go (undo asks again)
        pipeline.cancel(uid, below=math.inf)
    # nodes that disappeared: undo and deleting a node group call no node's free()
    for cache in (LIVE, STATUS, SETS, NOTES, DRIFT):
        for uid in [uid for uid in cache if uid not in contexts]:
            del cache[uid]
    WATCH.retain(contexts)  # stop watching the files of deleted nodes
    redraw()


def _process(tree, node, contexts):
    """Plan and answer the node's word sets, store new answers, rebuild its group and set its status. True if it stored
    any. A context plans each word tuple its evaluation reported (REPORTS, §5.7), else its static words (§5.3)."""
    if tree.library is not None:  # read-only: the library file holds this node's answers
        STATUS[node.uid] = NodeStatus("linked", f"Linked from {tree.library.name}: answers come from that file")
        return False
    node_store = store.load(node)
    if node_store.schema > store.SCHEMA:  # save() would refuse any answer, and the group holds the newer ones
        STATUS[node.uid] = _error(store.newer_message(node_store))
        return False
    spec = SPECS[node.bl_idname]
    wanted = spec.interface(node)
    if inner.lacks(node, wanted):  # saved before the node gained sockets, e.g. Geometry and Trigger (§6.0)
        inner.sync_interface(node, wanted)
    entries = node_store.entries
    contexts = contexts or [Ctx(None, None, (), tree)]
    jobs, shown, live = {}, [], []  # words -> (Job, ctx), one per distinct words; the entries shown; live gns
    # Stored entries whose words mean something else now stay stored for their rows, but leave the graph (§5.4):
    hidden = {}  # their (gn, words key) -> the partial table shown in their place, or None while nothing is known
    instead = {}  # (uid, ctx key) -> the words key of the partial table that context shows in a hidden entry's place
    problems, warnings, failed = [], [], []
    uses, resolved, asked = [], {}, set()  # (label, Job) per planned word set; words -> entry shown; cells missing
    sent = set()  # the missing cells the model is asked: sent now, or once the model that is starting is ready
    upstream = waiting = changed = False
    generation = GENERATIONS.get(node.uid, 0)
    planned = _scan(spec, node, _sets(spec, node, contexts, REPORTS.get(node.uid)))
    _moved(node.uid, contexts, planned, generation)
    counts = Counter(ctx.key() for ctx, _words, _job in planned)  # several word sets in a context: a zone's (§5.7)
    # §6.0: the sockets follow the live jobs (Rules, Classify, Dials). Jobs that disagree answer nothing, so the last
    # good table stays. A context without a job (Pending; Unresolvable: a CSV with errors, too many items...; or a Hint
    # whose words still name a stored entry: an Import CSV node without a Path, an empty collection) still shows its
    # last table, whose shape is unknown. While there is one, agree() keeps the sockets, and jobs that don't fit them
    # wait (None) until that context plans again or goes away; jobs that fit them are answered
    gns = {entry.gn for entry in entries}
    hinted = {(ctx.key(), words): gn for ctx, words, job in planned
              if isinstance(job, Hint) and (gn := _gn(spec, node, ctx, words)) in gns}
    hints = [str(job) for ctx, words, job in planned if (ctx.key(), words) in hinted]  # first: they hold the node
    unknown = bool(hinted) or any(isinstance(job, (Pending, Unresolvable)) for _ctx, _words, job in planned)
    conflict = spec.agree(node, [(ctx, job) for ctx, _words, job in planned if not isinstance(job, Exception)],
                          unknown, entries) if hasattr(spec, "agree") else ""
    held = conflict is None
    if conflict:
        problems.append(conflict)
    for ctx, words, job in planned:
        key = (node.uid, ctx.key())
        # A context with several word sets (a zone, §5.7) keeps no last-shown entry: they have no identity from one
        # evaluation to the next, so one word set would show another's rows while refilling
        last_shown = LAST_SHOWN if counts[ctx.key()] == 1 else {}
        if isinstance(job, Pending):  # its words wait for another node's answer: the entry it showed last stays live
            upstream = True
            last = next((entry for entry in entries if entry.words_key == last_shown.get(key)), None)
            if last is not None and last.gn not in live:
                live.append(last.gn)
            continue
        if isinstance(job, Hint):
            gn = hinted.get((ctx.key(), words))
            if gn is None:
                hints.append(str(job))
            elif gn not in live:  # the stored entry its words name still shows: kept, as for an error
                live.append(gn)
            continue
        if isinstance(job, Unresolvable):  # e.g. a CSV with errors: its words stay live, so the last good table shows
            problems.append(str(job))
            _live(live, spec, node, ctx, words)
            continue
        uses.append((_label(ctx, words, counts[ctx.key()]), job))
        warnings += job.problems
        problems += job.errors  # rows in error in a table that still shows, e.g. cells no outcome applies to
        if job.gn not in live:
            live.append(job.gn)
        if conflict or held:  # nothing is answered: the sidebar shows the stored entry resolve() keeps, if any
            found = next((entry for entry in entries if entry.gn == job.gn), None)
            kept = spec.resolve(job, entries, last_shown.get(key, ""), partial=True)[0] if found is not None else None
            if kept is not None and kept.words_key == found.words_key:  # it, its table so far or a refreshed copy
                resolved[job.gn] = found
            continue
        other = jobs.setdefault(job.gn, (job, ctx))
        if other[0] is not job and other[0].words_key != job.words_key:
            problems.append(CLASH.format(other[1].label(), ctx.label()))
            continue
        found = next((entry for entry in entries if entry.gn == job.gn), None)
        if found is not None:  # after a load, a context shows its stored entry
            last_shown.setdefault(key, found.words_key)
        if other[0] is not job:  # the same words and meaning as an earlier context: the graph shows it that entry
            if any(entry is found for entry in shown):
                last_shown[key] = found.words_key
            continue
        # While cells are missing, the table so far shows: stale-while-refilling (§5.4)
        entry, missing, errors = spec.resolve(job, entries, last_shown.get(key, ""), partial=True)
        if entry is not None and found is not None and entry is not found and dataclasses.replace(
                entry, t=found.t) == found:
            entry = found  # the partial table as stored already
        if found is not None and entry is found:  # the stored table holds
            if found.csv != job.csv:  # edited where no table follows it: loads and farms compare with it now (§14.2)
                entry = dataclasses.replace(found, csv=dict(job.csv))
        elif _opened and (drift := _differs(job, found)):  # edited elsewhere: re-baked as usual, with a note (R23)
            DRIFT.setdefault(node.uid, {}).update((path, DIFFERS.format(path, kinds)) for path, kinds in drift)
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
            last_shown[key] = entry.words_key
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
    if changed:
        try:
            store.save(node, node_store, frozenset(live))  # live contexts are never evicted
            NOTES[node.uid] = node_store.note
        except ValueError as error:  # a damaged or too large entry: the node keeps what it has
            problems.append(str(error))
            changed = False
            node_store = store.load(node)
    warnings += orphaned([job for job, _ctx in jobs.values()])  # locks no context's rows use (§7.6)
    # An unstored partial table has the gn of the entry it shows instead of, so it takes that entry's place in the table
    graph = node_store
    if hidden:  # a hidden entry leaves the table, or leaves its place to the partial table shown instead
        graph = store.NodeStore([entry for entry in (hidden.get((stored.gn, stored.words_key), stored)
                                                     for stored in node_store.entries) if entry is not None])
    problems += [COLLIDE.format(_named(kept), _named(other)) for kept, other in inner.collisions(graph.entries, live)]
    keys = [(node.uid, ctx.key()) for ctx, _words, _job in planned]  # one per word set: a zone's several show none
    last = [instead.get(key, LAST_SHOWN.get(key)) for key in keys]  # a miss shows what the table showed last
    default = inner.k_default_key(last, blocked=bool(problems or hints or failed))
    # The table object is written only when its rows change: a write re-evaluates every modifier that reads it, whose
    # depsgraph update schedules a pass, which then finds them unchanged and writes nothing
    if not inner.update_values(node, graph, live, default):  # values only (§8.8)
        inner.build(node, graph, live, default)
    warnings += spec.node_warnings(node)  # known once built, e.g. a renamed output that lost its link
    faults, warnings = list(dict.fromkeys(problems + failed)), list(dict.fromkeys(warnings))  # contexts share words
    state, text, eta = _status(spec, node.uid, jobs, shown, faults, hints, upstream, waiting, warnings)
    STATUS[node.uid] = NodeStatus(
        state, text, eta_s=eta, contexts=len(live), uses=[(label, job, resolved.get(job.gn)) for label, job in uses],
        problems=[_problem(ERROR, fault) for fault in faults] + [_problem(WARNING, w) for w in warnings]
        + [_problem(INFO, note) for note in (NOTES.get(node.uid), *DRIFT.get(node.uid, {}).values()) if note],
        asking=frozenset(sent), **_counts(spec, jobs, resolved, asked))
    return changed


def _live(live, spec, node, ctx, words=None):
    """Add the words of a word set to the live gns, when they can be read."""
    gn = _gn(spec, node, ctx, words)
    if gn is not None and gn not in live:
        live.append(gn)


def _gn(spec, node, ctx, words=None):
    """The words of a word set: the reported ones, else the ones the static evaluator reads in ctx now, or None when
    it can't."""
    if words is not None:
        return words
    try:
        return spec.gn(node, ctx)
    except (Hint, Pending, Unresolvable):
        return None


def _named(gn):
    """A word tuple as a status line names it: its words, empty ones left out, each cut to 40 characters."""
    return " · ".join(word if len(word) <= 40 else word[:39] + "…" for word in gn if word)


def _sets(spec, node, contexts, found):
    """(ctx, words) of each word set the node plans (§5.3): each word tuple the context's modifier reported for the
    node (found: its REPORTS, §5.7), sorted, else (ctx, None), its static words, for a context no evaluation reported
    (a tree no modifier uses, an unused or muted node, an object the viewport doesn't evaluate). Group paths that share
    a modifier share its report: a tuple goes to the first path whose static words equal it, else to the first path,
    so a path whose words merged into another's plans none. A tuple of another length (a group laid out for other
    words) is left out."""
    size, paths, owned = len(spec.words(node)), {}, {}
    for ctx in contexts:
        paths.setdefault(ctx.key()[:2], []).append(ctx)
    reported = {at: sorted(words for words in (found or {}).get(at, ()) if len(words) == size) for at in paths}
    for at, shared in paths.items():
        statics = [_gn(spec, node, ctx) for ctx in shared] if len(shared) > 1 and reported[at] else []
        for words in reported[at]:
            owner = next((ctx for ctx, gn in zip(shared, statics) if gn == words), shared[0])
            owned.setdefault(owner.key(), []).append(words)
    return [(ctx, words) for ctx in contexts
            for words in (owned.get(ctx.key(), ()) if reported[ctx.key()[:2]] else (None,))]


def _moved(uid, contexts, planned, generation):
    """Bump the node's generation when a context it still has lost a word set its last visit planned: its words
    changed, so its queued chunks ask about old words (§5.5). Once a visit: _plan() bumps for a changed fingerprint of
    the same word set, and a pass for a context that went away."""
    now = {_key(uid, ctx, words) for ctx, words, _job in planned}
    present = {ctx.key() for ctx in contexts}
    gone = any(key[1] in present and key not in now for key in SETS.get(uid, ()))
    SETS[uid] = now
    if gone and GENERATIONS.get(uid, 0) == generation:
        _bump(uid)


def _label(ctx, words, count):
    """A word set's label in the sidebar: its context's, followed by its first word when the context has several (a
    zone's, §5.7)."""
    if count < 2 or not words:
        return ctx.label()
    word = next((word.strip() for word in words if word.strip()), "")
    return f"{ctx.label()}: {word if len(word) <= 32 else word[:31] + '…'}"


def _scan(spec, node, sets):
    """(ctx, words, its Job, or the Hint, Pending or Unresolvable that planning raised) of each word set: a visit plans
    each once, for agree() and for answering. A word of an unreported context that only Geometry Nodes computes
    (Unreadable: Scene Time, a simulation's state...) is no error: its first evaluation reports it (§5.7), so it shows
    a hint."""
    planned = []
    for ctx, words in sets:
        try:
            planned.append((ctx, words, _plan(spec, node, ctx, words)))
        except (Hint, Pending, Unresolvable) as error:
            if words is None and isinstance(error, Unreadable) and error.socket in spec.words(node):
                error = Hint(READ_LATER.format(error.socket))
            planned.append((ctx, words, error))
    return planned


def _differs(job, entry):
    """(path, "text", "numbers" or "text/numbers") of each CSV the job read whose hashes differ from the ones the
    entry was stored with (§14.2). A file the entry holds no hashes of (stored before they were kept) isn't
    compared."""
    found = []
    for path, now in job.csv.items():
        saved = entry.csv.get(path) if entry is not None else None
        if not isinstance(saved, list) or len(saved) != 2:
            continue
        kinds = [kind for kind, new, old in zip(("text", "numbers"), now, saved) if new != old]
        if kinds:
            found.append((path, "/".join(kinds)))
    return found


def _report(scene=None):
    """Print once per file each CSV that differs from the one the stored table of its words was read from, for a pass
    or a background render, which leave the saved tables as they are (§7.8). Only where the table its words plan now
    differs from the stored one: an edit that changes no table reports nothing.

    The words are the ones each context has now or, given a render's scene, at the frame it renders: a background
    render sets its frame without evaluating the original data, so they are read through the F-curves (FrameWords),
    one context at a time. A context whose words only setting the frame would tell (a driver, an NLA-animated or
    blended owner, Time Stretching, a keyframed menu socket) is read as the file holds it; the others still at the
    frame. Each context is planned afresh, a render's once per word set; nothing is asked, stored or built. Linked
    nodes, whose answers are their library's, are left out."""
    reader = FrameWords(scene or bpy.context.scene)
    if scene is not None:
        reader.first = reader.last = scene.frame_current
    for node, spec, ctx in list(reader.contexts):
        entries = store.load(node).entries
        if not entries:
            continue
        values = None
        if scene is not None:  # its own reads only: another context's driver leaves this one's frame readable
            reader.contexts = [(node, spec, ctx)]
            values = reader.values.get(reader.first) if reader.read_curves() else None
        evaluator.OVERRIDES = values
        try:
            if scene is not None:
                words = _read(node, spec, ctx)[0]
                if words is None or (node.uid, words) in RENDERED:
                    continue
                RENDERED.add((node.uid, words))
            job = _planned(spec, node, ctx)
        except (Hint, Pending, Unresolvable):
            continue
        finally:
            evaluator.OVERRIDES = None
        found = next((entry for entry in entries if entry.gn == job.gn), None)
        if found is None or spec.resolve(job, entries, partial=True)[0] is found:  # the files give the stored table
            continue
        for path, kinds in _differs(job, found):
            if path not in REPORTED:
                REPORTED.add(path)
                print(DIFFERS.format(path, kinds) + RENDERING, flush=True)


def _submit(job, missing):
    """Ask for the missing cells once the model is ready, starting it if the preferences allow. True when they are
    asked: sent now, or sent by the pass that runs once the model is ready, as a start is under way, queued or begun
    here."""
    state, _message = server.status()
    if state == "ready":
        pipeline.submit(job, missing, job.priority, on_rows=_answered)
        return True
    if state in ("stopped", "unknown", "installing") and P.get_prefs().auto_start and not server.STATE.detached:
        return server.start()  # asked by the pass that runs when the server is ready; an install starts it as it ends
    return state == "starting" or server.STATE.start_queued


def _answered(_job):
    """A chunk came back, answered or failed: the stall clock starts again, and a pass shows the answers."""
    global _quiet_since
    _quiet_since = time.monotonic()
    schedule(immediate=True)


def stalled():
    """True when cells are asked of a ready model and no chunk has come back for STALL seconds (§17.6)."""
    return bool(pipeline.PENDING) and server.is_ready() and time.monotonic() - _quiet_since >= STALL


def slow_note():
    """The Model box's note once /health's ms_per_row has been above SLOW_MS for SLOW_FOR seconds while cells were
    asked (§17.3), else ""."""
    return SLOW if _slow_since is not None and time.monotonic() - _slow_since >= SLOW_FOR else ""


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
        if state in ("stopped", "unknown") and (not P.get_prefs().auto_start or server.STATE.detached):
            return "paused", "Model server is stopped (OpenAct tab)", None
        if upstream and not waiting:
            return "waiting", "Waiting for another OpenAct node...", None
        if stalled():  # its chunks stay queued
            return "warning", STALLED.format(STALL), None
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


def _counts(spec, jobs, resolved, asked):
    """The row counts of the node's tables (§17.1): by the source of each row of the entry a table shows, with its
    unsure answers (spec.unsure()), else of its plan's rows, whose cells are asked (pending), failed (errors) or
    answered this session."""
    counts = dict.fromkeys(("rows", "answered", "pending", "stale", "unsure", "locked", "code", "errors"), 0)
    letters = {"m": "answered", "p": "pending", "s": "stale", "l": "locked", "c": "code", "e": "errors"}
    for gn, (job, _ctx) in jobs.items():
        counts["rows"] += len(job.rows)
        entry = resolved.get(gn)
        if entry is not None:
            for letter in entry.rows.src:
                counts[letters.get(letter, "answered")] += 1
            counts["unsure"] += spec.unsure(entry)
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
    words = {job.gn for job in jobs}
    node_store.entries = [entry for entry in node_store.entries if entry.gn not in words]
    store.save(node, node_store)  # no live set: nothing is evicted (§13.3)
    pipeline.forget(row.cell.key for job in jobs for row in job.rows if row.cell is not None)
    for key in [key for key in LAST_SHOWN if key[0] == node.uid]:
        del LAST_SHOWN[key]
    # as the next pass writes it: the live set, and no default now that nothing was shown; values only when it can
    live = list(dict.fromkeys(job.gn for job in jobs))
    if not inner.update_values(node, node_store, live, None):
        inner.build(node, node_store, live, None)
    schedule(immediate=True)


def redraw():
    wm = getattr(bpy.context, "window_manager", None)
    if wm is None:
        return
    for window in wm.windows:
        for area in window.screen.areas:
            if area.type in ('NODE_EDITOR', 'PREFERENCES'):
                area.tag_redraw()


# ----------------------------------------------------------- words at other frames (§7.8)


class FrameWords:
    """The words each context of each OpenAct node has at other frames of the scene's frame range, read through the
    F-curves of the values the words read, with no scene evaluation (read_curves()): a background render sets its
    frame without evaluating the original data, so _report() compares the words it renders this way."""

    def __init__(self, scene):
        self.scene, self.first, self.last = scene, scene.frame_start, scene.frame_end
        uses = find_uses()
        self.contexts = [(node, SPECS[node.bl_idname], ctx) for tree, node in iter_nodes()
                         if tree.library is None and node.uid  # linked nodes stay as saved
                         for ctx in uses.get(tree.as_pointer()) or [Ctx(None, None, (), tree)]]
        self.seen = {}  # (uid, ctx key) -> {words: the frames the context has them at}
        self.values = {}  # frame -> the values the words read there (evaluator.OVERRIDES)
        self.problems = {}  # why some words can't be read -> the first frame
        self.reason = ""  # why only setting the frames would tell some words, found by read_curves()

    def _record(self, frame, node, ctx, words, problem):
        if words is not None:
            self.seen.setdefault((node.uid, ctx.key()), {}).setdefault(words, []).append(frame)
        if problem:
            self.problems.setdefault(problem, frame)

    def read_curves(self):
        """Read the words through the F-curves, purely in Python: each context's words are read at the first frame,
        and at a later one only when they read a value an F-curve animates, which then takes that F-curve's value there
        (evaluate() applies its modifiers). False as soon as a value only setting the frame would tell: self.reason
        says why."""
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
                                self.reason = why
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


def _unread(node, error):
    """Why a node's words can't be read at some frame (FrameWords.problems)."""
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
    global _last_refresh, _dirty, _quiet_since, _slow_since
    try:
        if server.pump():
            redraw()
        rendering, playing = _rendering(), _animation_playing()
        _publish(server.Worker.render_running, rendering)  # for the worker, which never calls bpy (§5.5)
        _publish(server.Worker.playing, playing)
        now = time.monotonic()
        asking = server.is_ready() and bool(pipeline.PENDING) and not rendering  # the worker sends: rows get scored
        if not asking:  # both clocks run only then: an idle server's ms_per_row is its last fill's
            _quiet_since = now
        ms_per_row = server.STATE.health.get("ms_per_row") if asking else None
        if ms_per_row is None or ms_per_row <= SLOW_MS:
            _slow_since = None
        elif _slow_since is None:
            _slow_since = now
        if now - _last_refresh > 5.0 and (_has_nodes or server.STATE.status in ("unknown", "ready", "starting")):
            _last_refresh = now
            server.refresh()
        if not (rendering or playing):  # node trees are left alone during playback and render jobs
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
def on_frame_change(*_args):
    # A new frame may give nodes new words (§5.7), and sends no depsgraph update: a pass reads them, once the timeline
    # stops (tick())
    if _has_nodes:
        schedule()


@persistent
def on_load_post(*_args):
    global _opened
    pipeline.cancel_all()  # asked for the file just closed; the chunk in flight still lands in MEMORY
    STATUS.clear()
    pipeline.FAILED.clear()
    PLANS.clear()
    LAST_SHOWN.clear()
    LIVE.clear()
    CONTEXTS.clear()
    REPORTS.clear()
    SETS.clear()
    NOTES.clear()
    DRIFT.clear()
    REPORTED.clear()
    RENDERED.clear()
    _opened = True
    schedule(immediate=True)


@persistent
def on_render_pre(scene, *_args):
    # blender -b -f / -a runs no timer, so no pass: a farm render compares the words of each frame it renders here
    # (§7.8). _saved_only() means background mode, where the render runs on the main thread
    if _saved_only():
        _report(scene)


@persistent
def on_undo_redo(*_args):
    schedule(immediate=True)


def on_server_status():
    schedule(immediate=True)
    redraw()


def _handlers():
    h = bpy.app.handlers
    return ((h.depsgraph_update_post, on_depsgraph_update), (h.frame_change_post, on_frame_change),
            (h.load_post, on_load_post), (h.undo_post, on_undo_redo), (h.redo_post, on_undo_redo),
            (h.render_pre, on_render_pre))


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
