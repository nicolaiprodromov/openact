"""The CSV file watcher (design 7.8): notices edits to the files OpenAct nodes read.

Blender sends no event when a file on disk changes, so the engine's tick() calls WATCH.poll(): at most every 0.5 s
it stats up to 32 watched files, round-robin. A changed stamp re-reads the file through the shared cache
(csvfmt.load(), so the next pass parses nothing again) and compares its hashes with the last good read:

- "text": text_hash changed; a pass re-asks the cells whose words changed.
- "numbers": only num_hash changed; a values-only update, 0 requests.
- "error", "missing": the node shows the problem and keeps its last good table. The last good read is kept too,
  so a half-saved file never blanks a table.
- "restored": readable again, with the content of the last good read.

A save without a real change reports nothing. Owners (node uids) of a changed file are marked dirty (take_dirty())
and the listeners get the changes. Pure Python apart from register(), so the pure suites can run it.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from . import csvfmt

INTERVAL = 0.5  # seconds between polls
BATCH = 32  # files stat'ed per poll


@dataclass(frozen=True)
class Change:
    path: str
    kind: str  # "text" | "numbers" | "error" | "missing" | "restored"
    owners: frozenset
    message: str = ""  # the problem, for "error" and "missing"


@dataclass
class _File:
    path: str
    stamp: tuple | None  # (st_mtime_ns, st_size) last seen; None while missing
    reads: dict = field(default_factory=dict)  # (schema, delimiter) -> the last good CsvDoc, None before one
    owners: set = field(default_factory=set)
    problem: str = ""  # the current error or "missing" message


def _errors(doc):
    return [p.message for p in doc.problems if p.severity == csvfmt.ERROR]


def _stamp(path):
    """(st_mtime_ns, st_size), or None when the file can't be stat'ed."""
    try:
        status = os.stat(path)
    except OSError:
        return None
    return status.st_mtime_ns, status.st_size


class Watcher:
    def __init__(self):
        self.files = {}  # csvfmt.identity() -> _File
        self.dirty = {}  # owner -> kind of its latest change, until take_dirty()
        self.listeners = []  # each called with the changes of a poll that found any
        self._next = 0
        self._last = float("-inf")

    def use(self, owner, doc, delimiter=None):
        """owner read doc (with this delimiter): watch its file until forget() or retain() drops the owner."""
        ident = csvfmt.identity(doc.path)
        watched = self.files.get(ident)
        if watched is None:
            watched = self.files[ident] = _File(doc.path, None)
        watched.owners.add(owner)
        # A file it could not read (held open by another program, a folder) has a stamp all the same: poll() then
        # reports it when it changes, not on every poll
        watched.stamp = doc.stamp if doc.stamp != (0, 0) else _stamp(doc.path)
        key = (doc.schema, delimiter)
        errors = _errors(doc)
        if errors:
            watched.reads.setdefault(key, None)
        else:
            watched.reads[key] = doc
        watched.problem = errors[0] if errors else ""

    def forget(self, owner):
        """owner reads no file any more (e.g. before it is planned again)."""
        self._keep(lambda owners: owners - {owner})

    def retain(self, owners):
        """Keep watching for these owners only (e.g. the nodes that still exist)."""
        owners = set(owners)
        self._keep(lambda current: current & owners)

    def _keep(self, which):
        for ident in list(self.files):
            watched = self.files[ident]
            watched.owners = which(watched.owners)
            if not watched.owners:  # nobody reads it any more
                del self.files[ident]

    def clear(self):
        self.files.clear()
        self.dirty.clear()
        self._next = 0

    def problem(self, path) -> str:
        watched = self.files.get(csvfmt.identity(path))
        return watched.problem if watched else ""

    def take_dirty(self) -> dict:
        """The owners with a changed file since the last call, each with the kind of its latest change."""
        dirty, self.dirty = self.dirty, {}
        return dirty

    def poll(self, now=None, force=False) -> list[Change]:
        """Stat the next BATCH files, at most every INTERVAL seconds unless forced; returns what changed."""
        now = time.monotonic() if now is None else now
        if not self.files or (not force and now - self._last < INTERVAL):
            return []
        self._last = now
        idents = list(self.files)
        start = self._next % len(idents)
        count = min(BATCH, len(idents))
        self._next = (start + count) % len(idents)
        changes = []
        for i in range(count):
            change = self._check(self.files[idents[(start + i) % len(idents)]])
            if change is not None:
                changes.append(change)
                for owner in change.owners:
                    self.dirty[owner] = change.kind
        if changes:
            for listener in list(self.listeners):
                listener(changes)
        return changes

    def _check(self, watched):
        stamp = _stamp(watched.path)
        if stamp == watched.stamp:
            return None
        watched.stamp = stamp
        owners = frozenset(watched.owners)
        if stamp is None:
            watched.problem = f"Can't find {watched.path}"
            return Change(watched.path, "missing", owners, watched.problem)
        broken, kinds, problem = bool(watched.problem), set(), ""
        for (schema, delimiter), good in list(watched.reads.items()):
            doc = csvfmt.load(watched.path, delimiter, schema)
            errors = _errors(doc)
            if errors:
                problem = problem or errors[0]
                continue
            watched.reads[(schema, delimiter)] = doc
            if good is None or doc.text_hash != good.text_hash:
                kinds.add("text")
            elif doc.num_hash != good.num_hash:
                kinds.add("numbers")
        watched.problem = problem
        if problem:
            return Change(watched.path, "error", owners, problem)
        kind = next((k for k in ("text", "numbers") if k in kinds), "restored" if broken else "")
        return Change(watched.path, kind, owners) if kind else None


WATCH = Watcher()


def on_load_post(*_args):
    """Stamps are not trusted across sessions: after loading, every file is read again."""
    WATCH.clear()
    csvfmt.clear_cache()


def register():
    import bpy
    from bpy.app.handlers import persistent

    persistent(on_load_post)  # marks the function to survive loading files
    if on_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(on_load_post)


def unregister():
    import bpy

    if on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(on_load_post)
