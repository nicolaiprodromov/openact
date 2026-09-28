"""A node's stored answers: the `entries` JSON, schema 2.

The JSON is the single source of truth: the inner group and the table object are rebuilt from it.
Pure Python (no bpy), so the reading and the eviction rules also run under plain CPython.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, fields

SCHEMA = 2
MAX_ENTRIES = 16  # warm or cold entries per node; live and pinned entries come on top
MB = 1024 * 1024
SOFT_CAP = 2 * MB  # per node; only warm and cold entries are evicted to meet it
HARD_CAP = 4 * MB  # per entry
_HEAD, _TAIL = f'{{"schema":{SCHEMA},"entries":[', "]}"


@dataclass
class Rows:
    """A table, column by column: every list, and src, has one item per row."""

    name: list[str] = field(default_factory=list)
    wkey: list[str] = field(default_factory=list)
    ask: list[int] = field(default_factory=list)  # index into Entry.asks
    src: str = ""  # per row: m model, c code, l lock, s stale, p pending, e error
    v: dict[str, list] = field(default_factory=dict)
    payload: dict[str, list] = field(default_factory=dict)
    top: list | None = None  # optional: the top 3 [answer row, probability] of each row


@dataclass
class Entry:
    """The answers stored for one tuple of words, gn."""

    words_key: str = ""
    asks: list[str] = field(default_factory=list)
    pv: int = 0
    t: float = 0.0  # last use: warm and cold entries are evicted least recently used first
    pinned: bool = False
    frames: list[list[int]] = field(default_factory=list)  # [start, end] ranges of a pinned entry's words
    gn: tuple[str, ...] = ()
    model: str = ""
    kind: str = ""
    answer_type: str = ""
    shape: dict = field(default_factory=dict)
    rows: Rows = field(default_factory=Rows)
    single: dict = field(default_factory=dict)
    csv: dict[str, list[str]] = field(default_factory=dict)  # path -> [text_hash, num_hash]


@dataclass
class NodeStore:
    """A node's entries, as load() reads them and save() writes them."""

    entries: list[Entry] = field(default_factory=list)
    note: str = field(default="", compare=False)  # what the last save() evicted or kept, for the status
    schema: int = field(default=SCHEMA, compare=False)  # above SCHEMA: JSON of a newer OpenAct, left unread


def load(node) -> NodeStore:
    """The node's entries. A damaged entry is skipped, and unreadable JSON gives an empty store, as does a node with
    none yet (a new node holds "[]"). JSON of a newer schema also gives an empty store, and save() refuses to write
    over that JSON."""
    try:
        data = json.loads(node.entries or "[]")
    except (ValueError, RecursionError):
        return NodeStore()
    if isinstance(data, dict) and data.get("schema") == SCHEMA and isinstance(data.get("entries"), list):
        items = data["entries"]
    elif isinstance(data, dict) and type(data.get("schema")) is int and data["schema"] > SCHEMA:
        return NodeStore(schema=data["schema"])
    else:
        return NodeStore()
    store = NodeStore()
    for item in items:
        try:
            entry = _read(item)
            _check(entry)
        except Exception:  # whatever the damage, it costs only this entry's answers
            continue
        store.entries.append(entry)
    return store


def save(node, store: NodeStore, live_gns: frozenset[tuple[str, ...]] | None = None) -> None:
    """Write the store into the node as schema 2.

    live_gns holds the gn tuples of the node's live contexts. "Live" is not saved, so only a caller
    that knows it may evict: warm and cold entries go, least recently used first, beyond MAX_ENTRIES
    and then until the JSON fits SOFT_CAP. Live and pinned entries are never evicted, even when they
    alone exceed SOFT_CAP. None means the live set is unknown, and nothing is evicted.

    Sets store.note for the node's status. Raises ValueError, writing nothing, if the store holds a
    newer OpenAct's JSON, or if an entry is damaged (load() would skip it) or larger than HARD_CAP.
    """
    if store.schema > SCHEMA:
        raise ValueError(newer_message(store))
    parts = []  # (entry, JSON, bytes): each entry is serialised once, because its size decides the caps
    for entry in store.entries:
        _check(entry)
        text = _dump(entry)
        size = len(text.encode("utf-8"))
        if size > HARD_CAP:
            raise ValueError(f"A table is too large to store in the file ({_mb(size)} MB; "
                             f"the limit is {HARD_CAP // MB} MB)")
        parts.append((entry, text, size))
    evicted, note = set(), ""
    if live_gns is not None:
        old = sorted((i for i, (entry, _, _) in enumerate(parts) if not entry.pinned and entry.gn not in live_gns),
                     key=lambda i: parts[i][0].t)
        evicted.update(old[:-MAX_ENTRIES])  # all but the MAX_ENTRIES most recent
        # bytes of the JSON: _HEAD, the kept entries with a comma between each two, _TAIL
        total = len(_HEAD) + len(_TAIL) - 1 + sum(size + 1 for i, (_, _, size) in enumerate(parts) if i not in evicted)
        dropped = 0
        for i in old[len(evicted):]:
            if total <= SOFT_CAP:
                break
            evicted.add(i)
            total -= parts[i][2] + 1
            dropped += 1
        drops = f"{dropped} old context{'s' if dropped > 1 else ''} dropped" if dropped else ""
        if total > SOFT_CAP:
            note = (f"{drops}; entries still use" if drops else "entries use") + (
                f" {_mb(total)} MB: live and pinned contexts are kept; pin fewer word sets to shrink the file")
        elif drops:
            note = f"{drops} to keep the file small"
    kept = [part for i, part in enumerate(parts) if i not in evicted]
    node.entries = _HEAD + ",".join(part[1] for part in kept) + _TAIL
    store.entries = [part[0] for part in kept]
    store.note = note


def newer_message(store: NodeStore) -> str:
    """Why a store that holds a newer OpenAct's JSON can't change: save() raises it, the pass shows it."""
    return f"These answers were saved by a newer OpenAct (schema {store.schema}): update OpenAct to change them"


def _mb(size):
    return f"{math.ceil(size * 10 / MB) / 10:.1f}"  # rounded up: just above a cap never reads as the cap


def _dump(entry):
    data = dict(vars(entry), rows=dict(vars(entry.rows)))
    if not entry.frames:
        del data["frames"]
    if entry.rows.top is None:
        del data["rows"]["top"]
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))  # compact: the caps count bytes


def _check(entry):
    """Raise ValueError unless the entry has the types of schema 2, with one item per row in every
    column. load() skips an entry that fails, and save() refuses to write it."""
    rows = entry.rows
    texts = (entry.words_key, entry.model, entry.kind, entry.answer_type)
    if not (isinstance(entry.gn, tuple) and set(map(type, entry.gn + texts)) <= {str} and _list_of(entry.asks, str)):
        raise ValueError("an entry's gn must be a tuple of strings, and its keys and names strings")
    if not (type(entry.pv) is int and type(entry.pinned) is bool
            and type(entry.t) in (int, float) and math.isfinite(entry.t)):
        raise ValueError("an entry's pv, pinned or t has the wrong type")
    if not (isinstance(entry.frames, list)
            and all(isinstance(table, dict) for table in (entry.shape, entry.single, entry.csv))):
        raise ValueError("an entry's frames, shape, single or csv has the wrong type")
    if not (isinstance(rows, Rows) and _list_of(rows.name, str) and _list_of(rows.wkey, str) and type(rows.src) is str
            and _list_of(rows.ask, int) and set(rows.ask) <= set(range(len(entry.asks)))
            and (rows.top is None or isinstance(rows.top, list))):
        raise ValueError("an entry's rows have the wrong types")
    columns = [rows.wkey, rows.ask, rows.src]
    for table in (rows.v, rows.payload):
        if not (isinstance(table, dict) and all(_list_of(column, int, float) for column in table.values())):
            raise ValueError("an entry's answer and payload columns must hold numbers")
        columns += table.values()
    if rows.top is not None:
        columns.append(rows.top)
    if any(len(column) != len(rows.name) for column in columns):
        raise ValueError("an entry's row columns have different lengths")


def _list_of(value, *kinds):
    return isinstance(value, list) and set(map(type, value)) <= set(kinds)  # exact types: True is not a number


def _read(data):
    """A schema-2 entry. Keys that Entry and Rows don't have are ignored; missing ones default, except gn."""
    entry = _known(Entry, data)
    entry.gn = tuple(data["gn"]) if isinstance(data["gn"], list) else data["gn"]  # JSON has no tuples
    entry.rows = _known(Rows, data.get("rows", {}))
    return entry


def _known(cls, data):
    if not isinstance(data, dict):
        raise TypeError(f"{cls.__name__} must be a JSON object")
    return cls(**{f.name: data[f.name] for f in fields(cls) if f.name in data})
