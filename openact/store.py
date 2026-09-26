"""A node's stored answers: the `entries` JSON, schema 2.

The JSON is the single source of truth: the inner group and the table object are rebuilt from it.
Pure Python (no bpy), so the conversion and the eviction rules also run under plain CPython.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

SCHEMA = 2
MAX_ENTRIES = 16  # warm or cold entries per node; live and pinned entries come on top
MB = 1024 * 1024
SOFT_CAP = 2 * MB  # per node; only warm and cold entries are evicted to meet it
HARD_CAP = 4 * MB  # per entry
V010 = "jev_nodes 0.1.0"  # model and ask key of the entries converted from v0.1.0
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
    legacy: bool = False
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


def load(node) -> NodeStore:
    """The node's entries. A v0.1.0 list (schema 1) is converted but not written back, because opening
    a file must not rewrite it. Unreadable JSON gives an empty store, and a damaged entry is skipped."""
    try:
        data = json.loads(node.entries or "[]")
    except (ValueError, RecursionError):
        return NodeStore()
    if isinstance(data, list):
        items, read = data, _from_v010
    elif isinstance(data, dict) and data.get("schema") == SCHEMA and isinstance(data.get("entries"), list):
        items, read = data["entries"], _read
    else:
        return NodeStore()
    store = NodeStore()
    for item in items:
        try:
            store.entries.append(read(item))
        except (AttributeError, LookupError, TypeError, ValueError):
            pass
    return store


def save(node, store: NodeStore, live_gns: frozenset[tuple[str, ...]] | None = None) -> None:
    """Write the store into the node as schema 2.

    live_gns holds the gn tuples of the node's live contexts. "Live" is not saved, so only a caller
    that knows it may evict: warm and cold entries go, least recently used first, beyond MAX_ENTRIES
    and then until the JSON fits SOFT_CAP. Live and pinned entries are never evicted, even when they
    alone exceed SOFT_CAP. None means the live set is unknown, and nothing is evicted.

    Sets store.note for the node's status. Raises ValueError, writing nothing, if an entry is larger
    than HARD_CAP.
    """
    parts = []  # (entry, JSON, bytes): each entry is serialised once, because its size decides the caps
    for entry in store.entries:
        text = _dump(entry)
        size = len(text.encode("utf-8"))
        if size > HARD_CAP:
            raise ValueError(f"A table is too large to store in the file ({size / MB:.1f} MB; "
                             f"the limit is {HARD_CAP // MB} MB)")
        parts.append((entry, text, size))
    evicted = set()
    dropped = 0
    if live_gns is not None:
        old = sorted((i for i, (entry, _, _) in enumerate(parts) if not entry.pinned and entry.gn not in live_gns),
                     key=lambda i: parts[i][0].t)
        evicted.update(old[:-MAX_ENTRIES])  # all but the MAX_ENTRIES most recent
        # bytes of the JSON: _HEAD, the kept entries with a comma between each two, _TAIL
        total = len(_HEAD) + len(_TAIL) - 1 + sum(size + 1 for i, (_, _, size) in enumerate(parts) if i not in evicted)
        for i in old[len(evicted):]:
            if total <= SOFT_CAP:
                break
            evicted.add(i)
            total -= parts[i][2] + 1
            dropped += 1
    kept = [part for i, part in enumerate(parts) if i not in evicted]
    text = _HEAD + ",".join(part[1] for part in kept) + _TAIL
    size = len(text.encode("utf-8"))
    note = ""
    if live_gns is not None and size > SOFT_CAP:
        note = (f"entries use {size / MB:.1f} MB: live and pinned contexts are kept; "
                "pin fewer word sets to shrink the file")
    elif dropped:
        note = f"{dropped} old context{'s' if dropped > 1 else ''} dropped to keep the file small"
    node.entries = text
    store.entries = [part[0] for part in kept]
    store.note = note


def _dump(entry):
    data = dict(vars(entry), rows=dict(vars(entry.rows)))
    if not entry.frames:
        del data["frames"]
    if entry.rows.top is None:
        del data["rows"]["top"]
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))  # compact: the caps count bytes


def _read(data):
    table = _typed(data.get("rows", {}), dict)
    top = table.get("top")
    entry = Entry(
        words_key=str(data.get("words_key", "")),
        asks=[str(key) for key in _typed(data.get("asks", []), list)],
        pv=int(data.get("pv", 0)),
        t=float(data.get("t", 0.0)),
        legacy=bool(data.get("legacy", False)),
        pinned=bool(data.get("pinned", False)),
        frames=_typed(data.get("frames", []), list),
        gn=tuple(_typed(word, str) for word in _typed(data["gn"], list)),
        model=str(data.get("model", "")),
        kind=str(data.get("kind", "")),
        answer_type=str(data.get("answer_type", "")),
        shape=_typed(data.get("shape", {}), dict),
        rows=Rows(
            name=_typed(table.get("name", []), list),
            wkey=_typed(table.get("wkey", []), list),
            ask=_typed(table.get("ask", []), list),
            src=_typed(table.get("src", ""), str),
            v={key: _typed(column, list) for key, column in _typed(table.get("v", {}), dict).items()},
            payload={key: _typed(column, list) for key, column in _typed(table.get("payload", {}), dict).items()},
            top=None if top is None else _typed(top, list),
        ),
        single=_typed(data.get("single", {}), dict),
        csv=_typed(data.get("csv", {}), dict),
    )
    rows = entry.rows
    columns = [rows.wkey, rows.ask, rows.src, *rows.v.values(), *rows.payload.values()]
    if rows.top is not None:
        columns.append(rows.top)
    if any(len(column) != len(rows.name) for column in columns):
        raise ValueError("the row columns have different lengths")
    return entry


def _from_v010(item):
    """A v0.1.0 entry {"key", "gn", "a", "t"}, in schema 2. Its words go to their positions in the
    fixed word tuples; the old key is dropped."""
    first, second = (_typed(word, str) for word in _typed(item["gn"], list))
    answer = _typed(item["a"], dict)
    entry = Entry(asks=[V010], pv=0, t=float(item.get("t", 0.0)), legacy=True, model=V010)
    if "mode" in answer:  # Ask: (Question, Context) -> Question, Context, Answers, Overrides, Key
        value = float(answer["value"])
        entry.gn = (first, second, "", "", "")
        entry.kind, entry.answer_type = "ask", str(answer["mode"])
        entry.shape = {"rows": 0, "columns": {}}
        entry.single = {"Answer": value, "Yes": value >= 0.5, "Confidence": float(answer["confidence"])}
    else:  # Pick: (Prompt, Items) -> Prompt, Context, Items, Overrides, Key
        names = [_typed(name, str) for name in _typed(answer["items"], list)]
        prob = [float(p) for p in _typed(answer["probabilities"], list)]
        if len(prob) != len(names):
            raise ValueError("one probability per item")
        entry.gn = (first, "", second, "", "")
        entry.kind, entry.answer_type = "pick", "ONE_OF"
        entry.shape = {"rows": len(names), "columns": {"prob": "FLOAT"}}
        entry.rows = Rows(name=names, wkey=[""] * len(names), ask=[0] * len(names), src="m" * len(names),
                          v={"prob": prob})
        entry.single = {"Index": int(answer["index"]), "Item": str(answer["item"]),
                        "Confidence": float(answer["confidence"])}
    return entry


def _typed(value, kind):
    if not isinstance(value, kind):
        raise TypeError(f"expected {kind.__name__}, not {type(value).__name__}")
    return value
