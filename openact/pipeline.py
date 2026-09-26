"""What a node asks the model, and this session's caches of the answers (§5.3, §5.4).

A Job is one node context's table: one Row per table row. A row is answered by a lock, by code, by a stored row
with the same words (its wkey), or by its Cell, one question to the model and the unit of caching: MEMORY holds
this session's answers by cell key.

Two key families keep what the artist said apart from how OpenAct asked. The words keys (wkey per row, words_key
per entry) decide whether a stored answer still holds. The ask key and the cell key only label rows and key the
caches, so new templates, a node's Balance Option Order or another backend never discard an answer.

Pure Python: store.py and the pure test suites use the keys too. Only submit() talks to the model server.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, OrderedDict
from dataclasses import dataclass, field

MEMORY_MAX = 100_000  # cells, about 40 MB
MEMORY: OrderedDict[str, dict] = OrderedDict()  # cell key -> parse() of its answer and its model, least recent first
PENDING: dict[str, set] = {}  # cell key -> {(node uid, generation)} waiting for the answer in flight
FAILED: dict[str, str] = {}  # cell key -> why: not asked again until the words change or Ask Again


@dataclass(frozen=True)
class Cell:
    """One question to the model: the unit of caching and batching."""

    key: str  # cell_key()
    state: str  # the shared situation: every cell of a job has the job's state
    question: dict  # Jev question: {"type": "noul" | "choice" | "score", "instructions", "criteria"}
    cost: int  # scored rows, cost()


@dataclass(frozen=True)
class Row:
    """One table row as Geometry Nodes reads it."""

    name: str  # identity: item name, level names ("alive|3") or setting name
    wkey: str  # what the artist said for this row, wkey()
    source: str = "pending"  # "lock" | "code" | "model" | "pending" | "stale" | "error"
    values: dict = field(default_factory=dict)  # answer columns, e.g. {"p": 0.93, "c": 0.86}
    ask: str | None = None  # the ask key the answer was produced with
    cell: Cell | None = None  # None for lock and code rows


@dataclass(frozen=True)
class TableShape:
    """What inner.build() lays out for a table, stored as the entry's shape (§14.2)."""

    rows: int = 0
    columns: tuple[tuple[str, str], ...] = ()  # (name, type) of each answer column of a row

    def as_dict(self) -> dict:
        return {"rows": self.rows, "columns": dict(self.columns)}

    def digest(self) -> str:
        return digest("shape", self.as_dict())


@dataclass
class Job:
    """One node context's table: its words, its rows, and how its questions are asked."""

    uid: str  # node uid
    gn: tuple[str, ...]  # the raw words the match chain compares, Key included
    words_key: str  # words_key()
    shape: TableShape
    rows: list[Row]
    priority: int = 0  # 0 interactive, 1 the active node's table, 2 other tables (§5.5)
    generation: int = 0  # bumped when the node's words change, so stale chunks are dropped
    answer_type: str = ""  # 'YES_NO', 'AMOUNT' or 'ONE_OF'
    debias: bool = True  # the node's Balance Option Order: its requests carry it
    problems: tuple[str, ...] = ()  # warnings of a usable plan (e.g. a CSV's), for the node's status
    options: tuple = ()  # the answer names rows choose among (Dials: per row), which Overrides answers name (§7.6)
    locks: dict = field(default_factory=dict)  # Overrides lock -> "" when a row uses it, else its orphan warning


def digest(*parts) -> str:
    """sha1 over canonical JSON, like the prototype's request key."""
    return hashlib.sha1(json.dumps(parts, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def wkey(spec: str, answer_type: str, shared_words, row_facts, offered_options) -> str:
    """A row's words key: what the artist said for it, before templating. Never the Key word (§6.0)."""
    return digest("row", spec, answer_type, shared_words, row_facts, offered_options)


def words_key(gn, csv_text_hashes, answer_type: str, shape_digest: str, wkeys, lock_digest: str = "") -> str:
    """An entry's words key: a stored entry holds while it matches, whatever its ask keys."""
    return digest("words", gn, sorted(csv_text_hashes), answer_type, shape_digest, wkeys, lock_digest)


def ask_key(prompt_version: int, debias: bool, backend_model_id: str) -> str:
    """How OpenAct asked: templates, the node's Balance Option Order and the backend. It only labels rows."""
    return digest("ask", prompt_version, debias, backend_model_id)


def cell_key(kind: str, state: str, instructions: str, options, debias: bool) -> str:
    """The exact question sent, its options in the order sent, and debias: MEMORY never answers another question."""
    return digest("cell", kind, state, instructions, options, debias)


def cost(options: int, debias: bool) -> int:
    """Scored rows of a question: more than 16 options run as a tournament of groups of up to 16 and a final."""
    groups = 1 if options <= 16 else -(-options // 16) + 1
    return (2 if debias else 1) * groups


def provenance(health: dict) -> str:
    """The backend and model a server's /health reports, the backend_model_id of the ask key."""
    return " ".join(filter(None, (health.get("backend"), health.get("model_id"))))


def cached(cell_key: str) -> dict | None:
    """This session's answer to the cell with this key, or None (§13.3)."""
    answer = MEMORY.get(cell_key)
    if answer is not None:
        MEMORY.move_to_end(cell_key)
    return answer


def remember(key: str, answer: dict) -> None:
    MEMORY[key] = answer
    MEMORY.move_to_end(key)
    if len(MEMORY) > MEMORY_MAX:
        MEMORY.popitem(last=False)


def forget(keys) -> None:
    """Drop this session's answers and failures of these cells, so they are asked again."""
    for key in keys:
        MEMORY.pop(key, None)
        FAILED.pop(key, None)


def parse(cell: Cell, answer) -> dict:
    """What the rows of a cell need from the server's answer: p for noul; probabilities in option order for choice;
    those and amount = score / (levels - 1) for score. The server's confidence is not used: the specs compute it
    (4.e). Raises KeyError, TypeError or ValueError when the answer doesn't fit the question."""
    kind = cell.question["type"]
    if not isinstance(answer, dict) or answer.get("type") != kind:
        raise ValueError(f"not a {kind} answer")
    if kind == "noul":
        parsed = {"p": float(answer["noul"])}
        numbers = [parsed["p"]]
    else:
        criteria = cell.question["criteria"]
        ids = list(criteria) if kind == "choice" else [str(i) for i in range(len(criteria))]
        given = answer["probabilities"]
        if not isinstance(given, dict):
            raise TypeError("probabilities must be an object")
        parsed = {"probabilities": [float(given.get(oid, 0.0)) for oid in ids]}
        numbers = list(parsed["probabilities"])
        if kind == "score":
            score = float(answer["score"])
            numbers.append(score)
            parsed["amount"] = min(max(score / (len(ids) - 1), 0.0), 1.0)
    if not all(math.isfinite(number) for number in numbers):
        raise ValueError("a number in the answer is not finite")
    return parsed


def submit(job: Job, missing: list[Cell], priority: int, on_rows=None) -> None:
    """Ask the model for the job's missing cells (§13.3). on_rows(job) then runs on the main thread, with their answers
    in MEMORY or their errors in FAILED. Each call sends one request on the server's executor; the priority orders
    nothing yet. A cell already in flight only gains the job as a waiter."""
    from . import server  # imports bpy; everything else here is pure Python

    waiter = (job.uid, job.generation)
    cells = []
    for cell in missing:
        if cell.key in PENDING:
            PENDING[cell.key].add(waiter)
        else:
            PENDING[cell.key] = {waiter}
            cells.append(cell)
    if not cells:
        return
    ids = _question_ids(cells)
    payload = {"model": server.MODEL_NAME, "state": cells[0].state,
               "questions": {qid: cell.question for qid, cell in zip(ids, cells)}, "options": {"debias": job.debias}}
    server.ask(payload, lambda kind, data: _answered(job, cells, ids, kind, data, on_rows))


def _question_ids(cells):
    """The first 16 hex digits of each cell key, or the whole key where two cells share them (§10.3)."""
    counts = Counter(cell.key[:16] for cell in cells)
    return [cell.key[:16] if counts[cell.key[:16]] == 1 else cell.key for cell in cells]


def _answered(job, cells, ids, kind, data, on_rows):
    from . import server

    model = provenance(server.STATE.health)
    for cell in cells:
        PENDING.pop(cell.key, None)
    if kind == "ok":
        answers = data.get("answers") if isinstance(data, dict) else None
        for qid, cell in zip(ids, cells):
            try:
                remember(cell.key, dict(parse(cell, answers[qid]), model=model))
            except (KeyError, TypeError, ValueError) as error:
                FAILED[cell.key] = f"Unexpected answer from the model: {error}"
    elif kind == "error":
        for cell in cells:
            FAILED[cell.key] = f"The model refused: {data}"
    # "offline" keeps nothing: the pass asks again once the model is ready
    if on_rows is not None:
        on_rows(job)
