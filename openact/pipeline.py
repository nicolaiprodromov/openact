"""What a node asks the model, and this session's caches of the answers (§5.3, §5.4).

A Job is one node context's table: one Row per table row. A row is answered by a lock, by code, by a stored row
with the same words (its wkey), or by its Cell, one question to the model and the unit of caching: MEMORY holds
this session's answers by cell key.

Two key families keep what the artist said apart from how OpenAct asked. The words keys (wkey per row, words_key
per entry) decide whether a stored answer still holds. The ask key and the cell key only label rows and key the
caches, so new templates, a node's Balance Option Order or another backend never discard an answer.

Missing cells wait in QUEUE, in chunks, for the decision worker (server.Worker), which sends one chunk at a time
(§5.5). Pure Python: store.py and the pure test suites use the keys too. Only submit(), answered() and progress()
look at the model server.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field

MODEL_NAME = "semif-qwen3.5-4b"  # the requests' model (§13.2)
MEMORY_MAX = 100_000  # cells, about 40 MB
MEMORY: OrderedDict[str, dict] = OrderedDict()  # cell key -> parse() of its answer and its model, least recent first
PENDING: dict[str, set] = {}  # cell key -> {(node uid, generation)} waiting for the answer, queued or in flight
FAILED: dict[str, str] = {}  # cell key -> why: not asked again until the words change or Ask Again
LIMITS = (32, 64)  # questions and scored rows per chunk (§5.5), until /v1/capabilities sizes them (size_chunks())
MS_PER_ROW = 21.0  # a scored row's time before /health reports ms_per_row (§5.4)
QUEUE = (deque(), deque(), deque())  # P0, P1, P2 (§5.5): the jobs waiting in each class, each a deque of its chunks
FILLS: dict[str, list] = {}  # node uid -> [cells answered, cells asked, rows answered, rows asked] of its fill
_QUEUED = threading.Condition()  # guards QUEUE: the main thread submits and cancels, the worker takes


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


@dataclass(frozen=True, eq=False)
class Chunk:
    """Missing cells of one job that travel in one request: one state, at most LIMITS (§5.5)."""

    job: Job  # its uid and generation for cancel(), its debias for the request
    cells: tuple
    priority: int = 2
    on_rows: object = None  # on_rows(job) runs on the main thread once the chunk is answered or has failed


def digest(*parts) -> str:
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
    """The exact question sent, its options in the order sent, and debias: MEMORY never answers another question.
    options: the question's criteria; a dict keeps its order, which digest() would lose by sorting its keys."""
    if isinstance(options, dict):
        options = list(options.items())
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
    """Queue the job's missing cells for the decision worker (§13.3, §5.5), in chunks (chunked()) that join class
    `priority` (0 interactive, 1 the active node's table, 2 other tables), where jobs take turns. on_rows(job) runs on
    the main thread after each chunk, with its answers in MEMORY or its errors in FAILED. A cell already queued or in
    flight only gains the job as a waiter. Main thread only."""
    from . import server  # imports bpy; everything else here is pure Python

    waiter, cells = (job.uid, job.generation), []
    for cell in missing:
        waiters = PENDING.setdefault(cell.key, set())
        if not any(uid == job.uid for uid, _generation in waiters):
            _fill(job.uid, cell, asked=1)
        if not waiters:
            cells.append(cell)
        waiters.add(waiter)
    if cells:
        with _QUEUED:
            QUEUE[priority].append(deque(Chunk(job, tuple(part), priority, on_rows) for part in chunked(cells)))
            _QUEUED.notify()
    server.worker()  # also when every cell was pending: unregistering stops the worker, and its chunk may be requeued


def chunked(cells) -> list[list[Cell]]:
    """The chunk builder (§5.5): consecutive cells of one state, at most LIMITS (questions, scored rows) in each. A
    cell that costs more rows than the limit travels alone."""
    questions, rows = LIMITS
    parts, cost = [], 0
    for cell in cells:
        if not parts or len(parts[-1]) >= questions or cost + cell.cost > rows or parts[-1][0].state != cell.state:
            parts.append([])
            cost = 0
        parts[-1].append(cell)
        cost += cell.cost
    return parts


def size_chunks(capabilities: dict) -> None:
    """The chunk limits of a server's /v1/capabilities (§5.5): batch_rows questions, at most max_questions, and two
    batches of rows, so a debiased chunk of yes/no questions is two score_shared batches."""
    global LIMITS
    try:
        batch, most = int(capabilities["batch_rows"]), int(capabilities["max_questions"])
    except (KeyError, TypeError, ValueError):
        return
    if batch > 0 and most > 0:
        LIMITS = (min(batch, most), 2 * batch)


def build_request(chunk: Chunk):
    """(request, question ids) of a chunk (§10.3): its cells under their question ids, with its job's debias."""
    ids = _question_ids(chunk.cells)
    return {"model": MODEL_NAME, "state": chunk.cells[0].state,
            "questions": {qid: cell.question for qid, cell in zip(ids, chunk.cells)},
            "options": {"debias": chunk.job.debias}}, ids


def _question_ids(cells):
    """The first 16 hex digits of each cell key, or the whole key where two cells share them (§10.3)."""
    counts = Counter(cell.key[:16] for cell in cells)
    return [cell.key[:16] if counts[cell.key[:16]] == 1 else cell.key for cell in cells]


def take(timeout: float) -> Chunk | None:
    """The worker's next chunk: from the first class that has one, whose jobs take turns; None after `timeout` s."""
    with _QUEUED:
        if not any(QUEUE):
            _QUEUED.wait(timeout)
        for jobs in QUEUE:
            if jobs:
                chunks = jobs.popleft()
                chunk = chunks.popleft()
                if chunks:
                    jobs.append(chunks)
                return chunk
    return None


def requeue(*chunks: Chunk) -> None:
    """Put chunks back at the head of their classes, in this order, ahead of the rest of their job: a chunk that met a
    loading model or no server, or the parts of a refused one (§10.5)."""
    with _QUEUED:
        for chunk in reversed(chunks):
            QUEUE[chunk.priority].appendleft(deque([chunk]))
        _QUEUED.notify()


def cancel(uid: str, below: float) -> None:
    """Drop the node's queued chunks of generations below `below` (§5.5, §13.3), all of them with math.inf (a deleted
    node). The one in flight completes, and its answers land in MEMORY. The dropped cells stop being pending, so
    another node waiting for one asks again. Main thread only."""
    _drop(lambda job: job.uid == uid and job.generation < below)


def cancel_all() -> None:
    """Drop every queued chunk, as cancel() does: another file was opened, or the add-on unregisters. Main thread
    only."""
    _drop(lambda job: True)


def _drop(wanted) -> None:
    dropped = []
    with _QUEUED:
        for jobs in QUEUE:
            for chunks in jobs:
                for chunk in [chunk for chunk in chunks if wanted(chunk.job)]:
                    chunks.remove(chunk)
                    dropped += chunk.cells
            kept = [chunks for chunks in jobs if chunks]
            jobs.clear()
            jobs.extend(kept)
    for cell in dropped:
        for waiter in {waiter for waiter, _generation in PENDING.pop(cell.key, ())}:
            _fill(waiter, cell, asked=-1)


def answered(chunk: Chunk, kind: str, data) -> None:
    """A chunk's result, posted by the worker to the main thread: "ok" with the server's reply, whose answers go into
    MEMORY; "error" with the message its cells keep in FAILED (no automatic retry); or "refused" (403), which leaves
    them unanswered, so a pass asks again once a server takes this Blender's token (§10.5). Then its job's on_rows."""
    from . import server

    model = provenance(server.STATE.health)
    answers = data.get("answers") if kind == "ok" and isinstance(data, dict) else None
    for qid, cell in zip(_question_ids(chunk.cells), chunk.cells):
        for waiter in {waiter for waiter, _generation in PENDING.pop(cell.key, ())}:
            _fill(waiter, cell, answered=1)
        if kind == "refused":
            continue
        if kind != "ok":
            FAILED[cell.key] = data
            continue
        try:
            remember(cell.key, dict(parse(cell, answers[qid]), model=model))
        except (KeyError, TypeError, ValueError) as error:
            FAILED[cell.key] = f"Unexpected answer from the model: {error}"
    if chunk.on_rows is not None:
        chunk.on_rows(chunk.job)


def _fill(uid, cell, answered=0, asked=0):
    fill = FILLS.setdefault(uid, [0, 0, 0, 0])
    fill[0] += answered
    fill[1] += asked
    fill[2] += answered * cell.cost
    fill[3] += asked * cell.cost
    if fill[0] >= fill[1]:  # nothing left queued or in flight for the node: its fill is over
        del FILLS[uid]


def progress(uid: str) -> tuple[int, int, float] | None:
    """The node's fill (PR-22's "Filling n/N · ~ETA", §17.1): (done, total, ms_per_row) while cells are queued or in
    flight for it, else None. done and total count its cells answered (or failed) and asked since the fill began; a
    cell is one question, so one table row for Match, Classify, Rules and Dials and the whole table for Ask and Pick.
    ms_per_row is what each remaining one should take: /health's ms_per_row (MS_PER_ROW until it reports one) times
    the scored rows it costs, so (total - done) * ms_per_row / 1000 is the ETA in seconds. Main thread only."""
    from . import server

    fill = FILLS.get(uid)
    if fill is None:
        return None
    done, total, rows_done, rows_total = fill
    scored = server.STATE.health.get("ms_per_row") or MS_PER_ROW
    return done, total, scored * (rows_total - rows_done) / (total - done)
