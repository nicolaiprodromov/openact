"""What every node spec shares (§10.2): the Spec interface, the helpers behind the shared formats (4.g), and the
validity rule that answers a context's table from stored rows and cached cells (§5.4).

Pure Python when imported, so the pure suites plan from plain words; the methods that read a node import the
evaluator, and with it bpy, when they run.
"""

from __future__ import annotations

import dataclasses
import math
import os
import time

from .. import csvfmt, pipeline, store

NO_CONTEXT = "No additional context."
KEY = ('INPUT', "Key", 'NodeSocketString',
       "Tells apart objects that share every other word, e.g. typed per object on the modifier. Compared, never "
       "sent to the model", "word", True)  # the last word of every node (§6.0), hidden until Ctrl H
SOURCES = {"lock": "l", "code": "c", "model": "m", "pending": "p", "stale": "s", "error": "e"}  # Row.source -> src
OVERRIDES = ("Optional Overrides CSV (.csv path). Rows with lock = 1 replace the model's answer and are never asked. "
             "Write Overrides CSV (sidebar) starts one from the current answers")
UNSURE_BELOW = 0.30  # 4.e: the exported unsure column


def option_text(option) -> str:
    """An option as the model reads it: "name: when", or the name alone. The one definition (sources.py
    re-exports it): takes a csvfmt.Option or lists.Item, or a (name, when) pair."""
    name, when = (option.name, option.when) if hasattr(option, "name") else option
    return f"{name}: {when}" if when else name


def warnings(problems) -> tuple[str, ...]:
    """The warnings among csvfmt Problems (ItemList.problems, AnswerSet.problems) as status texts, after the file's
    name and line: only its name when the message starts with the line."""
    texts = []
    for p in problems:
        if p.severity != "warning":
            continue
        where = os.path.basename(p.file) if p.file and p.line else ""
        if where and not p.message.startswith(f"Line {p.line}"):
            where += f" line {p.line}"
        texts.append(f"{where}: {p.message}" if where else p.message)
    return tuple(texts)


def state_text(context: str, empty: str = NO_CONTEXT) -> str:
    """A job's state, the situation its questions share: the Context text, else `empty`."""
    return context.strip() or empty


def confidence(answer_type: str, distribution) -> float:
    """4.e, from the probabilities in option order: 0 for an answer that says nothing (p = 0.5, a uniform choice or
    scale), 1 for a certain one. Yes/No and One-of: the top minus the runner-up (|2p - 1| for [p, 1 - p]). Amount:
    1 - sigma / sigma_u, where sigma is the spread of the level distribution and sigma_u = sqrt((L^2 - 1) / 12) that
    of a uniform one over its L levels, clamped to 0..1."""
    probabilities = [float(p) for p in distribution]
    total = math.fsum(probabilities)
    if total <= 0:
        return 0.0
    if len(probabilities) < 2:
        return 1.0
    probabilities = [p / total for p in probabilities]
    if answer_type == 'AMOUNT':
        levels = len(probabilities)
        mean = math.fsum(i * p for i, p in enumerate(probabilities))
        variance = math.fsum(p * (i - mean) ** 2 for i, p in enumerate(probabilities))
        value = 1.0 - math.sqrt(variance * 12 / (levels * levels - 1))
    else:
        top, runner_up = sorted(probabilities, reverse=True)[:2]
        value = top - runner_up
    return round(min(max(value, 0.0), 1.0), 12)  # rounded, so a uniform answer gives exactly 0


def overrides_inputs() -> list:
    """The Overrides word and its CSV companion, hidden like Key (§6.0)."""
    from ..sources import csv_inputs  # imports bpy

    return csv_inputs("Overrides", OVERRIDES, hidden=True)


def named(answer: str, names) -> int | None:
    """The position of answer among names, ignoring case; None when absent."""
    folded = [name.casefold() for name in names]
    return folded.index(answer.casefold()) if answer.casefold() in folded else None


def parse_number(text):
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def lock_answer(answer_type: str, answer: str, levels=()):
    """A Yes/No lock's p (yes, no or a probability 0-1) or an Amount lock's 0-1 (a number 0-1 or a level name,
    at its position / (L - 1)); else why the answer names nothing valid (str)."""
    number = parse_number(answer)
    if answer_type == 'AMOUNT':
        at = named(answer, levels) if number is None else None
        if number is not None and 0.0 <= number <= 1.0:
            return number
        if at is not None and len(levels) > 1:
            return at / (len(levels) - 1)
        return f"'{answer}' is not a number from 0 to 1 or a level name"
    if answer.casefold() in ("yes", "no"):
        return float(answer.casefold() == "yes")
    if number is not None and 0.0 <= number <= 1.0:
        return number
    return f"'{answer}' is not yes, no or a number from 0 to 1"


def number_text(value: float) -> str:
    """0.625 -> "0.625", 1.0 -> "1": how exported numbers read."""
    return f"{value:.3f}".rstrip("0").rstrip(".") or "0"


def yes_no_text(p: float) -> str:
    return "yes" if p >= 0.5 else "no"


def orphaned(jobs) -> list[str]:
    """The orphaned-lock warnings of a node's jobs: locks that no context's rows use (§7.6)."""
    locks = {}
    for job in jobs:
        for lock, warning in job.locks.items():
            locks[lock] = "" if locks.get(lock) == "" or not warning else warning
    return [warning for warning in locks.values() if warning]


def stored_rows(entry) -> list[pipeline.Row]:
    """The answered rows of a stored entry: model and code rows (locks come from the Overrides CSV every time)."""
    rows, kinds = entry.rows, {src: source for source, src in SOURCES.items()}
    return [pipeline.Row(rows.name[i], wkey, kinds[rows.src[i]], {column: v[i] for column, v in rows.v.items()},
                         entry.asks[rows.ask[i]])
            for i, wkey in enumerate(rows.wkey) if wkey and rows.src[i] in "mc"]


class Spec:
    """A node type: its sockets and words, how the words of a context become a Job, and how answers become the
    Job's rows, the entry the node stores and its outputs."""

    name = ""  # "ask", "pick", ...
    idname = ""  # "GeometryNodeOpenActAsk", ...
    label = ""  # the node's label, also in its inner group's name
    PROMPT_VERSION = 1  # the templates: stored per entry as pv, part of the ask key and never of a words key
    WORDS: tuple[str, ...] = ()  # the fixed word tuple, in match-chain order, Key last (§8.2)
    FALLBACK: dict = {}  # single-value output -> its value before the first answer (§6.0)
    empty_hint = ""

    def words(self, node) -> tuple[str, ...]:
        """The String words the node can have, whatever its dropdowns (§8.2)."""
        return self.WORDS

    def interface(self, node) -> list:
        """The sockets the node should have: SocketDefs for inner.sync_interface()."""
        raise NotImplementedError

    def gn(self, node, ctx) -> tuple[str, ...]:
        """The raw words in ctx, as the match chain compares them: "" for a word the node has no socket for."""
        from ..evaluator import as_text, eval_input  # imports bpy

        sockets = [node.inputs.get(name) for name in self.words(node)]
        return tuple("" if socket is None else as_text(eval_input(socket, ctx)) for socket in sockets)

    def listed(self, node, ctx):
        """What a list source adds to the fingerprint, e.g. a collection's children."""
        return None

    def fingerprint(self, node, ctx) -> str:
        """Everything plan() reads, hashed: while it is unchanged, the Job planned from it still holds (§5.3). The
        trust state is part of it, as it decides which files plan() may read (§16.1)."""
        from .. import trust  # imports bpy

        return pipeline.digest(self.name, self.PROMPT_VERSION, self.gn(node, ctx), getattr(node, "answer_type", ""),
                               node.balance_order, self.listed(node, ctx), self.overrides_stamp(node, ctx),
                               trust.is_trusted())

    def plan(self, node, ctx) -> pipeline.Job:
        """The node's Job in ctx. Raises Hint, Pending or Unresolvable."""
        raise NotImplementedError

    def values(self, index: int, answer: dict) -> dict:
        """The answer columns of row `index` from its cell's parsed answer."""
        raise NotImplementedError

    def single(self, rows: list[pipeline.Row]) -> dict:
        """The entry-level values of an answered table: the single-value outputs."""
        return {}

    def ask_key(self, debias: bool, model: str) -> str:
        return pipeline.ask_key(self.PROMPT_VERSION, debias, model)

    def rows(self, job: pipeline.Job, answers: dict) -> list[pipeline.Row]:
        """Server answers -> table rows: each pending row whose cell is answered in `answers` (cell key -> the parsed
        answer and the model that gave it) becomes a model row, labelled with the ask key it was asked with."""
        rows = []
        for index, row in enumerate(job.rows):
            answer = answers.get(row.cell.key) if row.source == "pending" and row.cell is not None else None
            if answer is not None:
                row = dataclasses.replace(row, source="model", values=self.values(index, answer),
                                          ask=self.ask_key(job.debias, answer["model"]))
            rows.append(row)
        return rows

    def resolve(self, job: pipeline.Job, entries, last_shown: str = ""):
        """Answer the job by the validity rule (§5.4), from the node's stored entries and this session's caches.

        The stored entry with the job's words holds while its words key matches, whatever its ask keys. Otherwise
        each row takes the answer of a row with its wkey in that entry or in the one this context showed last
        (words key last_shown), else its cell's answer in MEMORY. Returns (entry, missing cells, failure messages):
        the stored entry, or a new one once every row is answered, else None."""
        found = next((entry for entry in entries if entry.gn == job.gn), None)
        if found is not None and found.words_key == job.words_key:
            return found, [], []
        shown = next((entry for entry in entries if entry.words_key == last_shown), None) if last_shown else None
        sources = [entry for entry in (found, shown) if entry is not None]
        carried = {}
        for source in sources:
            for row in stored_rows(source):
                carried.setdefault(row.wkey, row)
        rows, answers, missing, failed = [], {}, {}, {}
        for row in job.rows:
            old = carried.get(row.wkey) if row.source != "lock" else None  # a lock wins over any stored answer
            if old is not None:
                row = dataclasses.replace(row, source=old.source, values=old.values, ask=old.ask)
            elif row.cell is not None:
                key = row.cell.key
                answer = pipeline.cached(key)
                if answer is not None:
                    answers[key] = answer
                elif key in pipeline.FAILED:
                    failed[key] = pipeline.FAILED[key]
                else:
                    missing[key] = row.cell
            rows.append(row)
        if missing or failed:
            return None, list(missing.values()), list(failed.values())
        if answers:
            model = next(iter(answers.values()))["model"]
            model = " ".join(filter(None, (model, "debias" if job.debias else "")))
        else:  # every row carried over
            model = sources[0].model if sources else ""
        return self.entry(job, self.rows(dataclasses.replace(job, rows=rows), answers), model), [], []

    def entry(self, job: pipeline.Job, rows: list[pipeline.Row], model: str = "") -> store.Entry:
        """An answered table as the entry the node stores (§14.2): columnar rows and the ask keys behind them. A row
        that asked nothing (a lock or code row) has the ask key ""."""
        asks = list(dict.fromkeys(row.ask or "" for row in rows))
        return store.Entry(
            words_key=job.words_key, asks=asks, pv=self.PROMPT_VERSION, t=time.time(), gn=job.gn, model=model,
            kind=self.name, answer_type=job.answer_type, shape=job.shape.as_dict(),
            rows=store.Rows(name=[row.name for row in rows], wkey=[row.wkey for row in rows],
                            ask=[asks.index(row.ask or "") for row in rows],
                            src="".join(SOURCES[row.source] for row in rows),
                            v={column: [row.values[column] for row in rows] for column, _type in job.shape.columns}),
            single=self.single(rows))

    # ------------------------------------------------------------------ the Overrides CSV (§7.6)

    OVERRIDE_KEYS: tuple[str, ...] = ("item",)  # the Overrides CSV's key columns

    def override_keys(self, job) -> list[str]:
        return list(self.OVERRIDE_KEYS)

    def override_units(self, job) -> list:
        """(key cells, row indices) of each lockable unit of the table: by default each named row, by its name."""
        return [((row.name,), (i,)) for i, row in enumerate(job.rows) if row.name]

    def lock_values(self, job, rows, answer: str):
        """The values a lock with this answer gives the unit's rows (one dict per row), or why the answer names
        nothing valid (str)."""
        raise NotImplementedError

    def answer_text(self, job, index: int, values: dict) -> str:
        """Row `index`'s answer as the Overrides CSV writes it: what lock_values() reads back."""
        raise NotImplementedError

    def export_unit(self, job, rows, table) -> tuple[str, float | None, str]:
        """(answer, confidence, source) of a unit for the Overrides CSV; table: (source, values) of every row."""
        source, values = table[rows[0]]
        if not values:  # pending: nothing to write yet
            return "", None, source
        return self.answer_text(job, rows[0], values), values.get("c"), source

    def overrides(self, node, ctx):
        """The Overrides CSV of ctx: (CsvSource, CsvDoc), or None when the node has none. Raises Hint or
        Unresolvable."""
        from .. import lists  # imports bpy

        source = lists.csv_source(node, "Overrides", ctx)
        return None if source is None else (source, lists.load_csv(node, source))

    def overrides_stamp(self, node, ctx):
        """The Overrides CSV in the fingerprint, so an edit plans again: (path, delimiter, stamp)."""
        from ..evaluator import Hint, Pending, Unresolvable  # imports bpy

        try:
            found = self.overrides(node, ctx)
        except (Hint, Pending, Unresolvable) as error:  # plan() raises it again, and the plan cache keeps it
            return f"{type(error).__name__}: {error}"
        return None if found is None else [found[1].path, found[0].delimiter, list(found[1].stamp)]

    def locked(self, node, ctx, job):
        """The job with the node's Overrides CSV applied (lock()). Raises FileError for a file with errors."""
        from .. import lists  # imports bpy

        found = self.overrides(node, ctx)
        if found is None:
            return job
        source, doc = found
        try:
            locks = csvfmt.parse_overrides(doc, self.override_keys(job))
        except csvfmt.CsvError as error:
            raise lists.file_error(doc, error) from None
        job = self.lock(job, locks, doc.path)
        if source.warning:
            job.problems += (source.warning,)
        return job

    def lock(self, job, locks: dict, path: str = ""):
        """Apply csvfmt.parse_overrides() locks to the job. A valid lock makes its rows lock rows (c = 1, never
        asked) and enters the words key through lock_digest; an invalid answer is an error on its row, whose model
        answer stays. job.locks maps each lock to "" when a row uses it, else to its orphaned-lock warning. Pure."""
        name = os.path.basename(path) or "Overrides"
        rows, errors, used = list(job.rows), [], set()
        for cells, indices in self.override_units(job):
            key = csvfmt.KEY_SEP.join(cell.strip().casefold() for cell in cells)
            lock = locks.get(key)
            if lock is None:
                continue
            used.add(key)
            answer = lock.answer.strip()
            values = self.lock_values(job, indices, answer) if answer else "the answer is empty"
            if isinstance(values, str):
                errors.append(f"{name} line {lock.line}: {values}; the model's answer stays")
                continue
            for i, value in zip(indices, values):
                rows[i] = dataclasses.replace(rows[i], source="lock", values=value, ask="", cell=None)
        locked = [[i, row.values] for i, row in enumerate(rows) if row.source == "lock"]
        words_key = job.words_key
        if locked:  # the unlocked words key stands in for the CSV hashes it already covers
            words_key = pipeline.words_key(job.gn, [job.words_key], job.answer_type, job.shape.digest(),
                                           [row.wkey for row in rows], pipeline.digest("locks", locked))
        marks = {f"{path}{csvfmt.KEY_SEP}{key}": "" if key in used else
                 f"{name} line {lock.line}: orphaned lock '{key.replace(csvfmt.KEY_SEP, ', ')}' matches no row"
                 for key, lock in locks.items()}
        return dataclasses.replace(job, rows=rows, words_key=words_key, problems=(*errors, *job.problems),
                                   locks=marks)

    def output(self, entry, name: str):
        """A single-value output's value in a stored entry, for chaining. KeyError: not a single value."""
        return entry.single[name]

    def fallback(self, name: str):
        return self.FALLBACK.get(name)

    def summary(self, entry) -> str:
        """The answer in a few words, for the status line and the sidebar."""
        raise NotImplementedError

    def node_warnings(self, node) -> tuple[str, ...]:
        """Warnings about the node itself once it is built, for its status."""
        return ()

    def build(self, b, gi, go, entries) -> None:
        """Lay out the inner group's nodes for these entries (inner.build())."""
        raise NotImplementedError
