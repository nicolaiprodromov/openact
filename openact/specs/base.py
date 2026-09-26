"""What every node spec shares (§10.2): the Spec interface, the helpers behind the shared formats (4.g), and the
validity rule that answers a context's table from stored rows and cached cells (§5.4).

Pure Python when imported, so the pure suites plan from plain words; the methods that read a node import the
evaluator, and with it bpy, when they run.
"""

from __future__ import annotations

import dataclasses
import math
import time

from .. import pipeline, store

NO_CONTEXT = "No additional context."
KEY = ('INPUT', "Key", 'NodeSocketString',
       "Tells apart objects that share every other word, e.g. typed per object on the modifier. Compared, never "
       "sent to the model", "word", True)  # the last word of every node (§6.0), hidden until Ctrl H
SOURCES = {"lock": "l", "code": "c", "model": "m", "pending": "p", "stale": "s", "error": "e"}  # Row.source -> src


def option_text(option) -> str:
    """An option (name, when) as the model reads it: "name: when", or the name alone."""
    name, when = option
    return f"{name}: {when}" if when else name


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
    SOCKETS: tuple = ()  # SocketDef arguments of the node's sockets, but Key
    FALLBACK: dict = {}  # single-value output -> its value before the first answer (§6.0)
    empty_hint = ""

    def words(self, node) -> tuple[str, ...]:
        """The String words the node can have, whatever its dropdowns (§8.2)."""
        return self.WORDS

    def interface(self, node) -> list:
        """The sockets the node should have: SocketDefs for inner.sync_interface()."""
        from ..inner import SocketDef  # imports bpy

        return [SocketDef(*args) for args in (*self.SOCKETS, KEY)]

    def gn(self, node, ctx) -> tuple[str, ...]:
        """The raw words in ctx, as the match chain compares them: "" for a word the node has no socket for."""
        from ..evaluator import as_text, eval_input  # imports bpy

        sockets = [node.inputs.get(name) for name in self.words(node)]
        return tuple("" if socket is None else as_text(eval_input(socket, ctx)) for socket in sockets)

    def listed(self, node, ctx):
        """What a list source adds to the fingerprint, e.g. a collection's children."""
        return None

    def fingerprint(self, node, ctx) -> str:
        """Everything plan() reads, hashed: while it is unchanged, the Job planned from it still holds (§5.3)."""
        return pipeline.digest(self.name, self.PROMPT_VERSION, self.gn(node, ctx), getattr(node, "answer_type", ""),
                               node.balance_order, self.listed(node, ctx))

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
            old = carried.get(row.wkey)
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
        """An answered table as the entry the node stores (§14.2): columnar rows and the ask keys behind them."""
        asks = list(dict.fromkeys(row.ask for row in rows))
        return store.Entry(
            words_key=job.words_key, asks=asks, pv=self.PROMPT_VERSION, t=time.time(), gn=job.gn, model=model,
            kind=self.name, answer_type=job.answer_type, shape=job.shape.as_dict(),
            rows=store.Rows(name=[row.name for row in rows], wkey=[row.wkey for row in rows],
                            ask=[asks.index(row.ask) for row in rows], src="".join(SOURCES[row.source] for row in rows),
                            v={column: [row.values[column] for row in rows] for column, _type in job.shape.columns}),
            single=self.single(rows))

    def output(self, entry, name: str):
        """A single-value output's value in a stored entry, for chaining. KeyError: not a single value."""
        return entry.single[name]

    def fallback(self, name: str):
        return self.FALLBACK.get(name)

    def summary(self, entry) -> str:
        """The answer in a few words, for the status line and the sidebar."""
        raise NotImplementedError

    def word_sockets(self, gi) -> list:
        """The inner group's input of each word, None for a word the node has no socket for."""
        return [gi.outputs.get(name) for name in self.WORDS]

    def build(self, b, gi, go, entries) -> None:
        """Lay out the inner group's nodes for these entries (inner.build())."""
        raise NotImplementedError
