"""Planning helpers shared by the node specs (design 4.g, 10.2): CSV companion sockets, list-source sockets, answer
sets, and the per-item cells of Match and Classify.

Their home is specs/base.py (design 13.3), which PR-09 creates; they wait here as functions and data that the specs
call. Only answer_set() and the socket helpers need Blender, so the pure suites can import this module.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace

from . import csvfmt
from .csvfmt import AnswerSet, Option
from .specs.base import option_text  # noqa: F401  (the one definition; callers import it from either module)

try:
    import bpy
except ImportError:  # the pure suites
    bpy = None
if bpy is not None:
    from . import lists
    from .evaluator import Hint, Unresolvable, as_text, eval_input
    from .inner import SocketDef

NO_CONTEXT = "No additional context."
CLASSIFY_QUESTION = "Which label fits this item best?"
AMOUNT_LEVELS = ("Not at all", "Slightly", "Moderately", "Very", "Extremely")
KINDS = {'YES_NO': "yes_no", 'AMOUNT': "amount", 'ONE_OF': "one_of"}  # answer_type -> answer-set kind
QUESTION_TYPES = {"yes_no": "noul", "amount": "score", "one_of": "choice"}
GROUP = 16  # options per tournament group: more run as a tournament (design 10.3)
LIST_SOURCE_ITEMS = (  # the list_source dropdown (design 6.0)
    ('TEXT', "Text or CSV", "Items typed inline, or the rows of a CSV file"),
    ('COLLECTION', "Collection", "The children of a collection, in Collection Info order"),
    ('MATERIALS', "Materials", "The material slots of an object: ID = material index"),
    ('ATTRIBUTES', "Attributes", "The attribute names of an object's own data"),
)


# ----------------------------------------------------------------------------- sockets


def csv_inputs(name: str, description: str, hidden: bool = False) -> list:
    """A CSV-capable String word and its '<name> CSV' Geometry companion, directly below it (design 6.0).

    Every node declares its CSV inputs this way. The inner group leaves the companion unconnected, so Geometry
    Nodes never evaluates an Import CSV node plugged into it; lists.csv_source() reads that node's Path instead."""
    return [
        SocketDef('INPUT', name, 'NodeSocketString', description, kind="word", hidden=hidden),
        SocketDef('INPUT', f"{name} CSV", 'NodeSocketGeometry',
                  f"Or plug an Import CSV node here: OpenAct reads the file at its Path, with its Delimiter. "
                  f"It wins over a path in {name}", kind="csv", hidden=hidden),
    ]


def list_inputs(list_source: str) -> list:
    """The sockets of a list source, swapped by the list_source dropdown (design 6.0)."""
    if list_source == 'TEXT':
        return csv_inputs("Items", "The items: inline ('chair, table', or 'Pine: grows in cold places; ...'), "
                                   "or a .csv path with an 'answer' column")
    if list_source == 'COLLECTION':
        return [SocketDef('INPUT', "Collection", 'NodeSocketCollection',
                          "The children of this collection, in the order Collection Info gives them", kind="word")]
    if list_source == 'MATERIALS':
        return [SocketDef('INPUT', "Object", 'NodeSocketObject',
                          "The material slots of this object; ID = material index", kind="word")]
    if list_source == 'ATTRIBUTES':
        return [SocketDef('INPUT', "Object", 'NodeSocketObject',
                          "The attribute names of this object's own data. Empty: the modifier's object", kind="word")]
    raise ValueError(f"unknown list source {list_source!r}")


# ------------------------------------------------------------------------- answer sets


def default_set(kind: str) -> AnswerSet | None:
    """What an empty input means: Yes/No without whens ('Yes.'/'No.'), the five AMOUNT_LEVELS; None for One-of."""
    if kind == "yes_no":
        return AnswerSet((Option("yes", "", "", {}), Option("no", "", "", {})), None)
    if kind == "amount":
        return AnswerSet(tuple(Option(level, "", "", {}) for level in AMOUNT_LEVELS), None)
    return None


def answer_set(node, ctx, socket, kind: str, spec: str = "", empty_hint: str = "") -> AnswerSet:
    """The answer set of a node input in this context (design 6.0, 7.3): the Import CSV node linked to its
    companion, a typed CSV path, inline text, or the default for an empty input.

    socket: the input or its name. kind: "yes_no" | "amount" | "one_of". spec: the node's Spec name, for the CSV
    rules of csvfmt.parse_answer_set(). An empty One-of input raises Hint(empty_hint). Raises Hint, Pending or
    Unresolvable (FileError for a CSV with errors)."""
    name = socket if isinstance(socket, str) else socket.name
    source = lists.csv_source(node, name, ctx)
    if source is not None:
        doc = lists.load_csv(node, source)
        try:
            answers = csvfmt.parse_answer_set(doc, kind, spec)
        except csvfmt.CsvError as error:
            raise lists.file_error(doc, error) from None
        problems = answers.problems
        if source.warning:
            problems += (csvfmt.Problem(csvfmt.WARNING, doc.path, 0, "", source.warning),)
        return replace(answers, problems=problems, source=source)
    sock = lists.find_input(node, name)
    text = as_text(eval_input(sock, ctx)) if sock is not None else ""
    if not text.strip():
        default = default_set(kind)
        if default is None:
            raise Hint(empty_hint or f"Add {name.lower()}")
        return default
    try:
        return csvfmt.parse_inline_set(text, kind)
    except csvfmt.CsvError as error:
        raise Unresolvable(f"{name}: {error}") from None


def criteria(answers: AnswerSet, kind: str):
    """A question's criteria for an answer set: the Yes/No whens as noul 'true'/'false' ('Yes.'/'No.' without one),
    the Amount levels lowest first, the One-of options keyed by row index."""
    if kind == "yes_no":
        whens = {option.name.casefold(): option.when for option in answers.options}
        return {"true": whens.get("yes") or "Yes.", "false": whens.get("no") or "No."}
    if kind == "amount":
        return [option_text(option) for option in answers.options]
    return {str(i): option_text(option) for i, option in enumerate(answers.options)}


def question(kind: str, instructions: str, answers: AnswerSet) -> dict:
    """The question for one cell: {"type": "noul" | "score" | "choice", "instructions", "criteria"}."""
    return {"type": QUESTION_TYPES[kind], "instructions": instructions, "criteria": criteria(answers, kind)}


def decided(answers: AnswerSet) -> bool:
    """A single-option set is decided by code and never sent (4.d)."""
    return len(answers.options) == 1


def amount_value(score: float, levels: int) -> float:
    """An Amount answer in 0-1: the server's probability-weighted level position over L - 1."""
    return min(max(float(score) / (levels - 1), 0.0), 1.0)


def cost(question: dict, debias: bool) -> int:
    """Scored rows of one question: (2 if debias else 1) * (groups + 1 when it runs as a tournament)."""
    options = len(question["criteria"])
    rounds = math.ceil(options / GROUP) + 1 if question["type"] == "choice" and options > GROUP else 1
    return (2 if debias else 1) * rounds


# ---------------------------------------------------------------------- per-item cells


@dataclass(frozen=True)
class ItemCell:
    """One row of a Match or Classify table and the question it needs: pipeline.Row and Cell (PR-09) in waiting."""

    index: int  # the item's ID
    name: str  # the row name: the item's name
    wkey: str  # what the artist said for this row (design 5.4); Key never enters it
    source: str  # "model" | "code" (a single label) | "fallback" (an empty material slot: never asked or flagged)
    values: dict  # a code row's answer, {"label": 0, "c": 1.0}; else {}
    key: str = ""  # the cell key of a model row
    state: str = ""
    question: dict | None = None
    cost: int = 0


def _key(*parts):
    return hashlib.sha1(json.dumps(parts, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def per_item_cells(spec: str, answer_type: str, items, answers: AnswerSet, question_text: str, context: str,
                   debias: bool) -> list[ItemCell]:
    """One row per item, all sharing one state (design 6.3, 6.4, 10.2): Match asks a Yes/No or Amount question about
    each item, Classify a choice among the labels.

    spec: "match" | "classify". answer_type: 'YES_NO' | 'AMOUNT' | 'ONE_OF'. items: a lists.ItemList, or its items.
    question_text and context: the words as given (Classify's empty question takes CLASSIFY_QUESTION)."""
    kind = KINDS[answer_type]
    question_text, context = question_text.strip(), context.strip()
    state = context or NO_CONTEXT
    asked = question_text or (CLASSIFY_QUESTION if spec == "classify" else "")
    offered = [[option.name, option.when] for option in answers.options]
    wanted = criteria(answers, kind)
    ordered = list(wanted.items()) if isinstance(wanted, dict) else wanted  # the cell key keeps option order
    rows = []
    for index, item in enumerate(getattr(items, "items", items)):
        wkey = _key("row", spec, answer_type, [context, question_text], [item.name, item.when], offered)
        if not item.offered:
            rows.append(ItemCell(index, item.name, wkey, "fallback", {}))
        elif kind == "one_of" and decided(answers):
            rows.append(ItemCell(index, item.name, wkey, "code", {"label": 0, "c": 1.0}))
        else:
            instructions = f"Item: {item.name}" + (f" — {item.when}" if item.when else "") + f"\n{asked}"
            asks = {"type": QUESTION_TYPES[kind], "instructions": instructions, "criteria": wanted}
            key = _key("cell", asks["type"], state, instructions, ordered, debias)
            rows.append(ItemCell(index, item.name, wkey, "model", {}, key, state, asks, cost(asks, debias)))
    return rows
