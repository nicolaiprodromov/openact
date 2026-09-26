"""The Rules engine in pure Python (design 6.5 and 9): cells, `if` restrictions, the question for one cell, the limits,
and the references for interpolation and dithering that the inner graph must match within 1e-5.

Only the standard library, csvfmt and pipeline are imported, so the unit tests also run under plain CPython.
specs/rules.py builds the Rules Spec on it. Model rows take their confidence from specs/base.py's confidence() (4.e);
this module only sets it for rows that code decides (1) and blends it like the value (9.6).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import NamedTuple

from . import csvfmt, pipeline
from .csvfmt import ERROR, WARNING, CsvError, Problem

PROMPT_VERSION = 1  # stored per entry as "pv"; never part of a key (5.4)
QUESTION_TYPES = {"ONE_OF": "choice", "YES_NO": "noul", "AMOUNT": "score"}
NO_CONTEXT = "No additional context."
FACTS_INTRO = "Each question describes one situation with these facts:"
SITUATION = "Situation:"
DEFAULT_QUESTION = "Which outcome applies in this situation?"  # One-of only
QUESTION_HINT = "Type a question"  # Yes/No and Amount: a scale without a question is meaningless
MAX_SCORED_ROWS = 8192  # per context, before any request (5.4)
MAX_INTERPOLATED = 3  # Amount factors that Yes/No and Amount interpolate over; later ones use the nearest band
BLEND_EPSILON = 1e-6  # a blend whose answered corners weigh no more than this gives the fallbacks
DITHER_SEED_STEP = 7919  # factor j's Random Value uses Seed + 7919 * j (j in factor order)


class RulesError(ValueError):
    """A limit that stops planning: the node shows the message, keeps its last good table and sends nothing."""


class RulesHint(Exception):
    """Planning needs more words: the node shows the message as a hint (evaluator.Hint)."""


class IfError(ValueError):
    """An `if` expression with a syntax error or an unknown factor or level. The message names no line."""


class Clause(NamedTuple):
    factor: int  # index in factor order
    negate: bool  # "!=": the cell's level must not be one of levels
    levels: frozenset  # level indices


@dataclass(frozen=True)
class CellPlan:
    levels: tuple[int, ...]  # level index per factor, in factor order
    name: str  # row identity (Row.name): level names joined by "|", e.g. "alive|3"
    allowed: tuple[int, ...]  # option rows offered to the cell, in row order (Amount: every level)
    source: str  # "code" | "error" | "model" (the cell needs its question answered: cache or model)
    values: dict | None  # row values for code and error rows; None for model rows
    question: dict | None  # {"type", "instructions", "criteria"} for model rows; None otherwise
    cost: int  # scored rows the question costs (10.3); 0 for code and error rows


@dataclass(frozen=True)
class RulesPlan:
    state: str  # shared by every cell of the context
    cells: tuple[CellPlan, ...]  # one per table row, in cell order: cell r is table row r
    rows: int  # estimated scored rows: the sum of the cells' costs
    error: str  # "" or the status for cells no outcome applies to (9.4); their rows are error rows
    warnings: tuple[Problem, ...]  # the table is used; the status shows them after the error, once it is complete


# ------------------------------------------------------------------------ cells (9.3)


def level_counts(factors) -> tuple[int, ...]:
    return tuple(len(factor.levels) for factor in factors)


def strides(counts) -> tuple[int, ...]:
    """Row-major, so the last factor varies fastest: stride[K-1] = 1, stride[j] = stride[j+1] * n[j+1]."""
    out = [1] * len(counts)
    for j in range(len(counts) - 2, -1, -1):
        out[j] = out[j + 1] * counts[j + 1]
    return tuple(out)


def cell_rows(counts):
    """The level indices of every cell in table-row order: itertools.product order, which is a C-order ravel."""
    return itertools.product(*(range(n) for n in counts))


def cell_index(levels, counts) -> int:
    """cell = sum of levels[j] * stride[j]; -1 when a level is None (a category value that matches no level)."""
    if any(b is None for b in levels):
        return -1
    return sum(b * stride for b, stride in zip(levels, strides(counts)))


def cell_levels(cell: int, counts) -> tuple[int, ...]:
    """The level indices of a flat cell: the inverse of cell_index."""
    levels = []
    for n in reversed(counts):
        cell, b = divmod(cell, n)
        levels.append(b)
    return tuple(reversed(levels))


def row_name(factors, levels) -> str:
    return "|".join(factor.levels[b].name for factor, b in zip(factors, levels))


# ------------------------------------------------------------------------ level index (9.2)


def band_edges(factor) -> tuple[float, ...]:
    """An Amount factor's n + 1 band edges, from 0 to 1."""
    n = len(factor.levels)
    if factor.levels[0].band is None:
        return tuple(b / n for b in range(n + 1))
    return tuple(level.band[0] for level in factor.levels) + (factor.levels[-1].band[1],)


def band_centres(factor) -> tuple[float, ...]:
    edges = band_edges(factor)
    return tuple((low + high) / 2 for low, high in zip(edges, edges[1:]))


def band_width(factor, b: int) -> float:
    edges = band_edges(factor)
    return edges[b + 1] - edges[b]


def level_index(factor, value) -> int | None:
    """The level of one socket value; None when a category value matches no level (the cell is invalid).

    amount: the band that holds clamp(value, 0, 1), the last band including 1. category: the level with that value.
    flag: the level whose value is 1 when the value is true, else the level whose value is 0.
    """
    if factor.kind == "amount":
        x = _clamp01(value)
        if factor.levels[0].band is None:
            n = len(factor.levels)
            return min(math.floor(x * n), n - 1)
        return sum(x >= level.band[0] for level in factor.levels[1:])
    if factor.kind == "flag":
        value = 1 if value else 0
    return next((b for b, level in enumerate(factor.levels) if level.value == value), None)


def category_map(factor) -> tuple[int, ...] | None:
    """None when the category's values count up by one (the graph subtracts u_0), else the map rows the table object
    stores after the cell rows: the level of each value 0..u_max, or -1 where no level has the value."""
    values = [level.value for level in factor.levels]
    if all(value == values[0] + i for i, value in enumerate(values)):
        return None
    rows = [-1] * (max(values) + 1)
    for b, value in enumerate(values):
        rows[value] = b
    return tuple(rows)


def cell_of(factors, values) -> int:
    """The cell of one element with these socket values (one per factor), every Amount factor at its nearest band;
    -1 when a category value matches no level."""
    return cell_index([level_index(f, v) for f, v in zip(factors, values)], level_counts(factors))


# ------------------------------------------------------------------------ `if` restrictions (9.4)


def parse_if(text: str, factors) -> tuple[Clause, ...]:
    """The clauses of one `if` expression, all of which must hold; () when it is empty.

    Grammar: clause (";" clause)*, where clause := factor ("=" | "!=") level ("|" level)*. Names are trimmed and
    matched ignoring case. A category or flag level may also be given by its value. Raises IfError.
    """
    if not text.strip():
        return ()
    names = {factor.name.casefold(): j for j, factor in enumerate(factors)}
    clauses = []
    for part in text.split(";"):
        clause = part.strip()
        if not clause:
            raise IfError("'if' has an empty clause")
        at = clause.find("=")
        if at < 0:
            raise IfError(f"'if' clause '{clause}' needs '=' or '!='")
        negate = clause[at - 1 : at] == "!"
        name = clause[: at - negate].strip()
        if not name:
            raise IfError(f"'if' clause '{clause}' names no factor")
        j = names.get(name.casefold())
        if j is None:
            raise IfError(f"'if' names unknown factor '{name}'")
        levels = set()
        for token in clause[at + 1 :].split("|"):
            level = token.strip()
            if not level:
                raise IfError(f"'if' clause '{clause}' has an empty level")
            b = _find_level(factors[j], level)
            if b is None:
                raise IfError(f"'if' names unknown level '{level}' of factor '{factors[j].name}'")
            levels.add(b)
        clauses.append(Clause(j, negate, frozenset(levels)))
    return tuple(clauses)


def restrictions(answers, factors) -> list[tuple[Clause, ...]]:
    """Every option's parsed `if`, in row order. Raises CsvError naming the line of each bad expression (7.9)."""
    problems, parsed = [], []
    for r, option in enumerate(answers.options):
        try:
            parsed.append(parse_if(option.if_expr, factors))
        except IfError as error:
            problems.append(_problem(ERROR, answers.doc, r, str(error), prefix=True))
            parsed.append(())
    if problems:
        raise CsvError(problems)
    return parsed


def allowed_sets(clauses, counts) -> list[tuple[int, ...]]:
    """Each cell's allowed set, in cell order: the rows of the options whose clauses the cell satisfies."""
    everyone = (1 << len(clauses)) - 1
    masks = [[everyone] * n for n in counts]  # masks[j][b]: bit r is set while option r allows level b of factor j
    for r, option in enumerate(clauses):
        for clause in option:
            for b in range(counts[clause.factor]):
                if (b in clause.levels) == clause.negate:
                    masks[clause.factor][b] &= ~(1 << r)
    decoded, out = {}, []
    for levels in cell_rows(counts):
        mask = everyone
        for j, b in enumerate(levels):
            mask &= masks[j][b]
        if mask not in decoded:
            rows, rest = [], mask
            while rest:  # visits only the set bits, lowest first
                low = rest & -rest
                rows.append(low.bit_length() - 1)
                rest ^= low
            decoded[mask] = tuple(rows)
        out.append(decoded[mask])
    return out


# ------------------------------------------------------------------------ the question for one cell (9.5)


def state_text(context: str, factors) -> str:
    lines = [context.strip() or NO_CONTEXT, FACTS_INTRO]
    lines += [f"- {factor.name}: " + " | ".join(level.name for level in factor.levels) for factor in factors]
    return "\n".join(lines)


def instructions(factors, levels, question: str) -> str:
    """The cell's facts, then the question (already defaulted: see plan_table)."""
    facts = [_fact(factor.name, factor.levels[b]) for factor, b in zip(factors, levels)]
    return "\n".join([SITUATION, *facts, question])


def criteria(answer_type: str, options, allowed):
    """One-of: {row: "answer: when"} over the allowed rows. Yes/No: {"true": yes when, "false": no when}, "Yes." and
    "No." when absent. Amount: every level as "name: when", lowest first."""
    if answer_type == "ONE_OF":
        return {str(r): _option_text(options[r]) for r in allowed}
    if answer_type == "YES_NO":
        whens = {option.name.casefold(): option.when for option in options}
        return {"true": whens.get("yes") or "Yes.", "false": whens.get("no") or "No."}
    return [_option_text(option) for option in options]


# ------------------------------------------------------------------------ planning (9.1, steps 2-4)


def plan_table(factors, answer_type: str, answers, question: str, context: str, debias: bool = True) -> RulesPlan:
    """Every cell of one context: decided by code, an error row, or a question for the model.

    factors: csvfmt.parse_factors(), which keeps the cells within csvfmt.MAX_CELLS. answers: the Outcomes AnswerSet
    (sources.answer_set(), whose default sets stand in for an empty Yes/No or Amount input). question, context: the
    words as evaluated. debias: the node's balance_order. Raises RulesHint (empty Question for Yes/No or Amount),
    CsvError (a bad `if`, naming its line) or RulesError (over MAX_SCORED_ROWS).
    """
    counts = level_counts(factors)
    text = question.strip() or (DEFAULT_QUESTION if answer_type == "ONE_OF" else "")
    if not text:
        raise RulesHint(QUESTION_HINT)
    options, doc = answers.options, answers.doc
    warnings = []
    if answer_type == "AMOUNT":
        clauses = [()] * len(options)
        used = [r for r, option in enumerate(options) if option.if_expr]
        if used:
            warnings.append(_problem(WARNING, doc, used[0], "'if' is not used for Amount answers"))
    else:
        clauses = restrictions(answers, factors)
    allowed = allowed_sets(clauses, counts)
    offered = set().union(*allowed)
    for r, option in enumerate(options):
        if r not in offered:
            warnings.append(_problem(WARNING, doc, r, f"'{option.name}' is never offered: its 'if' matches no cell"))
    if answer_type != "ONE_OF":
        for factor in [factor for factor in factors if factor.kind == "amount"][MAX_INTERPOLATED:]:
            message = f"interpolating over {MAX_INTERPOLATED} Amount factors; '{factor.name}' uses its nearest band"
            warnings.append(Problem(WARNING, "", 0, "", message))
    rows = sum(pipeline.cost(len(a), debias) for a in allowed if len(a) > 1)
    if rows > MAX_SCORED_ROWS:
        limit = f"(limit {csvfmt._n(MAX_SCORED_ROWS)})"
        raise RulesError(f"Rules needs {csvfmt._n(rows)} scored rows {limit}: offer at most 16 outcomes per cell, "
                         f"or use fewer bands")

    cells, empty, shared = [], [], {}  # shared: one criteria object per allowed set (never mutated)
    for levels, offer in zip(cell_rows(counts), allowed):
        name = row_name(factors, levels)
        if len(offer) == 1:
            values = _code_values(answer_type, options, offer[0])
            cells.append(CellPlan(levels, name, offer, "code", values, None, 0))
        elif not offer:
            empty.append(levels)
            cells.append(CellPlan(levels, name, offer, "error", _error_values(answer_type), None, 0))
        else:
            if offer not in shared:
                shared[offer] = criteria(answer_type, options, offer)
            asked = {
                "type": QUESTION_TYPES[answer_type],
                "instructions": instructions(factors, levels, text),
                "criteria": shared[offer],
            }
            cells.append(CellPlan(levels, name, offer, "model", None, asked, pipeline.cost(len(offer), debias)))
    error = ""
    if empty:
        shown = [", ".join(f"{f.name}={f.levels[b].name}" for f, b in zip(factors, levels)) for levels in empty[:3]]
        more = len(empty) - len(shown)
        error = "No outcome applies to " + "; ".join(shown)
        error += f" and {more} more cell{'s' if more > 1 else ''}" if more else ""
    return RulesPlan(state_text(context, factors), tuple(cells), rows, error, tuple(warnings))


# ------------------------------------------------------------------------ interpolation (9.6)


def interpolated_factors(factors) -> tuple[int, ...]:
    """The factors Yes/No and Amount interpolate over: the first 3 Amount factors, in factor order."""
    return tuple(j for j, factor in enumerate(factors) if factor.kind == "amount")[:MAX_INTERPOLATED]


def axis(factor, value) -> tuple[int, float]:
    """(b0, t) of an interpolated Amount factor: the lower corner's band, and the weight of the upper corner."""
    n = len(factor.levels)
    x = _clamp01(value)
    if factor.levels[0].band is None:
        u = min(max(x * n - 0.5, 0.0), n - 1.0)
        b0 = min(math.floor(u), n - 2)
        return b0, u - b0
    centres = band_centres(factor)
    b0 = sum(x >= centres[b] for b in range(1, n - 1))
    return b0, min(max((x - centres[b0]) / (centres[b0 + 1] - centres[b0]), 0.0), 1.0)


def corners(factors, values) -> list[tuple[int, float]]:
    """(cell, weight) of the 2^Ka corners, s in itertools.product((0, 1), repeat=Ka) order, bit j for the j-th
    interpolated factor (interpolated_factors()). The other factors take their level (Amount: nearest band). [] when a
    category value matches no level."""
    interpolate = interpolated_factors(factors)
    steps = strides(level_counts(factors))
    exact, axes = 0, []
    for j, (factor, value) in enumerate(zip(factors, values)):
        if j in interpolate:
            axes.append((steps[j], *axis(factor, value)))
            continue
        b = level_index(factor, value)
        if b is None:
            return []
        exact += b * steps[j]
    out = []
    for s in itertools.product((0, 1), repeat=len(axes)):
        cell, weight = exact, 1.0
        for bit, (stride, b0, t) in zip(s, axes):
            cell += (b0 + bit) * stride
            weight *= t if bit else 1.0 - t
        out.append((cell, weight))
    return out


def blend(factors, values, table) -> tuple[float, float, bool]:
    """(value, confidence, valid) of one element: the corner rows blended with weight w * ok, renormalised.

    table[cell] = (v, c, ok), v being p (Yes/No) or amount. Renormalising stops a pending corner from dragging the
    value towards 0 while a table fills. Invalid, (0.0, 0.0, False), when a category value matches no level or the
    answered corners weigh no more than BLEND_EPSILON. Yes = value >= 0.5 is decided after the blend.
    """
    total = value = confidence = 0.0
    for cell, weight in corners(factors, values):
        v, c, ok = table[cell]
        total += weight * ok
        value += weight * ok * v
        confidence += weight * ok * c
    if total <= BLEND_EPSILON:
        return 0.0, 0.0, False
    return value / total, confidence / total, True


# ------------------------------------------------------------------------ dithering (9.7, One-of only)


def dithered_level(factor, value, dither: float, r: float) -> int:
    """The band of an Amount value jittered by (r - 0.5) * Dither widths of its undithered band; r is the element's
    Random Value (0..1)."""
    x = _clamp01(value)
    width = band_width(factor, level_index(factor, x))
    return level_index(factor, _clamp01(x + (r - 0.5) * dither * width))


def dither_distribution(factor, value, dither: float) -> tuple[float, ...]:
    """P(band) of dithered_level for r uniform on 0..1: the share of the jitter interval in each band. The clamp moves
    what falls below 0 or above 1 into the first or last band."""
    n = len(factor.levels)
    x = _clamp01(value)
    b = level_index(factor, x)
    half = dither * band_width(factor, b) / 2
    if half <= 0:
        return tuple(float(k == b) for k in range(n))
    edges = band_edges(factor)
    low, high = x - half, x + half
    probabilities = []
    for k in range(n):
        start = edges[k] if k > 0 else -math.inf
        end = edges[k + 1] if k < n - 1 else math.inf
        probabilities.append(max(0.0, min(end, high) - max(start, low)) / (2 * half))
    return tuple(probabilities)


def dithered_cell(factors, values, dither: float, r) -> int:
    """The One-of cell of one element: r[j] is factor j's Random Value, read for Amount factors only (categories and
    flags never dither). -1 when a category value matches no level."""
    levels = [
        dithered_level(factor, value, dither, r[j]) if factor.kind == "amount" else level_index(factor, value)
        for j, (factor, value) in enumerate(zip(factors, values))
    ]
    return cell_index(levels, level_counts(factors))


def dither_cells(factors, values, dither: float) -> dict[int, float]:
    """P(cell) of dithered_cell with independent uniform r; {} when a category value matches no level."""
    steps = strides(level_counts(factors))
    distribution = {0: 1.0}
    for j, (factor, value) in enumerate(zip(factors, values)):
        if factor.kind == "amount":
            spread = [(b, p) for b, p in enumerate(dither_distribution(factor, value, dither)) if p > 0]
        else:
            b = level_index(factor, value)
            if b is None:
                return {}
            spread = [(b, 1.0)]
        distribution = {cell + b * steps[j]: p * q for cell, p in distribution.items() for b, q in spread}
    return distribution


# ------------------------------------------------------------------------ helpers


def _clamp01(value) -> float:
    return min(max(float(value), 0.0), 1.0)


def _find_level(factor, text):
    folded = text.casefold()
    for b, level in enumerate(factor.levels):
        if level.name.casefold() == folded:
            return b
    value = csvfmt._int32(text) if factor.kind != "amount" else None
    return next((b for b, level in enumerate(factor.levels) if value is not None and level.value == value), None)


def _fact(name, level):
    return f"- {name}: {level.name} ({level.when})" if level.when else f"- {name}: {level.name}"


def _option_text(option):
    """As specs/base.option_text: "name: when", or the name alone."""
    return f"{option.name}: {option.when}" if option.when else option.name


def value_column(answer_type: str) -> str:
    """The value column of a Yes/No or Amount table (9.6); One-of's is "outcome"."""
    return {"ONE_OF": "outcome", "YES_NO": "p", "AMOUNT": "amount"}[answer_type]


def _code_values(answer_type, options, r):
    if answer_type == "ONE_OF":
        return {"outcome": r, "c": 1.0}
    if answer_type == "AMOUNT":  # a single level: its position is 0
        return {"amount": 0.0, "c": 1.0, "ok": 1.0}
    return {"p": 1.0 if options[r].name.casefold() == "yes" else 0.0, "c": 1.0, "ok": 1.0}


def _error_values(answer_type):
    if answer_type == "ONE_OF":
        return {"outcome": -1, "c": 0.0}
    return {value_column(answer_type): 0.0, "c": 0.0, "ok": 0.0}


def _problem(severity, doc, r, message, prefix=False):
    """A problem about option row r of the Outcomes document (None for an inline set), in its 'if' column."""
    if doc is None:
        return Problem(severity, "", 0, "if", message)
    line = doc.lines[r]
    column = next((c for c in doc.header if c.casefold() == "if"), "if")
    return Problem(severity, doc.path, line, column, f"Line {line}: {message}" if prefix else message)
