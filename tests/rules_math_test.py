"""The Rules engine's pure-Python core, openact/rules_core.py, with no Blender and no model.

Checks that itertools.product order equals a C-order ravel, the level and cell index references, the `if` allowed
sets, cells decided by code, error rows, the 9.5 question text of the worked example, the 9.8 limits, the 9.6
interpolation reference (hand-computed corners, renormalisation) and the 9.7 dithering probabilities against 100 000
seeded samples, plus PR-09's three 4.e confidence formulas (specs.base.confidence).

Run:  python tests/rules_math_test.py
      blender -b --factory-startup --python tests/rules_math_test.py
"""

import math
import os
import random
import shutil
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import ROOT, check, finish  # noqa: E402

# openact/__init__.py imports bpy, so the pure modules are loaded without running it.
_package = types.ModuleType("openact")
_package.__path__ = [os.path.join(ROOT, "openact")]
sys.modules["openact"] = _package
from openact import csvfmt  # noqa: E402
from openact import rules_core as rc  # noqa: E402
from openact.specs.base import confidence  # noqa: E402

try:
    import numpy
except ImportError:  # the pure-Python ravel still checks the order
    numpy = None

FIXTURES = os.path.join(ROOT, "tests", "fixtures")
TMP = tempfile.mkdtemp(prefix="openact_rules_")
SAMPLES = 100_000
GOL_CONTEXT = "A cellular automaton on a square grid; every cell has eight neighbours."
GOL_QUESTION = "What happens to this cell in the next generation?"
MIXED = (  # dead cells with 3 neighbours and live cells with 2 or 3 offer two outcomes, every other cell one
    "answer,when,if\nBirth,comes to life,state=dead;neighbours=3\nStays empty,stays empty,state=dead\n"
    "Survives,stays alive,state=alive;neighbours=2|3\nDies,dies,state=alive\n"
)


def write(name, content):
    path = os.path.join(TMP, name)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(content)
    return path


def factors_csv(name):
    return csvfmt.parse_factors(csvfmt.read_csv(os.path.join(FIXTURES, name), schema="factors"))


def answers(name, content=None, kind="one_of"):
    """An Outcomes answer set: a fixture, or a generated file."""
    path = os.path.join(FIXTURES, name) if content is None else write(name, content)
    return csvfmt.parse_answer_set(csvfmt.read_csv(path), kind, node="rules")


def inline(*names):
    return csvfmt.AnswerSet(tuple(csvfmt.Option(name, "", "", {}) for name in names), None)


def amount(name, n, edges=None):
    bands = [None] * n if edges is None else list(zip(edges, edges[1:]))
    return csvfmt.Factor(name, "amount", tuple(csvfmt.Level(f"{name}{b}", "", None, bands[b], 0) for b in range(n)), 0)


def category(name, values):
    return csvfmt.Factor(name, "category", tuple(csvfmt.Level(f"v{v}", "", v, None, 0) for v in values), 0)


def flag(name, values=(0, 1)):
    return csvfmt.Factor(name, "flag", tuple(csvfmt.Level(f"{name}{v}", "", v, None, 0) for v in values), 0)


def close(a, b, tolerance=1e-9):
    return abs(a - b) <= tolerance


def raised(function, *args, **kwargs):
    """The exception function raises, or None."""
    try:
        function(*args, **kwargs)
    except Exception as error:  # noqa: BLE001 - the checks look at its type and message
        return error
    return None


def summary(problems):
    return [(p.severity, p.line, p.message) for p in problems or ()]


def b3s23(alive, neighbours):
    """Game of Life's outcome row: Birth 0, Stays empty 1, Survives 2, Dies 3."""
    if alive:
        return 2 if neighbours in (2, 3) else 3
    return 0 if neighbours == 3 else 1


def ravel(shape):
    """The index tuples of a nested list of this shape, flattened in C order."""

    def nest(prefix):
        if len(prefix) == len(shape):
            return prefix
        return [nest(prefix + (i,)) for i in range(shape[len(prefix)])]

    def flatten(item):
        return [item] if isinstance(item, tuple) else [leaf for part in item for leaf in flatten(part)]

    return flatten(nest(()))


def test_cells():
    for shape in ((2, 9), (3, 3, 3), (2, 3, 4, 5), (7,), (2,) * 8, (16, 16, 16)):
        rows = list(rc.cell_rows(shape))
        good = len(rows) == math.prod(shape) and all(
            rc.cell_index(levels, shape) == r and rc.cell_levels(r, shape) == levels for r, levels in enumerate(rows)
        )
        check(f"cells {shape}: row r is cell r, and cell_levels inverts cell_index", good)
        check(f"cells {shape}: product order equals a C-order ravel", rows == ravel(shape))
        if numpy is not None:
            flat = numpy.ravel_multi_index(tuple(numpy.array(rows).T), shape).tolist()
            back = list(zip(*(axis.tolist() for axis in numpy.unravel_index(range(len(rows)), shape))))
            check(f"cells {shape}: numpy ravel and unravel agree", flat == list(range(len(rows))) and back == rows)
    check(
        "cells: Game of Life strides (9, 1), biome (9, 3, 1)",
        rc.strides((2, 9)) == (9, 1) and rc.strides((3,) * 3) == (9, 3, 1),
    )
    life, biome = factors_csv("life_factors.csv"), factors_csv("biome_factors.csv")
    check("cells: a live cell with 3 neighbours is cell 12", rc.cell_of(life, (True, 3)) == 12)
    check("cells: row 12 is alive|3", rc.row_name(life, rc.cell_levels(12, (2, 9))) == "alive|3")
    check("cells: biome (hot, wet, low) is cell 24", rc.cell_of(biome, (0.9, 0.9, 0.1)) == 24)
    invalid = rc.cell_of(life, (True, 9)), rc.cell_index((1, None), (2, 9))
    check("cells: a category value that matches no level gives -1", invalid == (-1, -1), invalid)

    equal = amount("a", 3)
    got = [rc.level_index(equal, x) for x in (-0.5, 0.0, 0.33, 0.34, 0.66, 0.67, 0.999, 1.0, 1.5)]
    check("level: equal-width bands, clamped, the last band including 1", got == [0, 0, 0, 1, 1, 2, 2, 2, 2], got)
    got = [rc.level_index(biome[0], x) for x in (0.0, 0.29, 0.3, 0.69, 0.7, 1.0)]
    check("level: custom edges count the interior 'from' values reached", got == [0, 0, 1, 1, 2, 2], got)
    sparse = category("c", (0, 5, 3))
    got = [rc.level_index(sparse, u) for u in (0, 5, 3, 4, -1, 4096)]
    check("level: category values, unknown ones invalid", got == [0, 1, 2, None, None, None], got)
    contiguous = rc.category_map(life[1]) is None and rc.category_map(category("c", (-3, -2, -1))) is None
    check("level: a contiguous category needs no map", contiguous)
    check("level: a non-contiguous category's map rows", rc.category_map(sparse) == (0, -1, -1, 2, -1, 1))
    got = [rc.level_index(flag("f", (1, 0)), v) for v in (True, False)] + [rc.level_index(life[0], v) for v in (0, 1)]
    check("level: a flag takes the level whose value matches", got == [0, 1, 0, 1], got)


def test_allowed():
    life = factors_csv("life_factors.csv")
    plan = rc.plan_table(life, "ONE_OF", answers("life_outcomes.csv"), GOL_QUESTION, GOL_CONTEXT)
    allowed = [cell.allowed for cell in plan.cells]
    check(
        "allowed: dead cells offer Birth and Stays empty, live ones Survives and Dies",
        allowed == [(0, 1)] * 9 + [(2, 3)] * 9,
    )
    sent = all(list(cell.question["criteria"]) == [str(r) for r in cell.allowed] for cell in plan.cells)
    check("allowed: only the allowed outcomes are sent, keyed by row", sent)
    good = len(plan.cells) == 18 and plan.rows == 36 and {cell.source for cell in plan.cells} == {"model"}
    check("allowed: 18 cells and 36 scored rows with debias", good and not plan.error and not plan.warnings, plan.rows)
    plain = rc.plan_table(life, "ONE_OF", answers("life_outcomes.csv"), GOL_QUESTION, GOL_CONTEXT, debias=False)
    check("allowed: 18 scored rows without debias", plain.rows == 18, plain.rows)
    plan = rc.plan_table(life, "ONE_OF", answers("mixed.csv", MIXED), GOL_QUESTION, GOL_CONTEXT)
    expected = [(0, 1) if n == 3 else (1,) for n in range(9)] + [(2, 3) if n in (2, 3) else (3,) for n in range(9)]
    check("allowed: sets of one and two outcomes", [cell.allowed for cell in plan.cells] == expected)
    rng, shape = random.Random(5), (3, 4, 5)
    clauses = []
    for _ in range(40):
        picked = rng.sample(range(3), rng.randint(0, 3))
        clauses.append(
            tuple(rc.Clause(j, rng.random() < 0.3, frozenset(rng.sample(range(shape[j]), 2))) for j in picked)
        )
    brute = [
        tuple(r for r, option in enumerate(clauses) if all((levels[c.factor] in c.levels) != c.negate for c in option))
        for levels in rc.cell_rows(shape)
    ]
    check(
        "allowed: the bit masks agree with checking every clause of every cell",
        rc.allowed_sets(clauses, shape) == brute,
    )
    open_ = rc.plan_table(life, "ONE_OF", inline("Birth", "Stays empty", "Survives", "Dies"), GOL_QUESTION, "")
    check(
        "allowed: without 'if' every cell offers every outcome",
        {cell.allowed for cell in open_.cells} == {(0, 1, 2, 3)},
    )


def test_code_decided():
    life = factors_csv("life_factors.csv")
    exact = answers(
        "exact.csv",
        "answer,when,if\nBirth,comes to life,state=dead;neighbours=3\n"
        "Stays empty,stays empty,state=dead; neighbours!=3\nSurvives,stays alive,state=alive;neighbours=2|3\n"
        "Dies,dies, state = ALIVE ; neighbours != 2 | 3 \n",
    )
    plan = rc.plan_table(life, "ONE_OF", exact, GOL_QUESTION, GOL_CONTEXT)
    got = [cell.values["outcome"] if cell.source == "code" else None for cell in plan.cells]
    check(
        "code: an exact rule set decides all 18 cells as B3/S23",
        got == [b3s23(a, n) for a in (0, 1) for n in range(9)],
        got,
    )
    good = all(
        c.values == {"outcome": c.allowed[0], "c": 1.0} and c.question is None and c.cost == 0 for c in plan.cells
    )
    check("code: confidence 1, no question and no scored rows", good and plan.rows == 0)
    plan = rc.plan_table(life, "ONE_OF", answers("mixed.csv", MIXED), GOL_QUESTION, GOL_CONTEXT)
    sources = "".join(cell.source[0] for cell in plan.cells)
    check("code: single-outcome cells are code, the rest model", sources == "cccmccccc" + "ccmmccccc", sources)
    check("code: 3 asked cells cost 6 scored rows", plan.rows == 6 and plan.cells[1].values["outcome"] == 1, plan.rows)

    yes_no = answers(
        "yes_no.csv", "answer,when,if\nno,it is empty next generation,\nyes,it is alive,state=alive\n", "yes_no"
    )
    plan = rc.plan_table(life, "YES_NO", yes_no, "Is this cell alive in the next generation?", GOL_CONTEXT)
    dead, alive = plan.cells[3], plan.cells[12]
    check(
        "code: Yes/No with only 'no' allowed gives p = 0",
        dead.source == "code" and dead.values == {"p": 0.0, "c": 1.0, "ok": 1.0},
    )
    good = alive.source == "model" and alive.question["type"] == "noul"
    whens = {"true": "it is alive", "false": "it is empty next generation"}
    check("code: Yes/No asks noul with both whens where both rows apply", good and alive.question["criteria"] == whens)
    three = answers("yes_no_three.csv", "answer,if\nyes,neighbours=3\nno,neighbours!=3\n", "yes_no")
    plan = rc.plan_table(life, "YES_NO", three, "Is it three?", "")
    got = [cell.values["p"] for cell in plan.cells]
    check(
        "code: Yes/No with only 'yes' allowed gives p = 1",
        got == [float(n == 3) for _ in (0, 1) for n in range(9)],
        got,
    )


def test_error_rows():
    life = factors_csv("life_factors.csv")
    missing = answers(
        "missing.csv",
        "answer,when,if\nBirth,comes to life,state=dead;neighbours=3\nSurvives,stays alive,state=alive;neighbours=2|3\n"
        "Dies,dies,state=alive;neighbours!=2|3\n",
    )
    plan = rc.plan_table(life, "ONE_OF", missing, GOL_QUESTION, GOL_CONTEXT)
    errors = [cell.index for cell in plan.cells if cell.source == "error"]
    check("error: dead cells without three neighbours have no outcome", errors == [0, 1, 2, 4, 5, 6, 7, 8], errors)
    rows = [cell for cell in plan.cells if cell.source == "error"]
    good = all(
        c.values == {"outcome": -1, "c": 0.0} and c.cost == 0 and c.allowed == () and c.question is None for c in rows
    )
    check("error: error rows hold the fallbacks and send nothing", good)
    message = "No outcome applies to state=dead, neighbours=0; state=dead, neighbours=1; state=dead, neighbours=2"
    check("error: the status lists the first 3 cells", plan.error == message + " and 5 more cells", plan.error)
    one = answers(
        "one_missing.csv",
        "answer,when,if\nBirth,comes to life,state=dead;neighbours=3\n"
        "Stays empty,stays empty,state=dead;neighbours!=3|4\nSurvives,stays alive,state=alive\n",
    )
    plan = rc.plan_table(life, "ONE_OF", one, GOL_QUESTION, GOL_CONTEXT)
    check(
        "7.9 a cell with no allowed outcome", plan.error == "No outcome applies to state=dead, neighbours=4", plan.error
    )
    yes_no = answers("yes_no_alive.csv", "answer,if\nyes,state=alive\nno,state=alive\n", "yes_no")
    plan = rc.plan_table(life, "YES_NO", yes_no, "Alive?", "")
    good = plan.cells[0].source == "error" and plan.cells[0].values == {"p": 0.0, "c": 0.0, "ok": 0.0}
    check("error: a Yes/No cell with neither row allowed", good and plan.cells[9].source == "model")

    ghost = answers(
        "ghost.csv", "answer,when,if\nBirth,x,state=dead\nGhost,never,state=dead;state=alive\nSurvives,y,state=alive\n"
    )
    plan = rc.plan_table(life, "ONE_OF", ghost, GOL_QUESTION, GOL_CONTEXT)
    warning = ("warning", 3, "'Ghost' is never offered: its 'if' matches no cell")
    check("7.9 an outcome never offered", summary(plan.warnings) == [warning], summary(plan.warnings))
    bad = answers(
        "bad_if.csv", "answer,when,if\nBirth,x,state=dead\nStays empty,y,state=dead\nSurvives,z,state=alvie\n"
    )
    error = raised(rc.plan_table, life, "ONE_OF", bad, GOL_QUESTION, GOL_CONTEXT)
    expected = [("error", 4, "Line 4: 'if' names unknown level 'alvie' of factor 'state'")]
    check(
        "error: a bad 'if' is a CsvError naming its line",
        isinstance(error, csvfmt.CsvError) and summary(error.problems) == expected,
    )


def test_templates():
    life = factors_csv("life_factors.csv")
    plan = rc.plan_table(life, "ONE_OF", answers("life_outcomes.csv"), GOL_QUESTION, GOL_CONTEXT)
    state = (
        "A cellular automaton on a square grid; every cell has eight neighbours.\n"
        "Each question describes one situation with these facts:\n"
        "- state: dead | alive\n"
        "- neighbours: 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8"
    )
    check("9.5 the shared state", plan.state == state, plan.state)
    question = {
        "type": "choice",
        "instructions": "Situation:\n"
        "- state: alive (The cell holds a living cell.)\n"
        "- neighbours: 3 (Exactly three live neighbours.)\n"
        "What happens to this cell in the next generation?",
        "criteria": {
            "2": "Survives: A live cell with two or three live neighbours stays alive.",
            "3": "Dies: A live cell with fewer than two or more than three live neighbours dies.",
        },
    }
    check("9.5 the question for cell 12", plan.cells[12].question == question, plan.cells[12].question)
    check("9.5 PROMPT_VERSION is 1", rc.PROMPT_VERSION == 1)
    plan = rc.plan_table([flag("wet")], "ONE_OF", inline("Grow", "Wait"), "  ", " ")
    good = (
        plan.state
        == "No additional context.\nEach question describes one situation with these facts:\n- wet: wet0 | wet1"
    )
    question = {"type": "choice", "instructions": "Situation:\n- wet: wet1\nWhich outcome applies in this situation?"}
    question["criteria"] = {"0": "Grow", "1": "Wait"}
    check(
        "9.5 no context, One-of's default question, bare names and levels", good and plan.cells[1].question == question
    )
    for answer_type in ("YES_NO", "AMOUNT"):
        error = raised(rc.plan_table, life, answer_type, None, " ", GOL_CONTEXT)
        check(
            f"9.5 {answer_type} without a question is a hint",
            isinstance(error, rc.RulesHint) and str(error) == "Type a question",
        )
    plan = rc.plan_table(life, "YES_NO", None, "Alive?", "")
    check(
        "9.5 Yes/No without Outcomes: Yes. and No.",
        plan.cells[0].question["criteria"] == {"true": "Yes.", "false": "No."},
    )
    plan = rc.plan_table(life, "AMOUNT", None, "How lively?", "")
    good = plan.cells[0].question["type"] == "score" and plan.cells[0].question["criteria"] == list(rc.AMOUNT_LEVELS)
    check("9.5 Amount without Outcomes: the five AMOUNT_LEVELS", good and plan.rows == 36)
    levels = answers(
        "snow.csv", "answer,when,if\nnone,bare ground,state=dead\nsome,a thin layer,\nlots,deep snow,\n", "amount"
    )
    plan = rc.plan_table(life, "AMOUNT", levels, "How much snow?", "")
    criteria = ["none: bare ground", "some: a thin layer", "lots: deep snow"]
    good = all(cell.question["criteria"] == criteria and cell.allowed == (0, 1, 2) for cell in plan.cells)
    warning = [("warning", 2, "'if' is not used for Amount answers")]
    check("9.5 Amount levels lowest first; 'if' ignored with a warning", good and summary(plan.warnings) == warning)


def test_limits():
    check("limits: 4 096 cells are fine", rc.cells_error((16, 16, 16)) == "")
    message = "Rules has 5 400 cells (limit 4 096): use fewer bands"
    check("limits: over 4 096 cells", rc.cells_error((15, 15, 24)) == message, rc.cells_error((15, 15, 24)))
    cube = [amount("a", 16), amount("b", 16), amount("c", 16)]
    error = raised(rc.plan_table, cube + [flag("f")], "YES_NO", None, "Snowy?", "")
    message = "Rules has 8 192 cells (limit 4 096): use fewer bands"
    check(
        "limits: planning over 4 096 cells is a RulesError", isinstance(error, rc.RulesError) and str(error) == message
    )
    costs = [rc.cell_cost(m) for m in (0, 1, 2, 16, 17, 20, 255)], [rc.cell_cost(m, False) for m in (1, 2, 16, 20, 255)]
    check(
        "limits: scored rows per cell, tournaments above 16", costs == ([0, 0, 2, 2, 6, 6, 34], [0, 1, 1, 3, 17]), costs
    )
    check("limits: 8 192 scored rows are fine", rc.rows_error(8192) == "")
    message = "Rules needs 8 500 scored rows (limit 8 192): offer at most 16 outcomes per cell, or use fewer bands"
    check("limits: over 8 192 scored rows", rc.rows_error(8500) == message, rc.rows_error(8500))
    plan = rc.plan_table(cube, "YES_NO", None, "Snowy?", "")
    check(
        "limits: the largest table, 4 096 cells of 2 options, is 8 192 rows",
        plan.rows == 8192 and len(plan.cells) == 4096,
    )
    error = raised(rc.plan_table, cube, "ONE_OF", inline(*(f"o{i}" for i in range(20))), "", "")
    message = "Rules needs 24 576 scored rows (limit 8 192): offer at most 16 outcomes per cell, or use fewer bands"
    check(
        "limits: planning over 8 192 rows is a RulesError", isinstance(error, rc.RulesError) and str(error) == message
    )
    five = [amount("a", 2), amount("b", 2), flag("f"), amount("c", 2), amount("d", 3)]
    plan = rc.plan_table(five, "AMOUNT", None, "How much?", "")
    warning = [("warning", 0, "interpolating over 3 Amount factors; 'd' uses its nearest band")]
    good = summary(plan.warnings) == warning and rc.interpolated_factors(five) == (0, 1, 3)
    check("limits: a 4th Amount factor uses its nearest band, with a warning", good, summary(plan.warnings))
    plan = rc.plan_table(five, "ONE_OF", inline("x", "y"), "", "")
    check("limits: One-of never interpolates, so it doesn't warn", not plan.warnings)


def test_interpolation():
    four = amount("x", 4)
    table = [(0.0, 1.0, 1.0), (0.2, 0.8, 1.0), (0.6, 0.6, 1.0), (1.0, 0.4, 1.0)]
    for x, b0, t, value, conf in (
        (0.5, 1, 0.5, 0.4, 0.7),
        (0.125, 0, 0.0, 0.0, 1.0),  # a band centre gives its row exactly
        (0.05, 0, 0.0, 0.0, 1.0),  # the outer half-bands are flat
        (0.8, 2, 0.7, 0.88, 0.46),
        (0.9, 2, 1.0, 1.0, 0.4),
        (1.5, 2, 1.0, 1.0, 0.4),  # clamped
    ):
        got = rc.axis(four, x), rc.blend([four], [x], table)
        good = got[0][0] == b0 and close(got[0][1], t) and close(got[1][0], value) and close(got[1][1], conf)
        check(f"interpolation: equal widths at x = {x}", good and got[1][2], got)

    biome = factors_csv("biome_factors.csv")
    factors = [flag("lit"), biome[0], biome[1]]  # custom edges (centres .15, .5, .85), then 3 equal bands
    got = rc.corners(factors, (True, 0.3, 0.75))
    want = [(10, 1 / 7), (11, 3 / 7), (13, 3 / 28), (14, 9 / 28)]  # t = 3/7 and 0.75 from b0 = 0 and 1
    good = [cell for cell, _ in got] == [cell for cell, _ in want] and all(close(a[1], b[1]) for a, b in zip(got, want))
    check("interpolation: hand-computed corners and weights", good, got)
    table = [(cell / 20, 1 - cell / 20, 1.0) for cell in range(18)]
    value, confidence, valid = rc.blend(factors, (True, 0.3, 0.75), table)
    check(
        "interpolation: the blend of those corners",
        close(value, 16.85 / 28) and close(confidence, 1 - 16.85 / 28) and valid,
    )

    d = amount("d", 3)
    factors = [amount("a", 3), flag("f"), amount("b", 4, (0, 0.1, 0.5, 0.8, 1)), amount("c", 2), d]
    centres = [rc.band_centres(factor) if factor.kind == "amount" else () for factor in factors]
    table = []
    for levels in rc.cell_rows(rc.level_counts(factors)):
        v = 0.2 * centres[0][levels[0]] + 0.3 * centres[2][levels[2]] + 0.5 * centres[3][levels[3]]
        table.append((v + 0.01 * levels[1] + 0.001 * levels[4], 0.5, 1.0))
    rng, worst = random.Random(3), 0.0
    for _ in range(500):
        xa, xb, xc, xd, f = (
            rng.uniform(1 / 6, 5 / 6),
            rng.uniform(0.05, 0.9),
            rng.uniform(0.25, 0.75),
            rng.random(),
            rng.random() < 0.5,
        )
        got = rc.blend(factors, (xa, f, xb, xc, xd), table)[0]
        worst = max(worst, abs(got - (0.2 * xa + 0.3 * xb + 0.5 * xc + 0.01 * f + 0.001 * rc.level_index(d, xd))))
    check(
        "interpolation: 8 corners reproduce a table linear in the band centres; a 4th Amount factor stays at its band",
        worst < 1e-9,
        worst,
    )
    got = rc.corners(factors, (0.3, True, 0.3, 0.3, 0.3))
    check("interpolation: 8 corners whose weights sum to 1", len(got) == 8 and close(sum(w for _, w in got), 1.0), got)
    equal, custom = amount("e", 5), amount("k", 5, [b / 5 for b in range(6)])
    table = [(v, 1.0, 1.0) for v in (0.1, 0.7, 0.2, 0.9, 0.4)]
    same = all(
        close(rc.blend([equal], [x], table)[0], rc.blend([custom], [x], table)[0])
        for x in (i / 997 for i in range(998))
    )
    check("interpolation: custom edges of equal width blend like equal widths", same)


def test_renormalisation():
    biome = factors_csv("biome_factors.csv")
    factors = [flag("lit"), biome[0], biome[1]]
    table = [(cell / 20, 1 - cell / 20, 1.0) for cell in range(18)]
    table[14] = (0.0, 0.0, 0.0)  # the corner of weight 9/28 is pending
    value, confidence, valid = rc.blend(factors, (True, 0.3, 0.75), table)
    good = close(value, 10.55 / 19) and close(confidence, 1 - 10.55 / 19) and valid
    check("renormalisation: a pending corner drops out instead of pulling towards 0", good, value)
    pending = [(0.0, 0.0, 0.0)] * 18
    check(
        "renormalisation: no answered corner gives the fallbacks",
        rc.blend(factors, (True, 0.3, 0.75), pending) == (0.0, 0.0, False),
    )
    table = [(cell / 20, 0.5, 0.0 if cell in (14, 16, 17) else 1.0) for cell in range(18)]
    value, _, valid = rc.blend(factors, (True, 0.5, 0.5), table)
    check(
        "renormalisation: pending corners of weight 0 change nothing at a band centre",
        close(value, 13 / 20) and valid,
        value,
    )
    four = amount("x", 4)
    value = rc.blend([four], [(1.5 + 1e-7) / 4], [(0.0, 0.0, 1.0), (0.2, 0.5, 0.0), (0.6, 0.5, 1.0), (1.0, 0.5, 1.0)])
    check("renormalisation: answered corners weighing under 1e-6 give the fallbacks", value == (0.0, 0.0, False), value)
    value = rc.blend([category("c", (0, 1)), amount("a", 2)], [5, 0.5], [(1.0, 1.0, 1.0)] * 4)
    check("renormalisation: an unknown category value gives the fallbacks", value == (0.0, 0.0, False), value)
    got = [rc.ok_value(source) for source in ("model", "lock", "code", "pending", "error")]
    check(
        "renormalisation: ok is 1 for answered, locked and code rows, 0 for pending and error rows",
        got == [1, 1, 1, 0, 0],
    )


def test_dither():
    rng = random.Random(20260926)
    four = amount("x", 4)
    got = rc.dither_distribution(four, 0.3, 0.6)
    check(
        "dither: 0.05 inside an edge, half-width 0.075: 1/6 cross",
        all(close(a, b) for a, b in zip(got, (1 / 6, 5 / 6, 0, 0))),
        got,
    )
    counts = [0] * 4
    for _ in range(SAMPLES):
        counts[rc.dithered_level(four, 0.3, 0.6, rng.random())] += 1
    worst = max(abs(count / SAMPLES - p) for count, p in zip(counts, got))
    check("dither: 100 000 samples of one factor within 1 %", worst <= 0.01, worst)
    five, dither = amount("y", 5), 0.8  # band width 0.2, jitter half-width 0.08
    distances = (0.0001, 0.02, 0.04, 0.06, 0.08, 0.1)
    ramp = [rc.dither_distribution(five, 0.4 - d, dither)[2] for d in distances]
    want = [max(0.0, 0.5 - d / (dither * 0.2)) for d in distances]
    check(
        "dither: the next band's chance falls linearly from 1/2 at the edge to 0",
        all(close(a, b) for a, b in zip(ramp, want)),
        ramp,
    )
    kept = rc.dither_distribution(four, 0.3, 0.0) == (0.0, 1.0, 0.0, 0.0) and {
        rc.dithered_level(four, 0.3, 0.0, r) for r in (0, 0.5, 0.99)
    } == {1}
    check("dither: Dither 0 keeps the nearest band", kept)
    clamped = close(rc.dither_distribution(four, 0.02, 1.0)[0], 1.0) and close(
        rc.dither_distribution(four, 0.99, 1.0)[3], 1.0
    )
    check("dither: jitter beyond 0 or 1 is clamped into the outer bands", clamped)

    narrow = amount("n", 3, (0, 0.1, 0.12, 1))
    got = rc.dither_distribution(narrow, 0.5, 1.0)
    check(
        "dither: a wide band's jitter can jump a narrow one",
        all(close(a, b) for a, b in zip(got, (0.04 / 0.88, 0.02 / 0.88, 0.82 / 0.88))),
        got,
    )
    counts = [0] * 3
    for _ in range(SAMPLES):
        counts[rc.dithered_level(narrow, 0.5, 1.0, rng.random())] += 1
    worst = max(abs(count / SAMPLES - p) for count, p in zip(counts, got))
    check("dither: 100 000 samples across a narrow band within 1 %", worst <= 0.01, worst)

    factors = [amount("t", 3), flag("f"), amount("m", 3, (0, 0.2, 0.5, 1)), category("c", (0, 7))]
    values = (0.36, True, 0.45, 7)
    analytic = rc.dither_cells(factors, values, 1.0)
    want = {9: 0.42 * 2 / 3, 11: 0.42 / 3, 21: 0.58 * 2 / 3, 23: 0.58 / 3}  # strides (12, 6, 2, 1)
    good = sorted(analytic) == sorted(want) and all(close(analytic[cell], p) for cell, p in want.items())
    check("dither: the analytic cell distribution", good, analytic)
    counts = {}
    for _ in range(SAMPLES):
        cell = rc.dithered_cell(factors, values, 1.0, [rng.random() for _ in factors])
        counts[cell] = counts.get(cell, 0) + 1
    worst = max(abs(counts.get(cell, 0) / SAMPLES - analytic.get(cell, 0.0)) for cell in set(counts) | set(analytic))
    check("dither: 100 000 seeded samples of the cell within 1 % of its distribution", worst <= 0.01, worst)
    shape = rc.level_counts(factors)
    exact = all(rc.cell_levels(cell, shape)[1] == 1 and rc.cell_levels(cell, shape)[3] == 1 for cell in counts)
    check("dither: categories and flags never dither", exact)
    invalid = rc.dither_cells(factors, (0.36, True, 0.45, 3), 1.0), rc.dithered_cell(
        factors, (0.36, True, 0.45, 3), 1.0, [0.5] * 4
    )
    check("dither: an unknown category value gives no cell", invalid == ({}, -1), invalid)


LEVELS = range(2, 17)
UNSURE_BELOW = 0.30  # the preference's default


def near_uniform(levels, weight=0.1):
    """A uniform answer with `weight` of it moved onto the middle level or option."""
    distribution = [(1 - weight) / levels] * levels
    distribution[levels // 2] += weight
    return distribution


def phase_endpoints():
    for kind in ('AMOUNT', 'ONE_OF'):
        uniform = {levels: confidence(kind, [1 / levels] * levels) for levels in LEVELS}
        check(f"{kind}: a uniform answer gives exactly 0, for L = 2..16", set(uniform.values()) == {0.0}, uniform)
        certain = {(levels, at): confidence(kind, [float(i == at) for i in range(levels)])
                   for levels in LEVELS for at in (0, levels // 2, levels - 1)}
        check(f"{kind}: a certain answer gives exactly 1, on any level", set(certain.values()) == {1.0},
              {key: value for key, value in certain.items() if value != 1.0})
        near = {levels: confidence(kind, near_uniform(levels)) for levels in LEVELS}
        check(f"{kind}: near-uniform answers stay below {UNSURE_BELOW}", 0 < max(near.values()) < UNSURE_BELOW, near)
    got = [confidence('YES_NO', (p, 1 - p)) for p in (0.5, 0.0, 1.0, 0.45)]
    check("YES_NO: p = 0.5 gives 0, p = 0 or 1 gives 1, and near 0.5 stays below the threshold",
          got[:3] == [0.0, 1.0, 1.0] and 0 < got[3] < UNSURE_BELOW, got)


def phase_examples():
    between = [0, 0.45, 0.55, 0, 0]
    got = confidence('AMOUNT', between), confidence('ONE_OF', between)
    check("AMOUNT: an answer between two neighbouring levels scores 0.65 with sigma, 0.10 as top minus runner-up",
          (round(got[0], 2), round(got[1], 2)) == (0.65, 0.10), got)
    got = confidence('AMOUNT', [0.5, 0, 0, 0, 0.5])
    check("AMOUNT: a split between the two ends scores 0 (its sigma 2 exceeds the uniform 1.41)", got == 0.0, got)
    got = confidence('YES_NO', (0.93, 0.07)), confidence('ONE_OF', [0.1, 0.7, 0.2])
    check("YES_NO |2p - 1| and ONE_OF top minus runner-up", got == (0.86, 0.5), got)
    got = confidence('ONE_OF', [2, 1, 1]), confidence('AMOUNT', []), confidence('ONE_OF', [1.0])
    check("unnormalised answers are normalised; no options give 0 and one option 1", got == (0.25, 0.0, 1.0), got)


test_cells()
test_allowed()
test_code_decided()
test_error_rows()
test_templates()
test_limits()
test_interpolation()
test_renormalisation()
test_dither()
phase_endpoints()
phase_examples()
shutil.rmtree(TMP, ignore_errors=True)
finish()
