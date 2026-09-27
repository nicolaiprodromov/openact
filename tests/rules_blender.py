"""The Rules node (§6.5, §9) in a headless Blender. Without arguments no model is used: synthetic answers go
through the real pass (the Game of Life, in a Simulation Zone, against a Python B3/S23 reference) or straight into
the store (the cell index for every combination, interpolation, renormalisation and dithering against rules_core's
references), and a rebuild with the CSV files deleted must leave every output unchanged. A Factors CSV edit that
keeps every row rebuilds the graph for the new factors, and a payload header of any length builds. Outcomes edits
keep the table (§5.4), the status shows the plan's problems (§17.1), contexts must agree on the factors (§6.0), and
only the pass writes the interface: not a draw, and again after an undo.

With -- --model PORT GPU, the real model answers the Game of Life rule sets (18/18 exact, 0 unsure each).

Run:  blender -b --factory-startup --python tests/rules_blender.py [-- --model 18769 0]
"""

import dataclasses
import itertools
import os
import random
import shutil
import sys
import tempfile
import time

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import ROOT, check, drive, enable_addon, finish, store  # noqa: E402

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
MODEL = ARGS[ARGS.index("--model") + 1:ARGS.index("--model") + 3] if "--model" in ARGS else None
if MODEL:
    engine, server, P = enable_addon()
else:
    from openact import engine, nodes  # noqa: E402
    from openact import prefs as P  # noqa: E402

    P._Defaults.auto_start = False  # nothing here may start a model server
    nodes.register()
from openact import csvfmt, evaluator, inner, pipeline, tables  # noqa: E402
from openact import rules_core as rc  # noqa: E402
from openact import store as S  # noqa: E402
from openact.specs import rules as R  # noqa: E402

FIXTURES = os.path.join(ROOT, "tests", "fixtures")
TMP = tempfile.mkdtemp(prefix="openact_rules_")
SPEC = engine.SPECS[R.RulesSpec.idname]
GOL_QUESTION = "What happens to this cell in the next generation?"
GOL_CONTEXT = "A cellular automaton on a square grid; every cell has eight neighbours."
RULE_SETS = {  # Outcomes CSV -> (birth, survival) neighbour counts
    "life_outcomes.csv": ({3}, {2, 3}), "highlife_outcomes.csv": ({3, 6}, {2, 3}),
    "seeds_outcomes.csv": ({2}, set()), "daynight_outcomes.csv": ({3, 6, 7, 8}, {3, 4, 6, 7, 8}),
}
TYPES = {'VALUE': 'FLOAT', 'INT': 'INT', 'BOOLEAN': 'BOOLEAN'}


def write(name, text):
    path = os.path.join(TMP, name)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    return path


def enabled(sockets, name):
    return next(s for s in sockets if s.name == name and s.enabled)


def scene(name, factors, outcomes, values, answer_type='ONE_OF', question="", context=""):
    """A tree with a Rules node, used by an object whose points carry the factor values as in_<factor>."""
    tree = bpy.data.node_groups.new(name, 'GeometryNodeTree')
    tree.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    tree.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    gi, go = tree.nodes.new("NodeGroupInput"), tree.nodes.new("NodeGroupOutput")
    rules = tree.nodes.new(R.RulesSpec.idname)
    rules.answer_type = answer_type
    for word, text in (("Question", question), ("Context", context), ("Factors", factors), ("Outcomes", outcomes)):
        rules.inputs[word].default_value = text
    count = len(next(iter(values.values()))[1]) if values else 1
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([(i * 0.01, 0.0, 0.0) for i in range(count)], [], [])
    for attribute, (kind, data) in values.items():
        mesh.attributes.new(attribute, kind, 'POINT').data.foreach_set("value", data)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.modifiers.new("Rules", 'NODES').node_group = tree
    return tree, rules, obj, gi, go


def wire_inputs(tree, rules):
    """Factor sockets <- Named Attribute in_<factor>."""
    for socket in rules.inputs:
        if socket.name.casefold() in csvfmt.RULES_INPUTS or socket.type not in TYPES or socket.is_linked:
            continue
        attribute = tree.nodes.new("GeometryNodeInputNamedAttribute")
        attribute.data_type = TYPES[socket.type]
        attribute.inputs["Name"].default_value = "in_" + socket.name
        tree.links.new(enabled(attribute.outputs, "Attribute"), socket)


def wire(tree, rules, gi, go, outputs, geometry=None):
    """Factor sockets <- Named Attribute in_<factor>; each output -> Store Named Attribute out_<output>."""
    wire_inputs(tree, rules)
    geometry = geometry or gi.outputs["Geometry"]
    for name in outputs:
        geometry = store(tree, geometry, rules.outputs[name], "out_" + name, TYPES[rules.outputs[name].type])
    tree.links.new(geometry, go.inputs[0])


def read(obj, names):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        found = {name: mesh.attributes.get("out_" + name) for name in names}
        return {name: [d.value for d in a.data] if a else None for name, a in found.items()}
    finally:
        evaluated.to_mesh_clear()


def context(rules, obj):
    return next(ctx for ctx in evaluator.find_uses()[rules.id_data.as_pointer()] if ctx.obj == obj)


def synthetic(rules, obj, values):
    """Plan in obj's context, give the node the sockets the Job declares (as the pass does), store the table
    values(i, row) gives, rebuild."""
    ctx = context(rules, obj)
    job = SPEC.plan(rules, ctx)
    SPEC.agree(rules, [(ctx, job)])
    rows = []
    for i, row in enumerate(job.rows):
        given = values(i, row)
        rows.append(dataclasses.replace(row, source=given.pop("source", "model"), values=given, ask="fake"))
    entry = SPEC.entry(job, rows, "fake")
    S.save(rules, S.NodeStore([entry]))
    inner.build(rules, S.load(rules))
    return job, entry


def factors_of(path):
    return csvfmt.parse_factors(csvfmt.read_csv(path, schema="factors"))


# --------------------------------------------------------------------------- cell index (§9.2, §9.3)

MIXED_FACTORS = """factor,kind,level,value,from,to
lit,flag,dark,0,,
lit,flag,bright,1,,
soil,category,clay,9,,
soil,category,sand,2,,
soil,category,loam,5,,
age,category,young,3,,
age,category,mature,4,,
age,category,old,5,,
wet,amount,dry,,,
wet,amount,damp,,,
wet,amount,soaked,,,
heat,amount,cold,,0,0.2
heat,amount,mild,,0.2,0.7
heat,amount,hot,,0.7,1
"""
TWO = "answer,when\nA,the first\nB,the second\n"


def representatives(factor):
    """Values that hit every level of a factor, and for categories two that match none."""
    if factor.kind == "flag":
        return [False, True]
    if factor.kind == "category":
        return [level.value for level in factor.levels] + [-1, 7]
    centres = list(rc.band_centres(factor))
    return centres + [-0.5, 1.5]


def phase_cell_index():
    path = write("mixed_factors.csv", MIXED_FACTORS)
    factors = factors_of(path)
    combos = list(itertools.product(*(representatives(f) for f in factors)))
    columns = {f"in_{f.name}": ({'flag': 'BOOLEAN', 'category': 'INT', 'amount': 'FLOAT'}[f.kind],
                                [combo[j] for combo in combos]) for j, f in enumerate(factors)}
    tree, rules, obj, gi, go = scene("CellIndex", path, write("two.csv", TWO), columns, question="Which?")
    job, entry = synthetic(rules, obj, lambda i, row: {"outcome": i, "c": 0.5})
    wire(tree, rules, gi, go, ["Outcome", "Confidence"])
    got = read(obj, ["Outcome", "Confidence"])
    expected = [rc.cell_of(factors, combo) for combo in combos]
    bad = [(combo, g, e) for combo, g, e in zip(combos, got["Outcome"], expected) if g != e]
    check(f"cell index: every combination of {len(job.rows)} cells, invalid category values and clamped amounts "
          f"({len(combos)} points) equal rules_core.cell_of", not bad and len(set(expected)) == 163, bad[:3])
    check("cell index: the non-contiguous map sits in the table after every cell row",
          rc.category_map(factors[1]) is not None and entry.shape["rows"] == 162)
    confidences = {round(c, 6) for c, e in zip(got["Confidence"], expected) if e >= 0}
    check("cell index: Confidence reads the cell's c, 0 where the cell is invalid",
          confidences == {0.5} and all(c == 0.0 for c, e in zip(got["Confidence"], expected) if e < 0))

    before = read(obj, ["Outcome", "Confidence"])
    os.remove(path)
    os.remove(os.path.join(TMP, "two.csv"))
    csvfmt.clear_cache()
    inner.build(rules, S.load(rules))
    engine.run_pass()  # plans with the files missing: an error, and the last good table stays
    check("rebuild with every CSV file deleted leaves the outputs unchanged (shape suffices)",
          read(obj, ["Outcome", "Confidence"]) == before and engine.STATUS[rules.uid][0] == 'ERROR',
          engine.STATUS.get(rules.uid))


# --------------------------------------------------------------------------- Game of Life (§6.5)

W = H = 8


def neighbours_of(i):
    row, col = divmod(i, W)
    return [((row + dr) % H) * W + (col + dc) % W for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]


def step(alive, birth=(3,), survive=(2, 3)):
    counts = [sum(alive[k] for k in neighbours_of(i)) for i in range(W * H)]
    return [int(n in survive) if a else int(n in birth) for a, n in zip(alive, counts)]


def outcome_name(state, n, birth, survive):
    if state:
        return "Survives" if n in survive else "Dies"
    return "Birth" if n in birth else "Stays empty"


def life_scene():
    start = [0] * (W * H)
    for row, col in ((0, 1), (1, 2), (2, 0), (2, 1), (2, 2), (5, 5), (5, 6), (5, 7)):  # a glider and a blinker
        start[row * W + col] = 1
    columns = {"alive": ('INT', start)}
    for k in range(8):
        columns[f"n{k}"] = ('INT', [neighbours_of(i)[k] for i in range(W * H)])
    tree, rules, obj, gi, go = scene("Life", os.path.join(FIXTURES, "life_factors.csv"),
                                     os.path.join(FIXTURES, "life_outcomes.csv"), columns,
                                     question=GOL_QUESTION, context=GOL_CONTEXT)
    return tree, rules, obj, gi, go, start


def life_zone(tree, rules, gi, go):
    """state = alive > 0 and neighbours = 8 toroidal Sample Index lookups, inside a Simulation Zone that stores
    the Rules payload output 'alive' back."""
    nodes, links = tree.nodes, tree.links
    sim_in, sim_out = nodes.new("GeometryNodeSimulationInput"), nodes.new("GeometryNodeSimulationOutput")
    sim_in.pair_with_output(sim_out)
    links.new(gi.outputs["Geometry"], sim_in.inputs["Geometry"])
    geometry = sim_in.outputs["Geometry"]  # outputs[0] is Delta Time
    alive = nodes.new("GeometryNodeInputNamedAttribute")
    alive.data_type = 'INT'
    alive.inputs["Name"].default_value = "alive"
    state = nodes.new("FunctionNodeCompare")
    state.data_type, state.operation = 'INT', 'GREATER_THAN'
    links.new(enabled(alive.outputs, "Attribute"), next(s for s in state.inputs if s.identifier == "A_INT"))
    links.new(state.outputs[0], rules.inputs["state"])
    total = None
    for k in range(8):
        index = nodes.new("GeometryNodeInputNamedAttribute")
        index.data_type = 'INT'
        index.inputs["Name"].default_value = f"n{k}"
        sampled = nodes.new("GeometryNodeSampleIndex")
        sampled.data_type = 'INT'
        links.new(geometry, sampled.inputs["Geometry"])
        links.new(enabled(alive.outputs, "Attribute"), enabled(sampled.inputs, "Value"))
        links.new(enabled(index.outputs, "Attribute"), sampled.inputs["Index"])
        value = enabled(sampled.outputs, "Value")
        if total is None:
            total = value
        else:
            add = nodes.new("FunctionNodeIntegerMath")
            add.operation = 'ADD'
            links.new(total, add.inputs[0])
            links.new(value, add.inputs[1])
            total = add.outputs[0]
    links.new(total, rules.inputs["neighbours"])
    stored = store(tree, geometry, rules.outputs["alive"], "alive", 'INT')
    links.new(stored, sim_out.inputs["Geometry"])  # inputs[0] is Skip
    links.new(sim_out.outputs["Geometry"], go.inputs[0])


def gol_answers(job, birth, survive, names):
    """The right answer to every missing cell, as pipeline.parse() gives it."""
    for row in job.rows:
        if row.cell is None:
            continue
        state, n = row.name.split("|")
        right = outcome_name(state == "alive", int(n), birth, survive)
        keys = list(row.cell.question["criteria"])
        pipeline.remember(row.cell.key, {"probabilities": [0.96 if names[int(key)] == right else 0.04 for key in keys],
                                         "model": "fake"})


def life_read(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        return [d.value for d in mesh.attributes["alive"].data]
    finally:
        evaluated.to_mesh_clear()


def phase_life():
    tree, rules, obj, gi, go, start = life_scene()
    jobs = []
    submit, engine._submit = engine._submit, lambda job, missing: jobs.append((job, missing))
    try:
        engine.run_pass()
        check("life: the pass asks 18 cells, each offering 2 outcomes (36 scored rows)",
              len(jobs) == 1 and len(jobs[0][1]) == 18 and sum(c.cost for c in jobs[0][1]) == 36,
              [len(m) for _, m in jobs])
        names = [option.name for option in csvfmt.parse_answer_set(
            csvfmt.read_csv(os.path.join(FIXTURES, "life_outcomes.csv")), "one_of", "rules").options]
        gol_answers(jobs[0][0], {3}, {2, 3}, names)
        engine.run_pass()
    finally:
        engine._submit = submit
    entries = S.load(rules).entries
    table = [names[o] for o in entries[0].rows.v["outcome"]] if entries else []
    reference = [outcome_name(s, n, {3}, {2, 3}) for s in (0, 1) for n in range(9)]
    check("life: the stored table is B3/S23, 18/18", table == reference, table)
    check("life: payload 'alive' is baked per cell", entries and entries[0].rows.payload.get("alive") ==
          [int(t in ("Birth", "Survives")) for t in reference], entries and entries[0].rows.payload)
    check("life: the node shows its sockets and '18 cells · 0 unsure'",
          all(name in rules.inputs for name in ("state", "neighbours")) and "alive" in rules.outputs
          and engine.STATUS.get(rules.uid) == ('CHECKMARK', "18 cells · 0 unsure"), engine.STATUS.get(rules.uid))
    life_zone(tree, rules, gi, go)
    scene_ = bpy.context.scene
    scene_.frame_start, scene_.frame_end = 1, 40
    scene_.frame_set(1)
    first = life_read(obj)
    offset = 1 if first == step(start) else 0
    expected, frames, bad = start, 16, []
    for _ in range(offset):
        expected = step(expected)
    if first != expected:
        bad.append(1)
    for frame in range(2, frames + 1):
        scene_.frame_set(frame)
        expected = step(expected)
        if life_read(obj) != expected:
            bad.append(frame)
    check(f"life: {frames} frames in a Simulation Zone equal the Python B3/S23 reference (toroidal)", not bad,
          f"bad frames {bad}; the zone's first frame steps {offset}")
    scene_.frame_set(1)

    # Seeds: every live cell has one outcome ("Dies"), decided by code and never asked
    path = os.path.join(FIXTURES, "seeds_outcomes.csv")
    rules.inputs["Outcomes"].default_value = path
    jobs.clear()
    submit, engine._submit = engine._submit, lambda job, missing: jobs.append((job, missing))
    try:
        engine.run_pass()
        names = [o.name for o in csvfmt.parse_answer_set(csvfmt.read_csv(path), "one_of", "rules").options]
        asked = len(jobs[0][1]) if jobs else 0
        if jobs:
            gol_answers(jobs[0][0], {2}, set(), names)
        engine.run_pass()
    finally:
        engine._submit = submit
    entry = next((e for e in S.load(rules).entries if e.gn[3] == path), None)
    table = [names[o] for o in entry.rows.v["outcome"]] if entry else []
    check("life: Seeds asks only the 9 dead cells; its 9 code rows are stored with the answers (B2/S)",
          asked == 9 and table == [outcome_name(s, n, {2}, set()) for s in (0, 1) for n in range(9)]
          and entry.rows.src == "m" * 9 + "c" * 9, (asked, table, engine.STATUS.get(rules.uid)))
    return tree, rules, obj


# --------------------------------------------------------------------------- interpolation (§9.6)

MIXED_YES_NO = """factor,kind,level,value,from,to
warmth,amount,cold,,,
warmth,amount,cool,,,
warmth,amount,warm,,,
warmth,amount,hot,,,
lit,flag,dark,0,,
lit,flag,bright,1,,
soil,category,clay,9,,
soil,category,sand,2,,
soil,category,loam,5,,
slope,amount,flat,,0,0.25
slope,amount,steep,,0.25,0.5
slope,amount,cliff,,0.5,1
"""


def random_values(factors, count, rng):
    out = []
    for _ in range(count):
        combo = []
        for f in factors:
            if f.kind == "amount":
                combo.append(rng.uniform(-0.1, 1.1))
            elif f.kind == "flag":
                combo.append(rng.random() < 0.5)
            else:
                combo.append(rng.choice([level.value for level in f.levels] + [7]))
        out.append(combo)
    return out


def phase_interpolation():
    rng = random.Random(7)
    cases = (("Snow", os.path.join(FIXTURES, "biome_factors.csv"), 'AMOUNT', "How much snow covers the ground?"),
             ("Shade", write("shade_factors.csv", MIXED_YES_NO), 'YES_NO', "Is this spot shaded?"))
    for name, path, answer_type, question in cases:
        factors = factors_of(path)
        combos = random_values(factors, 400, rng)
        columns = {f"in_{f.name}": ({'flag': 'BOOLEAN', 'category': 'INT', 'amount': 'FLOAT'}[f.kind],
                                    [combo[j] for combo in combos]) for j, f in enumerate(factors)}
        tree, rules, obj, gi, go = scene(name, path, "", columns, answer_type=answer_type, question=question)
        value = rc.value_column(answer_type)
        cells = rc.level_counts(factors)
        count = 1
        for n in cells:
            count *= n
        failed = set(rng.sample(range(count), count // 6))  # error rows: ok = 0, renormalised away
        table = [(0.0, 0.0, 0.0) if i in failed else (rng.random(), rng.random(), 1.0) for i in range(count)]

        def values(i, row, table=table, failed=failed, value=value):
            v, c, ok = table[i]
            return {value: v, "c": c, "ok": ok, **({"source": "error"} if i in failed else {})}

        started = time.perf_counter()
        synthetic(rules, obj, values)
        built = (time.perf_counter() - started) * 1000
        outputs = ["Amount", "Confidence"] if answer_type == 'AMOUNT' else ["Probability", "Yes", "Confidence"]
        wire(tree, rules, gi, go, outputs)
        got = read(obj, outputs)
        worst, wrong = 0.0, []
        for i, combo in enumerate(combos):
            v, c, ok = rc.blend(factors, combo, table)
            worst = max(worst, abs(got[outputs[0]][i] - v), abs(got["Confidence"][i] - c))
            if answer_type == 'YES_NO' and got["Yes"][i] != (v >= 0.5):
                wrong.append(i)
        corners = 2 ** len(rc.interpolated_factors(factors))
        check(f"interpolation {answer_type}: {len(combos)} points over {corners} corners (renormalised, "
              f"{len(failed)} error rows) within 1e-5 of rules_core.blend", worst <= 1e-5 and not wrong,
              f"worst {worst:.2e}, Yes wrong at {wrong[:5]}")
        nodes_ = len(rules.node_tree.nodes)
        print(f"BENCH rules build {answer_type} ({nodes_} nodes, plan + store + build): {built:.0f} ms "
              f"(budget 250 ms for the interpolated demo)", flush=True)
        if built > 250:
            print(f"WARN rules build {answer_type} took {built:.0f} ms", flush=True)
            check(f"rules build {answer_type} within twice its budget", built <= 500, f"{built:.0f} ms")


# --------------------------------------------------------------------------- dithering (§9.7)

DITHER_FACTORS = """factor,kind,level,value,from,to
x,amount,a,,,
x,amount,b,,,
x,amount,c,,,
x,amount,d,,,
lit,flag,dark,0,,
lit,flag,bright,1,,
y,amount,low,,0,0.2
y,amount,mid,,0.2,0.7
y,amount,high,,0.7,1
"""


def phase_dither():
    path = write("dither_factors.csv", DITHER_FACTORS)
    factors = factors_of(path)
    count = 20000
    point = (0.3, True, 0.65)
    columns = {"in_x": ('FLOAT', [point[0]] * count), "in_lit": ('BOOLEAN', [point[1]] * count),
               "in_y": ('FLOAT', [point[2]] * count)}
    tree, rules, obj, gi, go = scene("Dither", path, write("dither_two.csv", TWO), columns, question="Which?")
    synthetic(rules, obj, lambda i, row: {"outcome": i, "c": 1.0})
    wire(tree, rules, gi, go, ["Outcome"])
    exact = rc.cell_of(factors, point)
    check("dither 0: every element takes its nearest cell", set(read(obj, ["Outcome"])["Outcome"]) == {exact})
    for dither, seed in ((0.6, 0), (1.0, 5)):
        rules.inputs["Dither"].default_value = dither
        rules.inputs["Seed"].default_value = seed
        got = read(obj, ["Outcome"])["Outcome"]
        expected = rc.dither_cells(factors, point, dither)
        share = {cell: got.count(cell) / count for cell in set(got) | set(expected)}
        worst = max(abs(share.get(cell, 0.0) - expected.get(cell, 0.0)) for cell in share)
        check(f"dither {dither} (seed {seed}): each cell's fraction within 2% of rules_core.dither_cells "
              f"({len(expected)} cells)", worst <= 0.02, f"worst {worst:.4f}; {share} vs {expected}")
    rules.inputs["Dither"].default_value = 0.0


# --------------------------------------------------------------------------- layout edits and payload names

EDGES = "factor,kind,level,when,from,to\nheat,amount,cold,Cold weather.,0,{0}\nheat,amount,hot,Hot weather.,{0},1\n"
LEVELS = "factor,kind,level,when\nheat,amount,cold,Cold.\n{mild}heat,amount,hot,Hot.\n"


def planned(rules):
    """Run a pass and return the Jobs it asks for this node; nothing reaches a server."""
    jobs = []
    submit = engine._submit
    engine._submit = lambda job, missing: jobs.append(job) if job.uid == rules.uid else None
    try:
        engine.run_pass()
    finally:
        engine._submit = submit
    return jobs


def answer_cells(jobs, outcome):
    """Remember an answer for every cell the jobs ask: outcome(row name) is the row of the outcome that wins."""
    for job in jobs:
        for row in job.rows:
            if row.cell is not None:
                pipeline.remember(row.cell.key, {"model": "fake", "probabilities": [
                    0.9 if int(key) == outcome(row.name) else 0.1 for key in row.cell.question["criteria"]]})


def phase_layout_edit():
    """A Factors CSV edit that keeps every row (band edges moved, a level removed) replaces the entry in place with 0
    requests. The graph bakes the factors, so the pass rebuilds it, and the outputs follow the new layout (§8.8)."""
    cases = (("Edges", EDGES.format("0.5"), EDGES.format("0.2"), {"cold": 0, "hot": 1}, [0.1, 0.3, 0.6, 0.9],
              [0, 0, 1, 1], [0, 1, 1, 1]),
             ("Levels", LEVELS.format(mild="heat,amount,mild,Mild.\n"), LEVELS.format(mild=""),
              {"cold": 0, "mild": 1, "hot": 0}, [0.1, 0.5, 0.9], [0, 1, 0], [0, 0, 0]))
    for name, factors, edited, table, heat, first, then in cases:
        path = write(f"{name}_factors.csv", factors)
        tree, rules, obj, gi, go = scene(name, path, write(f"{name}_outcomes.csv", TWO), {"in_heat": ('FLOAT', heat)},
                                         question="Which one?")
        answer_cells(planned(rules), table.__getitem__)  # the first pass plans, and the heat socket appears
        wire(tree, rules, gi, go, ["Outcome"])
        planned(rules)
        before = read(obj, ["Outcome"])["Outcome"]
        write(f"{name}_factors.csv", edited)
        csvfmt.clear_cache()
        asked = planned(rules)
        got = read(obj, ["Outcome"])["Outcome"]
        check(f"layout edit ({name.lower()}): 0 requests, and the rebuilt graph follows the new factors",
              before == first and not asked and got == then, (before, len(asked), got))


def phase_long_header():
    """A payload header of 62 bytes: as oa_pl_<header> its table column would need 68 bytes, which Blender 4.5 refuses,
    so it is named by position (§8.5). The node builds, and its payload output reads the baked value."""
    header = "h" * 62
    factors = write("long_factors.csv", "factor,kind,level,when\nheat,amount,cold,Cold.\nheat,amount,hot,Hot.\n")
    outcomes = write("long_outcomes.csv", f"answer,when,{header}\nA,the first,1\nB,the second,2\n")
    tree, rules, obj, gi, go = scene("LongHeader", factors, outcomes, {"in_heat": ('FLOAT', [0.1, 0.9])},
                                     question="Which payload?")
    answer_cells(planned(rules), lambda _name: 1)  # B, whose payload is 2, everywhere
    wire(tree, rules, gi, go, [header])
    planned(rules)
    table = tables.find(rules)
    names = sorted(a.name for a in table.data.attributes if a.name.startswith(tables.ATTR)) if table else []
    status = engine.STATUS.get(rules.uid)
    check("payload: a 62-byte header builds, its table column is named by position (oa_pl_0), and its output reads "
          "it", status is not None and status[0] == 'CHECKMARK' and "oa_pl_0" in names
          and read(obj, [header])[header] == [2, 2], (status, names))


# --------------------------------------------------------------------------- Outcomes edits, status, agreement, writes

CELLS = [(state, n) for state in (False, True) for n in range(9)]  # one element per Game of Life cell, in cell order


def passed(rules):
    """Run a pass: (Job, missing cells) of each context of this node that asks the model; nothing reaches a server."""
    asked = []
    submit = engine._submit
    engine._submit = lambda job, missing: asked.append((job, list(missing))) if job.uid == rules.uid else None
    try:
        engine.run_pass()
    finally:
        engine._submit = submit
    return asked


def factor_names(rules):
    return [name for name, _kind in R._factor_sockets(rules.node_tree)]


def phase_outcome_edits():
    """Moving, inserting or deleting an Outcomes row keeps the Seeds table (B2/S) through the pass: a carried row's
    outcome is renumbered by name (§5.4), so no output names another outcome, and only the cells whose offered
    outcomes changed are asked."""
    with open(os.path.join(FIXTURES, "seeds_outcomes.csv"), encoding="utf-8") as handle:
        header, birth, empty, dies = handle.read().splitlines()
    explodes = "Explodes,A live cell with eight live neighbours explodes.,state=alive;neighbours=8,0"
    path = write("seeds_edit.csv", "\n".join((header, birth, empty, dies)) + "\n")
    columns = {"in_state": ('BOOLEAN', [s for s, _n in CELLS]), "in_neighbours": ('INT', [n for _s, n in CELLS])}
    tree, rules, obj, gi, go = scene("SeedsEdit", os.path.join(FIXTURES, "life_factors.csv"), path, columns,
                                     question=GOL_QUESTION, context="Seeds on a square grid.")  # not phase_life's cells
    reference = [outcome_name(s, n, {2}, set()) for s, n in CELLS]
    alive = [int(name == "Birth") for name in reference]

    def settle(lines, first=False):
        """Write the Outcomes rows, run the pass, answer what it asks (B2/S), run it again: (cells asked, each
        element's outcome by name, its alive output, the status)."""
        write("seeds_edit.csv", "\n".join((header, *lines)) + "\n")
        csvfmt.clear_cache()
        names = [line.split(",")[0] for line in lines]
        asked = passed(rules)
        for job, _missing in asked:
            gol_answers(job, {2}, set(), names)
        passed(rules)
        if first:
            wire(tree, rules, gi, go, ["Outcome", "alive"])
            passed(rules)
        got = read(obj, ["Outcome", "alive"])
        outcomes = [names[o] if 0 <= o < len(names) else o for o in got["Outcome"]]
        return sum(len(missing) for _job, missing in asked), outcomes, got["alive"], engine.STATUS.get(rules.uid)

    got = settle((birth, empty, dies), first=True)
    check("outcome edits: Seeds answered through the pass (9 cells asked)",
          got == (9, reference, alive, ('CHECKMARK', "18 cells · 0 unsure")), got)
    for label, lines, want in (("Dies moved to the top", (dies, birth, empty), 0),
                               ("Explodes inserted on top, which alive|8 offers", (explodes, dies, birth, empty), 1),
                               ("Explodes deleted again", (dies, birth, empty), 0)):
        got = settle(lines)
        check(f"outcome edits: {label}: {want} cells asked, and the outputs keep B2/S",
              got == (want, reference, alive, ('CHECKMARK', "18 cells · 0 unsure")), got)


def phase_status():
    """The status shows the plan's problems once the table is complete (§17.1): the cells no outcome applies to
    (§9.4), then the Factors CSV's warnings (a linked Import CSV wins over the typed path), the Outcomes CSV's (a
    number-like payload column gives no output) and the plan's (an outcome never offered)."""
    factors = os.path.join(FIXTURES, "life_factors.csv")
    outcomes = write("status_outcomes.csv", 'answer,when,if,scale\n'
                                            'Birth,comes to life,state=dead;neighbours=3,"1,5"\n'
                                            'Survives,stays alive,state=alive;neighbours=2|3,2\n'
                                            'Dies,dies,state=alive;neighbours!=2|3,3\n'
                                            'Ghost,never,state=dead;state=alive,4\n')
    tree, rules, obj, gi, go = scene("Status", factors, outcomes,
                                     {"in_state": ('BOOLEAN', [False]), "in_neighbours": ('INT', [3])},
                                     question="What happens?")
    reader = tree.nodes.new("GeometryNodeImportCSV")
    reader.inputs["Path"].default_value = factors
    tree.links.new(reader.outputs[0], rules.inputs["Factors CSV"])
    asked = passed(rules)
    job = engine._jobs(rules)[0][1]
    want = ("No outcome applies to state=dead, neighbours=0; state=dead, neighbours=1; state=dead, neighbours=2 and "
            "5 more cells",
            "Factors: using the linked Import CSV; the typed path is ignored",
            "status_outcomes.csv line 2: Column 'scale' has '1,5' in line 2; treated as text, no output",
            "status_outcomes.csv line 5: 'Ghost' is never offered: its 'if' matches no cell")
    status = engine.STATUS.get(rules.uid)
    check("status: nothing is asked, and the status shows the cells without an outcome first; the Job holds the "
          "Factors, Outcomes and plan warnings after it",
          not asked and job.problems == want and "scale" not in rules.outputs
          and status == ('ERROR', f"{want[0]}   (18 cells · 0 unsure · 8 with no outcome)"),
          (len(asked), job.problems, status))


def shared_scene(name, words, values):
    """A tree whose Rules node takes these words from the modifier (word -> one text per object), used by one
    object per item of values (its point attributes)."""
    tree = bpy.data.node_groups.new(name, 'GeometryNodeTree')
    tree.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    for word in words:
        tree.interface.new_socket(word, in_out='INPUT', socket_type='NodeSocketString')
    tree.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    gi, go = tree.nodes.new("NodeGroupInput"), tree.nodes.new("NodeGroupOutput")
    rules = tree.nodes.new(R.RulesSpec.idname)
    rules.inputs["Question"].default_value = "Which one?"
    for word in words:
        tree.links.new(gi.outputs[word], rules.inputs[word])
    objs = []
    for k, columns in enumerate(values):
        count = len(next(iter(columns.values()))[1])
        mesh = bpy.data.meshes.new(f"{name}{k}")
        mesh.from_pydata([(i * 0.01, 0.0, 0.0) for i in range(count)], [], [])
        for attribute, (kind, data) in columns.items():
            mesh.attributes.new(attribute, kind, 'POINT').data.foreach_set("value", data)
        obj = bpy.data.objects.new(f"{name}{k}", mesh)
        bpy.context.scene.collection.objects.link(obj)
        modifier = obj.modifiers.new("Rules", 'NODES')
        modifier.node_group = tree
        for word, texts in words.items():
            modifier[tree.interface.items_tree[word].identifier] = texts[k]
        obj.update_tag()
        objs.append(obj)
    return tree, rules, objs, gi, go


def phase_agreement():
    """§6.0: the contexts of one node must agree on what the graph bakes. Factors CSVs that differ only in their
    band edges, or Outcomes CSVs whose payload columns differ, give the error naming both contexts; nothing is
    asked, stored or rebuilt, so the last good table stays. Payload values and texts may differ. While a context has
    no job (a missing or broken CSV, an inline error), the sockets, their links and the graph stay: another context's
    new words are still asked and shown if they fit them, and wait if they would change them."""
    a, b = write("agree_a.csv", EDGES.format("0.5")), write("agree_b.csv", EDGES.format("0.5"))
    sizes = "answer,when,size\nA,the first,{0}\nB,the second,{1}\n"
    oa, ob = write("agree_oa.csv", sizes.format(1, 2)), write("agree_ob.csv", sizes.format(3, 4))
    heat = [0.1, 0.3, 0.6, 0.9]
    tree, rules, objs, gi, go = shared_scene("Agree", {"Factors": [a, b], "Outcomes": [oa, ob], "Context": ["", ""]},
                                             [{"in_heat": ('FLOAT', heat)}] * 2)
    answer_cells(planned(rules), lambda name: 0 if name == "cold" else 1)
    wire(tree, rules, gi, go, ["Outcome", "size"])
    planned(rules)
    labels = [ctx.label() for ctx in evaluator.find_uses()[tree.as_pointer()]]
    before = [read(obj, ["Outcome", "size"]) for obj in objs]
    check("agreement: two contexts with the same factors, and a payload column in both Outcomes CSVs, each show "
          "their own table", before == [{"Outcome": [0, 0, 1, 1], "size": [1, 1, 2, 2]},
                                        {"Outcome": [0, 0, 1, 1], "size": [3, 3, 4, 4]}]
          and engine.STATUS[rules.uid][0] == 'CHECKMARK', (before, engine.STATUS.get(rules.uid)))

    build, builds = inner.build, []

    def step(change):
        """Make a change, run a pass: (status, the Jobs it asks, whether it rebuilt the graph, whether the entries
        stayed, what both objects read)."""
        stored = [(entry.gn, entry.words_key) for entry in S.load(rules).entries]
        change()
        csvfmt.clear_cache()
        builds.clear()
        inner.build = lambda node_, *args: (builds.append(node_ == rules), build(node_, *args))[1]
        try:
            asked = planned(rules)
        finally:
            inner.build = build
        kept = [(entry.gn, entry.words_key) for entry in S.load(rules).entries] == stored
        return engine.STATUS.get(rules.uid), asked, any(builds), kept, [read(obj, ["Outcome", "size"]) for obj in objs]

    def edited(name, text, what):
        """Write one CSV, run a pass: (the status is the conflict, cells asked, builds, whether the entries and the
        outputs stayed)."""
        status, asked, built, kept, got = step(lambda: write(name, text))
        message = f"Rules: '{labels[0]}' and '{labels[1]}' use {what}"
        return status == ('ERROR', message), len(asked), built, kept, got == before

    got = edited("agree_b.csv", EDGES.format("0.2"), "Factors CSVs with different factors")
    check("agreement: Factors CSVs that differ only in their band edges give the error; nothing is asked, stored or "
          "rebuilt, and both objects keep the last good table", got == (True, 0, False, True, True),
          (got, engine.STATUS.get(rules.uid)))
    write("agree_a.csv", EDGES.format("0.2"))
    csvfmt.clear_cache()
    asked = planned(rules)
    got = [read(obj, ["Outcome"])["Outcome"] for obj in objs]
    check("agreement: once both use the new edges the rows carry over (0 requests) and the graph follows them",
          not asked and got == [[0, 1, 1, 1]] * 2 and engine.STATUS[rules.uid][0] == 'CHECKMARK',
          (len(asked), got, engine.STATUS.get(rules.uid)))
    before = [read(obj, ["Outcome", "size"]) for obj in objs]
    got = edited("agree_ob.csv", sizes.format(3.5, 4), "Outcomes CSVs with different payload columns")
    check("agreement: an Outcomes CSV whose payload column changes its type gives the error, and the last good table "
          "stays", got == (True, 0, False, True, True), (got, engine.STATUS.get(rules.uid)))

    # the hold is on the sockets and the graph, not on the answers: while a context has no job, another context's
    # new words that fit them are answered, and the ones that would change them wait
    def use(obj, word, text):
        obj.modifiers["Rules"][tree.interface.items_tree[word].identifier] = text
        obj.update_tag()

    def sockets():
        return [(s.name, s.is_linked) for s in rules.outputs] + [(n, rules.inputs[n].is_linked)
                                                                for n in factor_names(rules)]

    got = step(lambda: write("agree_ob.csv", sizes.format(3, 4)))
    good, face = got[4], sockets()
    check("agreement: the payload column's type back, both agree again", got[0][0] == 'CHECKMARK' and not got[1]
          and good == [{"Outcome": [0, 1, 1, 1], "size": [1, 2, 2, 2]}, {"Outcome": [0, 1, 1, 1], "size": [3, 4, 4, 4]}],
          got)
    missing = ('ERROR', f"Can't find {os.path.normpath(oa)}")
    os.remove(oa)
    waited = step(lambda: use(objs[1], "Context", "On a windy hill."))
    answer_cells(waited[1], lambda name: 1 if name == "cold" else 0)
    got = step(lambda: None)
    check("agreement: while an Outcomes CSV is missing, another context's new Context is asked, and shown once "
          "answered; the sockets keep their links, and the object whose CSV is missing its last table",
          waited[0] == got[0] == missing and [job.gn[1] for job in waited[1]] == ["On a windy hill."] and not got[1]
          and sockets() == face and got[4] == [good[0], {"Outcome": [1, 0, 0, 0], "size": [4, 3, 3, 3]}],
          (waited, got, sockets()))
    got = step(lambda: (write("agree_oa.csv", sizes.format(1, 2)), use(objs[1], "Context", "")))
    check("agreement: once the CSV is back, both show their tables with a checkmark, 0 requests",
          got[0][0] == 'CHECKMARK' and not got[1] and got[4] == good, got)
    payload = ('ERROR', f"Rules: '{labels[0]}' and '{labels[1]}' use Outcomes CSVs with different payload columns")
    got = (step(lambda: write("agree_oa.csv", "answer,when,weight\nA,the first,5\nB,the second,6\n")),
           step(lambda: use(objs[1], "Outcomes", "A: the first; A: again")), step(lambda: use(objs[1], "Outcomes", ob)))
    check("agreement: while the context whose payload the outputs show has an error of its own, the sockets, their "
          "links, the entries and the other object's table stay; fixed, the error is back",
          got[0][0] == got[2][0] == payload and got[1][0] == ('ERROR', "Outcomes: Entry 2: 'A' appears twice")
          and all(not g[1] and g[3] and g[4][0] == good[0] for g in got) and sockets() == face,
          (got, sockets()))
    got = step(lambda: write("agree_oa.csv", sizes.format(1, 2)))
    check("agreement: the first context's payload column back, a checkmark", got[0][0] == 'CHECKMARK'
          and got[4] == good, got)
    got = step(lambda: (write("agree_ob.csv", "answer,when,size\nA,the first,3\nA,again,4\n"),
                        write("agree_a.csv", EDGES.format("0.5"))))
    check("agreement: while an Outcomes CSV has errors, another context's new band edges wait: nothing is asked, "
          "stored or rebuilt, so the object whose CSV has errors is never read through them",
          got[0] == ('ERROR', "agree_ob.csv: Line 3: 'A' appears twice") and not got[1] and not got[2] and got[3]
          and got[4] == good, got)
    got = edited("agree_ob.csv", sizes.format(3, 4), "Factors CSVs with different factors")
    check("agreement: that CSV fixed, the band edges give the error, and both objects keep the last good table",
          got == (True, 0, False, True, True), got)
    got = step(lambda: write("agree_a.csv", EDGES.format("0.2")))
    check("agreement: the band edges back, a checkmark", got[0][0] == 'CHECKMARK' and got[4] == good, got)


class _PollProbe(bpy.types.Operator):
    """Runs a function from poll(), where Blender refuses ID writes as in a draw callback."""

    bl_idname = "openact_test.poll_probe"
    bl_label = "Poll probe"
    run = None
    result = None

    @classmethod
    def poll(cls, context):
        if cls.run is not None:
            try:
                cls.result = ("ok", cls.run())
            except Exception as error:  # noqa: BLE001 - the check reports it
                cls.result = (type(error).__name__, str(error))
        return False

    def execute(self, context):
        return {'FINISHED'}


def in_poll(function):
    _PollProbe.run, _PollProbe.result = function, None
    bpy.ops.openact_test.poll_probe.poll()
    _PollProbe.run = None
    return _PollProbe.result


ONE = "factor,kind,level,value,when\nheat,amount,cold,,Cold.\nheat,amount,hot,,Hot.\n"
TWO_FACTORS = ONE + "wind,flag,calm,0,Calm.\nwind,flag,windy,1,Windy.\n"


def phase_writes():
    """Only the pass writes the node (§5.3): planning from a draw, here Operator.poll(), changes nothing, and after an
    undo, which restores the interface, the entries and the group but not the plans, the next pass gives the node
    the sockets of its Job again, so the outputs come back."""
    bpy.utils.register_class(_PollProbe)
    points = {"in_heat": ('FLOAT', [0.1, 0.9]), "in_wind": ('BOOLEAN', [False, True])}
    path = write("draw_factors.csv", TWO_FACTORS)
    tree, rules, obj, gi, go = scene("Draw", path, write("draw_two.csv", TWO), points, question="Which one?")
    answer_cells(planned(rules), lambda name: 0 if name.startswith("cold") else 1)
    wire(tree, rules, gi, go, ["Outcome", "Confidence"])
    planned(rules)

    def state():
        return (factor_names(rules), len(rules.node_tree.nodes), inner.BUILT in rules.node_tree,
                read(obj, ["Outcome", "Confidence"]))

    before = state()
    write("draw_factors.csv", ONE)  # a factor removed on disk, planned first by a draw
    csvfmt.clear_cache()
    result = in_poll(lambda: len(engine.answers_for(rules)))
    after = state()
    check("writes: planning from Operator.poll() leaves the sockets, the graph and the outputs as they are",
          result == ("ok", 1) and after == before and before[0] == ["heat", "wind"], (result, before, after))
    planned(rules)
    check("writes: the next pass gives the node its Job's sockets", factor_names(rules) == ["heat"],
          factor_names(rules))

    path = write("undo_factors.csv", ONE)
    tree, rules, obj, gi, go = scene("Undo", path, write("undo_two.csv", TWO), points, question="Which one?")
    answer_cells(planned(rules), lambda name: 0 if name.startswith("cold") else 1)
    wire(tree, rules, gi, go, ["Outcome"])
    planned(rules)
    old_defs, old_entries = SPEC.interface(rules), rules.entries
    write("undo_factors.csv", TWO_FACTORS)
    csvfmt.clear_cache()
    answer_cells(planned(rules), lambda name: 0 if name.startswith("cold") else 1)
    wire_inputs(tree, rules)  # the new wind socket
    planned(rules)
    two = (factor_names(rules), read(obj, ["Outcome"])["Outcome"])
    inner.sync_interface(rules, old_defs)  # what an undo restores; the plans are Python state and stay
    rules.entries = old_entries
    inner.build(rules, S.load(rules))
    asked = planned(rules)
    got = (factor_names(rules), read(obj, ["Outcome"])["Outcome"])
    check("writes: after an undo the next pass gives the node its sockets back and the outputs return, 0 requests",
          two == (["heat", "wind"], [0, 1]) and not asked and got == two, (two, len(asked), got))
    bpy.utils.unregister_class(_PollProbe)


# --------------------------------------------------------------------------- the real model

def phase_model(port, gpu):
    prefs = P.get_prefs()
    prefs.port, prefs.gpu = port, gpu
    tree, rules, obj, gi, go, _start = life_scene()
    started = time.time()
    for outcomes, (birth, survive) in RULE_SETS.items():
        path = os.path.join(FIXTURES, outcomes)
        rules.inputs["Outcomes"].default_value = path
        engine.schedule(immediate=True)
        t0 = time.time()

        shown = [0.0]

        def done(path=path):
            if time.time() - shown[0] > 15:
                shown[0] = time.time()
                print(f"  ... {server.status()} {engine.STATUS.get(rules.uid)} pending {len(pipeline.PENDING)}",
                      flush=True)
            entries = [e for e in S.load(rules).entries if e.gn[3] == path]
            return bool(entries) and not pipeline.PENDING and engine.STATUS.get(rules.uid, ("",))[0] == 'CHECKMARK'

        answered = drive(engine, done, timeout=600)
        entry = next((e for e in S.load(rules).entries if e.gn[3] == path), None)
        names = [o.name for o in csvfmt.parse_answer_set(csvfmt.read_csv(path), "one_of", "rules").options]
        table = [names[o] if o >= 0 else "?" for o in entry.rows.v["outcome"]] if entry else []
        reference = [outcome_name(s, n, birth, survive) for s in (0, 1) for n in range(9)]
        exact = sum(t == r for t, r in zip(table, reference))
        unsure = sum(1 for s, c in zip(entry.rows.src, entry.rows.v["c"]) if s == "m" and c < R.UNSURE_BELOW) \
            if entry else -1
        print(f"TIMING {outcomes}: {time.time() - t0:.1f}s; {engine.STATUS.get(rules.uid)}", flush=True)
        check(f"model: {outcomes} is {exact}/18 exact with {unsure} unsure", answered and exact == 18 and unsure == 0,
              [(i, t, r) for i, (t, r) in enumerate(zip(table, reference)) if t != r])
    print(f"TIMING model total {time.time() - started:.1f}s", flush=True)
    server.stop()


try:
    if MODEL:
        phase_model(int(MODEL[0]), int(MODEL[1]))
    else:
        phase_cell_index()
        phase_life()
        phase_interpolation()
        phase_dither()
        phase_layout_edit()
        phase_long_header()
        phase_outcome_edits()
        phase_status()
        phase_agreement()
        phase_writes()
except Exception:
    import traceback

    traceback.print_exc()
    check("no exception", False)
finally:
    shutil.rmtree(TMP, ignore_errors=True)
finish()