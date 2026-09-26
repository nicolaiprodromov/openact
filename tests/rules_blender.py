"""The Rules node (§6.5, §9) in a headless Blender. Without arguments no model is used: synthetic answers go
through the real pass (the Game of Life, in a Simulation Zone, against a Python B3/S23 reference) or straight into
the store (the cell index for every combination, interpolation, renormalisation and dithering against rules_core's
references), and a rebuild with the CSV files deleted must leave every output unchanged.

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
from openact import csvfmt, evaluator, inner, pipeline  # noqa: E402
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


def wire(tree, rules, gi, go, outputs, geometry=None):
    """Factor sockets <- Named Attribute in_<factor>; each output -> Store Named Attribute out_<output>."""
    for socket in rules.inputs:
        if socket.name in R.HEAD or socket.type not in TYPES or socket.is_linked:
            continue
        attribute = tree.nodes.new("GeometryNodeInputNamedAttribute")
        attribute.data_type = TYPES[socket.type]
        attribute.inputs["Name"].default_value = "in_" + socket.name
        tree.links.new(enabled(attribute.outputs, "Attribute"), socket)
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
    """Plan in obj's context (which syncs the sockets), store the table values(i, row) gives, rebuild."""
    job = SPEC.plan(rules, context(rules, obj))
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
          read(obj, ["Outcome", "Confidence"]) == before and engine.STATUS[rules.uid].icon == 'CANCEL',
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
          and (engine.STATUS[rules.uid].icon, engine.STATUS[rules.uid].text) == ('CHECKMARK', "18 cells · 0 unsure"),
          engine.STATUS.get(rules.uid))
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
            return bool(entries) and not pipeline.PENDING and getattr(engine.STATUS.get(rules.uid), "icon", "") == \
                'CHECKMARK'

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
except Exception:
    import traceback

    traceback.print_exc()
    check("no exception", False)
finally:
    shutil.rmtree(TMP, ignore_errors=True)
finish()