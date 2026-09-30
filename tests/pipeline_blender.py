"""The decision pipeline (§5.5, §10.3, §10.5) in a headless Blender, against the fake model server that the add-on
starts itself (OPENACT_BACKEND=fake), whose --record log holds every request it gets:
- the launch token: in the server's environment, never in its argv; the server this Blender started refuses a request
  without it or with a wrong one; every request the add-on sends carries it; the 403 path, with a server another
  Blender started: its refusal lasts while it runs (refresh, Stop Model), nothing fails, and once it has gone the
  pass starts this Blender's own server and asks again;
- chunks sized from /v1/capabilities before the first send, of at most 32 questions and 64 scored rows (a 40-label
  tournament costs 8 rows a question); a list of 300 items, above the server's 256 questions per request, answered
  correctly in chunks; the server's per-question cache (cached_questions);
- priorities: a P0 job overtakes a P2 fill after the chunk in flight; the jobs of one class take turns; the pass's
  classes;
- cancellation: queued chunks go and the one in flight completes; a node's new words drop its old queued chunks; so
  do deleting the node, opening another file and one of its contexts going away (an object deleted), whose words are
  then never asked while the other's cells are each asked once;
- 422 isolation: the question the server names fails alone, else the chunk is halved until it travels alone;
- the render pause (no new chunk) and the playback pause (the worker finishes, no pass writes a table), driven by the
  render_running and playing events tick() publishes; the worker never touches bpy;
- stale-while-refilling: carried rows keep their answers, an edited row shows its old one (stale) and a new row its
  neutral value (pending), and the status record counts them with the ETA of pipeline.progress(); a new table with
  no row known yet stores nothing; a row whose question failed shows its neutral value, not its old answer; the
  table so far shown in the old entry's place (an Items CSV edit: the same words) while nothing refills it is
  written once, and then only the safety passes run, writing nothing;
- a simulated 5 s cold load: a P0 job queued after a P2 chunk (503 loading) is answered first once the model is ready;
  a failed load (503 error) fails the cells without requeueing them;
- unregistering and registering again mid-fill (disable and enable, Reload Scripts): the fill completes;
- entry validity (§5.4), from the --record log: a template bump and a Balance Option Order toggle send 0 requests;
  after a text edit only the row whose wkey changed is asked, with the node's current options.debias; Ask Again
  re-asks with it; two nodes with different toggles send their own debias;
- the plan cache: a node read as words from a nested group of a tree no modifier uses (asked through Ask Again, as
  nothing evaluates it) keeps the context only the evaluator reaches, so the scene settles (only safety passes,
  nothing planned again, no generation bumped); TIMING per unchanged node and per idle pass, with a 4096-cell Rules
  node used by 16 contexts.

Every node this suite expects to ask gets a Trigger that is on (_harness.trigger()), as a node asks only then (§6.0).

Run:  blender -b --factory-startup --python tests/pipeline_blender.py
"""

import contextlib
import http.client
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections import Counter

TMP = tempfile.mkdtemp(prefix="openact_pipeline_")
RECORD = os.path.join(TMP, "record.jsonl")
os.environ["OPENACT_BACKEND"] = "fake"
os.environ["OPENACT_SERVER_PYTHON"] = sys.executable  # the fake backend needs no runtime: Blender's Python runs it
os.environ["OPENACT_STATE_DIR"] = os.path.join(TMP, "state")  # the server log stays out of any shared runtime

import bpy  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import FAILURES, check, drive, finish, store, trigger  # noqa: E402
from openact import csvfmt, engine, lists, nodes, pipeline, server, sources, tables  # noqa: E402
from openact import prefs as P  # noqa: E402
from openact import store as S  # noqa: E402
from openact.specs import SPECS  # noqa: E402
from openact.specs.base import Spec  # noqa: E402

ASK, MATCH = SPECS["GeometryNodeOpenActAsk"], SPECS["GeometryNodeOpenActMatch"]
CLASSIFY, DIALS, PICK, RULES = (SPECS[f"GeometryNodeOpenAct{name}"] for name in ("Classify", "Dials", "Pick", "Rules"))
POINTS = 5
REFUSED = 2  # the 403s this suite causes on purpose in the log: 2 requests sent straight to the server
LONG = "x " * 20000  # far above the fake model's default limit of 8192 tokens, like a prompt too long for SemIf


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


PORT = free_port()
P._Defaults.port, P._Defaults.auto_start = PORT, True  # the add-on isn't enabled, so these are its preferences
server.FAKE_ARGS[:] = ["--record", RECORD]
nodes.register()
engine.register()


class Tap:
    """Wraps server.http around /v1/systemone, which the worker calls from its thread: every exchange (request, status,
    reply) in order, with an optional delay before each (a slow model) and an optional rewrite of the reply."""

    def __init__(self):
        self.http, self.log, self.delay, self.rewrite = server.http, [], 0.0, None
        self.started = threading.Event()  # set as each request starts
        server.http = self

    def __call__(self, method, path, body=None, port=None, timeout=10.0):
        if path != "/v1/systemone":
            return self.http(method, path, body, port, timeout)
        self.started.set()
        time.sleep(self.delay)
        code, reply = self.http(method, path, body, port, timeout)
        if self.rewrite is not None:
            code, reply = self.rewrite(code, reply)
        self.log.append((body, code, reply))
        return code, reply

    def since(self, start):
        return self.log[start:]


TAP = Tap()


# ------------------------------------------------------------------------------ helpers

def records(start=0):
    """The fake server's --record log from line `start` on: {"status", "request", "answers"} of each request."""
    with open(RECORD, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle][start:]


def questions(start):
    """(instructions, options.debias) of every question the server got since record `start`."""
    return [(question["instructions"], record["request"]["options"]["debias"]) for record in records(start)
            if record["request"] for question in record["request"]["questions"].values()]


def health():
    return server.http("GET", "/health")[1]


def post(headers):
    """The status of a request sent straight to the fake server, around the add-on."""
    connection = http.client.HTTPConnection("127.0.0.1", PORT, timeout=30)
    try:
        request = {"state": "A plain room.", "questions": {"q": {"type": "noul", "instructions": "Is it dark?"}}}
        connection.request("POST", "/v1/systemone", body=json.dumps(request),
                           headers={"Content-Type": "application/json", **headers})
        response = connection.getresponse()
        response.read()
        return response.status
    finally:
        connection.close()


def settle(timeout=30):
    """Tick until nothing is queued or in flight and no pass is due."""
    return drive(engine, lambda: not pipeline.PENDING and not engine._dirty, timeout)


@contextlib.contextmanager
def paused():
    """The worker starts no chunk, as during a render job. Passes run through run_pass() meanwhile: a tick would
    publish that no render job runs."""
    server.Worker.render_running.set()
    time.sleep(0.3)  # a worker waiting for work has seen it
    try:
        yield
    finally:
        server.Worker.render_running.clear()


def queued(uid):
    with pipeline._QUEUED:
        return [chunk for jobs in pipeline.QUEUE for chunks in jobs for chunk in chunks if chunk.job.uid == uid]


def cells(job):
    return [row.cell for row in job.rows if row.cell is not None]


def states(start):
    return [body["state"] for body, _code, _reply in TAP.since(start)]


def match_job(uid, state, count, whens=None):
    """A Match job asking "Is it red?" about `count` items that share the state (their Context)."""
    items = [lists.Item(f"item {i}", (whens or {}).get(i, "")) for i in range(count)]
    return MATCH.job(("Is it red?", state, "", "", "", ""), 'YES_NO', items, sources.default_set("yes_no"), True, uid)


def ask_job(uid, context):
    return ASK.job(("Is it night?", context, "", "", ""), 'YES_NO', True, uid)


def match_tree(name, *words):
    """A tree used by one object of POINTS points, with a Match node per (question, items) typed on it, whose
    Probability is stored as p0, p1, ..., and whose Trigger is on."""
    tree = bpy.data.node_groups.new(name, 'GeometryNodeTree')
    tree.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    tree.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    geometry, found = tree.nodes.new("NodeGroupInput").outputs["Geometry"], []
    for i, (question, items) in enumerate(words):
        node = tree.nodes.new(MATCH.idname)
        node.inputs["Question"].default_value = question
        node.inputs["Items"].default_value = items
        geometry = store(tree, geometry, node.outputs["Probability"], f"p{i}", 'FLOAT', 300 * i)
        trigger(node)
        found.append(node)
    tree.links.new(geometry, tree.nodes.new("NodeGroupOutput").inputs["Geometry"])
    return tree, found, user(name, tree)


def user(name, tree):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([(i, 0, 0) for i in range(POINTS)], [], [])
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.modifiers.new("OpenAct", 'NODES').node_group = tree
    return obj


def read(obj, name="p0"):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        found = mesh.attributes.get(name)
        return [round(d.value, 4) for d in found.data] if found else None
    finally:
        evaluated.to_mesh_clear()


def entry_of(node):
    """The stored entry of the words the node's one context has now."""
    gn = engine._jobs(node)[0][1].gn
    return next((entry for entry in S.load(node).entries if entry.gn == gn), None)


def whole(node, rows):
    """True once the node's table of `rows` rows is stored, every row answered by the model."""
    entry = entry_of(node)
    return entry is not None and entry.rows.src == "m" * rows


def line(node):
    """The node's status line as its body draws it: (icon, text)."""
    found = engine.STATUS.get(node.uid)
    return (found.icon, found.text) if found else ("", "")


def answers_health(port):
    """True while a server answers /health on the port."""
    try:
        return server.http("GET", "/health", port=port, timeout=1.0)[0] == 200
    except OSError:
        return False


def foreign_server():
    """A fake server another Blender started, on a port of its own, with a launch token this Blender hasn't got:
    (port, process)."""
    port = free_port()
    env = dict(server._child_env(), OPENACT_TOKEN=secrets.token_hex(32))
    process = subprocess.Popen([sys.executable, server.SERVER_SCRIPT, "--port", str(port), "--backend", "fake",
                                "--parent-pid", str(os.getpid())], env=env, stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return port, process


# ------------------------------------------------------------------------------ the token

def phase_token():
    started = server.start()
    check("start: the add-on starts the fake server on its port, which gets ready",
          started and drive(engine, server.is_ready, 30), server.status())
    token, argv = server.STATE.token, " ".join(server.STATE.process.args)
    check("token: a new 64-digit launch token, passed in the environment and never in argv",
          len(token) == 64 and set(token) <= set("0123456789abcdef") and token not in argv, argv)
    got = [post({}), post({"X-OpenAct-Token": "0" * 64}), post({"X-OpenAct-Token": token})]
    check("token: the server this Blender started refuses a request without its token or with a wrong one",
          got == [403, 403, 200], got)


def phase_refused():
    """The 403 path (§10.5): a server another Blender started holds the port. Its refusal lasts while it runs: the
    status stays an error that says what to do, Stop Model can't stop it and says so, and nothing fails. Once it has
    gone, the pass starts this Blender's own server, and the node is answered with no Ask Again."""
    port, foreign = foreign_server()
    message = server.TOKEN_REFUSED.format(port)
    server.stop()
    P._Defaults.port = port
    try:
        up = drive(engine, lambda: answers_health(port), 15)
        _tree, (node,), _obj = match_tree("Foreign", ("Is it foreign?", "coin, stamp, flag"))
        start = len(TAP.log)
        engine.run_pass()  # auto-start attaches to that server, and the next pass asks it
        refused = drive(engine, lambda: line(node) == ('CANCEL', message), 15)
        keys = [cell.key for cell in cells(engine._jobs(node)[0][1])]
        got = (server.status(), [code for _body, code, _reply in TAP.since(start)],
               [key for key in keys if key in pipeline.FAILED or key in pipeline.PENDING])
        check("403: a server another Blender started refuses this Blender's token: the status and the node show an "
              "error that says what to do, after one request, and its cells neither fail nor stay pending",
              up and refused and got == (("error", message), [403], []), got)
        server.refresh()
        drive(engine, lambda: False, 1.5)  # the probe, and 1.5 s of passes
        check("403: while it runs, a refresh keeps the error (its pid is remembered), and nothing is asked again",
              server.status() == ("error", message) and len(TAP.log) == start + 1,
              (server.status(), len(TAP.log) - start))
        server.stop()
        check("403: Stop Model can't stop it (/shutdown needs its token): the status keeps the error, and it runs on",
              server.status() == ("error", message) and answers_health(port), server.status())
        foreign.terminate()
        foreign.wait(10)
        server.refresh()  # as tick() does every 5 s
        done = drive(engine, lambda: whole(node, 3), 30)
        check("403: once it has gone, the pass starts this Blender's own server and the node is answered, with no Ask "
              "Again", done and server.STATE.process is not None and line(node)[0] == 'CHECKMARK',
              (server.status(), engine.STATUS.get(node.uid)))
    finally:
        if foreign.poll() is None:
            foreign.kill()
        server.stop()
        P._Defaults.port = PORT
    server.start()
    check("403: this Blender's server is ready again on its own port", drive(engine, server.is_ready, 30),
          server.status())


def phase_token_audit():
    statuses = Counter(record["status"] for record in records())
    check(f"token: every request the add-on sent carried the token: the only 403s are the {REFUSED} this suite caused",
          statuses[403] == REFUSED, statuses)


# ------------------------------------------------------------------------------ chunks

def phase_chunks():
    """The chunk limits (§5.5), and a list of 300 items where a request may hold 256 questions (§10.4)."""
    pipeline.LIMITS = (4, 8)  # not the server's: the worker must size the chunks before its first send to it
    server.worker().sized = None  # as for a server process it hasn't sent to yet
    sized = match_job("sized", "A measured room.", 3)
    pipeline.submit(sized, cells(sized), 0)
    settle()
    check("chunks: before its first send to a server, the worker sizes the chunks from its /v1/capabilities (32 "
          "questions and 2 x 32 rows)", pipeline.LIMITS == (32, 64), pipeline.LIMITS)
    big = match_job("three hundred", "A long list of things.", 300)
    labels = csvfmt.parse_inline_set("; ".join(f"label {i}" for i in range(40)), "one_of")
    tournament = CLASSIFY.job(("", "A shelf of things.", "", "", "", ""), [lists.Item(f"thing {i}") for i in range(20)],
                              labels, True, "tournament")
    start = len(records())
    pipeline.submit(big, cells(big), 2)
    pipeline.submit(tournament, cells(tournament), 2)
    settle(60)
    sent, by_id = records(start), {}
    for cell in cells(big) + cells(tournament):
        by_id[cell.key[:16]] = by_id[cell.key] = cell
    sizes = sorted((len(r["request"]["questions"]), sum(by_id[q].cost for q in r["request"]["questions"])) for r in sent)
    check("chunks: at most 32 questions and 64 scored rows each: 32 yes/no questions with Balance Option Order, 8 "
          "questions of a 40-label tournament (8 rows each)",
          sizes == sorted([(32, 64)] * 9 + [(12, 24)] + [(8, 64)] * 2 + [(4, 32)]), sizes)
    requests = [r for r in sent if r["request"]["state"] == "A long list of things."]
    once = Counter(qid for r in requests for qid in r["request"]["questions"])
    right = len(once) == 300 and set(once.values()) == {1} and all(r["status"] == 200 for r in requests)
    for r in requests:
        for qid, answer in r["answers"].items():
            got = pipeline.cached(by_id[qid].key)
            right = right and got is not None and dict(got, model=None) == dict(
                pipeline.parse(by_id[qid], answer), model=None)
    check("300 items, more than the 256 questions a request may hold: 10 chunks, each item asked once and given its "
          "own question's answer", len(big.rows) == 300 and len(requests) == 10 and right, (len(requests), len(once)))
    return big


def phase_cache(big):
    """The server's per-question cache (§10.4): cells asked again come back as cached_questions and score nothing."""
    again = cells(big)[:40]
    pipeline.forget(cell.key for cell in again)
    before, start = health(), len(TAP.log)
    pipeline.submit(big, again, 2)
    settle()
    after = health()
    got = [reply.get("cached_questions") for _body, _code, reply in TAP.since(start)]
    check("server cache: 40 cells asked again come from the server's cache (cached_questions 32 + 8), none scored",
          got == [32, 8] and after["rows_scored"] == before["rows_scored"]
          and after["cache_hits"] - before["cache_hits"] == 40, (got, before["rows_scored"], after["rows_scored"]))


# ------------------------------------------------------------------------------ priorities and cancellation

def phase_priorities():
    fill, quick = match_job("fill", "A big warehouse.", 150), ask_job("quick", "A dark street.")
    TAP.delay, start = 0.3, len(TAP.log)
    TAP.started.clear()
    try:
        pipeline.submit(fill, cells(fill), 2)
        TAP.started.wait(10)
        pipeline.submit(quick, cells(quick), 0)
        settle(60)
    finally:
        TAP.delay = 0.0
    got = states(start)
    check("priorities: a P0 job overtakes a P2 fill after the chunk in flight",
          got == ["A big warehouse.", "A dark street."] + ["A big warehouse."] * 4, got)
    with paused():
        a, b = match_job("turn a", "Room A.", 96), match_job("turn b", "Room B.", 96)
        start = len(TAP.log)
        pipeline.submit(a, cells(a), 2)
        pipeline.submit(b, cells(b), 2)
    settle(60)
    got = states(start)
    check("priorities: the jobs of one class take turns, a chunk each", got == ["Room A.", "Room B."] * 3, got)
    got = [engine._priority(spec, None, [None] * count) for spec, count in (
        (ASK, 1), (DIALS, 20), (MATCH, 8), (MATCH, 9), (RULES, 4096))]
    check("priorities: the pass asks Ask, Pick and Dials, and any job of at most 8 missing cells, at P0; other tables "
          "at P2 (P1 is the active node of a node editor)", got == [0, 0, 0, 2, 2], got)


def phase_cancel():
    job = match_job("cancelled", "A cancelled order.", 96)  # 3 chunks
    job.generation = 1
    TAP.delay = 0.3
    TAP.started.clear()
    try:
        pipeline.submit(job, cells(job), 2)
        TAP.started.wait(10)
        pipeline.cancel("cancelled", below=2)
        dropped = cells(job)[32:]
        gone = not queued("cancelled") and not any(cell.key in pipeline.PENDING for cell in dropped)
        settle()
    finally:
        TAP.delay = 0.0
    check("cancel: the queued chunks of older generations go at once, and their cells stop being pending", gone)
    check("cancel: the chunk in flight completes and its answers land in MEMORY; the dropped ones were never sent",
          all(pipeline.cached(cell.key) for cell in cells(job)[:32])
          and not any(pipeline.cached(cell.key) for cell in dropped)
          and [body["state"] for body, _code, _reply in TAP.log].count("A cancelled order.") == 1)

    _tree, (node,), _obj = match_tree("Cancel", ("Is it heavy?", ", ".join(f"thing {i}" for i in range(60))))
    with paused():
        engine.run_pass()
        old = queued(node.uid)
        node.inputs["Question"].default_value = "Is it light?"
        start = len(records())
        engine.run_pass()
        new = queued(node.uid)
        pending = {cell.key for chunk in old for cell in chunk.cells} & set(pipeline.PENDING)
    check("cancel: a node's new words bump its generation and drop its queued chunks; the new words' chunks wait",
          len(old) == 2 and len(new) == 2 and not pending
          and {chunk.job.generation for chunk in new} == {engine.GENERATIONS[node.uid]}
          and {chunk.job.generation for chunk in old} == {engine.GENERATIONS[node.uid] - 1}, (len(old), len(new)))
    settle()
    sent = questions(start)
    check("cancel: then only the new words are asked", len(sent) == 60 and all(
        text.endswith("\nIs it light?") for text, _debias in sent), len(sent))


def phase_isolation():
    job = match_job("isolated", "A pantry.", 10, {3: LONG})
    start = len(TAP.log)
    pipeline.submit(job, cells(job), 0)
    settle()
    bad = cells(job)[3]
    codes = [code for _body, code, _reply in TAP.since(start)]
    check("422: the question the server names fails alone ('... exceed limit 8192'), and the other 9 are answered by "
          "one more request", codes == [422, 200] and "exceed limit 8192" in pipeline.FAILED.get(bad.key, "")
          and all(pipeline.cached(cell.key) for cell in cells(job) if cell is not bad),
          (codes, pipeline.FAILED.get(bad.key)))

    job = match_job("halved", "A cellar.", 16, {11: LONG})
    TAP.rewrite = lambda code, reply: (code, dict(reply, question=None)) if code == 422 else (code, reply)
    start = len(TAP.log)
    try:
        pipeline.submit(job, cells(job), 0)
        settle()
    finally:
        TAP.rewrite = None
    bad = cells(job)[11]
    sizes = [(len(body["questions"]), code) for body, code, _reply in TAP.since(start)]
    check("422 with no question named: the chunk is halved until the bad question travels alone, 9 requests for 16",
          sizes == [(16, 422), (8, 200), (8, 422), (4, 422), (2, 200), (2, 422), (1, 200), (1, 422), (4, 200)], sizes)
    check("422 with no question named: only that question fails; the other 15 are answered",
          [cell.key for cell in cells(job) if cell.key in pipeline.FAILED] == [bad.key]
          and all(pipeline.cached(cell.key) for cell in cells(job) if cell is not bad))


# ------------------------------------------------------------------------------ pauses and partial tables

def phase_pauses():
    worker = (server.Worker.run, server.Worker.send, server.Worker.wait, server.Worker.size, TAP.http, server.post,
              server._offline, server._State.set, server._refused_by, pipeline.take, pipeline.requeue,
              pipeline.build_request, pipeline.size_chunks)
    check("pauses: the worker never touches bpy (neither its code nor what it calls names it); tick() publishes what "
          "it reads", not [fn.__name__ for fn in worker if "bpy" in fn.__code__.co_names])
    job, start = match_job("render", "A render farm.", 4), len(records())
    rendering, engine._rendering = engine._rendering, lambda: True  # stands in for bpy.app.is_job_running('RENDER')
    try:
        engine.tick()
        published = server.Worker.render_running.is_set()
        pipeline.submit(job, cells(job), 0)
        drive(engine, lambda: False, 1.0)  # a second of ticks while the render job runs
        held = len(records()) == start and all(cell.key in pipeline.PENDING for cell in cells(job))
    finally:
        engine._rendering = rendering
    check("render: tick() publishes render_running, and no new chunk starts while a render job runs", published and held,
          (published, held))
    settle()
    check("render: once it ends, the chunk goes out and is answered", not server.Worker.render_running.is_set()
          and len(records()) == start + 1 and all(pipeline.cached(cell.key) for cell in cells(job)))

    _tree, (node,), _obj = match_tree("Playback", ("Is it bright?", ", ".join(f"light {i}" for i in range(40))))
    with paused():
        engine.run_pass()  # two chunks wait
        unknown = not S.load(node).entries
    check("stale-while-refilling: while no row of a new table is known, nothing is stored (fallbacks show)", unknown)
    playing, engine._animation_playing = engine._animation_playing, lambda: True
    try:
        before = node.entries
        answered = drive(engine, lambda: not pipeline.PENDING, 30)
        published, kept = server.Worker.playing.is_set(), node.entries == before
    finally:
        engine._animation_playing = playing
    check("playback: tick() publishes playing; the worker finishes the queued chunks, and no pass writes a table",
          published and answered and kept, (published, answered, kept))
    settle()
    entry = entry_of(node)
    check("playback: once it stops, the pass writes the answered table", entry is not None
          and entry.rows.src == "m" * 40, entry and entry.rows.src)


def phase_stale():
    items = "spoon: a kitchen tool; chair; lamp; anvil"
    _tree, (node,), obj = match_tree("Stale", ("Is it made of metal?", items))
    settle()
    first = read(obj)
    with paused():
        node.inputs["Items"].default_value = items.replace("chair", "chair: a steel chair") + "; rug"
        start = len(records())
        engine.run_pass()
        entry, got = entry_of(node), read(obj)
        status, fill = engine.STATUS[node.uid], pipeline.progress(node.uid)
    check("stale-while-refilling: carried rows keep their answers, the edited row shows its old one (stale), the new "
          "row its neutral value (pending)", entry is not None and entry.rows.src == "msmmp" and got == first
          and entry.rows.v["p"][4] == 0.0, (entry and entry.rows.src, got, first))
    check("stale-while-refilling: the status record counts 5 rows (3 answered, 1 stale, 1 pending), with the ETA of "
          "pipeline.progress(): 0 of 2 cells done", (status.rows, status.answered, status.stale, status.pending) == (
              5, 3, 1, 1) and fill is not None and fill[:2] == (0, 2) and fill[2] > 0 and status.eta_s > 0,
          (status.rows, status.answered, status.stale, status.pending, fill, status.eta_s))
    settle()
    entry, sent = entry_of(node), sorted(questions(start))
    check("stale-while-refilling: then the table is whole, and only the edited and the new row were asked",
          entry is not None and entry.rows.src == "mmmmm" and pipeline.progress(node.uid) is None and sent == [
              ("Item: chair — a steel chair\nIs it made of metal?", True), ("Item: rug\nIs it made of metal?", True)],
          (entry and entry.rows.src, sent))


def phase_failed():
    """A row whose new question failed shows its neutral value, not its old meaning's answer: nothing refills it
    (§5.4, §6.0)."""
    items = "spoon; chair; lamp"
    _tree, (node,), obj = match_tree("Failed", ("Is it made of glass?", items))
    settle()
    first, error = read(obj), "The model refused: simulated model failure"
    TAP.rewrite = lambda code, reply: (500, {"error": "simulated model failure"}) if code == 200 else (code, reply)
    try:
        node.inputs["Items"].default_value = items.replace("lamp", "lamp: a glass lamp")
        engine.run_pass()
        failed = drive(engine, lambda: line(node) == ('CANCEL', error) and not pipeline.PENDING, 15)
    finally:
        TAP.rewrite = None
    entry, got, status = entry_of(node), read(obj), engine.STATUS[node.uid]
    counts = (status.rows, status.answered, status.stale, status.pending)
    check("failed row: once its question failed, the row shows its neutral value, pending, not its old meaning's "
          "answer; the status shows the failure, and the fill is over", failed and entry is not None
          and entry.rows.src == "mmp" and first[2] != 0.0 and got == first[:2] + [0.0] + first[3:]
          and counts == (3, 2, 0, 1) and pipeline.progress(node.uid) is None,
          (entry and entry.rows.src, first, got, counts))


def phase_idle():
    """The table so far shown in the old entry's place (§5.4) while nothing refills it: an Items CSV edit (the same
    words) with the model stopped. Its table object is written once, then only the 2 s safety passes run, writing
    nothing. Blender's event loop evaluates the depsgraph after each timer, so each tick here is followed by an
    evaluation: the update of a table write would schedule another pass."""
    path = os.path.join(TMP, "idle.csv")

    def items(lamp, stamp):  # each edit gets a new stamp, which the fingerprint holds
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(f"answer,when\nspoon,a kitchen tool\nchair,a seat\nlamp,{lamp}\n")
        os.utime(path, ns=(stamp, stamp))

    def ui(seconds):
        end = time.time() + seconds
        while time.time() < end:
            engine.tick()
            bpy.context.evaluated_depsgraph_get()
            time.sleep(0.05)

    def counted_pass(everything=True):
        passes.append(everything)  # the safety pass visits every node; a scheduled one passes False
        return run_pass(everything)

    def counted_write(node_, *args, **kwargs):
        writes.append(node_.uid)
        return write_rows(node_, *args, **kwargs)

    items("gives light", 10 ** 18)
    _tree, (node,), obj = match_tree("Idle", ("Is it made of metal?", path))
    settle()
    first, stored = read(obj), [entry.words_key for entry in S.load(node).entries]
    passes, writes, run_pass, write_rows = [], [], engine.run_pass, tables.write_rows
    engine.run_pass, tables.write_rows = counted_pass, counted_write
    P._Defaults.auto_start = False
    try:
        server.stop()
        ui(1.0)  # the pass the stop schedules
        writes.clear()
        items("gives light at night", 2 * 10 ** 18)  # only lamp's wkey changes
        ui(1.5)  # the watcher sees the edit, and a pass shows the table so far
        edited, status = len(writes), engine.STATUS[node.uid]
        passes.clear()
        ui(6.0)
        idle = (list(passes), len(writes) - edited)
    finally:
        engine.run_pass, tables.write_rows = run_pass, write_rows
        P._Defaults.auto_start = True
    counts = (status.rows, status.answered, status.stale, status.pending)
    check("partial in place: an Items CSV edit with the model stopped shows the table so far in the old entry's place "
          "(2 rows answered, 1 stale), and the old entry stays stored", status.icon == 'PAUSE' and counts == (3, 2, 1, 0)
          and [entry.words_key for entry in S.load(node).entries] == stored and read(obj) == first,
          ((status.state, status.text), counts))
    check("partial in place: with nothing to refill it, its table object is written once, and then only the 2 s "
          "safety passes run, writing nothing", edited <= 1 and idle[1] == 0 and 2 <= len(idle[0]) <= 4
          and all(idle[0]), (edited, idle))
    server.start()
    words_key = engine._jobs(node)[0][1].words_key
    done = drive(engine, lambda: any(entry.words_key == words_key and entry.rows.src == "mmm"
                                     for entry in S.load(node).entries), 30)
    got = read(obj)
    check("partial in place: once the model is back, the edited row is answered and stored, and the table object "
          "shows its answer", done and got[:2] == first[:2] and got[2] != first[2], (done, first, got))


# ------------------------------------------------------------------------------ entry validity (§5.4)

def phase_validity():
    _tree, (kitchen, soft), _obj = match_tree("Validity", ("Would you find it in a kitchen?", "pan, sofa, kettle"),
                                             ("Is it soft?", "pillow, brick"))
    soft.balance_order = False
    start = len(records())
    settle()
    got = {text.split("\n")[0]: debias for text, debias in questions(start)}
    check("validity: two nodes with different Balance Option Order toggles in one scene send their own debias",
          got == {"Item: pan": True, "Item: sofa": True, "Item: kettle": True, "Item: pillow": False,
                  "Item: brick": False}, got)

    saved, start = [kitchen.entries, soft.entries], len(records())
    Spec.PROMPT_VERSION += 1  # an add-on upgrade with new templates
    try:
        engine.run_pass()
        settle()
        kept = [kitchen.entries, soft.entries] == saved
    finally:
        Spec.PROMPT_VERSION -= 1
    engine.run_pass()
    settle()
    check("validity: a template bump sends 0 requests and keeps the stored answers", kept and len(records()) == start,
          len(records()) - start)

    saved = kitchen.entries
    kitchen.balance_order = False
    settle()
    check("validity: toggling a node's Balance Option Order sends 0 requests and keeps its stored answers",
          kitchen.entries == saved and len(records()) == start, len(records()) - start)

    kitchen.inputs["Items"].default_value = "pan, sofa, teapot"
    engine.run_pass()
    settle()
    sent = questions(start)
    check("validity: after a text edit only the row whose wkey changed is asked, with the node's current debias",
          sent == [("Item: teapot\nWould you find it in a kitchen?", False)], sent)

    start = len(records())
    engine.reask(kitchen)
    settle()
    sent = sorted(questions(start))
    check("validity: Ask Again re-asks every row with the node's current setting", sent == [
        (f"Item: {item}\nWould you find it in a kitchen?", False) for item in ("pan", "sofa", "teapot")], sent)


# ------------------------------------------------------------------------------ cold loads

def restart(*flags):
    server.stop()
    server.FAKE_ARGS[:] = ["--record", RECORD, *flags]
    server.start()


def phase_cold_load():
    restart("--fake-load-seconds", "5")
    loading = drive(engine, lambda: server.STATE.health.get("status") == "loading", 15)
    fill, quick = match_job("cold fill", "A cold warehouse.", 40), ask_job("cold quick", "A cold office.")
    start = len(TAP.log)
    pipeline.submit(fill, cells(fill), 2)
    bounced = drive(engine, lambda: len(TAP.log) > start, 10)
    pipeline.submit(quick, cells(quick), 0)
    ready = drive(engine, server.is_ready, 20)
    settle()
    sent = [(body["state"], code, reply.get("status")) for body, code, reply in TAP.since(start)]
    check("cold load: while the model loads, a chunk gets 503 loading at once and goes back to the head of the queue",
          loading and bounced and sent[:1] == [("A cold warehouse.", 503, "loading")], sent)
    check("cold load: a P0 job queued after that P2 chunk is answered first once the model is ready",
          ready and [state for state, code, _status in sent if code == 200] == [
              "A cold office.", "A cold warehouse.", "A cold warehouse."]
          and all(pipeline.cached(cell.key) for cell in cells(fill) + cells(quick)), sent)

    restart("--fake-load-seconds", "2", "--fake-load-fail")
    drive(engine, lambda: server.STATE.health.get("status") == "loading", 15)
    job, start = match_job("failed load", "A broken machine.", 3), len(TAP.log)
    pipeline.submit(job, cells(job), 0)
    failed = drive(engine, lambda: all(cell.key in pipeline.FAILED for cell in cells(job)), 20)
    drive(engine, lambda: False, 1.5)  # nothing is sent again
    sent = [(code, reply.get("status")) for _body, code, reply in TAP.since(start)]
    check("failed load: a 503 error fails the cells with the server's message and never requeues them",
          failed and sent == [(503, "loading"), (503, "error")] and server.status()[0] == "error"
          and all("simulated load failure" in pipeline.FAILED[cell.key] for cell in cells(job)), (sent, server.status()))
    pipeline.forget(cell.key for cell in cells(job))
    restart()
    check("cold load: the server starts again", drive(engine, server.is_ready, 30), server.status())


# ------------------------------------------------------------------------------ leaving a fill

def phase_reregister():
    """Disabling and enabling the add-on, or Reload Scripts, mid-fill: the add-on's modules stay loaded. Unregistering
    stops the worker and drops the queued chunks (the one it held may come back to the queue); after registering, the
    next pass starts a worker again, which completes the fill."""
    job = match_job("restart", "A silent room.", 40)
    with paused():
        pipeline.submit(job, cells(job), 2)
        server.worker().stopped = True  # gone, as after an unregister, with its chunks still queued
        server._worker = None
        pipeline.submit(job, cells(job), 2)  # every cell is pending already
        restarted = server._worker is not None and server._worker.is_alive()
    server.worker()  # so that a failed check leaves no work behind
    settle()
    check("reregister: a submit whose cells are all pending already starts a worker when none runs",
          restarted and all(pipeline.cached(cell.key) for cell in cells(job)))

    _tree, (node,), _obj = match_tree("Reregister", ("Is it loud?", ", ".join(f"bell {i}" for i in range(100))))
    TAP.delay = 2.0
    TAP.started.clear()
    try:
        engine.run_pass()  # 4 chunks: the worker holds the first
        TAP.started.wait(10)
        asked = cells(engine._jobs(node)[0][1])
        held, dropped = {cell.key for cell in asked[:32]}, {cell.key for cell in asked[32:]}
        engine.unregister()  # what the add-on's unregister() does to the engine and the server
        server.shutdown_owned()
        left = {cell.key for chunk in queued(node.uid) for cell in chunk.cells}
    finally:
        TAP.delay = 0.0
    check("reregister: unregistering drops the queued chunks, whose cells stop being pending; only the chunk the "
          "worker held may stay", not dropped & set(pipeline.PENDING) and left <= held,
          (len(dropped & set(pipeline.PENDING)), len(left)))
    engine.register()
    server.refresh()
    start = len(records())
    done = drive(engine, lambda: whole(node, 100) and not pipeline.PENDING, 30)
    sent = Counter(text for text, _debias in questions(start))
    check("reregister: after registering, the next pass starts a worker, and the fill completes with no bell asked "
          "twice", done and max(sent.values(), default=0) == 1, (done, len(sent), max(sent.values(), default=0)))


def fill_then(name, question, leave):
    """Start a 100-item fill of a new node, and leave(tree, obj) it while the worker holds the first chunk: (its cells,
    the chunks still queued for it, the requests the fake server answered about it since)."""
    tree, (node,), obj = match_tree(name, (question, ", ".join(f"{name.lower()} {i}" for i in range(100))))
    start = len(records())
    TAP.delay = 2.0
    TAP.started.clear()
    try:
        engine.run_pass()  # 4 chunks
        TAP.started.wait(10)
        asked, uid = cells(engine._jobs(node)[0][1]), node.uid
        leave(tree, obj)
        left = queued(uid)
    finally:
        TAP.delay = 0.0
    settle()
    sent = [record for record in records(start) if record["request"] and any(
        one["instructions"].endswith(f"\n{question}") for one in record["request"]["questions"].values())]
    return asked, left, sent


def phase_gone():
    """The queued chunks of a deleted node, and of a closed file, go (§5.5): the chunk in flight completes into MEMORY,
    and nothing more is sent for them."""
    def delete(tree, obj):
        bpy.data.objects.remove(obj)
        bpy.data.node_groups.remove(tree)
        engine.run_pass()

    for what, name, question, leave in (
            ("a deleted node: the next pass drops its queued chunks", "Ghost", "Is it a ghost?", delete),
            ("another file opened: every queued chunk of the closed one goes", "Door", "Is it a door?",
             lambda _tree, _obj: bpy.ops.wm.read_homefile(use_empty=True))):
        asked, left, sent = fill_then(name, question, leave)
        check(f"{what}; the chunk in flight completes into MEMORY, and no other request is sent about it",
              not left and len(sent) == 1 and all(pipeline.cached(cell.key) for cell in asked[:32])
              and not any(pipeline.cached(cell.key) for cell in asked[32:]), (len(left), len(sent)))


def phase_lost():
    """A context that goes away while its node stays (§5.5): one of the two objects whose modifiers give a node its
    Question is deleted while the worker holds the node's first chunk. The chunks queued for its words go and none is
    sent; the chunk in flight completes, and the other object's cells are queued again, each asked once."""
    tree = bpy.data.node_groups.new("Lost", 'GeometryNodeTree')
    tree.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    question = tree.interface.new_socket("Question", in_out='INPUT', socket_type='NodeSocketString').identifier
    tree.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    group_in, node = tree.nodes.new("NodeGroupInput"), tree.nodes.new(MATCH.idname)
    node.inputs["Items"].default_value = ", ".join(f"sound {i}" for i in range(100))
    tree.links.new(group_in.outputs["Question"], node.inputs["Question"])
    tree.links.new(store(tree, group_in.outputs["Geometry"], node.outputs["Probability"], "p0", 'FLOAT'),
                   tree.nodes.new("NodeGroupOutput").inputs["Geometry"])
    trigger(node)
    loud, quiet = user("Lost A", tree), user("Lost B", tree)  # A's context comes first: its first chunk is in flight
    loud.modifiers["OpenAct"][question], quiet.modifiers["OpenAct"][question] = "Is it loud?", "Is it quiet?"
    start = len(records())
    TAP.delay = 2.0
    TAP.started.clear()
    try:
        engine.run_pass()  # 4 chunks for each object's words
        TAP.started.wait(10)
        bpy.data.objects.remove(quiet)
        engine.run_pass()
        left = queued(node.uid)
    finally:
        TAP.delay = 0.0
    settle()
    sent = Counter(text for text, _debias in questions(start))
    about = [sum(count for text, count in sent.items() if text.endswith(f"\n{words}"))
             for words in ("Is it loud?", "Is it quiet?")]
    check("a context gone mid-fill (one of the node's two objects deleted): the next pass drops the chunks queued for "
          "its words, and none is sent; the chunk in flight completes, and each of the other's 100 cells is asked once",
          not left and about == [100, 0] and max(sent.values(), default=0) == 1 and whole(node, 100),
          (len(left), about, max(sent.values(), default=0)))


def phase_nested():
    """A node read as words from inside a nested group, in a tree no modifier uses: the evaluator plans it in a context
    find_uses() doesn't list, which the pass keeps (§5.3). Nothing evaluates either node, so Ask Again asks them, once
    (§6.0): the reader once the Pick it waits for has its answer. The scene settles: only the 2 s safety passes run,
    nothing is planned again and no generation is bumped, and the reader has its answer."""
    inner = bpy.data.node_groups.new("Nested inner", 'GeometryNodeTree')
    inner.interface.new_socket("Item", in_out='OUTPUT', socket_type='NodeSocketString')
    pick = inner.nodes.new(PICK.idname)
    pick.inputs["Prompt"].default_value = "something to sit on"
    pick.inputs["Items"].default_value = "chair, lamp, rug"
    inner.links.new(pick.outputs["Item"], inner.nodes.new("NodeGroupOutput").inputs["Item"])
    outer = bpy.data.node_groups.new("Nested outer", 'GeometryNodeTree')  # used by no modifier
    group = outer.nodes.new("GeometryNodeGroup")
    group.node_tree = inner
    reader = outer.nodes.new(ASK.idname)
    reader.inputs["Question"].default_value = "Can a person sit on this?"
    outer.links.new(group.outputs["Item"], reader.inputs["Context"])
    engine.run_pass()
    engine.reask(pick)
    engine.reask(reader)
    answered = drive(engine, lambda: bool(S.load(reader).entries), 30)
    drive(engine, lambda: False, 1.0)  # the passes the answers schedule
    passes, run_pass = [], engine.run_pass
    plans, generation = engine._plans_made, engine.GENERATIONS.get(pick.uid, 0)

    def counted_pass(everything=True):
        passes.append(everything)  # the safety pass visits every node; a scheduled one passes False
        return run_pass(everything)

    engine.run_pass = counted_pass
    try:
        drive(engine, lambda: False, 4.0)
    finally:
        engine.run_pass = run_pass
    got = (passes, engine._plans_made - plans, engine.GENERATIONS.get(pick.uid, 0) - generation)
    check("nested reader: a Pick read as words from a nested group of a tree no modifier uses is answered after Ask "
          "Again, and so is its reader, once the Pick is; then the scene settles: in 4 s only the 2 s safety passes, "
          "nothing planned again and no generation bumped", answered and 1 <= len(passes) <= 3 and all(passes)
          and got[1:] == (0, 0), (answered, got))


# ------------------------------------------------------------------------------ the plan cache

def plain_tree(name, idname, **words):
    """A tree with one OpenAct node whose words are typed on it; its geometry passes through."""
    tree = bpy.data.node_groups.new(name, 'GeometryNodeTree')
    tree.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    tree.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    node = tree.nodes.new(idname)
    for word, value in words.items():
        node.inputs[word].default_value = value
    tree.links.new(tree.nodes.new("NodeGroupInput").outputs["Geometry"],
                   tree.nodes.new("NodeGroupOutput").inputs["Geometry"])
    return tree, node


def fake(cell):
    """A made-up answer as pipeline.parse() gives it: yes/no 0.7, else the first option the most likely."""
    if cell.question["type"] == "noul":
        return {"p": 0.7, "model": "fake"}
    count = len(cell.question["criteria"])
    return {"probabilities": [0.5] + [0.5 / (count - 1)] * (count - 1), "model": "fake"}


def phase_timing():
    """§3.2: at most 5 ms per unchanged node, and 50 ms per idle pass with a 4096-cell Rules node used by 16 contexts
    (planned once each, then only fingerprinted), beside an Ask and a 20-item Match node used once each, all
    triggered, so each context reports its words."""
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    for group in list(bpy.data.node_groups):
        bpy.data.node_groups.remove(group)
    engine.run_pass()
    path = os.path.join(TMP, "factors_4096.csv")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("factor,kind,level,value,when\n")
        for factor in ("warmth", "damp", "height"):
            handle.writelines(f"{factor},category,{factor} {v},{v},The {factor} is at step {v} of 16.\n" for v in range(16))
    tree, rules = plain_tree("Timing", RULES.idname, Question="What happens to the plant here?", Factors=path,
                             Outcomes="grows; stays; shrinks")
    trigger(rules)
    for i in range(16):
        user(f"Timing {i}", tree)
    small = []
    for name, idname, words in (("Timing ask", ASK.idname, {"Question": "Is it raining?", "Context": "A grey sky."}),
                                ("Timing match", MATCH.idname, {"Question": "Is it round?",
                                                                "Items": ", ".join(f"thing {i}" for i in range(20))})):
        small_tree, node = plain_tree(name, idname, **words)
        trigger(node)
        user(name, small_tree)
        small.append(node)
    asked, submit = [], engine._submit
    engine._submit = lambda job, missing: asked.append((job.uid, missing))
    try:
        engine.run_pass()
        for _uid, missing in asked:
            for cell in missing:
                pipeline.remember(cell.key, fake(cell))
        engine.run_pass()
        entries = S.load(rules).entries
        check("plan cache: the 4096-cell table of the 16 contexts is asked once and stored whole",
              [len(missing) for uid, missing in asked if uid == rules.uid] == [4096] and len(entries) == 1
              and entries[0].rows.src == "m" * 4096 and all(S.load(node).entries for node in small), len(asked))
        times, process, plans, count = {}, engine._process, engine._plans_made, len(asked)

        def timed(tree_, node_, contexts):
            started = time.perf_counter()
            try:
                return process(tree_, node_, contexts)
            finally:
                times.setdefault(node_.uid, []).append((time.perf_counter() - started) * 1000)

        engine._process, passes = timed, []
        try:
            for _ in range(5):
                started = time.perf_counter()
                engine.run_pass()
                passes.append((time.perf_counter() - started) * 1000)
        finally:
            engine._process = process
    finally:
        engine._submit = submit
    # the fastest run of each: the one least slowed by other work on the machine (§18.2)
    node_ms, rules_ms, pass_ms = max(min(times[node.uid]) for node in small), min(times[rules.uid]), min(passes)
    print(f"TIMING plan cache: unchanged node {node_ms:.1f} ms (budget 5), idle pass {pass_ms:.1f} ms (budget 50) with "
          f"a 4096-cell Rules node used by 16 contexts (that node {rules_ms:.1f} ms)", flush=True)
    check("plan cache: idle passes plan nothing again and ask nothing", engine._plans_made == plans
          and len(asked) == count)
    for what, ms, budget in (("unchanged node", node_ms, 5), ("idle pass", pass_ms, 50)):
        if ms > budget:
            print(f"WARN plan cache: {what} took {ms:.1f} ms, above its {budget} ms budget", flush=True)
        check(f"plan cache: {what} within 2x its {budget} ms budget (§18.2)", ms <= 2 * budget, f"{ms:.1f} ms")


def main():
    print("BLENDER", bpy.app.version_string, "fake model server on port", PORT, flush=True)
    try:
        phase_token()
        phase_refused()
        big = phase_chunks()
        phase_cache(big)
        phase_priorities()
        phase_cancel()
        phase_isolation()
        phase_pauses()
        phase_stale()
        phase_failed()
        phase_idle()
        phase_validity()
        phase_cold_load()
        phase_reregister()
        phase_gone()
        phase_lost()
        phase_nested()
        phase_token_audit()
        phase_timing()
    except Exception:
        check("no unexpected error", False, traceback.format_exc())
    finally:
        server.stop()
    log = os.path.join(TMP, "state", "server.log")
    if FAILURES and os.path.isfile(log):
        with open(log, encoding="utf-8", errors="replace") as handle:
            print("--- the last fake server's log:\n" + handle.read()[-3000:].rstrip(), flush=True)
    if not FAILURES:
        shutil.rmtree(TMP, ignore_errors=True)
    finish()


main()
