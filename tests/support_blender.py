"""Templates, diagnostics, the request log, alerts and server sharing (§7.10, §10.5, §13.5, §16.3, §17.3-§17.6), in a
headless Blender, against the fake model server the add-on starts itself (OPENACT_BACKEND=fake) on a free port, with
--record and a --fake-script whose "delay" stalls one question:
- the manifest declares the clipboard permission, and `blender --command extension validate` passes (and refuses a
  permission that ends with a period);
- the templates parse with their sockets' schemas without a problem; New CSV from Template never overwrites (it
  creates name 2.csv, or the next free number), fills only an input that is empty and unlinked (its Import CSV
  companion too), refuses a node without templates and an unsaved .blend, and the Rules templates plan and fill;
- the diagnostics text has versions and counters but no prompt or CSV text, and cuts problems to 200 characters;
- the JSONL request log appears only while log_requests is on;
- the Unsure Below preference: its defaults, and it moves the status's unsure count, the sidebar's icons and the
  exported unsure column;
- drawing: Copy Diagnostics in the Model box, the template button in the Files panel, the new preferences, and the
  warning for a runtime folder outside the user folder;
- server sharing: the server writes its token file once it holds the port (mode 0600 on POSIX); a second Blender,
  which this suite starts, attaches through it without starting a server, and the one server answers it; a wrong
  token file gives §10.5's message, and a token file put right is tried at the next health probe;
- crash-loop protection, time-scaled: 3 unexpected exits within 10 minutes pause auto-start until Start Model, and
  exits further apart don't;
- the stalled-fill alert, time-scaled, while the scripted question stalls; the queued chunk stays queued;
- the slow-server note once a faked ms_per_row stays above 3 x 21 ms for a minute, time-scaled.

Run:  blender -b --factory-startup --python tests/support_blender.py
(The second Blender runs this file with -- attach <port>.)
"""

import json
import os
import platform
import re
import secrets
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
import tomllib
import traceback
from types import SimpleNamespace

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ATTACH = ARGS[:1] == ["attach"]  # the second Blender of the sharing check
os.environ["OPENACT_BACKEND"] = "fake"  # openact_server.py --backend fake: no torch, GPU or runtime
os.environ["OPENACT_SERVER_PYTHON"] = sys.executable  # Blender's own Python runs it
if not ATTACH:  # the token files and logs/, which the second Blender inherits
    os.environ["OPENACT_STATE_DIR"] = os.path.join(tempfile.mkdtemp(prefix="openact_support_"), "state")
STATE_DIR = os.environ["OPENACT_STATE_DIR"]
TMP = os.path.dirname(STATE_DIR)

import bpy  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import FAILURES, ROOT, check, drive, finish  # noqa: E402
from openact import csvfmt, engine, nodes, ops, pipeline, server, trust, ui, watch  # noqa: E402
from openact import prefs as P  # noqa: E402
from openact.specs.rules import FACTORS_HINT  # noqa: E402

RECORD, SCRIPT = os.path.join(TMP, "record.jsonl"), os.path.join(TMP, "script.json")
STALL_DELAY = 8.0  # seconds the fake model takes over "Is it stalled?"
MATCH, RULES, CLASSIFY, ASK = ("GeometryNodeOpenAct" + name for name in ("Match", "Rules", "Classify", "Ask"))
ICONS = set(bpy.types.UILayout.bl_rna.functions["label"].parameters["icon"].enum_items.keys())
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # Windows: no console for the Blenders this suite starts


def free_port():
    """A free local port, never 18765 or 18769: the user's own OpenAct session and another job use those."""
    while True:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        if port not in (18765, 18769):
            return port


PORT = int(ARGS[1]) if ATTACH else free_port()
P._Defaults.port, P._Defaults.runtime_dir, P._Defaults.auto_start = PORT, os.path.join(TMP, "runtime"), True
server.STATE.port = PORT  # before anything talks to a server
if not ATTACH:
    with open(SCRIPT, "w", encoding="utf-8") as handle:
        json.dump([{"question": "Is it stalled?", "answer": "yes", "delay": STALL_DELAY}], handle)
    server.FAKE_ARGS[:] = ["--record", RECORD, "--fake-script", SCRIPT]
for module in (ops, nodes, ui, trust, watch):
    module.register()
engine.register()  # the server status listener; timers don't run under --python, so the test ticks


# ------------------------------------------------------------------------------------------ helpers

def new_tree(name):
    tree = bpy.data.node_groups.new(name, 'GeometryNodeTree')
    tree.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    tree.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    tree.nodes.new("NodeGroupOutput")  # else Blender warns about the modifiers using the tree
    return tree


def used(name, idname, **words):
    """A node of this type, with these String inputs typed, in a new tree that an object's modifier uses."""
    tree = new_tree(name)
    node = tree.nodes.new(idname)
    for word, value in words.items():
        node.inputs[word].default_value = value
    obj = bpy.data.objects.new(name, bpy.data.meshes.new(name))
    bpy.context.scene.collection.objects.link(obj)
    obj.modifiers.new("OpenAct", 'NODES').node_group = tree
    return node


def status(node):
    return engine.STATUS.get(node.uid)


def line(node):
    """The node's status line: (state, text)."""
    found = status(node)
    return (found.state, found.text) if found else ("", "")


def complete(node):
    return line(node)[0] == "complete"


def settle(done, timeout=60):
    """Tick the add-on until done() holds and no cell is asked any more."""
    return drive(engine, lambda: done() and not pipeline.PENDING, timeout)


def records():
    """The fake server's --record log: every /v1/systemone request, with its status and answers."""
    try:
        with open(RECORD, encoding="utf-8") as handle:
            return [json.loads(text) for text in handle if text.strip()]
    except OSError:
        return []


def asked(text):
    """The status of each request whose questions include one with this text in its instructions."""
    return [record["status"] for record in records() if record["request"] and any(
        text in question["instructions"] for question in record["request"]["questions"].values())]


def health():
    return server.http("GET", "/health", timeout=2.0)[1]


def listening(port):
    try:
        return server.http("GET", "/health", port=port, timeout=1.0)[0] == 200
    except OSError:
        return False


def kill_server():
    """End this Blender's server as a crash would, with an exit code other than 0; returns its pid."""
    process = server.STATE.process
    process.kill()
    process.wait(10)
    return process.pid


def read(path):
    with open(path, "rb") as handle:
        return handle.read()


class FakeLayout:
    """Stands in for the UILayout a headless Blender has none of: records labels, properties and buttons, and, as
    Blender does, refuses unknown icons, operators, operator properties and layout attributes."""

    __slots__ = ("log", "alert", "active", "enabled", "use_property_split", "operator_context")

    def __init__(self, log):
        self.log = log

    def _child(self, *_args, **_kwargs):
        return FakeLayout(self.log)

    column = row = box = split = _child

    def label(self, text="", icon='NONE'):
        assert icon in ICONS, icon
        self.log.append(text)

    def prop(self, data, name, **_kwargs):
        assert hasattr(data, name), name
        self.log.append(("prop", name))

    def progress(self, factor=0.0, type='BAR', text=""):  # noqa: A002 - UILayout's own keyword
        self.log.append(text)

    def operator(self, idname, text=None, icon='NONE', depress=False):
        assert icon in ICONS, icon
        module, name = idname.split(".")
        self.log.append(("op", idname, text))
        return FakeProperties(set(getattr(getattr(bpy.ops, module), name).get_rna_type().properties.keys()), idname)


class FakeProperties:
    def __init__(self, names, idname):
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "idname", idname)

    def __setattr__(self, name, value):
        assert name in self.names, (self.idname, name)


def drawn(draw):
    log = []
    draw(FakeLayout(log))
    return log


def text_of(log):
    return " ".join(entry for entry in log if isinstance(entry, str))


# ------------------------------------------------------------------------------------------ manifest

def validate(folder):
    """blender --command extension validate on a folder: (exit code, output)."""
    result = subprocess.run([bpy.app.binary_path, "-t", "4", "--factory-startup", "--command", "extension", "validate",
                             folder], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300,
                            stdin=subprocess.DEVNULL, creationflags=NO_WINDOW)
    return result.returncode, (result.stdout + result.stderr).strip()


def phase_manifest():
    manifest = os.path.join(ROOT, "openact", "blender_manifest.toml")
    with open(manifest, "rb") as handle:
        permissions = tomllib.load(handle).get("permissions", {})
    check("manifest: the clipboard permission of §13.5; every permission under 64 characters, with no period at the end",
          permissions.get("clipboard") == "Copy a diagnostics report for bug reports"
          and all(len(text) < 64 and not text.endswith(".") for text in permissions.values()), permissions)
    code, output = validate(os.path.join(ROOT, "openact"))
    check("manifest: blender --command extension validate passes", code == 0 and "Success" in output, output[-1500:])
    broken = os.path.join(TMP, "broken_manifest")
    os.makedirs(broken)
    with open(manifest, encoding="utf-8") as handle:
        text = handle.read().replace("for bug reports\"", "for bug reports.\"")
    with open(os.path.join(broken, "blender_manifest.toml"), "w", encoding="utf-8") as handle:
        handle.write(text)
    shutil.copy(os.path.join(ROOT, "openact", "__init__.py"), broken)
    code, output = validate(broken)
    check("manifest: ...which checks the permission strings (it refuses a clipboard permission ending in a period)",
          code != 0 and "clipboard" in output, output[-1500:])


# ----------------------------------------------------------------------------------------- templates

def template_problems(name, schema, parse):
    """The messages of every problem a template has, read as the socket that takes it reads it."""
    doc = csvfmt.read_csv(os.path.join(ops.TEMPLATE_DIR, name), schema=schema)
    try:
        found = parse(doc)
    except csvfmt.CsvError as error:
        return [problem.message for problem in error.problems]
    return [problem.message for problem in (*doc.problems, *getattr(found, "problems", ()))]


def run_template(node):
    """New CSV from Template on the node, as its button runs it: {'FINISHED'}, or the error it reported."""
    try:
        return bpy.ops.openact.new_csv_from_template('EXEC_DEFAULT', uid=node.uid)
    except RuntimeError as error:  # how an operator's ERROR report reaches a Python caller
        return str(error).strip()


def phase_templates():
    got = {
        "life_factors.csv as Factors": template_problems("life_factors.csv", "factors", csvfmt.parse_factors),
        "life_outcomes.csv as Rules Outcomes": template_problems(
            "life_outcomes.csv", "answer_set", lambda doc: csvfmt.parse_answer_set(doc, "one_of", "rules")),
        "labels.csv as Classify Labels": template_problems(
            "labels.csv", "answer_set", lambda doc: csvfmt.parse_answer_set(doc, "one_of", "classify")),
        **{f"items.csv as {spec} Items": template_problems(
            "items.csv", "answer_set", lambda doc, spec=spec: csvfmt.parse_answer_set(doc, "items", spec))
           for spec in ("pick", "match", "classify")},
        "weather_dials.csv as Settings": template_problems("weather_dials.csv", "settings", csvfmt.parse_settings),
    }
    files = {file for pairs in ops.TEMPLATES.values() for _name, file in pairs}
    check("templates: the five files of §7.10, each a node's starter and parsed with its socket's schema without a "
          "problem", not any(got.values()) and files == set(os.listdir(ops.TEMPLATE_DIR)) == {
              "life_factors.csv", "life_outcomes.csv", "labels.csv", "items.csv", "weather_dials.csv"}, got)

    rules = used("Templates", RULES)
    engine.run_pass()  # the trust rule sees this file created in this session, before it is saved
    got = run_template(rules)
    check("templates: an unsaved .blend is refused, and its input is left empty",
          "Save the .blend first" in str(got) and rules.inputs["Factors"].default_value == "", got)
    folder = os.path.join(TMP, "project")
    os.makedirs(folder)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(folder, "scene.blend"))
    got = run_template(rules)
    shipped = all(read(os.path.join(folder, name)) == read(os.path.join(ops.TEMPLATE_DIR, name))
                  for name in ("life_factors.csv", "life_outcomes.csv"))
    check("templates: a Rules node gets life_factors.csv and life_outcomes.csv next to the .blend, as shipped, and its "
          "empty, unlinked Factors and Outcomes read them", got == {'FINISHED'} and shipped
          and rules.inputs["Factors"].default_value == "//life_factors.csv"
          and rules.inputs["Outcomes"].default_value == "//life_outcomes.csv",
          (got, rules.inputs["Factors"].default_value, rules.inputs["Outcomes"].default_value))

    with open(os.path.join(folder, "life_factors 2.csv"), "w", encoding="utf-8") as handle:
        handle.write("factor,kind,level\n")  # the artist's own file
    before = {name: read(os.path.join(folder, name)) for name in os.listdir(folder)}
    typed = used("Templates typed", RULES, Factors="//mine.csv")
    typed.id_data.links.new(typed.id_data.nodes.new("FunctionNodeInputString").outputs[0], typed.inputs["Outcomes"])
    got = run_template(typed)
    new = set(os.listdir(folder)) - set(before)
    kept = all(read(os.path.join(folder, name)) == data for name, data in before.items())
    check("templates: a copy never overwrites: 'name 2.csv', or the next free number, and every file there before is "
          "unchanged", got == {'FINISHED'} and new == {"life_factors 3.csv", "life_outcomes 2.csv"} and kept,
          (got, sorted(new)))
    check("templates: a typed input and a linked one keep what they have",
          typed.inputs["Factors"].default_value == "//mine.csv" and typed.inputs["Outcomes"].default_value == "",
          (typed.inputs["Factors"].default_value, typed.inputs["Outcomes"].default_value))

    classify = used("Templates classify", CLASSIFY)
    classify.id_data.links.new(classify.id_data.nodes.new("GeometryNodeImportCSV").outputs[0],
                               classify.inputs["Items CSV"])
    got = run_template(classify)
    check("templates: a Classify node gets items.csv and labels.csv; its empty Labels reads its copy, and Items, whose "
          "Import CSV companion is linked, stays empty", got == {'FINISHED'}
          and {"items.csv", "labels.csv"} <= set(os.listdir(folder))
          and classify.inputs["Labels"].default_value == "//labels.csv" and classify.inputs["Items"].default_value == "",
          (got, classify.inputs["Labels"].default_value, classify.inputs["Items"].default_value))

    ask = used("Templates ask", ASK)
    listing = set(os.listdir(folder))
    got = run_template(ask)
    check("templates: a node type without a starter CSV (Ask) is refused, and nothing is written",
          "No starter CSV" in str(got) and set(os.listdir(folder)) == listing, got)

    engine.schedule(immediate=True)
    done = settle(lambda: complete(rules), 90)
    factors = [socket.name for socket in rules.inputs if socket.name in ("state", "neighbours")]
    check("templates: the Rules node plans the Game of Life templates, and its 18 cells are answered",
          done and status(rules).rows == 18 and factors == ["state", "neighbours"], (line(rules), factors))
    return rules, classify, ask


# ---------------------------------------------------------------------------------------- diagnostics

def phase_diagnostics():
    import openact

    folder = os.path.dirname(bpy.data.filepath)
    with open(os.path.join(folder, "savanna.csv"), "w", encoding="utf-8") as handle:
        handle.write("answer,when\nokapi-sentinel,a quokka-sentinel of the forest\nzebra-sentinel,a striped grazer\n")
    node = used("Diagnosed", MATCH, Question="Is the aardvark-sentinel awake?", Context="dusk-sentinel",
                Items="//savanna.csv")
    missing = used("Missing", MATCH, Question="Is it there?", Items="//" + "m" * 170 + ".csv")
    engine.schedule(immediate=True)
    done = settle(lambda: complete(node) and line(missing)[0] == "error", 60)
    text = ops.diagnostics()
    version = ".".join(map(str, openact.bl_info["version"]))
    parts = [f"OpenAct {version}, Blender {bpy.app.version_string}, {platform.platform()}", "Backend: fake, model fake",
             f"Runtime: {P.runtime_root()}", f"Server: ready on port {PORT}", "Health: status ready, requests "]
    check("diagnostics: OpenAct, Blender and OS versions, the backend and model, the runtime, the server's status and "
          "its /health counters", done and all(part in text for part in parts)
          and all(re.search(rf"\b{key} \d", text) for key in ("requests", "questions", "rows_scored", "cache_hits")),
          text)
    check("diagnostics: the queue, a count per node type, and per node its rows per table, materialised entries and "
          "inner-group nodes", "Queue: 0/0/0 chunks (P0/P1/P2), 0 cells in flight, 0 pending" in text
          and re.search(r"^Nodes: .*\b\d+ Match\b", text, re.M) is not None and re.search(
              r"^  Match: complete; rows per table 2; 1 of 1 stored entries materialised; \d+ nodes in the inner "
              r"group$", text, re.M) is not None, text)
    cut = [row.strip() for row in text.splitlines() if "Can't find" in row and "mmm" in row]
    check("diagnostics: the problems, each cut to 200 characters", len(cut) == 1 and len(cut[0]) == 200, cut)
    leaked = [word for word in ("aardvark", "dusk-sentinel", "okapi", "quokka", "zebra") if word in text]
    check("diagnostics: no prompt, context, item name or other CSV text", not leaked, leaked)
    reports, wm = [], SimpleNamespace(clipboard="")  # a stand-in: the user's own clipboard is left alone
    got = ops.OPENACT_OT_copy_diagnostics.execute(SimpleNamespace(report=lambda _kind, message: reports.append(message)),
                                                  SimpleNamespace(window_manager=wm))
    check("diagnostics: Copy Diagnostics puts the report on the clipboard", got == {'FINISHED'}
          and wm.clipboard.startswith(f"OpenAct {version}, ") and "\nProblems: " in wm.clipboard, (got, reports))


# ---------------------------------------------------------------------------------------- request log

def phase_log():
    logs = os.path.join(STATE_DIR, "logs")

    def logged():
        try:
            with open(os.path.join(logs, time.strftime("requests-%Y%m%d.jsonl")), encoding="utf-8") as handle:
                return [json.loads(text) for text in handle if text.strip()]
        except OSError:
            return []

    off = used("Unlogged", MATCH, Question="Is it logged with the preference off?", Items="ledger, diary")
    done = settle(lambda: complete(off))
    check("log: with log_requests off, its default, requests are answered and no log is written",
          done and P._Defaults.log_requests is False and not os.path.exists(logs), os.path.exists(logs))
    P._Defaults.log_requests = True
    server.refresh()  # publishes the preference to the worker, as tick() does every 5 s
    try:
        on = used("Logged", MATCH, Question="Is it logged with the preference on?", Items="ledger, diary")
        done = settle(lambda: complete(on))
        got = logged()
    finally:
        P._Defaults.log_requests = False
        server.refresh()
    fields = {"t", "status", "state", "questions", "answers", "ms", "rows"}
    good = done and got and all(
        set(record) == fields and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d", record["t"]) and record["status"] == 200
        and set(record["answers"]) == set(record["questions"]) and record["rows"] == 2 * len(record["questions"])
        and record["ms"] >= 0 for record in got)
    check("log: while it is on, one JSON line per request in <state>/logs/requests-YYYYMMDD.jsonl, with t, state, "
          "questions, answers, ms and rows (and the status)", good and any(
              "preference on" in question["instructions"] for record in got for question in record["questions"].values()),
          got)
    later = used("Unlogged again", MATCH, Question="Is it logged once the preference is off again?", Items="ledger")
    done = settle(lambda: complete(later))
    check("log: once it is off again, nothing more is written", done and len(logged()) == len(got), len(logged()))


# ---------------------------------------------------------------------------------------- unsure below

def phase_unsure():
    try:
        for cls in P.classes:
            bpy.utils.register_class(cls)
        rna = P.OpenActPreferences.bl_rna.properties
        defaults = (round(rna["unsure_below"].default, 6), rna["unsure_below"].hard_min, rna["unsure_below"].hard_max,
                    rna["log_requests"].default)
    except Exception as error:  # noqa: BLE001 - reported by the check
        defaults = repr(error)
    finally:
        for cls in reversed(P.classes):
            if getattr(cls, "is_registered", False):
                bpy.utils.unregister_class(cls)
    check("preferences: Unsure Below defaults to 0.30 within 0-1, Log Requests to off, as without the add-on enabled",
          defaults == (0.3, 0.0, 1.0, False) and P._Defaults.unsure_below == P.UNSURE_BELOW == 0.30, defaults)

    node = used("Unsure", MATCH, Question="Is it hazy?", Items="fog, mist, haze, smoke, steam, cloud, dust, spray")
    done = settle(lambda: complete(node))
    spec, (_label, job, entry) = engine.SPECS[MATCH], status(node).uses[0]
    confidences = entry.rows.v["c"]
    ordered = sorted(confidences)
    threshold = (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2
    expected = sum(c < threshold for c in confidences)
    P._Defaults.unsure_below = threshold
    try:
        engine.run_pass()
        got = status(node)
        icons = [icon for _name, _answer, _c, icon, _active in ui.table_page(ui.table_rows(spec, job, entry))[0]]
        header, rows = ops.merged_rows(spec, [(job, entry)])
        exported = [row[header.index("unsure")] for row in rows]
    finally:
        P._Defaults.unsure_below = P.UNSURE_BELOW
    engine.run_pass()
    check("unsure below: the preference sets the status's unsure count and line, the sidebar's unsure icons and the "
          "exported unsure column", done and 0 < expected < len(confidences) and got.unsure == expected
          and got.text.endswith(f" {expected} unsure") and icons.count('QUESTION') == expected
          and exported.count("1") == expected, (expected, got.unsure, got.text, icons, exported))
    check("unsure below: back at 0.30, the count follows it", status(node).unsure == sum(c < 0.30 for c in confidences),
          status(node).unsure)


# ------------------------------------------------------------------------------------------- drawing

def phase_draw(rules, classify, ask):
    log = drawn(lambda layout: ui.draw_server(layout))
    check("draw: the Model box has Copy Diagnostics, next to Open Log",
          ("op", "openact.copy_diagnostics", "") in log and ("op", "openact.open_log", "") in log, log)
    buttons = {}
    for node in (rules, classify, ask):
        buttons[node.name] = [entry[2] for entry in drawn(lambda layout, node=node: ui.draw_files(layout, node))
                              if isinstance(entry, tuple) and entry[1] == "openact.new_csv_from_template"]
    check("draw: the Files panel offers the templates: 'New Rules Template' on Rules, as its hint says, 'New CSV from "
          "Template' on Classify, and nothing on Ask", "New Rules Template" in FACTORS_HINT and buttons == {
              rules.name: ["New Rules Template"], classify.name: ["New CSV from Template"], ask.name: []}, buttons)
    saved, warning = P._Defaults.runtime_dir, "This folder is outside your user folder"
    try:
        P._Defaults.runtime_dir = os.path.join(os.path.expanduser("~"), "OpenAct runtime")
        inside = drawn(lambda layout: ui.draw_preferences(P.get_prefs(), bpy.context, layout))
        P._Defaults.runtime_dir = os.path.abspath(os.sep + "OpenAct runtime")
        outside = drawn(lambda layout: ui.draw_preferences(P.get_prefs(), bpy.context, layout))
    finally:
        P._Defaults.runtime_dir = saved
    check("draw: the preferences show Unsure Below and Log Requests, and warn when the runtime folder lies outside the "
          "user folder, as the token file's readers then depend on its permissions",
          {("prop", "unsure_below"), ("prop", "log_requests")} <= set(entry for entry in inside if isinstance(entry, tuple))
          and warning not in text_of(inside) and warning in text_of(outside), (text_of(inside), text_of(outside)))


# ------------------------------------------------------------------------------------------- sharing

def attach():
    """The second Blender: it finds the suite's server on PORT, attaches through the token file, asks one question and
    reports what it saw. It never stops the server: that would stop it for the first Blender too."""
    started = server.start()
    node = used("Second Blender", MATCH, Question="Is it asked by the second Blender?", Items="brick, feather")
    done = settle(lambda: complete(node), 90)
    process = server.STATE.process
    print("ATTACHED " + json.dumps({
        "started": started, "process": None if process is None else process.pid, "pid": server.STATE.health.get("pid"),
        "status": list(server.status()), "complete": done,
        "token": len(server.STATE.token) == 64 and server.STATE.token == server._read_token(PORT)}), flush=True)


def phase_sharing():
    ready = server.is_ready() or (server.start() and drive(engine, server.is_ready, 30))
    token_file = os.path.join(STATE_DIR, f"server-{PORT}.token")
    try:
        written, mode = read(token_file).decode("ascii"), stat.S_IMODE(os.stat(token_file).st_mode)
    except OSError as error:
        written, mode = repr(error), None
    argv = " ".join(server.STATE.process.args) if server.STATE.process else ""
    check("sharing: the server this Blender started writes its launch token to <state>/server-<port>.token once it "
          "holds the port, mode 0600 on POSIX; argv names only the file",
          ready and written == server.STATE.token and (sys.platform == "win32" or mode == 0o600)
          and "--token-file" in argv and server.STATE.token not in argv, (written[:80], oct(mode or 0), argv))
    before, pid = health(), server.STATE.process.pid
    try:
        result = subprocess.run([bpy.app.binary_path, "-b", "--factory-startup", "-t", "4", "--python",
                                 os.path.abspath(__file__), "--", "attach", str(PORT)], capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=300, stdin=subprocess.DEVNULL,
                                creationflags=NO_WINDOW)
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired as error:  # run() killed it
        output = f"the second Blender timed out: {error}"
    report = next((json.loads(text[len("ATTACHED "):]) for text in output.splitlines() if text.startswith("ATTACHED ")),
                  None)
    after = health()
    check("sharing: a second Blender attaches through the token file: it starts no server of its own, and its status "
          "is ready, with this Blender's server (the same pid) and its token",
          report is not None and report["process"] is None and report["pid"] == pid == after["pid"]
          and report["status"][0] == "ready" and report["token"], report or output[-3000:])
    statuses = asked("asked by the second Blender")
    check("sharing: ...and the one server answers its requests, with no second model loaded",
          report is not None and report["complete"] and statuses and set(statuses) == {200}
          and after["requests"] > before["requests"] and after["load_seconds"] == before["load_seconds"]
          and server.STATE.process.poll() is None, (report, statuses, before, after))


def phase_wrong_token():
    port, token = free_port(), secrets.token_hex(32)
    foreign = subprocess.Popen([sys.executable, server.SERVER_SCRIPT, "--port", str(port), "--backend", "fake",
                                "--parent-pid", str(os.getpid())], env=dict(server._child_env(), OPENACT_TOKEN=token),
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               creationflags=NO_WINDOW)
    token_file, message = os.path.join(STATE_DIR, f"server-{port}.token"), server.TOKEN_REFUSED.format(port)
    with open(token_file, "w", encoding="ascii") as handle:
        handle.write("0" * 64)  # another server's, or damaged
    server.stop()
    P._Defaults.port = port
    try:
        up = drive(engine, lambda: listening(port), 20)
        node = used("Wrong token", MATCH, Question="Is its token wrong?", Items="key, lock")
        engine.schedule(immediate=True)
        refused = drive(engine, lambda: line(node) == ("error", message), 20)
        check("sharing: a server whose token file holds another token refuses this Blender, and the status says so in "
              "§10.5's words", up and refused and server.status() == ("error", message)
              and message == f"A model server started elsewhere is running on port {port} without a readable token: "
                             "stop it or choose another port", (server.status(), line(node)))
        with open(token_file, "w", encoding="ascii") as handle:
            handle.write(token)
        server.refresh()  # as tick() does every 5 s
        done = settle(lambda: complete(node), 30)
        check("sharing: once the token file holds its token, the next health probe tries it, and the node is answered "
              "with no Ask Again", done, (server.status(), line(node)))
    finally:
        server.stop()  # the other server, whose token this Blender now has
        if foreign.poll() is None:
            foreign.kill()
        foreign.wait(10)
        P._Defaults.port = PORT
    check("sharing: this Blender's own server starts again on its port",
          server.start() and drive(engine, server.is_ready, 30), server.status())


# ---------------------------------------------------------------------------------------- crash loop

def restart_after_crash(node, question):
    """Kill the server as a crash would, then give the node new words, whose cells the pass asks, so auto-start starts
    the server again. Returns (the node answered, and the server runs as a new process)."""
    killed = kill_server()
    node.inputs["Question"].default_value = question
    engine.run_pass()  # the new words now: until then the status says complete, and "ready" is what it was
    done = settle(lambda: complete(node) and server.is_ready(), 60)
    return done, server.STATE.process is not None and server.STATE.process.pid != killed


def phase_crash_loop():
    window = server.CRASH_WINDOW
    server.CRASH_WINDOW = 30.0  # 10 minutes, time-scaled
    node = used("Crashes", MATCH, Question="Does it crash? 0", Items="glass, vase")
    try:
        settle(lambda: complete(node) and server.is_ready())
        got = [restart_after_crash(node, f"Does it crash? {crash}") for crash in (1, 2)]
        check("crash loop: after each of 2 unexpected exits, auto-start starts the server again and the node is "
              "answered", got == [(True, True)] * 2, got)
        kill_server()
        node.inputs["Question"].default_value = "Does it crash? 3"
        engine.run_pass()
        paused = drive(engine, lambda: server.status() == ("error", server.KEEPS_STOPPING), 30)
        drive(engine, lambda: False, 2.5)  # passes, and nothing starts
        box = text_of(drawn(lambda layout: ui.draw_server(layout)))
        check("crash loop: the 3rd unexpected exit within 10 minutes (time-scaled) pauses auto-start: the node and the "
              "Model box say 'The model server keeps stopping: see the log', and no server starts",
              paused and line(node) == ("error", server.KEEPS_STOPPING) and server.STATE.process is None
              and not listening(PORT) and server.KEEPS_STOPPING in box, (server.status(), line(node), box))
        got = bpy.ops.openact.server_start()
        done = settle(lambda: complete(node) and server.is_ready(), 60)
        check("crash loop: Start Model starts it again, and the cells that waited are answered",
              got == {'FINISHED'} and done, (got, server.status(), line(node)))
        spread = [restart_after_crash(node, "Does it crash? 4")]
        server.STATE.exits = [t - server.CRASH_WINDOW for t in server.STATE.exits]  # that exit is 10 minutes old
        spread += [restart_after_crash(node, f"Does it crash? {crash}") for crash in (5, 6)]
        check("crash loop: 3 exits further apart than 10 minutes don't pause it", spread == [(True, True)] * 3
              and server.status()[0] == "ready", (spread, server.status()))
    finally:
        server.CRASH_WINDOW = window


# -------------------------------------------------------------------------------------- stall, slow

def phase_stall():
    stall = engine.STALL
    engine.STALL = 1.5  # 60 s, time-scaled
    try:
        stalled = used("Stalls", MATCH, Question="Is it stalled?", Items="clock")
        stalled.balance_order = False  # one question and one scored row, which the fake model takes 8 s over
        engine.schedule(immediate=True)
        sent = drive(engine, lambda: pipeline.queue_counts()[1] == 1, 10)  # in flight
        waiting = used("Waits", MATCH, Question="Is it waiting behind?", Items="queue")
        engine.schedule(immediate=True)
        alert = ("warning", engine.STALLED.format(engine.STALL))
        shown = drive(engine, lambda: line(stalled) == alert == line(waiting), STALL_DELAY - 2)
        per_class, flight = pipeline.queue_counts()
        check("stall: with no chunk answered for 60 s (time-scaled) while the model is ready, the filling nodes show "
              "'No answers for 60 s: see the log' as a warning, and the queued chunk stays queued",
              sent and shown and status(stalled).icon == 'ERROR' and per_class == [1, 0, 0] and flight == 1
              and engine.STALLED.format(60.0) == "No answers for 60 s: see the log",
              (line(stalled), line(waiting), per_class, flight))
        done = settle(lambda: complete(stalled) and complete(waiting), 60)
        check("stall: once the stalled chunk comes back, the alert goes and both tables complete",
              done and asked("Is it stalled?") == [200] and asked("Is it waiting behind?") == [200],
              (line(stalled), line(waiting)))
    finally:
        engine.STALL = stall


def phase_slow():
    slow_for, real = engine.SLOW_FOR, server.STATE.health
    engine.SLOW_FOR = 1.0  # a minute, time-scaled

    def reading(ms_per_row, seconds):
        """Tick for this long with /health reporting this ms_per_row (no probe replaces it meanwhile)."""
        server.STATE.health = dict(real, ms_per_row=ms_per_row)
        engine._last_refresh = time.monotonic()
        drive(engine, lambda: False, seconds)
        return engine.slow_note()

    try:
        at_limit = reading(3 * 21.0, 1.5)
        early = reading(100.0, 0.4)
        late = reading(100.0, 1.0)
        box = text_of(drawn(lambda layout: ui.draw_server(layout)))
        gone = reading(20.0, 0.2)
    finally:
        engine.SLOW_FOR, server.STATE.health = slow_for, real
    check("slow server: at 3 x 21 ms per row, no note", engine.SLOW_MS == 63.0 and at_limit == "", at_limit)
    check("slow server: above it for a minute (time-scaled), and not before, the Model box notes 'Running slower than "
          "expected: is the GPU busy (rendering, another app)?'; it goes once the rate is back",
          early == "" and late == engine.SLOW and engine.SLOW in box and gone == "", (early, late, box, gone))


# -------------------------------------------------------------------------------------------- run

def main():
    print("BLENDER", bpy.app.version_string, "fake model server on port", PORT, flush=True)
    try:
        phase_manifest()
        rules, classify, ask = phase_templates()
        phase_diagnostics()
        phase_log()
        phase_unsure()
        phase_draw(rules, classify, ask)
        phase_sharing()
        phase_wrong_token()
        phase_crash_loop()
        phase_stall()
        phase_slow()
    except Exception:
        check("no unexpected error", False, traceback.format_exc())
    finally:
        if server.STATE.port == PORT:  # never another port: /shutdown would stop someone else's server
            server.stop()
    log = os.path.join(STATE_DIR, "server.log")
    if FAILURES and os.path.isfile(log):
        with open(log, encoding="utf-8", errors="replace") as handle:
            print("--- the last fake server's log:\n" + handle.read()[-3000:].rstrip(), flush=True)
    if not FAILURES:
        shutil.rmtree(TMP, ignore_errors=True)
    finish()


if ATTACH:
    try:
        attach()
    except Exception:
        traceback.print_exc()
    os._exit(0)  # without stopping the server: it is the first Blender's
main()
