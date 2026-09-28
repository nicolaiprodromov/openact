"""Templates, diagnostics, the request log, alerts and server sharing (§7.10, §10.5, §13.5, §16.3, §17.3-§17.6), in a
headless Blender, against the fake model server the add-on starts itself (OPENACT_BACKEND=fake) on a free port, with
--record and a --fake-script whose "delay" stalls one question:
- the manifest declares the clipboard permission, and `blender --command extension validate` passes (and refuses a
  permission that ends with a period);
- the templates parse with their sockets' schemas without a problem; New CSV from Template never overwrites (it
  creates name 2.csv, or the next free number), fills only an input that is empty and unlinked (its Import CSV
  companion too), refuses a node without templates and an unsaved .blend, and the Rules templates plan and fill;
- the diagnostics text has versions and counters but no prompt or CSV text: a problem is copied as its node type and
  severity, never its text;
- the JSONL request log appears only while log_requests is on, which reaches the worker as it changes; its folder
  and file are created 0700 and 0600;
- the Unsure Below preference: its defaults, and it moves the status's unsure count, the sidebar's icons and the
  exported unsure column;
- drawing: Copy Diagnostics in the Model box, the template button in the Files panel (only for a node with such an
  input), the new preferences, and the warning for a token folder outside the user folder;
- server sharing: the server writes its token file once it holds the port (mode 0600 on POSIX), keeps the port to
  itself (a second server exits with the port-taken code, the file unchanged) and removes the file on a clean exit;
  a second Blender, which this suite starts, attaches through it without starting a server, the one server answers
  it, its Model box says the server is shared, and its Stop only detaches; a wrong token file gives §10.5's message,
  and a token file put right is tried at the next health probe; a lost start race attaches to the winner at once,
  and neither it nor a port some other program holds counts as a crash; Detach lasts until Start Model, also while
  the other server loads and in a lost start race;
- the client's state follows from what can be observed: a health probe that times out changes nothing (a ready
  server stays ready, a Detach lasts); a chunk the worker took before a Detach never goes out without a token; the
  other Blender's server dying during its load leaves this Blender stopped within seconds, and auto-start starts its
  own; Stop Model during this Blender's own load ends stopped, not in an "exited" error;
- crash-loop protection, time-scaled: 3 unexpected exits within 10 minutes pause auto-start until Start Model, and
  exits further apart don't;
- the stalled-fill alert, time-scaled, while the scripted question stalls; the queued chunk stays queued;
- the slow-server note once a faked ms_per_row stays above 3 x 21 ms for a minute while cells are asked (never while
  idle), time-scaled;
- a profile saved with the Device preference's old CPU choice, which Blender reads as "", starts the server with
  --device auto.

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
import threading
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

HOME = os.path.expanduser("~")
ELSEWHERE = os.path.abspath(os.sep + "OpenAct elsewhere")  # a folder outside the user folder

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
    """This Blender's port's /health, {} when nothing answers."""
    try:
        return server.http("GET", "/health", timeout=2.0)[1]
    except OSError:
        return {}


def listening(port):
    try:
        return server.http("GET", "/health", port=port, timeout=1.0)[0] == 200
    except OSError:
        return False


def kill_server():
    """End this Blender's server as a crash would, with an exit code other than 0; returns its pid (None: none ran)."""
    process = server.STATE.process
    if process is None:
        return None
    process.kill()
    process.wait(10)
    return process.pid


def other_server(port, token, *flags):
    """A fake server as another Blender starts it, writing its token file: /health names no launching Blender of this
    process (no --parent-pid, so the suite ends it itself)."""
    return subprocess.Popen([sys.executable, server.SERVER_SCRIPT, "--port", str(port), "--backend", "fake",
                             "--token-file", os.path.join(STATE_DIR, f"server-{port}.token"), *flags],
                            env=dict(server._child_env(), OPENACT_TOKEN=token), stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=NO_WINDOW)


def ended(process):
    if process.poll() is None:
        process.kill()
    process.wait(10)


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
    doubled = used("Doubled", MATCH, Question="Is it there twice?", Items="okapi-sentinel, okapi-sentinel")
    missing = used("Missing", MATCH, Question="Is it there?", Items="//folder-sentinel/names.csv")
    with open(os.path.join(folder, "zebra_factors.csv"), "w", encoding="utf-8") as handle:
        handle.write("factor,kind,level,value\nzebrafactor,flag,quokkalevel,0\nzebrafactor,flag,otherlevel,1\n")
    with open(os.path.join(folder, "zebra_outcomes.csv"), "w", encoding="utf-8") as handle:
        handle.write("answer,when,if\nStays,It stays.,zebrafactor=otherlevel\nGoes,It goes.,zebrafactor=otherlevel\n")
    uncovered = used("Uncovered", RULES, Factors="//zebra_factors.csv", Outcomes="//zebra_outcomes.csv")
    engine.schedule(immediate=True)
    done = settle(lambda: complete(node) and line(missing)[0] == line(doubled)[0] == "error"
                  and "No outcome applies" in line(uncovered)[1], 60)
    quoted, pathed, named = line(doubled)[1], line(missing)[1], line(uncovered)[1]
    config = os.path.join(P._Defaults.runtime_dir, "venv", "pyvenv.cfg")  # names the venv's base Python
    os.makedirs(os.path.dirname(config), exist_ok=True)
    roaming = os.path.join(os.sep * 2 + "fileserver", "users$", "artist", "AppData", "Roaming", "uv", "python",
                           "cpython-3.12.11-windows-x86_64-none")  # uv's Pythons in a %APPDATA% redirected to a share
    with open(config, "w", encoding="utf-8") as handle:
        handle.write(f"home = {roaming}\n")
    runtime, venv_home = P._Defaults.runtime_dir, P.venv_home
    try:
        text = ops.diagnostics()
        P._Defaults.runtime_dir, P.venv_home = HOME, lambda _root: ""  # a runtime in the home folder itself
        in_home = ops.diagnostics()
    finally:
        P._Defaults.runtime_dir, P.venv_home = runtime, venv_home
        os.remove(config)
    version = ".".join(map(str, openact.bl_info["version"]))
    parts = [f"OpenAct {version}, Blender {bpy.app.version_string}, {platform.platform()}", "Backend: fake, model fake",
             "\nRuntime: ", f"; Python from cpython-3.12.11-windows-x86_64-none\nServer: ready on port {PORT}",
             "Health: status ready, requests "]
    check("diagnostics: OpenAct, Blender and OS versions, the backend and model, the runtime and the name of its venv's "
          "Python folder, the server's status and its /health counters", done and all(part in text for part in parts)
          and all(re.search(rf"\b{key} \d", text) for key in ("requests", "questions", "rows_scored", "cache_hits")),
          text)
    check("diagnostics: no server or account name: the Python only by its folder's name, and a runtime in the home "
          "folder itself as ~", "fileserver" not in text and "users$" not in text and "\nRuntime: ~\n" in in_home,
          (text, in_home))
    check("diagnostics: the queue, a count per node type, and per node its rows per table, materialised entries and "
          "inner-group nodes", "Queue: 0/0/0 chunks (P0/P1/P2), 0 cells in flight, 0 pending" in text
          and re.search(r"^Nodes: .*\b\d+ Match\b", text, re.M) is not None and re.search(
              r"^  Match: complete; rows per table 2; 1 of 1 stored entries materialised; \d+ nodes in the inner "
              r"group$", text, re.M) is not None, text)
    check("diagnostics: a problem is copied as its node type and severity, never its text",
          "'okapi-sentinel' appears twice" in quoted and "folder-sentinel" in pathed
          and "zebrafactor=quokkalevel" in named and re.search(r"^  Match error \(x2\)$", text, re.M)
          and re.search(r"^  Rules error( \(x\d+\))?$", text, re.M), (quoted, pathed, named, text))
    leaked = [word for word in ("aardvark", "dusk-sentinel", "okapi", "quokka", "zebra", "folder-sentinel",
                                "appears twice", "No outcome", HOME + os.sep) if word in text]
    check("diagnostics: no prompt, context, item name, factor or level name, folder, message or other CSV text",
          not leaked, leaked)
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

    # The preference as the add-on declares it, on a stand-in property group: its update reaches the worker at once
    annotation = P.OpenActPreferences.__annotations__["log_requests"]  # prefs postpones annotations: a string
    declared = eval(annotation, vars(P)) if isinstance(annotation, str) else annotation  # the add-on's own source
    group = type("OpenActLogProbe", (bpy.types.PropertyGroup,), {"__annotations__": {"log_requests": declared}})
    bpy.utils.register_class(group)
    bpy.types.WindowManager.openact_log_probe = bpy.props.PointerProperty(type=group)
    try:
        probe, seen = bpy.context.window_manager.openact_log_probe, []
        for value in (True, False):
            probe.log_requests = value
            seen.append(server.STATE.log_requests)
    finally:
        del bpy.types.WindowManager.openact_log_probe
        bpy.utils.unregister_class(group)
    check("log: switching Log Requests reaches the worker at once, not at the next refresh", seen == [True, False],
          seen)

    calls, makedirs, opener = [], os.makedirs, os.open

    def made(path, mode=0o777, exist_ok=False):
        calls.append(("folder", mode))
        return makedirs(path, mode, exist_ok)

    def opened(path, flags, mode=0o777, *args, **kwargs):
        calls.append(("file", bool(flags & os.O_APPEND), mode))
        return opener(path, flags, mode, *args, **kwargs)

    os.makedirs, os.open = made, opened
    try:
        server._log_request({"state": "s", "questions": {}}, 200, {}, 0.0, 0)
    finally:
        os.makedirs, os.open = makedirs, opener
    modes = [stat.S_IMODE(os.stat(path).st_mode) for path in (logs, os.path.join(
        logs, time.strftime("requests-%Y%m%d.jsonl")))]
    check("log: its folder is created 0700 and its file 0600, appended to: prompts and CSV text for this user only on "
          "POSIX", calls == [("folder", 0o700), ("file", True, 0o600)]
          and (sys.platform == "win32" or modes == [0o700, 0o600]), (calls, [oct(mode) for mode in modes]))


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
    check("draw: the Model box has Copy Diagnostics next to Open Log; with this Blender's own server, Stop Model and no "
          "shared line", ("op", "openact.copy_diagnostics", "") in log and ("op", "openact.open_log", "") in log
          and ("op", "openact.server_stop", None) in log and server.SHARED not in text_of(log), log)
    collection = used("Collection match", MATCH)
    collection.list_source = 'COLLECTION'  # no Items input, so no starter file to copy
    buttons = {}
    for node in (rules, classify, ask, collection):
        buttons[node.name] = [entry[2] for entry in drawn(lambda layout, node=node: ui.draw_files(layout, node))
                              if isinstance(entry, tuple) and entry[1] == "openact.new_csv_from_template"]
    check("draw: the Files panel offers New CSV from Template, the button the Rules hint names, on a node with an input "
          "that has a starter file: Rules and Classify, not Ask, nor a Match whose list comes from a collection",
          "(sidebar: New CSV from Template)" in FACTORS_HINT
          and ops.OPENACT_OT_new_csv_from_template.bl_label == "New CSV from Template" and buttons == {
              rules.name: [None], classify.name: [None], ask.name: [], collection.name: []}, buttons)

    # The warning follows the folder that holds the token file: OPENACT_STATE_DIR when set, else the runtime folder
    saved, warning, found = (os.environ.get("OPENACT_STATE_DIR"), P._Defaults.runtime_dir), \
        "The folder that holds the model server's token", {}
    try:
        for name, state, runtime in (("state inside, runtime outside", os.path.join(HOME, "OpenAct state"), ELSEWHERE),
                                     ("state outside, runtime inside", ELSEWHERE, os.path.join(HOME, "OpenAct")),
                                     ("runtime outside", None, ELSEWHERE),
                                     ("runtime inside", None, os.path.join(HOME, "OpenAct"))):
            if state is None:
                os.environ.pop("OPENACT_STATE_DIR", None)
            else:
                os.environ["OPENACT_STATE_DIR"] = state
            P._Defaults.runtime_dir = runtime
            log = drawn(lambda layout: ui.draw_preferences(P.get_prefs(), bpy.context, layout))
            found[name] = warning in text_of(log)
    finally:
        os.environ["OPENACT_STATE_DIR"], P._Defaults.runtime_dir = saved
    check("draw: the preferences show Unsure Below and Log Requests, and warn when the folder that holds the token lies "
          "outside the user folder", {("prop", "unsure_below"), ("prop", "log_requests")} <= set(
              entry for entry in log if isinstance(entry, tuple)) and found == {
              "state inside, runtime outside": False, "state outside, runtime inside": True, "runtime outside": True,
              "runtime inside": False}, found)


# ------------------------------------------------------------------------------------------- sharing

def attach():
    """The second Blender: it finds the suite's server on PORT, attaches through the token file, asks one question,
    looks at its Model box, presses Stop Model, which must only detach, and reports what it saw."""
    started = server.start()
    node = used("Second Blender", MATCH, Question="Is it asked by the second Blender?", Items="brick, feather")
    done = settle(lambda: complete(node), 90)
    process = server.STATE.process
    report = {
        "started": started, "process": None if process is None else process.pid, "pid": server.STATE.health.get("pid"),
        "status": list(server.status()), "complete": done,
        "token": len(server.STATE.token) == 64 and server.STATE.token == server._read_token(PORT)}
    box = drawn(lambda layout: ui.draw_server(layout))
    report.update(shared=server.SHARED in text_of(box), detach=("op", "openact.server_stop", "Detach") in box,
                  tooltip=ops.OPENACT_OT_server_stop.description(bpy.context, None))
    server.stop()
    report.update(stopped=list(server.status()), token_after=server.STATE.token, alive=listening(PORT))
    print("ATTACHED " + json.dumps(report), flush=True)


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
    pid = server.STATE.process.pid
    try:  # as when another Blender starts one at the same moment: the port is taken
        second = subprocess.run([sys.executable, server.SERVER_SCRIPT, "--port", str(PORT), "--backend", "fake",
                                 "--token-file", token_file], env=dict(server._child_env(),
                                                                       OPENACT_TOKEN=secrets.token_hex(32)),
                                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20,
                                stdin=subprocess.DEVNULL, creationflags=NO_WINDOW)
        code, output = second.returncode, second.stdout + second.stderr
    except subprocess.TimeoutExpired as error:  # it went on serving; run() killed it
        code, output = None, str(error)
    check("sharing: a second server on this port can't bind it: it exits with the port-taken code, and the token file "
          "and /health are still this server's", code == server.PORT_TAKEN and "cannot listen" in output
          and read(token_file).decode("ascii") == server.STATE.token and health().get("pid") == pid,
          (code, output[-500:]))
    before = health()
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
          report is not None and report["process"] is None and report["pid"] == pid == after.get("pid")
          and report["status"][0] == "ready" and report["token"], report or output[-3000:])
    statuses = asked("asked by the second Blender")
    check("sharing: ...and the one server answers its requests, with no second model loaded",
          report is not None and report["complete"] and statuses and set(statuses) == {200}
          and after.get("requests", 0) > before["requests"] and after.get("load_seconds") == before["load_seconds"], (
              report, statuses, before, after))
    check("sharing: its Model box says the server is shared and its Stop button reads Detach, whose tooltip says the "
          "server keeps running for the other Blender", report is not None and report["shared"] and report["detach"]
          and "another Blender started: it keeps running" in report["tooltip"], report)
    check("sharing: its Stop Model only detaches: its status is stopped and it forgets the token, while this Blender's "
          "server runs on", report is not None and report["stopped"] == ["stopped", ""] and report["token_after"] == ""
          and report["alive"] and server.STATE.process.poll() is None and health().get("pid") == pid, report)


def phase_wrong_token():
    port, token = free_port(), secrets.token_hex(32)
    foreign = subprocess.Popen([sys.executable, server.SERVER_SCRIPT, "--port", str(port), "--backend", "fake"],
                               env=dict(server._child_env(), OPENACT_TOKEN=token), stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=NO_WINDOW)
    token_file, message = os.path.join(STATE_DIR, f"server-{port}.token"), server.TOKEN_REFUSED.format(port)
    with open(token_file, "w", encoding="ascii") as handle:
        handle.write("0" * 64)  # another server's, or damaged
    own_file = os.path.join(STATE_DIR, f"server-{PORT}.token")
    kept = os.path.exists(own_file)
    server.stop()
    check("sharing: a server this Blender stops removes its token file as it exits", kept and not os.path.exists(
        own_file), (kept, os.path.exists(own_file)))
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
        server.stop()  # only detaches from the other server, which the suite ends
        ended(foreign)
        P._Defaults.port = PORT
    check("sharing: this Blender's own server starts again on its port",
          server.start() and drive(engine, server.is_ready, 30), server.status())


def racing_start():
    """server.start() whose /health check finds nothing listening: it checked a moment before the server of another
    Blender, started at the same moment, listened."""
    real, first = server.http, True

    def http(method, path, body=None, port=None, timeout=10.0):
        nonlocal first
        if first and method == "GET" and path == "/health":
            first = False
            raise ConnectionRefusedError("nothing listened yet")
        return real(method, path, body, port, timeout)

    server.http = http
    try:
        return server.start()
    finally:
        server.http = real


def phase_lost_race():
    """§10.5: this Blender loses a start race for its port. Against another Blender's server (loading for 2 s) it
    attaches at once, with no health probe; against a program that isn't OpenAct's it says the port can't be used.
    Neither counts as a crash."""
    port, token = free_port(), secrets.token_hex(32)
    token_file = os.path.join(STATE_DIR, f"server-{port}.token")
    winner = other_server(port, token, "--fake-load-seconds", "2")
    server.stop()
    P._Defaults.port, exits, seen, own = port, list(server.STATE.exits), [], None
    try:
        up = drive(engine, lambda: listening(port) and os.path.exists(token_file), 20)
        started = racing_start()
        own = server.STATE.process
        engine._last_refresh = time.monotonic()  # no health probe for 5 s: only the lost port may attach it
        node = used("Race", MATCH, Question="Who won the race?", Items="left, right")
        engine.schedule(immediate=True)

        def ready():
            now = (server.status(), line(node))
            if not seen or seen[-1] != now:
                seen.append(now)
            return server.is_ready()

        quick = drive(engine, ready, 4.5)
        done = settle(lambda: ready() and complete(node), 30)
        try:
            code = own.wait(10) if own is not None else None
        except subprocess.TimeoutExpired:  # it is serving too: ended below
            code = None
        server.refresh()  # forgets its process, if nothing did yet
        check("lost race: this Blender's server can't bind the port the other holds: it exits with the port-taken "
              "code, and the token file keeps the other's token", up and started and code == server.PORT_TAKEN
              and read(token_file).decode("ascii") == token, (up, started, code))
        check("lost race: it attaches to the other at once (ready within 4.5 s with no health probe, the other loading "
              "for 2 s) with its token, shows no error meanwhile, and the node is answered", quick and done
              and server.STATE.token == token and not any(status[0] == "error" for status, _line in seen), seen)
        check("lost race: losing the port is no crash", server.STATE.exits == exits and server.STATE.process is None,
              (server.STATE.exits, exits, server.STATE.process))
    finally:
        server.stop()  # only detaches: the winner is another Blender's
        ended(winner)
        if own is not None:
            ended(own)

    blocker = socket.socket()  # a program that isn't OpenAct's holds the port
    if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
        blocker.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    blocker.bind(("127.0.0.1", port))
    blocker.listen(1)
    try:
        started = server.start()  # its /health check gets no answer: nothing OpenAct can use listens
        failed = drive(engine, lambda: server.status()[0] == "error", 20)
        check("lost race: when a program that isn't OpenAct's holds the port, the server exits with the port-taken "
              "code, the status says it can't listen, and that is no crash either", started and failed
              and "cannot listen" in server.status()[1] and server.STATE.process is None
              and server.STATE.exits == exits, (server.status(), server.STATE.exits))
    finally:
        blocker.close()
        if server.STATE.process is not None:  # it is serving too
            ended(server.STATE.process)
            server.STATE.process = None
        server.STATE.set("stopped", "")
        P._Defaults.port = PORT
    check("lost race: this Blender's own server starts again on its port",
          server.start() and drive(engine, server.is_ready, 30), server.status())


def attach_other(port, token, *flags):
    """This Blender attached to a server another Blender started on its own port: (that server, attached)."""
    other = other_server(port, token, *flags)
    server.stop()
    P._Defaults.port = port
    up = drive(engine, lambda: listening(port) and os.path.exists(os.path.join(STATE_DIR, f"server-{port}.token")), 20)
    return other, up and server.start() and drive(engine, server.is_ready, 20) and server.attached()


def phase_own_after_other():
    """§10.5: the other Blender's server goes away, and this Blender starts its own: before that one's first /health,
    the Model box must not call it shared, and Stop Model must shut it down (not detach and leave it on the port)."""
    port = free_port()
    other, attached = attach_other(port, secrets.token_hex(32))
    try:
        ended(other)  # the other Blender closes
        server.refresh()
        gone = drive(engine, lambda: server.status()[0] == "stopped", 10)
        started, own = server.start(), server.STATE.process
        box = drawn(lambda layout: ui.draw_server(layout))
        shown = (server.attached(), server.SHARED in text_of(box), ("op", "openact.server_stop", None) in box)
        server.stop()
        ended_own = own is not None and own.poll() is not None and not listening(port)
        check("own after other: once the other Blender's server has gone, this Blender's own starting server shows "
              "Stop Model, not a shared line and Detach", attached and gone and started and shown == (False, False, True),
              (attached, gone, started, shown))
        check("own after other: and Stop Model shuts it down, leaving the port free", ended_own,
              (own and own.poll(), listening(port)))
    finally:
        ended(other)
        P._Defaults.port = PORT
    check("own after other: this Blender's own server starts again on its port",
          server.start() and drive(engine, server.is_ready, 30), server.status())


def phase_detach_sticks():
    """§10.5: after Detach, this Blender doesn't attach again by itself, auto-start or not, until Start Model, or until
    the other server goes away."""
    port = free_port()
    other, attached = attach_other(port, secrets.token_hex(32))
    found = {}
    try:
        server.stop()  # Detach
        node = used("Detached", MATCH, Question="Is it detached?", Items="rope, knot")  # cells to ask meanwhile
        for auto in (False, True):
            P._Defaults.auto_start = auto
            server.refresh()  # a health probe, which finds the other server
            drive(engine, lambda: False, 1.5)
            found[auto] = (server.status()[0], line(node)[0])
        again = server.start() and drive(engine, lambda: server.is_ready() and complete(node), 20)
        server.stop()
        ended(other)  # the other server goes: auto-start (on) starts this Blender's own for new cells
        server.refresh()
        drive(engine, lambda: False, 1.0)
        node.inputs["Question"].default_value = "Is it detached for good?"
        engine.run_pass()
        own = drive(engine, lambda: server.is_ready() and not server.attached(), 20)
        check("detach: after Detach a health probe doesn't attach again, with auto-start off or on, and the node "
              "waits as paused", attached and found == {False: ("stopped", "paused"), True: ("stopped", "paused")},
              found)
        check("detach: Start Model attaches again and the node is answered; once the other server has gone, auto-start "
              "starts this Blender's own", again and own, (again, own, server.status()))
    finally:
        P._Defaults.auto_start = True
        ended(other)
        server.stop()
        P._Defaults.port = PORT
    check("detach: this Blender's own server starts again on its port",
          server.start() and drive(engine, server.is_ready, 30), server.status())


def phase_detach_while_starting():
    """§10.5: Detach lasts also while the other Blender's server still loads (the start poller doesn't attach again
    once it is ready), and in a lost start race while this Blender's losing server process still runs."""
    port, token = free_port(), secrets.token_hex(32)
    token_file = os.path.join(STATE_DIR, f"server-{port}.token")
    other = other_server(port, token, "--fake-load-seconds", "4")
    server.stop()
    P._Defaults.port = port
    try:
        up = drive(engine, lambda: listening(port) and os.path.exists(token_file), 20)
        loading = up and server.start() and server.status()[0] == "starting" and server.attached()
        server.stop()  # Detach, which the Model box offers while the other server loads
        node = used("Detached while loading", MATCH, Question="Is it loaded yet?", Items="sand, clay")
        engine.schedule(immediate=True)
        loaded = drive(engine, lambda: health().get("status") == "ready", 10)
        drive(engine, lambda: False, 1.5)  # the start poller asks /health every 0.5 s
        server.refresh()  # and a health probe
        drive(engine, lambda: False, 1.0)
        after = (server.status()[0], server.attached(), server.STATE.token, server.STATE.detached, line(node)[0])
        again = server.start() and settle(lambda: server.is_ready() and complete(node), 20)
        check("detach: pressed while the other Blender's server loads, Detach lasts: once that server is ready, this "
              "Blender is still stopped and not attached, with no token, and the node waits as paused; Start Model "
              "then attaches again and the node is answered", loading and loaded
              and after == ("stopped", False, "", True, "paused") and again, (loading, loaded, after, again))
    finally:
        server.stop()
        ended(other)

    port, token = free_port(), secrets.token_hex(32)
    token_file, script, own = os.path.join(STATE_DIR, f"server-{port}.token"), server.SERVER_SCRIPT, None
    slow = os.path.join(TMP, "slow_server.py")  # this Blender's server, 2 s late to the port it then loses
    with open(slow, "w", encoding="utf-8") as handle:
        handle.write(f"import runpy, time\ntime.sleep(2)\nrunpy.run_path({script!r}, run_name='__main__')\n")
    winner = other_server(port, token)
    P._Defaults.port = port
    try:
        up = drive(engine, lambda: listening(port) and os.path.exists(token_file), 20)
        server.SERVER_SCRIPT = slow
        started = racing_start()
        server.SERVER_SCRIPT = script
        own = server.STATE.process
        shared = drive(engine, lambda: server.attached() and server.is_ready(), 1.5)
        alive = own is not None and own.poll() is None
        server.stop()  # the Model box's Detach, while the losing process still runs
        server.refresh()  # a health probe, which finds the winner
        drive(engine, lambda: False, 1.5)
        after = (server.status()[0], server.attached(), server.STATE.token, server.STATE.detached,
                 own.poll() if own is not None else None, winner.poll())
        check("detach: pressed in a lost start race while this Blender's losing server still runs, Detach lasts: that "
              "process ends with the port-taken code, the winner runs on, and after a health probe this Blender is "
              "still stopped and not attached, with no token", up and started and shared and alive
              and after == ("stopped", False, "", True, server.PORT_TAKEN, None), (up, started, shared, alive, after))
    finally:
        server.SERVER_SCRIPT = script
        server.stop()
        ended(winner)
        if own is not None:
            ended(own)
        P._Defaults.port = PORT
    check("detach: this Blender's own server starts again on its port",
          server.start() and drive(engine, server.is_ready, 30), server.status())


def phase_timeouts():
    """§10.5: a health probe that times out says nothing: an attached, ready server stays ready, and a Detach lasts."""
    port = free_port()
    other, attached = attach_other(port, secrets.token_hex(32))
    real = server.http

    def slow(method, path, body=None, port=None, timeout=10.0):
        if method == "GET" and path == "/health":
            raise TimeoutError("timed out")  # as from a busy server, or a Blender busy on its main thread
        return real(method, path, body, port, timeout)

    try:
        server.http = slow
        server.refresh()
        drive(engine, lambda: False, 1.0)
        ready = (server.status()[0], server.attached())
        server.stop()  # Detach
        server.refresh()
        drive(engine, lambda: False, 1.0)
        timed = server.STATE.detached
        server.http = real
        server.refresh()  # the other server answers
        drive(engine, lambda: False, 1.5)
        after = (server.status()[0], server.attached(), server.STATE.detached, server.STATE.token)
        check("timeout: a health probe that times out leaves an attached, ready server ready: a timeout is no verdict",
              attached and ready == ("ready", True), ready)
        check("timeout: ...and doesn't end a Detach: after it, a probe that finds the other server leaves this Blender "
              "stopped and detached, with no token", timed and after == ("stopped", False, True, ""), (timed, after))
    finally:
        server.http = real
        ended(other)
        server.stop()
        P._Defaults.port = PORT
    check("timeout: this Blender's own server starts again on its port",
          server.start() and drive(engine, server.is_ready, 30), server.status())


def phase_detach_taken_chunk():
    """§10.5: Detach while the worker holds a chunk it has taken from the queue: the chunk never goes out without a
    token (nothing reaches the other server), and no "…without a readable token" error follows."""
    port, record = free_port(), os.path.join(TMP, "other_record.jsonl")
    other, attached = attach_other(port, secrets.token_hex(32), "--record", record)
    armed, held, go, take = threading.Event(), threading.Event(), threading.Event(), pipeline.take

    def holding(timeout):
        armed.set()  # the worker's takes come here from now on
        chunk = take(timeout)
        if chunk is not None and not held.is_set():
            held.set()  # taken: the test detaches now
            go.wait(10)
        return chunk

    pipeline.take = holding
    try:
        server.worker()
        armed.wait(5)  # a take the worker began before the patch would send the chunk unheld
        node = used("Taken", MATCH, Question="Is it taken?", Items="cup, saucer")
        engine.schedule(immediate=True)
        took = drive(engine, held.is_set, 10)
        server.stop()  # Detach while the worker holds the chunk
        sent = len(read(record).splitlines())
        go.set()
        drive(engine, lambda: False, 1.5)
        after = (server.status(), server.STATE.detached, line(node)[0], len(read(record).splitlines()) - sent)
        check("detach: a chunk the worker took before Detach isn't sent without a token: nothing reaches the other "
              "server, no '…without a readable token' error follows, and the node waits as paused",
              attached and took and after == (("stopped", ""), True, "paused", 0), after)
    finally:
        pipeline.take = take
        go.set()
        ended(other)
        server.stop()
        P._Defaults.port = PORT
    check("detach: this Blender's own server starts again on its port",
          server.start() and drive(engine, server.is_ready, 30), server.status())


def phase_other_gone_while_loading():
    """§10.5: the other Blender's server goes away during its load while this Blender is attached to it: this Blender
    is stopped within seconds, not "Loading the model..." for up to 10 minutes, its start poller ends, and auto-start
    then starts this Blender's own server."""
    port, token = free_port(), secrets.token_hex(32)
    token_file = os.path.join(STATE_DIR, f"server-{port}.token")
    other = other_server(port, token, "--fake-load-seconds", "60")
    server.stop()
    P._Defaults.port, P._Defaults.auto_start = port, False
    try:
        up = drive(engine, lambda: listening(port) and os.path.exists(token_file), 20)
        loading = up and server.start() and server.status() == ("starting", "Loading the model...") \
            and server.attached()
        node = used("Orphaned", MATCH, Question="Is anyone there?", Items="echo, silence")
        engine.schedule(immediate=True)
        ended(other)  # the other Blender closes during its load
        gone = drive(engine, lambda: server.status() == ("stopped", "The model server stopped"), 8)
        polled = drive(engine, lambda: not server.STATE.poll_thread.is_alive(), 2)
        paused = drive(engine, lambda: line(node)[0] == "paused", 3)
        P._Defaults.auto_start = True
        own = drive(engine, lambda: server.is_ready() and not server.attached() and complete(node), 30)
        check("other gone while loading: the other Blender's server dies during its load: within seconds this Blender "
              "is stopped, not 'Loading the model...', its start poller has ended, and the node waits as paused",
              loading and gone and polled and paused, (loading, gone, polled, paused, server.status()))
        check("other gone while loading: ...then auto-start starts this Blender's own server, which answers the node",
              own, (server.status(), line(node)))
    finally:
        P._Defaults.auto_start = True
        ended(other)
        server.stop()
        P._Defaults.port = PORT
    check("other gone while loading: this Blender's own server starts again on its port",
          server.start() and drive(engine, server.is_ready, 30), server.status())


def phase_stop_while_loading():
    """§17.6: Stop Model while this Blender's own server still loads ends stopped: the exit it asks for is no crash,
    and is not reported as one."""
    fake, exits = list(server.FAKE_ARGS), list(server.STATE.exits)
    server.stop()
    server.FAKE_ARGS[:] = [*fake, "--fake-load-seconds", "5"]
    P._Defaults.auto_start = False  # auto-start would start it again at once, which would hide an error
    try:
        loading = server.start() and drive(engine, lambda: server.status() == ("starting", "Loading the model..."), 15)
        own = server.STATE.process
        server.stop()  # Stop Model during the load
        drive(engine, lambda: False, 2.0)  # the start poller's rounds
        after = (server.status(), own.poll() if own is not None else None, server.STATE.exits == exits)
        check("stop while loading: Stop Model during this Blender's own load ends stopped, not in 'The model server "
              "exited', and is no crash", loading and after == (("stopped", ""), 0, True), after)
    finally:
        server.FAKE_ARGS[:] = fake
        P._Defaults.auto_start = True
    check("stop while loading: this Blender's own server starts again on its port",
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
        idle = reading(100.0, 1.5)  # nothing asked: the rate is the last fill's, not the GPU's now
        node = used("Slow fill", MATCH, Question="Is it stalled? It is slow", Items="snail")
        node.balance_order = False  # one scored row, which the fake model takes 8 s over: cells asked all along
        engine.schedule(immediate=True)
        asking = drive(engine, lambda: bool(pipeline.PENDING), 10)
        at_limit = reading(3 * 21.0, 1.5)
        early = reading(100.0, 0.4)
        late = reading(100.0, 1.0)
        box = text_of(drawn(lambda layout: ui.draw_server(layout)))
        done = settle(lambda: complete(node), 30)
        after = reading(100.0, 1.5)
    finally:
        engine.SLOW_FOR, server.STATE.health = slow_for, real
    check("slow server: while nothing is asked, even a rate above 3 x 21 ms per row raises no note, before a fill and "
          "after it", idle == "" and done and after == "", (idle, done, after))
    check("slow server: while cells are asked, no note at 3 x 21 ms per row", asking and engine.SLOW_MS == 63.0
          and at_limit == "", (asking, at_limit))
    check("slow server: above it for a minute (time-scaled) of asking, and not before, the Model box notes 'Running "
          "slower than expected: is the GPU busy (rendering, another app)?'", early == "" and late == engine.SLOW
          and engine.SLOW in box, (early, late, box))


# ------------------------------------------------------------------------------------ preferences

def phase_old_device():
    """A profile saved with the Device preference's old CPU choice, which Blender now reads as "": the server starts
    with --device auto, a value it takes, and the diagnostics say so."""
    device = P._Defaults.device
    server.stop()
    P._Defaults.device = ""  # AddonPreferences refuse the old value, so the defaults stand in for such a profile
    try:
        started = server.start()
        argv = list(server.STATE.process.args) if server.STATE.process else []
        ready = drive(engine, lambda: server.status()[0] in ("ready", "error"), 30) and server.is_ready()
        shown = re.search(r"\(device (\w*), GPU", ops.diagnostics())
    finally:
        P._Defaults.device = device
    at = argv.index("--device") + 1 if "--device" in argv else len(argv)
    check("old device preference: a profile saved with the CPU choice, read as \"\", starts the server with --device "
          "auto, and the diagnostics say device auto",
          started and ready and argv[at:at + 1] == ["auto"] and shown is not None and shown[1] == "auto",
          (server.status(), argv[at:at + 1], shown and shown[0]))


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
        phase_lost_race()
        phase_own_after_other()
        phase_detach_sticks()
        phase_detach_while_starting()
        phase_timeouts()
        phase_detach_taken_chunk()
        phase_other_gone_while_loading()
        phase_stop_while_loading()
        phase_crash_loop()
        phase_stall()
        phase_slow()
        phase_old_device()
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
