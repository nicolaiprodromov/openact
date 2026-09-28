"""The local model server: install, start, stop, and ask (Blender side).

The model runs in its own Python environment (the "runtime") as a separate process that
serves a Jev-compatible API on 127.0.0.1. Nothing heavy is imported into Blender. All network
and process work happens on background threads: the decision worker sends the questions, the
executor does the rest. Results are handed back to Blender's main thread through a queue that a
timer drains, because bpy may only be touched from there.
"""

from __future__ import annotations

import dataclasses
import glob
import hashlib
import json
import os
import pathlib
import platform
import queue
import secrets
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import traceback
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

import bpy

from . import pipeline
from . import prefs as P

SERVER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runtime", "openact_server.py")
SERVER_ID = "openact-semif"
TOKEN_REFUSED = ("A model server started elsewhere is running on port {} without a readable token: stop it or choose "
                 "another port")
CRASH_EXITS, CRASH_WINDOW = 3, 600.0  # unexpected exits of this Blender's server within 10 minutes that pause auto-start
KEEPS_STOPPING = "The model server keeps stopping: see the log"
PORT_TAKEN = 4  # openact_server.py's exit code when it can't bind its port: attach to the server that holds it
SHARED = "Shared: started by another Blender"
# What a /health probe of the port finds (§10.5): its answer; a refused connection, so nothing listens; or a timeout,
# which says nothing (a busy server, or a Blender busy on its main thread)
ANSWERED, REFUSED, TIMED_OUT = "answered", "refused", "timed out"
PROBE_TIMEOUT = 3.0  # longer than a refused localhost connection takes on Windows (about 2 s): nothing there is REFUSED
FAKE_ARGS: list[str] = []  # tests: more openact_server.py flags for the fake backend, e.g. --record <path.jsonl>

SEMIF_COMMIT = "23cf1f39fc9534fe81437200959b6dfc7106e45a"
SEMIF_URL = f"https://github.com/TheoLeeCJ/SemIf-OpenJev/archive/{SEMIF_COMMIT}.zip"
SEMIF_SHA256 = "6eaab1415902b8c3b703fe2d02b23851c2c0c9a0f09cfdc4f8df176e2b7ce463"  # GitHub may re-pack archives
UV_VERSION = "0.8.17"
UV_SHA256 = {  # the release's published checksums, for the platforms that can have an NVIDIA GPU
    "uv-x86_64-pc-windows-msvc.zip": "0d051779fbcb173b183efeae1c3e96148764fd82709bbbf0966df3efe48b67c5",
    "uv-x86_64-unknown-linux-gnu.tar.gz": "920cbcaad514cc185634f6f0dcd71df5e8f4ee4456d440a22e0f8c0f142a8203",
    "uv-aarch64-unknown-linux-gnu.tar.gz": "9a20d65b110770bbaa2ee89ed76eb963d8c6a480b9ebef584ea9df2ae85b4f0f",
}
# torch and every dependency of torch and SemIf, pinned with hashes (tools/make_locks.ps1). v1 is CUDA only.
LOCK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runtime", "requirements-cu128.lock")
MIN_FREE_GB = 15
MIN_VRAM_GB = 10
PROBE_CODE = "import torch, semif_phase1"
CHECKING = "Checking the runtime..."
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # Windows only: no console window for child processes

# ---------------------------------------------------------------------------------- state


class _State:
    def __init__(self):
        # The facts below change under it: a probe's result is applied whole, or not at all once stale (_observe())
        self.lock = threading.RLock()
        self.epoch = 0  # bumped by each start, Stop Model and start poller: older probes and requests say nothing
        self.status = "unknown"  # unknown, not_installed, stopped, starting, ready, error, installing
        self.message = ""
        self.health = {}
        self.process = None  # the Popen we started, if any
        self.port = P.DEFAULT_PORT
        self.poll_thread = None
        self.install_thread = None
        self.install_step = ""
        self.install_warning = ""  # e.g. little VRAM: the status moves on to the start, the Model box keeps it
        self.log_path = ""
        self.conflict = ""  # the error shown while an OpenAct server this Blender can't use holds the port
        self.refused = None  # the pid of the one that refused this Blender's token (403), else None
        self.probes = {}  # runtime folder -> "" once its import probe passed this session, else the probe's error
        self.checking = ""  # the runtime folder whose import probe is running
        self.probe_id = 0  # that probe's: a result with another id comes from a superseded probe
        self.start_queued = False  # a start that runs when that probe passes
        # Sent as X-OpenAct-Token (§16.1): the launch token of the server this Blender started, else the one in the
        # token file of the server it attached to (§10.5)
        self.token = ""
        self.folder = ""  # state_dir() of the runtime: token files and logs/, for threads, which can't read preferences
        self.log_requests = False  # the preference, for the worker
        self.detached = False  # Stop Model left another Blender's server: no attaching until Start Model (§10.5)
        self.exits = []  # time.monotonic() of recent unexpected exits of the server this Blender started (§17.6)

    def set(self, status, message=""):
        with self.lock:
            changed = (status, message) != (self.status, self.message)
            self.status, self.message = status, message
        if changed:
            post(_status_changed)


STATE = _State()
_results: queue.Queue = queue.Queue()
_executor: ThreadPoolExecutor | None = None
_worker: Worker | None = None
_status_listeners = []


def post(callback, *args):
    """Run callback(*args) on Blender's main thread (from any thread)."""
    _results.put((callback, args))


def pump():
    """Drain the main-thread queue. Called by the engine timer."""
    handled = 0
    while True:
        try:
            callback, args = _results.get_nowait()
        except queue.Empty:
            return handled
        handled += 1
        try:
            callback(*args)
        except Exception:
            traceback.print_exc()


def _status_changed():
    for listener in list(_status_listeners):
        try:
            listener()
        except Exception:
            traceback.print_exc()


def add_status_listener(fn):
    if fn not in _status_listeners:
        _status_listeners.append(fn)


def executor():
    """For the /health probes and the runtime's import probe; the questions go through the worker (§5.5)."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="openact")
    return _executor


def status():
    with STATE.lock:
        return STATE.status, STATE.message


def is_ready():
    return STATE.status == "ready"


# ----------------------------------------------------------------------------------- http


def _opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never proxy localhost


class NoToken(Exception):
    """A POST while no token is in use (after a Detach, or a token file that can't be read): it never leaves."""


def http(method, path, body=None, port=None, timeout=10.0):
    """Blocking JSON request to the local server. Returns (status_code, json)."""
    url = f"http://127.0.0.1:{port or STATE.port}{path}"
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    # Every POST carries the token, read as it leaves: that of the server in use then (§10.5). GETs never carry it: the
    # probes may reach a program on the port that isn't ours
    if method == "POST":
        token = STATE.token
        if not token:
            raise NoToken(path)
        headers["X-OpenAct-Token"] = token
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with _opener().open(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as error:
        try:
            payload = json.loads(error.read().decode("utf-8") or "{}")
        except ValueError:
            payload = {"error": f"HTTP {error.code}"}
        return error.code, payload


def _offline(error):
    """True when nothing listens: the connection was refused or reset. A timeout is no verdict (§10.5)."""
    return isinstance(getattr(error, "reason", error), (ConnectionRefusedError, ConnectionResetError))


def _timed_out(error):
    return isinstance(getattr(error, "reason", error), TimeoutError)


class Worker(threading.Thread):
    """The decision worker 'openact-decide' (§5.5, §10.5): sends pipeline's queued chunks, highest priority first, one
    request at a time, and posts each result to the main thread. It never touches bpy: engine.tick() publishes the
    two events it reads."""

    render_running = threading.Event()  # no new chunk starts during a render job; the one in flight completes
    playing = threading.Event()  # queued chunks may still finish during playback: only the pass waits for its end

    def __init__(self):
        super().__init__(name="openact-decide", daemon=True)
        self.stopped = False
        self.sized = None  # (port, pid) of the server whose /v1/capabilities sized the chunks

    def run(self):
        while not self.stopped:
            if self.render_running.is_set():
                time.sleep(0.1)
                continue
            chunk = pipeline.take(timeout=0.25)
            if chunk is None:
                continue
            if self.stopped or self.render_running.is_set():
                pipeline.requeue(chunk)
                continue
            try:
                self.send(chunk)
            except Exception as error:  # a bug must not stop the thread, nor leave the chunk's cells pending
                traceback.print_exc()
                post(pipeline.answered, chunk, "error", f"Internal error: {error}")

    def send(self, chunk):
        self.size()
        request, ids = pipeline.build_request(chunk)
        token, epoch, started = STATE.token, STATE.epoch, time.perf_counter()
        try:
            code, body = http("POST", "/v1/systemone", request, timeout=180.0)
        except NoToken:  # none in use (a Detach, a token file that can't be read): nothing went out
            post(pipeline.answered, chunk, "refused", None)  # asked again once a server is in use (§10.5)
            return
        except Exception as error:
            if not (_offline(error) or _timed_out(error)):
                post(pipeline.answered, chunk, "error", f"{type(error).__name__}: {error}")
                return
            pipeline.requeue(chunk)  # its cells stay queued, and auto-start applies (§13.2)
            if _offline(error):  # a timeout says nothing about the server
                post(_lost_connection, epoch)
            self.wait()
            return
        if STATE.log_requests:
            _log_request(request, code, body, time.perf_counter() - started, sum(cell.cost for cell in chunk.cells))
        if code == 200:
            post(pipeline.answered, chunk, "ok", body)
        elif code == 503 and body.get("status") == "loading":
            pipeline.requeue(chunk)  # at the head: P0 work may overtake it, and cancel() acts, while the model loads
            self.wait()
        elif code in (413, 422) and len(chunk.cells) > 1:
            if code == 413:  # never expected: chunks stay far below the 4 MB limit (§13.2)
                print(f"[OpenAct] the model server found a request of {len(chunk.cells)} questions too large; "
                      "splitting it")
            named = body.get("question") if code == 422 else None
            if named in ids:  # only the question the server names fails; the others are sent again
                bad = ids.index(named)
                post(pipeline.answered, dataclasses.replace(chunk, cells=chunk.cells[bad:bad + 1]), "error",
                     f"The model refused: {body.get('error')}")
                pipeline.requeue(dataclasses.replace(chunk, cells=chunk.cells[:bad] + chunk.cells[bad + 1:]))
            else:  # unnamed: halves, until the bad question travels alone (§10.3)
                half = len(chunk.cells) // 2
                pipeline.requeue(dataclasses.replace(chunk, cells=chunk.cells[:half]),
                                 dataclasses.replace(chunk, cells=chunk.cells[half:]))
        elif code == 403 and (STATE.token != token or _adopt_token()):
            # Sent with a token that has changed since: that of a server that has gone, whose replacement took the
            # request (a refused connection takes seconds on Windows), or one its token file no longer holds. Asked
            # again with the new one
            pipeline.requeue(chunk)
        elif code == 403:  # a server whose token file is missing or wrong: nothing fails, asked again once it has gone
            _refused_by(STATE.health.get("pid"))
            post(pipeline.answered, chunk, "refused", None)
        elif code == 503:  # the model failed to load: a failure, never requeued
            error = body.get("error") or "The model failed to load"
            STATE.set("error", error)
            post(pipeline.answered, chunk, "error", error)
        else:  # one refused question, or a 500: Ask Again retries
            post(pipeline.answered, chunk, "error", f"The model refused: {body.get('error') or f'HTTP {code}'}")

    def wait(self):
        """Poll /health every 0.5 s until the server answers with its model loaded or failed (§10.5). Nothing is in
        flight meanwhile, so cancel() acts and P0 work overtakes the requeued chunk."""
        while not self.stopped:
            time.sleep(0.5)
            try:
                code, health = http("GET", "/health", timeout=1.0)
            except Exception:
                continue
            if code == 200 and health.get("status") != "loading":
                return

    def size(self):
        """Size the chunks from /v1/capabilities, once per server process (§5.5); the defaults stay without it."""
        sized = (STATE.port, STATE.health.get("pid"))
        if sized == self.sized:
            return
        self.sized = sized
        try:
            code, capabilities = http("GET", "/v1/capabilities", timeout=2.0)
        except Exception:
            return
        if code == 200:
            pipeline.size_chunks(capabilities)


def _log_request(request, code, reply, seconds, rows):
    """With the log_requests preference on: one JSON line per request in <state>/logs/requests-YYYYMMDD.jsonl (§17.4).
    rows: the scored rows its questions cost. Worker thread."""
    if not STATE.folder:
        return
    now, folder = time.localtime(), os.path.join(STATE.folder, "logs")
    line = {"t": time.strftime("%Y-%m-%dT%H:%M:%S", now), "status": code, "state": request["state"],
            "questions": request["questions"], "answers": reply.get("answers"), "ms": round(seconds * 1000, 1),
            "rows": rows}
    path = os.path.join(folder, time.strftime("requests-%Y%m%d.jsonl", now))
    try:
        os.makedirs(folder, mode=0o700, exist_ok=True)  # prompts and CSV text: this user's only, on POSIX (§16.1)
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600), "a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")
    except OSError as error:  # the answers matter more than their log
        print(f"[OpenAct] could not write the request log: {error}")


def worker():
    """The decision worker, started on first use."""
    global _worker
    if _worker is None or not _worker.is_alive():
        _worker = Worker()
        _worker.start()
    return _worker


def _lost_connection(epoch):
    """The worker's request found nothing listening: what a refused probe finds. Main thread only."""
    _reap()  # a server of this Blender's that crashed: counted, and forgotten
    _observe(REFUSED, {}, epoch)


# ------------------------------------------------------------------------------ lifecycle


def _backend():
    """The server's --backend. OPENACT_BACKEND=fake needs no GPU or installed runtime (tests)."""
    return "fake" if os.environ.get("OPENACT_BACKEND") == "fake" else "semif"


def _other_backend(health):
    """The backend of the OpenAct server that sent this /health, if it is not the one this session uses."""
    running = health.get("backend", "semif-torch")  # servers before 1.0.0 only ran SemIf
    return None if running == ("fake" if _backend() == "fake" else "semif-torch") else running


def _refuse_backend(other):
    """Treat an OpenAct server with another backend on the port like any other program there."""
    STATE.conflict = (f"Port {STATE.port} is used by an OpenAct model server with the {other} backend. "
                      "Stop it or pick another port in the preferences")
    STATE.set("error", STATE.conflict)


def _refused_by(pid):
    """The server on the port refused this Blender's token (403), or its token file can't be read: it was started
    elsewhere. It stays unusable, and the status an error, while /health reports its pid (§10.5). Any thread."""
    with STATE.lock:
        STATE.refused, STATE.conflict = pid, TOKEN_REFUSED.format(STATE.port)
        STATE.set("error", STATE.conflict)


def state_dir(root):
    """Writable state such as server.log, probe.json, the token files and logs/: OPENACT_STATE_DIR, so the runtime can
    be shared read-only, else the runtime folder."""
    return os.environ.get("OPENACT_STATE_DIR") or root


def _token_file(port):
    """Where the server on this port keeps its launch token for other Blenders (§10.5). The server writes it once it
    holds the port, so a Blender whose server lost the port to another never overwrites it."""
    return os.path.join(STATE.folder, f"server-{port}.token")


def _read_token(port):
    """The launch token in the token file of the server on this port, "" when there is none or it can't be read."""
    if not STATE.folder:  # no refresh() or start() has said where the runtime is
        return ""
    try:
        with open(_token_file(port), encoding="ascii") as handle:
            return handle.read(256).strip()
    except (OSError, ValueError):
        return ""


def attached():
    """True while the server on the port is another Blender's (§10.5): its /health names another launching Blender.
    Any thread."""
    return bool(STATE.health) and STATE.health.get("parent_pid") != os.getpid()


def _adopt_token():
    """For another Blender's server: use the token its file holds, when that differs from the one in use, and forget a
    refusal. True when the token changed. Any thread."""
    with STATE.lock:
        token = _read_token(STATE.port) if attached() else STATE.token
        if token == STATE.token:
            return False
        STATE.token, STATE.refused = token, None
        return True


def _child_env():
    """os.environ without the variables that would point the runtime's Python at other packages."""
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(key, None)
    return env


def _imports_ok(python):
    """The runtime probe: does this Python import torch and SemIf? Takes seconds on a cold disk."""
    try:
        return subprocess.run([python, "-c", PROBE_CODE], capture_output=True, stdin=subprocess.DEVNULL,
                              env=_child_env(), timeout=300, creationflags=_NO_WINDOW).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _probe_key(root):
    """What a passed probe depends on: the folder, and the mtimes of pyvenv.cfg and of torch's dist-info."""
    venv = os.path.join(root, "venv")
    site = ("Lib", "site-packages") if sys.platform == "win32" else ("lib", "python3*", "site-packages")
    try:
        torch = glob.glob(os.path.join(glob.escape(venv), *site, "torch-*.dist-info"))[0]
        return [root, os.path.getmtime(os.path.join(venv, "pyvenv.cfg")), os.path.getmtime(torch)]
    except (IndexError, OSError):
        return None


def _probe(root, probe_id):
    """Executor: probe the runtime, unless probe.json holds a pass for it as it is now. Posts the result."""
    key = _probe_key(root)
    path = os.path.join(state_dir(root), "probe.json")
    try:
        with open(path, encoding="utf-8") as handle:
            passed = key is not None and json.load(handle) == key
    except (OSError, ValueError):
        passed = False
    if not passed and _imports_ok(P.venv_python(root)):
        passed = True
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(key, handle)  # only passes are stored
        except OSError:
            pass  # the next session probes again
    error = "" if passed else (f"The runtime's Python at {P.venv_home(root)} is missing or broken (was it removed "
                               "with `uv python uninstall`?): reinstall the runtime")
    post(_probed, root, error, probe_id)


def _check(root):
    """Start the probe of this runtime folder on the executor, unless it is running: nothing waits for it."""
    if STATE.checking != root:
        STATE.checking = root
        STATE.probe_id += 1
        STATE.set("unknown", CHECKING)
        executor().submit(_probe, root, STATE.probe_id)


def _probed(root, error, probe_id):
    if probe_id != STATE.probe_id:
        return  # superseded by another probe, an install or unregistering: neither stored nor shown
    STATE.checking = ""
    if root != P.runtime_root():
        return  # the runtime folder changed to one that needed no probe: this result is stale
    STATE.probes[root] = error
    queued, STATE.start_queued = STATE.start_queued, False
    if error:
        STATE.set("error", error)  # this also cancels a queued start
    elif queued:
        start()
    else:
        refresh()


def probe_runtime(prefs=None):
    """Probe the runtime folder again in the background, e.g. after Locate Existing Runtime chose it."""
    STATE.probes.pop(P.runtime_root(prefs), None)
    refresh(prefs)


def refresh(prefs=None):
    """Update the status from the file system and a quick /health probe (non-blocking)."""
    prefs = prefs or P.get_prefs()
    STATE.port = prefs.port
    root = P.runtime_root(prefs)
    STATE.folder, STATE.log_requests = state_dir(root), prefs.log_requests
    if STATE.status in ("installing", "starting") or _reap():
        return
    if _backend() != "fake":
        if not P.runtime_installed(root):
            if STATE.status != "error":  # keep an install error visible
                STATE.set("not_installed", "The model runtime is not installed")
            return
        error = STATE.probes.get(root)
        if error is None:  # once per session; the status reads "Checking the runtime..." meanwhile
            _check(root)
            return
        if error:
            STATE.set("error", error)  # until Start Model or Locate Existing Runtime probes again
            return
    if STATE.status in ("unknown", "not_installed"):
        STATE.set("stopped", "")
    port, epoch = STATE.port, STATE.epoch
    executor().submit(lambda: _observe(*_probe_port(port), epoch))


def _probe_port(port, timeout=PROBE_TIMEOUT):
    """GET /health on the port: (ANSWERED, its /health, or {} for an answer that isn't one), (REFUSED, {}) or
    (TIMED_OUT, {}). Any thread."""
    try:
        code, health = http("GET", "/health", port=port, timeout=timeout)
    except Exception as error:
        return (REFUSED if _offline(error) else TIMED_OUT), {}
    return ANSWERED, health if code == 200 else {}


def _observe(outcome, health, epoch, poller=False):
    """Apply what a probe of the port found. Every probe comes here: refresh()'s, the start poller's, a start's, and a
    request of the worker's that was refused. The facts decide the state (§10.5):
    - the probe's outcome;
    - whose server answered (/health's parent_pid);
    - its token file;
    - a Detach, a refusal (403);
    - this Blender's own process.
    A timeout says nothing, and neither does a probe made before the last start, Stop Model or start poller (epoch).
    Starts the poller for a server that loads, unless a poller asks. True when a start poller can end. Any thread."""
    with STATE.lock:
        if epoch != STATE.epoch:
            return True
        if outcome == TIMED_OUT:  # a busy server, or a busy Blender: no verdict
            return False
        if outcome == REFUSED:  # nothing listens on the port
            STATE.detached = False  # the server this Blender detached from has gone
            if STATE.process is not None:
                return False  # its own server doesn't listen yet, or has exited: the poller or _reap() says which
            conflict = (STATE.status, STATE.message) == ("error", STATE.conflict)
            if STATE.status in ("ready", "starting") or conflict:  # the server it used or couldn't use has gone
                if attached():
                    STATE.token = ""  # another Blender's token is of no use any more
                STATE.health, STATE.refused = {}, None
                STATE.set("stopped", "" if conflict else "The model server stopped")  # auto-start may start one
            return True
        if health.get("server") != SERVER_ID:
            STATE.set("error", f"Port {STATE.port} is used by another program. Pick another port in the preferences")
            return True
        other = _other_backend(health)
        if other:
            _refuse_backend(other)  # also ends a "ready" left over from the server it replaced
            return True
        mine = health.get("parent_pid") == os.getpid()
        if STATE.detached and not mine:
            return True  # Detach lasts until Start Model, or until that server goes
        STATE.health = health
        if not mine:
            _adopt_token()  # shared through its token file (§10.5)
        if not STATE.token:  # its token file can't be read: nothing may be sent to it
            _refused_by(health.get("pid"))
            return True
        if STATE.refused is not None and health.get("pid") == STATE.refused:  # still the server that refused
            STATE.set("error", STATE.conflict)
            return True
        STATE.refused = None
        state = health.get("status")
        if state == "ready":
            STATE.set("ready", f"{health.get('source', '')} on {health.get('device') or 'unknown device'}")
        elif state == "error":
            STATE.set("error", health.get("error") or "The model failed to load")
        elif state == "loading":
            STATE.set("starting", "Loading the model...")
            if not poller:
                _start_polling(None)
        return state in ("ready", "error")


def start(prefs=None):
    """Start the model server, or attach to one that is already running through its token file (§10.5). False when
    it can't, e.g. once auto-start paused for a server that keeps stopping (§17.6). Main thread only."""
    prefs = prefs or P.get_prefs()
    if STATE.status in ("starting", "ready", "installing"):
        return True
    if _reap():
        return False
    with STATE.lock:
        STATE.detached = False
        STATE.epoch += 1  # what probes found before this start says nothing about it
    STATE.port = prefs.port
    root = P.runtime_root(prefs)
    STATE.folder = state_dir(root)
    backend = _backend()
    if backend != "fake":
        if not P.runtime_installed(root):
            STATE.set("not_installed", "The model runtime is not installed")
            return False
        if STATE.probes.get(root) != "":  # queued, not waited for: _probed() starts it once the probe passes
            STATE.probes.pop(root, None)  # a failed runtime is probed again, it may have been repaired
            STATE.start_queued = True
            _check(root)
            return True
    found = _attach()
    if found is not None:
        return found
    STATE.health = {}  # a server that has gone: the one started here must not look like another Blender's

    python = os.environ.get("OPENACT_SERVER_PYTHON") or P.venv_python(root)
    folder = STATE.folder
    STATE.log_path = os.path.join(folder, "server.log")
    STATE.token = secrets.token_hex(32)  # new for every start; in the environment, as argv is readable by others
    env = _child_env()
    env.update({
        "OPENACT_TOKEN": STATE.token,
        "HF_HOME": P.hf_home(root),
        "HF_HUB_OFFLINE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    })
    args = [python, "-u", SERVER_SCRIPT, "--port", str(STATE.port), "--backend", backend, "--device", P.device(prefs),
            "--parent-pid", str(os.getpid()), "--token-file", os.path.abspath(_token_file(STATE.port)),
            "--gpu", str(prefs.gpu)]
    if backend == "fake":
        args += FAKE_ARGS
    try:
        os.makedirs(folder, exist_ok=True)
        log = open(STATE.log_path, "w", encoding="utf-8")
        STATE.process = subprocess.Popen(
            args, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            cwd=folder, env=env, creationflags=_NO_WINDOW,
        )
        log.close()  # the child keeps its own handle
    except OSError as error:
        STATE.set("error", f"Could not start the model server: {error}")
        return False
    STATE.set("starting", "Starting the model (the first start takes about 20 seconds)...")
    _start_polling(STATE.process)
    return True


def _attach():
    """Use what answers /health on the port: an OpenAct server with this session's backend (another Blender's is shared
    through its token file, §10.5), else an error that names what holds the port. None when nothing answers in time,
    else what start() returns. Main thread only."""
    outcome, health = _probe_port(STATE.port, timeout=0.5)  # a refusal takes 2 s on Windows: the bind decides then
    if outcome != ANSWERED:
        return None
    _observe(outcome, health, STATE.epoch)
    return health.get("server") == SERVER_ID and not _other_backend(health)


def _lost_port():
    """The server this Blender started couldn't bind its port: one started meanwhile, e.g. by another Blender at the
    same moment, holds it. Attach to that one at once; if nothing answers, the port can't be used. Main thread only."""
    _reap()
    if STATE.status == "starting" and _attach() is None:  # else a health probe attached meanwhile
        STATE.set("error", "The model server exited: " + _log_tail())


def _start_polling(process):
    """Follow a start until the server is ready or has failed, gone or been left: process, the server this Blender
    started, else None for another Blender's that loads (§10.5). It ends once a newer poller, start or Stop Model
    changes the epoch. Any thread."""
    with STATE.lock:
        STATE.epoch += 1
        epoch, port = STATE.epoch, STATE.port

    def poll():
        deadline = time.time() + 600
        while time.time() < deadline and epoch == STATE.epoch:
            if process is not None and process.poll() is not None:
                with STATE.lock:
                    if epoch != STATE.epoch or process is not STATE.process:
                        return  # Stop Model released it: its exit is no crash (§17.6)
                    if process.poll() == PORT_TAKEN:
                        post(_lost_port)
                    else:
                        STATE.set("error", "The model server exited: " + _log_tail())
                        post(_reap)
                return
            if _observe(*_probe_port(port), epoch, poller=True):
                if STATE.status == "ready":
                    post(_became_ready)
                return
            time.sleep(0.5)
        with STATE.lock:
            if epoch == STATE.epoch:
                STATE.set("error", "The model server did not become ready in 10 minutes")

    STATE.poll_thread = threading.Thread(target=poll, name="openact-poll", daemon=True)
    STATE.poll_thread.start()


_ready_listeners = []


def _became_ready():
    for listener in list(_ready_listeners):
        listener()


def add_ready_listener(fn):
    if fn not in _ready_listeners:
        _ready_listeners.append(fn)


def _log_tail(lines=6):
    try:
        with open(STATE.log_path, encoding="utf-8", errors="replace") as handle:
            tail = [line.strip() for line in handle.readlines()[-lines:] if line.strip()]
        return " | ".join(tail)[-600:] or "(empty log)"
    except OSError:
        return "(no log)"


def _reap():
    """Forget the server this Blender started once it has exited (one Stop Model ended never comes here: stop() releases
    it first). An exit that is neither clean (0) nor a port another server holds (PORT_TAKEN) counts: the CRASH_EXITS-th
    within CRASH_WINDOW seconds pauses auto-start (§17.6), as the status becomes an error, which auto-start leaves
    alone, until Start Model. True when it paused. Main thread only."""
    process = STATE.process
    code = process.poll() if process is not None else None
    if code is None:
        return False
    STATE.process = None
    if code in (0, PORT_TAKEN):
        return False
    now = time.monotonic()
    STATE.exits = [t for t in STATE.exits if now - t < CRASH_WINDOW] + [now]
    if len(STATE.exits) < CRASH_EXITS:
        return False
    STATE.exits = []  # Start Model begins a new count
    STATE.set("error", KEEPS_STOPPING)
    return True


def stop():
    """Stop the server this Blender started (/shutdown needs its token: one that refuses keeps running, which the status
    then says). Another Blender's server is never shut down from here (§10.5): this Blender only stops using it, and
    forgets its token until it attaches again; one whose token it can't read keeps its error. The process is released
    first, and the start poller ends: the exit this asks for is never reported as a crash (§17.6). Main thread only."""
    with STATE.lock:
        process, shared = STATE.process, attached()
        own = process is not None and process.poll() is None
        STATE.process = None
        STATE.epoch += 1
        if shared or not (own or STATE.token):  # another Blender's server, or nothing of this Blender's to shut down
            if STATE.refused is None:  # stays detached until Start Model, or until that server goes
                STATE.health, STATE.token, STATE.detached = {}, "", shared
                pipeline.cancel_all()  # queued chunks would go out without a token
                STATE.set("stopped", "")
            if not own:
                return
    try:  # a process of this Blender's while another's server answers lost the port: it is only ended
        code = 0 if shared else http("POST", "/shutdown", body={}, timeout=2.0)[0]
    except Exception:
        code = 0
    if own:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
    elif code == 403:
        _refused_by(STATE.health.get("pid"))
        return
    with STATE.lock:
        STATE.health = {}
        STATE.set("stopped", "")


def shutdown_owned():
    """On add-on unregister: stop the server only if this Blender started it, and the worker, whose queued chunks go.
    The modules stay loaded (re-enabling, Reload Scripts): the next registration's passes ask again."""
    if STATE.process is not None and STATE.process.poll() is None:
        stop()
    STATE.probes.clear()  # the next registration probes again: a probe cancelled below would never report
    STATE.checking, STATE.start_queued = "", False
    STATE.probe_id += 1  # and one that can't be cancelled reports to the next session, which drops it
    global _executor, _worker
    if _executor is not None:
        _executor.shutdown(wait=False, cancel_futures=True)
        _executor = None
    if _worker is not None:
        _worker.stopped = True  # it ends within 0.25 s, or when its request in flight returns
        _worker = None
    pipeline.cancel_all()  # a chunk the worker still holds may come back to the queue: the next submit() sends it


# ------------------------------------------------------------------------------ installer


def _uv_asset():
    machine = platform.machine().lower()
    arch = "aarch64" if machine in ("arm64", "aarch64") else "x86_64"
    if sys.platform == "win32":
        return f"uv-{arch}-pc-windows-msvc.zip"
    if sys.platform == "darwin":
        return f"uv-{arch}-apple-darwin.tar.gz"
    return f"uv-{arch}-unknown-linux-gnu.tar.gz"


def _nvidia_gpus():
    """[(name, MiB of VRAM)] as nvidia-smi reports them: empty without an NVIDIA GPU and driver."""
    tool = shutil.which("nvidia-smi")
    if not tool:
        return []
    try:
        result = subprocess.run([tool, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                                capture_output=True, stdin=subprocess.DEVNULL, timeout=20, creationflags=_NO_WINDOW)
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in result.stdout.decode("utf-8", "replace").splitlines() if result.returncode == 0 else ():
        name, _, memory = line.rpartition(",")
        if memory.strip().isdigit():
            gpus.append((name.strip(), int(memory)))
    return gpus


def install(prefs=None, dry_run=False):
    """Download and set up the runtime in the background (about 13 GB). Main thread only.

    dry_run writes every command and download to install.log without running it."""
    prefs = prefs or P.get_prefs()
    if STATE.install_thread and STATE.install_thread.is_alive():
        return False
    if not bpy.app.online_access:
        STATE.set("error", "Online access is disabled. Enable Preferences > System > Network > Allow Online Access "
                           "to download the model")
        return False
    root = P.runtime_root(prefs)
    previous = status()
    STATE.probes.pop(root, None)  # the runtime is about to change: probe it again afterwards
    STATE.checking, STATE.probe_id = "", STATE.probe_id + 1  # a running probe describes the runtime before it
    STATE.set("installing", "Preparing...")
    warning = ""

    def run(cmd, log, env=None):
        log.write("\n$ " + " ".join(cmd) + "\n")
        log.flush()
        if dry_run:
            return
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                env=env, creationflags=_NO_WINDOW)
        if result.returncode != 0:
            raise RuntimeError(f"'{os.path.basename(cmd[0])}' failed with code {result.returncode} (see install.log)")

    def fetch(url, path, sha256, log):
        log.write(f"downloading {url} (SHA-256 {sha256})\n")
        log.flush()
        if dry_run:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(url, path)
        with open(path, "rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != sha256:
            os.remove(path)
            log.write(f"SHA-256 mismatch: got {digest}\n")
            raise RuntimeError(f"{os.path.basename(path)} does not match its recorded SHA-256: update OpenAct")

    def step(text):
        STATE.install_step = text
        STATE.set("installing", f"{text} {warning}".strip())

    def work():
        nonlocal warning
        log_path = os.path.join(root, "install.log")
        STATE.log_path = log_path
        try:
            os.makedirs(root, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as log:
                log.write(f"\n===== OpenAct runtime install{' (dry run)' if dry_run else ''} {time.ctime()} =====\n")
                if "#" in os.path.abspath(root):  # uv cuts the SemIf archive's file: URL at the '#'
                    raise RuntimeError(f"the runtime folder {os.path.abspath(root)} contains '#', which uv can't "
                                       "install from: choose another Runtime Folder")
                free = shutil.disk_usage(root).free / 1024 ** 3
                log.write(f"free disk space: {free:.1f} GB\n")
                if free < MIN_FREE_GB:
                    raise RuntimeError(f"not enough disk space: the runtime needs {MIN_FREE_GB} GB free for {root}, "
                                       f"{free:.1f} GB is free")
                gpus = _nvidia_gpus()
                log.write(f"NVIDIA GPUs (MiB of VRAM): {gpus}\n")
                if not gpus:
                    raise RuntimeError("OpenAct 1.0 needs an NVIDIA GPU")  # v1 has no CPU runtime
                vram = max(memory for _name, memory in gpus) / 1024
                if vram < MIN_VRAM_GB:
                    warning = (f"Warning: the GPU has {vram:.0f} GB of VRAM; the model needs about 9 GB, and "
                               f"{MIN_VRAM_GB} GB or more is recommended.")
                    log.write(warning + "\n")
                # Without the user's uv settings (UV_* variables, uv.toml): UV_LINK_MODE=symlink, for one, would link
                # the venv into the uv-cache that the last step deletes.
                env = {key: value for key, value in _child_env().items() if not key.startswith("UV_")}
                env["UV_NO_CONFIG"] = "1"
                env["HF_HOME"] = P.hf_home(root)
                env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
                # Keep everything inside the runtime folder: uv's own Python and its download cache.
                env["UV_PYTHON_INSTALL_DIR"] = os.path.join(root, "python")
                env["UV_PYTHON_PREFERENCE"] = "only-managed"
                env["UV_CACHE_DIR"] = os.path.join(root, "uv-cache")
                python = P.venv_python(root)

                step("1/5 Getting uv (Python package manager)...")
                exe = "uv.exe" if sys.platform == "win32" else "uv"
                uv = os.path.join(root, "tools", exe)  # never a uv on PATH: its version isn't the pinned one
                if not os.path.isfile(uv):
                    asset = _uv_asset()
                    if asset not in UV_SHA256:
                        raise RuntimeError(f"there is no pinned uv for this platform ({asset})")
                    archive = os.path.join(root, "tools", asset)
                    fetch(f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/{asset}", archive,
                          UV_SHA256[asset], log)
                    if not dry_run and archive.endswith(".zip"):
                        with zipfile.ZipFile(archive) as zf:
                            member = next(n for n in zf.namelist() if n.endswith("/" + exe) or n == exe)
                            with zf.open(member) as src, open(uv, "wb") as dst:
                                shutil.copyfileobj(src, dst)
                    elif not dry_run:
                        with tarfile.open(archive) as tf:
                            member = next(m for m in tf.getmembers() if m.name.endswith("/" + exe) or m.name == exe)
                            with tf.extractfile(member) as src, open(uv, "wb") as dst:
                                shutil.copyfileobj(src, dst)
                        os.chmod(uv, 0o755)

                step("2/5 Creating the Python environment...")
                # A venv that fails the probe is rebuilt: installing into it again can't restore files uv already
                # counts as installed, or a base Python that is gone (`uv python uninstall`).
                installed = not dry_run and _imports_ok(python)
                if not installed:
                    run([uv, "venv", "--clear", "--python", "3.12", os.path.join(root, "venv")], log, env)

                step("3/5 Installing PyTorch (about 3 GB)...")
                if not installed:  # --no-build: an sdist would build with packages that aren't in the lock
                    run([uv, "pip", "install", "--python", python, "--require-hashes", "--no-build", "--torch-backend",
                         "cu128", "-r", LOCK_FILE], log, env)

                step("4/5 Installing SemIf...")
                if not installed:
                    archive = os.path.join(root, "tools", f"SemIf-OpenJev-{SEMIF_COMMIT}.zip")
                    fetch(SEMIF_URL, archive, SEMIF_SHA256, log)
                    # Its dependencies came from the lock. Building with the lock's setuptools, not a fresh
                    # download, keeps every installed package hash-checked.
                    run([uv, "pip", "install", "--python", python, "--no-deps", "--no-build-isolation",
                         f"semif-phase1 @ {pathlib.Path(archive).absolute().as_uri()}"], log, env)

                step("5/5 Downloading the model (8.7 GB)...")
                if not P.runtime_installed(root):  # resumes an interrupted download
                    code = ("from huggingface_hub import snapshot_download; "
                            f"print(snapshot_download(repo_id={P.MODEL_SOURCE!r}, revision={P.MODEL_REVISION!r}))")
                    run([python, "-c", code], log, env)
                if not dry_run and not P.runtime_installed(root):
                    raise RuntimeError("the model files are incomplete after the download")
                try:  # the venv holds its own copies; the download cache is not needed any more
                    run([uv, "cache", "clean"], log, env)
                except RuntimeError:
                    pass
                log.write("===== done =====\n")
            STATE.install_step = ""
            if dry_run:
                STATE.set(*previous)
                return
            STATE.install_warning = warning
            STATE.set("stopped", f"Runtime installed. {warning}" if warning else "Runtime installed")
            post(_became_installed)
        except Exception as error:
            STATE.install_step = ""
            STATE.set("error", f"Install failed: {error}")
            try:
                with open(log_path, "a", encoding="utf-8") as log:
                    log.write(traceback.format_exc())
            except OSError:
                pass

    STATE.install_thread = threading.Thread(target=work, name="openact-install", daemon=True)
    STATE.install_thread.start()
    return True


def _became_installed():
    prefs = P.get_prefs()
    if prefs.auto_start:
        start(prefs)
