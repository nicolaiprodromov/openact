"""The local model server: install, start, stop, and ask (Blender side).

The model runs in its own Python environment (the "runtime") as a separate process that
serves a Jev-compatible API on 127.0.0.1. Nothing heavy is imported into Blender. All network
and process work happens on background threads; results are handed back to Blender's main
thread through a queue that a timer drains, because bpy may only be touched from there.
"""

from __future__ import annotations

import json
import os
import platform
import queue
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

from . import prefs as P

SERVER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runtime", "openact_server.py")
SERVER_ID = "openact-semif"
LEGACY_SERVER_ID = "jev-nodes-semif"  # started by the jev_nodes v0.1.0 add-on
LEGACY_SERVER_MESSAGE = ("An old Jev Nodes model server is running on this port: disable the Jev Nodes add-on "
                         "or choose another port.")
MODEL_NAME = "semif-qwen3.5-4b"

SEMIF_COMMIT = "23cf1f39fc9534fe81437200959b6dfc7106e45a"
SEMIF_URL = f"https://github.com/TheoLeeCJ/SemIf-OpenJev/archive/{SEMIF_COMMIT}.zip"
TORCH_VERSION = "2.10.0"
UV_VERSION = "0.8.17"

# ---------------------------------------------------------------------------------- state


class _State:
    def __init__(self):
        self.lock = threading.Lock()
        self.status = "unknown"  # unknown, not_installed, stopped, starting, ready, error, installing
        self.message = ""
        self.health = {}
        self.process = None  # the Popen we started, if any
        self.port = P.DEFAULT_PORT
        self.poll_thread = None
        self.install_thread = None
        self.install_step = ""
        self.log_path = ""

    def set(self, status, message=""):
        with self.lock:
            changed = (status, message) != (self.status, self.message)
            self.status, self.message = status, message
        if changed:
            post(_status_changed)


STATE = _State()
_results: queue.Queue = queue.Queue()
_executor: ThreadPoolExecutor | None = None
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
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="openact")
    return _executor


def status():
    with STATE.lock:
        return STATE.status, STATE.message


def is_ready():
    return STATE.status == "ready"


# ----------------------------------------------------------------------------------- http


def _opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never proxy localhost


def http(method, path, body=None, port=None, timeout=10.0):
    """Blocking JSON request to the local server. Returns (status_code, json)."""
    url = f"http://127.0.0.1:{port or STATE.port}{path}"
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
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
    reason = getattr(error, "reason", error)
    return isinstance(reason, (ConnectionRefusedError, ConnectionResetError)) or isinstance(
        error, (ConnectionRefusedError, ConnectionResetError, TimeoutError)
    )


def ask(payload, on_done, timeout=180.0):
    """Send one /v1/systemone request in the background.

    on_done(kind, data) runs on the main thread; kind is "ok", "error" or "offline"."""
    port = STATE.port

    def work():
        try:
            code, body = http("POST", "/v1/systemone", payload, port=port, timeout=timeout)
            if code == 200:
                result = ("ok", body)
            elif code == 503:
                result = ("offline", body.get("error") or "model is not ready")
            else:
                result = ("error", body.get("error") or f"HTTP {code}")
        except Exception as error:
            if _offline(error):
                result = ("offline", "model server is not running")
                post(_lost_connection)
            else:
                result = ("error", f"{type(error).__name__}: {error}")
        post(on_done, *result)

    executor().submit(work)


def _lost_connection():
    if STATE.status == "ready":
        STATE.set("stopped", "The model server stopped")


# ------------------------------------------------------------------------------ lifecycle


def refresh(prefs=None):
    """Update the status from the file system and a quick /health probe (non-blocking)."""
    prefs = prefs or P.get_prefs()
    STATE.port = prefs.port
    root = P.runtime_root(prefs)
    if STATE.status in ("installing", "starting"):
        return
    if not P.runtime_installed(root):
        if STATE.status != "error":  # keep an install error visible
            STATE.set("not_installed", "The model runtime is not installed")
        return
    if STATE.status in ("unknown", "not_installed"):
        STATE.set("stopped", "")
    port = STATE.port

    def probe():
        try:
            code, health = http("GET", "/health", port=port, timeout=1.0)
        except Exception:
            if STATE.status == "ready":
                STATE.set("stopped", "The model server stopped")
            return
        if code == 200 and health.get("server") == SERVER_ID:
            _apply_health(health)
            if health.get("status") == "loading":
                _start_polling(None)

    executor().submit(probe)


def _apply_health(health):
    STATE.health = health
    state = health.get("status")
    if state == "ready":
        STATE.set("ready", f"{health.get('source', '')} on {health.get('device') or 'unknown device'}")
    elif state == "loading":
        STATE.set("starting", "Loading the model...")
    elif state == "error":
        STATE.set("error", health.get("error") or "The model failed to load")


def start(prefs=None):
    """Start the model server (or attach to one that is already running). Main thread only."""
    prefs = prefs or P.get_prefs()
    if STATE.status in ("starting", "ready", "installing"):
        return True
    STATE.port = prefs.port
    root = P.runtime_root(prefs)
    if not P.runtime_installed(root):
        STATE.set("not_installed", "The model runtime is not installed")
        return False
    try:
        code, health = http("GET", "/health", timeout=0.5)
        if code == 200 and health.get("server") == SERVER_ID:
            _apply_health(health)
            if health.get("status") == "loading":
                _start_polling(None)
            return True
        if code == 200 and health.get("server") == LEGACY_SERVER_ID:
            STATE.set("error", LEGACY_SERVER_MESSAGE)
        else:
            STATE.set("error", f"Port {STATE.port} is used by another program. Pick another port in the preferences")
        return False
    except Exception:
        pass  # nothing is listening: start our own

    python = P.venv_python(root)
    STATE.log_path = os.path.join(root, "server.log")
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(key, None)
    env.update({
        "HF_HOME": P.hf_home(root),
        "HF_HUB_OFFLINE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    })
    args = [python, "-u", SERVER_SCRIPT, "--port", str(STATE.port), "--device", prefs.device,
            "--parent-pid", str(os.getpid())]
    if prefs.device != "cpu":
        args += ["--gpu", str(prefs.gpu)]
    try:
        log = open(STATE.log_path, "w", encoding="utf-8")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
        STATE.process = subprocess.Popen(
            args, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            cwd=root, env=env, creationflags=flags,
        )
        log.close()  # the child keeps its own handle
    except OSError as error:
        STATE.set("error", f"Could not start the model server: {error}")
        return False
    STATE.set("starting", "Starting the model (the first start takes about 20 seconds)...")
    _start_polling(STATE.process)
    return True


def _start_polling(process):
    if STATE.poll_thread and STATE.poll_thread.is_alive():
        return
    port = STATE.port

    def poll():
        deadline = time.time() + 600
        while time.time() < deadline:
            if process is not None and process.poll() is not None:
                STATE.set("error", "The model server exited: " + _log_tail())
                return
            try:
                code, health = http("GET", "/health", port=port, timeout=1.0)
                if code == 200:
                    _apply_health(health)
                    if health.get("status") in ("ready", "error"):
                        if health.get("status") == "ready":
                            post(_became_ready)
                        return
            except Exception:
                pass
            time.sleep(0.5)
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


def stop():
    """Stop the server (ours, or one another Blender started on the same port)."""
    try:
        http("POST", "/shutdown", body={}, timeout=2.0)
    except Exception:
        pass
    process = STATE.process
    if process is not None:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        STATE.process = None
    STATE.health = {}
    STATE.set("stopped", "")


def shutdown_owned():
    """On add-on unregister: stop the server only if this Blender started it."""
    if STATE.process is not None and STATE.process.poll() is None:
        stop()
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=False, cancel_futures=True)
        _executor = None


# ------------------------------------------------------------------------------ installer


def _uv_asset():
    machine = platform.machine().lower()
    arch = "aarch64" if machine in ("arm64", "aarch64") else "x86_64"
    if sys.platform == "win32":
        return f"uv-{arch}-pc-windows-msvc.zip"
    if sys.platform == "darwin":
        return f"uv-{arch}-apple-darwin.tar.gz"
    return f"uv-{arch}-unknown-linux-gnu.tar.gz"


def _has_nvidia():
    tool = shutil.which("nvidia-smi")
    if not tool:
        return False
    try:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
        return subprocess.run([tool, "-L"], capture_output=True, timeout=20, creationflags=flags).returncode == 0
    except Exception:
        return False


def install(prefs=None):
    """Download and set up the runtime in the background (about 12 GB). Main thread only."""
    prefs = prefs or P.get_prefs()
    if STATE.install_thread and STATE.install_thread.is_alive():
        return False
    root = P.runtime_root(prefs)
    STATE.set("installing", "Preparing...")

    def run(cmd, log, env=None):
        log.write("\n$ " + " ".join(cmd) + "\n")
        log.flush()
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                env=env, creationflags=flags)
        if result.returncode != 0:
            raise RuntimeError(f"'{os.path.basename(cmd[0])}' failed with code {result.returncode} (see install.log)")

    def step(text):
        STATE.install_step = text
        STATE.set("installing", text)

    def work():
        os.makedirs(root, exist_ok=True)
        log_path = os.path.join(root, "install.log")
        STATE.log_path = log_path
        try:
            with open(log_path, "a", encoding="utf-8") as log:
                log.write(f"\n===== OpenAct runtime install {time.ctime()} =====\n")
                env = dict(os.environ)
                for key in ("PYTHONPATH", "PYTHONHOME"):
                    env.pop(key, None)
                env["HF_HOME"] = P.hf_home(root)
                env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
                # Keep everything inside the runtime folder: uv's own Python and its download cache.
                env["UV_PYTHON_INSTALL_DIR"] = os.path.join(root, "python")
                env["UV_PYTHON_PREFERENCE"] = "only-managed"
                env["UV_CACHE_DIR"] = os.path.join(root, "uv-cache")
                python = P.venv_python(root)

                step("1/5 Getting uv (Python package manager)...")
                exe = "uv.exe" if sys.platform == "win32" else "uv"
                uv = shutil.which("uv") or os.path.join(root, "tools", exe)
                if not os.path.isfile(uv):
                    asset = _uv_asset()
                    url = f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/{asset}"
                    archive = os.path.join(root, "tools", asset)
                    os.makedirs(os.path.dirname(archive), exist_ok=True)
                    log.write(f"downloading {url}\n")
                    urllib.request.urlretrieve(url, archive)
                    if archive.endswith(".zip"):
                        with zipfile.ZipFile(archive) as zf:
                            member = next(n for n in zf.namelist() if n.endswith("/" + exe) or n == exe)
                            with zf.open(member) as src, open(uv, "wb") as dst:
                                shutil.copyfileobj(src, dst)
                    else:
                        with tarfile.open(archive) as tf:
                            member = next(m for m in tf.getmembers() if m.name.endswith("/" + exe) or m.name == exe)
                            with tf.extractfile(member) as src, open(uv, "wb") as dst:
                                shutil.copyfileobj(src, dst)
                        os.chmod(uv, 0o755)

                step("2/5 Creating the Python environment...")
                if not os.path.isfile(python):
                    run([uv, "venv", "--python", "3.12", os.path.join(root, "venv")], log, env)

                def importable(module):
                    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
                    return subprocess.run([python, "-c", f"import {module}"], capture_output=True, env=env,
                                          creationflags=flags).returncode == 0

                step("3/5 Installing PyTorch (about 3 GB)...")
                if not importable("torch"):
                    cmd = [uv, "pip", "install", "--python", python, f"torch=={TORCH_VERSION}"]
                    if sys.platform != "darwin":
                        index = "cu128" if _has_nvidia() else "cpu"
                        cmd += ["--index-url", f"https://download.pytorch.org/whl/{index}"]
                    run(cmd, log, env)

                step("4/5 Installing SemIf...")
                if not importable("semif_phase1"):
                    run([uv, "pip", "install", "--python", python, f"semif-phase1 @ {SEMIF_URL}"], log, env)

                step("5/5 Downloading the model (8.7 GB)...")
                if not P.runtime_installed(root):  # resumes an interrupted download
                    code = ("from huggingface_hub import snapshot_download; "
                            f"print(snapshot_download(repo_id={P.MODEL_SOURCE!r}, revision={P.MODEL_REVISION!r}))")
                    run([python, "-c", code], log, env)
                if not P.runtime_installed(root):
                    raise RuntimeError("the model files are incomplete after the download")
                try:  # the venv holds its own copies; the download cache is not needed any more
                    run([uv, "cache", "clean"], log, env)
                except RuntimeError:
                    pass
                log.write("===== done =====\n")
            STATE.install_step = ""
            STATE.set("stopped", "Runtime installed")
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
