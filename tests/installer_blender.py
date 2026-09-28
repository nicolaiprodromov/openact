"""The runtime installer and the runtime probe, in a headless Blender. Nothing is downloaded or installed.

Every runtime is a temporary folder, and so is OPENACT_STATE_DIR. Downloads, nvidia-smi, the free disk space and
every subprocess are fakes; the installed JevNodes runtime is only read (design section 20.6):
  - the lock pins every package with hashes;
  - the dry run logs the pinned cu128 commands with the runtime's own uv, even with another uv on PATH;
  - refused before any download: online access off (also through the Install operator), a runtime folder whose path
    has a '#', less than 15 GB free, no nvidia-smi; a runtime folder that can't be created fails the install;
  - an 8 GB GPU gets a warning, and a full install (fake uv, fake SemIf) ends with it; uv gets no UV_* setting of
    the user's;
  - a download whose SHA-256 differs (uv, then SemIf) stops the install with a clear status;
  - an existing venv that fails the probe is rebuilt before the lock goes in; one that passes is kept;
  - runtime_installed() rejects a legacy runtime whose venv home is missing (a copy of its pyvenv.cfg);
  - the import probe, in a runtime folder with a '#': register() returns at once while it runs, a start waits for
    it, the next session reads probe.json without a subprocess, a changed venv probes again, a failure cancels the
    start;
  - a probe's late result is dropped once its folder was left, an install began or the add-on was re-registered;
  - Locate Existing Runtime sets the runtime folder and probes it in the background.

Run:  blender -b --factory-startup --python tests/installer_blender.py
"""

import collections
import hashlib
import io
import json
import os
import pathlib
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.request
import zipfile

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import check, drive, finish  # noqa: E402

import openact  # noqa: E402  (from this checkout, which _harness puts on sys.path; not registered yet)
from openact import engine, server  # noqa: E402
from openact import prefs as P  # noqa: E402

JEVNODES = os.path.join(os.path.dirname(P.default_runtime_root()), "JevNodes")  # the installed one: read only
BASE = tempfile.mkdtemp(prefix="openact_installer_")
os.environ["LOCALAPPDATA" if sys.platform == "win32" else "XDG_DATA_HOME"] = BASE
os.environ["OPENACT_STATE_DIR"] = os.path.join(BASE, "state")
for name in ("OPENACT_BACKEND", "OPENACT_SERVER_PYTHON"):
    os.environ.pop(name, None)
EXE = "uv.exe" if sys.platform == "win32" else "uv"
ASSET = server._uv_asset()  # before the fakes: platform.uname() may run a subprocess once, then caches


class Fake:
    """The machine as the add-on sees it."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.files = {}  # url -> the bytes its download writes; other urls get bytes that match no hash
        self.downloads = []
        self.runs = []  # every subprocess.run command
        self.uv_envs = []  # the environment of every uv command
        self.nvidia_smi = "nvidia-smi"  # what shutil.which finds: None on a machine without it
        self.gpus = "NVIDIA GeForce RTX 3090, 24576\n"
        self.free = 100 * 1024 ** 3
        self.probe_seconds, self.probe_code = 0.0, 1  # the import probe's duration and exit code
        self.probes = []  # (ran on the main thread, seconds) per import probe


FAKE = Fake()
REAL_WHICH = shutil.which
Usage = collections.namedtuple("Usage", "total used free")


def fake_urlretrieve(url, filename, *_args, **_kwargs):
    FAKE.downloads.append(url)
    with open(filename, "wb") as handle:
        handle.write(FAKE.files.get(url, b"not the pinned file"))
    return filename, None


def fake_which(cmd, *args, **kwargs):
    return FAKE.nvidia_smi if cmd == "nvidia-smi" else REAL_WHICH(cmd, *args, **kwargs)


def fake_run(args, **kwargs):
    """nvidia-smi answers from FAKE.gpus, the import probe from FAKE.probe_* as they were when it started. uv venv
    and the model download leave the files they would; every other command succeeds without doing anything."""
    seconds, code = FAKE.probe_seconds, FAKE.probe_code
    FAKE.runs.append(list(args))
    if os.path.basename(args[0]) == EXE:
        FAKE.uv_envs.append(kwargs.get("env") or {})
    if "--query-gpu=name,memory.total" in args:
        return subprocess.CompletedProcess(args, 0, FAKE.gpus.encode(), b"")
    if server.PROBE_CODE in args:
        started = time.perf_counter()
        time.sleep(seconds)
        FAKE.probes.append((threading.current_thread() is threading.main_thread(), time.perf_counter() - started))
        return subprocess.CompletedProcess(args, code, b"", b"")
    if args[1:2] == ["venv"]:
        fake_runtime(os.path.dirname(args[-1]), model=False)
    elif "snapshot_download" in args[-1]:
        fake_runtime(os.path.dirname(os.path.dirname(os.path.dirname(args[0]))), venv=False)
    return subprocess.CompletedProcess(args, 0, b"", b"")


class FakeProcess:
    """Replaces subprocess.Popen: records each launch of the model server, which has already exited."""

    launched = []

    def __init__(self, args, **_kwargs):
        FakeProcess.launched.append(args)

    def poll(self):
        return 1

    def wait(self, timeout=None):
        return 1


urllib.request.urlretrieve, shutil.which, subprocess.run, subprocess.Popen = (
    fake_urlretrieve, fake_which, fake_run, FakeProcess)
shutil.disk_usage = lambda _path: Usage(2 * FAKE.free, FAKE.free, FAKE.free)


def fake_runtime(root, venv=True, model=True):
    """What runtime_installed() and the probe key read: the venv's Python, a pyvenv.cfg whose home exists, torch's
    dist-info, and a one-shard model."""
    if venv:
        os.makedirs(os.path.dirname(P.venv_python(root)), exist_ok=True)
        open(P.venv_python(root), "wb").close()
        with open(os.path.join(root, "venv", "pyvenv.cfg"), "w", encoding="utf-8") as handle:
            handle.write(f"home = {BASE}\n")
        site = ("Lib", "site-packages") if sys.platform == "win32" else ("lib", "python3.12", "site-packages")
        os.makedirs(os.path.join(root, "venv", *site, "torch-2.10.0+cu128.dist-info"), exist_ok=True)
    if model:
        folder = P.model_dir(root)
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "model.safetensors.index.json"), "w", encoding="utf-8") as handle:
            json.dump({"weight_map": {"w": "model-1.safetensors"}}, handle)
        with open(os.path.join(folder, "model-1.safetensors"), "wb") as handle:
            handle.write(b"x")


def fake_uv_archive(asset):
    """An archive laid out like the uv release's, holding a fake uv."""
    buffer = io.BytesIO()
    if asset.endswith(".zip"):
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(EXE, b"fake uv")
    else:
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            member = tarfile.TarInfo(asset[:-len(".tar.gz")] + "/" + EXE)
            member.size = len(b"fake uv")
            archive.addfile(member, io.BytesIO(b"fake uv"))
    return buffer.getvalue()


def snapshot(root):
    """Names, sizes and mtimes of a runtime's top level and venv folder, read without writing anything."""
    found = {}
    for folder in (root, os.path.join(root, "venv")):
        try:
            for entry in os.scandir(folder):
                info = entry.stat()
                found[entry.path] = (info.st_size, info.st_mtime_ns)
        except OSError:
            pass
    return found


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def install(name, dry_run=False):
    """Run the installer into a new folder under BASE and wait for it: (root, started, install.log)."""
    FAKE.runs.clear()
    root = os.path.join(BASE, name)
    P._Defaults.runtime_dir = root
    started = server.install(dry_run=dry_run)
    if started:
        server.STATE.install_thread.join(60)
    try:
        with open(os.path.join(root, "install.log"), encoding="utf-8") as handle:
            return root, started, handle.read()
    except OSError:
        return root, started, ""


def in_order(text, parts):
    """The first of parts that is not found in text after the previous one, or None."""
    position = 0
    for part in parts:
        found = text.find(part, position)
        if found < 0:
            return part
        position = found + len(part)
    return None


def phase_lock():
    requirements, name = {}, None
    with open(server.LOCK_FILE, encoding="utf-8") as handle:
        for line in handle:
            if line[:1].isalnum():
                name = line.split(";")[0].rstrip(" \\\n")
                requirements[name] = 0
            elif name and line.strip().startswith("--hash=sha256:"):
                requirements[name] += 1
    bad = [r for r, hashes in requirements.items() if "==" not in r or not hashes]
    check("lock: every package is pinned with hashes", len(requirements) > 30 and not bad, bad or len(requirements))
    check("lock: torch 2.10.0 for CUDA 12.8", "torch==2.10.0+cu128" in requirements)
    check("lock: not SemIf itself, which comes from its checked archive",
          not [r for r in requirements if r.startswith("semif")])


def phase_online_off():
    FAKE.reset()
    bpy.context.preferences.system.use_online_access = False
    root, started, _log = install("offline")
    state = server.status()
    check("online access off: the install refuses",
          not started and state[0] == "error" and state[1].startswith("Online access is disabled"), state)
    check("online access off: nothing is downloaded or created", not FAKE.downloads and not os.path.exists(root))
    bpy.context.preferences.system.use_online_access = True


def phase_dry_run():
    FAKE.reset()
    path_uv = os.path.join(BASE, "bin", EXE)
    os.makedirs(os.path.dirname(path_uv))
    open(path_uv, "wb").close()
    os.chmod(path_uv, 0o755)
    saved_path = os.environ["PATH"]
    os.environ["PATH"] = os.path.dirname(path_uv) + os.pathsep + saved_path
    found = shutil.which("uv")
    check("setup: another uv is first on PATH", found and os.path.samefile(found, path_uv), found)
    before = server.status()
    root, started, log = install("dry", dry_run=True)
    os.environ["PATH"] = saved_path
    print("dry run install.log:" + log.replace("\n", "\n    "), flush=True)

    asset = ASSET
    uv, python = os.path.join(root, "tools", EXE), P.venv_python(root)
    semif = pathlib.Path(root, "tools", f"SemIf-OpenJev-{server.SEMIF_COMMIT}.zip").as_uri()
    expected = [
        "===== OpenAct runtime install (dry run)",
        f"downloading https://github.com/astral-sh/uv/releases/download/{server.UV_VERSION}/{asset} "
        f"(SHA-256 {server.UV_SHA256[asset]})",
        f"$ {uv} venv --clear --python 3.12 {os.path.join(root, 'venv')}\n",
        f"$ {uv} pip install --python {python} --require-hashes --no-build --torch-backend cu128 "
        f"-r {server.LOCK_FILE}\n",
        f"downloading {server.SEMIF_URL} (SHA-256 {server.SEMIF_SHA256})",
        f"$ {uv} pip install --python {python} --no-deps --no-build-isolation semif-phase1 @ {semif}\n",
        f"$ {python} -c from huggingface_hub import snapshot_download; ",
        f"revision={P.MODEL_REVISION!r}",
        f"$ {uv} cache clean\n",
    ]
    missing = in_order(log, expected)
    check("dry run: logs the pinned cu128 commands, in order", started and missing is None, missing)
    check("dry run: no CPU wheels", "/whl/cpu" not in log and "--index-url" not in log)
    check("dry run: the uv on PATH is not used, only the runtime's own",
          path_uv not in log and log.count(f"$ {uv} ") == 4, log.count(f"$ {uv} "))
    check("dry run: nothing is downloaded, run or created but install.log",
          not FAKE.downloads and not FAKE.probes and os.listdir(root) == ["install.log"]
          and all("--query-gpu=name,memory.total" in run for run in FAKE.runs), (FAKE.downloads, FAKE.runs))
    check("dry run: the status is left as it was", server.status() == before, server.status())


def phase_refusals():
    FAKE.reset()
    root, started, log = install("C# tools")
    state = server.status()
    check("path: a runtime folder whose path has a '#' is refused before anything runs or downloads",
          started and state == ("error", f"Install failed: the runtime folder {root} contains '#', which uv can't "
                                         "install from: choose another Runtime Folder")
          and not FAKE.downloads and not FAKE.runs, (state, FAKE.runs))

    FAKE.reset()
    open(os.path.join(BASE, "a file"), "wb").close()
    _root, started, log = install(os.path.join("a file", "OpenAct"))
    state = server.status()
    check("path: a runtime folder that can't be created fails the install with its error",
          started and state[0] == "error" and state[1].startswith("Install failed: ") and not FAKE.downloads, state)

    FAKE.reset()
    FAKE.free = 10 * 1024 ** 3
    _root, started, log = install("small-disk")
    state = server.status()
    check("disk: less than 15 GB free is refused before any download",
          started and state[0] == "error" and "not enough disk space" in state[1] and "15 GB" in state[1]
          and not FAKE.downloads and not FAKE.runs, (state, FAKE.runs))

    FAKE.reset()
    FAKE.nvidia_smi = None
    _root, started, log = install("no-nvidia")
    state = server.status()
    check("GPU: a machine without nvidia-smi is refused before any download",
          state == ("error", "Install failed: OpenAct 1.0 needs an NVIDIA GPU") and not FAKE.downloads, state)

    FAKE.reset()
    FAKE.gpus = "NVIDIA GeForce RTX 3070, 8192\n"
    _root, started, log = install("8gb-dry", dry_run=True)
    check("GPU: an 8 GB GPU gets the warning, and the install goes on",
          in_order(log, ["Warning: the GPU has 8 GB of VRAM", "cache clean"]) is None,
          [line for line in log.splitlines() if line.startswith("Warning")])


def phase_hashes():
    asset = ASSET
    uv_url = f"https://github.com/astral-sh/uv/releases/download/{server.UV_VERSION}/{asset}"
    semif_name = f"SemIf-OpenJev-{server.SEMIF_COMMIT}.zip"

    FAKE.reset()
    root, started, log = install("bad-uv")
    state = server.status()
    check("hash: a uv download with another SHA-256 stops the install with a clear status",
          state == ("error", f"Install failed: {asset} does not match its recorded SHA-256: update OpenAct")
          and FAKE.downloads == [uv_url], state)
    check("hash: ...before anything else runs, and the download is deleted",
          all("--query-gpu=name,memory.total" in run for run in FAKE.runs)
          and not os.path.exists(os.path.join(root, "tools", asset))
          and not os.path.exists(os.path.join(root, "tools", EXE)), FAKE.runs)

    archive = fake_uv_archive(asset)
    real_uv_sha, real_semif_sha = server.UV_SHA256[asset], server.SEMIF_SHA256
    server.UV_SHA256[asset] = hashlib.sha256(archive).hexdigest()  # the fake uv passes its check
    try:
        FAKE.reset()
        FAKE.files[uv_url] = archive
        root, started, log = install("bad-semif")
        state = server.status()
        ran = [" ".join(run[1:3]) for run in FAKE.runs]
        check("hash: a SemIf download with another SHA-256 stops the install with a clear status",
              state == ("error", f"Install failed: {semif_name} does not match its recorded SHA-256: update OpenAct")
              and FAKE.downloads == [uv_url, server.SEMIF_URL], state)
        check("hash: ...after the lock, before SemIf is installed, and the download is deleted",
              "pip install" in ran and not any("semif-phase1 @" in " ".join(run) for run in FAKE.runs)
              and not any("snapshot_download" in run[-1] for run in FAKE.runs)
              and not os.path.exists(os.path.join(root, "tools", semif_name)), ran)

        FAKE.reset()
        FAKE.files[uv_url] = archive
        FAKE.files[server.SEMIF_URL] = b"fake SemIf archive"
        server.SEMIF_SHA256 = hashlib.sha256(b"fake SemIf archive").hexdigest()
        FAKE.gpus = "NVIDIA GeForce RTX 3070, 8192\n"
        os.environ["UV_LINK_MODE"] = "symlink"  # would link the venv into the uv-cache that the install deletes
        root, started, log = install("full")
        os.environ.pop("UV_LINK_MODE")
        state = server.status()
        with open(os.path.join(root, "tools", EXE), "rb") as handle:
            extracted = handle.read()
        python = P.venv_python(root)
        commands = [" ".join(run) for run in FAKE.runs if "--query-gpu=name,memory.total" not in run]
        check("install: a checked uv archive is unpacked into the runtime's tools folder", extracted == b"fake uv")
        check("install: every step runs, with the lock and the checked SemIf archive",
              in_order("\n".join(commands), [
                  f"{python} -c {server.PROBE_CODE}", "venv --clear --python 3.12",
                  f"--require-hashes --no-build --torch-backend cu128 -r {server.LOCK_FILE}",
                  "--no-deps --no-build-isolation semif-phase1 @ file:", "snapshot_download", "cache clean"]) is None,
              commands)
        check("install: uv runs without the user's UV_* variables and uv.toml",
              len(FAKE.uv_envs) == 4 and all(env.get("UV_NO_CONFIG") == "1" and "UV_LINK_MODE" not in env
                                             for env in FAKE.uv_envs),
              [(env.get("UV_NO_CONFIG"), env.get("UV_LINK_MODE")) for env in FAKE.uv_envs])
        check("install: an 8 GB GPU's warning stays in the final status, and in STATE.install_warning, which the "
              "Model box keeps after the status moves on",
              state[0] == "stopped" and state[1].startswith("Runtime installed. Warning: the GPU has 8 GB of VRAM")
              and server.STATE.install_warning.startswith("Warning: the GPU has 8 GB of VRAM")
              and P.runtime_installed(root), (state, server.STATE.install_warning))

        FAKE.reset()
        FAKE.files[uv_url] = archive
        FAKE.files[server.SEMIF_URL] = b"fake SemIf archive"
        fake_runtime(os.path.join(BASE, "broken"), model=False)  # a venv and its home, but its imports fail
        root, started, log = install("broken")
        commands = [" ".join(run) for run in FAKE.runs if "--query-gpu=name,memory.total" not in run]
        check("install: an existing venv that fails the probe is rebuilt before the lock goes in",
              in_order("\n".join(commands), [
                  f"{P.venv_python(root)} -c {server.PROBE_CODE}", "venv --clear --python 3.12",
                  "--require-hashes", "semif-phase1 @ file:"]) is None and server.status()[0] == "stopped", commands)

        FAKE.reset()
        FAKE.files[uv_url] = archive
        FAKE.probe_code = 0
        fake_runtime(os.path.join(BASE, "resumed"), model=False)  # its imports work; the model download stopped
        root, started, log = install("resumed")
        check("install: an existing venv that passes the probe is kept, and only the model is downloaded",
              [run[1] for run in FAKE.runs if os.path.basename(run[0]) == EXE] == ["cache"]
              and any("snapshot_download" in run[-1] for run in FAKE.runs) and FAKE.downloads == [uv_url]
              and server.status()[0] == "stopped", FAKE.runs)
    finally:
        server.UV_SHA256[asset], server.SEMIF_SHA256 = real_uv_sha, real_semif_sha


def phase_legacy_home():
    legacy, new = os.path.join(BASE, "JevNodes"), os.path.join(BASE, "OpenAct")
    fake_runtime(legacy)
    copied = os.path.join(JEVNODES, "venv", "pyvenv.cfg")
    try:
        with open(copied, encoding="utf-8") as handle:
            text = handle.read()
    except OSError:  # no Jev Nodes runtime on this machine: the same keys
        copied, text = "(written)", "home = C:\\x\nimplementation = CPython\nuv = 0.8.17\nversion_info = 3.12.11\n"
    missing = os.path.join(BASE, "uv", "python", "cpython-3.12.11-removed")
    with open(os.path.join(legacy, "venv", "pyvenv.cfg"), "w", encoding="utf-8") as handle:
        handle.write(re.sub(r"(?m)^home\s*=.*$", lambda _match: f"home = {missing}", text))
    P._Defaults.runtime_dir = ""
    check("legacy: a runtime whose venv home is missing is not installed", not P.runtime_installed(legacy), copied)
    check("legacy: runtime_root() passes over it to a new install", P.runtime_root() == new, P.runtime_root())
    os.makedirs(missing)
    check("legacy: with its home back it is used in place",
          P.runtime_installed(legacy) and P.runtime_root() == legacy, P.runtime_root())
    shutil.rmtree(legacy)


def phase_probe():
    root = os.path.join(BASE, "Runtime #1")  # only installing refuses a '#'
    fake_runtime(root)
    P._Defaults.runtime_dir, P._Defaults.port, P._Defaults.auto_start = root, free_port(), True
    cached = os.path.join(os.environ["OPENACT_STATE_DIR"], "probe.json")
    checking = ("unknown", server.CHECKING)
    FAKE.reset()
    FAKE.probe_seconds, FAKE.probe_code = 5.0, 0

    started = time.perf_counter()
    openact.register()
    seconds = time.perf_counter() - started
    check("probe: register() returns in < 50 ms while the probe sleeps 5 s", seconds < 0.05, f"{seconds * 1000:.1f} ms")
    check("probe: meanwhile the status reads 'Checking the runtime...'", server.status() == checking, server.status())
    started = time.perf_counter()
    queued = server.start()
    seconds = time.perf_counter() - started
    check("probe: a start meanwhile returns at once and waits", queued and seconds < 0.05 and not FakeProcess.launched,
          f"{seconds * 1000:.1f} ms")
    drive(engine, lambda: FakeProcess.launched, timeout=20)
    check("probe: ran once, off the main thread, for the fake's 5 s",
          len(FAKE.probes) == 1 and not FAKE.probes[0][0] and FAKE.probes[0][1] >= 5.0, FAKE.probes)
    check("probe: the queued start runs once it passed",
          len(FakeProcess.launched) == 1 and server.SERVER_SCRIPT in FakeProcess.launched[0], FakeProcess.launched)
    check("probe: the pass is cached in OPENACT_STATE_DIR, not in the runtime",
          os.path.isfile(cached) and not os.path.exists(os.path.join(root, "probe.json")))

    def session():
        """Unregister and register again, like a new Blender session, and wait for the probe's result."""
        server.STATE.poll_thread.join(5)  # the fake server has exited
        openact.unregister()
        openact.register()
        status = server.status()
        drive(engine, lambda: server.status()[0] != "unknown", timeout=20)
        return status

    FAKE.probe_seconds = 0.0
    runs = len(FAKE.runs)
    status = session()
    check("second session: probe.json is read, no subprocess starts",
          status == checking and server.status() == ("stopped", "") and len(FAKE.runs) == runs,
          (status, server.status(), FAKE.runs[runs:]))

    cfg = os.path.join(root, "venv", "pyvenv.cfg")
    later = os.path.getmtime(cfg) + 60
    os.utime(cfg, (later, later))
    session()
    check("a changed pyvenv.cfg: the probe runs again", len(FAKE.probes) == 2 and server.status() == ("stopped", ""),
          (FAKE.probes, server.status()))

    FAKE.probe_code = 1
    os.remove(cached)
    server.STATE.poll_thread.join(5)
    openact.unregister()
    openact.register()
    queued = server.start()
    drive(engine, lambda: server.status()[0] != "unknown", timeout=20)
    failed = ("error", f"The runtime's Python at {BASE} is missing or broken (was it removed with `uv python "
                       "uninstall`?): reinstall the runtime")
    check("failed probe: the queued start is cancelled with its message",
          queued and server.status() == failed and len(FakeProcess.launched) == 1, server.status())
    check("failed probe: nothing is cached", not os.path.exists(cached))
    server.refresh()
    check("failed probe: refresh() keeps the message and does not probe again",
          server.status() == failed and len(FAKE.probes) == 3, (server.status(), FAKE.probes))
    FAKE.probe_code = 0
    server.start()
    drive(engine, lambda: len(FakeProcess.launched) == 2, timeout=20)
    check("failed probe: Start Model probes again and starts once it passes",
          len(FAKE.probes) == 4 and len(FakeProcess.launched) == 2, (FAKE.probes, FakeProcess.launched))


def late_failing_probe(root):
    """Choose root and start its probe, which fails after 2 s. Returns an event that is set once it has posted."""
    posted, real = threading.Event(), server._probe

    def probe(*args):
        real(*args)
        posted.set()

    FAKE.probe_seconds, FAKE.probe_code, runs = 2.0, 1, len(FAKE.runs)
    P._Defaults.runtime_dir, server._probe = root, probe
    server.refresh()
    server._probe = real
    drive(engine, lambda: any(server.PROBE_CODE in run for run in FAKE.runs[runs:]), timeout=20)
    FAKE.probe_seconds, FAKE.probe_code = 0.0, 0  # the running probe keeps what it started with
    return posted


def phase_late_probe():
    passed, left, repaired, away, reinstalled = (os.path.join(BASE, name)
                                                 for name in ("Passed", "Left", "Repaired", "Away", "Reinstalled"))
    for root in (passed, left, repaired, away, reinstalled):
        fake_runtime(root)
    server.STATE.poll_thread.join(5)
    openact.unregister()
    P._Defaults.runtime_dir, P._Defaults.auto_start = passed, False
    FAKE.probe_seconds, FAKE.probe_code = 0.0, 0
    openact.register()
    drive(engine, lambda: server.status() == ("stopped", ""), timeout=20)

    posted = late_failing_probe(left)
    P._Defaults.runtime_dir = passed  # back to a folder whose probe passed: nothing to wait for
    server.refresh()
    posted.wait(20)
    server.pump()
    server.refresh()
    check("late probe: the failure of a folder left meanwhile leaves the status alone and is not stored",
          server.status() == ("stopped", "") and left not in server.STATE.probes,
          (server.status(), server.STATE.probes))

    posted = late_failing_probe(repaired)
    openact.unregister()  # the probe runs on: it can't be cancelled any more
    openact.register()  # the runtime was repaired meanwhile, so this session's probe passes
    drive(engine, lambda: server.STATE.probes.get(repaired) == "", timeout=20)
    posted.wait(20)
    server.pump()
    server.refresh()
    check("late probe: a failure from before re-registering doesn't replace this session's pass",
          server.status() == ("stopped", "") and server.STATE.probes.get(repaired) == "",
          (server.status(), server.STATE.probes))

    posted = late_failing_probe(away)
    openact.unregister()
    index = os.path.join(P.model_dir(away), "model.safetensors.index.json")
    os.rename(index, index + ".away")  # unavailable while the add-on registers again, so this session doesn't probe
    openact.register()
    posted.wait(20)
    server.pump()
    unavailable = server.status()
    os.rename(index + ".away", index)  # back, and repaired
    server.refresh()
    drive(engine, lambda: server.status()[0] != "unknown", timeout=20)
    check("late probe: ...nor sticks to a runtime that was unavailable meanwhile: it is probed again",
          unavailable[0] == "not_installed" and server.status() == ("stopped", "")
          and server.STATE.probes.get(away) == "", (unavailable, server.status(), server.STATE.probes))

    posted = late_failing_probe(reinstalled)
    FAKE.files.clear()  # the install fails at once: its uv download doesn't match the SHA-256
    started = server.install()
    server.STATE.install_thread.join(60)
    failed = server.status()
    posted.wait(20)
    server.pump()
    check("late probe: an install begun meanwhile drops the probe of the folder's old runtime",
          started and failed[1].startswith("Install failed: ") and server.status() == failed
          and reinstalled not in server.STATE.probes, (failed, server.status(), server.STATE.probes))


def phase_locate():
    import addon_utils

    server.STATE.poll_thread.join(5)
    openact.unregister()
    addon_utils.enable("openact", default_set=True)  # real preferences, which the operator writes
    prefs = P.get_prefs()
    prefs.port = free_port()
    located = os.path.join(BASE, "Located")
    fake_runtime(located)
    try:
        bpy.ops.openact.locate_runtime('EXEC_DEFAULT', directory=BASE)
        refused = False
    except RuntimeError:
        refused = True
    check("locate: a folder without a runtime is refused and not kept", refused and prefs.runtime_dir == "",
          prefs.runtime_dir)
    FAKE.probe_seconds = 1.0
    probes = len(FAKE.probes)
    result = bpy.ops.openact.locate_runtime('EXEC_DEFAULT', directory=located)
    status = server.status()
    drive(engine, lambda: server.status()[0] != "unknown", timeout=20)
    check("locate: sets the runtime folder and probes it in the background",
          result == {'FINISHED'} and prefs.runtime_dir == located and status == ("unknown", server.CHECKING)
          and len(FAKE.probes) == probes + 1 and server.STATE.probes.get(located) == "",
          (result, prefs.runtime_dir, status, FAKE.probes[probes:]))


def phase_install_operator():
    """The add-on is enabled (phase_locate): the operator reports why install() refused."""
    bpy.context.preferences.system.use_online_access = False
    try:
        bpy.ops.openact.install_runtime('EXEC_DEFAULT')
        reported = ""
    except RuntimeError as error:  # how bpy.ops passes on an ERROR report
        reported = str(error)
    bpy.context.preferences.system.use_online_access = True
    check("install operator: with online access off it reports why", "Online access is disabled" in reported,
          reported)


def main():
    before = snapshot(JEVNODES)
    P._Defaults.auto_start = False  # a finished install must not start anything here
    phase_lock()
    phase_online_off()
    phase_dry_run()
    phase_refusals()
    phase_hashes()
    server.pump()  # the installer's posted callbacks, before the add-on registers
    phase_legacy_home()
    phase_probe()
    phase_late_probe()
    phase_locate()
    phase_install_operator()
    check("the installed JevNodes runtime is untouched", snapshot(JEVNODES) == before, f"{len(before)} entries")
    finish()


main()
