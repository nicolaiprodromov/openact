"""OpenAct's handling of what jev_nodes left behind, in a headless Blender. No model or runtime needed.

Blender 4.5 and newer:
  - runtime_root() order: the Runtime Folder preference, an installed OpenAct runtime, an installed
    Jev Nodes runtime (used in place), then OpenAct for a new install;
  - an old jev-nodes-semif server on the port gives its error message and nothing is launched; once
    that server has gone, or the port changed, the status clears and auto-start tries again.
Older Blender: enabling the add-on fails before anything is registered.

Run:  blender -b --factory-startup --python tests/legacy_blender.py
"""

import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import check, finish  # noqa: E402


def phase_old_blender():
    import addon_utils

    errors = []
    module = addon_utils.enable("openact", default_set=False, handle_error=errors.append)
    check("register() refuses this Blender",
          module is None and [str(error) for error in errors] == ["OpenAct needs Blender 4.5 or newer"], errors)
    registered = (bpy.types.AddonPreferences.bl_rna_get_subclass_py("OpenActPreferences"),
                  bpy.types.Node.bl_rna_get_subclass_py("GeometryNodeOpenActAsk"),
                  bpy.types.Panel.bl_rna_get_subclass_py("NODE_PT_openact"))
    check("nothing was registered", registered == (None, None, None), registered)


def load_package():
    """Import the add-on without enabling it: only the node types are registered, so no timer, handler or
    /health probe runs, and prefs.get_prefs() returns the _Defaults that the test sets."""
    from openact import engine, nodes, server
    from openact import prefs as P

    nodes.register()
    return engine, server, P


def fake_runtime(P, root, model=True):
    """Just enough for runtime_installed(): the venv's Python, a pyvenv.cfg whose home exists and, optionally, a
    one-shard model."""
    os.makedirs(os.path.dirname(P.venv_python(root)), exist_ok=True)
    open(P.venv_python(root), "wb").close()
    with open(os.path.join(root, "venv", "pyvenv.cfg"), "w", encoding="utf-8") as handle:
        handle.write(f"home = {root}\n")
    if model:
        folder = P.model_dir(root)
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "model.safetensors.index.json"), "w", encoding="utf-8") as handle:
            json.dump({"weight_map": {"w": "model-1.safetensors"}}, handle)
        with open(os.path.join(folder, "model-1.safetensors"), "wb") as handle:
            handle.write(b"x")


def phase_runtime_root(P, base):
    new, old = os.path.join(base, "OpenAct"), os.path.join(base, "JevNodes")
    check("root: nothing installed -> OpenAct", P.runtime_root() == new, P.runtime_root())
    fake_runtime(P, old)
    check("root: only the Jev Nodes runtime -> JevNodes, in place", P.runtime_root() == old, P.runtime_root())
    fake_runtime(P, new, model=False)
    check("root: an incomplete OpenAct runtime -> JevNodes", P.runtime_root() == old, P.runtime_root())
    fake_runtime(P, new)
    check("root: an installed OpenAct runtime -> OpenAct", P.runtime_root() == new, P.runtime_root())
    P._Defaults.runtime_dir = os.path.join(base, "Custom")
    check("root: the Runtime Folder preference wins", P.runtime_root() == P._Defaults.runtime_dir, P.runtime_root())
    return new


class OldServer(BaseHTTPRequestHandler):
    """/health of the model server that the jev_nodes v0.1.0 add-on starts."""

    def do_GET(self):
        body = json.dumps({"server": "jev-nodes-semif", "version": "0.1.0", "status": "ready"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


class FakeProcess:
    """Replaces subprocess.Popen: records each launch of the model server, which has already exited."""

    launched = []

    def __init__(self, args, **_kwargs):
        FakeProcess.launched.append(args)

    def poll(self):
        return 1

    def wait(self, timeout=None):
        return 1


class Inline:
    """Replaces server.executor(): refresh()'s /health probe runs at once, so its effect is checked directly."""

    @staticmethod
    def submit(fn, *args):
        fn(*args)


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def phase_old_server(engine, server, P, root):
    old = ThreadingHTTPServer(("127.0.0.1", 0), OldServer)
    threading.Thread(target=old.serve_forever, daemon=True).start()
    port = old.server_address[1]
    P._Defaults.runtime_dir, P._Defaults.port = root, port
    subprocess.Popen, server.executor = FakeProcess, Inline
    server.STATE.probes[P.runtime_root()] = ""  # the fake runtime has no Python to probe: count it as passed
    legacy = ("error", server.LEGACY_SERVER_MESSAGE)
    tree = bpy.data.node_groups.new("Legacy", 'GeometryNodeTree')
    ask = tree.nodes.new(engine.ASK_ID)
    ask.inputs["Question"].default_value = "Is it night?"

    engine.run_pass()  # the node needs an answer, so auto-start meets the old server
    check("old server: the status names it", server.status() == legacy, server.status())
    check("old server: the node shows it", engine.STATUS.get(ask.uid) == ('ERROR', legacy[1]),
          engine.STATUS.get(ask.uid))
    check("old server: nothing is launched", not FakeProcess.launched, FakeProcess.launched)
    server.refresh()
    check("old server: the message stays while it runs", server.status() == legacy, server.status())

    P._Defaults.port = free_port()
    server.refresh()
    check("another port: the status clears", server.status() == ("stopped", ""), server.status())
    P._Defaults.port = port
    engine.run_pass()
    check("back on the old port: auto-start meets it again",
          server.status() == legacy and not FakeProcess.launched, server.status())

    old.shutdown()
    old.server_close()
    server.refresh()
    check("old server gone: the status clears", server.status() == ("stopped", ""), server.status())
    engine.run_pass()
    args = FakeProcess.launched[0] if len(FakeProcess.launched) == 1 else []
    check("old server gone: auto-start launches OpenAct's server on the port",
          "--port" in args and args[args.index("--port") + 1] == str(port), FakeProcess.launched)


def main():
    if bpy.app.version < (4, 5, 0):
        phase_old_blender()
    else:
        engine, server, P = load_package()
        base = tempfile.mkdtemp(prefix="openact_legacy_")
        os.environ["LOCALAPPDATA" if sys.platform == "win32" else "XDG_DATA_HOME"] = base
        root = phase_runtime_root(P, base)
        phase_old_server(engine, server, P, root)
    finish()


main()
