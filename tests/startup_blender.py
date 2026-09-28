"""Where OpenAct starts from, in a headless Blender. No model or runtime needed.

Blender 4.5 and newer:
  - runtime_root(): the Runtime Folder preference, else the default folder, installed or not;
  - a server of another program on the port gives the port error and nothing is launched, also when it answers
    again; once it has gone, Start Model launches OpenAct's server on the port.
Older Blender: enabling the add-on fails before anything is registered.

Run:  blender -b --factory-startup --python tests/startup_blender.py
"""

import json
import os
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
    new = os.path.join(base, "OpenAct")
    check("root: nothing installed -> OpenAct, for a new install", P.runtime_root() == new, P.runtime_root())
    fake_runtime(P, new, model=False)
    check("root: an incomplete runtime -> OpenAct", P.runtime_root() == new and not P.runtime_installed(new),
          P.runtime_root())
    fake_runtime(P, new)
    check("root: an installed runtime -> OpenAct", P.runtime_root() == new and P.runtime_installed(new),
          P.runtime_root())
    P._Defaults.runtime_dir = os.path.join(base, "Custom")
    check("root: the Runtime Folder preference wins", P.runtime_root() == P._Defaults.runtime_dir, P.runtime_root())
    return new


class OtherServer(BaseHTTPRequestHandler):
    """/health of a server another program runs on the port: it answers, but it isn't OpenAct's."""

    def do_GET(self):
        body = json.dumps({"server": "some-other-server", "version": "2.0", "status": "ready"}).encode()
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


def phase_other_server(engine, server, P, root):
    other = ThreadingHTTPServer(("127.0.0.1", 0), OtherServer)
    threading.Thread(target=other.serve_forever, daemon=True).start()
    port = other.server_address[1]
    P._Defaults.runtime_dir, P._Defaults.port = root, port
    subprocess.Popen, server.executor = FakeProcess, Inline
    server.STATE.probes[P.runtime_root()] = ""  # the fake runtime has no Python to probe: count it as passed
    taken = ("error", f"Port {port} is used by another program. Pick another port in the preferences")
    tree = bpy.data.node_groups.new("Startup", 'GeometryNodeTree')
    ask = tree.nodes.new(engine.ASK_ID)
    ask.inputs["Question"].default_value = "Is it night?"

    engine.run_pass()  # the node needs an answer, so auto-start meets the other server
    check("other server: the status names the port", server.status() == taken, server.status())
    status = engine.STATUS.get(ask.uid)
    check("other server: the node shows it", status is not None and (status.icon, status.text) == ('CANCEL', taken[1]),
          status)
    check("other server: nothing is launched", not FakeProcess.launched, FakeProcess.launched)
    server.refresh()
    check("other server: the message stays while it runs", server.status() == taken, server.status())
    engine.run_pass()
    check("other server: a pass launches nothing either", server.status() == taken and not FakeProcess.launched,
          server.status())

    other.shutdown()
    other.server_close()
    started = server.start()  # Start Model: the port is free now
    args = FakeProcess.launched[0] if len(FakeProcess.launched) == 1 else []
    check("other server gone: Start Model launches OpenAct's server on the port",
          started and "--port" in args and args[args.index("--port") + 1] == str(port), FakeProcess.launched)


def main():
    if bpy.app.version < (4, 5, 0):
        phase_old_blender()
    else:
        engine, server, P = load_package()
        base = tempfile.mkdtemp(prefix="openact_startup_")
        os.environ["LOCALAPPDATA" if sys.platform == "win32" else "XDG_DATA_HOME"] = base
        root = phase_runtime_root(P, base)
        phase_other_server(engine, server, P, root)
    finish()


main()
