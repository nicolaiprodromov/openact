"""Shared helpers for the test suites.

The line protocol, read by tests/run_all.ps1: a "PASS <check>" or "FAIL <check>" line per check,
then one RESULT line from finish(). The *_test.py suites also run under plain CPython, so only the
Blender helpers touch bpy.
"""

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)  # suites import openact from this checkout; Blender does not install it
os.environ["OPENACT_BACKGROUND_PASSES"] = "1"  # the suites author headlessly (§7.8): openact.engine reads it on import
FAILURES = []
_NO_DETAIL = object()  # details like 0, None or [] must still print


def check(name, condition, detail=_NO_DETAIL):
    print(("PASS " if condition else "FAIL ") + name + ("" if detail is _NO_DETAIL else f"  [{detail}]"), flush=True)
    if not condition:
        FAILURES.append(name)


def finish():
    """Print the RESULT line and exit with its code at once: after the script, Blender itself would
    exit with 0."""
    bpy = sys.modules.get("bpy")
    where = f"BLENDER {bpy.app.version_string}" if bpy else f"PYTHON {sys.version.split()[0]}"
    print(f"{where} RESULT {'OK' if not FAILURES else 'FAILED: ' + ', '.join(FAILURES)}", flush=True)
    os._exit(0 if not FAILURES else 1)


def sock(sockets, identifier):
    return next(s for s in sockets if s.identifier == identifier)


def enable_addon():
    import addon_utils

    module = addon_utils.enable("openact", default_set=True, handle_error=None)
    assert module is not None, "add-on failed to enable"
    from openact import engine, server
    from openact import prefs as P

    return engine, server, P


def drive(engine, done, timeout):
    """Tick the add-on until done() is true. Blender runs no timers while a --python script runs."""
    started = time.time()
    while time.time() - started < timeout:
        engine.tick()
        if done():
            return True
        time.sleep(0.05)
    return False


def store(tree, geometry_socket, value_socket, name, data_type, x=0):
    """Store the value as a point attribute, so the test can read it from the evaluated mesh."""
    node = tree.nodes.new("GeometryNodeStoreNamedAttribute")
    node.data_type = data_type
    node.domain = 'POINT'
    node.location = (x, 0)
    node.inputs["Name"].default_value = name
    tree.links.new(geometry_socket, node.inputs["Geometry"])
    value = next(s for s in node.inputs if s.name == "Value" and s.enabled)
    tree.links.new(value_socket, value)
    return node.outputs["Geometry"]
