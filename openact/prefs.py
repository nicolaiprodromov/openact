"""Add-on preferences and the paths of the local model runtime."""

from __future__ import annotations

import os
import sys

import bpy

ADDON_ID = __package__  # "openact", or "bl_ext.<repo>.openact" when installed as an extension

DEFAULT_PORT = 18765
MODEL_SOURCE = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
MODEL_LABEL = "Qwen3.5-4B via SemIf"
UNSURE_BELOW = 0.30  # 4.e: the Unsure Below preference's default


def default_runtime_root() -> str:
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Local")
    elif sys.platform == "darwin":
        base = os.path.join(os.path.expanduser("~"), "Library", "Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
    return os.path.join(base, "OpenAct")


class _Defaults:
    """Used when the package is loaded without being enabled as an add-on (tests, scripts)."""

    runtime_dir = ""
    port = DEFAULT_PORT
    gpu = 0
    device = "auto"
    auto_start = True
    debias = True
    unsure_below = UNSURE_BELOW
    log_requests = False
    trusted_folders = []  # plain path strings here; PropertyGroups with a path in the preferences
    trusted_files = []


def get_prefs():
    addon = bpy.context.preferences.addons.get(ADDON_ID)
    return addon.preferences if addon else _Defaults()


def runtime_root(prefs=None) -> str:
    """The folder set in the preferences, else an installed OpenAct runtime, else an installed
    Jev Nodes runtime, else the folder for a new install."""
    prefs = prefs or get_prefs()
    if prefs.runtime_dir:
        return os.path.normpath(bpy.path.abspath(prefs.runtime_dir))
    default = default_runtime_root()
    # The old runtime is used in place: its venv and uv shims hold absolute paths, so it can't move.
    for root in (default, os.path.join(os.path.dirname(default), "JevNodes")):
        if runtime_installed(root):
            return os.path.normpath(root)
    return os.path.normpath(default)


def venv_python(root: str) -> str:
    if sys.platform == "win32":
        return os.path.join(root, "venv", "Scripts", "python.exe")
    return os.path.join(root, "venv", "bin", "python")


def venv_home(root: str) -> str:
    """The base interpreter's folder named by the venv's pyvenv.cfg ("" if unreadable). It can lie outside the
    runtime: the Jev Nodes runtime's venv uses the user's global uv Python."""
    try:
        with open(os.path.join(root, "venv", "pyvenv.cfg"), encoding="utf-8") as handle:
            for line in handle:
                key, sep, value = line.partition("=")
                if sep and key.strip().lower() == "home":
                    return value.strip()
    except (OSError, ValueError):
        pass
    return ""


def hf_home(root: str) -> str:
    return os.path.join(root, "hf")


def model_dir(root: str) -> str:
    repo = "models--" + MODEL_SOURCE.replace("/", "--")
    return os.path.join(hf_home(root), "hub", repo, "snapshots", MODEL_REVISION)


_installed_cache = {}


def runtime_installed(root: str) -> bool:
    """The venv's Python, the base interpreter folder it points at, and every weight shard listed in the model index
    are on disk. File-system checks only: whether that Python imports torch is the probe's job (server.py)."""
    if not os.path.isfile(venv_python(root)) or not os.path.isdir(venv_home(root)):
        return False
    folder = model_dir(root)
    index = os.path.join(folder, "model.safetensors.index.json")
    try:
        stamp = os.path.getmtime(index)
    except OSError:
        return False
    cached = _installed_cache.get(folder)
    if cached and cached[0] == stamp and cached[1]:
        return True
    try:
        import json

        with open(index, encoding="utf-8") as handle:
            shards = set(json.load(handle).get("weight_map", {}).values())
        complete = bool(shards) and all(os.path.getsize(os.path.join(folder, shard)) > 0 for shard in shards)
    except (OSError, ValueError):
        complete = False
    _installed_cache[folder] = (stamp, complete)
    return complete


class OpenActPath(bpy.types.PropertyGroup):
    path: bpy.props.StringProperty(name="Path", default="")


def publish_log_requests(self, _context):
    """The decision worker reads the preference from the server state: turning it off stops the log at once."""
    from . import server  # imports this module

    server.STATE.log_requests = self.log_requests


class OpenActPreferences(bpy.types.AddonPreferences):
    bl_idname = ADDON_ID

    runtime_dir: bpy.props.StringProperty(
        name="Runtime Folder",
        subtype='DIR_PATH',
        default="",
        description="Where the model runtime (Python environment and model weights, about 12 GB) lives. "
                    "Leave empty for the default location",
    )
    port: bpy.props.IntProperty(
        name="Port", default=DEFAULT_PORT, min=1024, max=65535,
        description="Local port of the model server (127.0.0.1 only)",
    )
    device: bpy.props.EnumProperty(
        name="Device",
        items=(
            ('auto', "Auto", "Use the NVIDIA GPU. Without CUDA the model server reports an error instead of "
                             "using the CPU"),
            ('cuda', "GPU (CUDA)", "Use an NVIDIA GPU"),
            ('cpu', "CPU", "Use the CPU (much slower)"),
        ),
        default='auto',
    )
    gpu: bpy.props.IntProperty(
        name="GPU", default=0, min=0, max=15,
        description="Which NVIDIA GPU runs the model (0 = first). The model needs about 9 GB of VRAM",
    )
    auto_start: bpy.props.BoolProperty(
        name="Start Automatically", default=True,
        description="Start the model server as soon as an OpenAct node needs an answer",
    )
    debias: bpy.props.BoolProperty(
        name="Balance Option Order for New Nodes", default=True,
        description="The Balance Option Order setting that new OpenAct nodes start with: every question asked with "
                    "the options in both orders, at twice the cost. Existing nodes keep their own setting",
    )
    unsure_below: bpy.props.FloatProperty(
        name="Unsure Below", default=UNSURE_BELOW, min=0.0, max=1.0,
        description="Flag the model's answers whose confidence is below this: in the node's status line, the "
                    "sidebar's table and the unsure column of an exported Overrides CSV",
    )
    log_requests: bpy.props.BoolProperty(
        name="Log Requests", default=False, update=publish_log_requests,
        description="Write full requests and responses to <runtime>/logs/, one JSON line each. The prompts and CSV "
                    "text they hold stay on this computer until you delete them",
    )
    trusted_folders: bpy.props.CollectionProperty(
        type=OpenActPath, name="Trusted Folders",
        description="Folders (and their subfolders) whose .blend files may read files outside their own folder",
    )
    trusted_files: bpy.props.CollectionProperty(
        type=OpenActPath, name="Trusted Files",
        description=".blend files that may read files outside their own folder (Allow this file adds to it)",
    )

    def draw(self, context):
        from . import ui

        ui.draw_preferences(self, context, self.layout)


classes = (OpenActPath, OpenActPreferences)  # the PropertyGroup first: the preferences use it
