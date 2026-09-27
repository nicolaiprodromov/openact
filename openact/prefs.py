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
JEV_SETTINGS = ("runtime_dir", "port", "device", "gpu", "auto_start", "debias")  # what OpenAct takes from jev_nodes


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
    trusted_folders = []  # plain path strings here; PropertyGroups with a path in the preferences
    trusted_files = []


def get_prefs():
    addon = bpy.context.preferences.addons.get(ADDON_ID)
    return addon.preferences if addon else _Defaults()


def is_jev(module: str) -> bool:
    """True for jev_nodes' module name: a legacy install, or an extension (bl_ext.<repo>.jev_nodes)."""
    return module == "jev_nodes" or module.endswith(".jev_nodes")


def jev_addon():
    """The enabled jev_nodes add-on, or None."""
    addons = bpy.context.preferences.addons
    return next((addons[name] for name in addons.keys() if is_jev(name)), None)


def import_jev():
    """On OpenAct's first registration, take jev_nodes' settings while that add-on is enabled, so a custom runtime
    folder or port carries over (§12.2). Once: the preferences remember it, and later changes are the user's own."""
    prefs = getattr(bpy.context.preferences.addons.get(ADDON_ID), "preferences", None)
    if prefs is None or prefs.jev_imported:
        return
    prefs.jev_imported = True
    old = getattr(jev_addon(), "preferences", None)  # None until jev_nodes has registered its preferences
    if old is None:
        return
    for name in JEV_SETTINGS:
        try:
            setattr(prefs, name, getattr(old, name))
        except (AttributeError, TypeError, ValueError):  # a setting it lacks, or a value OpenAct refuses
            pass


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
    trusted_folders: bpy.props.CollectionProperty(
        type=OpenActPath, name="Trusted Folders",
        description="Folders (and their subfolders) whose .blend files may read files outside their own folder",
    )
    trusted_files: bpy.props.CollectionProperty(
        type=OpenActPath, name="Trusted Files",
        description=".blend files that may read files outside their own folder (Allow this file adds to it)",
    )
    jev_imported: bpy.props.BoolProperty(options={'HIDDEN'})  # import_jev() ran

    def draw(self, context):
        from . import ui

        ui.draw_preferences(self, context, self.layout)


classes = (OpenActPath, OpenActPreferences)  # the PropertyGroup first: the preferences use it
