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


def hf_home(root: str) -> str:
    return os.path.join(root, "hf")


def model_dir(root: str) -> str:
    repo = "models--" + MODEL_SOURCE.replace("/", "--")
    return os.path.join(hf_home(root), "hub", repo, "snapshots", MODEL_REVISION)


_installed_cache = {}


def runtime_installed(root: str) -> bool:
    """The venv exists and every weight shard listed in the model index is on disk."""
    if not os.path.isfile(venv_python(root)):
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
            ('auto', "Auto", "Use an NVIDIA GPU if there is one, otherwise the CPU"),
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
        name="Balance Option Order", default=True,
        description="Ask every question with the options in both orders and average the answers. "
                    "Removes the small model's bias toward early options, at twice the cost",
    )

    def draw(self, context):
        from . import ui

        ui.draw_preferences(self, context, self.layout)
