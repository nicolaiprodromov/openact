"""The file trust rule (design 16.1).

A .blend received from someone else could point a word at a private file, and OpenAct would store what it read in
the .blend. So in a file that is not trusted, every file read (CSVs and Import Text) stays inside the .blend's own
folder tree, and a blocked file is never opened.

A file is trusted when it was created in this session, when Blender's Auto Run Python Scripts covers it, or when
its path (Allow this file) or one of its folders is in the trusted lists of the preferences.
"""

from __future__ import annotations

import fnmatch
import os

import bpy
from bpy.app.handlers import persistent

from . import prefs as P

TRUST = {"opened": None}  # the .blend as load_post saw it: "" for a file created in this session; None: not seen


def ident(path: str) -> str:
    return os.path.normcase(os.path.realpath(path))


def _inside(path, folder):
    """True when path is folder or lies under it (both from ident())."""
    try:
        return os.path.commonpath([path, folder]) == folder
    except ValueError:  # another drive
        return False


def _paths(prefs, name):
    """A trusted list: PropertyGroups with a path in the preferences, plain strings in prefs._Defaults."""
    return [item if isinstance(item, str) else item.path for item in getattr(prefs, name, ())]


def _auto_run(path):
    """Blender's own decision (BKE_autoexec_match): Auto Run Python Scripts on, and path under no excluded path."""
    preferences = bpy.context.preferences
    if not preferences.filepaths.use_scripts_auto_execute:
        return False
    for excluded in preferences.autoexec_paths:
        if not excluded.path:
            continue
        if excluded.use_glob:
            if fnmatch.fnmatch(path, excluded.path):  # case-insensitive on Windows, as Blender's FNM_CASEFOLD
                return False
        elif os.path.normcase(path).startswith(os.path.normcase(excluded.path)):
            return False
    return True


def is_trusted() -> bool:
    """Whether this .blend may read files outside its own folder tree."""
    path = bpy.data.filepath
    if TRUST["opened"] == "" or not path:  # created in this session, saved since or not
        return True
    if _auto_run(path):
        return True
    prefs = P.get_prefs()
    own = ident(path)
    if any(p and ident(p) == own for p in _paths(prefs, "trusted_files")):
        return True
    return any(p and _inside(own, ident(p)) for p in _paths(prefs, "trusted_folders"))


def blocked(path: str) -> str:
    """Why this file may not read path, or "" when it may. realpath is compared, so neither a symlink nor '..'
    escapes the folder tree."""
    if is_trusted() or _inside(ident(path), ident(os.path.dirname(bpy.data.filepath))):
        return ""
    # normpath: Blender 5 saves file paths relative, so '//../x' arrives as '<folder>/../x'
    return (f"Blocked reading {os.path.normpath(path)}: outside this file's folder. Allow this file (or trust its "
            f"folder) to read it")


def allow(path: str | None = None) -> bool:
    """Allow this file: trust one .blend, by path. False when it was trusted already."""
    path = os.path.normpath(path or bpy.data.filepath)
    prefs = P.get_prefs()
    if any(p and ident(p) == ident(path) for p in _paths(prefs, "trusted_files")):
        return False
    files = prefs.trusted_files
    if isinstance(files, list):  # prefs._Defaults: the add-on is not enabled
        files.append(path)
    else:
        files.add().path = path
        bpy.context.preferences.is_dirty = True  # saved with the other preferences
    return True


@persistent
def on_load_post(*_args):
    TRUST["opened"] = bpy.data.filepath


def register():
    if not bpy.data.filepath:  # the add-on starts on an unsaved file: it was created in this session
        TRUST["opened"] = ""
    if on_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(on_load_post)


def unregister():
    if on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(on_load_post)
