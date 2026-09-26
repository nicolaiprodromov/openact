"""The file trust rule (design 16.1).

A .blend received from someone else could point a word at a private file, and OpenAct would store what it read in
the .blend. So in a file that is not trusted, every file read (CSVs and Import Text) stays inside the .blend's own
folder tree, and a blocked file is never opened.

A file is trusted when it was created in this session, or when its path (Allow this file) or one of its folders is
in the trusted lists of the preferences. Blender's Auto Run Python Scripts doesn't count: Blender can refuse it for
one file (Trusted Source unchecked in File > Open), and an add-on can't see that refusal.
"""

from __future__ import annotations

import os

import bpy
from bpy.app.handlers import persistent

from . import prefs as P
from .csvfmt import identity as ident

TRUST = {"opened": None}  # the .blend as load_post saw it: "" for a file created in this session; None: not seen


def _inside(path, folder):
    """True when path is folder or lies under it (both from ident(), or both normcase'd absolute paths)."""
    try:
        return os.path.commonpath([path, folder]) == folder
    except ValueError:  # another drive
        return False


def _paths(prefs, name):
    """A trusted list: PropertyGroups with a path in the preferences, plain strings in prefs._Defaults."""
    return [item if isinstance(item, str) else item.path for item in getattr(prefs, name, ())]


def is_trusted() -> bool:
    """Whether this .blend may read files outside its own folder tree."""
    path = bpy.data.filepath
    if TRUST["opened"] is None and not path:  # enabled on an unsaved file: it was created in this session
        TRUST["opened"] = ""
    if TRUST["opened"] == "" or not path:  # created in this session, saved since or not
        return True
    prefs = P.get_prefs()
    own = ident(path)
    if any(p and ident(p) == own for p in _paths(prefs, "trusted_files")):
        return True
    return any(p and _inside(own, ident(p)) for p in _paths(prefs, "trusted_folders"))


def blocked(path: str) -> str:
    """Why this file may not read path, or "" when it may. A path outside the folder tree is blocked without touching
    it (realpath opens a UNC path's server); one inside is compared by ident() too, so neither a symlink nor '..'
    escapes."""
    if is_trusted():
        return ""
    # abspath: Blender 5 saves file paths relative ('//../x' arrives as '<folder>/../x'), and a bare path is read
    # from the working folder
    full, folder = os.path.abspath(path), os.path.abspath(os.path.dirname(bpy.data.filepath))
    if _inside(os.path.normcase(full), os.path.normcase(folder)) and _inside(ident(full), ident(folder)):
        return ""
    return f"Blocked reading {full}: outside this file's folder. Allow this file (or trust its folder) to read it"


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


def register():  # bpy.data is off limits while add-ons register: is_trusted() looks at the file later
    TRUST["opened"] = None  # a file opened while the add-on was disabled was never seen
    if on_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(on_load_post)


def unregister():
    TRUST["opened"] = None
    if on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(on_load_post)
