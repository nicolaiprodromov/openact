"""The upgrade from jev_nodes v0.1.0 (§12.3): each old Jev node becomes the OpenAct node it grew into, keeping its
words, links, settings, uid, answers and inner group (§12.4), in the one undo step of openact.upgrade_legacy."""

from __future__ import annotations

import json
import traceback

import bpy
from bpy.app.handlers import persistent

from . import engine, inner, store
from .specs import SPECS

LEGACY_TYPES = {"GeometryNodeJevAsk": engine.ASK_ID, "GeometryNodeJevPick": engine.PICK_ID}  # old node -> its upgrade
SETTINGS = ("label", "width", "mute", "hide", "select", "use_custom_color", "color")  # copied after the parent frame
NUMBERS = ('INT', 'VALUE', 'BOOLEAN')


def find_legacy():
    """(tree, node) of the old Jev nodes in this file's own node trees: a linked one is upgraded in its library."""
    return [(tree, node) for tree, node in engine.iter_nodes(LEGACY_TYPES) if tree.library is None]


def _nodes(count):
    return f"{count} old Jev node{'s' if count != 1 else ''}"


def banner():
    """The sidebar's banner while the file has old Jev nodes (ui.LEGACY): (text, the upgrade's idname), or None."""
    count = len(find_legacy())
    return (f"This file has {_nodes(count)}: Upgrade", "openact.upgrade_legacy") if count else None


def notice():
    """The INFO line after a file with old Jev nodes loads, else "". Only a line: opening a file never rewrites it."""
    count = len(find_legacy())
    if not count:
        return ""
    return f"This file has {_nodes(count)}: upgrade {'them' if count > 1 else 'it'} in the OpenAct sidebar"


@persistent
def on_load_post(*_args):
    """Point at the upgrade after a file with old Jev nodes loads. Only a report from a modal step reaches the status
    bar and the Info editor (one through bpy.ops stays with its Python caller), so a timer starts one in a window; in
    background mode, which has no window to run it, the report goes to the console."""
    if not find_legacy():
        return
    if bpy.app.background:
        bpy.ops.openact.legacy_report()
    else:
        bpy.app.timers.register(_report, first_interval=0.1)  # once the file's windows are set up


def _report():
    if bpy.context.window is not None:  # a timer's context has the first window: no temp_override (engine._undo_push)
        bpy.ops.openact.legacy_report('INVOKE_DEFAULT')


def upgrade_all():
    """Replace each old Jev node of this file's own trees with its OpenAct node (§12.3). Returns (nodes upgraded, node
    trees they are in, problems, changed). A problem names a node left frozen, and why; changed is True once anything
    changed: an internal error may leave its node half done, which the operator's undo step then reverts."""
    upgraded, trees, problems, changed = 0, set(), [], False
    for tree, node in find_legacy():
        name = node.name
        try:
            problem = _upgrade(tree, node)
        except Exception as error:  # noqa: BLE001 - one node's failure leaves the others upgraded
            traceback.print_exc()
            problem, changed = f"internal error: {error}", True
        if problem:
            problems.append(f"'{name}' in {tree.name}: {problem}")
        else:
            upgraded += 1
            trees.add(tree.name)
    return upgraded, len(trees), problems, changed or bool(upgraded)


def report(upgraded, trees, problems):
    """The operator's report, e.g. "Upgraded 9 nodes in 3 node groups; 0 problems"."""
    text = (f"Upgraded {upgraded} node{'s' if upgraded != 1 else ''} in {trees} node group{'s' if trees != 1 else ''}; "
            f"{len(problems)} problem{'s' if len(problems) != 1 else ''}")
    return f"{text}: {'; '.join(problems)}" if problems else text


def _upgrade(tree, old):
    """Replace one old Jev node (§12.3 steps 1-6). Returns why it can't, having left it as it was, or ""."""
    new_id = LEGACY_TYPES[old.bl_idname]
    source = None  # a Pick's list source
    if new_id == engine.PICK_ID:
        collection = old.inputs.get("Collection")
        used = collection is not None and (collection.default_value is not None
                                           or any(not link.is_muted for link in collection.links))
        source = 'COLLECTION' if used else 'TEXT'  # v0.1.0's precedence: a set Collection wins over the Items text
    words = ("Question", "Context") if source is None else ("Prompt", "Items" if source == 'TEXT' else "Collection")
    # Everything read from the old node before anything changes: the inner group it shares with the new node changes
    inputs = [(s.name, s.default_value, [(link.from_socket, link.is_muted) for link in s.links])
              for s in old.inputs if s.name in words]
    outputs = [(s.name, link.to_socket, link.is_muted, link) for s in old.outputs for link in s.links]
    place = {key: getattr(old, key) for key in SETTINGS}
    place["color"] = tuple(old.color)  # not a view of the old node, which goes
    parent, location, name, active = old.parent, tuple(old.location), old.name, tree.nodes.active == old
    passed = {link.to_socket.name: link.from_socket.name for link in old.internal_links}  # what, muted, it passes on
    node_store = store.load(old)  # its v0.1.0 answers, as v1 stores them (§14.4)

    new = tree.nodes.new(new_id)  # init(): a fresh uid and inner group, and Balance Option Order from the preferences
    fresh = new.node_tree
    if source is None:
        new.answer_type = old.mode
    else:
        items = old.inputs.get("Items")  # its text and its answers' texts keep v0.1.0's parsing (lists.py)
        texts = {entry.gn[2] for entry in node_store.entries} | ({items.default_value} if items is not None else set())
        new.list_source, new.items_parse_v0 = source, True
        new.items_v0_texts = json.dumps(sorted(texts), ensure_ascii=False)
    problem = _refused(old, new, {output for output, *_link in outputs}, passed)
    if problem:
        tree.nodes.remove(new)
        bpy.data.node_groups.remove(fresh)
        return problem
    new.uid = old.uid or new.uid
    group = old.node_tree if old.node_tree is not None and old.node_tree.library is None else fresh
    if group is not fresh:  # the old inner group becomes the new node's, so nothing is left orphaned
        new.node_tree = group
        bpy.data.node_groups.remove(fresh)  # first: a node that keeps its fresh uid needs the name it holds
    group.name = inner.inner_name(new)
    inner.sync_interface(new, SPECS[new_id].interface(new))
    store.save(new, node_store)  # no live set: nothing is evicted (§13.3)
    inner.build(new, node_store)
    for socket_name, value, links in inputs:
        socket = new.inputs[socket_name]
        socket.default_value = value
        for from_socket, muted in links:
            tree.links.new(from_socket, socket).is_muted = muted
    for output, target, muted, old_link in outputs:
        link = tree.links.new(new.outputs[output], target)
        link.is_muted = muted
        if target.is_multi_input:  # the new link takes the old one's place in the join order
            link.swap_multi_input_sort_id(old_link)
    new.parent = parent  # before the location, which is relative to it
    new.location = location
    for key, value in place.items():
        setattr(new, key, value)
    tree.nodes.remove(old)
    new.name = name
    if active:
        tree.nodes.active = new
    return ""


def _refused(old, new, linked, passed):
    """Why the new node can't take the old one's place without changing an output, or "": a linked output it lacks, or,
    muted, a linked number it would pass another input to. Blender passes a muted node's inputs to outputs of their
    type, and the OpenAct Pick's ID input reaches Index and Confidence, which v0.1.0's muted Pick left at 0."""
    lost = sorted(linked - set(new.outputs.keys()))
    if lost:
        return f"its {lost[0]} output is linked, and the OpenAct node has none for this list: unlink it, then upgrade"
    for link in new.internal_links if old.mute else ():
        output = link.to_socket
        if output.name in linked and output.type in NUMBERS and passed.get(output.name) != link.from_socket.name:
            return (f"it is muted, and a muted OpenAct {SPECS[new.bl_idname].label} passes its {link.from_socket.name} "
                    f"input to {output.name}: unmute or unlink it, then upgrade")
    return ""


def register():
    if on_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(on_load_post)


def unregister():
    if bpy.app.timers.is_registered(_report):
        bpy.app.timers.unregister(_report)
    if on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(on_load_post)
