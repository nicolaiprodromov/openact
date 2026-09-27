"""The upgrade from jev_nodes v0.1.0 (§12.3): each old Jev node becomes the OpenAct node it grew into, keeping its
words, links, settings, uid, answers and inner group (§12.4), in the one undo step of openact.upgrade_legacy."""

from __future__ import annotations

import traceback

import bpy

from . import engine, inner, store
from .specs import SPECS

SETTINGS = ("label", "width", "mute", "hide", "use_custom_color", "color")  # copied once the parent frame is set


def find_legacy():
    """(tree, node) of the old Jev nodes in this file's own node trees: a linked one is upgraded in its library."""
    return [(tree, node) for tree, node in engine.iter_nodes(engine.LEGACY_TYPES) if tree.library is None]


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


def upgrade_all():
    """Replace each old Jev node of this file's own trees with its OpenAct node (§12.3). Returns (nodes upgraded, node
    trees they are in, problems): a problem names a node left as it was, frozen, and why."""
    upgraded, trees, problems = 0, set(), []
    for tree, node in find_legacy():
        name = node.name
        try:
            problem = _upgrade(tree, node)
        except Exception as error:  # noqa: BLE001 - one node's failure leaves the others upgraded; undo reverts all
            traceback.print_exc()
            problem = f"internal error: {error}"
        if problem:
            problems.append(f"'{name}' in {tree.name}: {problem}")
        else:
            upgraded += 1
            trees.add(tree.name)
    return upgraded, len(trees), problems


def report(upgraded, trees, problems):
    """The operator's report, e.g. "Upgraded 9 nodes in 3 node groups; 0 problems"."""
    text = (f"Upgraded {upgraded} node{'s' if upgraded != 1 else ''} in {trees} node group{'s' if trees != 1 else ''}; "
            f"{len(problems)} problem{'s' if len(problems) != 1 else ''}")
    return f"{text}: {'; '.join(problems)}" if problems else text


def _upgrade(tree, old):
    """Replace one old Jev node (§12.3 steps 1-6). Returns why it can't, having left it as it was, or ""."""
    new_id = engine.LEGACY_TYPES[old.bl_idname]
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
    outputs = [(s.name, link.to_socket, link.is_muted, _order(link.to_socket, old)) for s in old.outputs
               for link in s.links]
    place = {key: getattr(old, key) for key in SETTINGS}
    place["color"] = tuple(old.color)  # not a view of the old node, which goes
    parent, location, name, active = old.parent, tuple(old.location), old.name, tree.nodes.active == old
    node_store = store.load(old)  # its v0.1.0 answers, as v1 stores them (§14.4)

    new = tree.nodes.new(new_id)  # init(): a fresh uid and inner group, and Balance Option Order from the preferences
    fresh = new.node_tree
    if source is None:
        new.answer_type = old.mode
    else:
        new.list_source, new.items_parse_v0 = source, True
    lost = [output for output, *_link in outputs if output not in new.outputs]
    if lost:
        tree.nodes.remove(new)
        bpy.data.node_groups.remove(fresh)
        return f"its {lost[0]} output is linked, and the OpenAct node has none for this list: unlink it, then upgrade"
    new.uid = old.uid or new.uid
    group = old.node_tree if old.node_tree is not None and old.node_tree.library is None else fresh
    group.name = inner.inner_name(new)
    if group is not fresh:  # the old inner group becomes the new node's, so nothing is left orphaned
        new.node_tree = group
        bpy.data.node_groups.remove(fresh)
    inner.sync_interface(new, SPECS[new_id].interface(new))
    store.save(new, node_store)  # no live set: nothing is evicted (§13.3)
    inner.build(new, node_store)
    for socket_name, value, links in inputs:
        socket = new.inputs[socket_name]
        socket.default_value = value
        for from_socket, muted in links:
            tree.links.new(from_socket, socket).is_muted = muted
    for output, target, muted, order in outputs:
        if order is None:
            tree.links.new(new.outputs[output], target).is_muted = muted
            continue
        for link in list(target.links):  # made again oldest first, as each new link joins at the top
            tree.links.remove(link)
        for from_socket, link_muted in reversed(order):
            from_socket = new.outputs[from_socket] if isinstance(from_socket, str) else from_socket
            tree.links.new(from_socket, target).is_muted = link_muted
    new.parent = parent  # before the location, which is relative to it
    new.location = location
    for key, value in place.items():
        setattr(new, key, value)
    tree.nodes.remove(old)
    new.name = name
    if active:
        tree.nodes.active = new
    return ""


def _order(target, old):
    """The links of a multi-input socket in the order they are joined, as the evaluator reads them: (source socket,
    muted), with the old node's outputs by name. None for a single-input socket."""
    if not target.is_multi_input:
        return None
    return [(link.from_socket.name if link.from_node == old else link.from_socket, link.is_muted)
            for link in sorted(target.links, key=lambda link: -link.multi_input_sort_id)]
