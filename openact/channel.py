"""The Warning channel's reader (§5.7): what each evaluation reported, per node and modifier.

Every inner group reports "OpenAct <uid> <0|1> <key text>" through one Warning node, which its modifier lists in
node_warnings after each evaluation that evaluates the node (inner._Builder.warning()). read() collects those lists
once the viewport depsgraph has evaluated: the words of every evaluated context, words the static evaluator can't
read included, and whether the node's Trigger was on (§6.0). Only the viewport depsgraph logs warnings, so a render
reports nothing (§7.8).
"""

from __future__ import annotations

import bpy

from . import inner, words


def evaluated():
    """The context's depsgraph, evaluated: evaluated_depsgraph_get() evaluates whatever changed since the last
    evaluation, so a timer or a headless script reads the lists of the current one. None without a scene."""
    try:
        return bpy.context.evaluated_depsgraph_get()
    except RuntimeError:
        return None


def read(nodes, contexts, depsgraph) -> dict:
    """What the evaluation reported: node uid -> {(object pointer, modifier name): {words: triggered}}, each word
    tuple as words.parse() reads it, in the order the modifier lists them. The same words reported with the Trigger
    on and off count as triggered.

    nodes: (tree, node) of every OpenAct node; contexts: evaluator.Ctx of the modifiers to read, each read once.
    A modifier that didn't evaluate in depsgraph (disabled in the viewport or in edit mode; its object disabled in
    viewports, in an excluded or hidden collection or in no scene) reports nothing, whatever list an earlier
    evaluation left on it. A uid that another group may report too is left out: a copy shares its original's group
    until engine.ensure_owned() gives it its own, whose uid node then holds the original's until the pass rewrites
    it (inner.update_values()). Each group reports its own uid from the evaluation after."""
    holders = {}  # the uid a group reports -> the nodes whose group reports it
    for _tree, node in nodes:
        ident = node.node_tree.nodes.get(inner.UID) if node.node_tree is not None else None
        if ident is not None:
            holders.setdefault(ident.string, []).append(node)
    own = {uid for uid, found in holders.items() if len(found) == 1 and found[0].uid == uid}
    reports, seen = {}, set()
    for ctx in contexts:
        at = (ctx.obj.as_pointer(), ctx.mod.name) if ctx.mod is not None else None
        if at is None or at in seen:
            continue
        seen.add(at)
        if depsgraph is None or not evaluates(ctx.obj, ctx.mod, depsgraph):
            continue
        for warning in ctx.mod.node_warnings:
            report = words.parse(warning.message)
            if report is None or report.uid not in own:
                continue
            found = reports.setdefault(report.uid, {}).setdefault(at, {})
            found[report.words] = found.get(report.words, False) or report.triggered
    return reports


def evaluates(obj, mod, depsgraph) -> bool:
    """True when the modifier took part in the depsgraph's evaluation, so its node_warnings are that evaluation's. An
    object hidden with the eye still evaluates, and so does one only another object's tree reads (Object Info)."""
    if not mod.show_viewport or (obj.mode == 'EDIT' and not mod.show_in_editmode):
        return False
    return obj.evaluated_get(depsgraph).is_evaluated
