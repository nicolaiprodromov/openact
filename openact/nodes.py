"""The Ask, Pick and Dials nodes."""

from __future__ import annotations

import bpy

from . import engine, inner
from . import prefs as P
from .specs import dials


def _changed(self, context):
    engine.schedule(immediate=True, node=self)


class _OpenActNode:
    """Shared behaviour. Each node wraps a hidden node group that holds its answers."""

    bl_width_default = 240
    bl_width_min = 160

    uid: bpy.props.StringProperty(options={'HIDDEN'})
    entries: bpy.props.StringProperty(options={'HIDDEN'}, default="[]")
    balance_order: bpy.props.BoolProperty(
        name="Balance Option Order", default=True, update=_changed,
        description="Ask every question with its options in both orders and average the answers. Removes the small "
                    "model's bias toward early options, at twice the cost. Changing it keeps the answers given so far",
    )

    @classmethod
    def poll(cls, ntree):
        return ntree.bl_idname == 'GeometryNodeTree' and not ntree.name.startswith(inner.INNER_PREFIX)

    def init(self, context):
        self.uid = engine.new_uid()
        self.balance_order = P.get_prefs().debias
        inner.create_inner(self)
        engine.schedule(immediate=True)

    def copy(self, other):
        # Don't reassign node_tree here: when Blender duplicates a whole node tree, a pointer set in
        # copy() is not user-counted (Blender 4.2) and the original's hidden group is lost on save.
        # The copy keeps sharing the group (same answers) until the next pass gives it its own.
        self.uid = ""
        engine.schedule(immediate=True)

    def free(self):
        # The hidden group is left without users and is not saved; removing data-blocks while
        # Blender frees a node is not safe.
        engine.STATUS.pop(self.uid, None)

    def update(self):
        engine.schedule(node=self)

    def socket_value_update(self, context):
        engine.schedule(node=self)

    def draw_status(self, layout, full=False):
        icon, text = engine.STATUS.get(self.uid, ('SORTTIME', "Waiting..."))
        if full and len(text) > 48:
            col = layout.column(align=True)
            words, line = text.split(), ""
            first = True
            for word in words:
                if len(line) + len(word) > 44:
                    col.label(text=line, icon=icon if first else 'BLANK1')
                    first, line = False, ""
                line = f"{line} {word}".strip()
            col.label(text=line, icon=icon if first else 'BLANK1')
        else:
            layout.label(text=text, icon=icon)

    def draw_answers(self, layout):
        rows = engine.answers_for(self)
        spec = engine.SPECS[self.bl_idname]
        if not rows:
            return
        box = layout.box()
        for label, answer in rows:
            col = box.column(align=True)
            col.label(text=label, icon='MODIFIER' if not label.startswith("(") else 'NODETREE')
            if answer is None:
                col.label(text="Asking...", icon='SORTTIME')
                continue
            col.label(text=spec.summary(answer), icon='CHECKMARK')
            if engine.older_settings(self, answer):
                col.label(text="Asked with older settings: Ask Again refreshes it", icon='BLANK1')
            if self.bl_idname == engine.PICK_ID:
                ranked = sorted(zip(answer.rows.name, answer.rows.v["prob"]), key=lambda pair: -pair[1])[:5]
                for name, probability in ranked:
                    row = col.row()
                    row.label(text=name, icon='BLANK1')
                    row.label(text=f"{probability * 100:.1f}%")
            else:
                col.label(text=f"Confidence {answer.single['Confidence']:.2f}", icon='BLANK1')

    def draw_settings(self, layout):
        layout.prop(self, "balance_order")

    def draw_buttons(self, context, layout):
        self.draw_settings(layout)
        self.draw_status(layout)

    def draw_buttons_ext(self, context, layout):
        self.draw_settings(layout)
        self.draw_status(layout, full=True)
        self.draw_answers(layout)
        row = layout.row()
        row.enabled = self.id_data.library is None  # a linked node's answers come from its library file
        row.operator("openact.reask", icon='FILE_REFRESH').uid = self.uid


class GeometryNodeOpenActAsk(_OpenActNode, bpy.types.GeometryNodeCustomGroup):
    """Ask the local decision model a question; outputs its answer as a number"""

    bl_idname = engine.ASK_ID
    bl_label = "Ask"

    answer_type: bpy.props.EnumProperty(
        name="Answer",
        items=(
            ('YES_NO', "Yes / No", "Probability that the answer is yes (0-1)"),
            ('AMOUNT', "Amount", "How much, from 0 (not at all) to 1 (extremely)"),
        ),
        default='YES_NO',
        update=_changed,
    )

    def draw_settings(self, layout):
        layout.row().prop(self, "answer_type", expand=True)
        super().draw_settings(layout)


class GeometryNodeOpenActPick(_OpenActNode, bpy.types.GeometryNodeCustomGroup):
    """Pick the item from a list (or a collection) that best fits a prompt"""

    bl_idname = engine.PICK_ID
    bl_label = "Pick"


class GeometryNodeOpenActDials(_OpenActNode, bpy.types.GeometryNodeCustomGroup):
    """Judge many named settings against one prompt, in one request: one output per setting"""

    bl_idname = dials.DialsSpec.idname
    bl_label = "Dials"

    def draw_status(self, layout, full=False):
        super().draw_status(layout, full)
        warning = dials.WARNINGS.get(self.uid)
        if warning:
            layout.label(text=warning, icon='ERROR')


class NODE_MT_openact_add(bpy.types.Menu):
    bl_idname = "NODE_MT_openact_add"
    bl_label = "OpenAct"

    def draw(self, context):
        layout = self.layout
        for idname, label, icon in ((engine.ASK_ID, "Ask", 'QUESTION'),
                                    (engine.PICK_ID, "Pick", 'RESTRICT_SELECT_OFF'),
                                    (dials.DialsSpec.idname, "Dials", 'OPTIONS')):
            op = layout.operator("node.add_node", text=label, icon=icon)
            op.type = idname
            op.use_transform = True


def draw_add_menu(self, context):
    space = context.space_data
    if getattr(space, "tree_type", None) != 'GeometryNodeTree':
        return
    tree = getattr(space, "edit_tree", None)
    if tree is not None and tree.name.startswith(inner.INNER_PREFIX):
        return
    self.layout.separator()
    self.layout.menu(NODE_MT_openact_add.bl_idname, icon='NODETREE')


classes = (GeometryNodeOpenActAsk, GeometryNodeOpenActPick, GeometryNodeOpenActDials, NODE_MT_openact_add)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.NODE_MT_add.append(draw_add_menu)


def unregister():
    bpy.types.NODE_MT_add.remove(draw_add_menu)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
