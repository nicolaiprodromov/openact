"""The OpenAct nodes: Ask and Pick (whole situation), Match and Classify (each item)."""

from __future__ import annotations

import bpy

from . import engine, inner, sources, store
from . import prefs as P
from .specs.classify import ClassifySpec
from .specs.match import MatchSpec

PREVIEW_ROWS = 8  # per-item rows the sidebar lists


def _changed(self, context):
    engine.schedule(immediate=True, node=self)


def _list_source_changed(self, context):
    # the dropdown swaps sockets (§6.0); the group is rebuilt for them, as a stored entry may show again unchanged
    if self.node_tree is not None:
        inner.sync_interface(self, engine.SPECS[self.bl_idname].interface(self))
        inner.build(self, store.load(self))
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
    list_source: bpy.props.EnumProperty(  # Pick, Match and Classify (§6.0); unused by the others
        name="List Source", items=sources.LIST_SOURCE_ITEMS, default='TEXT', update=_list_source_changed,
        description="Where the items come from",
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
            elif hasattr(spec, "preview"):  # Match and Classify: every output is per element
                rows = spec.preview(answer)
                for name, text in rows[:PREVIEW_ROWS]:
                    row = col.row()
                    row.label(text=name, icon='BLANK1')
                    row.label(text=text)
                if len(rows) > PREVIEW_ROWS:
                    col.label(text=f"... {len(rows) - PREVIEW_ROWS} more", icon='BLANK1')
            elif "Confidence" in answer.single:
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

    def draw_settings(self, layout):
        layout.prop(self, "list_source", text="")
        super().draw_settings(layout)


class GeometryNodeOpenActMatch(_OpenActNode, bpy.types.GeometryNodeCustomGroup):
    """Ask the local decision model the same question about every item of a list; outputs per element"""

    bl_idname = MatchSpec.idname
    bl_label = "Match"

    answer_type: bpy.props.EnumProperty(
        name="Answer",
        items=(
            ('YES_NO', "Yes / No", "Probability that the answer is yes for each item (0-1)"),
            ('AMOUNT', "Amount", "How much for each item, from 0 (not at all) to 1 (extremely)"),
        ),
        default='YES_NO',
        update=_changed,
    )

    def init(self, context):
        super().init(context)
        # Threshold is 0.5 by default (§6.3); SocketDef carries no default value
        item = self.node_tree.interface.items_tree["Threshold"]
        item.min_value, item.max_value, item.default_value = 0.0, 1.0, 0.5
        self.inputs["Threshold"].default_value = 0.5

    def draw_settings(self, layout):
        layout.row().prop(self, "answer_type", expand=True)
        layout.prop(self, "list_source", text="")
        super().draw_settings(layout)


class GeometryNodeOpenActClassify(_OpenActNode, bpy.types.GeometryNodeCustomGroup):
    """Let the local decision model give every item of a list one of several labels; outputs per element"""

    bl_idname = ClassifySpec.idname
    bl_label = "Classify"

    def draw_settings(self, layout):
        layout.prop(self, "list_source", text="")
        super().draw_settings(layout)


MENU = (  # the Add menu's groups (§6.0)
    ("Whole situation", ((engine.ASK_ID, "Ask", 'QUESTION'), (engine.PICK_ID, "Pick", 'RESTRICT_SELECT_OFF'))),
    ("Each item", ((MatchSpec.idname, "Match", 'CHECKMARK'), (ClassifySpec.idname, "Classify", 'SORTALPHA'))),
)


class NODE_MT_openact_add(bpy.types.Menu):
    bl_idname = "NODE_MT_openact_add"
    bl_label = "OpenAct"

    def draw(self, context):
        layout = self.layout
        for i, (group, entries) in enumerate(MENU):
            if i:
                layout.separator()
            layout.label(text=group)
            for idname, label, icon in entries:
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


classes = (GeometryNodeOpenActAsk, GeometryNodeOpenActPick, GeometryNodeOpenActMatch, GeometryNodeOpenActClassify,
           NODE_MT_openact_add)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.NODE_MT_add.append(draw_add_menu)


def unregister():
    bpy.types.NODE_MT_add.remove(draw_add_menu)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
