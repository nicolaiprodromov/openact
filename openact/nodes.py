"""The six OpenAct nodes: Ask, Pick and Dials (whole situation), Match and Classify (each item), Rules (each
situation)."""

from __future__ import annotations

import bpy

from . import engine, inner, sources, ui
from . import prefs as P
from .specs import dials
from .specs.classify import ClassifySpec
from .specs.match import MatchSpec
from .specs.rules import RulesSpec


def _changed(self, context):
    engine.schedule(immediate=True, node=self)


def _interface_changed(self, context):
    # the dropdown swaps sockets at once (§6.0); the pass rebuilds the group, as update_values() sees the change
    if self.node_tree is not None:
        inner.sync_interface(self, engine.SPECS[self.bl_idname].interface(self))
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
        name="List Source", items=sources.LIST_SOURCE_ITEMS, default='TEXT', update=_interface_changed,
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
        """One line on the node; with full, the whole status record (§17.1)."""
        status = engine.STATUS.get(self.uid)
        if not full:
            layout.label(text=status.text if status else "Waiting...", icon=status.icon if status else 'SORTTIME')
            return
        col = layout.column(align=True)
        for icon, text in ui.status_rows(status):
            ui.wrap(col, text, icon=icon)

    def draw_answers(self, layout, uses, selected):
        """One summary per context; with several, each one a button that shows its table."""
        spec = engine.SPECS[self.bl_idname]
        sent = getattr(engine.STATUS.get(self.uid), "asking", frozenset())  # the cells the last pass asks the model
        box = layout.box()
        for i, (label, job, answer) in enumerate(uses):
            col = box.column(align=True)
            icon = 'MODIFIER' if not label.startswith("(") else 'NODETREE'
            if len(uses) > 1:
                op = col.operator("openact.table_view", text=label, icon=icon, depress=i == selected)
                op.context_index, op.page = i, 0
            else:
                col.label(text=label, icon=icon)
            # no table, or a table so far with no answer yet (a Pick's Index is -1): "Asking..." while the pass asks one
            # of its cells, else it waits, disagrees or failed
            if answer is None or ("p" in answer.rows.src and not {"m", "s"} & set(answer.rows.src)):
                asking = bool(sent) and any(row.cell is not None and row.cell.key in sent for row in job.rows)
                col.label(text="Asking..." if asking else "Not asked: see the status above",
                          icon='SORTTIME' if asking else 'INFO')
                continue
            col.label(text=spec.summary(answer), icon='CHECKMARK')
            if engine.older_settings(self, answer):
                col.label(text="Asked with older settings: Ask Again refreshes it", icon='BLANK1')
            if self.bl_idname == engine.PICK_ID:  # the top 5 items, without empty material slots
                named = [(name, p) for name, p in zip(answer.rows.name, answer.rows.v["prob"]) if name]
                for name, probability in sorted(named, key=lambda pair: -pair[1])[:5]:
                    row = col.row()
                    row.label(text=name, icon='BLANK1')
                    row.label(text=f"{probability * 100:.1f}%")
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
        uses = engine.answers_for(self)
        selected = min(context.window_manager.openact_context, len(uses) - 1) if uses else 0
        if uses:
            self.draw_answers(layout, uses, selected)
            ui.draw_table(layout, context, self, uses, selected)
        row = layout.row()
        row.enabled = self.id_data.library is None  # a linked node's answers come from its library file
        row.operator("openact.reask", icon='FILE_REFRESH').uid = self.uid
        op = row.operator("openact.write_overrides", icon='EXPORT')
        op.uid, op.context_index = self.uid, selected


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

    # Set by the upgrade from v0.1.0: its old Items texts (a JSON list) keep v0.1.0's parsing until another text comes
    # (lists.py, §12.3)
    items_parse_v0: bpy.props.BoolProperty(options={'HIDDEN'})
    items_v0_texts: bpy.props.StringProperty(options={'HIDDEN'}, default="[]")

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
    ("Whole situation", ((engine.ASK_ID, "Ask", 'QUESTION'), (engine.PICK_ID, "Pick", 'RESTRICT_SELECT_OFF'),
                         (dials.DialsSpec.idname, "Dials", 'OPTIONS'))),
    ("Each item", ((MatchSpec.idname, "Match", 'CHECKMARK'), (ClassifySpec.idname, "Classify", 'SORTALPHA'))),
    ("Each situation", ((RulesSpec.idname, "Rules", 'PRESET'),)),
)


class GeometryNodeOpenActDials(_OpenActNode, bpy.types.GeometryNodeCustomGroup):
    """Judge many named settings against one prompt, in one request: one output per setting"""

    bl_idname = dials.DialsSpec.idname
    bl_label = "Dials"


class GeometryNodeOpenActRules(_OpenActNode, bpy.types.GeometryNodeCustomGroup):
    """Ask one question per combination of factor levels, then look up each element's cell"""

    bl_idname = RulesSpec.idname
    bl_label = "Rules"

    answer_type: bpy.props.EnumProperty(
        name="Answer",
        items=(
            ('ONE_OF', "One of", "The outcome that applies to each element's cell"),
            ('YES_NO', "Yes / No", "Probability of yes, blended across up to 3 Amount factors"),
            ('AMOUNT', "Amount", "How much (0-1), blended across up to 3 Amount factors"),
        ),
        default='ONE_OF',
        update=_interface_changed,  # the outputs, and Dither and Seed, follow the answer type
    )

    def draw_settings(self, layout):
        layout.row().prop(self, "answer_type", expand=True)
        super().draw_settings(layout)


LEGACY_STATUS = "Old Jev node: upgrade in the OpenAct sidebar"


class _LegacyNode:
    """An old Jev node, loaded from a jev_nodes v0.1.0 file as a defined node (§12.3). Frozen: it keeps evaluating the
    inner group it was saved with and is never asked anything, until openact.upgrade_legacy replaces it."""

    bl_width_default = 240
    bl_width_min = 160

    uid: bpy.props.StringProperty(options={'HIDDEN'})
    entries: bpy.props.StringProperty(options={'HIDDEN'}, default="[]")

    @classmethod
    def poll(cls, ntree):
        return False  # never in the Add menu: only old files bring them

    def poll_instance(self, ntree):
        return True  # else the default, poll(), would make a node group that holds one invalid in its parent

    def draw_buttons(self, context, layout):
        layout.label(text=LEGACY_STATUS, icon='INFO')


class GeometryNodeJevAsk(_LegacyNode, bpy.types.GeometryNodeCustomGroup):
    """An old Ask Jev node: upgrade it in the OpenAct sidebar"""

    bl_idname = "GeometryNodeJevAsk"
    bl_label = "Ask Jev"

    mode: bpy.props.EnumProperty(  # v0.1.0's items, in its order: files store the position
        name="Answer", items=(('YES_NO', "Yes / No", ""), ('AMOUNT', "Amount", "")), default='YES_NO')


class GeometryNodeJevPick(_LegacyNode, bpy.types.GeometryNodeCustomGroup):
    """An old Jev Pick node: upgrade it in the OpenAct sidebar"""

    bl_idname = "GeometryNodeJevPick"
    bl_label = "Jev Pick"


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


classes = (GeometryNodeOpenActAsk, GeometryNodeOpenActPick, GeometryNodeOpenActDials, GeometryNodeOpenActMatch,
           GeometryNodeOpenActClassify, GeometryNodeOpenActRules, NODE_MT_openact_add)
legacy_classes = (GeometryNodeJevAsk, GeometryNodeJevPick)


def hold_legacy_types():
    """Register the aliases of the old Jev node types that no class holds (§12.2): jev_nodes' own classes win while it
    is enabled, and when it unregisters, or is listed but failed to load, the aliases take over, so the upgrade is
    offered. A timer, once a second: classes can't be registered while Blender draws."""
    for cls in legacy_classes:
        if bpy.types.Node.bl_rna_get_subclass_py(cls.bl_idname) is None:
            bpy.utils.register_class(cls)
    return 1.0


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    hold_legacy_types()
    if not bpy.app.timers.is_registered(hold_legacy_types):
        bpy.app.timers.register(hold_legacy_types, first_interval=1.0, persistent=True)
    bpy.types.NODE_MT_add.append(draw_add_menu)


def unregister():
    bpy.types.NODE_MT_add.remove(draw_add_menu)
    if bpy.app.timers.is_registered(hold_legacy_types):
        bpy.app.timers.unregister(hold_legacy_types)
    for cls in reversed(legacy_classes):
        if cls.is_registered:  # jev_nodes, enabled later, replaced it
            bpy.utils.unregister_class(cls)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
