"""The node specs, one per OpenAct node type: SPECS (bl_idname -> spec) is the one registry."""

from .ask import AskSpec
from .dials import DialsSpec
from .pick import PickSpec

SPECS = {spec.idname: spec for spec in (AskSpec(), PickSpec(), DialsSpec())}
