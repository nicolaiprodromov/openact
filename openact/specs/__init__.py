"""The node specs, one per OpenAct node type: SPECS (bl_idname -> spec) is the one registry."""

from .ask import AskSpec
from .pick import PickSpec
from .rules import RulesSpec

SPECS = {spec.idname: spec for spec in (AskSpec(), PickSpec(), RulesSpec())}
