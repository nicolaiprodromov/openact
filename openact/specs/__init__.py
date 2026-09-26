"""The node specs, one per OpenAct node type: SPECS (bl_idname -> spec) is the one registry."""

from .ask import AskSpec
from .classify import ClassifySpec
from .dials import DialsSpec
from .match import MatchSpec
from .pick import PickSpec

SPECS = {spec.idname: spec for spec in (AskSpec(), PickSpec(), MatchSpec(), ClassifySpec(), DialsSpec())}
