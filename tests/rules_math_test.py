"""The maths behind Rules and every confidence, with no Blender and no model: so far the three confidence formulas
of 4.e (specs.base.confidence). The Rules cell index, interpolation and dithering references join them later.

A know-nothing answer (p = 0.5, a uniform choice or scale) gives exactly 0 and a certain one 1, for 2 to 16 levels
or options, so one Unsure below threshold means the same for every answer type; near-uniform answers stay below the
0.30 default; and the documented examples hold.

Run:  python tests/rules_math_test.py
      blender -b --factory-startup --python tests/rules_math_test.py
"""

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import ROOT, check, finish  # noqa: E402

# openact/__init__.py imports bpy, so the pure modules are loaded without running it.
_package = types.ModuleType("openact")
_package.__path__ = [os.path.join(ROOT, "openact")]
sys.modules["openact"] = _package
from openact.specs.base import confidence  # noqa: E402

LEVELS = range(2, 17)
UNSURE_BELOW = 0.30  # the preference's default


def near_uniform(levels, weight=0.1):
    """A uniform answer with `weight` of it moved onto the middle level or option."""
    distribution = [(1 - weight) / levels] * levels
    distribution[levels // 2] += weight
    return distribution


def phase_endpoints():
    for kind in ('AMOUNT', 'ONE_OF'):
        uniform = {levels: confidence(kind, [1 / levels] * levels) for levels in LEVELS}
        check(f"{kind}: a uniform answer gives exactly 0, for L = 2..16", set(uniform.values()) == {0.0}, uniform)
        certain = {(levels, at): confidence(kind, [float(i == at) for i in range(levels)])
                   for levels in LEVELS for at in (0, levels // 2, levels - 1)}
        check(f"{kind}: a certain answer gives exactly 1, on any level", set(certain.values()) == {1.0},
              {key: value for key, value in certain.items() if value != 1.0})
        near = {levels: confidence(kind, near_uniform(levels)) for levels in LEVELS}
        check(f"{kind}: near-uniform answers stay below {UNSURE_BELOW}", 0 < max(near.values()) < UNSURE_BELOW, near)
    got = [confidence('YES_NO', (p, 1 - p)) for p in (0.5, 0.0, 1.0, 0.45)]
    check("YES_NO: p = 0.5 gives 0, p = 0 or 1 gives 1, and near 0.5 stays below the threshold",
          got[:3] == [0.0, 1.0, 1.0] and 0 < got[3] < UNSURE_BELOW, got)


def phase_examples():
    between = [0, 0.45, 0.55, 0, 0]
    got = confidence('AMOUNT', between), confidence('ONE_OF', between)
    check("AMOUNT: an answer between two neighbouring levels scores 0.65 with sigma, 0.10 as top minus runner-up",
          (round(got[0], 2), round(got[1], 2)) == (0.65, 0.10), got)
    got = confidence('AMOUNT', [0.5, 0, 0, 0, 0.5])
    check("AMOUNT: a split between the two ends scores 0 (its sigma 2 exceeds the uniform 1.41)", got == 0.0, got)
    got = confidence('YES_NO', (0.93, 0.07)), confidence('ONE_OF', [0.1, 0.7, 0.2])
    check("YES_NO |2p - 1| and ONE_OF top minus runner-up", got == (0.86, 0.5), got)
    got = confidence('ONE_OF', [2, 1, 1]), confidence('AMOUNT', []), confidence('ONE_OF', [1.0])
    check("unnormalised answers are normalised; no options give 0 and one option 1", got == (0.25, 0.0, 1.0), got)


phase_endpoints()
phase_examples()
finish()
