"""Port of Blender's ``BLI_strcasecmp_natural()``.

Collection Info (with Separate Children) sorts a collection's objects and child collections with
this comparison, so Pick sorts the same way and its Index lines up with Collection Info and
Instance on Points > Pick Instance. Bytes compare unsigned, which is what Blender does in
practice (verified: non-ASCII names sort after '~').
"""

from __future__ import annotations

import functools


def _is_digit(byte):
    return 48 <= byte <= 57


def _lower(byte):
    return byte + 32 if 65 <= byte <= 90 else byte


def _left_number_compare(s1, i1, s2, i2, tiebreaker):
    p1, p2 = i1, i2
    zeros1 = zeros2 = 0
    while p1 < len(s1) and s1[p1] == 48:
        p1 += 1
        zeros1 += 1
    while p2 < len(s2) and s2[p2] == 48:
        p2 += 1
        zeros2 += 1
    digits = 0
    while True:
        d1 = p1 + digits < len(s1) and _is_digit(s1[p1 + digits])
        d2 = p2 + digits < len(s2) and _is_digit(s2[p2 + digits])
        if d1 and d2:
            digits += 1
            continue
        if d1:
            return 1
        if d2:
            return -1
        break
    if digits:
        a, b = s1[p1:p1 + digits], s2[p2:p2 + digits]
        if a != b:
            return -1 if a < b else 1
    if tiebreaker[0] == 0:
        if zeros1 > zeros2:
            tiebreaker[0] = 1
        elif zeros1 < zeros2:
            tiebreaker[0] = -1
    return 0


def compare(a: str, b: str) -> int:
    s1, s2 = a.encode("utf-8"), b.encode("utf-8")
    n1, n2 = len(s1), len(s2)
    d1 = d2 = 0
    tiebreaker = [0]
    while True:
        if d1 < n1 and d2 < n2 and _is_digit(s1[d1]) and _is_digit(s2[d2]):
            result = _left_number_compare(s1, d1, s2, d2, tiebreaker)
            if result:
                return result
            d1 += 1
            while d1 < n1 and _is_digit(s1[d1]):
                d1 += 1
            d2 += 1
            while d2 < n2 and _is_digit(s2[d2]):
                d2 += 1
        if d1 >= n1 or d2 >= n2:
            break
        c1, c2 = _lower(s1[d1]), _lower(s2[d2])
        if c1 != c2:
            if c1 == 46:  # '.' so "foo.bar" sorts before "foo 1.bar"
                return -1
            if c2 == 46:
                return 1
            return -1 if c1 < c2 else 1
        d1 += 1
        d2 += 1
    if tiebreaker[0]:
        return tiebreaker[0]
    return (s1 > s2) - (s1 < s2)


sort_key = functools.cmp_to_key(compare)
