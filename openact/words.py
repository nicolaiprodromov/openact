"""A node's words as Geometry Nodes builds them: the key text, its hash, and the Warning message (§5.7, §8.2).

Every inner group joins its live words, in its spec's WORDS order, into the key text: each word as its length in code
points, a colon and the word itself. The lengths make the text unambiguous whatever the words hold, so no word needs
escaping. Hash Value (String) of the key text is the key the lookup finds the entry by (§8.2), and the group's Warning
node reports the key text with the node's uid and its Trigger (§5.7), which Python reads from the modifier after the
evaluation. This module is the Python side of both.

Pure Python (no bpy): the pure suites check it against hashes that Blender computed.
"""

from __future__ import annotations

from dataclasses import dataclass

MARK = "OpenAct"  # the first word of every message an inner group reports
_M = 0xFFFFFFFF


def key_text(words) -> str:
    """The key text of a word tuple: "<length>:<word>" per word, lengths in code points as String Length counts."""
    return "".join(f"{len(word)}:{word}" for word in words)


def split(text: str) -> tuple[str, ...] | None:
    """The words of a key text, or None when the text isn't one (a stray message)."""
    words, at = [], 0
    while at < len(text):
        colon = text.find(":", at)
        if colon <= at or not text[at:colon].isdigit() or not text[at:colon].isascii():
            return None
        end = colon + 1 + int(text[at:colon])
        if end > len(text):
            return None
        words.append(text[colon + 1:end])
        at = end
    return tuple(words)


def blender_hash(text: str, seed: int = 0) -> int:
    """Blender's Hash Value node on a String, noise::hash(BLI_hash_string(text), seed), as a signed 32-bit Int.

    BLI_hash_string() multiplies by 37 and adds each byte of the UTF-8 text, unsigned, until the first NUL; the words
    Geometry Nodes passes never hold one. The mix is Bob Jenkins' final() of lookup3, as noise::hash() applies it."""
    pre = 0
    for byte in text.encode("utf-8"):
        if byte == 0:
            break
        pre = (pre * 37 + byte) & _M
    a = b = c = (0xDEADBEEF + (2 << 2) + 13) & _M
    a = (a + pre) & _M
    b = (b + seed) & _M
    c ^= b
    c = (c - _rotate(b, 14)) & _M
    a ^= c
    a = (a - _rotate(c, 11)) & _M
    b ^= a
    b = (b - _rotate(a, 25)) & _M
    c ^= b
    c = (c - _rotate(b, 16)) & _M
    a ^= c
    a = (a - _rotate(c, 4)) & _M
    b ^= a
    b = (b - _rotate(a, 14)) & _M
    c ^= b
    c = (c - _rotate(b, 24)) & _M
    return c - (1 << 32) if c >> 31 else c


def _rotate(x, k):
    return ((x << k) | (x >> (32 - k))) & _M


def key(words) -> int:
    """The key of a word tuple, as the inner group's Hash Value computes it (§8.2)."""
    return blender_hash(key_text(words))


def shared_keys(gns) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    """(kept, refused) for each word tuple whose key an earlier, different one already has: the lookup would find the
    earlier entry for both, so the later one is refused (§8.2). In the order given, duplicates ignored."""
    seen, found = {}, []
    for gn in dict.fromkeys(tuple(gn) for gn in gns):
        other = seen.setdefault(key(gn), gn)
        if other != gn:
            found.append((other, gn))
    return found


@dataclass(frozen=True)
class Report:
    """What one message says: the node, whether its Trigger was on, and the words it evaluated."""

    uid: str
    triggered: bool
    words: tuple[str, ...]


def message(uid: str, triggered: bool, words) -> str:
    """The message an inner group reports, "OpenAct <uid> <0|1> <key text>" (§5.7): what its Warning node builds."""
    return f"{MARK} {uid} {int(bool(triggered))} {key_text(words)}"


def parse(text: str) -> Report | None:
    """The Report in a message, or None for any other warning (a user's own Warning node, a Blender message)."""
    parts = text.split(" ", 3)
    if len(parts) != 4 or parts[0] != MARK or not parts[1] or parts[2] not in ("0", "1"):
        return None
    words = split(parts[3])
    return None if words is None else Report(parts[1], parts[2] == "1", words)
