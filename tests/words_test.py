"""A node's words as Geometry Nodes builds them, openact/words.py: the key text, the hash replica, the messages of the
Warning channel, and the words that share a key (§5.7, §8.2).

No bpy: the hashes below are the ones Blender's Hash Value node computed, on 4.5 and 5.1 alike.

Run:  python tests/words_test.py
      blender -b --factory-startup --python tests/words_test.py
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
from openact import words  # noqa: E402

LONG = ("The quick brown fox jumps over the lazy dog, 0123456789. " * 40)[:2000]  # the long text Blender hashed
MEASURED = (  # (text, seed, Hash Value's result in Blender)
    ("", 0, -599952135), ("", -1, 951247345), ("", 12345, 2135815942),
    ("a", 0, -2123640003), ("a", -1, 1885535868), ("a", 12345, -1261803330),
    ("Is it night?", 0, 130709544), ("Is it night?", -1, -1648578624), ("Is it night?", 12345, 899435134),
    ("A stormy night in a fishing village", 0, 1883907674), ("A stormy night in a fishing village", 12345, -1127428253),
    ("caf\u00e9", 0, -1008283445), ("\u96ea", 0, -1276599583), ("\U0001f327 rain", 0, -1129753792),
    (LONG, 0, 1679778669),
    ("aZ", 0, -539991699), ("b5", 0, -539991699), ("az", 0, -2008811345), ("bU", 0, -2008811345),
    ("aZ", -1, 1827813100), ("b5", -1, 1827813100),
)


def phase_hash():
    got = [(text[:20], seed, words.blender_hash(text, seed), expected) for text, seed, expected in MEASURED
           if words.blender_hash(text, seed) != expected]
    check("hash: the replica gives Blender's Hash Value for every measured text and seed", not got, got)
    check("hash: results are signed 32-bit Ints", all(-2**31 <= words.blender_hash(text, seed) < 2**31
                                                       for text, seed, _ in MEASURED))
    check("hash: a NUL ends the text, as in Blender's C loop", words.blender_hash("ab\x00cd") == words.blender_hash("ab"))
    check("hash: seeds re-mix the same pre-hash, so 'aZ' and 'b5' share a hash under every seed",
          all(words.blender_hash("aZ", seed) == words.blender_hash("b5", seed) for seed in (0, 1, 2, -1, 12345)))


def phase_key_text():
    cases = [(), ("",), ("Is it night?", "", "", "", ""), ("a:b", "12:x", ":"), ("caf\u00e9 \u96ea \U0001f327",),
             ("\x1f", "line\nbreak", "tab\t"), (LONG, "")]
    check("key text: each word as its length, a colon and the word",
          words.key_text(("Is it night?", "", "caf\u00e9")) == "12:Is it night?0:4:caf\u00e9")
    check("key text: lengths count code points, as String Length does", words.key_text(("\U0001f327",)) == "1:\U0001f327")
    check("key text: split() gives back every word tuple, whatever the words hold",
          all(words.split(words.key_text(gn)) == gn for gn in cases),
          [gn for gn in cases if words.split(words.key_text(gn)) != gn])
    check("key text: word tuples that join to the same text with a separator stay apart",
          words.key_text(("a\x1fb", "c")) != words.key_text(("a", "b\x1fc")))
    broken = ["x", "3:ab", ":ab", "-1:a", "1a:b", "\u0663:abc", "2:ab1"]
    check("key text: split() refuses text that isn't a key text", all(words.split(text) is None for text in broken),
          [text for text in broken if words.split(text) is not None])
    check("key text: key() is the hash of the key text",
          words.key(("Is it night?",)) == words.blender_hash("12:Is it night?"))


def phase_messages():
    gn = ("Is it night?", "A beach at noon", "", "", "caf\u00e9 \u96ea")
    text = words.message("3f2a9c1b4d5e", True, gn)
    check("messages: 'OpenAct <uid> <0|1> <key text>'",
          text == "OpenAct 3f2a9c1b4d5e 1 " + words.key_text(gn), text)
    check("messages: parse() reads back the uid, the Trigger and the words",
          words.parse(text) == words.Report("3f2a9c1b4d5e", True, gn)
          and words.parse(words.message("ab", False, ())) == words.Report("ab", False, ()))
    check("messages: words with spaces and colons survive", words.parse(words.message("u", 0, ("a b: c", " "))).words
          == ("a b: c", " "))
    stray = ["", "OpenAct", "OpenAct uid 1", "OpenAct  1 0:", "OpenAct uid 2 0:", "OpenAct uid 1 3:ab",
             "Openact uid 1 0:", "Input geometry has unsupported type", "OpenAct uid 1 x"]
    check("messages: any other warning is not a report", all(words.parse(text) is None for text in stray),
          [text for text in stray if words.parse(text) is not None])


def phase_shared_keys():
    check("shared keys: none among ordinary sentences",
          words.shared_keys([("Is it night?", ""), ("Is it day?", ""), ("Is it night?", "")]) == [])
    found = words.shared_keys([("aZ",), ("Is it night?",), ("b5",), ("aZ",)])
    check("shared keys: a later tuple with an earlier one's key is refused, the earlier kept",
          found == [(("aZ",), ("b5",))], found)
    check("shared keys: the key texts of 'aZ' and 'b5' collide too", words.key(("aZ",)) == words.key(("b5",)))


phase_hash()
phase_key_text()
phase_messages()
phase_shared_keys()
finish()
