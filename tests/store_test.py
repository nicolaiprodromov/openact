"""The node store, openact/store.py: the schema-2 entries JSON, the v0.1.0 conversion and eviction.

No bpy and no model. tests/fixtures/jev_v010_entries.json holds the exact entries strings that the
v0.1.0 prototype's set_entries() wrote for Ask Yes/No, Ask Amount, Pick from text and Pick from a
collection, once with an empty Items text and once with Items text that the collection overrode.

Run:  python tests/store_test.py
      blender -b --factory-startup --python tests/store_test.py
"""

import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import ROOT, check, finish  # noqa: E402


def import_store():
    """openact/store.py on its own: importing the openact package needs bpy."""
    spec = importlib.util.spec_from_file_location("openact_store", os.path.join(ROOT, "openact", "store.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look up the module of the classes they build
    spec.loader.exec_module(module)
    return module


store = import_store()
NodeStore, MB, SOFT_CAP = store.NodeStore, store.MB, store.SOFT_CAP
EMPTY = '{"schema":2,"entries":[]}'
KEPT = "live and pinned contexts are kept; pin fewer word sets to shrink the file"
with open(os.path.join(ROOT, "tests", "fixtures", "jev_v010_entries.json"), encoding="utf-8") as handle:
    V010 = json.load(handle)


class Node:
    """Stands in for an OpenAct node: the store only reads and writes its entries string."""

    def __init__(self, entries="[]"):
        self.entries = entries


def size_of(node):
    return len(node.entries.encode("utf-8"))


def mb(size):
    """Megabytes as the notes give them: rounded up to 0.1."""
    return f"{math.ceil(size * 10 / MB) / 10:.1f}"


def words(entries):
    return [entry.gn[0] for entry in entries]


def gn(first):
    return (first, "", "//factors.csv", "//outcomes.csv", "", "")


def saving(node, kept, live_gns=None):
    """The ValueError that save() raises, or None."""
    try:
        store.save(node, kept, live_gns)
    except ValueError as error:
        return error
    return None


def loading(text):
    """The first words of the entries that load() returns, or the exception it raises."""
    try:
        return words(store.load(Node(text)).entries)
    except Exception as error:  # noqa: BLE001  (a raising load() is the failure being checked)
        return error


def rules(first, t, rows, pinned=False):
    """A Rules One-of entry with `rows` rows, last used at t."""
    names = [f"level {r // 64}|level {r % 64}" for r in range(rows)]
    return store.Entry(
        words_key=hashlib.sha1(first.encode()).hexdigest(), asks=["ask"], pv=1, t=t, pinned=pinned,
        frames=[[1, 250]] if pinned else [], gn=gn(first), model="fake", kind="rules", answer_type="ONE_OF",
        shape={"rows": rows, "columns": {"outcome": "INT", "c": "FLOAT"}},
        rows=store.Rows(name=names, wkey=[hashlib.sha1(name.encode()).hexdigest() for name in names],
                        ask=[0] * rows, src="m" * rows,
                        v={"outcome": [r % 7 for r in range(rows)], "c": [1 - r / (2 * rows) for r in range(rows)]}))


def life():
    """Every schema-2 field set: a pinned Game of Life entry with payload and top answers."""
    names = [f"{state}|{n}" for state in ("dead", "alive") for n in range(9)]
    outcome = [0 if n == 3 else 1 for n in range(9)] + [2 if n in (2, 3) else 3 for n in range(9)]
    return store.Entry(
        words_key=hashlib.sha1(b"life").hexdigest(), asks=["b77a", "c0de"], pv=1, t=1790000123.4, pinned=True,
        frames=[[1, 48], [60, 90]],
        gn=("What happens to this cell in the next generation?", "A cellular automaton.", "//life_factors.csv",
            "//life_outcomes.csv", "", ""),
        model="semif-torch Qwen/Qwen3.5-4B@851bf6e8 debias", kind="rules", answer_type="ONE_OF",
        shape={"rows": 18, "columns": {"outcome": "INT", "c": "FLOAT"}, "payload": [["alive", "INT"]],
               "outcomes": ["Birth", "Stays empty", "Survives", "Dies"],
               "factors": [{"name": "state", "kind": "flag", "levels": ["dead", "alive"], "values": [0, 1]},
                           {"name": "neighbours", "kind": "category", "levels": [str(n) for n in range(9)],
                            "values": list(range(9)), "contiguous": True}]},
        rows=store.Rows(name=names, wkey=[hashlib.sha1(name.encode()).hexdigest() for name in names],
                        ask=[0] * 17 + [1], src="m" * 16 + "cl", v={"outcome": outcome, "c": [0.99] * 18},
                        payload={"alive": [int(o in (0, 2)) for o in outcome]},
                        top=[[[o, 0.995], [(o + 1) % 4, 0.005]] for o in outcome]),
        csv={"//life_factors.csv": ["a41f", "0000"], "//life_outcomes.csv": ["9c2e", "77b1"]})


def phase_conversion():
    for case in ("ask_yes_no", "ask_amount", "pick_text", "pick_collection", "pick_collection_and_text"):
        old = json.loads(V010[case])
        node = Node(V010[case])
        new = store.load(node).entries
        pairs = list(zip(new, old))
        check(f"{case}: every v0.1.0 entry is converted", len(new) == len(old) > 0, len(new))
        check(f"{case}: load does not rewrite the v0.1.0 JSON", node.entries == V010[case])
        check(f"{case}: legacy, pv 0, model and ask key 'jev_nodes 0.1.0', t kept, not pinned",
              all((e.legacy, e.pv, e.model, e.asks, e.t, e.pinned) ==
                  (True, 0, "jev_nodes 0.1.0", ["jev_nodes 0.1.0"], o["t"], False) for e, o in pairs))
        if case.startswith("ask"):
            mode = "YES_NO" if case == "ask_yes_no" else "AMOUNT"
            check(f"{case}: gn (Question, Context) becomes (Question, Context, '', '', '')",
                  all(e.gn == (o["gn"][0], o["gn"][1], "", "", "") for e, o in pairs), [e.gn for e in new])
            check(f"{case}: kind ask, answer type {mode}, no rows",
                  all((e.kind, e.answer_type, e.rows) == ("ask", mode, store.Rows()) for e in new))
            check(f"{case}: single holds Answer, Yes and Confidence",
                  all(e.single == {"Answer": o["a"]["value"], "Yes": o["a"]["value"] >= 0.5,
                                   "Confidence": o["a"]["confidence"]} for e, o in pairs), [e.single for e in new])
        else:
            check(f"{case}: gn (Prompt, Items) becomes (Prompt, '', Items, '', ''), with no Items for a collection",
                  all(e.gn == (o["gn"][0], "", "" if o["a"]["collection"] else o["gn"][1], "", "") for e, o in pairs),
                  [e.gn for e in new])
            check(f"{case}: kind pick, answer type ONE_OF",
                  all((e.kind, e.answer_type) == ("pick", "ONE_OF") for e in new))
            check(f"{case}: single holds Index, Item and Confidence",
                  all(e.single == {"Index": o["a"]["index"], "Item": o["a"]["item"],
                                   "Confidence": o["a"]["confidence"]} for e, o in pairs), [e.single for e in new])
            check(f"{case}: one model row per item, with its probability",
                  all((e.rows.name, e.rows.v, e.rows.src, e.rows.ask, len(e.rows.wkey)) ==
                      (o["a"]["items"], {"prob": o["a"]["probabilities"]}, "m" * len(o["a"]["items"]),
                       [0] * len(o["a"]["items"]), len(o["a"]["items"])) for e, o in pairs),
                  [(e.rows.name, e.rows.src) for e in new])
        saved = Node()
        store.save(saved, store.load(node))
        data = json.loads(saved.entries)
        check(f"{case}: saved as schema 2, without the old key",
              data["schema"] == 2 and all("key" not in entry for entry in data["entries"]), saved.entries[:60])
        check(f"{case}: converted entries survive save and load", store.load(saved).entries == new)

    answers = [e.single["Yes"] for e in store.load(Node(V010["ask_yes_no"])).entries]
    check("ask_yes_no: the fixture has a Yes and a No", answers == [True, False], answers)
    for case, prompt, text in (("pick_collection", "something that gives light", ""),
                               ("pick_collection_and_text", "something to put a cup on", "tray, saucer")):
        stored = json.loads(V010[case])[0]["gn"][1]
        new = [(e.gn, e.rows.name) for e in store.load(Node(V010[case])).entries]
        check(f"{case}: the Items text {text!r} is left out, and the rows are the collection's children",
              stored == text and new == [((prompt, "", "", "", ""),
                                          ["Armchair", "Bookshelf", "Floor Lamp", "Rug", "Table"])], new)


def phase_round_trip():
    ask = store.load(Node(V010["ask_amount"])).entries[1]  # entry-level values and a non-ASCII word
    full = NodeStore([life(), ask])
    node = Node()
    store.save(node, full)
    check("round trip: load returns what save wrote", store.load(node) == full)
    data = json.loads(node.entries)
    pinned, plain = data["entries"]
    check("round trip: the documented schema-2 fields, in order",
          data["schema"] == 2 and list(pinned) == ["words_key", "asks", "pv", "t", "legacy", "pinned", "frames", "gn",
                                                   "model", "kind", "answer_type", "shape", "rows", "single", "csv"]
          and list(pinned["rows"]) == ["name", "wkey", "ask", "src", "v", "payload", "top"], list(pinned))
    check("round trip: frames only on pinned entries, top only when given",
          "frames" not in plain and "top" not in plain["rows"], list(plain))
    check("round trip: non-ASCII words are stored as they are", "\u2014" in node.entries)


def phase_max_entries():
    live = [rules(f"live {i}", 100 + i, 64) for i in range(18)]  # more live entries than MAX_ENTRIES
    pinned = rules("pinned", 0, 64, pinned=True)  # the least recently used of all
    old = [rules(f"old {i}", 10 + (7 * i) % 20, 64) for i in range(20)]
    entries = live[:9] + old[:10] + [pinned] + live[9:] + old[10:]
    live_gns = frozenset(gn(f"live {i}") for i in range(18))
    recent = set(words(sorted(old, key=lambda e: e.t)[-store.MAX_ENTRIES:]))
    expected = [first for first in words(entries) if not first.startswith("old") or first in recent]
    kept, node = NodeStore(list(entries)), Node()
    store.save(node, kept, live_gns)
    check("16 entries: keeps every live and pinned entry and the 16 most recent others, in order",
          words(kept.entries) == expected, words(kept.entries))
    check("16 entries: the node holds exactly what save() kept", store.load(node).entries == kept.entries)
    check("16 entries: no note", kept.note == "", kept.note)
    kept = NodeStore(list(entries))
    store.save(node, kept)
    check("16 entries with live_gns=None: nothing is evicted",
          kept.entries == entries and store.load(node).entries == entries, len(kept.entries))


def phase_soft_cap():
    live = rules("live", 1, 3000)
    pinned = rules("pinned", 2, 3000, pinned=True)
    old = [rules(f"old {i}", 10 + (5 * i) % 8, 3000) for i in range(8)]
    entries = [old[0], live, *old[1:5], pinned, *old[5:]]
    whole, alone = Node(), Node()
    store.save(whole, NodeStore(list(entries)))
    store.save(alone, NodeStore([live, pinned]))
    check("soft cap: the store exceeds 2 MB, its live and pinned entries do not",
          size_of(whole) > SOFT_CAP > size_of(alone), (size_of(whole), size_of(alone)))
    kept, node = NodeStore(list(entries)), Node()
    store.save(node, kept, frozenset({gn("live")}))
    by_age = words(sorted(old, key=lambda e: e.t))
    dropped = len(old) - sum(first.startswith("old") for first in words(kept.entries))
    check("soft cap: keeps live and pinned, evicts the least recently used others first",
          0 < dropped < len(old) and set(words(kept.entries)) == {"live", "pinned", *by_age[dropped:]},
          words(kept.entries))
    check("soft cap: kept entries keep their order",
          words(kept.entries) == [first for first in words(entries) if first in words(kept.entries)])
    check("soft cap: the node fits 2 MB", size_of(node) <= SOFT_CAP, size_of(node))
    if dropped:
        probe = Node()
        store.save(probe, NodeStore(kept.entries + [next(e for e in old if e.gn[0] == by_age[dropped - 1])]))
        check("soft cap: evicts no more than needed", size_of(probe) > SOFT_CAP, size_of(probe))
    check("soft cap: the note counts the dropped contexts",
          kept.note == f"{dropped} old context{'s' if dropped > 1 else ''} dropped to keep the file small", kept.note)


def phase_both_caps():
    live = rules("live", 1, 2000)
    old = [rules(f"old {i}", 10 + (7 * i) % 20, 2000) for i in range(20)]
    by_age = sorted(old, key=lambda e: e.t)
    recent = Node()
    store.save(recent, NodeStore([live, *by_age[-store.MAX_ENTRIES:]]))
    check("both caps: live and the 16 most recent others still exceed 2 MB", size_of(recent) > SOFT_CAP,
          size_of(recent))
    kept, node = NodeStore(old[:10] + [live] + old[10:]), Node()
    store.save(node, kept, frozenset({gn("live")}))
    others = [e for e in kept.entries if e.gn[0] != "live"]
    check("both caps: keeps live and the most recent others that fit",
          "live" in words(kept.entries) and 0 < len(others) < store.MAX_ENTRIES
          and set(words(others)) == set(words(by_age[-len(others):])), words(kept.entries))
    check("both caps: the node fits 2 MB", size_of(node) <= SOFT_CAP, size_of(node))
    probe = Node()
    store.save(probe, NodeStore(kept.entries + [by_age[-len(others) - 1]]))
    check("both caps: evicts no more than needed", size_of(probe) > SOFT_CAP, size_of(probe))
    drops = store.MAX_ENTRIES - len(others)
    check("both caps: the note counts only the drops for 2 MB",
          kept.note == f"{drops} old context{'s' if drops > 1 else ''} dropped to keep the file small", kept.note)


def phase_precedence():
    live = [rules(f"live {i}", 10 + i, 4096) for i in range(4)]
    pinned = [rules(f"pinned {i}", 1 + i, 4096, pinned=True) for i in range(4)]
    others = [rules(f"old {i}", t, rows) for i, (t, rows) in enumerate(
        ((50, 64), (60, 4096), (5, 64), (70, 64), (40, 4096), (55, 64)))]  # most are more recent than live
    entries = [others[0], live[0], pinned[0], others[1], live[1], live[2], pinned[1], others[2], pinned[2],
               others[3], live[3], pinned[3], others[4], others[5]]
    live_gns = frozenset(gn(f"live {i}") for i in range(4))  # built from the words, as the pass will
    alone, both = Node(), NodeStore(live + pinned)
    store.save(alone, both, live_gns)
    check("precedence: live and pinned entries alone exceed 2 MB", size_of(alone) > SOFT_CAP, size_of(alone))
    check("precedence: with nothing to drop, the note gives the size",
          both.note == f"entries use {mb(size_of(alone))} MB: {KEPT}", both.note)

    kept, node = NodeStore(list(entries)), Node()
    store.save(node, kept, live_gns)
    check("precedence: every live and pinned entry is kept, and every warm and cold one evicted",
          words(kept.entries) == [first for first in words(entries) if not first.startswith("old")],
          words(kept.entries))
    check("precedence: the node holds exactly what save() kept", store.load(node).entries == kept.entries)
    check("precedence: the note reports the drops and the size",
          size_of(node) > SOFT_CAP
          and kept.note == f"{len(others)} old contexts dropped; entries still use {mb(size_of(node))} MB: {KEPT}",
          kept.note)

    kept, node = NodeStore(list(entries)), Node()
    store.save(node, kept)
    check("live_gns=None: the same over-limit store keeps every entry",
          kept.entries == entries and size_of(node) > SOFT_CAP, words(kept.entries))
    check("live_gns=None: the node holds every entry", store.load(node).entries == entries)
    check("live_gns=None: no note", kept.note == "", kept.note)


def phase_hard_cap():
    fits = store.Entry(words_key="w", gn=gn("fits"), kind="ask", single={"Item": "x" * (4 * MB - 1024)})
    huge = store.Entry(words_key="w", gn=gn("huge"), kind="ask", single={"Item": "x" * (4 * MB)})
    node = Node()
    store.save(node, NodeStore([fits]))
    check("hard cap: an entry within 4 MB is saved", store.load(node).entries == [fits], size_of(node))
    before, kept = node.entries, NodeStore([fits, huge])
    error = saving(node, kept, frozenset({gn("fits"), gn("huge")}))
    check("hard cap: an entry above 4 MB is an error, even when live, and its size rounds up",
          error is not None and "(4.1 MB; the limit is 4 MB)" in str(error), error)
    check("hard cap: then nothing is written or evicted", node.entries == before and kept.entries == [fits, huge])


def phase_damaged():
    for label, text in (("the node default '[]'", "[]"), ("an empty string", ""), ("broken JSON", '{"schema": 2, "en'),
                        ("a store without a schema", '{"entries": []}'), ("a JSON string", '"entries"')):
        check(f"damaged: {label} loads as an empty store", store.load(Node(text)) == NodeStore())

    ask, pick = (store.load(Node(V010[case])).entries[0] for case in ("ask_yes_no", "pick_text"))
    written = Node()
    store.save(written, NodeStore([ask, pick]))
    a, p = json.loads(written.entries)["entries"]
    damaged = [
        42, {"words_key": "no gn"}, dict(a, gn="Is it night?"), dict(a, gn=["Is it night?", 7]), dict(a, rows=[]),
        dict(a, pv="1e400"), dict(a, pv=float("inf")), dict(a, t=10 ** 400), dict(a, t=float("nan")),
        dict(a, t=float("inf")), dict(a, pinned="false"), dict(a, legacy=1), dict(a, words_key=None),
        dict(a, kind=["ask"]), dict(a, single=[]),
        *(dict(p, rows=dict(p["rows"], **change)) for change in (
            {"name": ["table", 7, "armchair", "bookshelf", "rug"]}, {"wkey": [None] * 5}, {"wkey": [""]},
            {"ask": [0, 0, 1, 0, 0]}, {"ask": [0, -1, 0, 0, 0]}, {"v": {"prob": [0.1, "x", 0.2, 0.3, 0.4]}},
            {"payload": {"alive": [0, 1, None, 0, 1]}}, {"top": []}))]
    text = json.dumps({"schema": 2, "entries": [a, *damaged, p]}).replace('"1e400"', "1e400")
    loaded = loading(text)
    check("damaged: schema 2 skips only the damaged entries, overflowing numbers included",
          loaded == words([ask, pick]) and store.load(Node(text)).entries == [ask, pick], loaded)

    old_ask, old_pick = (json.loads(V010[case])[0] for case in ("ask_yes_no", "pick_text"))
    damaged = ["junk", {"key": "k"}, dict(old_pick, gn=["one word"]),
               dict(old_pick, a=dict(old_pick["a"], probabilities=[1.0])),
               dict(old_pick, a=dict(old_pick["a"], index="1e400")), dict(old_pick, t=10 ** 400),
               dict(old_pick, t=float("nan")), dict(old_ask, a=dict(old_ask["a"], value=10 ** 400)),
               dict(old_ask, a=dict(old_ask["a"], mode=None)),
               # shapes that would unpack or iterate into well-typed but wrong words and numbers
               dict(old_pick, gn="pq"), dict(old_ask, gn={"Question": 1, "Context": 2}),
               dict(old_pick, a=dict(old_pick["a"], probabilities="12345")),
               dict(old_pick, a=dict(old_pick["a"], probabilities={str(i): 0.2 for i in range(5)}))]
    text = json.dumps([old_ask, *damaged, old_pick]).replace('"1e400"', "1e400")
    loaded = loading(text)
    check("damaged: v0.1.0 skips only the damaged entries, including overflowing numbers and non-list shapes",
          loaded == words([ask, pick]) and store.load(Node(text)).entries == [ask, pick], loaded)

    for label, bad in (
            ("a top list of the wrong length", dataclasses.replace(pick, rows=dataclasses.replace(pick.rows, top=[]))),
            ("a short wkey column", dataclasses.replace(pick, rows=dataclasses.replace(pick.rows, wkey=[""]))),
            ("words in a list", dataclasses.replace(pick, gn=list(pick.gn)))):
        node = Node()
        error = saving(node, NodeStore([ask, bad]))
        check(f"damaged: save refuses {label}, which load would skip, and writes nothing",
              error is not None and node.entries == "[]", error)


def phase_newer_schema():
    text = json.dumps({"schema": 3, "entries": [{"written by": "a newer OpenAct"}]})
    node = Node(text)
    loaded = store.load(node)
    check("newer schema: loads as an empty store", loaded == NodeStore() and loaded.schema == 3, loaded.schema)
    error = saving(node, loaded, frozenset())
    check("newer schema: save refuses to write over it", error is not None and node.entries == text, error)
    store.save(node, NodeStore())
    check("newer schema: a new store replaces it on purpose", node.entries == EMPTY, node.entries)
    broken = Node('{"schema": 2, "en')
    store.save(broken, store.load(broken))
    check("broken JSON: the next save replaces it, since nothing in it can be read", broken.entries == EMPTY,
          broken.entries)


def main():
    phase_conversion()
    phase_round_trip()
    phase_max_entries()
    phase_soft_cap()
    phase_both_caps()
    phase_precedence()
    phase_hard_cap()
    phase_damaged()
    phase_newer_schema()
    finish()


main()
