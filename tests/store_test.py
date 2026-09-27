"""The node store, openact/store.py: the schema-2 entries JSON, the v0.1.0 conversion and eviction; and the
entries the specs make from their plans (§5.4): the two key families and the validity rule.

No bpy and no model. tests/fixtures/jev_v010_entries.json holds the exact entries strings that the
v0.1.0 prototype's set_entries() wrote for Ask Yes/No, Ask Amount, Pick from text and Pick from a
collection, once with an empty Items text and once with Items text that the collection overrode.

Run:  python tests/store_test.py
      blender -b --factory-startup --python tests/store_test.py
"""

import contextlib
import dataclasses
import hashlib
import json
import math
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import ROOT, check, finish  # noqa: E402

# openact/__init__.py imports bpy, so the pure modules are loaded without running it.
_package = types.ModuleType("openact")
_package.__path__ = [os.path.join(ROOT, "openact")]
sys.modules["openact"] = _package
from openact import csvfmt, pipeline, store  # noqa: E402
from openact.specs import ask, base, pick  # noqa: E402

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


def parse_v0(text):
    """Items as an upgraded Pick parses its Items text while it keeps v0.1.0's rule (§12.3)."""
    return csvfmt.parse_inline(text, mode="v0")


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
            column = "amount" if mode == "AMOUNT" else "p"
            check(f"{case}: gn (Question, Context) becomes (Question, Context, '', '', '')",
                  all(e.gn == (o["gn"][0], o["gn"][1], "", "", "") for e, o in pairs), [e.gn for e in new])
            check(f"{case}: kind ask, answer type {mode}, one model row with the answer, as v1 stores it",
                  all((e.kind, e.answer_type, e.rows.name, e.rows.src, e.rows.ask, e.rows.v)
                      == ("ask", mode, [o["gn"][0].strip()], "m", [0],
                          {column: [o["a"]["value"]], "c": [o["a"]["confidence"]]}) for e, o in pairs),
                  [(e.rows.name, e.rows.v) for e in new])
            check(f"{case}: single holds Answer, Yes and Confidence",
                  all(e.single == {"Answer": o["a"]["value"], "Yes": o["a"]["value"] >= 0.5,
                                   "Confidence": o["a"]["confidence"]} for e, o in pairs), [e.single for e in new])
            plans = [[ASK.job(e.gn, mode, debias) for debias in (True, False)] for e in new]
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
                  all((e.rows.name, e.rows.v, e.rows.src, e.rows.ask) ==
                      (o["a"]["items"], {"prob": o["a"]["probabilities"]}, "m" * len(o["a"]["items"]),
                       [0] * len(o["a"]["items"])) for e, o in pairs),
                  [(e.rows.name, e.rows.src) for e in new])
            # the items the upgraded node plans: a collection's children, or its Items text parsed as v0.1.0 did
            items = [o["a"]["items"] if o["a"]["collection"] else [name for name, _ in parse_v0(o["gn"][1])]
                     for o in old]
            if case == "pick_text":
                check(f"{case}: legacy parsing of the Items text gives the stored items",
                      items == [o["a"]["items"] for o in old], items)
            plans = [[PICK.job(e.gn, names, debias) for debias in (True, False)] for e, names in zip(new, items)]
        check(f"{case}: the words key, wkeys and shape are the ones the spec plans for the same words, with either "
              "Balance Option Order (§14.4)",
              all((e.words_key, e.rows.wkey, e.shape) == (job.words_key, wkeys(job), job.shape.as_dict())
                  for e, jobs in zip(new, plans) for job in jobs), [e.words_key for e in new])
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

    # As set_entries() writes it: v0.1.0 split this Items text on commas only, and kept ": " in the names
    text, names = "Chair: comfy; Stool, Bench: long", ["Chair: comfy; Stool", "Bench: long"]
    entry = store.load(Node(json.dumps([{"key": "k", "gn": ["somewhere to rest", text], "t": 1.0, "a": {
        "index": 1, "item": names[1], "confidence": 0.5, "probabilities": [0.25, 0.75], "items": names,
        "collection": ""}}], ensure_ascii=False))).entries[0]
    v1 = [name for name, _when in csvfmt.parse_inline(text)]
    check("Items with ';' and ': ': legacy parsing keeps v0.1.0's names, and with them the words key; v1's parsing "
          "would not",
          [name for name, _ in parse_v0(text)] == names == entry.rows.name
          and entry.words_key == PICK.job(entry.gn, names, True).words_key
          and entry.words_key != PICK.job(entry.gn, v1, True).words_key, (entry.rows.name, v1))


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


ASK, PICK = ask.AskSpec(), pick.PickSpec()
ASK_WORDS = ("Is it night?", "A snowy mountain village, warm windows glowing.", "", "", "")  # gn: Key last
PICK_WORDS = ("something to sit on", "", "table, lamp, armchair", "", "")
ITEMS = ["table", "lamp", "armchair"]
MODEL = "semif-torch Qwen/Qwen3.5-4B@851bf6e8"
PLANS = {"Ask": lambda gn, debias: ASK.job(gn, 'YES_NO', debias),
         "Ask Amount": lambda gn, debias: ASK.job(gn, 'AMOUNT', debias),
         "Pick": lambda gn, debias: PICK.job(gn, ITEMS, debias)}


def wkeys(job):
    return [row.wkey for row in job.rows]


def cells(job):
    return [row.cell.key for row in job.rows]


@contextlib.contextmanager
def bumped():
    """PROMPT_VERSION one higher inside, as new templates make it."""
    base.Spec.PROMPT_VERSION += 1
    try:
        yield
    finally:
        base.Spec.PROMPT_VERSION -= 1


def answered(spec, job, answer):
    """The entry spec.resolve() makes once the job's one question has this parsed answer in MEMORY."""
    key = job.rows[0].cell.key
    pipeline.remember(key, dict(answer, model=MODEL))
    try:
        return spec.resolve(job, [], "")[0]
    finally:
        pipeline.forget([key])


def phase_keys():
    for name, plan in PLANS.items():
        gn = PICK_WORDS if name == "Pick" else ASK_WORDS
        on, off = plan(gn, True), plan(gn, False)
        check(f"keys, {name}: Balance Option Order changes the cell keys and the ask key, and no words key",
              (on.words_key, wkeys(on)) == (off.words_key, wkeys(off)) and cells(on) != cells(off)
              and ASK.ask_key(True, MODEL) != ASK.ask_key(False, MODEL))
        before = ASK.ask_key(True, MODEL)
        with bumped():
            newer, after = plan(gn, True), ASK.ask_key(True, MODEL)
        check(f"keys, {name}: a new PROMPT_VERSION changes the ask key, and no words key",
              (newer.words_key, wkeys(newer)) == (on.words_key, wkeys(on)) and after != before)
        keyed = plan(gn[:-1] + ("left",), True)
        check(f"keys, {name}: changing only Key leaves every wkey and cell key unchanged; the words key follows gn",
              wkeys(keyed) == wkeys(on) and cells(keyed) == cells(on) and keyed.words_key != on.words_key)
        edited = plan((gn[0] + " Really?",) + gn[1:], True)
        check(f"keys, {name}: other words change every wkey", not set(wkeys(edited)) & set(wkeys(on)))
    check("keys: the answer type is part of the words keys",
          PLANS["Ask"](ASK_WORDS, True).words_key != PLANS["Ask Amount"](ASK_WORDS, True).words_key)
    check("keys: the backend is part of the ask key", ASK.ask_key(True, MODEL) != ASK.ask_key(True, "fake fake"))
    check("keys: a question's cost in scored rows (§5.5)",
          (pipeline.cost(2, True), pipeline.cost(16, False), pipeline.cost(17, False), pipeline.cost(255, True))
          == (2, 1, 3, 34))


def phase_parse():
    score = pipeline.Cell("k", "s", {"type": "score", "instructions": "q", "criteria": list("abcde")}, 2)
    parsed = pipeline.parse(score, {"type": "score", "score": 1.55, "confidence": 0.1,
                                    "probabilities": {"4": 0.0, "3": 0.0, "2": 0.55, "1": 0.45, "0": 0.0}})
    check("parse: score gives the level probabilities in level order and amount = score / (L - 1)",
          parsed == {"probabilities": [0.0, 0.45, 0.55, 0.0, 0.0], "amount": 1.55 / 4}, parsed)
    choice = pipeline.Cell("k", "s", {"type": "choice", "instructions": "q", "criteria": {"1": "b", "0": "a"}}, 2)
    parsed = pipeline.parse(choice, {"type": "choice", "choice": "0", "probabilities": {"0": 0.75, "1": 0.25}})
    check("parse: choice gives the probabilities in the order the options were sent",
          parsed == {"probabilities": [0.25, 0.75]}, parsed)
    noul = pipeline.Cell("k", "s", {"type": "noul", "instructions": "q"}, 2)
    refused = []
    for answer in ({"type": "choice", "noul": 0.5}, {"type": "noul"}, {"type": "noul", "noul": "x"},
                   {"type": "noul", "noul": float("nan")}, [0.5], None):
        try:
            pipeline.parse(noul, answer)
        except (KeyError, TypeError, ValueError):
            refused.append(True)
    check("parse: noul gives p, and a malformed or non-finite answer raises",
          pipeline.parse(noul, {"type": "noul", "noul": 0.25}) == {"p": 0.25} and len(refused) == 6, len(refused))
    saved = pipeline.MEMORY_MAX
    pipeline.MEMORY_MAX = 2
    try:
        for key in "abc":
            pipeline.remember(key, {"p": 0.5})
            pipeline.cached("a")  # still in use
        kept = sorted(pipeline.MEMORY)
    finally:
        pipeline.MEMORY_MAX = saved
        pipeline.forget("abc")
    check("MEMORY: least recently used cells go first beyond the cap", kept == ["a", "c"], kept)


def phase_validity():
    job = ASK.job(ASK_WORDS, 'YES_NO', True)
    entry, missing, failed = ASK.resolve(job, [], "")
    check("validity: a new question is missing its one cell",
          entry is None and [cell.key for cell in missing] == cells(job) and not failed)
    entry = answered(ASK, job, {"p": 0.93})
    check("validity: its answer becomes an entry for the words, labelled with the ask key it was asked with",
          (entry.gn, entry.words_key, entry.asks, entry.pv, entry.model, entry.rows.src, entry.rows.wkey, entry.single)
          == (ASK_WORDS, job.words_key, [ASK.ask_key(True, MODEL)], 1, MODEL + " debias", "m", wkeys(job),
              {"Answer": 0.93, "Yes": True, "Confidence": 0.86}), entry)
    node = Node()
    store.save(node, NodeStore([entry]))
    stored = store.load(node).entries
    with bumped():
        again = [ASK.resolve(ASK.job(ASK_WORDS, 'YES_NO', debias), stored, "") for debias in (True, False)]
    check("validity: after a new PROMPT_VERSION, with either Balance Option Order, the stored entry holds: 0 requests",
          all(held is stored[0] and not missing for held, missing, _failed in again))
    keyed = ASK.job(ASK_WORDS[:-1] + ("left",), 'YES_NO', True)
    moved, missing, _failed = ASK.resolve(keyed, stored, stored[0].words_key)
    check("validity: a new Key carries the row over from the entry the context showed last, with its old ask key",
          moved is not None and not missing and (moved.gn, moved.words_key, moved.asks, moved.single)
          == (keyed.gn, keyed.words_key, stored[0].asks, stored[0].single), moved)
    unseen, missing, _failed = ASK.resolve(keyed, stored, "")
    check("validity: without an entry to carry from, the new words are asked", unseen is None and len(missing) == 1)
    amount = ASK.job(ASK_WORDS, 'AMOUNT', True)
    other, missing, _failed = ASK.resolve(amount, stored, stored[0].words_key)
    check("validity: the same words meaning something else (another answer type) are asked again",
          other is None and [cell.key for cell in missing] == cells(amount))
    pipeline.FAILED[cells(amount)[0]] = "prompt too long"
    try:
        _entry, missing, failed = ASK.resolve(amount, stored, "")
    finally:
        pipeline.FAILED.clear()
    check("validity: a failed cell is not asked again and says why", not missing and failed == ["prompt too long"])
    entry = answered(ASK, amount, {"probabilities": [0.0, 0.45, 0.55, 0.0, 0.0], "amount": 1.55 / 4})
    check("validity: Amount answers score / (L - 1), with the sigma confidence (4.e)",
          entry.single["Answer"] == 1.55 / 4 and round(entry.single["Confidence"], 2) == 0.65, entry.single)

    job = PICK.job(PICK_WORDS, ITEMS, True)
    entry = answered(PICK, job, {"probabilities": [0.2, 0.1, 0.7]})
    check("validity, Pick: a row per item with its probability, then the pick and top minus runner-up",
          (entry.rows.name, entry.rows.v, entry.single, entry.shape)
          == (ITEMS, {"prob": [0.2, 0.1, 0.7]}, {"Index": 2, "Item": "armchair", "Confidence": 0.5},
              {"rows": 3, "columns": {"prob": "FLOAT"}}), entry)
    grown = PICK.job(PICK_WORDS[:2] + ("table, lamp, armchair, rug",) + PICK_WORDS[3:], ITEMS + ["rug"], True)
    other, missing, _failed = PICK.resolve(grown, [entry], entry.words_key)
    check("validity, Pick: a new item changes what every row was offered, so the question is asked again",
          other is None and len(missing) == 1)


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
    phase_keys()
    phase_parse()
    phase_validity()
    finish()


main()
