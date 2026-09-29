"""The node store, openact/store.py: the schema-2 entries JSON and eviction; and the entries the specs make from
their plans (§5.4): the two key families and the validity rule.

No bpy and no model: the sample entries are the ones the specs make from a parsed answer.

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
from openact import pipeline, store  # noqa: E402
from openact.specs import ask, base, pick  # noqa: E402

NodeStore, MB, SOFT_CAP = store.NodeStore, store.MB, store.SOFT_CAP
EMPTY = '{"schema":2,"entries":[]}'
KEPT = "live contexts are kept"


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
    """A Rules One-of entry with `rows` rows, last used at t; pinned: with the pin and frames OpenAct 1.0 saved."""
    names = [f"level {r // 64}|level {r % 64}" for r in range(rows)]
    return store.Entry(
        words_key=hashlib.sha1(first.encode()).hexdigest(), asks=["ask"], pv=1, t=t, pinned=pinned,
        frames=[[1, 250]] if pinned else [], gn=gn(first), model="fake", kind="rules", answer_type="ONE_OF",
        shape={"rows": rows, "columns": {"outcome": "INT", "c": "FLOAT"}},
        rows=store.Rows(name=names, wkey=[hashlib.sha1(name.encode()).hexdigest() for name in names],
                        ask=[0] * rows, src="m" * rows,
                        v={"outcome": [r % 7 for r in range(rows)], "c": [1 - r / (2 * rows) for r in range(rows)]}))


def life():
    """Every schema-2 field set: a Game of Life entry with payload and top answers, pinned with its frames as OpenAct
    1.0 saved it."""
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


SAMPLE_ITEMS = ["table", "lamp", "armchair", "bookshelf", "rug"]


def sample_ask(mode='YES_NO'):
    """An Ask entry as the spec makes it from a parsed answer, with a non-ASCII word in its Context."""
    job = ASK.job(("How cold is it?", "A sunny tropical beach at noon \u2014 people swimming.", "", "", ""), mode, True)
    if mode == 'AMOUNT':
        return answered(ASK, job, {"probabilities": [0.6, 0.3, 0.1, 0.0, 0.0], "amount": 0.5 / 4})
    return answered(ASK, job, {"p": 0.08})


def sample_pick():
    """A Pick entry over five items, as the spec makes it from a parsed answer."""
    job = PICK.job(("something to sit on", "", ", ".join(SAMPLE_ITEMS), "", ""), SAMPLE_ITEMS, True)
    return answered(PICK, job, {"probabilities": [0.05, 0.02, 0.85, 0.05, 0.03]})


def phase_round_trip():
    ask = sample_ask('AMOUNT')  # entry-level values and a non-ASCII word
    full = NodeStore([life(), ask])
    node = Node()
    store.save(node, full)
    check("round trip: load returns what save wrote", store.load(node) == full)
    data = json.loads(node.entries)
    whole, plain = data["entries"]
    check("round trip: the documented schema-2 fields, in order, a 1.0 pin's pinned and frames kept",
          data["schema"] == 2 and list(whole) == ["words_key", "asks", "pv", "t", "pinned", "frames", "gn",
                                                  "model", "kind", "answer_type", "shape", "rows", "single", "csv"]
          and (whole["pinned"], whole["frames"]) == (True, [[1, 48], [60, 90]])
          and list(whole["rows"]) == ["name", "wkey", "ask", "src", "v", "payload", "top"], list(whole))
    check("round trip: frames only when an entry has some, top only when given",
          "frames" not in plain and "top" not in plain["rows"], list(plain))
    check("round trip: non-ASCII words are stored as they are", "\u2014" in node.entries)


def phase_max_entries():
    live = [rules(f"live {i}", 100 + i, 64) for i in range(18)]  # more live entries than MAX_ENTRIES
    pinned = rules("pinned", 0, 64, pinned=True)  # the least recently used of all
    old = [rules(f"old {i}", 10 + (7 * i) % 20, 64) for i in range(20)]
    entries = live[:9] + old[:10] + [pinned] + live[9:] + old[10:]
    live_gns = frozenset(gn(f"live {i}") for i in range(18))
    recent = set(words(sorted(old, key=lambda e: e.t)[-store.MAX_ENTRIES:]))
    expected = [first for first in words(entries) if first.startswith("live") or first in recent]
    kept, node = NodeStore(list(entries)), Node()
    store.save(node, kept, live_gns)
    check("16 entries: keeps every live entry and the 16 most recent others, in order; an entry OpenAct 1.0 pinned "
          "is evicted like the others", words(kept.entries) == expected and "pinned" not in words(kept.entries),
          words(kept.entries))
    check("16 entries: the node holds exactly what save() kept", store.load(node).entries == kept.entries)
    check("16 entries: no note", kept.note == "", kept.note)
    kept = NodeStore(list(entries))
    store.save(node, kept)
    check("16 entries with live_gns=None: nothing is evicted",
          kept.entries == entries and store.load(node).entries == entries, len(kept.entries))


def phase_soft_cap():
    live = rules("live", 1, 3000)
    pinned = rules("pinned", 2, 3000, pinned=True)  # older than every other warm or cold entry
    old = [pinned, *(rules(f"old {i}", 10 + (5 * i) % 8, 3000) for i in range(8))]
    entries = [old[1], live, *old[2:6], pinned, *old[6:]]
    whole, alone = Node(), Node()
    store.save(whole, NodeStore(list(entries)))
    store.save(alone, NodeStore([live]))
    check("soft cap: the store exceeds 2 MB, its live entry does not",
          size_of(whole) > SOFT_CAP > size_of(alone), (size_of(whole), size_of(alone)))
    kept, node = NodeStore(list(entries)), Node()
    store.save(node, kept, frozenset({gn("live")}))
    by_age = words(sorted(old, key=lambda e: e.t))
    dropped = len(old) + 1 - len(kept.entries)
    check("soft cap: keeps live, evicts the least recently used others first, an entry OpenAct 1.0 pinned first",
          0 < dropped < len(old) and set(words(kept.entries)) == {"live", *by_age[dropped:]} and by_age[0] == "pinned",
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
    live = [rules(f"live {i}", 10 + i, 4096) for i in range(8)]
    others = [rules(f"old {i}", t, rows) for i, (t, rows) in enumerate(
        ((50, 64), (60, 4096), (5, 64), (70, 64), (40, 4096), (55, 64)))]  # most are more recent than live
    others += [rules(f"pinned {i}", 80 + i, 4096, pinned=True) for i in range(2)]  # 1.0 pins, the most recent of all
    entries = [others[0], live[0], others[6], others[1], live[1], live[2], live[3], others[2], live[4], others[7],
               others[3], live[5], live[6], others[4], live[7], others[5]]
    live_gns = frozenset(gn(f"live {i}") for i in range(8))  # built from the words, as the pass will
    alone, both = Node(), NodeStore(list(live))
    store.save(alone, both, live_gns)
    check("precedence: the live entries alone exceed 2 MB", size_of(alone) > SOFT_CAP, size_of(alone))
    check("precedence: with nothing to drop, the note gives the size",
          both.note == f"entries use {mb(size_of(alone))} MB: {KEPT}", both.note)

    kept, node = NodeStore(list(entries)), Node()
    store.save(node, kept, live_gns)
    check("precedence: every live entry is kept, and every warm and cold one evicted, the ones OpenAct 1.0 pinned too",
          words(kept.entries) == [first for first in words(entries) if first.startswith("live")],
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
                        ("a store without a schema", '{"entries": []}'), ("a JSON string", '"entries"'),
                        ("a bare list of entries", '[{"gn": ["Is it night?", ""], "t": 1.0}]')):
        check(f"damaged: {label} loads as an empty store", store.load(Node(text)) == NodeStore())

    ask, pick = sample_ask(), sample_pick()
    written = Node()
    store.save(written, NodeStore([ask, pick]))
    a, p = json.loads(written.entries)["entries"]
    damaged = [
        42, {"words_key": "no gn"}, dict(a, gn="Is it night?"), dict(a, gn=["Is it night?", 7]), dict(a, rows=[]),
        dict(a, pv="1e400"), dict(a, pv=float("inf")), dict(a, t=10 ** 400), dict(a, t=float("nan")),
        dict(a, t=float("inf")), dict(a, pinned="false"), dict(a, words_key=None),
        dict(a, kind=["ask"]), dict(a, single=[]),
        *(dict(p, rows=dict(p["rows"], **change)) for change in (
            {"name": ["table", 7, "armchair", "bookshelf", "rug"]}, {"wkey": [None] * 5}, {"wkey": [""]},
            {"ask": [0, 0, 1, 0, 0]}, {"ask": [0, -1, 0, 0, 0]}, {"v": {"prob": [0.1, "x", 0.2, 0.3, 0.4]}},
            {"payload": {"alive": [0, 1, None, 0, 1]}}, {"top": []}))]
    text = json.dumps({"schema": 2, "entries": [a, *damaged, p]}).replace('"1e400"', "1e400")
    loaded = loading(text)
    check("damaged: schema 2 skips only the damaged entries, overflowing numbers included",
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
