"""Node status, the sidebar's data and Show Question (§17.1-§17.3), in a headless Blender. The add-on starts the fake
model server itself (OPENACT_BACKEND=fake, §18.1) on a free port, with --record, so every request it sends is logged:
- a fill: waiting -> filling -> complete, with n/N and an ETA from pipeline.progress();
- the warning and error lines: a CSV warning, a missing CSV, Rules cells no outcome applies to (its other cells say
  "Asking..." while they are asked) and an outcome never offered, more than 16 live word sets;
- table_page(): rows, icons and paging; table_rows() of answered and pending tables, of an entry whose rows are in
  another order (by wkey, Show Question's ask key too) and of an empty material slot;
- while a Classify context's CSV has errors or the contexts disagree, a context whose words didn't change shows its
  stored answer and one without says it is not asked, as does one whose job waits while another context waits for
  another node, where a context whose cells are asked with no answer yet says "Asking...", also while another
  context's CSV is missing, and one with a table so far sums it up, its pending rows with no answer, also while the
  contexts disagree; a Pick of material slots before its first answer says "Asking...", with no top items; a failed
  cell is in error in the table, the counts and Show Question, and not asked again;
- the Model box's queue: the chunks waiting in each class and the cells in flight;
- the banner conditions, and live: reads blocked in an untrusted file (an Ask's and a Rules node's), which either
  trust button lets every node make and a deleted node stops counting in, a blocked read counted once when another
  node reads the blocked node's answer, a sidebar draw that resolves no path while nothing is blocked, an Import CSV
  node on a file a node reads, and over-cap;
- show_question returns exactly the state, instructions and options that the --record log shows for that row, and
  asks nothing; a row answered with older settings says so;
- the pass drops the status records of nodes that are gone;
- drawing the node body, the sidebar panels, the popup and the preferences into a stand-in layout: valid icons,
  operators and properties, paging, the install's VRAM warning, and nothing planned or written while drawing;
- Open File opens CSV files only.

Run:  blender -b --factory-startup --python tests/status_blender.py
"""

import dataclasses
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import check, finish  # noqa: E402

os.environ["OPENACT_BACKEND"] = "fake"  # openact_server.py --backend fake: no torch, GPU or runtime
os.environ["OPENACT_SERVER_PYTHON"] = sys.executable  # Blender's own Python runs it
TMP = tempfile.mkdtemp(prefix="openact_status_")
OUTSIDE = tempfile.mkdtemp(prefix="openact_status_outside_")  # outside the folder the .blend is saved in
os.environ.setdefault("OPENACT_STATE_DIR", os.path.join(TMP, "state"))  # where the server writes its log

from openact import csvfmt, engine, lists, nodes, ops, pipeline, server, trust, ui, watch  # noqa: E402
from openact import prefs as P  # noqa: E402
from openact import store as S  # noqa: E402
from openact.specs.base import number_text  # noqa: E402
from openact.specs.classify import ClassifySpec  # noqa: E402
from openact.specs.match import MatchSpec  # noqa: E402
from openact.specs.rules import RulesSpec  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
RECORD = os.path.join(TMP, "record.jsonl")
ICONS = set(bpy.types.UILayout.bl_rna.functions["label"].parameters["icon"].enum_items.keys())
_stamp = [1_700_000_000 * 10**9]


def free_port():
    """A free local port, never 18765 or 18769: the user's own OpenAct session and another job use those."""
    while True:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        if port not in (18765, 18769):
            return port


PORT = free_port()
P._Defaults.port, P._Defaults.runtime_dir, P._Defaults.auto_start = PORT, os.path.join(TMP, "runtime"), True
server.STATE.port = PORT  # before anything talks to a server
for module in (ops, nodes, ui, trust, watch):
    module.register()
engine.register()  # the server status listener; timers don't run under --python, so the test ticks
_popen = subprocess.Popen


def recording(args, **kwargs):
    """subprocess.Popen, with --record added to the model server's command line."""
    if isinstance(args, list) and server.SERVER_SCRIPT in args:
        args = [*args, "--record", RECORD]
    return _popen(args, **kwargs)


subprocess.Popen = recording


# ------------------------------------------------------------------------------------------ helpers

def write(folder, name, text):
    """A file with a new mtime every time: an edit within one clock tick still changes its stamp."""
    path = os.path.join(folder, name)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    _stamp[0] += 10**9
    os.utime(path, ns=(_stamp[0], _stamp[0]))
    return path


def new_tree(name):
    tree = bpy.data.node_groups.new(name, 'GeometryNodeTree')
    tree.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    tree.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    tree.nodes.new("NodeGroupOutput")  # else Blender warns about the modifiers using the tree
    return tree


def user(name, tree, **words):
    """An object whose modifier uses the tree, with modifier inputs by name."""
    obj = bpy.data.objects.new(name, bpy.data.meshes.new(name))
    bpy.context.scene.collection.objects.link(obj)
    mod = obj.modifiers.new("OpenAct", 'NODES')
    mod.node_group = tree
    idents = {i.name: i.identifier for i in tree.interface.items_tree if getattr(i, "in_out", "") == 'INPUT'}
    for word, value in words.items():
        mod[idents[word]] = value
    return obj


def status(node):
    return engine.STATUS.get(node.uid)


def state(node):
    return getattr(status(node), "state", "")


def settle(done, timeout=60, watched=()):
    """Tick the add-on until done() holds and no answer is in flight. Returns (done, {node name: [(state, text)]}),
    each watched node's status lines in the order they came."""
    seen = {node.name: [] for node in watched}
    started = time.time()
    while time.time() - started < timeout:
        engine.tick()
        for node in watched:
            found = status(node)
            if found is not None and (not seen[node.name] or seen[node.name][-1] != (found.state, found.text)):
                seen[node.name].append((found.state, found.text))
        if done() and not pipeline.PENDING:
            return True, seen
        time.sleep(0.02)
    return False, seen


def in_order(states, wanted):
    """True when the wanted states come in this order (others may come between)."""
    rest = iter(states)
    return all(any(found == one for found in rest) for one in wanted)


def recorded():
    """The fake server's --record log: every /v1/systemone request, as it arrived, with its status and answers."""
    try:
        with open(RECORD, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    except OSError:
        return []


def banner_kinds():
    return [banner[0] for banner in ui.live_banners(bpy.context)]


# ------------------------------------------------------------------------ waiting -> filling -> complete

def phase_fill():
    """Pick, then Ask reading Pick's Item as its Context: the fake server the add-on starts answers both."""
    tree = new_tree("Fill")
    pick = tree.nodes.new(engine.PICK_ID)
    pick.inputs["Prompt"].default_value = "something to sit on"
    pick.inputs["Items"].default_value = "table, lamp, armchair"
    ask = tree.nodes.new(engine.ASK_ID)
    ask.inputs["Question"].default_value = "Is it soft?"
    tree.links.new(pick.outputs["Item"], ask.inputs["Context"])
    user("Filler", tree)
    engine.schedule(immediate=True)
    done, seen = settle(lambda: state(pick) == state(ask) == "complete", 90, (pick, ask))
    check("fill: the add-on starts the fake server on a free port, and both nodes complete",
          done and server.STATE.port == PORT and PORT not in (18765, 18769), (server.status(), seen))
    states = {name: [found for found, _text in lines] for name, lines in seen.items()}
    check("fill: Pick goes starting -> filling -> complete",
          in_order(states[pick.name], ["starting", "filling", "complete"]), seen[pick.name])
    check("fill: Ask, reading Pick's answer, goes waiting -> filling -> complete, and waits with the waiting line",
          in_order(states[ask.name], ["waiting", "filling", "complete"])
          and {text for found, text in seen[ask.name] if found == "waiting"} == {"Waiting for another OpenAct node..."},
          seen[ask.name])
    filling = [text for lines in seen.values() for found, text in lines if found == "filling"]
    check("fill: with pipeline.progress(), the filling line shows n/N and an ETA",
          any(re.fullmatch(r"Filling \d+/\d+ · ~\d+ (s|min)", text) for text in filling), filling)
    picked, asked = status(pick), status(ask)
    check("complete: Pick's record: its summary as the line, 3 rows answered, none pending, no ETA, one context",
          picked.icon == 'CHECKMARK' and picked.text == engine.SPECS[pick.bl_idname].summary(picked.uses[0][2])
          and (picked.rows, picked.answered, picked.pending, picked.eta_s, picked.contexts) == (3, 3, 0, None, 1)
          and [label for label, _job, _entry in picked.uses] == ["Filler > OpenAct"] and not picked.problems, picked)
    check("complete: Ask's record: 'Yes/No  (N% yes)', 1 row answered",
          asked.icon == 'CHECKMARK' and re.fullmatch(r"(Yes|No)  \(\d+% yes\)", asked.text) is not None
          and (asked.rows, asked.answered, asked.pending) == (1, 1, 0), asked)
    return pick, ask


def phase_eta():
    saved = pipeline.progress
    pipeline.progress = lambda uid: {"a": (12, 50, 40.0), "b": (0, 3000, 40.0), "c": (0, 0, None)}.get(uid)
    try:
        got = [engine._filling(uid) for uid in "abcd"]
        del pipeline.progress
        try:
            gone = engine._filling("a")
        except AttributeError as error:
            gone = type(error).__name__
    finally:
        pipeline.progress = saved
    check("ETA: n/N and the remaining rows at ms_per_row (38 x 40 ms = ~2 s, 3000 x 40 ms = ~2 min); no progress: "
          "'Asking...'", got == [("Filling 12/50 · ~2 s", 1.52), ("Filling 0/3000 · ~2 min", 120.0),
                                ("Asking...", None), ("Asking...", None)], got)
    check("ETA: the filling line reads pipeline.progress() itself, with no fallback that would draw 'Asking...' "
          "without it", gone == "AttributeError", gone)
    rows = ui.status_rows(engine.NodeStatus("filling", "Filling 12/50 · ~2 s", rows=50, answered=12, pending=38,
                                            eta_s=1.52))
    check("status_rows: the filling line with its ETA, then the row counts",
          rows == [('SORTTIME', "Filling 12/50 · ~2 s"), ('BLANK1', "50 rows: 12 answered, 38 pending")], rows)


# --------------------------------------------------------------------------------------- Show Question

def phase_question(pick, ask):
    lines = [line for line in recorded() if line["status"] == 200]
    compared, wrong = 0, []
    for node in (pick, ask):
        job = status(node).uses[0][1]
        for row in range(len(job.rows)):
            found = ops.row_question(node, 0, row)
            sent = next(((line["request"], question) for line in lines
                         for qid, question in line["request"]["questions"].items()
                         if qid in (found["key"][:16], found["key"])), None)
            if sent is None:
                wrong.append((node.name, row, "not in the --record log", found))
                continue
            request, question = sent
            compared += 1
            same = (request["state"] == found["state"] and question["instructions"] == found["instructions"]
                    and question.get("criteria") == found["options"] and question["type"] == found["type"]
                    and list(question.get("criteria") or ()) == list(found["options"] or ())  # the order sent
                    and request["options"]["debias"] == found["debias"])
            if not same or found["source"] != "model" or not found["ask"]:
                wrong.append((node.name, row, request["state"], question, found))
    check("show_question: every row's state, instructions and options, in their order, are exactly those of its "
          "request in the fake server's --record log", compared == 4 and not wrong, wrong)

    found = ops.row_question(ask, 0, 0)
    text = ops.question_text(found)
    before = len(recorded())
    result = bpy.ops.openact.show_question('EXEC_DEFAULT', uid=ask.uid, context_index=0, row=0)
    settle(lambda: False, 0.5)
    check("show_question: the operator shows the question as sent and asks nothing",
          result == {'FINISHED'} and len(recorded()) == before and not pipeline.PENDING
          and f"State: {found['state']}" in text and "Question (noul): Is it soft?" in text
          and "Answered with ask key" in text, (result, len(recorded()) - before, text))

    ask.balance_order = False  # words keys hold no Balance Option Order: the answer asked with it on is kept
    settle(lambda: state(ask) == "complete" and not status(ask).uses[0][1].debias, 30)
    found = ops.row_question(ask, 0, 0)
    text = ops.question_text(found)
    check("show_question: a row answered with Balance Option Order on, shown once it is off, says it was answered "
          "with older settings and that this is what would be sent now; nothing is asked",
          len(recorded()) == before and found["older"] and not found["debias"] and "Balance Option Order: off" in text
          and "Answered with older settings" in text and "Answered with ask key" not in text, text)
    ask.balance_order = True
    settle(lambda: state(ask) == "complete" and status(ask).uses[0][1].debias, 30)

    solo = pick.id_data.nodes.new(engine.PICK_ID)
    solo.inputs["Prompt"].default_value = "something to hold"
    solo.inputs["Items"].default_value = "a cup"
    engine.schedule(immediate=True)
    settle(lambda: state(solo) == "complete", 30)
    found = ops.row_question(solo, 0, 0)
    check("show_question: a row decided by code says it is never sent",
          found["source"] == "code" and found["state"] is None
          and ops.question_text(found).endswith(ops.NEVER_SENT["code"]), found)
    try:
        ops.row_question(solo, 0, 5)
        refused = ""
    except ValueError as error:
        refused = str(error)
    check("show_question: a row the table doesn't have is refused", refused == "The table has no row 6", refused)


# ----------------------------------------------------------------------------------------- table view

def phase_table(pick, ask):
    rows = [(f"r{i:03}", "yes", 0.9, "model") for i in range(120)]
    rows[1:7] = [("unsure", "no", 0.1, "model"), ("locked", "yes", 1.0, "lock"), ("code", "no", 1.0, "code"),
                 ("pending", "", None, "pending"), ("stale", "yes", 0.8, "stale"), ("error", "", 0.0, "error")]
    lines, page, pages = ui.table_page(rows, 0)
    check("table_page: 50 rows a page, so 120 rows make 3 pages", len(lines) == 50 and (page, pages) == (0, 3),
          (len(lines), page, pages))
    check("table_page: the rows as given: name, answer, confidence",
          [line[:3] for line in lines[:3]] == [("r000", "yes", 0.9), ("unsure", "no", 0.1), ("locked", "yes", 1.0)],
          lines[:3])
    check("table_page: icons: QUESTION unsure, LOCKED lock, CHECKBOX_HLT code, CANCEL error, none for a sure answer",
          [line[3] for line in lines[:7]] == ['BLANK1', 'QUESTION', 'LOCKED', 'CHECKBOX_HLT', 'BLANK1', 'BLANK1',
                                              'CANCEL'], [line[3] for line in lines[:7]])
    check("table_page: stale and pending rows are greyed out", [line[4] for line in lines[:7]] == [
        True, True, True, True, False, False, True], [line[4] for line in lines[:7]])
    last, beyond, below = ui.table_page(rows, 2), ui.table_page(rows, 9), ui.table_page(rows, -3)
    check("table_page: the last page has the rest (20 rows); a page beyond clamps to the last, below it to the first",
          len(last[0]) == 20 and last[0][0][0] == "r100" and beyond[1:] == (2, 3) and below[1:] == (0, 3)
          and below[0] == lines, (len(last[0]), beyond[1:], below[1:]))
    check("table_page: an empty table is one empty page", ui.table_page([], 4) == ([], 0, 1), ui.table_page([], 4))

    label, job, entry = status(pick).uses[0]
    got = ui.table_rows(engine.SPECS[pick.bl_idname], job, entry)
    check("table_rows: an answered Pick lists each item with its probability, from the stored entry",
          got == [(name, f"{p * 100:.0f}%", None, "model") for name, p in
                  zip(["table", "lamp", "armchair"], entry.rows.v["prob"])], got)
    pending = ui.table_rows(engine.SPECS[pick.bl_idname], job)
    check("table_rows: before its answer, a table lists its plan's rows, pending, with no answer, greyed out",
          [row[1:] for row in pending] == [("", None, "pending")] * 3
          and [line[4] for line in ui.table_page(pending)[0]] == [False] * 3, pending)
    order, record = [2, 0, 1], status(pick)  # the entry of other words, whose rows are in another order
    moved = dataclasses.replace(entry, words_key="other", asks=["ask0", "ask1", "ask2"], rows=dataclasses.replace(
        entry.rows, name=[entry.rows.name[i] for i in order], wkey=[entry.rows.wkey[i] for i in order], ask=[0, 1, 2],
        src="".join(entry.rows.src[i] for i in order), v={c: [v[i] for i in order] for c, v in entry.rows.v.items()}))
    engine.STATUS[pick.uid] = dataclasses.replace(record, uses=[(label, job, moved)])
    try:
        got = ui.table_rows(engine.SPECS[pick.bl_idname], job, moved), [ops.row_question(pick, 0, row)["ask"]
                                                                        for row in range(3)]
    finally:
        engine.STATUS[pick.uid] = record
    check("table_rows and Show Question: an entry whose words key isn't the plan's gives each row the answer and the "
          "ask key of its row with the same wkey, as resolve() carries rows, whatever their order",
          got == ([(name, f"{p * 100:.0f}%", None, "model") for name, p in zip(["table", "lamp", "armchair"],
                                                                               entry.rows.v["prob"])],
                  ["ask1", "ask2", "ask0"]), got)
    _label, job, entry = status(ask).uses[0]
    got = ui.table_rows(engine.SPECS[ask.bl_idname], job, entry)
    check("table_rows: an answered Ask lists its question, its answer as the Overrides CSV writes it (the probability "
          "of yes), and its confidence",
          got == [("Is it soft?", number_text(entry.rows.v["p"][0], job.options), entry.rows.v["c"][0], "model")], got)
    spec = engine.SPECS[engine.PICK_ID]
    slots = spec.job(("Something shiny", ""), [lists.Item("Steel"), lists.Item("", offered=False),
                                                lists.Item("Gold")], True)
    got, text = ui.table_rows(spec, slots)[1], ops.question_text({"name": "", "source": "code", "state": None})
    check("table: an empty material slot is listed as one, with no answer or confidence; Show Question says it is "
          "never offered", got == (ui.EMPTY_SLOT, "", None, "code")
          and text == f"Row: {ui.EMPTY_SLOT}\n{ops.NEVER_OFFERED}", (got, text))
    kept = [each.label for each in engine.SPECS.values() if hasattr(each, "preview")]
    check("table: it replaced the Match and Classify previews, so no spec keeps a preview()", not kept, kept)


# ------------------------------------------------------------------------- held, in conflict, failed

def phase_held():
    """A Classify node used by two objects, each with its own Labels CSV (§6.0). While one CSV has errors, or their
    payload columns disagree, nothing is asked: a context whose words didn't change shows its stored answer, and one
    without it says it is not asked, where the sidebar said "Asking...". So does a context whose job waits while
    another waits for another node's answer; "Asking..." is for a context whose cells are asked with no row known yet,
    also while another context's CSV is missing, and a context with a table so far sums it up (§5.4)."""
    folder = os.path.join(TMP, "held")
    os.makedirs(folder)
    body = "answer,when,roughness\nmetal,steel or iron,{}\nwood,planks or bark,0.6\n"
    tree = new_tree("Held")
    tree.interface.new_socket("Labels", in_out='INPUT', socket_type='NodeSocketString')
    classify = tree.nodes.new(ClassifySpec.idname)
    classify.inputs["Items"].default_value = "pan, spoon"
    tree.links.new(tree.nodes.new("NodeGroupInput").outputs["Labels"], classify.inputs["Labels"])
    user("CA", tree, Labels=write(folder, "ca.csv", body.format(0.3)))
    user("CB", tree, Labels=write(folder, "cb.csv", body.format(0.4)))
    engine.schedule(immediate=True)
    done, _seen = settle(lambda: state(classify) == "complete", 60)
    summary = engine.SPECS[classify.bl_idname].summary(status(classify).uses[1][2])

    def after(text, until):
        """Write CA's Labels CSV and run the passes: (the status, each context's label and whether it shows its stored
        entry of its plan's words, the requests sent, what the Active Node panel draws)."""
        before = len(recorded())
        write(folder, "ca.csv", text)
        engine.schedule(immediate=True)
        settle(lambda: until(status(classify)), 30)
        for _ in range(3):
            engine.run_pass()
        got = status(classify)
        return (got, [(label, entry is not None and entry.words_key == job.words_key) for label, job, entry in got.uses],
                len(recorded()) - before, sidebar(classify))

    got = after("answer,Answer\nx,y\n", lambda found: found.state == "error")
    check("held: while CA's Labels CSV has errors, nothing is asked, and CB, whose words didn't change, shows its "
          "stored answer, not 'Asking...'",
          done and got[0].text == "ca.csv: Two columns are named 'Answer'" and got[1] == [("CB > OpenAct", True)]
          and not got[2] and summary in got[3] and "Asking..." not in got[3], got)
    got = after("answer,when,hardness\nmetal,steel or iron,9\nwood,planks or bark,3\n",
                lambda found: found.text.startswith("Classify: "))
    check("conflict: while the payload columns disagree, nothing is asked; CB still shows its stored answer, and CA, "
          "whose words changed, says it is not asked instead of 'Asking...'",
          got[0].text.startswith("Classify: 'CA > OpenAct' and 'CB > OpenAct' use Labels CSVs with different payload")
          and got[1] == [("CA > OpenAct", False), ("CB > OpenAct", True)] and not got[2] and summary in got[3]
          and "Not asked: see the status above" in got[3] and "Asking..." not in got[3], got)
    os.remove(os.path.join(folder, "ca.csv"))
    classify.inputs["Items"].default_value = "kettle, whisk"
    engine.run_pass()  # CB's new items fit the outputs, so they are asked; no tick, so no answer arrives meanwhile
    got, log = status(classify), sidebar(classify)
    check("asking: while CA's Labels CSV is missing, CB's new items are asked, and CB, with no row known yet, says "
          "'Asking...' although the status is CA's error",
          got.state == "error" and got.text.startswith("Can't find ") and got.text.endswith("ca.csv")
          and got.pending == 2 and [(label, entry) for label, _job, entry in got.uses] == [("CB > OpenAct", None)]
          and log[log.index("CB > OpenAct") + 1] == "Asking...", (got, log))
    classify.inputs["Items"].default_value = "pan, spoon, kettle"
    engine.run_pass()  # pan's and spoon's answers carry over, so CB shows its table so far (§5.4)
    got, log = status(classify), sidebar(classify)
    _label, job, entry = got.uses[0]
    line = log[log.index("CB > OpenAct") + 1]
    check("table so far: with pan and spoon carried and kettle asked, CB's line sums up its table so far, with the "
          "record's unsure count; its kettle row is pending with no answer, and the counts say 2 answered, 1 pending",
          entry is not None and entry.rows.src == "mmp" and (got.answered, got.pending) == (2, 1)
          and line == engine.SPECS[classify.bl_idname].summary(entry) and line.endswith(f" {got.unsure} unsure")
          and ui.table_rows(engine.SPECS[classify.bl_idname], job, entry)[2] == ("kettle", "", None, "pending"),
          (got, log))
    write(folder, "ca.csv", "answer,when,hardness\nmetal,steel or iron,9\nwood,planks or bark,3\n")
    engine.run_pass()  # the contexts disagree, so nothing is answered; kettle's answer hasn't arrived either
    got, log = status(classify), sidebar(classify)
    job, entry = next((job, entry) for label, job, entry in got.uses if label == "CB > OpenAct")
    lines = [log[i + 1] for i, found in enumerate(log) if found == ("openact.table_view", "page", 0)]
    check("conflict, table so far: while the contexts disagree, CB shows its stored table so far, whose words key is "
          "its plan's (§17.2): its line sums it up, and its table view keeps pan's and spoon's answers, kettle pending",
          got.text.startswith("Classify: 'CA > OpenAct' and 'CB > OpenAct'") and entry is not None
          and entry.rows.src == "mmp"
          and lines == ["Not asked: see the status above", engine.SPECS[classify.bl_idname].summary(entry)]
          and [row[1] != "" for row in ui.table_rows(engine.SPECS[classify.bl_idname], job, entry)]
          == [True, True, False], (got, log))
    bpy.data.node_groups.remove(tree)
    engine.run_pass()

    # a job also waits while another context waits for another node's answer: then nothing asks for it either
    tree = new_tree("Waits")
    for name in ("Labels", "Where"):
        tree.interface.new_socket(name, in_out='INPUT', socket_type='NodeSocketString')
    group_in = tree.nodes.new("NodeGroupInput")
    pick = tree.nodes.new(engine.PICK_ID)  # its answer is the Classify's Context
    pick.inputs["Items"].default_value = "a kitchen, a forge"
    tree.links.new(group_in.outputs["Where"], pick.inputs["Prompt"])
    classify = tree.nodes.new(ClassifySpec.idname)
    classify.inputs["Items"].default_value = "pan, spoon"
    tree.links.new(group_in.outputs["Labels"], classify.inputs["Labels"])
    tree.links.new(pick.outputs["Item"], classify.inputs["Context"])
    wa = user("WA", tree, Labels=write(folder, "wa.csv", body.format(0.3)), Where="Where is it made?")
    user("WB", tree, Labels=write(folder, "wb.csv", body.format(0.4)), Where="Where is it made?")
    engine.schedule(immediate=True)
    done, _seen = settle(lambda: state(classify) == "complete", 60)

    def drawn_lines():
        """The status, and the line under each context's button in the Active Node panel."""
        log = sidebar(classify)
        return status(classify), [log[i + 1] for i, entry in enumerate(log)
                                  if entry == ("openact.table_view", "page", 0)]

    classify.inputs["Items"].default_value = "fork, knife"
    engine.run_pass()  # asks both new cells for both contexts; no tick, so no answer arrives meanwhile
    asking = drawn_lines()
    settle(lambda: state(classify) == "complete", 30)
    wa.modifiers["OpenAct"][next(i.identifier for i in tree.interface.items_tree if i.name == "Where")] = "Where to?"
    write(folder, "wb.csv", "answer,when,hardness\nmetal,steel or iron,9\nwood,planks or bark,3\n")
    engine.run_pass()  # WA's Pick is asked again, so WA's Context waits, and WB's new payload columns with it
    got, log = status(classify), sidebar(classify)
    check("asking: while both contexts' cells are asked and no row is known yet, the sidebar says 'Asking...' under "
          "each", done and asking[0].state == "filling" and asking[1] == ["Asking...", "Asking..."], asking)
    check("waits: while one context waits for another node's answer, the other's new payload columns wait too: the "
          "status is the wait, and the sidebar says that context is not asked instead of 'Asking...'",
          (got.state, got.text) == ("waiting", "Waiting for another OpenAct node...")
          and [(label, entry) for label, _job, entry in got.uses] == [("WB > OpenAct", None)]
          and log[log.index("WB > OpenAct") + 1] == "Not asked: see the status above" and "Asking..." not in log,
          (got, log))
    bpy.data.node_groups.remove(tree)
    settle(lambda: True, 30)  # WA's Pick answer arrives for a node that is gone


def phase_failed():
    """A cell whose ask failed is not sent again until Ask Again: the table has its row in error, as the counts do,
    the sidebar says it is not asked, and Show Question that asking it failed."""
    tree = new_tree("Failed")
    ask = tree.nodes.new(engine.ASK_ID)
    ask.inputs["Question"].default_value = "Is it made of glass?"
    user("Failer", tree)
    key = engine._jobs(ask)[0][1].rows[0].cell.key
    pipeline.FAILED[key] = "The model refused: no"  # as a refused answer leaves it
    before = len(recorded())
    engine.schedule(immediate=True)
    settle(lambda: state(ask) == "error", 30)
    for _ in range(3):
        engine.run_pass()
    got = status(ask)
    _label, job, entry = got.uses[0]
    log, text = sidebar(ask), ops.question_text(ops.row_question(ask, 0, 0))
    check("failed cell: nothing is sent again; its row is in error in the table as in the counts, the sidebar says "
          "it is not asked, and Show Question that asking it failed",
          len(recorded()) == before and got.text == "The model refused: no" and entry is None
          and ui.table_rows(engine.SPECS[ask.bl_idname], job) == [("Is it made of glass?", "", None, "error")]
          and ('BLANK1', "1 row: 1 in error") in ui.status_rows(got) and "Not asked: see the status above" in log
          and "Asking..." not in log
          and text.endswith("Asking it failed (see the status): this is what was sent. Ask Again asks it again"),
          (got, log, text))
    pipeline.FAILED.pop(key, None)
    bpy.data.node_groups.remove(tree)
    engine.run_pass()


def phase_slots():
    """A Pick of material slots, one of them empty, before its first answer: its table so far knows only the empty
    slot, so it has no answer yet (Index -1). Its line says "Asking...", not a summary, with no top items, and its
    table lists the materials with no answer and the empty slot as one."""
    tree = new_tree("Slots")
    pick = tree.nodes.new(engine.PICK_ID)
    pick.list_source = 'MATERIALS'
    pick.inputs["Prompt"].default_value = "Something shiny"
    obj = user("Shiny", tree)
    for material in (bpy.data.materials.new("Steel"), bpy.data.materials.new("Gold"), None):
        obj.data.materials.append(material)
    pick.inputs["Object"].default_value = obj
    engine.run_pass()  # asks its one cell; no tick, so no answer arrives meanwhile
    got, log = status(pick), sidebar(pick)
    _label, job, entry = got.uses[0]
    check("slots: before its first answer, a Pick of material slots with an empty slot shows a table so far with no "
          "answer yet (Index -1): its line says 'Asking...', with no top items, and its table lists the materials "
          "with no answer and the empty slot as one",
          entry is not None and entry.rows.src == "ppc" and entry.single["Index"] == -1
          and log[log.index("Shiny > OpenAct") + 1] == "Asking..." and "0.0%" not in log
          and ui.table_rows(engine.SPECS[pick.bl_idname], job, entry) == [
              ("Steel", "", None, "pending"), ("Gold", "", None, "pending"), (ui.EMPTY_SLOT, "", None, "code")],
          (got, log))
    bpy.data.node_groups.remove(tree)
    bpy.data.objects.remove(obj)
    engine.run_pass()


def phase_queue():
    """The Model box's queue (§17.2): the chunks waiting in each class and the cells in flight. With the worker held
    as during a render job, a 100-item Match is queued in chunks, and no cell is in flight."""
    tree = new_tree("Queue")
    match = tree.nodes.new(MatchSpec.idname)
    match.inputs["Question"].default_value = "Is it heavy?"
    match.inputs["Items"].default_value = ", ".join(f"load {n:03}" for n in range(100))
    user("Queuer", tree)
    server.Worker.render_running.set()  # the worker takes no chunk, as during a render job
    time.sleep(0.3)  # a turn of its loop, so it sees that before anything is queued
    try:
        engine.run_pass()
        job = status(match).uses[0][1]
        chunks = len(pipeline.chunked([row.cell for row in job.rows if row.cell is not None]))
        log = drawn(lambda layout: ui.draw_server(layout))
    finally:
        server.Worker.render_running.clear()
    check("queue: the Model box counts the chunks waiting in each class (a 100-item table in the background is P2) and "
          "no cell in flight, where it said every queued cell was in flight",
          chunks > 1 and f"Queue: 0/0/{chunks} chunks (P0/P1/P2)" in log and "In flight: 0 cells" in log
          and len(pipeline.PENDING) == 100, (chunks, log))
    settle(lambda: state(match) == "complete", 60)
    bpy.data.node_groups.remove(tree)
    engine.run_pass()


# --------------------------------------------------------------------------------- warning and error lines

def phase_lines():
    folder = os.path.join(TMP, "lines")
    os.makedirs(folder)
    answers = write(folder, "answers.csv", "answer,when\nyes,It is soft to the touch.\n\nno,It is hard.\n")
    factors = shutil.copy(os.path.join(FIXTURES, "life_factors.csv"), folder)
    outcomes = write(folder, "outcomes.csv", "answer,when,if\nBirth,The cell comes to life.,state=dead\n"
                                             "Stays empty,The cell stays empty.,state=dead\n"
                                             "Ghost,Never offered.,state=dead; state=alive\n")
    tree = new_tree("Lines")
    warned = tree.nodes.new(engine.ASK_ID)
    warned.inputs["Question"].default_value = "Is it soft?"
    warned.inputs["Answers"].default_value = answers
    broken = tree.nodes.new(engine.ASK_ID)
    broken.inputs["Question"].default_value = "Is it hard?"
    broken.inputs["Answers"].default_value = os.path.join(folder, "missing.csv")
    rules = tree.nodes.new(RulesSpec.idname)
    rules.inputs["Factors"].default_value = factors
    rules.inputs["Outcomes"].default_value = outcomes
    user("Liner", tree)
    engine.run_pass()  # asks the Rules node's model cells; no tick, so no answer arrives meanwhile
    got, log = status(rules), sidebar(rules)
    entry = got.uses[0][2]
    check("asking: while a Rules node's cells that an outcome applies to are asked, its table so far has no answer yet "
          "(only the cells no outcome applies to are known), so its line says 'Asking...', although its status is the "
          "error of those cells",
          got.state == "error" and got.text.startswith("No outcome applies to ") and got.pending == 9
          and entry is not None and entry.rows.src == "p" * 9 + "e" * 9
          and log[log.index("Liner > OpenAct") + 1] == "Asking...", (got, log))
    engine.schedule(immediate=True)
    done, _seen = settle(lambda: state(warned) == "warning" and state(broken) == "error"
                         and bool(S.load(rules).entries), 60)
    check("lines: the nodes settle", done, {node.name: status(node) for node in (warned, broken, rules)})

    got = status(warned)
    warning = next((problem.message for problem in got.problems if problem.severity == "warning"), "")
    check("warning line: the summary, then the number of warnings and the first one; the triangle icon",
          got.state == "warning" and got.icon == 'ERROR' and warning.startswith("answers.csv: Line 3 is blank")
          and got.text == f"{engine.SPECS[warned.bl_idname].summary(got.uses[0][2])} · 1 warning: {warning}", got)
    check("warning line: status_rows lists the warning with the triangle", ('ERROR', warning) in ui.status_rows(got),
          ui.status_rows(got))
    got = status(broken)
    check("error line: a missing CSV is an error with the CANCEL icon, which status_rows lists once",
          got.state == "error" and got.icon == 'CANCEL' and got.text.startswith("Can't find ")
          and got.text.endswith("missing.csv") and ui.status_rows(got).count(('CANCEL', got.text)) == 1,
          ui.status_rows(got))
    got, entry = status(rules), S.load(rules).entries[0]
    check("Rules: cells no outcome applies to give the error line, and the rest of the table still shows",
          got.state == "error" and got.text == "No outcome applies to state=alive, neighbours=0; state=alive, "
          "neighbours=1; state=alive, neighbours=2 and 6 more cells" and (got.rows, got.answered, got.errors) == (
              18, 9, 9) and entry.rows.src == "m" * 9 + "e" * 9, (got, entry.rows.src))
    check("Rules: an outcome that is never offered is a warning of the record",
          ('ERROR', "outcomes.csv line 4: 'Ghost' is never offered: its 'if' matches no cell") in ui.status_rows(got),
          ui.status_rows(got))
    return tree, warned, answers


def phase_crowd():
    """More than 16 distinct live word sets on one node (§8.1)."""
    tree = new_tree("Crowd")
    tree.interface.new_socket("Question", in_out='INPUT', socket_type='NodeSocketString')
    ask = tree.nodes.new(engine.ASK_ID)
    tree.links.new(tree.nodes.new("NodeGroupInput").outputs["Question"], ask.inputs["Question"])
    users = [user(f"Crowd {i}", tree, Question=f"Is this object number {i}?") for i in range(17)]
    engine.schedule(immediate=True)
    done, _seen = settle(lambda: state(ask) == "error", 60)
    got = status(ask)
    check("over-cap: 17 live word sets on one node give the error line naming the limit",
          done and got.icon == 'CANCEL' and got.contexts == 17
          and got.text == "17 objects use this tree with different words; the limit is 16: split the tree", got)
    over = [banner for banner in ui.live_banners(bpy.context) if banner[0] == "over_cap"]
    check("over-cap banner: names the node and the limit", over == [
        ("over_cap", 'CANCEL', "Ask: 17 objects use this tree with different words; the limit is 16: split the tree",
         ())], over)
    for obj in users:
        bpy.data.objects.remove(obj)
    engine.schedule(immediate=True)
    settle(lambda: state(ask) == "hint", 30)
    check("over-cap banner: gone with the objects", "over_cap" not in banner_kinds(), banner_kinds())


# ------------------------------------------------------------------------------------------ banners

def phase_banners():
    install = ("install", 'ERROR', "Model not installed: Install Model downloads it once (about 12 GB); Locate "
               "Existing Runtime uses one already on this computer",
               ("openact.install_runtime", "openact.locate_runtime"))
    probe = "The runtime's Python at C:\\uv\\python is missing or broken: reinstall the runtime"
    port = f"Port {PORT} is used by another program. Pick another port in the preferences"
    cases = [
        (ui.banners("not_installed", "The model runtime is not installed"), [install]),
        (ui.banners("unknown", server.CHECKING), [("runtime", 'SORTTIME', server.CHECKING,
                                                     ("openact.locate_runtime",))]),
        (ui.banners("error", probe, probe=probe), [("runtime", 'ERROR', probe, ("openact.locate_runtime",))]),
        (ui.banners("error", port), [("port", 'ERROR', port, ())]),
        (ui.banners("error", "Could not start the model server: denied"), []),
        (ui.banners("ready", "Qwen/Qwen3.5-4B on cuda"), []),
        (ui.banners("ready", "", blocked=2), [("trust", 'ERROR', "This file is not trusted, so 2 reads outside its "
                                               "folder were blocked", ("openact.allow_file", "openact.trust_folder"))]),
        (ui.banners("ready", "", notes=["a.csv: note"]), [("import_csv", 'INFO', "a.csv: note", ())]),
        (ui.banners("ready", "", over=["Ask: too many"]), [("over_cap", 'CANCEL', "Ask: too many", ())]),
    ]
    wrong = [(got, wanted) for got, wanted in cases if got != wanted]
    check("banners: model not installed, checking the runtime, a failed probe, a port conflict, an untrusted file's "
          "blocked reads, Import CSV notes and over-cap; nothing otherwise", not wrong, wrong)
    both = ui.banners("not_installed", "", blocked=1, notes=["a.csv: note"], over=["Ask: too many"])
    check("banners: most urgent first", [banner[0] for banner in both] == ["install", "trust", "over_cap", "import_csv"],
          both)
    check("banners: while the fake server is ready, no server banner", server.status()[0] == "ready"
          and not set(banner_kinds()) & set(ui.SERVER_BANNERS), (server.status(), banner_kinds()))


def phase_import_csv(tree, answers):
    """§7.1: the artist's own Import CSV node on a file an OpenAct node reads."""
    plugged = tree.nodes.new("GeometryNodeImportCSV")
    plugged.name = "Plugged CSV"
    next(s for s in plugged.inputs if s.identifier == "Path").default_value = answers
    other = tree.nodes.new(engine.ASK_ID)
    other.inputs["Question"].default_value = "Is it very soft?"
    tree.links.new(plugged.outputs[0], other.inputs["Answers CSV"])
    engine.schedule(immediate=True)
    settle(lambda: state(other) in ("warning", "complete"), 30)
    ident = csvfmt.identity(answers)
    check("Import CSV note: none for an Import CSV node that only feeds an OpenAct node",
          ident in watch.WATCH.files and ident not in ui.import_csv_notes(), ui.import_csv_notes())
    artist = tree.nodes.new("GeometryNodeImportCSV")
    artist.name = "Artist CSV"
    next(s for s in artist.inputs if s.identifier == "Path").default_value = answers
    tree.links.new(artist.outputs[0], next(n for n in tree.nodes if n.bl_idname == "NodeGroupOutput").inputs[0])
    engine.run_pass()
    note = ui.import_csv_notes().get(ident)
    check("Import CSV note: an Import CSV node the artist uses on a node's file gets the note, naming it",
          note == ui.IMPORT_CSV_NOTE.format("Artist CSV") and note.startswith("Import CSV 'Artist CSV' reads the same "
                                                                             "file; Blender won't re-read it"), note)
    found = [banner for banner in ui.live_banners(bpy.context) if banner[0] == "import_csv"]
    check("Import CSV note: a banner names the file", found == [("import_csv", 'INFO', f"answers.csv: {note}", ())],
          found)


def phase_trust():
    """An untrusted .blend whose Ask and Rules nodes read CSVs outside its folder (§16.1). The banner counts the
    blocked reads of the nodes that exist, and either button lets every node read its CSVs: the trust state is in
    every plan's fingerprint, so the pass plans them again, the Rules node too."""
    far = write(OUTSIDE, "far.csv", "answer,when\nyes,It is far.\nno,It is near.\n")
    tree = new_tree("Trust")
    reader = tree.nodes.new(engine.ASK_ID)
    reader.inputs["Question"].default_value = "Is it far?"
    reader.inputs["Answers"].default_value = far
    rules = tree.nodes.new(RulesSpec.idname)
    rules.inputs["Question"].default_value = "Which one?"
    rules.inputs["Factors"].default_value = write(OUTSIDE, "far_factors.csv", "factor,kind,level,when\n"
                                                  "heat,amount,cold,Cold.\nheat,amount,hot,Hot.\n")
    rules.inputs["Outcomes"].default_value = write(TMP, "near_outcomes.csv", "answer,when\nA,The first.\n"
                                                   "B,The second.\n")
    user("Truster", tree)
    gone = new_tree("Gone")  # a second tree, deleted while its read is blocked
    stale = gone.nodes.new(engine.ASK_ID)
    stale.inputs["Question"].default_value = "Is it gone?"
    stale.inputs["Answers"].default_value = write(OUTSIDE, "gone.csv", "answer,when\nyes,It is gone.\nno,It stays.\n")
    blend = os.path.join(TMP, "status.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend)
    trust.TRUST["opened"] = blend  # as if opened from disk: not trusted
    engine.schedule(immediate=True)
    blocked = (reader, rules, stale)
    settle(lambda: all(status(node) and status(node).text.startswith(trust.BLOCKED) for node in blocked), 30)

    def trust_banner(count):
        return [("trust", 'ERROR', f"This file is not trusted, so {count} reads outside its folder were blocked",
                 ("openact.allow_file", "openact.trust_folder"))]

    found = [banner for banner in ui.live_banners(bpy.context) if banner[0] == "trust"]
    check("trust banner: reads blocked in an untrusted file show it, with Allow This File and Trust This Folder",
          not trust.is_trusted() and all(state(node) == "error" for node in blocked) and found == trust_banner(3),
          (found, [status(node) for node in blocked]))
    uid = stale.uid
    bpy.data.node_groups.remove(gone)  # frees no node, as an undo that takes a node away
    engine.run_pass()
    found = [banner for banner in ui.live_banners(bpy.context) if banner[0] == "trust"]
    check("trust banner: a deleted node's blocked read no longer counts: the pass drops its status record",
          uid not in engine.STATUS and found == trust_banner(2), (found, uid in engine.STATUS))
    chain = new_tree("Chain")  # an Ask reading the Item of a Pick whose Items CSV is blocked
    picker = chain.nodes.new(engine.PICK_ID)
    picker.inputs["Prompt"].default_value = "something far away"
    picker.inputs["Items"].default_value = write(OUTSIDE, "far_items.csv", "answer\nboat\ncar\n")
    asker = chain.nodes.new(engine.ASK_ID)
    asker.inputs["Question"].default_value = "Is it fast?"
    chain.links.new(picker.outputs["Item"], asker.inputs["Context"])
    user("Chainer", chain)
    engine.run_pass()
    found = [banner for banner in ui.live_banners(bpy.context) if banner[0] == "trust"]
    check("trust banner: a node that reads a blocked node's answer doesn't count its blocked read again",
          status(asker).text.startswith("Context ← ") and trust.BLOCKED in status(asker).text
          and found == trust_banner(3), (found, status(asker)))
    bpy.data.node_groups.remove(chain)
    engine.run_pass()

    bpy.ops.openact.allow_file()
    check("trust banner: gone once the file is allowed", trust.is_trusted() and "trust" not in banner_kinds(),
          banner_kinds())
    engine.schedule(immediate=True)
    settle(lambda: state(reader) == state(rules) == "complete", 30)
    check("Allow This File: the pass plans the blocked nodes again, and each reads its CSV, the Rules node's Factors "
          "CSV too", state(reader) == state(rules) == "complete" and bool(S.load(rules).entries),
          (status(reader), status(rules)))
    P._Defaults.trusted_files.clear()
    engine.schedule(immediate=True)
    settle(lambda: status(rules).text.startswith(trust.BLOCKED), 30)
    check("trust: no longer allowed, the reads are blocked again",
          all(status(node).text.startswith(trust.BLOCKED) for node in (reader, rules)), (status(reader), status(rules)))
    result = bpy.ops.openact.trust_folder()
    check("Trust This Folder: trusts the .blend's folder, and the banner goes",
          result == {'FINISHED'} and P._Defaults.trusted_folders == [os.path.dirname(os.path.normpath(blend))]
          and trust.is_trusted() and "trust" not in banner_kinds(), P._Defaults.trusted_folders)
    engine.schedule(immediate=True)
    settle(lambda: state(reader) == state(rules) == "complete", 30)
    check("Trust This Folder: the pass plans them again too, and each reads its CSV, the Rules node's too",
          state(reader) == state(rules) == "complete", (status(reader), status(rules)))
    ui.import_csv_notes()  # worked out once per pass, on its first draw
    touched, real_realpath = [], os.path.realpath
    os.path.realpath = lambda path, *args, **kwargs: touched.append(path) or real_realpath(path, *args, **kwargs)
    try:
        engine.answers_for(reader)
        ui.live_banners(bpy.context)
    finally:
        os.path.realpath = real_realpath
    check("sidebar draw: with no read blocked, answers_for() and the banners resolve no path (os.path.realpath), "
          "although is_trusted() would, for a file trusted by its folder", trust.is_trusted() and not touched, touched)
    P._Defaults.trusted_folders.clear()
    trust.TRUST["opened"] = ""  # created in this session again

    for group in list(bpy.data.node_groups):
        bpy.data.node_groups.remove(group)
    engine.run_pass()
    check("status: with no OpenAct node left, the pass keeps no status record", not engine.STATUS,
          list(engine.STATUS))


# ------------------------------------------------------------------------------------------ drawing

class FakeLayout:
    """Stands in for the UILayout a headless Blender has none of: records labels and buttons and, as Blender does,
    refuses unknown icons, operators, operator properties and layout attributes."""

    __slots__ = ("log", "alert", "active", "enabled", "use_property_split", "operator_context")

    def __init__(self, log):
        self.log = log

    def _child(self, *_args, **_kwargs):
        return FakeLayout(self.log)

    column = row = box = split = _child

    def label(self, text="", icon='NONE'):
        assert icon in ICONS, icon
        self.log.append(text)

    def prop(self, data, name, **_kwargs):
        assert hasattr(data, name), name

    def progress(self, factor=0.0, type='BAR', text=""):  # noqa: A002 - UILayout's own keyword
        assert 0.0 <= factor <= 1.0 and type == 'BAR', factor
        self.log.append(text)

    def operator(self, idname, text=None, icon='NONE', depress=False):
        assert icon in ICONS, icon
        module, name = idname.split(".")
        self.log.append(f"[{idname}]")
        return FakeProperties(getattr(getattr(bpy.ops, module), name).get_rna_type(), self.log, idname)


class FakeProperties:
    def __init__(self, rna, log, idname):
        object.__setattr__(self, "names", set(rna.properties.keys()))
        object.__setattr__(self, "log", log)
        object.__setattr__(self, "idname", idname)

    def __setattr__(self, name, value):
        assert name in self.names, (self.idname, name)
        self.log.append((self.idname, name, value))


def snapshot():
    """What drawing must leave alone: the plan cache, and every OpenAct node's answers, group and sockets."""
    return ({key: id(plan) for key, plan in engine.PLANS.items()},
            [(node.entries, [n.name for n in node.node_tree.nodes], [s.name for s in node.outputs])
             for _tree, node in engine.iter_nodes()])


def drawn(draw):
    log = []
    draw(FakeLayout(log))
    return log


def sidebar(node):
    """What the Active Node panel draws for the node (drawn())."""
    ctx = SimpleNamespace(window_manager=bpy.context.window_manager, preferences=bpy.context.preferences,
                          space_data=SimpleNamespace(tree_type='GeometryNodeTree', edit_tree=node.id_data))
    return drawn(lambda layout: type(node).draw_buttons_ext(node, ctx, layout))


def phase_draw(pick, warned):
    folder = os.path.join(TMP, "draw")
    os.makedirs(folder)
    labels = write(folder, "labels.csv", "answer,when,roughness\nmetal,steel or iron,0.3\nwood,planks or bark,0.6\n")
    tree = new_tree("Draw")
    classify = tree.nodes.new(ClassifySpec.idname)
    classify.inputs["Items"].default_value = "pan, spoon"
    classify.inputs["Labels"].default_value = labels
    match = tree.nodes.new(MatchSpec.idname)
    match.inputs["Question"].default_value = "Would you find it in a kitchen?"
    match.inputs["Items"].default_value = ", ".join(f"i{n:02}" for n in range(60))
    user("Drawer", tree)
    engine.schedule(immediate=True)
    done, _seen = settle(lambda: state(classify) == state(match) == "complete", 60)
    check("draw: the scene settles", done, (status(classify), status(match)))
    wm = bpy.context.window_manager
    ctx = SimpleNamespace(window_manager=wm, preferences=bpy.context.preferences,
                          space_data=SimpleNamespace(tree_type='GeometryNodeTree', edit_tree=None))
    frozen = snapshot()
    log = drawn(lambda layout: ui.NODE_PT_openact.draw(SimpleNamespace(layout=layout), ctx))
    check("draw: with no OpenAct node active, the panel shows the Model box and the hint",
          "Model ready" in log and "Add nodes with Shift A > OpenAct" in log and "[openact.server_stop]" in log, log)

    ctx.space_data.edit_tree = pick.id_data
    pick.id_data.nodes.active = pick
    # the node's own draw functions, as the panels call them: called on the node, RNA wants a real layout
    log = drawn(lambda layout: type(pick).draw_buttons(pick, ctx, layout))
    check("draw: the node body shows its one status line", status(pick).text in log, log)
    log = drawn(lambda layout: type(pick).draw_buttons_ext(pick, ctx, layout))
    check("draw: the Active Node panel shows the status record, the context, Pick's top items and its table, with "
          "Show Question on each row", status(pick).text in log and "Filler > OpenAct" in log and "Table" in log
          and all(name in log for name in ("table", "lamp", "armchair"))
          and ("openact.show_question", "row", 2) in log and "[openact.write_overrides]" in log, log)

    ctx.space_data.edit_tree = warned.id_data
    warned.id_data.nodes.active = warned
    log = drawn(lambda layout: ui.NODE_PT_openact_files.draw(SimpleNamespace(layout=layout), ctx))
    check("draw: the Files panel shows the file, its rows and columns, when it was read, the Import CSV note, its "
          "problems, and Open File", "answers.csv" in log and any(
              re.fullmatch(r"2 data rows · read \d\d:\d\d:\d\d", str(entry)) for entry in log)
          and "Text columns: answer, when" in log and any("Import CSV 'Artist CSV'" in str(entry) for entry in log)
          and any(str(entry).startswith("Line 3") for entry in log) and "[openact.open_file]" in log, log)
    log = drawn(lambda layout: ops.OPENACT_OT_show_question.draw(SimpleNamespace(
        layout=layout, text=ops.question_text(ops.row_question(pick, 0, 0)), bl_idname="openact.show_question",
        uid=pick.uid, context_index=0, row=0), ctx))
    check("draw: the Show Question popup lists the request and a Copy button",
          any(str(entry).startswith("State:") for entry in log) and ("openact.show_question", "copy", True) in log,
          log)
    log = drawn(lambda layout: ui.draw_preferences(P.get_prefs(), ctx, layout))
    log += drawn(lambda layout: ui.draw_server(layout, ui.banners("not_installed", "")[0]))
    check("draw: the preferences, and the Model box with the not-installed banner and its buttons",
          "Model ready" in log and "[openact.install_runtime]" in log and "[openact.locate_runtime]" in log, log)
    server.STATE.install_warning = "Warning: the GPU has 8 GB of VRAM"
    status_of = server.status
    try:
        log = drawn(lambda layout: ui.draw_server(layout))
        server.status = lambda: ("stopped", "Runtime installed. Warning: the GPU has 8 GB of VRAM")  # auto start off
        installed = drawn(lambda layout: ui.draw_server(layout))
    finally:
        server.status = status_of
        server.STATE.install_warning = ""
    check("draw: the Model box keeps the install's VRAM warning after the status moved on",
          "Warning: the GPU has 8 GB of VRAM" in log and "Model ready" in log, log)
    check("draw: ...and shows it once while the status message still holds it",
          "Model stopped" in installed and sum("8 GB" in str(entry) for entry in installed) == 1, installed)

    ctx.space_data.edit_tree = tree
    tree.nodes.active = match
    bpy.ops.openact.table_view('EXEC_DEFAULT', context_index=0, page=1)
    log = drawn(lambda layout: type(match).draw_buttons_ext(match, ctx, layout))
    check("paging: page 2 of 60 rows shows rows 51-60, with the page number in the WindowManager",
          wm.openact_page == 1 and "2 / 2" in log and "i50" in log and "i59" in log and "i49" not in log
          and ("openact.show_question", "row", 59) in log, log)
    bpy.ops.openact.table_view('EXEC_DEFAULT', context_index=0, page=0)
    check("draw: drawing all of that planned nothing and wrote nothing", snapshot() == frozen, "a plan or node changed")

    write(folder, "labels.csv", "answer,when,roughness,metallic\nmetal,steel or iron,0.3,1\n"
                                "wood,planks or bark,0.6,0\n")
    tree.nodes.active = classify
    frozen = snapshot()
    drawn(lambda layout: type(classify).draw_buttons_ext(classify, ctx, layout))
    drawn(lambda layout: ui.NODE_PT_openact.draw(SimpleNamespace(layout=layout), ctx))
    drawn(lambda layout: ui.NODE_PT_openact_files.draw(SimpleNamespace(layout=layout), ctx))
    check("draw: nothing is planned or written while drawing, even for a Classify whose Labels CSV gained a column",
          snapshot() == frozen and "metallic" not in classify.outputs, [s.name for s in classify.outputs])
    engine.run_pass()
    check("draw: the next pass plans that edit (its payload output appears)", "metallic" in classify.outputs,
          [s.name for s in classify.outputs])
    used = set(engine.ICONS.values()) | set(ui.ROW_ICONS.values()) | set(ui.PROBLEM_ICONS.values())
    check("icons: every state, row and problem icon exists in this Blender", used <= ICONS, used - ICONS)


def phase_open_file(answers):
    def run(path):
        try:
            return bpy.ops.openact.open_file('EXEC_DEFAULT', filepath=path)
        except RuntimeError as error:  # how an operator's ERROR report reaches a Python caller
            return str(error).strip()

    opened, real = [], ops._open_path
    ops._open_path = opened.append  # the system would open a spreadsheet program
    try:
        got = [run(answers), run(os.path.join(TMP, "lines", "tool.exe")), run(os.path.join(TMP, "gone.csv"))]
    finally:
        ops._open_path = real
    check("Open File: opens a CSV the node reads; refuses anything but a CSV, and a missing file",
          got[0] == {'FINISHED'} and opened == [answers] and "Only CSV files open from here" in got[1]
          and "Can't find" in got[2], (got, opened))


try:
    picked, asked = phase_fill()
    phase_eta()
    phase_question(picked, asked)
    phase_table(picked, asked)
    phase_held()
    phase_failed()
    phase_slots()
    lines_tree, warned_node, answers_csv = phase_lines()
    phase_crowd()
    phase_banners()
    phase_import_csv(lines_tree, answers_csv)
    phase_draw(picked, warned_node)
    phase_queue()
    phase_open_file(answers_csv)
    phase_trust()
except Exception:
    import traceback

    traceback.print_exc()
    check("no exception", False)
finally:
    subprocess.Popen = _popen
    if server.STATE.port == PORT:  # never another port: /shutdown would stop someone else's server
        server.stop()
    shutil.rmtree(TMP, ignore_errors=True)
    shutil.rmtree(OUTSIDE, ignore_errors=True)
finish()

