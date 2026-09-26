"""The model server with its fake backend: no GPU, torch or runtime needed.

Each server is openact/runtime/openact_server.py --backend fake, run by this Python on a free port.
- Auth: launched with OPENACT_TOKEN, a POST without the token or with a wrong one gets 403, and so
  does any request with an Origin header or a foreign Host, token or not. /health needs no token,
  /shutdown does. Launched without OPENACT_TOKEN, requests need no token, but Origin and Host are
  still checked.
- The per-question cache: cached_questions on a repeat, only new questions scored, the /health
  counters, options.debias honoured per request; reordered options or state keys miss the cache.
- /v1/capabilities; 422 for 257 questions and for an invalid question or one over the token limit,
  naming it; a 40-option tournament whose 40 probabilities sum to 1, also next to question ids
  that look like the tournament's own keys.
- Bad requests: a negative or non-numeric Content-Length, a real body over 4 MB, deeply nested JSON
  and lone surrogates all get a reply. Refused requests with bodies that are slow to parse, sent
  nonstop, leave /health within the add-on's 1 s probe timeout.
- --fake-script answers, the --record log of every request (refused ones too), a --record path
  that can't be written, and the same fake answers in every process.
- Cold loads: during --fake-load-seconds a POST gets an immediate 503 "loading"; after
  --fake-load-fail, or with --fake-no-cuda, a 503 "error".

Run:  python tests/server_test.py
"""

import http.client
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import FAILURES, ROOT, check, finish  # noqa: E402

SERVER = os.path.join(ROOT, "openact", "runtime", "openact_server.py")
TMP = tempfile.mkdtemp(prefix="openact_server_")
TOKEN = secrets.token_hex(32)
MAX_BODY = 4 * 1024 * 1024
SCRIPT = [  # --fake-script: the first matching rule answers
    {"state": "snowy", "question": "Is it night?", "answer": "yes"},
    {"question": "Is it night?", "answer": {"yes": 1, "no": 3}},
    {"question": "Which item", "answer": "3"},
]
LAUNCHED = []  # (name, process)
RECORDED = []  # (status, request, answers) of every /v1/systemone request to the token server


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def call(port, method, path, body=None, headers=None):
    """(status, JSON reply) of one request; body is JSON-encoded unless it is bytes. http.client, unlike
    urllib, never uses a proxy."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        data = body if body is None or isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        connection.request(method, path, body=data, headers={"Content-Type": "application/json", **(headers or {})})
        response = connection.getresponse()
        return response.status, json.loads(response.read() or b"null")
    finally:
        connection.close()


def health(port):
    return call(port, "GET", "/health")[1]


def wait_status(port, wanted, seconds):
    deadline = time.time() + seconds
    while time.time() < deadline:
        if health(port).get("status") == wanted:
            return True
        time.sleep(0.05)
    return False


def exited(process, seconds):
    try:
        process.wait(timeout=seconds)
        return True
    except subprocess.TimeoutExpired:
        return False


def launch(name, *flags, token=None):
    """Start a fake-backend server; returns (port, process) once it answers."""
    port = free_port()
    env = {key: value for key, value in os.environ.items() if key != "OPENACT_TOKEN"}
    if token:
        env["OPENACT_TOKEN"] = token
    with open(os.path.join(TMP, name + ".log"), "w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", SERVER, "--backend", "fake", "--port", str(port), "--parent-pid", str(os.getpid()),
             *flags], stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env)
    LAUNCHED.append((name, process))
    deadline = time.time() + 30
    while time.time() < deadline and process.poll() is None:
        try:
            call(port, "GET", "/health")
            return port, process
        except OSError:
            time.sleep(0.05)
    raise RuntimeError(f"the {name} server did not start (exit code {process.poll()})")


def noul(text):
    return {"type": "noul", "instructions": text}


def choice(text, criteria):
    return {"type": "choice", "instructions": text, "criteria": criteria}


def answer(reply, qid):
    return (reply.get("answers") or {}).get(qid) or {}


def ask(port, request, token=TOKEN, **headers):
    """POST /v1/systemone to the token server; remembers what its --record log must show."""
    if token:
        headers["X-OpenAct-Token"] = token
    code, reply = call(port, "POST", "/v1/systemone", request, headers)
    if isinstance(request, bytes) or code == 403:  # logged as the body's text: not JSON, or refused unparsed
        sent = request.decode("utf-8", "replace") if isinstance(request, bytes) else json.dumps(request)
    else:
        sent = request
    RECORDED.append((code, sent, reply.get("answers")))
    return code, reply


def reply_code(port, body, headers):
    """The status of a POST /v1/systemone, or the error's name when no reply arrives."""
    try:
        return call(port, "POST", "/v1/systemone", body, headers)[0]
    except OSError as error:
        return type(error).__name__


WINDOW = {"state": "A plain room.", "questions": {"q": noul("Is there a window?")}}


def phase_cold_load():
    loading, process = launch("loading", "--fake-load-seconds", "5")
    started = time.perf_counter()
    code, reply = call(loading, "POST", "/v1/systemone", WINDOW)
    seconds = time.perf_counter() - started
    check("cold load: a POST during loading gets 503 loading at once",
          code == 503 and reply == {"status": "loading"} and seconds < 2.0, (code, reply, f"{seconds:.2f}s"))
    failed, _process = launch("failed", "--fake-load-fail")
    check("failed load: /health reports the error", wait_status(failed, "error", 15), health(failed).get("status"))
    code, reply = call(failed, "POST", "/v1/systemone", WINDOW)
    check("failed load: a POST gets 503 error with the message",
          code == 503 and reply.get("status") == "error" and "--fake-load-fail" in reply.get("error", ""),
          (code, reply))
    no_cuda, _process = launch("no-cuda", "--fake-no-cuda")
    check("no CUDA: /health reports the error", wait_status(no_cuda, "error", 15), health(no_cuda).get("status"))
    code, reply = call(no_cuda, "POST", "/v1/systemone", WINDOW)
    check("no CUDA: a POST gets 503 error with the driver hint",
          code == 503 and "CUDA is not available to PyTorch (is the NVIDIA driver too old" in reply.get("error", ""),
          (code, reply))
    return loading, process


def phase_bad_record():
    missing = os.path.join(TMP, "missing", "record.jsonl")
    result = subprocess.run(
        [sys.executable, "-u", SERVER, "--backend", "fake", "--port", str(free_port()), "--record", missing],
        capture_output=True, text=True, timeout=60)
    output = result.stdout + result.stderr
    check("a --record path that can't be written stops the start, with a message",
          result.returncode != 0 and "cannot write --record" in output, (result.returncode, output[-300:]))


def phase_auth(port):
    code, reply = ask(port, WINDOW, token=None)
    check("403 without the token", code == 403 and "token" in reply.get("error", "").lower(), (code, reply))
    code, reply = ask(port, WINDOW, token=secrets.token_hex(32))
    check("403 with a wrong token", code == 403, (code, reply))
    code, window = ask(port, WINDOW)
    check("200 with the token", code == 200 and 0.0 < answer(window, "q").get("noul", -1) < 1.0, (code, window))
    code, reply = ask(port, WINDOW, Origin="http://evil.example")
    check("403 with an Origin header, even with the token", code == 403 and "Origin" in reply.get("error", ""),
          (code, reply))
    code, reply = ask(port, WINDOW, Host=f"evil.example:{port}")
    check("403 with a foreign Host, even with the token", code == 403 and "Host" in reply.get("error", ""),
          (code, reply))
    code, reply = ask(port, WINDOW, Host=f"localhost:{port}")
    check("localhost:<port> is an allowed Host", code == 200, (code, reply))
    code, reply = call(port, "GET", "/health")
    check("/health needs no token", code == 200 and reply.get("status") == "ready", (code, reply.get("status")))
    code, reply = call(port, "GET", "/health", headers={"Origin": "null"})
    check("a GET with an Origin header gets 403", code == 403, (code, reply))
    code, reply = call(port, "GET", "/v1/capabilities", headers={"Host": "evil.example"})
    check("a GET with a foreign Host gets 403", code == 403, (code, reply))
    return window


def phase_capabilities(port):
    expected = {"server": "openact-semif", "version": "1.0.0", "backend": "fake", "model_id": "fake", "slots": 16,
                "max_options": 255, "max_levels": 16, "max_questions": 256, "batch_rows": 32, "debias": True,
                "question_cache": True}
    code, caps = call(port, "GET", "/v1/capabilities")
    check("capabilities fields", code == 200 and {key: caps.get(key) for key in expected} == expected, caps)


def phase_cache(port):
    batch = {"state": "A quiet library in the afternoon.", "options": {"debias": True}, "questions": {
        "quiet": {"type": "noul", "instructions": "Is it quiet?", "criteria": {"true": "Quiet.", "false": "Loud."}},
        "read": choice("What would you read here?", {"0": "a book", "1": "a lamp", "2": "a chair", "3": "a rug"}),
        "age": {"type": "score", "instructions": "How old is the building?",
                "criteria": ["new", "recent", "old", "very old", "ancient"]},
    }}
    before = health(port)
    code, first = ask(port, batch)
    middle = health(port)
    code_again, again = ask(port, batch)
    after = health(port)
    check("first ask: nothing is cached",
          code == 200 and first.get("cached_questions") == 0 and first.get("cached") is False,
          (code, first.get("cached_questions"), first.get("cached")))
    check("a repeat: cached_questions counts every question",
          code_again == 200 and again.get("cached_questions") == 3 and again.get("cached") is True,
          (code_again, again.get("cached_questions"), again.get("cached")))
    check("a repeat: the same answers, and no rows scored",
          again.get("answers") == first.get("answers") and after["rows_scored"] == middle["rows_scored"],
          (middle["rows_scored"], after["rows_scored"]))
    quiet, read, age = answer(first, "quiet"), answer(first, "read"), answer(first, "age")
    check("answer shapes are unchanged",
          quiet.get("type") == "noul" and 0.0 <= quiet.get("noul", -1) <= 1.0
          and read.get("type") == "choice" and read.get("choice") in ("0", "1", "2", "3")
          and abs(sum(read.get("probabilities", {}).values()) - 1.0) < 1e-9 and "confidence" in read
          and age.get("type") == "score" and 0.0 <= age.get("score", -1) <= 4.0 and len(age.get("legend", {})) == 5,
          first.get("answers"))
    check("/health: 2 rows per question with debias", middle["rows_scored"] - before["rows_scored"] == 6,
          (before["rows_scored"], middle["rows_scored"]))
    counted = tuple(after[key] - before[key] for key in ("requests", "questions", "cache_hits"))
    check("/health: requests, questions and cache_hits", counted == (2, 6, 3), counted)
    check("/health: backend, model_id and ms_per_row",
          after.get("backend") == "fake" and after.get("model_id") == "fake"
          and isinstance(after.get("ms_per_row"), (int, float)) and after["ms_per_row"] >= 0,
          {key: after.get(key) for key in ("backend", "model_id", "ms_per_row")})

    grown = dict(batch, questions=dict(batch["questions"], wet=noul("Is it raining?")))
    before = health(port)
    code, reply = ask(port, grown)
    scored = health(port)["rows_scored"] - before["rows_scored"]
    check("a new question next to cached ones: only it is scored",
          code == 200 and reply.get("cached_questions") == 3 and reply.get("cached") is False and scored == 2,
          (code, reply.get("cached_questions"), scored))
    check("answers keep the request's order", list(reply.get("answers", {})) == list(grown["questions"]),
          list(reply.get("answers", {})))

    plain = dict(batch, questions={"quiet": batch["questions"]["quiet"]}, options={"debias": False})
    before = health(port)
    code, reply = ask(port, plain)
    scored = health(port)["rows_scored"] - before["rows_scored"]
    check("options.debias false: scored once, not taken from the debiased answer",
          code == 200 and reply.get("debias") is False and reply.get("cached_questions") == 0 and scored == 1,
          (code, reply.get("debias"), reply.get("cached_questions"), scored))

    # SemIf letters options by position and tournaments group them by position: order is part of the question
    forward = {str(i): f"tool {i}" for i in range(40)}
    backward = dict(reversed(list(forward.items())))
    code, reply = ask(port, {"state": "A tool shed.", "questions": {"t": choice("Which tool fits best?", forward)}})
    before = health(port)
    code, reply = ask(port, {"state": "A tool shed.", "questions": {"t": choice("Which tool fits best?", backward)}})
    scored = health(port)["rows_scored"] - before["rows_scored"]
    check("reordered options miss the cache and are scored again",
          code == 200 and reply.get("cached_questions") == 0 and scored == 8,
          (code, reply.get("cached_questions"), scored))
    check("the reordered answer lists its probabilities in the new order",
          list(answer(reply, "t").get("probabilities", {})) == list(backward))
    ask(port, {"state": {"walls": "wood", "roof": "tin"}, "questions": {"s": noul("Is it sturdy?")}})
    code, reply = ask(port, {"state": {"roof": "tin", "walls": "wood"}, "questions": {"s": noul("Is it sturdy?")}})
    check("a state with its keys in another order misses the cache",
          code == 200 and reply.get("cached_questions") == 0, (code, reply.get("cached_questions")))


def phase_limits(port):
    def questions(count):
        return {"state": "A long list.", "questions": {f"q{i}": noul(f"Is item {i} red?") for i in range(count)}}

    code, reply = ask(port, questions(257))
    check("422 for 257 questions", code == 422 and "256" in reply.get("error", "") and reply.get("question") is None,
          (code, reply))
    code, reply = ask(port, questions(256))
    check("256 questions are answered", code == 200 and len(reply.get("answers", {})) == 256, code)
    code, reply = ask(port, {"state": "s", "questions": {
        "fine": noul("Is it fine?"), "broken": {"type": "maybe", "instructions": "Is it?"}}})
    check("422 for an invalid question names it", code == 422 and reply.get("question") == "broken", (code, reply))
    code, reply = ask(port, {"state": "s", "questions": {
        "fine": noul("Is it fine?"), "wide": choice("Which one?", {str(i): f"option {i}" for i in range(256)})}})
    check("422 for 256 options names the question", code == 422 and reply.get("question") == "wide", (code, reply))
    code, reply = ask(port, {"state": "s", "questions": {
        "fine": noul("Is it fine?"), "long": noul("Is it " + "very " * 5000 + "long?")}})
    error = reply.get("error", "")
    check("422 for a question over the token limit names it, in the message too",
          code == 422 and reply.get("question") == "long" and error.startswith("long: ") and "limit" in error
          and "Row " not in error, (code, reply.get("question"), error[:80]))
    criteria = dict({str(i): f"option {i}" for i in range(40)}, **{"20": "option " + "x " * 20000})
    code, reply = ask(port, {"state": "s", "questions": {
        "fine": noul("Is it fine?"), "wide": choice("Which?", criteria)}})
    check("422 for a tournament group over the token limit names its question",
          code == 422 and reply.get("question") == "wide" and reply.get("error", "").startswith("wide: "),
          (code, reply.get("question"), reply.get("error", "")[:80]))


def phase_tournament(port):
    request = {"state": "A kitchen shelf.", "questions": {"t": choice(
        "What fits best on the shelf?", {str(i): f"thing {i}" for i in range(40)})}}
    before = health(port)
    code, reply = ask(port, request)
    scored = health(port)["rows_scored"] - before["rows_scored"]
    result = answer(reply, "t")
    probabilities = result.get("probabilities", {})
    check("40-option tournament: 40 probabilities",
          code == 200 and sorted(probabilities, key=int) == [str(i) for i in range(40)], (code, len(probabilities)))
    check("40-option tournament: they sum to 1", abs(sum(probabilities.values()) - 1.0) < 1e-9,
          sum(probabilities.values()))
    check("40-option tournament: the choice is the most probable item",
          probabilities and result.get("choice") == max(probabilities, key=probabilities.get), result.get("choice"))
    check("40-option tournament: 3 groups and a final, in both orders", scored == 8, scored)

    garage = {"state": "A garage.", "questions": {
        "x": choice("What belongs on the workbench?", {str(i): f"thing {i}" for i in range(40)}),
        "x#g0": noul("Is it tidy?"),
        "x#g1": choice("Which light?", {"0": "a lamp", "1": "a torch"}),
        "x#final": {"type": "score", "instructions": "How cold is it?", "criteria": ["warm", "cool", "cold"]},
    }}
    code, reply = ask(port, garage)
    x = answer(reply, "x").get("probabilities", {})
    check("question ids like the tournament's own keys: every answer is whole",
          code == 200 and len(x) == 40 and abs(sum(x.values()) - 1.0) < 1e-9 and min(x.values()) > 0
          and answer(reply, "x#g0").get("type") == "noul" and len(answer(reply, "x#g1").get("probabilities", {})) == 2
          and answer(reply, "x#final").get("type") == "score", (code, len(x), sum(x.values())))
    code, again = ask(port, {"state": "A garage.", "questions": {"y": garage["questions"]["x"]}})
    check("... and the cache holds the whole answer",
          code == 200 and again.get("cached_questions") == 1 and answer(again, "y") == answer(reply, "x"),
          (code, again.get("cached_questions")))


def phase_script(port):
    code, reply = ask(port, {"state": "A snowy village.", "questions": {"n": noul("Is it night?")}})
    check("script: the first matching rule answers", code == 200 and answer(reply, "n").get("noul") == 1.0,
          (code, answer(reply, "n")))
    code, reply = ask(port, {"state": "A sunny beach.", "questions": {"n": noul("Is it night?")}})
    check("script: a rule's weights become the probabilities",
          code == 200 and abs(answer(reply, "n").get("noul", -1) - 0.25) < 1e-12, (code, answer(reply, "n")))
    code, reply = ask(port, {"state": "A living room.", "questions": {
        "i": choice("Which item fits?", {str(i): f"item {i}" for i in range(40)})}})
    result = answer(reply, "i")
    check("script: the scripted item wins its group and the final",
          code == 200 and result.get("choice") == "3" and result.get("probabilities", {}).get("3") == 1.0,
          (code, result.get("choice")))


def phase_bad_requests(port):
    for length in ("-1", "abc"):
        code, reply = call(port, "POST", "/v1/systemone", None, {"Content-Length": length, "X-OpenAct-Token": TOKEN})
        check(f"Content-Length {length}: 400", code == 400, (code, reply))
    headers = {"Content-Length": str(100 << 20), "X-OpenAct-Token": TOKEN}
    code, reply = call(port, "POST", "/v1/systemone", None, headers)
    check("a Content-Length too large to skip: 413 at once, nothing read", code == 413, (code, reply))
    code = reply_code(port, b"x" * (MAX_BODY + 1), {"X-OpenAct-Token": TOKEN})
    check("a real body over 4 MB gets its 413, not a reset connection", code == 413, code)
    check("the server keeps serving after bad lengths", call(port, "GET", "/health")[0] == 200)
    code, reply = ask(port, b"[" * 100000 + b"]" * 100000)
    check("deeply nested JSON: 422, not a dropped connection",
          code == 422 and reply.get("error") == "body is not valid JSON", (code, reply))
    code, reply = ask(port, {"state": "s", "questions": {"bad": noul("Is it \ud800?")}})
    check("a lone surrogate in a question still gets its 422, --record on", code == 422, (code, reply))
    code, reply = ask(port, {"state": "s", "questions": {"\ud800": noul("Is it fine?")}})
    check("a lone surrogate in a question id still gets its answer",
          code == 200 and "\ud800" in reply.get("answers", {}), (code, reply))


def phase_record(path):
    with open(path, encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle]
    check("record: one line per request whose body was read, refused ones too", len(lines) == len(RECORDED),
          (len(lines), len(RECORDED)))
    check("record: each line holds the request as sent, its status and its answers",
          [(line.get("status"), line.get("request"), line.get("answers")) for line in lines] == RECORDED)


def phase_shutdown(port, process):
    code, reply = call(port, "POST", "/shutdown", {})
    check("/shutdown without the token gets 403", code == 403, (code, reply))
    check("the refused /shutdown leaves the server running", call(port, "GET", "/health")[0] == 200)
    code, reply = call(port, "POST", "/shutdown", {}, {"X-OpenAct-Token": TOKEN})
    check("/shutdown with the token stops the server", code == 200 and exited(process, 10), (code, reply))


def phase_hostile():
    """A web page can't pass the checks, but it can send refused requests nonstop: refusing must stay cheap."""
    port, _process = launch("hostile", token=TOKEN)
    body = b"[" + b"{}," * 1398000 + b"{}]"  # just under 4 MB of tiny objects: slow to parse
    until = time.time() + 2.5
    codes = []

    def flood(headers):
        while time.time() < until:
            codes.append(reply_code(port, body, headers))

    floods = [threading.Thread(target=flood, args=(headers,), daemon=True)
              for headers in [{"Origin": "http://evil.example"}] * 3 + [{}] * 3]
    for thread in floods:
        thread.start()
    time.sleep(0.3)
    seconds = []
    while time.time() < until:
        started = time.perf_counter()
        try:
            answered = call(port, "GET", "/health")[0] == 200
        except OSError:
            answered = False
        seconds.append(time.perf_counter() - started if answered else float("inf"))
        time.sleep(0.1)
    for thread in floods:
        thread.join(30)
    seconds.sort()
    median = seconds[len(seconds) // 2] if seconds else float("inf")
    check("refused requests sent nonstop: every /health probe answers within the add-on's 1 s timeout",
          seconds and seconds[-1] < 1.0, [round(value, 3) for value in seconds])
    check("refused requests sent nonstop: /health stays fast (median under 0.2 s)", median < 0.2, round(median, 3))
    check("refused requests sent nonstop: each gets its 403, its body skipped rather than the connection reset",
          codes and set(codes) == {403}, Counter(codes))


def phase_open_mode(port, process, window):
    check("cold load: ready once --fake-load-seconds are over", wait_status(port, "ready", 20),
          health(port).get("load_seconds"))
    code, reply = call(port, "POST", "/v1/systemone", WINDOW)
    check("without OPENACT_TOKEN, a POST needs no token", code == 200, (code, reply))
    check("fake answers are the same in every process", reply.get("answers") == window.get("answers"),
          (reply.get("answers"), window.get("answers")))
    code, reply = call(port, "POST", "/v1/systemone", WINDOW, {"Origin": "http://evil.example"})
    check("without OPENACT_TOKEN, an Origin header still gets 403", code == 403, (code, reply))
    code, reply = call(port, "POST", "/v1/systemone", WINDOW, {"Host": f"evil.example:{port}"})
    check("without OPENACT_TOKEN, a foreign Host still gets 403", code == 403, (code, reply))
    code, reply = call(port, "POST", "/shutdown", {})
    check("without OPENACT_TOKEN, /shutdown needs no token", code == 200 and exited(process, 10), (code, reply))


def main():
    try:
        loading, loading_process = phase_cold_load()
        phase_bad_record()
        record, script = os.path.join(TMP, "record.jsonl"), os.path.join(TMP, "script.json")
        with open(script, "w", encoding="utf-8") as handle:
            json.dump(SCRIPT, handle)
        port, process = launch("token", "--record", record, "--fake-script", script, token=TOKEN)
        check("the token server is ready", wait_status(port, "ready", 15), health(port).get("status"))
        window = phase_auth(port)
        phase_capabilities(port)
        phase_cache(port)
        phase_limits(port)
        phase_tournament(port)
        phase_script(port)
        phase_bad_requests(port)
        phase_record(record)
        phase_shutdown(port, process)
        phase_hostile()
        phase_open_mode(loading, loading_process, window)
    except Exception:
        check("no unexpected error", False, traceback.format_exc())
    finally:
        for _name, process in LAUNCHED:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
    if FAILURES:
        for name, _process in LAUNCHED:
            with open(os.path.join(TMP, name + ".log"), encoding="utf-8", errors="replace") as handle:
                print(f"--- {name} server log:\n{handle.read().rstrip()}", flush=True)
    else:
        shutil.rmtree(TMP, ignore_errors=True)
    finish()


main()
