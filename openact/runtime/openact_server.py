"""Local, Jev-compatible "System One" decision server used by the OpenAct add-on.

This file runs inside the OpenAct runtime (a separate Python environment that has PyTorch),
never inside Blender. It serves the request/response shape of TypeSafe's public API
(``POST /v1/systemone``) on 127.0.0.1 and answers with SemIf's method: the model is asked to
reply with an option letter and the option probabilities are read from a single forward pass.
No text is generated.

SemIf (formerly OpenJev), MIT license: https://github.com/TheoLeeCJ/SemIf-OpenJev
Default model: Qwen/Qwen3.5-4B (Apache-2.0), pinned to one revision.

Additions on top of SemIf:
* Order debiasing: every question is scored twice, with the options in the given order and in
  reverse order, and the two distributions are averaged. Small models favour early options.
* More than 16 options: SemIf has 16 answer letters (A-P), so larger Choice questions run as a
  tournament (groups of <= 16, then a final between the group winners). The returned
  probabilities are hierarchical: P(item) = P(item | its group) * P(group winner wins the final).
* A per-question LRU answer cache and a watchdog that exits when the parent process (Blender) is gone.
* Backends: SemIf on PyTorch, or a deterministic fake that needs neither torch nor a GPU (tests).
* Only local clients: a request with an Origin header, or a Host other than 127.0.0.1:<port> or
  localhost:<port>, gets 403. When OPENACT_TOKEN is set, every POST must send it as X-OpenAct-Token.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import sys
import threading
import time
import traceback
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Protocol

SERVER_NAME = "openact-semif"
SERVER_VERSION = "1.0.0"
DEFAULT_SOURCE = "Qwen/Qwen3.5-4B"
DEFAULT_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
MODEL_NAME = "semif-qwen3.5-4b"
SLOTS = 16  # SemIf answer letters A..P
MAX_CHOICE_OPTIONS = 255
MAX_SCORE_LEVELS = 16
MAX_QUESTIONS = 256
MAX_TOKENS = 4096  # SemIf's prompt limit per row
BATCH_ROWS = 32
CACHE_SIZE = 65536  # questions
MAX_BODY = 4 * 1024 * 1024
EMA_ALPHA = 2 / (64 + 1)  # ms_per_row: an EMA spanning about the last 64 scoring calls
NO_CUDA = "CUDA is not available to PyTorch (is the NVIDIA driver too old for cu128?)"


def log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


class RequestError(ValueError):
    """The request is malformed (HTTP 422)."""


def _text(value, what):
    """Instructions and criteria may be strings, objects or arrays (as in Jev's API)."""
    if isinstance(value, str):
        text = value.strip()
    elif isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        raise RequestError(f"{what} must be a string, object or array")
    if not text:
        raise RequestError(f"{what} must not be empty")
    return text


def _groups(options):
    """Split 17-255 options into 2-16 groups of at most SLOTS whose sizes differ by at most one
    (so every group has at least 8 options; e.g. 241 = 1 x 16 + 15 x 15, never 15 x 16 + 1)."""
    count = math.ceil(len(options) / SLOTS)
    base, extra = divmod(len(options), count)
    groups, start = [], 0
    for index in range(count):
        size = base + (1 if index < extra else 0)
        groups.append(options[start:start + size])
        start += size
    return groups


def _confidence(probabilities):
    top = sorted(probabilities, reverse=True)
    return top[0] - (top[1] if len(top) > 1 else 0.0)


def _sha256_json(value):
    # No sort_keys: the answer depends on the order of options and of keys in the state (SemIf's prompt).
    return hashlib.sha256(json.dumps(value, ensure_ascii=False).encode("utf-8")).hexdigest()


class Backend(Protocol):
    name: str  # "semif-torch" or "fake"
    model_id: str  # provenance for /health, e.g. "Qwen/Qwen3.5-4B@851bf6e8"
    device: str | None  # known once loaded

    def load(self) -> None: ...

    def score_rows(self, rows: list[dict]) -> list[dict]:
        """SemIf rows of up to SLOTS options sharing one state -> [{"option_ids", "probabilities",
        "input_tokens"}]. A ValueError caused by one row carries that row's id as `row`."""
        ...


class SemIfTorchBackend(Backend):
    name = "semif-torch"

    def __init__(self, args):
        self.args = args
        self.model_id = f"{args.model}@{args.revision[:8]}"
        self.device = None
        self.model = self.tokenizer = self.metadata = None

    def load(self):
        import torch
        from semif_phase1.core import load_causal_model

        device = self.args.device
        dtype = self.args.dtype
        if device == "auto":
            device = "cuda"  # never a silent CPU load: 16 GB of RAM and far too slow
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(NO_CUDA)
        if device == "cpu" and dtype == "bfloat16":
            dtype = "float32"  # SemIf's documented CPU reference path
        log(f"loading {self.args.model}@{self.args.revision[:12]} on {device} ({dtype})")
        self.model, self.tokenizer, self.metadata = load_causal_model(
            self.args.model, self.args.revision, device=device, dtype=dtype
        )
        if self.metadata["device"].startswith("cuda"):
            self.device = torch.cuda.get_device_name(0)
        else:
            self.device = self.metadata["device"]
        # Warm up kernels so the first real request is not the slow one.
        self.score_rows([
            {"id": "warmup", "state": "Warm-up.", "question": "Is this a warm-up?",
             "options": [{"id": "yes", "description": "Yes."}, {"id": "no", "description": "No."}]}
        ])

    def score_rows(self, rows):
        from semif_phase1.direct import encode_prompt, score
        from semif_phase1.shared import score_shared

        results = []
        for start in range(0, len(rows), BATCH_ROWS):
            chunk = rows[start:start + BATCH_ROWS]
            try:
                if len(chunk) == 1:
                    results.append(score(self.model, self.tokenizer, chunk[0], self.metadata, MAX_TOKENS))
                else:
                    scored, _timing = score_shared(self.model, self.tokenizer, chunk, self.metadata, MAX_TOKENS)
                    results.extend(scored)
            except ValueError as error:
                # Only tokenising again tells which row SemIf refused, e.g. one over the token limit.
                for row in chunk:
                    try:
                        encode_prompt(self.tokenizer, row, MAX_TOKENS)
                    except ValueError:
                        error.row = row["id"]
                        break
                raise
        return results


class FakeBackend(Backend):
    """Deterministic answers without torch or a GPU, for tests.

    An option's weight depends only on (state, question, option id), so answers are the same in
    every run, and a row's answer does not depend on the order of its options (a tournament still
    groups options by position). Rows over SemIf's token limit are refused as SemIf refuses them.
    --fake-script is a JSON list of rules: the first rule whose "state" and "question" texts (both
    optional) occur in a row's state and question decides that row, with its "answer": an option
    id, or a map of option ids to weights. A row with no matching rule, or none of whose options
    the answer names, keeps the hash weights.
    """

    name = "fake"
    model_id = "fake"
    device = "fake"

    def __init__(self, args):
        self.args = args
        self.script = []

    def load(self):
        time.sleep(self.args.fake_load_seconds)
        if self.args.fake_load_fail:
            raise RuntimeError("simulated load failure (--fake-load-fail)")
        if self.args.fake_no_cuda:
            raise RuntimeError(NO_CUDA)
        if self.args.fake_script:
            with open(self.args.fake_script, encoding="utf-8") as handle:
                self.script = json.load(handle)

    def score_rows(self, rows):
        results = []
        for row in rows:
            tokens = len(json.dumps(row, ensure_ascii=False)) // 4
            if tokens > MAX_TOKENS:
                error = ValueError(f"Row {row['id']}: about {tokens} input tokens exceed limit {MAX_TOKENS}")
                error.row = row["id"]
                raise error
            ids = [option["id"] for option in row["options"]]
            # exp() spreads the hash into decisive answers, like the real model's
            weights = self._scripted(row, ids) or [
                math.exp(8.0 * int(_sha256_json([row["state"], row["question"], oid])[:16], 16) / 16 ** 16)
                for oid in ids
            ]
            total = sum(weights)
            results.append({"option_ids": ids, "probabilities": [weight / total for weight in weights],
                            "input_tokens": tokens})
        return results

    def _scripted(self, row, ids):
        state = row["state"] if isinstance(row["state"], str) else json.dumps(row["state"], ensure_ascii=False)
        for rule in self.script:
            if rule.get("state", "") in state and rule.get("question", "") in row["question"]:
                answer = rule["answer"]
                named = answer if isinstance(answer, dict) else {str(answer): 1.0}
                weights = [float(named.get(oid, 0.0)) for oid in ids]
                return weights if sum(weights) > 0 else None
        return None


class Engine:
    def __init__(self, args, backend):
        self.args = args
        self.backend = backend
        self.status = "loading"
        self.error = None
        self.gpu = threading.Lock()  # one scoring call at a time
        self.cache = OrderedDict()  # sha256([sha256(state), question, debias]) -> answer
        self.cache_lock = threading.Lock()
        self.record_lock = threading.Lock()
        self.load_seconds = None
        self.requests = self.questions = self.cache_hits = self.rows_scored = 0
        self.ms_per_row = None

    # ------------------------------------------------------------------ model
    def load(self):
        started = time.perf_counter()
        try:
            self.backend.load()
            self.load_seconds = time.perf_counter() - started
            self.status = "ready"
            log(f"ready on {self.backend.device} in {self.load_seconds:.1f}s")
        except Exception as error:  # reported through /health and every POST
            self.error = f"{type(error).__name__}: {error}"
            self.status = "error"  # after the message, which HTTP threads read once they see "error"
            log("model failed to load:\n" + traceback.format_exc())

    def health(self):
        return {
            "server": SERVER_NAME,
            "version": SERVER_VERSION,
            "status": self.status,
            "error": self.error,
            "backend": self.backend.name,
            "model_id": self.backend.model_id,
            "model": MODEL_NAME,
            "source": self.args.model,
            "revision": self.args.revision,
            "device": self.backend.device,
            "debias": not self.args.no_debias,
            "load_seconds": self.load_seconds,
            "requests": self.requests,
            "questions": self.questions,
            "rows_scored": self.rows_scored,
            "cache_hits": self.cache_hits,
            "ms_per_row": None if self.ms_per_row is None else round(self.ms_per_row, 3),
            "pid": os.getpid(),
        }

    def capabilities(self):
        return {
            "server": SERVER_NAME,
            "version": SERVER_VERSION,
            "backend": self.backend.name,
            "model_id": self.backend.model_id,
            "slots": SLOTS,
            "max_options": MAX_CHOICE_OPTIONS,
            "max_levels": MAX_SCORE_LEVELS,
            "max_questions": MAX_QUESTIONS,
            "batch_rows": BATCH_ROWS,
            "debias": True,
            "question_cache": True,
        }

    def record(self, code, request, answers):
        """--record: one JSON line per /v1/systemone request whose body was read, refused ones too:
        its status, the request (the body's text when it is not JSON) and its answers (null unless 200)."""
        if not self.args.record:
            return
        try:
            line = json.dumps({"status": code, "request": request, "answers": answers})
            with self.record_lock, open(self.args.record, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except Exception:  # the reply matters more than the test log
            log("--record failed:\n" + traceback.format_exc())

    def _score_rows(self, rows):
        """Score SemIf rows that all share one state; returns results in row order."""
        with self.gpu:
            started = time.perf_counter()
            results = self.backend.score_rows(rows)
            ms = (time.perf_counter() - started) * 1000 / len(rows)
            self.ms_per_row = ms if self.ms_per_row is None else self.ms_per_row + EMA_ALPHA * (ms - self.ms_per_row)
            self.rows_scored += len(rows)
        return results

    # -------------------------------------------------------------- decisions
    def _distributions(self, state, tasks, debias):
        """tasks: {key: (instruction, [(option_id, description), ...])} with 2..16 options.

        Keys are tuples that start with the question id. Returns ({key: {option_id: probability}},
        input_tokens)."""
        rows, owners = [], []
        for key, (instruction, options) in tasks.items():
            orders = [("f", options)]
            if debias:
                orders.append(("r", list(reversed(options))))
            for tag, ordered in orders:
                owners.append((key, tag))
                rows.append({
                    "id": str(len(rows)),  # a running index: ids built from free-form question ids can collide
                    "state": state,
                    "question": instruction,
                    "options": [{"id": oid, "description": desc} for oid, desc in ordered],
                })
        try:
            scored = dict(zip(owners, self._score_rows(rows)))
        except ValueError as error:
            if getattr(error, "row", None) is not None:
                key, _tag = owners[int(error.row)]
                error.question = key[0]
            raise
        tokens = sum(result["input_tokens"] for result in scored.values())
        out = {}
        for key in tasks:
            tags = ("f", "r") if debias else ("f",)
            merged = {}
            for tag in tags:
                result = scored[(key, tag)]
                for oid, prob in zip(result["option_ids"], result["probabilities"]):
                    merged[oid] = merged.get(oid, 0.0) + prob / len(tags)
            out[key] = merged
        return out, tokens

    def _plan(self, qid, question):
        if not isinstance(question, dict):
            raise RequestError(f"question {qid!r} must be an object")
        kind = question.get("type")
        instruction = _text(question.get("instructions"), f"{qid}.instructions")
        criteria = question.get("criteria")
        if kind == "noul":
            criteria = criteria or {}
            if not isinstance(criteria, dict):
                raise RequestError(f"{qid}.criteria must be an object with 'true'/'false'")
            yes = criteria.get("true")
            no = criteria.get("false")
            options = [
                ("yes", _text(yes, f"{qid}.criteria.true") if yes is not None else "Yes."),
                ("no", _text(no, f"{qid}.criteria.false") if no is not None else "No."),
            ]
        elif kind == "choice":
            if not isinstance(criteria, dict) or not criteria:
                raise RequestError(f"{qid}.criteria must map options to descriptions")
            if len(criteria) > MAX_CHOICE_OPTIONS:
                raise RequestError(f"{qid}: at most {MAX_CHOICE_OPTIONS} options")
            options = [
                (str(oid), _text(desc, f"{qid}.criteria.{oid}") if desc is not None else str(oid))
                for oid, desc in criteria.items()
            ]
        elif kind == "score":
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= MAX_SCORE_LEVELS:
                raise RequestError(f"{qid}.criteria must list 2-{MAX_SCORE_LEVELS} ordered levels")
            options = [(str(i), _text(level, f"{qid}.criteria[{i}]")) for i, level in enumerate(criteria)]
        else:
            raise RequestError(f"{qid}.type must be 'noul', 'choice' or 'score'")
        return kind, instruction, options

    def _validate(self, request):
        if not isinstance(request, dict):
            raise RequestError("request body must be a JSON object")
        state = request.get("state")
        if isinstance(state, str):
            state = state.strip()
        if not isinstance(state, (str, dict, list)) or not state:
            raise RequestError("state must be a nonempty string, object or array")
        questions = request.get("questions")
        if not isinstance(questions, dict) or not questions:
            raise RequestError("questions must be a nonempty object")
        if len(questions) > MAX_QUESTIONS:
            raise RequestError(f"at most {MAX_QUESTIONS} questions per request, got {len(questions)}")
        opts = request.get("options") or {}
        if not isinstance(opts, dict):
            raise RequestError("options must be an object")
        return state, questions, bool(opts.get("debias", not self.args.no_debias))

    def decide(self, request):
        """Serve cached questions first and score only the rest."""
        started = time.perf_counter()
        state, questions, debias = self._validate(request)
        state_key = _sha256_json(state)  # once: the state can be megabytes, shared by 256 questions
        keys = {qid: _sha256_json([state_key, question, debias]) for qid, question in questions.items()}
        answers = {}
        with self.cache_lock:
            for qid, key in keys.items():
                if key in self.cache:
                    self.cache.move_to_end(key)
                    answers[qid] = self.cache[key]
        todo = {qid: question for qid, question in questions.items() if qid not in answers}
        tokens = 0
        if todo:
            fresh, tokens = self._decide_uncached(state, todo, debias)
            answers.update(fresh)
        with self.cache_lock:
            for qid in todo:
                self.cache[keys[qid]] = answers[qid]
                self.cache.move_to_end(keys[qid])
            while len(self.cache) > CACHE_SIZE:
                self.cache.popitem(last=False)
            self.requests += 1
            self.questions += len(questions)
            self.cache_hits += len(questions) - len(todo)
        return {
            "model": MODEL_NAME,
            "answers": {qid: answers[qid] for qid in questions},
            "usage": {"input_tokens": tokens, "output_tokens": 0},
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "debias": debias,
            "cached": not todo,
            "cached_questions": len(questions) - len(todo),
        }

    def _decide_uncached(self, state, questions, debias):
        """Returns ({qid: answer}, input_tokens)."""
        plans = {}
        for qid, question in questions.items():
            try:
                plans[qid] = self._plan(qid, question)
            except RequestError as error:
                error.question = qid
                raise
        tokens = 0
        dists = {}
        # Round 1: every question, with >16-option choices split into groups of <= 16.
        tasks, grouped = {}, {}
        for qid, (kind, instruction, options) in plans.items():
            if len(options) == 1:
                dists[qid] = {options[0][0]: 1.0}
            elif len(options) <= SLOTS:
                tasks[(qid,)] = (instruction, options)
            else:
                grouped[qid] = _groups(options)  # 2-16 groups of 8-16 options
                for index, group in enumerate(grouped[qid]):
                    tasks[(qid, index)] = (instruction, group)
        round1 = {}
        if tasks:
            round1, used = self._distributions(state, tasks, debias)
            tokens += used
        for qid in plans:
            if (qid,) in round1:
                dists[qid] = round1[(qid,)]
        # Round 2: a final between the group winners (<= 16 groups because options <= 255).
        finals = {}
        for qid, groups in grouped.items():
            lookup = dict(plans[qid][2])
            group_winners = [max(round1[(qid, i)], key=round1[(qid, i)].get) for i in range(len(groups))]
            finals[(qid,)] = (plans[qid][1], [(oid, lookup[oid]) for oid in group_winners])
        if finals:
            round2, used = self._distributions(state, finals, debias)
            tokens += used
            for qid, groups in grouped.items():
                final = round2[(qid,)]
                merged = {}
                for index in range(len(groups)):
                    probs = round1[(qid, index)]
                    group_winner = max(probs, key=probs.get)
                    for oid, prob in probs.items():  # P(item | group) * P(group winner wins)
                        merged[oid] = prob * final[group_winner]
                dists[qid] = merged

        answers = {}
        for qid, (kind, _instruction, options) in plans.items():
            probs = dists[qid]
            ordered = [probs.get(oid, 0.0) for oid, _ in options]
            if kind == "noul":
                answers[qid] = {"type": "noul", "noul": probs["yes"]}
            elif kind == "choice":
                # For tournaments this is the most probable item of the two-level distribution, so
                # the choice, its probability and the confidence always describe the same item.
                best = max(options, key=lambda option: probs.get(option[0], 0.0))[0]
                answers[qid] = {
                    "type": "choice",
                    "choice": best,
                    "probabilities": {oid: probs.get(oid, 0.0) for oid, _ in options},
                    "confidence": _confidence(ordered),
                }
            else:
                answers[qid] = {
                    "type": "score",
                    "score": sum(i * p for i, p in enumerate(ordered)),
                    "legend": {oid: desc for oid, desc in options},
                    "probabilities": {oid: probs.get(oid, 0.0) for oid, _ in options},
                    "confidence": _confidence(ordered),
                }
        return answers, tokens


ENGINE: Engine | None = None
SERVER: ThreadingHTTPServer | None = None
TOKEN = ""  # from OPENACT_TOKEN; empty means POSTs need no token


class Handler(BaseHTTPRequestHandler):
    server_version = "OpenActSemIf/" + SERVER_VERSION
    protocol_version = "HTTP/1.1"
    timeout = 30  # seconds a socket read or write may stall, so no client can hold a thread forever

    def log_message(self, fmt, *args):  # keep the log readable
        pass

    def _send(self, code, payload):
        body = json.dumps(payload).encode("utf-8")  # ASCII escapes: a lone surrogate can't break the reply
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _refused(self):
        """Why the request gets a 403, or None."""
        # Browsers add Origin to cross-site requests, and a foreign Host means DNS rebinding;
        # local clients send neither.
        if self.headers.get("Origin") is not None:
            return "requests with an Origin header are refused"
        port = self.server.server_address[1]
        if (self.headers.get("Host") or "").lower() not in (f"127.0.0.1:{port}", f"localhost:{port}"):
            return f"the Host header must be 127.0.0.1:{port} or localhost:{port}"
        return None

    def do_GET(self):
        refused = self._refused()
        if refused:
            self._send(403, {"error": refused})
        elif self.path == "/health":
            self._send(200, ENGINE.health())
        elif self.path == "/v1/capabilities":
            self._send(200, ENGINE.capabilities())
        elif self.path == "/v1/models":
            self._send(200, {"models": [{
                "name": MODEL_NAME,
                "description": f"SemIf direct option logits on {ENGINE.args.model}",
                "release_date": "2026-09-23",
            }]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if not 0 <= length <= MAX_BODY:
            self.close_connection = True  # the body stays unread and must not be taken for the next request
            code, error = (413, "request too large") if length > MAX_BODY else (400, "invalid Content-Length")
            self._send(code, {"error": error})
            return
        raw = self.rfile.read(length) if length else b""  # before any reply: an unread body resets the connection
        refused = self._refused()
        if refused is None and TOKEN and not hmac.compare_digest(
                self.headers.get("X-OpenAct-Token", "").encode(), TOKEN.encode()):
            refused = "missing or wrong X-OpenAct-Token"
        if self.path == "/v1/systemone":
            self._send(*self._systemone(raw, refused))
        elif refused:
            self._send(403, {"error": refused})
        elif self.path == "/shutdown":
            self._send(200, {"ok": True})
            threading.Thread(target=SERVER.shutdown, daemon=True).start()
        else:
            self._send(404, {"error": "not found"})

    def _systemone(self, raw, refused):
        try:
            request, readable = json.loads(raw.decode("utf-8") or "null"), True
        except (ValueError, RecursionError):  # deep nesting raises RecursionError
            request, readable = raw.decode("utf-8", "replace"), False
        status = ENGINE.status  # read once: the load may finish meanwhile
        if refused:
            code, payload = 403, {"error": refused}
        elif not readable:
            code, payload = 422, {"error": "body is not valid JSON", "question": None}
        elif status != "ready":  # never block behind a load: the client polls /health
            code, payload = 503, {"status": status}
            if status == "error":
                payload["error"] = ENGINE.error
        else:
            try:
                code, payload = 200, ENGINE.decide(request)
            except ValueError as error:  # RequestError, or SemIf refusing e.g. an over-long prompt
                # "question": the id of the question at fault, set by _decide_uncached() or _distributions()
                code, payload = 422, {"error": str(error), "question": getattr(error, "question", None)}
            except Exception as error:
                log("request failed:\n" + traceback.format_exc())
                code, payload = 500, {"error": f"{type(error).__name__}: {error}"}
        ENGINE.record(code, request, payload.get("answers"))
        return code, payload


def parent_alive(pid):
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        synchronize = 0x00100000
        handle = kernel32.OpenProcess(synchronize, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5  # access denied means it exists
        try:
            return kernel32.WaitForSingleObject(handle, 0) == 0x102  # WAIT_TIMEOUT: running
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)  # POSIX only; on Windows os.kill would terminate the process
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def watch_parent(pid):
    while True:
        time.sleep(2.0)
        if not parent_alive(pid):
            log(f"parent process {pid} is gone; exiting")
            os._exit(0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--backend", choices=("semif", "fake"), default="semif",
                        help="fake: deterministic answers without torch or a GPU, for tests")
    parser.add_argument("--model", default=DEFAULT_SOURCE)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--gpu", type=int, default=None, help="CUDA device index to use")
    parser.add_argument("--parent-pid", type=int, default=None)
    parser.add_argument("--no-debias", action="store_true", help="score each question once")
    parser.add_argument("--record", default=None, metavar="PATH.jsonl",
                        help="append every /v1/systemone request, with its answers, to this file")
    parser.add_argument("--fake-script", default=None, metavar="JSON", help="fake backend: answer rules")
    parser.add_argument("--fake-load-seconds", type=float, default=0.0, help="fake backend: how long the load takes")
    parser.add_argument("--fake-load-fail", action="store_true", help="fake backend: the load fails")
    parser.add_argument("--fake-no-cuda", action="store_true", help="fake backend: load as if PyTorch had no CUDA")
    args = parser.parse_args(argv)

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)  # SemIf wants exactly one visible GPU
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    if args.record:
        try:  # a bad path stops the start, rather than failing every request
            open(args.record, "a", encoding="utf-8").close()
        except OSError as error:
            log(f"cannot write --record {args.record}: {error}")
            return 3

    global ENGINE, SERVER, TOKEN
    TOKEN = os.environ.get("OPENACT_TOKEN", "")
    ENGINE = Engine(args, FakeBackend(args) if args.backend == "fake" else SemIfTorchBackend(args))
    try:
        SERVER = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as error:
        log(f"cannot listen on {args.host}:{args.port}: {error}")
        return 3
    SERVER.daemon_threads = True
    log(f"{SERVER_NAME} {SERVER_VERSION} listening on http://{args.host}:{args.port} (pid {os.getpid()}, "
        f"backend {ENGINE.backend.name}, token {'required' if TOKEN else 'not required'})")
    threading.Thread(target=ENGINE.load, name="load", daemon=True).start()
    if args.parent_pid:
        threading.Thread(target=watch_parent, args=(args.parent_pid,), name="watchdog", daemon=True).start()
    try:
        SERVER.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    log("server stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
