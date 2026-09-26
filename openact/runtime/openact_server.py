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
* An LRU answer cache and a watchdog that exits when the parent process (Blender) is gone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import threading
import time
import traceback
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SERVER_NAME = "openact-semif"
SERVER_VERSION = "0.1.0"
DEFAULT_SOURCE = "Qwen/Qwen3.5-4B"
DEFAULT_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
MODEL_NAME = "semif-qwen3.5-4b"
SLOTS = 16  # SemIf answer letters A..P
MAX_CHOICE_OPTIONS = 255
MAX_SCORE_LEVELS = 16
BATCH_ROWS = 32
CACHE_SIZE = 4096
MAX_BODY = 4 * 1024 * 1024


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


class Engine:
    def __init__(self, args):
        self.args = args
        self.status = "loading"
        self.error = None
        self.device_name = None
        self.model = self.tokenizer = self.metadata = None
        self.loaded = threading.Event()
        self.gpu = threading.Lock()
        self.cache = OrderedDict()
        self.cache_lock = threading.Lock()
        self.requests = 0
        self.load_seconds = None

    # ------------------------------------------------------------------ model
    def load(self):
        started = time.perf_counter()
        try:
            import torch
            from semif_phase1.core import load_causal_model

            device = self.args.device
            dtype = self.args.dtype
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu" and dtype == "bfloat16":
                dtype = "float32"  # SemIf's documented CPU reference path
            log(f"loading {self.args.model}@{self.args.revision[:12]} on {device} ({dtype})")
            self.model, self.tokenizer, self.metadata = load_causal_model(
                self.args.model, self.args.revision, device=device, dtype=dtype
            )
            if self.metadata["device"].startswith("cuda"):
                self.device_name = torch.cuda.get_device_name(0)
            else:
                self.device_name = self.metadata["device"]
            # Warm up kernels so the first real request is not the slow one.
            self._score_rows([
                {"id": "warmup", "state": "Warm-up.", "question": "Is this a warm-up?",
                 "options": [{"id": "yes", "description": "Yes."}, {"id": "no", "description": "No."}]}
            ])
            self.load_seconds = time.perf_counter() - started
            self.status = "ready"
            log(f"ready on {self.device_name} in {self.load_seconds:.1f}s")
        except Exception as error:  # reported through /health
            self.status = "error"
            self.error = f"{type(error).__name__}: {error}"
            log("model failed to load:\n" + traceback.format_exc())
        finally:
            self.loaded.set()

    def health(self):
        return {
            "server": SERVER_NAME,
            "version": SERVER_VERSION,
            "status": self.status,
            "error": self.error,
            "model": MODEL_NAME,
            "source": self.args.model,
            "revision": self.args.revision,
            "device": self.device_name,
            "debias": not self.args.no_debias,
            "load_seconds": self.load_seconds,
            "requests": self.requests,
            "pid": os.getpid(),
        }

    def _score_rows(self, rows):
        """Score SemIf rows that all share one state; returns results in row order."""
        from semif_phase1.direct import score
        from semif_phase1.shared import score_shared

        results = []
        with self.gpu:
            for start in range(0, len(rows), BATCH_ROWS):
                chunk = rows[start:start + BATCH_ROWS]
                if len(chunk) == 1:
                    results.append(score(self.model, self.tokenizer, chunk[0], self.metadata))
                else:
                    scored, _timing = score_shared(self.model, self.tokenizer, chunk, self.metadata)
                    results.extend(scored)
        return results

    # -------------------------------------------------------------- decisions
    def _distributions(self, state, tasks, debias):
        """tasks: {key: (instruction, [(option_id, description), ...])} with 2..16 options.

        Returns ({key: {option_id: probability}}, input_tokens)."""
        rows = []
        for key, (instruction, options) in tasks.items():
            orders = [("f", options)]
            if debias:
                orders.append(("r", list(reversed(options))))
            for tag, ordered in orders:
                rows.append({
                    "id": f"{key}|{tag}",
                    "state": state,
                    "question": instruction,
                    "options": [{"id": oid, "description": desc} for oid, desc in ordered],
                })
        scored = {row["id"]: result for row, result in zip(rows, self._score_rows(rows))}
        tokens = sum(result["input_tokens"] for result in scored.values())
        out = {}
        for key in tasks:
            tags = ("f", "r") if debias else ("f",)
            merged = {}
            for tag in tags:
                result = scored[f"{key}|{tag}"]
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

    def decide(self, request):
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
        opts = request.get("options") or {}
        debias = bool(opts.get("debias", not self.args.no_debias))

        cache_key = hashlib.sha256(
            json.dumps([state, questions, debias], sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        with self.cache_lock:
            if cache_key in self.cache:
                self.cache.move_to_end(cache_key)
                return dict(self.cache[cache_key], cached=True)

        started = time.perf_counter()
        plans = {qid: self._plan(qid, question) for qid, question in questions.items()}
        tokens = 0
        dists = {}
        # Round 1: every question, with >16-option choices split into groups of <= 16.
        tasks, grouped = {}, {}
        for qid, (kind, instruction, options) in plans.items():
            if len(options) == 1:
                dists[qid] = {options[0][0]: 1.0}
            elif len(options) <= SLOTS:
                tasks[qid] = (instruction, options)
            else:
                grouped[qid] = _groups(options)  # 2-16 groups of 8-16 options
                for index, group in enumerate(grouped[qid]):
                    tasks[f"{qid}#g{index}"] = (instruction, group)
        round1 = {}
        if tasks:
            round1, used = self._distributions(state, tasks, debias)
            tokens += used
        for qid in plans:
            if qid in round1:
                dists[qid] = round1[qid]
        # Round 2: a final between the group winners (<= 16 groups because options <= 255).
        finals = {}
        for qid, groups in grouped.items():
            lookup = dict(plans[qid][2])
            group_winners = [max(round1[f"{qid}#g{i}"], key=round1[f"{qid}#g{i}"].get) for i in range(len(groups))]
            finals[f"{qid}#final"] = (plans[qid][1], [(oid, lookup[oid]) for oid in group_winners])
        if finals:
            round2, used = self._distributions(state, finals, debias)
            tokens += used
            for qid, groups in grouped.items():
                final = round2[f"{qid}#final"]
                merged = {}
                for index in range(len(groups)):
                    probs = round1[f"{qid}#g{index}"]
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
        response = {
            "model": MODEL_NAME,
            "answers": answers,
            "usage": {"input_tokens": tokens, "output_tokens": 0},
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "debias": debias,
        }
        with self.cache_lock:
            self.cache[cache_key] = response
            while len(self.cache) > CACHE_SIZE:
                self.cache.popitem(last=False)
        self.requests += 1
        return dict(response, cached=False)


ENGINE: Engine | None = None
SERVER: ThreadingHTTPServer | None = None


class Handler(BaseHTTPRequestHandler):
    server_version = "OpenActSemIf/" + SERVER_VERSION
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # keep the log readable
        pass

    def _send(self, code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, ENGINE.health())
        elif self.path == "/v1/models":
            self._send(200, {"models": [{
                "name": MODEL_NAME,
                "description": f"SemIf direct option logits on {ENGINE.args.model}",
                "release_date": "2026-09-23",
            }]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            self._send(413, {"error": "request too large"})
            return
        raw = self.rfile.read(length) if length else b""
        if self.path == "/shutdown":
            self._send(200, {"ok": True})
            threading.Thread(target=SERVER.shutdown, daemon=True).start()
            return
        if self.path != "/v1/systemone":
            self._send(404, {"error": "not found"})
            return
        try:
            request = json.loads(raw.decode("utf-8") or "null")
        except ValueError:
            self._send(422, {"error": "body is not valid JSON"})
            return
        if not ENGINE.loaded.wait(timeout=300) or ENGINE.status != "ready":
            self._send(503, {"error": ENGINE.error or "model is still loading", "status": ENGINE.status})
            return
        try:
            self._send(200, ENGINE.decide(request))
        except ValueError as error:  # RequestError, or SemIf rejecting e.g. an over-long prompt
            self._send(422, {"error": str(error)})
        except Exception as error:
            log("request failed:\n" + traceback.format_exc())
            self._send(500, {"error": f"{type(error).__name__}: {error}"})


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
    parser.add_argument("--model", default=DEFAULT_SOURCE)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--gpu", type=int, default=None, help="CUDA device index to use")
    parser.add_argument("--parent-pid", type=int, default=None)
    parser.add_argument("--no-debias", action="store_true", help="score each question once")
    args = parser.parse_args(argv)

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)  # SemIf wants exactly one visible GPU
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    global ENGINE, SERVER
    ENGINE = Engine(args)
    try:
        SERVER = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as error:
        log(f"cannot listen on {args.host}:{args.port}: {error}")
        return 3
    SERVER.daemon_threads = True
    log(f"{SERVER_NAME} {SERVER_VERSION} listening on http://{args.host}:{args.port} (pid {os.getpid()})")
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
