"""CSV files for OpenAct nodes: the dialect, the text/numeric column split, and the schema parsers.

Only the standard library is imported, so the unit tests also run under plain CPython. read_csv() never raises for
what a file contains: every problem is in CsvDoc.problems. The parsers raise CsvError, which carries every problem,
when the file has an error.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
from dataclasses import dataclass

ERROR, WARNING, INFO = "error", "warning", "info"

MAX_BYTES = 1024 * 1024
MAX_ROWS = 4096
MAX_COLUMNS = 64
MAX_CELL = 2000  # characters: keeps prompts inside SemIf's context
MAX_ANSWER = 200
MAX_PAYLOAD = 16
MAX_NAME = 63  # bytes of UTF-8: Blender cuts longer socket names
MAX_FACTORS = 8
MAX_CELLS = 4096
MAX_SETTINGS = 64
KEY_SEP = "\x1f"  # joins the key cells of an Overrides row

KINDS = ("yes_no", "amount", "one_of", "items")
NODES = ("", "ask", "pick", "match", "classify", "rules", "dials")  # Spec names; "" when the caller doesn't know
NAME_COLUMNS = ("answer", "name", "item", "label", "outcome")  # the first one present names the options
WHEN_COLUMNS = ("when", "description")
FACTOR_REQUIRED = ("factor", "kind", "level")
SETTING_REQUIRED = ("setting", "type", "question")
# What the socket that reads a file expects it to be. "answer_set" also covers Items lists and Overrides files;
# Factors and Settings files carry no payload, so read_csv() classifies none of their columns as numbers.
SCHEMAS = ("answer_set", "factors", "settings")
# Never payload in an answer set (design 7.3); every other numeric column is.
ANSWER_COLUMNS = frozenset(
    NAME_COLUMNS + WHEN_COLUMNS + ("if", "lock", "unsure", "confidence", "model_answer", "source")
)
# Where an answer set's numeric columns become output sockets: (node, kind) -> the node's fixed output names, which
# those sockets must not take.
PAYLOAD_OUTPUTS = {
    ("pick", "items"): frozenset(("index", "item", "confidence", "selection", "probability")),
    ("classify", "one_of"): frozenset(("label", "confidence")),
    ("rules", "one_of"): frozenset(("outcome", "probability", "yes", "amount", "confidence")),
}
# Factor names become input sockets of the Rules node, next to these.
RULES_INPUTS = frozenset(
    ("question", "context", "factors", "outcomes", "overrides", "dither", "seed", "key")
    + ("factors csv", "outcomes csv", "overrides csv")
)
FACTOR_LEVELS = {"amount": (2, 16), "category": (2, 64), "flag": (2, 2)}
SETTING_TYPES = {
    **{"yes/no": "yes_no", "yes_no": "yes_no", "bool": "yes_no", "amount": "amount"},
    **{"one-of": "one_of", "one_of": "one_of", "choice": "one_of"},
}
LOCKED = frozenset(("1", "yes", "true", "x"))

_INT = re.compile(r"[+-]?[0-9]+")
_FLOAT = re.compile(r"[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?")
# Numbers as decimal-comma locales write them: "1,5", or with digit groups such as "1 000" or "1.234,5".
_LOCALE_NUMBER = re.compile("[+-]?[0-9]*,[0-9]+|[+-]?[0-9]{1,3}([ .,'\u00a0\u202f][0-9]{3})+([.,][0-9]+)?")
# Cells that make a column text although it was meant to hold numbers.
_NUMBER_LIKE = re.compile(f"{_LOCALE_NUMBER.pattern}|[+-]?(inf|infinity|nan)|true|false", re.IGNORECASE)


@dataclass(frozen=True)
class Problem:
    severity: str  # ERROR | WARNING | INFO
    file: str  # the document's path
    line: int  # physical line; 0 for the whole file
    column: str  # header name; "" when not about one column
    message: str


@dataclass(frozen=True)
class CsvDoc:
    path: str  # absolute and normalised (os.path.abspath)
    stamp: tuple[int, int]  # (st_mtime_ns, st_size) when read; (0, 0) if it could not be read
    header: tuple[str, ...]  # trimmed, as written
    rows: tuple[tuple[str, ...], ...]  # data rows, trimmed, padded to header width
    lines: tuple[int, ...]  # physical line number of each data row (for messages)
    text_cols: tuple[str, ...]  # reserved + non-numeric columns
    num_cols: dict[str, str]  # payload column -> "INT" | "FLOAT" (header order)
    numbers: dict[str, tuple[float | int, ...]]
    text_hash: str  # sha1: header order, row count, normalised text columns
    num_hash: str  # sha1: num_cols + numbers
    problems: tuple[Problem, ...]
    schema: str  # the SCHEMAS entry the columns were classified for; each parser insists on its own


@dataclass(frozen=True)
class Option:
    name: str
    when: str
    if_expr: str  # as written; nodes other than Rules drop it
    payload: dict[str, int | float]  # this row's numeric columns, where the node bakes them


@dataclass(frozen=True)
class AnswerSet:
    options: tuple[Option, ...]
    doc: CsvDoc | None  # None for an inline set
    problems: tuple[Problem, ...] = ()  # warnings and info only: the set is usable


@dataclass(frozen=True)
class Level:
    name: str
    when: str
    value: int | None  # category and flag levels
    band: tuple[float, float] | None  # (from, to) when the amount factor has custom edges; None for equal widths
    line: int


@dataclass(frozen=True)
class Factor:
    name: str
    kind: str  # "amount" | "category" | "flag"
    levels: tuple[Level, ...]
    line: int  # of its first row


@dataclass(frozen=True)
class Setting:
    name: str
    type: str  # "yes_no" | "amount" | "one_of"
    question: str
    answers: tuple[tuple[str, str], ...]  # the inline answer set as (name, when) pairs; () when not given
    min: float
    max: float
    line: int


@dataclass(frozen=True)
class Lock:
    answer: str  # as written: whether it names a valid answer depends on the node
    line: int


class CsvError(Exception):
    """A file with errors: the node keeps its last good table. problems holds every problem found in the file."""

    def __init__(self, problems):
        self.problems = tuple(problems)
        super().__init__(next((p.message for p in self.problems if p.severity == ERROR), ""))


class _Problems(list):
    def __init__(self, path, problems=()):
        super().__init__(problems)
        self.path = path

    def add(self, severity, line, column, message):
        self.append(Problem(severity, self.path, line, column, message))

    def check(self):
        if any(p.severity == ERROR for p in self):
            raise CsvError(self)


def read_csv(path: str, delimiter: str | None = None, schema: str = "answer_set") -> CsvDoc:
    """Reads a CSV file and sorts its columns into text and numeric payload.

    delimiter: an Import CSV node's Delimiter. None sniffs it: "," unless the header line has none but has ";" (as
    Excel writes in decimal-comma locales) or a tab. schema: what the reading socket expects (SCHEMAS): "factors"
    for Rules' Factors, "settings" for Dials' Settings, else "answer_set".
    """
    if schema not in SCHEMAS:
        raise ValueError(f"unknown schema {schema!r}")
    path = os.path.abspath(path)
    name = os.path.basename(path)
    problems = _Problems(path)
    stamp, data, text = (0, 0), b"", ""
    try:
        with open(path, "rb") as handle:
            status = os.fstat(handle.fileno())  # before reading: an edit during the read changes the stamp again
            data = handle.read(MAX_BYTES + 1)
        stamp = (status.st_mtime_ns, status.st_size)
    except FileNotFoundError:
        problems.add(ERROR, 0, "", f"Can't find {path}")
    except OSError as error:
        problems.add(ERROR, 0, "", f"Can't read {path}: {error.strerror or error}")
    if len(data) > MAX_BYTES:
        problems.add(ERROR, 0, "", f"{name} has {_n(max(len(data), stamp[1]))} bytes (limit {_n(MAX_BYTES)})")
        data = b""
    bom = 3 if data.startswith(b"\xef\xbb\xbf") else 0
    bad = data.find(b"\0")  # UTF-16 or binary data
    try:
        text = data[bom:].decode("utf-8")
    except UnicodeDecodeError as error:
        bad = min(at for at in (bad, bom + error.start) if at >= 0)
    if bad >= 0:
        problems.add(ERROR, 0, "", f"{name} is not UTF-8 text (byte {_n(bad)})")
        text = ""
    if delimiter is None:
        first = next((line for line in io.StringIO(text, newline="") if line.strip()), "")
        delimiter = next((sign for sign in (",", ";", "\t") if sign in first), ",")
    try:
        if len(delimiter) != 1 or delimiter in '"\r\n':  # Python 3.13 rejects these, earlier ones misparse them
            raise ValueError(delimiter)
        # Skipping the spaces after a delimiter lets a quote follow them; a space delimiter keeps its empty cells.
        reader = csv.reader(
            io.StringIO(text, newline=""), delimiter=delimiter, skipinitialspace=delimiter != " ", strict=True
        )
    except (TypeError, ValueError):  # also whatever a later Python's csv refuses
        message = f"The delimiter must be one character other than a quote or a line break, not {delimiter!r}"
        problems.add(ERROR, 0, "", message)
        reader = ()

    header, rows, lines = None, [], []
    end = width = counted = blank = 0
    wrapped = gap = False
    try:
        for cells in reader:
            start, end = end + 1, reader.line_num
            if len(rows) == MAX_ROWS:  # past the limit, rows are only counted
                counted += bool("".join(cells).strip())
                continue
            if end > start and not wrapped:  # Import CSV may not read it the same way
                problems.add(WARNING, start, "", f"Line {start}: a quoted cell has a line break")
                wrapped = True
            cells = [cell.replace("\r\n", "\n").replace("\r", "\n").strip() for cell in cells]
            if not any(cells):
                blank = blank or start
                continue
            if header is None:
                header, width, blank = cells, len(cells), 0
                if width > MAX_COLUMNS:
                    problems.add(ERROR, start, "", f"{name} has {_n(width)} columns (limit {MAX_COLUMNS})")
                    header, width = [], 0
                    break
                seen = set()
                for i, column in enumerate(header):
                    if not column:
                        problems.add(ERROR, start, "", f"Column {i + 1} has no name")
                    elif column.casefold() in seen:
                        problems.add(ERROR, start, column, f"Two columns are named '{column}'")
                    seen.add(column.casefold())
                continue
            if blank and rows and not gap:  # messages count physical lines, Import CSV may not
                problems.add(WARNING, blank, "", f"Line {blank} is blank: blank lines between rows are skipped")
                gap = True
            blank = 0
            if len(cells) > width:
                problems.add(ERROR, start, "", f"Line {start} has {len(cells)} cells; the header has {width}")
                del cells[width:]
            cells += [""] * (width - len(cells))
            for column, cell in zip(header, cells):
                if len(cell) > MAX_CELL:
                    size = f"{_n(len(cell))} characters (limit {_n(MAX_CELL)})"
                    problems.add(ERROR, start, column, f"Line {start}: column '{column}' has {size}")
            rows.append(tuple(cells))
            lines.append(start)
    except csv.Error as error:
        if "field limit" in str(error):  # the csv module's own cap on a cell, usually hit by a quote never closed
            size = f"over {_n(csv.field_size_limit())} characters (limit {_n(MAX_CELL)})"
            problems.add(ERROR, end + 1, "", f"Line {end + 1}: a cell has {size}, or a quote is not closed")
        else:
            problems.add(ERROR, end + 1, "", f"Line {end + 1}: bad quoting ({error})")
    header = header or []
    if counted:
        problems.add(ERROR, 0, "", f"{name} has {_n(MAX_ROWS + counted)} rows (limit {_n(MAX_ROWS)})")

    texts, num_cols, numbers = {}, {}, {}  # texts: text column -> its cells
    for i, column in enumerate(header):
        cells = [row[i] for row in rows]
        free = schema == "answer_set" and column.casefold() not in ANSWER_COLUMNS
        kind, values = _classify(cells) if free else (None, ())
        if kind:
            num_cols[column], numbers[column] = kind, values
            continue
        texts[column] = cells
        if not free:
            continue
        odd = [(line, cell) for line, cell in zip(lines, cells) if not _FLOAT.fullmatch(cell)]
        meant = any(_FLOAT.fullmatch(cell) or _LOCALE_NUMBER.fullmatch(cell) for cell in cells)
        if meant and all(not cell or _NUMBER_LIKE.fullmatch(cell) for _, cell in odd):
            first = f"'{odd[0][1]}'" if odd[0][1] else "an empty cell"
            where = ", ".join(str(line) for line, _ in odd[:5]) + (", ..." if len(odd) > 5 else "")
            if len(odd) == 1:
                what = f"{first} in line {where}"
            elif len({cell for _, cell in odd}) == 1:
                what = f"{first if odd[0][1] else 'empty cells'} in lines {where}"
            else:
                what = f"cells that don't read as numbers in lines {where} (first: {first})"
            problems.add(WARNING, odd[0][0], column, f"Column '{column}' has {what}; treated as text, no output")

    return CsvDoc(
        path=path,
        stamp=stamp,
        header=tuple(header),
        rows=tuple(rows),
        lines=tuple(lines),
        text_cols=tuple(texts),
        num_cols=num_cols,
        numbers=numbers,
        text_hash=_sha1([header, len(rows), texts]),
        num_hash=_sha1([num_cols, numbers]),
        problems=tuple(problems),
        schema=schema,
    )


def parse_inline(text: str) -> list[tuple[str, str]]:
    """An inline list or answer set as (name, when) pairs.

    Entries are split on new lines if there are any, else on ";" if present, else on ",". New lines at either end do
    not count, so the trailing one of an Import Text file changes nothing. Within an entry, the first ": " separates
    the name from its when.
    """
    if len(text.strip().splitlines()) > 1:
        entries = text.splitlines()
    else:
        entries = text.split(";" if ";" in text else ",")
    pairs = []
    for entry in entries:
        if entry.strip():
            name, _, when = entry.partition(": ")
            pairs.append((name.strip(), when.strip()))
    return pairs


def parse_answer_set(doc: CsvDoc, kind: str, node: str = "") -> AnswerSet:
    """The options of an answer-set CSV (Answers, Labels, Outcomes or Items). Row r is option r.

    kind: "yes_no" | "amount" | "one_of" | "items". node: the Spec name of the node that reads the set (one of NODES,
    lower case), or "" when not known. A known node other than Rules drops 'if', with a warning. Numeric columns are
    payload only where the node bakes them (Pick items, Classify labels, Rules one-of outcomes; for an unknown node,
    wherever one of those could), and there they must not take one of the node's fixed output names. Raises CsvError
    when the file has an error.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown answer-set kind {kind!r}")
    if node not in NODES:
        raise ValueError(f"unknown node {node!r}")
    _expect(doc, "answer_set")
    problems = _Problems(doc.path, doc.problems)
    problems.check()
    name_at = _find(doc.header, NAME_COLUMNS)
    if name_at is None:
        problems.add(ERROR, 0, "", "No 'answer' column (or name, item, label, outcome)")
        problems.check()
    when_at, if_at = _find(doc.header, WHEN_COLUMNS), _find(doc.header, ("if",))
    keep_if = node in ("", "rules")
    if if_at is not None and not keep_if:
        used = [line for row, line in zip(doc.rows, doc.lines) if row[if_at]]
        if used:
            problems.add(WARNING, used[0], doc.header[if_at], "'if' is only used by Rules; ignored")
    outputs = [names for (baker, baked), names in PAYLOAD_OUTPUTS.items() if baked == kind and node in ("", baker)]
    payload = doc.num_cols if outputs else {}
    fixed = frozenset().union(*outputs)
    for column in payload:
        size = len(column.encode("utf-8"))
        if column.casefold() in fixed:
            problems.add(ERROR, 0, column, f"Rename column '{column}': it clashes with a node output")
        elif size > MAX_NAME:
            problems.add(ERROR, 0, column, f"Rename column '{column}': it has {size} bytes in UTF-8 (limit {MAX_NAME})")
    if len(payload) > MAX_PAYLOAD:
        message = f"{os.path.basename(doc.path)} has {len(payload)} payload columns (limit {MAX_PAYLOAD})"
        problems.add(ERROR, 0, "", message)
    for i, column in enumerate(doc.header):
        if i not in (name_at, when_at, if_at) and column not in doc.num_cols:
            if column.casefold() in ANSWER_COLUMNS:
                problems.add(INFO, 0, column, f"Column '{column}' is not used: '{column}' is a reserved name")
            else:
                problems.add(INFO, 0, column, f"Column '{column}' is text and is not used")
    options, seen = [], set()
    for r, (row, line) in enumerate(zip(doc.rows, doc.lines)):
        message = _name_problem(row[name_at], seen)
        if message:
            problems.add(ERROR, line, doc.header[name_at], f"Line {line}: {message}")
        when = row[when_at] if when_at is not None else ""
        if_expr = row[if_at] if keep_if and if_at is not None else ""
        options.append(Option(row[name_at], when, if_expr, {column: doc.numbers[column][r] for column in payload}))
    message = _count_problem(kind, [option.name for option in options])
    if message:
        problems.add(ERROR, 0, "", message)
    problems.check()
    return AnswerSet(tuple(options), doc, tuple(problems))


def parse_factors(doc: CsvDoc) -> list[Factor]:
    """The factors of a Factors CSV, in the order of their first rows, each with its levels in row order.

    Raises CsvError when the file has an error.
    """
    _expect(doc, "factors")
    problems = _Problems(doc.path, doc.problems)
    problems.check()
    at = {column: _find(doc.header, (column,)) for column in FACTOR_REQUIRED + ("value", "from", "to")}
    at["when"] = _find(doc.header, WHEN_COLUMNS)
    for column in FACTOR_REQUIRED:
        if at[column] is None:
            problems.add(ERROR, 0, "", f"No '{column}' column")
    problems.check()

    groups, last = {}, None  # casefolded factor name -> [(cells, line)], in the order of first rows
    for row, line in zip(doc.rows, doc.lines):
        cells = {column: row[i] if i is not None else "" for column, i in at.items()}
        key = cells["factor"].casefold()
        if not key:
            problems.add(ERROR, line, "", f"Line {line}: the factor name is empty")
        elif key in groups and key != last:
            problems.add(ERROR, line, "", f"Line {line}: the rows of factor '{cells['factor']}' must be together")
        else:
            groups.setdefault(key, []).append((cells, line))
            last = key

    factors = []
    for rows in groups.values():
        first, line = rows[0]
        name, kind = first["factor"], first["kind"].casefold()
        size = len(name.encode("utf-8"))
        if size > MAX_NAME:
            problems.add(ERROR, line, "", f"Line {line}: the factor name has {size} bytes in UTF-8 (limit {MAX_NAME})")
        if name.casefold() in RULES_INPUTS:
            problems.add(ERROR, line, "", f"Line {line}: rename factor '{name}': it clashes with a node input")
        if kind not in FACTOR_LEVELS:
            problems.add(ERROR, line, "", f"Line {line}: factor '{name}' needs a kind: amount, category or flag")
            continue
        names, values = set(), []
        for cells, row_line in rows:
            if cells["kind"] and cells["kind"].casefold() != kind:
                message = f"Line {row_line}: kind '{cells['kind']}' differs from '{kind}' on line {line}"
                problems.add(ERROR, row_line, "", message)
            level = cells["level"]
            if not level:
                problems.add(ERROR, row_line, "", f"Line {row_line}: the level name is empty")
            elif level.casefold() in names:
                problems.add(ERROR, row_line, "", f"Line {row_line}: level '{level}' appears twice in factor '{name}'")
            names.add(level.casefold())
            value = None if kind == "amount" else _int32(cells["value"])
            if kind != "amount" and value is None:
                problems.add(ERROR, row_line, "", f"Line {row_line}: level '{level}' needs an integer 'value'")
            elif value is not None and value in values:
                problems.add(ERROR, row_line, "", f"Line {row_line}: value {value} appears twice in factor '{name}'")
            values.append(value)

        low, high = FACTOR_LEVELS[kind]
        if not low <= len(rows) <= high:
            what = "a flag has exactly 2 levels" if kind == "flag" else f"{kind} factors have {low} to {high} levels"
            problems.add(ERROR, line, "", f"{name}: {what}, not {len(rows)}")
        complete = None not in values and len(set(values)) == len(values)
        if kind == "flag" and complete and len(values) == 2 and sorted(values) != [0, 1]:
            problems.add(ERROR, line, "", f"{name}: a flag's values must be 0 and 1")
        if kind == "category" and complete:
            consecutive = all(value == values[0] + i for i, value in enumerate(values))
            if not consecutive and not all(0 <= value <= 4095 for value in values):
                message = f"{name}: category values must count up by one, or all lie between 0 and 4095"
                problems.add(ERROR, line, "", message)
        bands = _bands(name, rows, problems) if kind == "amount" else [None] * len(rows)
        levels = [Level(c["level"], c["when"], v, b, n) for (c, n), v, b in zip(rows, values, bands)]
        factors.append(Factor(name, kind, tuple(levels), line))

    if not 1 <= len(groups) <= MAX_FACTORS:
        problems.add(ERROR, 0, "", f"Rules need 1 to {MAX_FACTORS} factors; this file has {len(groups)}")
    count = math.prod(len(rows) for rows in groups.values())
    if count > MAX_CELLS:
        problems.add(ERROR, 0, "", f"Rules has {_n(count)} cells (limit {_n(MAX_CELLS)}): use fewer bands")
    problems.check()
    return factors


def _bands(name, rows, problems):
    """An amount factor's (from, to) per level, or Nones for equal-width bands."""
    edges = [(cells["from"], cells["to"]) for cells, _ in rows]
    if not any(a or b for a, b in edges):
        return [None] * len(rows)
    if not all(a and b for a, b in edges):
        problems.add(ERROR, rows[0][1], "", f"{name}: give 'from' and 'to' on every level, or on none")
        return [None] * len(rows)
    bands = []
    for (cells, line), raw in zip(rows, edges):
        band = (_number(raw[0]), _number(raw[1]))
        for column, text, value in zip(("from", "to"), raw, band):
            if value is None:
                message = f"Line {line}: '{column}' of {cells['level']} must be a number, not '{text}'"
                problems.add(ERROR, line, column, message)
        bands.append(band)
    if any(None in band for band in bands):
        return [None] * len(rows)
    level = [cells["level"] for cells, _ in rows]
    if bands[0][0] != 0:
        problems.add(ERROR, rows[0][1], "from", f"{name}: 'from' of {level[0]} ({edges[0][0]}) must be 0")
    if bands[-1][1] != 1:
        problems.add(ERROR, rows[-1][1], "to", f"{name}: 'to' of {level[-1]} ({edges[-1][1]}) must be 1")
    for b in range(len(bands) - 1):
        if bands[b][1] != bands[b + 1][0]:
            upper, lower = f"'to' of {level[b]} ({edges[b][1]})", f"'from' of {level[b + 1]} ({edges[b + 1][0]})"
            problems.add(ERROR, rows[b + 1][1], "from", f"{name}: {upper} must equal {lower}")
    for b, (low, high) in enumerate(bands):
        if not low < high:
            message = f"{name}: 'from' of {level[b]} ({edges[b][0]}) must be below its 'to' ({edges[b][1]})"
            problems.add(ERROR, rows[b][1], "", message)
    return bands


def parse_settings(doc: CsvDoc) -> list[Setting]:
    """The settings of a Settings CSV, in row order. Raises CsvError when the file has an error."""
    _expect(doc, "settings")
    problems = _Problems(doc.path, doc.problems)
    problems.check()
    at = {column: _find(doc.header, (column,)) for column in SETTING_REQUIRED + ("answers", "min", "max")}
    for column in SETTING_REQUIRED:
        if at[column] is None:
            problems.add(ERROR, 0, "", f"No '{column}' column")
    problems.check()
    settings, seen = [], set()
    for row, line in zip(doc.rows, doc.lines):
        cells = {column: row[i] if i is not None else "" for column, i in at.items()}
        name, kind = cells["setting"], SETTING_TYPES.get(cells["type"].casefold())
        size = len(name.encode("utf-8"))
        if not name:
            problems.add(ERROR, line, "", f"Line {line}: the setting name is empty")
        elif size > MAX_NAME:
            message = f"Line {line}: the setting name has {size} bytes in UTF-8 (limit {MAX_NAME})"
            problems.add(ERROR, line, "", message)
        elif name.casefold() == "confidence":
            problems.add(ERROR, line, "", f"Line {line}: rename setting '{name}': it clashes with a node output")
        elif name.casefold() in seen:
            problems.add(ERROR, line, "", f"Line {line}: '{name}' appears twice")
        seen.add(name.casefold())
        if kind is None:
            problems.add(ERROR, line, "", f"Line {line}: type '{cells['type']}' is not yes/no, amount or one-of")
        if not cells["question"]:
            problems.add(ERROR, line, "", f"Line {line}: the question is empty")
        answers = tuple(parse_inline(cells["answers"]))
        if kind == "one_of" and not answers:
            problems.add(ERROR, line, "", f"Line {line}: a one-of setting needs answers")
        elif answers and kind:
            names = set()
            messages = [_name_problem(option, names) for option, _ in answers]
            messages.append(_count_problem(kind, [option for option, _ in answers], least=2))
            for message in filter(None, messages):
                problems.add(ERROR, line, "", f"Line {line}, answers: {message}")
        bounds = [0.0, 1.0]
        for i, column in enumerate(("min", "max")):
            if kind == "amount" and cells[column]:
                value = _number(cells[column])
                if value is None:
                    message = f"Line {line}: '{column}' must be a number, not '{cells[column]}'"
                    problems.add(ERROR, line, column, message)
                else:
                    bounds[i] = value
        settings.append(Setting(name, kind or "", cells["question"], answers, bounds[0], bounds[1], line))
    if not settings:
        problems.add(ERROR, 0, "", "No settings")
    elif len(settings) > MAX_SETTINGS:
        message = f"{os.path.basename(doc.path)} has {len(settings)} settings (limit {MAX_SETTINGS})"
        problems.add(ERROR, 0, "", message)
    problems.check()
    return settings


def parse_overrides(doc: CsvDoc, keys: list[str]) -> dict[str, Lock]:
    """The locked rows of an Overrides CSV, by key: the row's cells in the key columns, casefolded and joined with
    KEY_SEP.

    keys: the node's key columns, e.g. ["question"] for Ask, or one per factor for Rules. Unlocked rows and the
    informational columns are ignored. Matching locks to table rows and checking their answers is the caller's.
    Factor key columns may hold numbers, so an edit to them can change only num_hash: re-parse when either hash
    changes. Raises CsvError when a key column is missing or two locked rows share a key.
    """
    _expect(doc, "answer_set")
    problems = _Problems(doc.path, doc.problems)
    problems.check()
    at = [_find(doc.header, (key.casefold(),)) for key in keys]
    for key, i in zip(keys, at):
        if i is None:
            problems.add(ERROR, 0, "", f"No '{key}' column")
    problems.check()
    lock_at, answer_at = _find(doc.header, ("lock",)), _find(doc.header, ("answer",))
    locks = {}
    for row, line in zip(doc.rows, doc.lines):
        if lock_at is None or row[lock_at].casefold() not in LOCKED:
            continue
        key = KEY_SEP.join(row[i].casefold() for i in at)
        if key in locks:
            shown = ", ".join(row[i] for i in at)
            problems.add(ERROR, line, "", f"Line {line}: '{shown}' is locked twice (also line {locks[key].line})")
            continue
        locks[key] = Lock(row[answer_at] if answer_at is not None else "", line)
    problems.check()
    return locks


def _n(number):
    """4096 -> "4 096", as the messages write numbers."""
    return f"{number:,}".replace(",", " ")


def _sha1(parts):
    return hashlib.sha1(json.dumps(parts, ensure_ascii=False).encode("utf-8")).hexdigest()


def _expect(doc, schema):
    """A document classified for another schema would give wrong payload or hashes: a programming error."""
    if doc.schema != schema:
        raise ValueError(f"{os.path.basename(doc.path)} was read as {doc.schema!r}, not {schema!r}")


def _find(header, names):
    """Index of the first of names (most preferred first) among the header's, ignoring case; None if absent."""
    folded = [column.casefold() for column in header]
    return next((folded.index(name) for name in names if name in folded), None)


def _int32(cell):
    """The cell's value if it is an integer that fits an Int socket, else None."""
    if _INT.fullmatch(cell) and len(cell.lstrip("+-").lstrip("0")) <= 10:  # int() refuses over 4300 digits
        value = int(cell)
        if -(2**31) <= value < 2**31:
            return value
    return None


def _number(cell):
    return float(cell) if _FLOAT.fullmatch(cell) else None


def _classify(cells):
    """("INT" | "FLOAT", values) when every cell is a number, else (None, ())."""
    ints = [_int32(cell) for cell in cells]
    if cells and None not in ints:
        return "INT", tuple(ints)
    floats = [_number(cell) for cell in cells]
    if cells and None not in floats:
        return "FLOAT", tuple(floats)
    return None, ()


def _name_problem(name, seen):
    """What is wrong with an option's name, or ""; seen collects the names so far, ignoring case."""
    if not name:
        return "the name is empty"
    if len(name) > MAX_ANSWER:
        return f"the name has {_n(len(name))} characters (limit {MAX_ANSWER})"
    if name.casefold() in seen:
        return f"'{name}' appears twice"
    seen.add(name.casefold())
    return ""


def _count_problem(kind, names, least=1):
    """What is wrong with the rows of an answer set of this kind, or ""."""
    count = len(names)
    if kind == "yes_no":
        if sorted(name.casefold() for name in names) != ["no", "yes"]:
            return "A Yes/No answer set needs rows 'yes' and 'no'"
    elif kind == "amount":
        if not 2 <= count <= 16:
            return f"An Amount answer set needs 2 to 16 levels; this one has {count}"
    elif not least <= count <= 255:
        what, noun = ("An item list", "items") if kind == "items" else ("A One-of answer set", "answers")
        return f"{what} needs {least} to 255 {noun}; this one has {count}"
    return ""
