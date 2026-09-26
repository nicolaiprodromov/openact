"""csvfmt and lists, with no Blender and no model.

Checks the CSV dialect (BOM, CRLF and CR, ";" and tab delimiters, quoting), the INT/FLOAT/text split, the limits,
every design 7.9 message that the reader and the parsers give (one broken fixture each), the Answers, Factors,
Settings and Overrides parsers, the Rules `if` grammar (rules_core.parse_if), the change hashes, the inline-list
rules, and the cells Dials asks.

Run:  python tests/csv_test.py
      blender -b --factory-startup --python tests/csv_test.py
"""

import os
import shutil
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import ROOT, check, finish  # noqa: E402

# openact/__init__.py imports bpy, so the pure modules are loaded without running it.
_package = types.ModuleType("openact")
_package.__path__ = [os.path.join(ROOT, "openact")]
sys.modules["openact"] = _package
from openact import csvfmt, lists, pipeline, rules_core, sources  # noqa: E402
from openact.specs.base import NO_CONTEXT  # noqa: E402

FIXTURES = os.path.join(ROOT, "tests", "fixtures")
TMP = tempfile.mkdtemp(prefix="openact_csv_")
SEP = csvfmt.KEY_SEP
FACTORS = "factor,kind,level,value,when,from,to\n"
SETTINGS = "setting,type,question,answers,min,max\n"


def fixture(name):
    return os.path.join(FIXTURES, name)


def write(name, content):
    """A generated file in the temporary folder: bytes as given, text as UTF-8."""
    path = os.path.join(TMP, name)
    with open(path, "wb") as handle:
        handle.write(content if isinstance(content, bytes) else content.encode("utf-8"))
    return path


def read(name, content, **kwargs):
    return csvfmt.read_csv(write(name, content), **kwargs)


def summary(problems):
    return [(p.severity, p.line, p.message) for p in problems or ()]


def only(problems, severity, line, message):
    """True when problems is exactly this one problem."""
    return summary(problems) == [(severity, line, message)]


def raised(parse, *args, **kwargs):
    """The problems of the CsvError that parse raises; None when it returns."""
    try:
        parse(*args, **kwargs)
    except csvfmt.CsvError as error:
        return error.problems
    return None


def lines(count, make):
    return "".join(make(i) + "\n" for i in range(count))


def test_read():
    path = fixture("life_factors.csv")
    doc = csvfmt.read_csv(os.path.join(FIXTURES, "..", "fixtures", "life_factors.csv"))
    status = os.stat(path)
    check("read: the path is absolute and normalised", doc.path == os.path.normpath(os.path.abspath(path)), doc.path)
    check("read: the stamp is (st_mtime_ns, st_size)", doc.stamp == (status.st_mtime_ns, status.st_size), doc.stamp)
    doc = csvfmt.read_csv(TMP)
    problem = doc.problems[0] if len(doc.problems) == 1 else None
    check(
        "read: an unreadable path is a problem, not an exception",
        problem and problem.message.startswith("Can't read ") and problem.file == doc.path and not doc.header,
        summary(doc.problems),
    )


def test_dialect():
    text = 'answer,when\nmetal,"steel, iron"\nwood,"planks\nand bark"\n'
    lf = read("lf.csv", text)
    check(
        "dialect: quoted delimiter and quoted line break",
        lf.rows == (("metal", "steel, iron"), ("wood", "planks\nand bark")),
        lf.rows,
    )
    check("dialect: physical line numbers", lf.lines == (2, 3), lf.lines)
    check(
        "dialect: a quoted line break warns",
        only(lf.problems, "warning", 3, "Line 3: a quoted cell has a line break"),
        summary(lf.problems),
    )
    for name, end in (("CRLF", "\r\n"), ("CR", "\r")):
        doc = read(name + ".csv", text.replace("\n", end))
        same = (doc.header, doc.rows, doc.lines, doc.text_hash, summary(doc.problems))
        check(
            f"dialect: {name} reads like LF",
            same == (lf.header, lf.rows, lf.lines, lf.text_hash, summary(lf.problems)),
            doc.rows,
        )
    doc = read("bom.csv", b"\xef\xbb\xbfanswer,when\nyes,cold\nno,warm\n")
    check("dialect: one leading BOM is stripped", doc.header == ("answer", "when") and not doc.problems, doc.header)
    check("dialect: the doc says it had a BOM, so a rewrite can keep it", doc.bom and not lf.bom, (doc.bom, lf.bom))

    doc = read("semicolon.csv", "answer;when;scale\nmetal;steel, iron;1,5\nwood;planks;0,6\n")
    check("dialect: ';' when the header line has no ','", doc.rows[0] == ("metal", "steel, iron", "1,5"), doc.rows)
    check("dialect: the doc says which delimiter it split with", doc.delimiter == ";" and lf.delimiter == ",",
          (doc.delimiter, lf.delimiter))
    warning = (
        "Column 'scale' has cells that don't read as numbers in lines 2, 3 (first: '1,5'); treated as text, no output"
    )
    check(
        "dialect: decimal commas make a text column, with a warning",
        not doc.num_cols and only(doc.problems, "warning", 2, warning),
        summary(doc.problems),
    )
    doc = read("grouped.csv", "answer;scale\na;1 000\nb;2 500\nc;1.234,5\n")
    warning = "Column 'scale' has cells that don't read as numbers in lines 2, 3, 4 (first: '1 000')"
    warning += "; treated as text, no output"
    check(
        "dialect: digit groups make a text column, with a warning",
        only(doc.problems, "warning", 2, warning),
        summary(doc.problems),
    )
    doc = read("tab.csv", "answer\twhen\nmetal\tsteel; iron, copper\n")
    check(
        "dialect: a tab when the header line has neither ',' nor ';'",
        doc.rows == (("metal", "steel; iron, copper"),),
        doc.rows,
    )
    doc = read("comma.csv", "answer,when;notes\nx,y;z\n")
    check("dialect: ',' wins over ';'", doc.header == ("answer", "when;notes"), doc.header)
    doc = read("explicit.csv", "a,b;c\n1,2;3\n", delimiter=";")
    check(
        "dialect: an explicit delimiter skips the sniffing",
        doc.header == ("a,b", "c") and doc.rows == (("1,2", "3"),),
        doc.header,
    )
    for number, delimiter in enumerate((";;", '"', "\n", "\r")):
        doc = read(f"delimiter_{number}.csv", 'a,b\n"x",y\n', delimiter=delimiter)
        message = f"The delimiter must be one character other than a quote or a line break, not {delimiter!r}"
        check(
            f"dialect: the delimiter {delimiter!r} is a problem, not an exception",
            only(doc.problems, "error", 0, message) and not doc.rows,
            summary(doc.problems),
        )
    doc = read("space_delimiter.csv", "x y z\na  b\n", delimiter=" ")
    check("dialect: a space delimiter keeps its empty cells", doc.rows == (("a", "", "b"),), doc.rows)
    doc = read("spaced_quotes.csv", 'answer, when, roughness\nmetal, "steel, iron", 0.35\n')
    good = (
        doc.rows == (("metal", "steel, iron", "0.35"),) and doc.num_cols == {"roughness": "FLOAT"} and not doc.problems
    )
    check(
        "dialect: a quote after the spaces that follow a delimiter still quotes",
        good,
        (doc.rows, summary(doc.problems)),
    )

    doc = read("quotes.csv", 'answer,when\nsay,"he said ""hi"", twice"\n')
    check("dialect: doubled quotes", doc.rows == (("say", 'he said "hi", twice'),), doc.rows)
    doc = read("unclosed.csv", 'answer,when\nx,"never closed\ny,z\n')
    check(
        "dialect: an unclosed quote is an error",
        [(p.severity, p.line) for p in doc.problems] == [("error", 2)],
        summary(doc.problems),
    )
    doc = read("layout.csv", "\n  \n  answer , when ,notes\n\n x , a when \n y,,z\n\n")
    check(
        "dialect: the header is the first non-blank row, trimmed", doc.header == ("answer", "when", "notes"), doc.header
    )
    check("dialect: cells trimmed, short rows padded", doc.rows == (("x", "a when", ""), ("y", "", "z")), doc.rows)
    check(
        "dialect: blank rows skipped, lines stay physical",
        doc.lines == (5, 6) and not doc.problems,
        (doc.lines, summary(doc.problems)),
    )
    doc = read("case.csv", "Answer,Alive,SCALE\na,1,0.5\n")
    check(
        "dialect: payload names keep their case",
        doc.num_cols == {"Alive": "INT", "SCALE": "FLOAT"} and doc.text_cols == ("Answer",),
        doc.num_cols,
    )


def test_classes():
    doc = read(
        "classes.csv",
        "answer,when,ints,big,floats,words,mixed,blank,inf,nan,truth,Confidence\n"
        "a,1,007,2147483648,0.5,x,3,,inf,NaN,true,1\n"
        "b,2,-2,1,.5,y,fragile,1,1,1,1,2\n"
        "c,3,+3,1,1e3,z,,2,2,2,2,3\n",
    )
    ints = doc.numbers.get("ints")
    check(
        "classes: INT",
        doc.num_cols.get("ints") == "INT" and ints == (7, -2, 3) and all(type(v) is int for v in ints),
        ints,
    )
    check(
        "classes: beyond int32 the column is FLOAT",
        doc.num_cols.get("big") == "FLOAT" and doc.numbers["big"] == (2147483648.0, 1.0, 1.0),
        doc.numbers.get("big"),
    )
    check(
        "classes: FLOAT",
        doc.num_cols.get("floats") == "FLOAT" and doc.numbers["floats"] == (0.5, 0.5, 1000.0),
        doc.numbers.get("floats"),
    )
    check(
        "classes: payload columns in header order", list(doc.num_cols) == ["ints", "big", "floats"], list(doc.num_cols)
    )
    check(
        "classes: reserved names are text whatever they hold",
        {"answer", "when", "Confidence"} <= set(doc.text_cols),
        doc.text_cols,
    )
    warned = {p.column: p.message for p in doc.problems if p.severity == "warning"}
    check(
        "classes: an empty cell, inf, nan or true in a number column warns",
        sorted(warned) == ["blank", "inf", "nan", "truth"] and len(doc.problems) == 4,
        summary(doc.problems),
    )
    check(
        "classes: the warning names the cell and its line",
        warned.get("inf") == "Column 'inf' has 'inf' in line 2; treated as text, no output",
        warned.get("inf"),
    )
    check(
        "classes: text columns with a stray number do not warn", {"words", "mixed"} <= set(doc.text_cols), doc.text_cols
    )
    doc = read("odd_cells.csv", "answer,a,b\nx,1,1\ny,,\nz,inf,\n")
    warned = [p.message for p in doc.problems]
    expected = [
        "Column 'a' has cells that don't read as numbers in lines 3, 4 (first: an empty cell); treated as text, "
        "no output",
        "Column 'b' has empty cells in lines 3, 4; treated as text, no output",
    ]
    check("classes: the warning names a value only when every odd cell holds it", warned == expected, warned)

    names = "factor,kind,level,value,from,to,setting,type,question,answers,min,max,prompt"
    doc = read("other_schemas.csv", f"answer,{names}\na,{','.join(['1'] * 13)}\n")
    check(
        "classes: an answer set may use every other schema's column names as payload",
        list(doc.num_cols) == names.split(","),
        doc.num_cols,
    )
    for name, content, node, payload in (
        (
            "items_factor_level.csv",
            "answer,when,factor,level,value\nchair,a seat,2,3,7\n",
            "pick",
            ("factor", "level", "value"),
        ),
        (
            "labels_settings.csv",
            "answer,when,setting,question,min,max\nmetal,shiny,1,2,0.5,1\n",
            "classify",
            ("setting", "question", "min", "max"),
        ),
        (
            "outcomes_factor_level.csv",
            "answer,when,factor,level,scale\nBirth,alive,1,3,0.5\n",
            "rules",
            ("factor", "level", "scale"),
        ),
    ):
        option = csvfmt.parse_answer_set(
            read(name, content), "one_of" if node != "pick" else "items", node=node
        ).options[0]
        check(
            f"classes: {name}: Factors and Settings names stay payload in an answer set",
            tuple(option.payload) == payload,
            option.payload,
        )
    for name, schema, columns in (
        ("life_factors.csv", "factors", ("value",)),
        ("biome_factors.csv", "factors", ("from", "to")),
        ("weather_dials.csv", "settings", ("min", "max")),
    ):
        doc = csvfmt.read_csv(fixture(name), schema=schema)
        good = set(columns) <= set(doc.text_cols) and not doc.num_cols and not doc.problems
        check(
            f"classes: {name} read as {schema} keeps every column as text, blanks and all",
            good,
            (doc.num_cols, summary(doc.problems)),
        )
    for parse, schema, content, message in (
        (csvfmt.parse_factors, "factors", "factor,level,from,to\nt,a,0,0.5\nt,b,0.5,\n", "No 'kind' column"),
        (csvfmt.parse_settings, "settings", "setting,question,min,max\nW,How windy?,0,\n", "No 'type' column"),
    ):
        problems = raised(parse, read(f"incomplete_{schema}.csv", content, schema=schema))
        check(
            f"classes: an incomplete {schema} file gives only its real error",
            only(problems, "error", 0, message),
            summary(problems),
        )
    wrong = []
    for parse, schema in ((csvfmt.parse_factors, "answer_set"), (csvfmt.parse_settings, "factors")):
        try:
            parse(csvfmt.read_csv(fixture("life_factors.csv"), schema=schema))
        except ValueError as error:
            wrong.append(str(error))
    try:
        csvfmt.parse_answer_set(csvfmt.read_csv(fixture("props.csv"), schema="settings"), "items", node="pick")
    except ValueError as error:
        wrong.append(str(error))
    try:
        csvfmt.read_csv(fixture("props.csv"), schema="factor")
    except ValueError as error:
        wrong.append(str(error))
    expected = [
        "life_factors.csv was read as 'answer_set', not 'factors'",
        "life_factors.csv was read as 'factors', not 'settings'",
        "props.csv was read as 'settings', not 'answer_set'",
        "unknown schema 'factor'",
    ]
    check(
        "classes: a document read for another schema, or an unknown one, is a programming error",
        wrong == expected,
        wrong,
    )
    doc = read("items_with_settings_names.csv", "answer,when,setting,type,question\n1,10,a,b,c\n")
    items = csvfmt.parse_answer_set(doc, "items", node="pick")
    check(
        "classes: an Items CSV with Settings column names keeps the answer-set names reserved",
        not doc.num_cols and items.options[0].payload == {},
        doc.num_cols,
    )
    doc = read(
        "dials_overrides.csv", "setting,type,question,answer,lock,confidence\nRain,yes/no,Is it raining?,yes,1,0.93\n"
    )
    good = (
        not doc.problems
        and not doc.num_cols
        and csvfmt.parse_overrides(doc, ["setting"]) == {"rain": csvfmt.Lock("yes", 2)}
    )
    check(
        "classes: a Dials Overrides file with type and question keeps lock and confidence reserved",
        good,
        (doc.num_cols, summary(doc.problems)),
    )

    digits = "7" * 5000
    doc = read("digits.csv", f"answer,v\na,{digits}\n")
    message = "Line 2: column 'v' has 5 000 characters (limit 2 000)"
    check(
        "classes: a cell of 5 000 digits is a problem, not an exception",
        only(doc.problems, "error", 2, message),
        summary(doc.problems),
    )
    doc = read("zeros.csv", "answer,v,w\na," + "0" * 30 + "7,12345678901\n")
    good = doc.numbers.get("v") == (7,) and doc.num_cols.get("w") == "FLOAT"
    check("classes: leading zeros don't count towards int32; 11 digits make FLOAT", good, (doc.num_cols, doc.numbers))


def test_limits():
    doc = read("big.csv", b"answer\n" + b"x" * csvfmt.MAX_BYTES)
    check(
        "limits: over 1 MB",
        only(doc.problems, "error", 0, "big.csv has 1 048 583 bytes (limit 1 048 576)") and not doc.header,
        summary(doc.problems),
    )
    doc = read("rows_4096.csv", "answer\n" + lines(4096, lambda i: f"r{i}"))
    check("limits: 4 096 rows are fine", len(doc.rows) == 4096 and not doc.problems, summary(doc.problems))
    doc = read("columns_64.csv", ",".join(f"c{i}" for i in range(64)) + "\n")
    check("limits: 64 columns are fine", len(doc.header) == 64 and not doc.problems, summary(doc.problems))
    doc = read("columns_65.csv", ",".join(f"c{i}" for i in range(65)) + "\n")
    check(
        "limits: 65 columns, and nothing more is read",
        only(doc.problems, "error", 1, "columns_65.csv has 65 columns (limit 64)") and not doc.header,
        summary(doc.problems),
    )
    doc = read("cell_2000.csv", "answer,when\na," + "w" * 2000 + "\n")
    check("limits: 2 000 characters in a cell are fine", not doc.problems, summary(doc.problems))
    doc = read("cell_2001.csv", "answer,when\na," + "w" * 2001 + "\n")
    message = "Line 2: column 'when' has 2 001 characters (limit 2 000)"
    check("limits: 2 001 characters in a cell", only(doc.problems, "error", 2, message), summary(doc.problems))
    doc = read("cell_huge.csv", "answer,when\na," + "w" * 140000 + "\nb,c\n")
    message = "Line 2: a cell has over 131 072 characters (limit 2 000), or a quote is not closed"
    check(
        "limits: a cell over the csv module's own cap", only(doc.problems, "error", 2, message), summary(doc.problems)
    )
    doc = read("runaway_quote.csv", 'answer,when\na,"never closed\n' + lines(49000, lambda i: "b,c"))
    check(
        "limits: a quote never closed in a big file names both causes",
        only(doc.problems, "error", 2, message),
        summary(doc.problems),
    )

    doc = read("rows_many.csv", "answer\n" + lines(6000, lambda i: f"r{i}"))
    good = only(doc.problems, "error", 0, "rows_many.csv has 6 000 rows (limit 4 096)") and len(doc.rows) == 4096
    check("limits: rows past the limit are counted, not kept", good, (len(doc.rows), summary(doc.problems)))
    doc = read("long_rows.csv", "answer\n" + lines(60, lambda i: f"r{i},extra"))
    listed = summary(doc.problems)
    good = listed == [("error", n, f"Line {n} has 2 cells; the header has 1") for n in range(2, 62)]
    check("limits: every long row is listed", good, (len(listed), listed[-1:]))
    doc = read("long_cells.csv", "answer,when\n" + lines(55, lambda i: f"r{i}," + "w" * 2001))
    listed = summary(doc.problems)
    good = listed == [("error", n, f"Line {n}: column 'when' has 2 001 characters (limit 2 000)") for n in range(2, 57)]
    check("limits: every long cell is listed", good, (len(listed), listed[-1:]))


def test_messages():
    """One broken fixture per design 7.9 message given by the reader or the parsers, alone in its file."""
    missing = fixture("broken_missing.csv")
    doc = csvfmt.read_csv(missing)
    message = f"Can't find {os.path.normpath(missing)}"
    check("7.9 missing file", only(doc.problems, "error", 0, message) and doc.stamp == (0, 0), summary(doc.problems))

    doc = csvfmt.read_csv(fixture("broken_rows.csv"))
    check(
        "7.9 over a limit",
        only(doc.problems, "error", 0, "broken_rows.csv has 5 000 rows (limit 4 096)"),
        summary(doc.problems),
    )

    with open(fixture("broken_utf8.csv"), "rb") as handle:
        at = handle.read().index(b"\xe9")
    doc = csvfmt.read_csv(fixture("broken_utf8.csv"))
    message = f"broken_utf8.csv is not UTF-8 text (byte {at})"
    check("7.9 not UTF-8", only(doc.problems, "error", 0, message) and not doc.rows, summary(doc.problems))
    data = b"answer,when\nyes,co\x00ld\n"
    doc = read("nul_byte.csv", data)  # not "nul.csv": NUL is a device name on Windows
    message = f"nul_byte.csv is not UTF-8 text (byte {data.index(0)})"
    check("7.9 a NUL byte", only(doc.problems, "error", 0, message), summary(doc.problems))

    doc = csvfmt.read_csv(fixture("broken_header.csv"))
    check("7.9 duplicate header", only(doc.problems, "error", 1, "Two columns are named 'when'"), summary(doc.problems))
    check("7.9 problems name their file", all(p.file == doc.path for p in doc.problems), [p.file for p in doc.problems])
    doc = read("empty_header.csv", "answer,,when\na,b,c\n")
    check("7.9 empty header", only(doc.problems, "error", 1, "Column 2 has no name"), summary(doc.problems))

    problems = raised(csvfmt.parse_answer_set, csvfmt.read_csv(fixture("broken_no_answer.csv")), "one_of")
    message = "No 'answer' column (or name, item, label, outcome)"
    check("7.9 no name column", only(problems, "error", 0, message), summary(problems))

    doc = csvfmt.read_csv(fixture("broken_answer.csv"))
    problems = raised(csvfmt.parse_answer_set, doc, "one_of", node="rules")
    check("7.9 duplicate answer", only(problems, "error", 5, "Line 5: 'Birth' appears twice"), summary(problems))

    doc = csvfmt.read_csv(fixture("broken_long_row.csv"))
    check(
        "7.9 row longer than the header",
        only(doc.problems, "error", 7, "Line 7 has 5 cells; the header has 4"),
        summary(doc.problems),
    )

    problems = raised(csvfmt.parse_answer_set, csvfmt.read_csv(fixture("broken_yes_no.csv")), "yes_no")
    message = "A Yes/No answer set needs rows 'yes' and 'no'"
    check("7.9 Yes/No set without yes and no", only(problems, "error", 0, message), summary(problems))

    problems = raised(csvfmt.parse_factors, csvfmt.read_csv(fixture("broken_bands.csv"), schema="factors"))
    message = "temperature: 'to' of cold (0.3) must equal 'from' of mild (0.35)"
    check("7.9 invalid band edges", only(problems, "error", 3, message), summary(problems))

    problems = raised(csvfmt.parse_answer_set, csvfmt.read_csv(fixture("broken_payload.csv")), "one_of", node="rules")
    message = "Rename column 'Amount': it clashes with a node output"
    check("7.9 payload name clash", only(problems, "error", 0, message), summary(problems))
    header = "answer," + ",".join(f"p{i}" for i in range(17))
    problems = raised(
        csvfmt.parse_answer_set, read("payload_17.csv", header + "\na," + ",".join("1" * 17) + "\n"), "one_of"
    )
    message = "payload_17.csv has 17 payload columns (limit 16)"
    check("7.9 more than 16 payload columns", only(problems, "error", 0, message), summary(problems))

    doc = csvfmt.read_csv(fixture("broken_number_like.csv"))
    message = "Column 'scale' has an empty cell in line 7; treated as text, no output"
    check("7.9 number-like column with a blank", only(doc.problems, "warning", 7, message), summary(doc.problems))

    labels = csvfmt.parse_answer_set(csvfmt.read_csv(fixture("broken_if.csv")), "one_of", node="classify")
    message = "'if' is only used by Rules; ignored"
    check("7.9 'if' outside Rules", only(labels.problems, "warning", 2, message), summary(labels.problems))
    check("7.9 'if' outside Rules is dropped", [o.if_expr for o in labels.options] == ["", ""], labels.options)

    doc = csvfmt.read_csv(fixture("broken_blank_line.csv"))
    message = "Line 4 is blank: blank lines between rows are skipped"
    check(
        "7.9 blank line between rows",
        only(doc.problems, "warning", 4, message) and doc.lines == (2, 3, 5),
        summary(doc.problems),
    )

    doc = csvfmt.read_csv(fixture("broken_quoted_break.csv"))
    message = "Line 3: a quoted cell has a line break"
    check(
        "7.9 quoted line break",
        only(doc.problems, "warning", 3, message) and doc.lines == (2, 3, 5),
        summary(doc.problems),
    )

    labels = csvfmt.parse_answer_set(csvfmt.read_csv(fixture("broken_unused_column.csv")), "one_of")
    message = "Column 'notes' is text and is not used"
    check("7.9 text column that nothing uses", only(labels.problems, "info", 0, message), summary(labels.problems))


def test_answer_sets():
    life = csvfmt.parse_answer_set(csvfmt.read_csv(fixture("life_outcomes.csv")), "one_of", node="rules")
    names = [o.name for o in life.options]
    check("answers: Game of Life outcomes in row order", names == ["Birth", "Stays empty", "Survives", "Dies"], names)
    when = "A dead cell with exactly three live neighbours comes to life."
    check("answers: whens", life.options[0].when == when, life.options[0].when)
    ifs = [o.if_expr for o in life.options]
    check("answers: Rules keep 'if'", ifs == ["state=dead", "state=dead", "state=alive", "state=alive"], ifs)
    payload = [o.payload for o in life.options]
    check("answers: payload per option", payload == [{"alive": 1}, {"alive": 0}, {"alive": 1}, {"alive": 0}], payload)
    check(
        "answers: a clean file has no problems",
        life.problems == () and life.doc.num_cols == {"alive": "INT"},
        summary(life.problems),
    )
    for name, count, birth in (
        ("highlife_outcomes.csv", 4, "three or six"),
        ("seeds_outcomes.csv", 3, "exactly two"),
        ("daynight_outcomes.csv", 4, "three, six, seven or eight"),
    ):
        rules = csvfmt.parse_answer_set(csvfmt.read_csv(fixture(name)), "one_of", node="rules")
        good = len(rules.options) == count and birth in rules.options[0].when and not rules.problems
        check(f"answers: {name}", good and rules.doc.num_cols == {"alive": "INT"}, [o.name for o in rules.options])
    families = csvfmt.parse_answer_set(csvfmt.read_csv(fixture("material_families.csv")), "one_of", node="classify")
    metal = families.options[0]
    good = metal.when == "steel, iron, copper, brass or aluminium surfaces" and metal.payload == {
        "roughness": 0.35,
        "metallic": 1,
    }
    check(
        "answers: material families", good and families.doc.num_cols == {"roughness": "FLOAT", "metallic": "INT"}, metal
    )
    biome = csvfmt.parse_answer_set(csvfmt.read_csv(fixture("biome_outcomes.csv")), "one_of", node="rules")
    check(
        "answers: biome outcomes",
        len(biome.options) == 7 and biome.doc.num_cols == {"instance": "INT", "scale": "FLOAT"},
        biome.doc.num_cols,
    )
    unknown = csvfmt.parse_answer_set(biome.doc, "one_of")
    check(
        "answers: with the node unknown, biome's 'instance' is payload",
        unknown.options[0].payload == {"instance": 0, "scale": 1.2},
        unknown.options[0],
    )
    props = csvfmt.parse_answer_set(csvfmt.read_csv(fixture("props.csv")), "items", node="pick")
    anvil = props.options[8]
    check(
        "answers: props, an Items CSV",
        len(props.options) == 10 and anvil.name == "anvil" and anvil.payload == {"instance": 8, "scale": 0.6},
        anvil,
    )
    for node in ("match", "ask", "classify", ""):
        items = csvfmt.parse_answer_set(props.doc, "items", node=node)
        baked = {"instance": 8, "scale": 0.6} if node == "" else {}
        check(
            f"answers: props as items for node {node!r}",
            items.options[8].payload == baked and not items.problems,
            items.options[8],
        )
    wide = read(
        "wide.csv",
        "answer,"
        + ",".join(f"p{i}" for i in range(17))
        + "\nyes,"
        + ",".join("1" * 17)
        + "\nno,"
        + ",".join("0" * 17)
        + "\n",
    )
    answers = csvfmt.parse_answer_set(wide, "yes_no", node="match")
    check(
        "answers: numbers where no node bakes them are no payload, whatever their count",
        not answers.problems and answers.options[0].payload == {},
        summary(answers.problems),
    )
    problems = raised(csvfmt.parse_answer_set, read("index_payload.csv", "answer,Index\na,1\n"), "items")
    message = "Rename column 'Index': it clashes with a node output"
    check(
        "answers: with the node unknown, items payload may not take Pick's outputs",
        only(problems, "error", 0, message),
        summary(problems),
    )
    long_name = "p" * 64
    problems = raised(
        csvfmt.parse_answer_set, read("payload_long.csv", f"answer,{long_name}\na,1\n"), "one_of", node="classify"
    )
    message = f"Rename column '{long_name}': it has 64 bytes in UTF-8 (limit 63)"
    check("answers: a payload name over 63 bytes", only(problems, "error", 0, message), summary(problems))
    rules = csvfmt.parse_answer_set(csvfmt.read_csv(fixture("life_outcomes.csv")), "one_of")
    check(
        "answers: with the node unknown, 'if' is kept, without a warning",
        [o.if_expr for o in rules.options] == ifs and not rules.problems,
        summary(rules.problems),
    )

    answers = csvfmt.parse_answer_set(read("yes_no.csv", "answer,when\nNo,warm and calm\nYES,cold or wet\n"), "yes_no")
    check(
        "answers: Yes/No rows in any order and case",
        [o.name for o in answers.options] == ["No", "YES"],
        answers.options,
    )
    for count, kind, message in (
        (1, "amount", "An Amount answer set needs 2 to 16 levels; this one has 1"),
        (17, "amount", "An Amount answer set needs 2 to 16 levels; this one has 17"),
        (0, "one_of", "A One-of answer set needs 1 to 255 answers; this one has 0"),
        (256, "items", "An item list needs 1 to 255 items; this one has 256"),
    ):
        doc = read(f"count_{kind}_{count}.csv", "answer\n" + lines(count, lambda i: f"a{i}"))
        problems = raised(csvfmt.parse_answer_set, doc, kind)
        check(f"answers: {count} rows for {kind}", only(problems, "error", 0, message), summary(problems))
    for count, kind in ((2, "amount"), (16, "amount"), (1, "one_of"), (255, "one_of"), (255, "items")):
        doc = read(f"count_{kind}_{count}.csv", "answer\n" + lines(count, lambda i: f"a{i}"))
        check(f"answers: {count} rows are fine for {kind}", len(csvfmt.parse_answer_set(doc, kind).options) == count)
    problems = raised(csvfmt.parse_answer_set, read("empty_name.csv", "answer,when\na,x\n,y\n"), "one_of")
    check("answers: an empty name", only(problems, "error", 3, "Line 3: the name is empty"), summary(problems))
    problems = raised(csvfmt.parse_answer_set, read("long_name.csv", "answer\n" + "n" * 201 + "\n"), "one_of")
    message = "Line 2: the name has 201 characters (limit 200)"
    check("answers: a name over 200 characters", only(problems, "error", 2, message), summary(problems))

    option = csvfmt.parse_answer_set(read("synonyms.csv", "Name,Description\nx,an x\n"), "one_of").options[0]
    check(
        "answers: 'name' and 'description' stand for 'answer' and 'when'",
        (option.name, option.when) == ("x", "an x"),
        option,
    )
    answers = csvfmt.parse_answer_set(read("preferred.csv", "label,answer\nL,A\n"), "one_of")
    message = "Column 'label' is not used: 'label' is a reserved name"
    check(
        "answers: 'answer' wins over its synonyms",
        answers.options[0].name == "A" and only(answers.problems, "info", 0, message),
        summary(answers.problems),
    )
    answers = csvfmt.parse_answer_set(read("empty_if.csv", "answer,if\na,\nb,\n"), "one_of", node="classify")
    check("answers: an empty 'if' column outside Rules is silent", not answers.problems, summary(answers.problems))
    problems = raised(csvfmt.parse_answer_set, csvfmt.read_csv(fixture("broken_utf8.csv")), "one_of")
    good = len(problems or ()) == 1 and problems[0].message.startswith("broken_utf8.csv is not UTF-8 text")
    check("answers: a file that could not be read gives only its read error", good, summary(problems))
    for case, args in (
        ("kind", ("one-of", "rules")),
        ("node 'Pick'", ("items", "Pick")),
        ("node 'Rules'", ("one_of", "Rules")),
    ):
        try:
            csvfmt.parse_answer_set(life.doc, *args)
            unknown = False
        except ValueError:
            unknown = True
        check(f"answers: an unknown {case} is a programming error", unknown)


def test_factors():
    life = csvfmt.parse_factors(csvfmt.read_csv(fixture("life_factors.csv"), schema="factors"))
    shape = [(f.name, f.kind, len(f.levels), f.line) for f in life]
    check("factors: Game of Life", shape == [("state", "flag", 2, 2), ("neighbours", "category", 9, 4)], shape)
    state = [(level.name, level.value, level.band) for level in life[0].levels]
    check("factors: flag levels with their values", state == [("dead", 0, None), ("alive", 1, None)], state)
    eight = life[1].levels[8]
    good = [level.value for level in life[1].levels] == list(range(9)) and eight.line == 12
    check(
        "factors: category values",
        good and eight.when == "Eight live neighbours; every surrounding cell is alive.",
        eight,
    )
    biome = csvfmt.parse_factors(csvfmt.read_csv(fixture("biome_factors.csv"), schema="factors"))
    check(
        "factors: biome order",
        [f.name for f in biome] == ["temperature", "moisture", "altitude"],
        [f.name for f in biome],
    )
    bands = [level.band for level in biome[0].levels]
    check("factors: custom band edges", bands == [(0.0, 0.3), (0.3, 0.7), (0.7, 1.0)], bands)
    equal = all(level.band is None and level.value is None for f in biome[1:] for level in f.levels)
    check("factors: equal-width bands, and amount levels carry no value", equal, biome[1].levels)
    for name, values in (("negative", (-1, 0, 1)), ("sparse", (0, 5, 4095)), ("far", (10000, 10001))):
        body = "".join(f"c,category,l{value},{value}\n" for value in values)
        levels = csvfmt.parse_factors(read(f"category_{name}.csv", FACTORS + body, schema="factors"))[0].levels
        check(f"factors: category values {values} are fine", tuple(level.value for level in levels) == values, levels)

    def amount(name, count):
        return lines(count, lambda i: f"{name},amount,l{i}")

    def category(name, count):
        return lines(count, lambda i: f"{name},category,l{i},{i}")

    def flag(name):
        return f"{name},flag,off,0\n{name},flag,on,1\n"

    long_name = "f" * 64
    for case, body, line, message in (
        ("together", flag("a") + flag("b") + "a,flag,z,1\n", 6, "Line 6: the rows of factor 'a' must be together"),
        ("no_kind", "a,,x,0\na,flag,y,1\n", 2, "Line 2: factor 'a' needs a kind: amount, category or flag"),
        ("bad_kind", "a,bool,x,0\na,bool,y,1\n", 2, "Line 2: factor 'a' needs a kind: amount, category or flag"),
        ("kind_differs", "a,flag,x,0\na,category,y,1\n", 3, "Line 3: kind 'category' differs from 'flag' on line 2"),
        ("empty_level", "a,flag,,0\na,flag,y,1\n", 2, "Line 2: the level name is empty"),
        ("level_twice", "a,flag,x,0\na,flag,X,1\n", 3, "Line 3: level 'X' appears twice in factor 'a'"),
        ("amount_1", amount("t", 1), 2, "t: amount factors have 2 to 16 levels, not 1"),
        ("amount_17", amount("t", 17), 2, "t: amount factors have 2 to 16 levels, not 17"),
        ("category_65", category("c", 65), 2, "c: category factors have 2 to 64 levels, not 65"),
        ("flag_3", flag("a") + "a,flag,z,2\n", 2, "a: a flag has exactly 2 levels, not 3"),
        ("flag_values", "a,flag,x,1\na,flag,y,2\n", 2, "a: a flag's values must be 0 and 1"),
        ("no_value", "c,category,x,\nc,category,y,1\n", 2, "Line 2: level 'x' needs an integer 'value'"),
        ("float_value", "c,category,x,1.5\nc,category,y,2\n", 2, "Line 2: level 'x' needs an integer 'value'"),
        ("value_twice", "c,category,x,1\nc,category,y,1\n", 3, "Line 3: value 1 appears twice in factor 'c'"),
        (
            "category_spread",
            "c,category,x,0\nc,category,y,5000\n",
            2,
            "c: category values must count up by one, or all lie between 0 and 4095",
        ),
        ("long_name", flag(long_name), 2, "Line 2: the factor name has 64 bytes in UTF-8 (limit 63)"),
        ("wide_name", flag("\u4e00" * 22), 2, "Line 2: the factor name has 66 bytes in UTF-8 (limit 63)"),
        ("clash", flag("Seed"), 2, "Line 2: rename factor 'Seed': it clashes with a node input"),
        ("clash_key", flag("key"), 2, "Line 2: rename factor 'key': it clashes with a node input"),
        ("clash_source", flag("source"), 2, "Line 2: rename factor 'source': it clashes with an Overrides CSV column"),
        ("clash_lock", flag("Lock"), 2, "Line 2: rename factor 'Lock': it clashes with an Overrides CSV column"),
        ("empty_name", ",flag,x,0\n" + flag("a"), 2, "Line 2: the factor name is empty"),
        ("nine", "".join(flag(f"f{i}") for i in range(9)), 0, "Rules need 1 to 8 factors; this file has 9"),
        ("none", "", 0, "Rules need 1 to 8 factors; this file has 0"),
        (
            "cells",
            amount("a", 15) + amount("b", 15) + category("c", 24),
            0,
            "Rules has 5 400 cells (limit 4 096): use fewer bands",
        ),
        (
            "partial_edges",
            "t,amount,a,,,0,0.5\nt,amount,b,,,,\n",
            2,
            "t: give 'from' and 'to' on every level, or on none",
        ),
        ("first_edge", "t,amount,a,,,0.1,0.5\nt,amount,b,,,0.5,1\n", 2, "t: 'from' of a (0.1) must be 0"),
        ("last_edge", "t,amount,a,,,0,0.5\nt,amount,b,,,0.5,0.9\n", 3, "t: 'to' of b (0.9) must be 1"),
        (
            "rising_edges",
            "t,amount,a,,,0,0.5\nt,amount,b,,,0.5,0.5\nt,amount,c,,,0.5,1\n",
            3,
            "t: 'from' of b (0.5) must be below its 'to' (0.5)",
        ),
        (
            "edge_text",
            "t,amount,a,,,zero,0.5\nt,amount,b,,,0.5,1\n",
            2,
            "Line 2: 'from' of a must be a number, not 'zero'",
        ),
    ):
        problems = raised(csvfmt.parse_factors, read(f"factors_{case}.csv", FACTORS + body, schema="factors"))
        check(f"factors: {case}", only(problems, "error", line, message), summary(problems))
    problems = raised(csvfmt.parse_factors, read("factors_columns.csv", "factor,level\na,x\n", schema="factors"))
    check("factors: a required column is missing", only(problems, "error", 0, "No 'kind' column"), summary(problems))
    name = "\u4e00" * 21
    factors = csvfmt.parse_factors(read("factors_wide_ok.csv", FACTORS + flag(name), schema="factors"))
    check("factors: a name of exactly 63 bytes is fine", factors[0].name == name)


def test_if():
    life = csvfmt.parse_factors(csvfmt.read_csv(fixture("life_factors.csv"), schema="factors"))
    biome = csvfmt.parse_factors(csvfmt.read_csv(fixture("biome_factors.csv"), schema="factors"))
    body = "size,category,small,10\nsize,category,large,20\n"
    sizes = csvfmt.parse_factors(read("if_sizes.csv", FACTORS + body, schema="factors"))
    clause = rules_core.Clause
    for text, factors, expected in (
        ("", life, ()),
        ("  ", life, ()),
        ("state=dead", life, (clause(0, False, frozenset({0})),)),
        (" State = ALIVE ", life, (clause(0, False, frozenset({1})),)),
        ("state=1", life, (clause(0, False, frozenset({1})),)),
        ("neighbours=2|3", life, (clause(1, False, frozenset({2, 3})),)),
        ("neighbours != 2 | 3", life, (clause(1, True, frozenset({2, 3})),)),
        ("state=dead;neighbours=3", life, (clause(0, False, frozenset({0})), clause(1, False, frozenset({3})))),
        ("size=large", sizes, (clause(0, False, frozenset({1})),)),
        ("size=10|LARGE", sizes, (clause(0, False, frozenset({0, 1})),)),
        ("temperature=cold|Mild", biome, (clause(0, False, frozenset({0, 1})),)),
        ("altitude!=high; moisture=wet", biome, (clause(2, True, frozenset({2})), clause(1, False, frozenset({2})))),
    ):
        got = rules_core.parse_if(text, factors)
        check(f"if: {text!r}", got == expected, got)
    for text, factors, message in (
        ("state", life, "'if' clause 'state' needs '=' or '!='"),
        ("state=dead;", life, "'if' has an empty clause"),
        (";state=dead", life, "'if' has an empty clause"),
        ("=dead", life, "'if' clause '=dead' names no factor"),
        ("!=dead", life, "'if' clause '!=dead' names no factor"),
        ("state=", life, "'if' clause 'state=' has an empty level"),
        ("state=dead|", life, "'if' clause 'state=dead|' has an empty level"),
        ("colour=red", life, "'if' names unknown factor 'colour'"),
        ("state=alvie", life, "'if' names unknown level 'alvie' of factor 'state'"),
        ("neighbours=9", life, "'if' names unknown level '9' of factor 'neighbours'"),
        ("size=15", sizes, "'if' names unknown level '15' of factor 'size'"),
        ("temperature=0", biome, "'if' names unknown level '0' of factor 'temperature'"),  # amount levels: names only
    ):
        try:
            rules_core.parse_if(text, factors)
            got = None
        except rules_core.IfError as error:
            got = str(error)
        check(f"if: {text!r} is an error", got == message, got)
    answers = csvfmt.parse_answer_set(csvfmt.read_csv(fixture("broken_if.csv")), "one_of", node="rules")
    problems = raised(rules_core.restrictions, answers, life)
    message = "Line 2: 'if' names unknown level 'shiny' of factor 'state'"
    check("7.9 'if' names an unknown level", only(problems, "error", 2, message), summary(problems))
    doc = read("if_lines.csv", "answer,IF\na,state=alvie\nb,\nc,colour=red\nd,state=dead\n")
    problems = raised(rules_core.restrictions, csvfmt.parse_answer_set(doc, "one_of", node="rules"), life)
    expected = [
        ("error", 2, "Line 2: 'if' names unknown level 'alvie' of factor 'state'"),
        ("error", 4, "Line 4: 'if' names unknown factor 'colour'"),
    ]
    good = summary(problems) == expected and {(p.file, p.column) for p in problems} == {(doc.path, "IF")}
    check("if: every bad line is listed, in the file's 'if' column", good, summary(problems))


def test_settings():
    weather = csvfmt.parse_settings(csvfmt.read_csv(fixture("weather_dials.csv"), schema="settings"))
    kinds = [(s.name, s.type) for s in weather]
    expected = [("Rain", "yes_no"), ("Wind speed", "amount"), ("Time of day", "one_of"), ("Street lights", "yes_no")]
    check("settings: weather dials", kinds == expected + [("Fog", "amount")], kinds)
    wind = weather[1]
    good = (wind.min, wind.max) == (0.0, 30.0) and wind.answers[0] == ("calm", "still air") and len(wind.answers) == 5
    check("settings: amount levels and range", good and wind.question == "How windy is it?" and wind.line == 3, wind)
    check(
        "settings: one-of options",
        [name for name, _ in weather[2].answers] == ["dawn", "day", "dusk", "night"],
        weather[2],
    )
    whens = (("yes", "it is dark or foggy"), ("no", "there is enough daylight"))
    check("settings: yes/no whens", weather[3].answers == whens, weather[3].answers)
    rain = weather[0]
    check("settings: defaults", (rain.answers, rain.min, rain.max) == ((), 0.0, 1.0) and weather[4].answers == (), rain)
    body = "A,bool,q?,,,\nB,yes_no,q?,,,\nC,choice,q?,x: 1; y: 2,,\nD,one_of,q?,x;y,,\nE,Yes/No,q?,,,\nF,AMOUNT,q?,,,\n"
    types_ = [s.type for s in csvfmt.parse_settings(read("settings_types.csv", SETTINGS + body, schema="settings"))]
    check("settings: type synonyms", types_ == ["yes_no", "yes_no", "one_of", "one_of", "yes_no", "amount"], types_)
    content = SETTINGS + "W,amount,q?,,30,0\nR,yes/no,q?,,abc,\n"
    inverted = csvfmt.parse_settings(read("settings_inverted.csv", content, schema="settings"))
    good = (inverted[0].min, inverted[0].max) == (30.0, 0.0) and (inverted[1].min, inverted[1].max) == (0.0, 1.0)
    check("settings: min above max is allowed; min and max only count for amount", good, inverted)
    content = SETTINGS + lines(32, lambda i: f"s{i},yes/no,q?,,,")
    at_limit = csvfmt.parse_settings(read("settings_32.csv", content, schema="settings"))
    check("settings: 32 settings, the limit, are accepted", len(at_limit) == 32, len(at_limit))

    for case, body, line, message in (
        ("one_of_none", "T,one-of,q?,,,\n", 2, "Line 2: a one-of setting needs answers"),
        (
            "one_of_one",
            "T,one-of,q?,dawn: first light,,\n",
            2,
            "Line 2, answers: A One-of answer set needs 2 to 255 answers; this one has 1",
        ),
        (
            "amount_one",
            "W,amount,q?,calm,,\n",
            2,
            "Line 2, answers: An Amount answer set needs 2 to 16 levels; this one has 1",
        ),
        (
            "yes_no",
            "R,yes/no,q?,yes: wet; maybe: unsure,,\n",
            2,
            "Line 2, answers: A Yes/No answer set needs rows 'yes' and 'no'",
        ),
        ("option_twice", "T,one-of,q?,x: a; X: b,,\n", 2, "Line 2, answers: 'X' appears twice"),
        ("bad_type", "T,maybe,q?,,,\n", 2, "Line 2: type 'maybe' is not yes/no, amount or one-of"),
        ("no_question", "T,yes/no,,,,\n", 2, "Line 2: the question is empty"),
        ("twice", "Rain,yes/no,q?,,,\nrain,yes/no,q?,,,\n", 3, "Line 3: 'rain' appears twice"),
        (
            "confidence",
            "Confidence,amount,q?,,,\n",
            2,
            "Line 2: rename setting 'Confidence': it clashes with a node output",
        ),
        ("long_name", "s" * 64 + ",yes/no,q?,,,\n", 2, "Line 2: the setting name has 64 bytes in UTF-8 (limit 63)"),
        (
            "wide_name",
            "\u00e9" * 32 + ",yes/no,q?,,,\n",
            2,
            "Line 2: the setting name has 64 bytes in UTF-8 (limit 63)",
        ),
        ("empty_name", ",yes/no,q?,,,\n", 2, "Line 2: the setting name is empty"),
        ("min_text", "W,amount,q?,,low,30\n", 2, "Line 2: 'min' must be a number, not 'low'"),
        ("many", lines(33, lambda i: f"s{i},yes/no,q?,,,"), 0, "settings_many.csv has 33 settings (limit 32)"),
        ("none", "", 0, "No settings"),
    ):
        problems = raised(csvfmt.parse_settings, read(f"settings_{case}.csv", SETTINGS + body, schema="settings"))
        check(f"settings: {case}", only(problems, "error", line, message), summary(problems))
    problems = raised(csvfmt.parse_settings, read("settings_columns.csv", "setting,type\nA,bool\n", schema="settings"))
    check(
        "settings: a required column is missing", only(problems, "error", 0, "No 'question' column"), summary(problems)
    )


def test_inline_settings():
    parse = csvfmt.parse_inline_settings
    got = [(s.name, s.type, s.question, s.min, s.max, s.answers)
           for s in parse("Rain (yes/no): Is it raining?; Wind speed (amount): How windy is it?")]
    check("inline settings: 'name (type): question' separated by ';'",
          got == [("Rain", "yes_no", "Is it raining?", 0.0, 1.0, ()),
                  ("Wind speed", "amount", "How windy is it?", 0.0, 1.0, ())], got)
    got = [(s.name, s.type, s.question) for s in parse("Wet (bool): Is it wet, cold or windy?\nFog (Amount):How foggy?\n")]
    check("inline settings: one per line keeps commas, type synonyms, no space after ':'",
          got == [("Wet", "yes_no", "Is it wet, cold or windy?"), ("Fog", "amount", "How foggy?")], got)
    got = [s.name for s in parse("Speed (km/h) (amount): How fast?")]
    check("inline settings: the last parentheses hold the type", got == ["Speed (km/h)"], got)
    at_limit = parse("; ".join(f"s{i} (yes/no): q?" for i in range(32)))
    check("inline settings: 32 settings, the limit, are accepted", len(at_limit) == 32, len(at_limit))
    for text, message in (
        ("; ".join(f"s{i} (yes/no): q?" for i in range(33)), "33 settings (limit 32)"),
        ("Time (one-of): What time is it?", "Entry 1: a one-of setting needs a Settings CSV"),
        ("Rain: Is it raining?", "Entry 1: write it as 'name (type): question', e.g. 'Rain (yes/no): Is it raining?'"),
        ("Rain (yes/no)", "Entry 1: write it as 'name (type): question', e.g. 'Rain (yes/no): Is it raining?'"),
        ("Rain (maybe): Is it raining?", "Entry 1: type 'maybe' is not yes/no or amount"),
        ("Rain (yes/no):  ", "Entry 1: the question is empty"),
        ("Rain (yes/no): q?; rain (amount): q?", "Entry 2: 'rain' appears twice"),
        ("Confidence (amount): q?", "Entry 1: rename setting 'Confidence': it clashes with a node output"),
        ("s" * 64 + " (yes/no): q?", "Entry 1: the setting name has 64 bytes in UTF-8 (limit 63)"),
        (" ; ", "No settings"),
    ):
        problems = raised(parse, text)
        check(f"inline settings: {message[:60]}", only(problems, "error", 0, message), summary(problems))


def test_overrides():
    doc = read(
        "life_overrides.csv",
        "state,neighbours,answer,lock,model_answer,confidence,unsure,source,note\n"
        "dead,3,Birth,0,Birth,0.98,0,model,\n"
        "dead,6,Birth,1,Stays empty,0.95,0,lock,HighLife\n"
        "Alive,2,Survives,YES,Survives,1.00,0,model,\n"
        " alive , 3 ,Survives,x,,,,,\n"
        "alive,4,Dies,True,,,,,\n"
        "alive,5,Dies,no,,,,,\n"
        "alive,6,Dies,,,,,,\n",
    )
    locks = csvfmt.parse_overrides(doc, ["state", "neighbours"])
    expected = {
        f"dead{SEP}6": csvfmt.Lock("Birth", 3),
        f"alive{SEP}2": csvfmt.Lock("Survives", 4),
        f"alive{SEP}3": csvfmt.Lock("Survives", 5),
        f"alive{SEP}4": csvfmt.Lock("Dies", 6),
    }
    check("overrides: locked rows only, by their casefolded key cells", locks == expected, locks)
    check("overrides: lock_key() is that key, trimmed and ignoring case, for rows and table units alike",
          csvfmt.lock_key([" Dead ", "6"]) == f"dead{SEP}6" and csvfmt.lock_key(["Pan"]) == "pan")
    got = [csvfmt.number(cell) for cell in ("0.5", "-.5", "1e-3", "1_000", "\u0665", "nan", "inf")]
    check("overrides: a lock's number follows the rule of every numeric cell", got == [0.5, -0.5, 0.001] + [None] * 4,
          got)
    check(
        "overrides: key columns match ignoring case", csvfmt.parse_overrides(doc, ["State", "NEIGHBOURS"]) == expected
    )
    problems = raised(csvfmt.parse_overrides, doc, ["state", "neighbours", "season"])
    check(
        "overrides: a missing key column makes the whole file an error",
        only(problems, "error", 0, "No 'season' column"),
        summary(problems),
    )
    doc = read("twice_locked.csv", "item,answer,lock\nPan,yes,1\npan,no,x\nsofa,no,0\nSofa,yes,1\n")
    problems = raised(csvfmt.parse_overrides, doc, ["item"])
    message = "Line 3: 'pan' is locked twice (also line 2)"
    check("overrides: two locked rows with one key", only(problems, "error", 3, message), summary(problems))
    doc = read("ask_overrides.csv", "question,answer,lock\nIs it night?,yes,1\nIs it raining?,0.2,\n")
    locks = csvfmt.parse_overrides(doc, ["question"])
    check("overrides: one key column", locks == {"is it night?": csvfmt.Lock("yes", 2)}, locks)
    check(
        "overrides: no lock column, no locks",
        csvfmt.parse_overrides(read("no_lock.csv", "item,answer\npan,yes\n"), ["item"]) == {},
    )
    locks = csvfmt.parse_overrides(read("no_answer.csv", "item,lock\npan,1\n"), ["item"])
    check(
        "overrides: without an answer column a lock's answer is empty, for the node to flag",
        locks == {"pan": csvfmt.Lock("", 2)},
        locks,
    )


def test_hashes():
    with open(fixture("material_families.csv"), encoding="utf-8") as handle:  # universal new lines
        text = handle.read()
    base = read("hash_base.csv", text)
    edited = read("hash_numbers.csv", text.replace("0.35,1", "0.4,1"))
    check(
        "hashes: a numeric edit changes num_hash only",
        edited.text_hash == base.text_hash and edited.num_hash != base.num_hash,
    )
    edited = read("hash_retyped.csv", text.replace("0.35,1", "0.35,1.5"))
    good = (
        edited.num_cols["metallic"] == "FLOAT"
        and edited.text_hash == base.text_hash
        and edited.num_hash != base.num_hash
    )
    check("hashes: a numeric edit that retypes a column changes num_hash only", good, edited.num_cols)
    edited = read("hash_words.csv", text.replace("planks, bark", "boards, bark"))
    check(
        "hashes: a text edit changes text_hash only",
        edited.num_hash == base.num_hash and edited.text_hash != base.text_hash,
    )
    edited = read("hash_row.csv", text + "glass,clear panes,0.05,0\n")
    check("hashes: a new row changes both", edited.text_hash != base.text_hash and edited.num_hash != base.num_hash)
    same = read("hash_same.csv", text.replace("0.6,0", "0.60,0").replace("\n", "\r\n"))
    good = (same.text_hash, same.num_hash) == (base.text_hash, base.num_hash)
    check("hashes: another file with CRLF and 0.60 for 0.6 hashes the same", good)
    doc = csvfmt.read_csv(fixture("material_families.csv"))
    check(
        "hashes: the fixture, whatever its line ends, hashes like its copy",
        (doc.text_hash, doc.num_hash) == (base.text_hash, base.num_hash),
    )


def test_inline():
    parse = lists.parse_inline
    check(
        "inline: new lines first",
        parse("a; b\nc, d\r\ne: f\rg") == [("a; b", ""), ("c, d", ""), ("e", "f"), ("g", "")],
        parse("a; b\nc, d\r\ne: f\rg"),
    )
    pairs = parse("yes: it is cold, wet or windy; no: warm and calm")
    check("inline: then ';'", pairs == [("yes", "it is cold, wet or windy"), ("no", "warm and calm")], pairs)
    check(
        "inline: then ','",
        parse("chair, table,lamp") == [("chair", ""), ("table", ""), ("lamp", "")],
        parse("chair, table,lamp"),
    )
    pairs = parse("Pine: grows where it is cold: or mild")
    check(
        "inline: the first ': ' separates the name from its when",
        pairs == [("Pine", "grows where it is cold: or mild")],
        pairs,
    )
    check(
        "inline: ':' without a space belongs to the name",
        parse("12:30; a") == [("12:30", ""), ("a", "")],
        parse("12:30; a"),
    )
    check(
        "inline: entries trimmed, blank ones dropped",
        parse(" a ;; b ; ") == [("a", ""), ("b", "")],
        parse(" a ;; b ; "),
    )
    check("inline: empty text", parse("") == [] and parse("  \n ") == [])
    check(
        "inline: a lone trailing new line is not a separator",
        parse("a, b\n") == [("a", ""), ("b", "")],
        parse("a, b\n"),
    )
    for text, pairs in (
        ("a: x; b: ", [("a", "x"), ("b", "")]),
        ("b: ; a: x", [("b", ""), ("a", "x")]),
        ("a: x\nb: ", [("a", "x"), ("b", "")]),
        ("\n\nb: \na: x\n", [("b", ""), ("a", "x")]),
    ):
        check(
            f"inline: a trailing ': ' parses the same wherever the entry is ({text!r})",
            parse(text) == pairs,
            parse(text),
        )
    check("inline: lists offers csvfmt's parser", lists.parse_inline is csvfmt.parse_inline)
    error = csvfmt.CsvError([csvfmt.Problem("warning", "f.csv", 1, "", "just a warning")])
    check("CsvError: built without an error, it still builds", str(error) == "" and len(error.problems) == 1)
    paths = ("//life_outcomes.csv", "  C:\\data\\Life.CSV \n")
    check("csv path: a single line ending in .csv, any case", all(lists.is_csv_path(text) for text in paths))
    texts = ("", "a.csv\nb.csv", "life.csv.txt", "yes: cold; no: warm", "a, b")
    check("csv path: anything else is inline text", not any(lists.is_csv_path(text) for text in texts))


def test_inline_sets():
    parse = csvfmt.parse_inline_set
    answers = parse("yes: it is cold, wet or windy; no: warm and calm", "yes_no")
    pairs = [(o.name, o.when) for o in answers.options]
    check("inline set: Yes/No whens", pairs == [("yes", "it is cold, wet or windy"), ("no", "warm and calm")], pairs)
    check("inline set: no doc, no 'if', no payload", answers.doc is None and answers.options[0] == csvfmt.Option(
        "yes", "it is cold, wet or windy", "", {}))
    rubric = "none: bare ground; dusting: a thin white layer; patchy: snow in patches; thick: most ground covered; " \
             "buried: deep snow everywhere"
    levels = [o.name for o in parse(rubric, "amount").options]
    check("inline set: Amount levels, lowest first", levels == ["none", "dusting", "patchy", "thick", "buried"], levels)
    for text, kind, message in (
        ("yes: cold; maybe: unsure", "yes_no", "A Yes/No answer set needs rows 'yes' and 'no'"),
        ("low", "amount", "An Amount answer set needs 2 to 16 levels; this one has 1"),
        (",".join(f"l{i}" for i in range(17)), "amount", "An Amount answer set needs 2 to 16 levels; this one has 17"),
        (",".join(f"o{i}" for i in range(256)), "one_of", "A One-of answer set needs 1 to 255 answers; this one has 256"),
        ("a; b; A", "one_of", "Entry 3: 'A' appears twice"),
        (": only a description", "one_of", "Entry 1: the name is empty"),
    ):
        problems = raised(parse, text, kind)
        check(f"inline set: {message}", only(problems, "error", 0, message), summary(problems))
    check("inline set: one option is fine for One-of", len(parse("only", "one_of").options) == 1)
    items = [o.name for o in parse("chair, table,lamp", "items").options]
    check("inline set: items", items == ["chair", "table", "lamp"], items)


def test_answer_set_helpers():
    yes_no = csvfmt.parse_inline_set("no: warm and calm; yes: it is cold", "yes_no")
    got = sources.criteria(yes_no, "yes_no")
    check("criteria: Yes/No whens become noul true/false", got == {"true": "it is cold", "false": "warm and calm"}, got)
    got = sources.criteria(sources.default_set("yes_no"), "yes_no")
    check("criteria: an empty Yes/No set asks 'Yes.'/'No.'", got == {"true": "Yes.", "false": "No."}, got)
    got = sources.criteria(csvfmt.parse_inline_set("yes: cold; no", "yes_no"), "yes_no")
    check("criteria: a side without a when says 'No.'", got == {"true": "cold", "false": "No."}, got)
    levels = csvfmt.parse_inline_set("none: bare ground; dusting; buried: deep snow", "amount")
    got = sources.question("amount", "How much snow?", levels)
    check("criteria: Amount levels as 'name: when', lowest first",
          got == {"type": "score", "instructions": "How much snow?",
                  "criteria": ["none: bare ground", "dusting", "buried: deep snow"]}, got)
    got = sources.criteria(sources.default_set("amount"), "amount")
    check("criteria: an empty Amount set gives the five levels", got == list(sources.AMOUNT_LEVELS), got)
    labels = csvfmt.parse_answer_set(csvfmt.read_csv(fixture("material_families.csv")), "one_of", node="classify")
    got = sources.criteria(labels, "one_of")
    check("criteria: One-of options keyed by row index",
          list(got) == ["0", "1", "2", "3"] and got["1"] == "wood: planks, bark and furniture timber", got)
    check("default: One-of has none", sources.default_set("one_of") is None)
    values = [sources.amount_value(score, levels) for score, levels in ((2, 5), (1, 3), (4, 5), (-0.1, 5), (1, 2))]
    check("amount: score / (L - 1), clamped", values == [0.5, 0.5, 1.0, 0.0, 1.0], values)
    check("decided: a single option only", sources.decided(csvfmt.parse_inline_set("only", "one_of"))
          and not sources.decided(labels))
    costs = [pipeline.cost(n, debias) for n, debias in ((2, True), (5, False), (16, True), (17, True), (20, True),
                                                         (255, True))]
    check("cost: 2 x (groups + 1) above 16 options", costs == [2, 1, 2, 6, 6, 34], costs)


def test_per_item_cells():
    Item = lists.Item
    items = [Item("pan", "a flat metal pan"), Item("", offered=False), Item("sofa")]
    yes_no = sources.default_set("yes_no")
    neutral = {"p": 0.0, "c": 0.0}
    rows = sources.per_item_cells("match", 'YES_NO', items, yes_no, " Would you find it in a kitchen? ", "", True,
                                  neutral)
    check("cells: one row per item, the empty slot a code row",
          [(r.name, r.source, r.cell is None) for r in rows] == [("pan", "pending", False), ("", "code", True),
                                                                 ("sofa", "pending", False)])
    pan, slot, sofa = rows
    check("cells: Match template", pan.cell.question == {
        "type": "noul", "instructions": "Item: pan — a flat metal pan\nWould you find it in a kitchen?",
        "criteria": {"true": "Yes.", "false": "No."}} and sofa.cell.question["instructions"] == (
        "Item: sofa\nWould you find it in a kitchen?"), pan.cell.question)
    check("cells: one shared state", pan.cell.state == sofa.cell.state == NO_CONTEXT, (pan.cell.state, sofa.cell.state))
    check("cells: an empty slot is never asked and holds the neutral values", slot.values == neutral
          and slot.values is not neutral, slot)
    check("cells: cell keys and wkeys differ per item", pan.cell.key != sofa.cell.key and pan.wkey != sofa.wkey)
    check("cells: debias doubles the cost", (pan.cell.cost, sofa.cell.cost) == (2, 2), (pan.cell.cost, sofa.cell.cost))
    check("cells: pipeline's keys and cost, the options in the order sent",
          pan.wkey == pipeline.wkey("match", 'YES_NO', ["", "Would you find it in a kitchen?"],
                                    ["pan", "a flat metal pan"], [["yes", ""], ["no", ""]])
          and pan.cell.key == pipeline.cell_key("noul", NO_CONTEXT, pan.cell.question["instructions"],
                                                [("true", "Yes."), ("false", "No.")], True)
          and pan.cell.cost == pipeline.cost(2, True))
    edited = sources.per_item_cells("match", 'YES_NO', [Item("pan", "a heavy pan")] + items[1:], yes_no,
                                    "Would you find it in a kitchen?", "", False)
    check("cells: editing one item's when changes only its wkey",
          edited[0].wkey != pan.wkey and edited[2].wkey == sofa.wkey and edited[2].cell.key != sofa.cell.key)
    labels = csvfmt.parse_inline_set("kitchen: cooking; office: work and paper", "one_of")
    rows = sources.per_item_cells("classify", 'ONE_OF', items, labels, "", "a house", True)
    check("cells: Classify's default question and its choice over the labels", rows[0].cell.question == {
        "type": "choice", "instructions": "Item: pan — a flat metal pan\nWhich label fits this item best?",
        "criteria": {"0": "kitchen: cooking", "1": "office: work and paper"}} and rows[0].cell.state == "a house",
        rows[0].cell.question)
    rows = sources.per_item_cells("classify", 'ONE_OF', items, csvfmt.parse_inline_set("only", "one_of"), "", "", True)
    check("cells: a single label is decided by code for every item",
          [(r.source, r.values, r.cell) for r in rows] == [("code", {"label": 0, "c": 1.0}, None), ("code", {}, None),
                                                           ("code", {"label": 0, "c": 1.0}, None)])
    levels = sources.default_set("amount")
    rows = sources.per_item_cells("match", 'AMOUNT', items[:1], levels, "How heavy?", "", True)
    check("cells: Amount asks a score over the levels",
          rows[0].cell.question["type"] == "score" and len(rows[0].cell.question["criteria"]) == len(levels.options))


def test_dials_cells():
    """Dials asks each setting as Ask asks its answer set (sources.question()), so equal questions share a cell. And the
    warning for contexts whose settings differ names both, and what only one of them has."""
    from openact.specs.ask import AskSpec
    from openact.specs.dials import DialsSpec, differ

    storm = "A stormy night in a fishing village"
    settings = csvfmt.parse_settings(csvfmt.read_csv(fixture("weather_dials.csv"), schema="settings"))
    job = DialsSpec().job((storm, "weather_dials.csv", "", ""), settings, True)
    rain, wind, time_, street, fog = (row.cell.question for row in job.rows)
    check("dials cells: yes/no asks 'Yes.'/'No.' without whens, and its whens with them",
          rain.get("criteria") == {"true": "Yes.", "false": "No."}
          and street["criteria"] == {"true": "it is dark or foggy", "false": "there is enough daylight"}, (rain, street))
    check("dials cells: amount scores its levels, else the five default ones; one-of chooses by row index",
          wind["criteria"][:2] == ["calm: still air", "breeze: leaves move"] and len(wind["criteria"]) == 5
          and fog["criteria"] == list(sources.AMOUNT_LEVELS) and time_["type"] == "choice"
          and time_["criteria"] == {"0": "dawn: first light", "1": "day: the sun is high", "2": "dusk: evening glow",
                                    "3": "night: dark sky"}, (wind, fog, time_))
    ask = AskSpec().job(("Is it raining?", storm, "", "", ""), 'YES_NO', True)
    check("dials cells: a yes/no setting and an Ask with the same question and situation share one cell",
          job.rows[0].cell.key == ask.rows[0].cell.key, (job.rows[0].cell.key, ask.rows[0].cell.key))
    check("dials cells: one state, 2 scored rows each with order balancing, wkeys without the name or range",
          {row.cell.state for row in job.rows} == {storm} and {row.cell.cost for row in job.rows} == {2}
          and job.rows[4].wkey == pipeline.wkey("dials", "amount", [storm], ["How foggy is it?"], []),
          [(row.cell.state, row.cell.cost) for row in job.rows])
    wet, snow, when = ("Wet", "yes_no"), ("Snow", "amount"), ("When", "one_of")
    got = [differ("A > Dials", [wet, when], "B > Dials", theirs) for theirs in ([wet, snow], [wet, when, snow],
                                                                                [when, wet])]
    tail = "; setting names and types must agree"
    check("dials cells: the disagreement warning names both contexts, and what only one of them has", got == [
        f"Dials: 'A > Dials' and 'B > Dials' use different settings ('When' (one-of) vs 'Snow' (amount)){tail}",
        f"Dials: 'A > Dials' and 'B > Dials' use different settings (only 'B > Dials' has 'Snow' (amount)){tail}",
        f"Dials: 'A > Dials' and 'B > Dials' use different settings (the same settings in another order){tail}"], got)


def test_cache():
    path = write("cached.csv", "answer,when,scale\npan,cooking,1\nsofa,sitting,2\n")
    first = csvfmt.load(path)
    check("cache: a second read of an unchanged file is the same document", csvfmt.load(path) is first)
    spelled = os.path.join(TMP, "..", os.path.basename(TMP), "cached.csv")
    check("cache: keyed by file identity, not spelling", csvfmt.load(spelled) is first)
    factors = csvfmt.load(path, schema="factors")
    check("cache: keyed by schema too", factors is not first and factors.schema == "factors"
          and first.num_cols == {"scale": "INT"} and factors.num_cols == {}, factors.num_cols)
    check("cache: each schema keeps its own document", csvfmt.load(path, schema="factors") is factors
          and csvfmt.load(path) is first)
    other = csvfmt.load(path, delimiter=";")
    check("cache: and by delimiter", other is not first and other.header == ("answer,when,scale",), other.header)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("lamp,lighting,3\n")
    edited = csvfmt.load(path)
    check("cache: an edit (new stamp) is read again", edited is not first and len(edited.rows) == 3, len(edited.rows))
    csvfmt.clear_cache(path)
    check("cache: clear_cache() forgets the file", csvfmt.load(path) is not edited)
    os.remove(path)
    gone = csvfmt.load(path)
    check("cache: a missing file is read each time, with its problem",
          gone.problems[0].message.startswith("Can't find") and csvfmt.load(path) is not gone, summary(gone.problems))
    check("identity: normcase(realpath)", csvfmt.identity(spelled) == os.path.normcase(os.path.realpath(path)))


test_read()
test_dialect()
test_classes()
test_limits()
test_messages()
test_answer_sets()
test_factors()
test_if()
test_settings()
test_inline_settings()
test_overrides()
test_hashes()
test_inline()
test_inline_sets()
test_answer_set_helpers()
test_per_item_cells()
test_dials_cells()
test_cache()
shutil.rmtree(TMP, ignore_errors=True)
finish()
