"""csvfmt and lists, with no Blender and no model.

Checks the CSV dialect (BOM, CRLF and CR, ";" and tab delimiters, quoting), the INT/FLOAT/text split, the limits,
every design 7.9 message that the reader and the parsers give (one broken fixture each), the Answers, Factors,
Settings and Overrides parsers, the change hashes, and the inline-list rules.

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
from openact import csvfmt, lists  # noqa: E402

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

    doc = read("semicolon.csv", "answer;when;scale\nmetal;steel, iron;1,5\nwood;planks;0,6\n")
    check("dialect: ';' when the header line has no ','", doc.rows[0] == ("metal", "steel, iron", "1,5"), doc.rows)
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

    # Neither factor + kind + level (a Factors file) nor setting + type + question (a Settings file).
    names = "value,level,type,min,max,kind,from,to,question,answers,prompt"
    doc = read("other_schemas.csv", f"answer,{names}\na,{','.join(['1'] * 11)}\n")
    check(
        "classes: an answer set may use other schemas' column names as payload",
        list(doc.num_cols) == names.split(","),
        doc.num_cols,
    )
    for name, columns in (
        ("life_factors.csv", ("value",)),
        ("biome_factors.csv", ("from", "to")),
        ("weather_dials.csv", ("min", "max")),
    ):
        doc = csvfmt.read_csv(fixture(name))
        good = set(columns) <= set(doc.text_cols) and not doc.num_cols and not doc.problems
        check(
            f"classes: {name} keeps its own columns as text, blanks and all",
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
    message = "Line 2: a cell has over 131 072 characters (limit 2 000)"
    check(
        "limits: a cell over the csv module's own cap", only(doc.problems, "error", 2, message), summary(doc.problems)
    )

    doc = read("rows_many.csv", "answer\n" + lines(6000, lambda i: f"r{i}"))
    good = only(doc.problems, "error", 0, "rows_many.csv has 6 000 rows (limit 4 096)") and len(doc.rows) == 4096
    check("limits: rows past the limit are counted, not kept", good, (len(doc.rows), summary(doc.problems)))
    doc = read("long_rows.csv", "answer\n" + lines(60, lambda i: f"r{i},extra"))
    listed = summary(doc.problems)
    good = len(listed) == 51 and listed[0] == ("error", 2, "Line 2 has 2 cells; the header has 1")
    check(
        "limits: at most 50 long rows are listed",
        good and listed[-1] == ("error", 0, "10 more rows have more cells than the header"),
        listed[-2:],
    )
    doc = read("long_cells.csv", "answer,when\n" + lines(55, lambda i: f"r{i}," + "w" * 2001))
    listed = summary(doc.problems)
    good = len(listed) == 51 and listed[-1] == ("error", 0, "5 more cells have over 2 000 characters")
    check("limits: at most 50 long cells are listed", good, listed[-2:])


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

    problems = raised(csvfmt.parse_factors, csvfmt.read_csv(fixture("broken_bands.csv")))
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
    try:
        csvfmt.parse_answer_set(life.doc, "one-of")
        unknown = False
    except ValueError:
        unknown = True
    check("answers: an unknown kind is a programming error", unknown)


def test_factors():
    life = csvfmt.parse_factors(csvfmt.read_csv(fixture("life_factors.csv")))
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
    biome = csvfmt.parse_factors(csvfmt.read_csv(fixture("biome_factors.csv")))
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
        levels = csvfmt.parse_factors(read(f"category_{name}.csv", FACTORS + body))[0].levels
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
        problems = raised(csvfmt.parse_factors, read(f"factors_{case}.csv", FACTORS + body))
        check(f"factors: {case}", only(problems, "error", line, message), summary(problems))
    problems = raised(csvfmt.parse_factors, read("factors_columns.csv", "factor,level\na,x\n"))
    check("factors: a required column is missing", only(problems, "error", 0, "No 'kind' column"), summary(problems))
    name = "\u4e00" * 21
    factors = csvfmt.parse_factors(read("factors_wide_ok.csv", FACTORS + flag(name)))
    check("factors: a name of exactly 63 bytes is fine", factors[0].name == name)


def test_settings():
    weather = csvfmt.parse_settings(csvfmt.read_csv(fixture("weather_dials.csv")))
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
    types_ = [s.type for s in csvfmt.parse_settings(read("settings_types.csv", SETTINGS + body))]
    check("settings: type synonyms", types_ == ["yes_no", "yes_no", "one_of", "one_of", "yes_no", "amount"], types_)
    inverted = csvfmt.parse_settings(read("settings_inverted.csv", SETTINGS + "W,amount,q?,,30,0\nR,yes/no,q?,,abc,\n"))
    good = (inverted[0].min, inverted[0].max) == (30.0, 0.0) and (inverted[1].min, inverted[1].max) == (0.0, 1.0)
    check("settings: min above max is allowed; min and max only count for amount", good, inverted)

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
        ("many", lines(65, lambda i: f"s{i},yes/no,q?,,,"), 0, "settings_many.csv has 65 settings (limit 64)"),
        ("none", "", 0, "No settings"),
    ):
        problems = raised(csvfmt.parse_settings, read(f"settings_{case}.csv", SETTINGS + body))
        check(f"settings: {case}", only(problems, "error", line, message), summary(problems))
    problems = raised(csvfmt.parse_settings, read("settings_columns.csv", "setting,type\nA,bool\n"))
    check(
        "settings: a required column is missing", only(problems, "error", 0, "No 'question' column"), summary(problems)
    )


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


test_read()
test_dialect()
test_classes()
test_limits()
test_messages()
test_answer_sets()
test_factors()
test_settings()
test_overrides()
test_hashes()
test_inline()
shutil.rmtree(TMP, ignore_errors=True)
finish()
