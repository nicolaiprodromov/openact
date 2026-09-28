# OpenAct

Decision nodes for Geometry Nodes, run by a local model.

OpenAct adds six Geometry Nodes that make judgments. A question, a prompt, a list of items or a table of rules goes
in; numbers, selections, picks and labels come out. The nodes read their words (typed on the node, set on the
modifier panel, or kept in CSV files) outside evaluation, ask a model on your GPU in the background, and store its
answers in the node as a table that Geometry Nodes looks up, per element where needed. The model runs only when the
words change, never during evaluation, playback or rendering. The answers are saved in the .blend, so a file opens,
plays and renders without the model, and without the add-on, on Blender 4.5 or newer.

The model never generates text. It reads the probability of each allowed answer from one forward pass. That is the
method of **SemIf** (formerly OpenJev, MIT), running **Qwen3.5-4B** (Apache-2.0) on your GPU. After the one-time
install, OpenAct talks only to its model server on 127.0.0.1: nothing leaves your computer.

## Requirements

- Blender 4.5 or newer. OpenAct 1.0 is tested on 4.5, 5.0 and 5.1.
- Windows 10 or 11 with an NVIDIA GPU (CUDA) with 10 GB of VRAM or more; the model uses about 9 GB. Linux with an
  NVIDIA GPU is best effort and untested.
- 15 GB of free disk space for the one-time install of the model runtime, which takes about 13 GB.

Only authoring needs the GPU and the runtime: a file that holds its answers renders wherever Blender 4.5 or newer
runs, with or without OpenAct.

## Install

1. Download `openact-1.0.0.zip` from the GitHub release, or build it from this repository:
   `blender --command extension build --source-dir openact --output-dir dist`.
2. In Blender: *Edit > Preferences > Get Extensions > Install from Disk...* and pick the zip.
3. Open a Geometry Node editor, press **N** and go to the **OpenAct** tab.
   - If it says *Model not installed*, click **Install Model** (below), or **Locate Existing Runtime...** to use a
     runtime already on this computer.
   - If it says *Disable Jev Nodes*, disable that add-on: OpenAct replaces it.
4. Add nodes with **Shift A > OpenAct**.

The model server starts when a node first needs an answer (the *Start Automatically* preference) and loads the model
onto the GPU in about 12-40 s. The sidebar's Model box shows its status, with *Start Model* / *Stop Model*, *Open
Log*, and *Copy Diagnostics*, which copies a report for bug reports without your prompts, CSV text or paths. Another
Blender on the same port uses the running server; there the button reads *Detach*.

### The model runtime (13 GB)

**Install Model** downloads about 12 GB once: uv, Python 3.12, PyTorch for CUDA 12.8, SemIf and the model. It
installs them into `%LOCALAPPDATA%\OpenAct` (on Linux `$XDG_DATA_HOME/OpenAct`, or `~/.local/share/OpenAct`), where
they take about 13 GB. It needs *Preferences > System > Network > Allow Online Access*, 15 GB of free disk space and
an NVIDIA GPU. uv, SemIf and the Python packages are pinned and hash-checked, and the model is fetched at a fixed
revision. The Model box shows the progress, and *Open Log* opens `install.log`.

**A Jev Nodes runtime is reused.** The runtime that Jev Nodes installed in `%LOCALAPPDATA%\JevNodes` is found and
used in place, so nothing is downloaded, and the Model box reads *Using the Jev Nodes runtime at ...*. OpenAct never
moves or changes that folder: its Python environment holds absolute paths. For a runtime in another folder, use
*Locate Existing Runtime...* or the *Runtime Folder* preference. If Jev Nodes is enabled the first time OpenAct is,
OpenAct takes over its settings once: runtime folder, port, device, GPU, *Start Automatically* and order balancing.

**Removing it.** *Open Runtime Folder* shows the folder; deleting it uninstalls the model. Uninstalling OpenAct leaves
it in place. Once OpenAct has its own runtime, you can delete `JevNodes`. A Jev Nodes runtime may use a Python from
your global uv folder (the Model box names it: *Python from ...*), which other tools may share: remove that one only
with `uv python uninstall`, and only if nothing else uses it.

### Coming from Jev Nodes

OpenAct grew from the Jev Nodes 0.1.0 prototype. With OpenAct installed, files made with Jev Nodes open with their
old nodes frozen: they keep their saved answers and are never asked. The sidebar then offers **Upgrade**: in one
undo step, every old Ask and Pick becomes its OpenAct node with its links, settings and answers, so the upgraded file
asks nothing until its words change. A node that can't be upgraded stays frozen, and the report names it. On Blender
4.2, keep using Jev Nodes 0.1.0.

## The six nodes

| Node | Judges | Answer | Words in | Outputs |
|---|---|---|---|---|
| **Ask** | the whole situation | Yes/No or Amount | Question, Context, Answers (optional) | **Answer** 0-1, **Yes**, **Confidence** |
| **Pick** | the whole situation | one entry of a list | Prompt, Context, the list | **Index**, **Item**, **Confidence**; per element **Selection** and **Probability**; the picked **Instance** (Collection) or **Material** (Materials), or the Items CSV's payload |
| **Dials** | the whole situation | per setting: Yes/No, Amount or One of | Prompt, Settings | one output per setting; **Confidence**, the lowest of theirs |
| **Match** | each item of a list | Yes/No or Amount | Question, Context, the list, Answers (optional) | per element **Probability**, **Selection** (Probability ≥ *Threshold*), **Confidence** |
| **Classify** | each item of a list | one of several labels | Question (optional), Context, the list, Labels | per element **Label** (its row), **Confidence**, the Labels CSV's payload |
| **Rules** | each measured situation | One of, Yes/No or Amount | Question, Context, Factors, Outcomes, and one field input per factor | per element: One of gives **Outcome** (its row) and the Outcomes CSV's payload, Yes/No **Probability** and **Yes**, Amount **Amount**; and **Confidence** |

Every node also has an **Overrides** input, a CSV that locks answers (below), and a hidden **Key**.

- **Words.** The text inputs are *words*: typed on the node, or linked from the modifier panel (Group Input), String,
  Join Strings, Menu Switch, Import Text, or another OpenAct node's single-value output. OpenAct reads them outside
  evaluation, so they must be single values: a field linked into a word is an error on the node.
- **One table per set of words.** A tree used by several objects, with words from each modifier, keeps one set of
  answers per object. The model is asked only when words change, and only for the rows whose words changed.
- **Lists** (Pick, Match, Classify) come from the node's dropdown: **Text or CSV** (inline items or a `.csv` path),
  **Collection** (its children, in the order Collection Info gives them with *Separate Children*), **Materials** (an
  object's material slots, so ID = material index) or **Attributes** (the attribute names of an object's own mesh,
  curve or point cloud data, not of the evaluated result).
- **ID.** Entry *i* of a list is ID *i*. Unlinked, the ID is the element's index; link *Material Index*, a Named
  Attribute or anything else that numbers the entries. Before the first answer, and for IDs outside the list, the
  outputs give 0, False, -1 or nothing.
- **Confidence** runs from 0 (no information) to 1 (certain). Answers below *Unsure Below* (0.30, in the preferences)
  are flagged as unsure in the node's status, the sidebar's table and an exported Overrides CSV.
- **Balance Option Order**, on each node and on by default, asks every question with its options in both orders,
  which removes the small model's bias toward early options, at twice the cost. Changing it keeps the answers given
  so far. The preference of the same name sets it for new nodes.
- **Key.** Objects whose words are identical but whose Collection or Object differ can't be told apart: the node
  says only one answer can show. Show the hidden **Key** input (Ctrl H), expose it on the modifier and give each
  object its own value. The model never sees it.
- **Sidebar** (N > OpenAct): the Model box; for the active node, its status, a summary per object, the table of
  answers with confidence and source (*Show Question* shows exactly what a row asks the model), **Ask Again** and
  **Write Overrides CSV**; and the *Files* panel: each CSV the node reads, with its columns and problems, *Open File*
  and *New CSV from Template*. The sidebar only inspects: every change goes through a socket or a file.
- **Animated words.** OpenAct asks nothing during playback or rendering. When words change over the frame range (a
  keyframed dropdown, say), run **Prepare for Playback** in the sidebar first: it answers every set of words the
  frames show and pins those answers in the file. *Unpin* in the node's panel releases them.

## CSV files

Answer sets, item lists, Rules factors, Dials settings and overrides can be typed inline or kept in CSV files next to
the .blend. OpenAct parses every column itself. Text columns (names, `when` descriptions, `if` restrictions) go to
the model. Numeric columns are *payload*: each becomes an output socket, and its values are baked into the node's
table. The model never sees the numbers, and a saved file needs no CSV to render.

### Two ways to give a node its CSV

Every input that takes a CSV has a Geometry input named `<input> CSV` below it, e.g. *Outcomes* and *Outcomes CSV*.

1. **A typed path** in the String input, e.g. `//life_outcomes.csv`. Relative paths, with or without `//`, are
   relative to the .blend (for a linked node group, to its library file). A path is a word, so it can come from the
   modifier panel, Join Strings or a Menu Switch: a dropdown on the modifier can choose the file.
2. **An Import CSV node** plugged into the `<input> CSV` socket. OpenAct reads that node's *Path* and *Delimiter* and
   parses the file itself; it never uses the Import CSV geometry. It follows Reroutes, node groups, and Switch, Index
   Switch and Menu Switch nodes whose selector it can read. Any other node in between is an error ("connect the
   Import CSV node directly"). A link wins over a typed path, which is then ignored with a warning.

OpenAct can't tell objects apart by a path it traced to an Import CSV node: objects that need different files use the
typed path, or different Keys.

Either way, OpenAct watches the files it reads. After you save an edit, only the rows whose text changed are asked
again, and an edit of payload values only updates the outputs. A file with an error leaves the node's last good
answers in place and shows the error with its line. OpenAct never edits your CSV files: it writes one only when you
click *Write Overrides CSV* or *New CSV from Template*.

In a .blend that isn't trusted, OpenAct reads files only inside the .blend's own folder and its subfolders, because it
stores what it reads in the file. The sidebar then offers *Allow This File* and *Trust This Folder*. A file created in
this session is trusted.

### Formats

Every CSV file is UTF-8 (a BOM is fine) with a header row. The delimiter is `,`, or `;` or a tab when the header has
no comma, as Excel writes in some locales; with a plugged-in Import CSV node, it is that node's *Delimiter*. Cells
may be quoted with `"`. Limits: 1 MB, 4096 data rows, 64 columns and 2000 characters per cell. The column names below
are matched ignoring case.

**Answer sets and item lists**: Answers (Ask, Match), Labels (Classify), Outcomes (Rules) and Items (Pick, Match,
Classify).

| Column | Meaning |
|---|---|
| `answer` (or `name`, `item`, `label`, `outcome`) | The option or item name: unique, at most 200 characters |
| `when` (or `description`) | When this answer applies, or what the item is. Recommended: it makes the answers much more reliable |
| `if` | Rules Outcomes only: the cells that may choose this outcome (below) |
| payload columns | Pick's Items, Classify's Labels and One-of Rules Outcomes: one output each, read at the chosen row; at most 16 |

A payload column holds only integers (an Int output) or only numbers (a Float output), and its header becomes the
output's name. An empty cell, a decimal comma, `inf` or `nan` makes it text, with a warning. Other text columns are
ignored.

Row *r* is option *r*: Pick's Index, Classify's Label and Rules' Outcome give the row, which is also Import CSV's
point *r*. A Yes/No set has exactly the rows `yes` and `no`; an Amount set has 2-16 levels, lowest first (by default
Not at all, Slightly, Moderately, Very, Extremely); a One-of set has 1-255 options, and a single one is chosen without
asking. Lists hold up to 255 items.

Typed inline, a set is `name: when` entries separated by new lines, else by `;`, else by `,`, e.g.
`yes: it is cold, wet or windy; no: warm and calm`. Inline sets have no `if` and no payload.

```
answer,when,roughness,metallic
metal,"steel, iron, copper, brass or aluminium surfaces",0.35,1
wood,"planks, bark and furniture timber",0.6,0
```

**Factors CSV** (Rules), one row per level:

| Column | Meaning |
|---|---|
| `factor` | The factor's name, which becomes an input socket (at most 63 bytes of UTF-8) |
| `kind` | `amount` (a Float 0-1 in 2-16 bands), `category` (an Int with 2-64 levels) or `flag` (a Bool with 2 levels) |
| `level` | The level's name, as the model sees it |
| `when` (or `description`) | What the level means |
| `value` | `category` and `flag`: the integer the socket carries for this level (a flag's are 0 and 1) |
| `from`, `to` | `amount`, optional: the band's edges within 0-1; equal bands by default |

A Rules node has 1-8 factors and at most 4096 cells, the product of their level counts. It asks one question per
cell, and each element reads its cell through the factor inputs. An Outcomes `if` names the cells an outcome applies
to: `factor=level` or `factor!=level` clauses, `|` between alternative levels and `;` between clauses that must all
hold, e.g. `state=alive; neighbours=2|3`. A cell that allows one outcome gets it without asking, and a cell that
allows none is an error. Yes/No and Amount blend between the band centres of up to 3 Amount factors; One of can mix
outcomes near band edges with *Dither* and *Seed*.

```
factor,kind,level,value,when
state,flag,dead,0,The cell is empty.
state,flag,alive,1,The cell holds a living cell.
neighbours,category,0,0,No live neighbours.
neighbours,category,1,1,Exactly one live neighbour.
```

**Settings CSV** (Dials), one row per setting, each of which becomes an output:

| Column | Meaning |
|---|---|
| `setting` | The output's name |
| `type` | `yes/no` (a Bool output), `amount` (Float) or `one-of` (Int: the chosen option's position) |
| `question` | What to ask about the Prompt |
| `answers` | An inline answer set: required for `one-of` (2-255 options), optional otherwise |
| `min`, `max` | `amount` only: the output's range, 0 and 1 by default |

A Dials node has at most 32 settings. Typed inline they are `name (type): question` entries, yes/no and amount
only, e.g. `Rain (yes/no): Is it raining?; Wind (amount): How windy is it?`.

**Overrides CSV** (every node). A row whose `lock` is `1` (or `yes`, `true`, `x`) replaces the model's answer, and
that row is never asked.

| Node | Key columns | `answer` |
|---|---|---|
| Ask | `question` | Yes/No: `yes`, `no` or 0-1. Amount: 0-1 or a level name |
| Pick | `prompt` | An item name |
| Match | `item` | As for Ask |
| Classify | `item` | A label name |
| Rules | one column per factor, holding level names | One of: an outcome name. Yes/No and Amount: as for Ask |
| Dials | `setting` | `yes` or `no`, an amount in the output's range, or an option name |

*Write Overrides CSV* writes the node's answers with `lock` 0 and the columns `model_answer`, `confidence`, `unsure`
and `source`, which are for reading only: set `lock` to 1 on the rows to keep. With an empty Overrides input it asks
where to write the file, then fills the input. It merges into an existing file without losing a row. One file applies
to every object that uses the node; for overrides per object, give each object its own file through the modifier.

### Templates

*New CSV from Template* (the sidebar's *Files* panel) copies a node's starter files, from `openact/templates/`, next
to the saved .blend, never over an existing file (`items.csv` becomes `items 2.csv`), and puts each copy's path in
its input when that input is empty and unlinked.

| Node | Starter files |
|---|---|
| Rules | `life_factors.csv` and `life_outcomes.csv`: Conway's Game of Life |
| Classify | `items.csv`, and `labels.csv`: material families with roughness and metallic |
| Pick, Match | `items.csv`: household objects with descriptions and a `scale` column |
| Dials | `weather_dials.csv`: rain, wind speed, time of day, street lights and fog |

## Render farms and headless scripts

In background mode (`blender -b`, e.g. a render farm's `blender -b scene.blend -f 1`), OpenAct renders the
answers and CSV payload saved in the file: it doesn't plan again or re-bake from the CSV files. For each frame it
renders, it prints once per file each CSV whose edits would change a saved table, e.g.
`CSV //life_outcomes.csv differs from the saved table (numbers); rendering the saved answers`. Words that only a
frame change reveals (a driver, NLA, a keyframed menu socket) are compared as the file holds them; the render shows
the saved answers either way. A render job in an open Blender leaves the tables as they are too. Opened
interactively, the file re-bakes from its CSVs as usual, and the node's status notes each file that differed.

A script that authors headlessly (sets words, fills tables, saves) opts back in, either way:
- set the environment variable `OPENACT_BACKGROUND_PASSES=1` (or `true`) before starting Blender;
- or, after enabling the add-on, call `allow_background_passes(True)` in its `engine` module. Installed from disk
  as above, that is `import bl_ext.user_default.openact.engine as engine`; from another extension repository, use
  its name instead of `user_default`.

Blender runs no timers while a `--python` script runs, so the script then calls `engine.tick()` every few
hundredths of a second until its nodes are answered, and saves, as `drive()` in `tests/_harness.py` does. The test
suites set the variable (`tests/_harness.py`).

## Limitations and known issues

- **Blender's Import CSV cache affects your own Import CSV nodes.** Blender reads each Import CSV path once per
  session. If one of your own Import CSV nodes (not one plugged into an OpenAct node) reads a file that an OpenAct
  node also reads, it keeps the old rows after you edit the file, while OpenAct's outputs update. Use the OpenAct
  node's payload outputs instead, or restart Blender. The sidebar notes each such node. OpenAct's own reading isn't
  affected, whichever way it gets the file.
- **Blender 4.5 or newer only.** On Blender 4.2, keep using Jev Nodes 0.1.0.
- **NVIDIA GPUs (CUDA) only.** The installer refuses to install without one, and there is no CPU backend yet (see
  Future work). macOS is not supported for authoring. Files that hold their answers still render anywhere.
- **At most 16 sets of words per node.** A tree used by more than 16 objects with different words is an error on the
  node: split the tree.
- **Blender 4.5 and 5.0 (Python 3.11):** Blender can crash when an operator runs under a context
  override while one of OpenAct's background threads briefly runs Python (a Blender bug). OpenAct
  makes no such call, but Blender does (the node editor's viewer shortcuts Ctrl+1…9 and 1…9, Quick
  Effects), and so may other add-ons. It is rare. Blender 5.1 is not affected.

## Future work

Neither of these is built or scheduled; both were left out of 1.0 by the maintainer's decision.

- **A CPU backend** through llama.cpp (a GGUF Q4_K_M model of about 3 GB), for machines without an NVIDIA GPU. It
  would plug in behind the same model server interface, and would be offered only once it passes the same acceptance
  tests.
- **A high-accuracy model**: SemIf's 27B "bridge" (EXL3, 5 bits per weight). On SemIf's benchmark of 144 authored
  decisions its balanced accuracy is 0.958, against 0.813 for the 4B model OpenAct uses. It needs a 24 GB GPU.

## Repository

- `openact/`: the add-on (extension) package.
- `tests/`: headless Blender tests.
- `demos/`: the demo scenes' build script and CSV files.
- `docs/DESIGN.md`: the v1 design.
- `tools/make_locks.ps1`: regenerates the runtime's hash-locked package list.

## Tests

```
tests\run_all.ps1 [-Model] [-Versions 4.5,5.0,5.1] [-Gpu 0] [-Port 18765] [-Threads 4]
```

Every suite runs at BelowNormal priority with `OMP_NUM_THREADS`/`MKL_NUM_THREADS` set to `-Threads`,
and every Blender with `-t <Threads>`, so its `--bench` and timing figures can differ from a single
suite run at full speed.

The `*_blender.py` suites run on each `-Versions` entry. The `*_test.py` suites run under every
CPython 3.11+ found, and always inside Blender 4.5 and 5.1. `-Model` adds the e2e phases with the
real model on the given GPU, once it has 9 GB free. The port defaults to 18765 on GPU 0 and 18767
on GPU 1, and must be free: close any Blender session running OpenAct or Jev Nodes, or pass
`-Port`. The runner prints a summary table and exits with 1 if anything failed.

To run a single suite:

```
blender -b --factory-startup --python tests/evaluator_blender.py
blender -b --factory-startup --python tests/legacy_blender.py
blender -b --factory-startup --python tests/e2e_blender.py -- build   <file.blend> [port] [gpu]
blender -b --factory-startup --python tests/e2e_blender.py -- noaddon <file.blend>
blender -b --factory-startup --python tests/e2e_blender.py -- reopen  <file.blend>
blender -b --factory-startup --python tests/e2e_blender.py -- reopen-bumped <file.blend>
```

The e2e phases drive the add-on with the real model. `tests/migrate_blender.py` builds its v0.1.0
file with `tests/fixtures/jev_nodes-0.1.0.zip`, the release of the Jev Nodes prototype that OpenAct
grew from (same author, GPL-3.0-or-later), so it needs nothing outside the repository.

## Demos

`demos/build_demos.py` builds the six demo scenes headlessly, one `.blend` each, beside copies of
their CSV files, which the scenes read through `//` paths. The out folder defaults to `demos/`:

```
blender -b --factory-startup --python demos/build_demos.py -- [<out folder>] --model [PORT GPU]
blender -b --factory-startup --python demos/build_demos.py -- [<out folder>] --noaddon
```

| Demo | What it shows |
|---|---|
| `life.blend` | The hero, Life from a CSV. A Rules node in a Simulation Zone updates a 64 × 64 grid that wraps around at its edges. The modifier's *Rule set* dropdown picks the Outcomes CSV: Conway, HighLife, Seeds or Day & Night. All four tables are stored, so a switch by hand shows at once and stays while the animation plays. For one continuous shot through the four rule sets, keyframe the dropdown and run *Prepare for Playback*: it pins the stored tables and asks nothing. `life_overrides.csv` locks one cell: a live cell with five neighbours dies |
| `biome.blend` | Rules One-of picks one of 7 plants from habitat descriptions of temperature, moisture and altitude, with Dither 0.3, and instances it. A second Rules node, in Amount mode, lays snow. The modifier's *Plant descriptions* dropdown switches to the bare plant names (`biome_outcomes_bare.csv`), whose confidence the sidebar shows much lower. The demo's `biome_outcomes.csv` has rewritten habitat descriptions, so it differs from the test fixture of that name |
| `kitchen.blend` | Match and Classify on a collection of 40 props: "Would you find it in a kitchen?" keeps the kitchen's props, "How heavy is it?" scales them, and their rooms place them |
| `materials.blend` | Classify gives each material slot a family. An Import CSV node plugged into *Labels CSV* supplies each family's roughness and metallic, and Match asks "Is this material metal?" |
| `weather.blend` | Dials: the modifier's prompt sets rain, wind, time of day, street lights and fog |
| `ask_pick.blend` | The prototype's scene rebuilt with OpenAct nodes: Ask "Is it night?" and Pick from a furniture collection, on two objects with their own prompts |

With `--model` the add-on starts the model on PORT and GPU. Without them it uses the add-on's
defaults, port 18765 and GPU 0. Give the port of a server no other Blender uses: the checks count
that server's requests. Every table fills, and each demo's acceptance checks run (design §18.4).
The files are saved with their answers, so they open, play and render without the model and without
the add-on. Without `--model` the scenes are only laid out, and nothing is asked.

The build also writes `evaluated.json` next to the files, which records what every demo object
evaluates to. `--noaddon` opens all six demos in a Blender without OpenAct, with each dropdown item
picked in turn, and checks that they evaluate to exactly the same attributes. Both commands end with
a RESULT line, and exit with 1 if a check failed.

## Changelog

### 1.0.0

The first release of OpenAct, which grew from the Jev Nodes 0.1.0 prototype and its two nodes, Ask and Pick.

- Four new nodes: **Match**, **Classify**, **Rules** and **Dials**. Pick gains a Context input, the per-element
  Selection and Probability outputs, and the Materials and Attributes list sources; Ask gains an Answers input.
- Answers apply per element, through an ID or the Rules factor inputs, from a table stored in the node. Saved files,
  per-element tables included, evaluate and render without the add-on.
- CSV authoring: answer sets written as "answer + when", Factors and Settings files, and payload columns that become
  outputs. A node takes a typed path or a plugged-in Import CSV node, and sees edits while Blender runs.
- An Overrides CSV with locks on all six nodes, and *Write Overrides CSV*.
- A confidence for every answer, unsure flags (*Unsure Below*), and *Balance Option Order* on each node.
- The sidebar shows each node's status, tables, questions and file problems, and offers starter templates and *Copy
  Diagnostics*; the *Log Requests* preference keeps a request log.
- *Prepare for Playback* for animated words.
- The model server batches questions, caches answers, checks a launch token, and can serve several Blender sessions.
- The installer pins and hash-checks what it downloads, checks the disk and the GPU, reuses a Jev Nodes runtime in
  place, and can locate an existing runtime.
- Background renders use the saved answers; headless scripts can opt in to asking.
- Jev Nodes 0.1.0 files upgrade in one step, keeping their answers.
- Blender 4.5 or newer (the prototype supported 4.2).
- Six demo scenes.

## Credits and licences

- **OpenAct** is licensed under GPL-3.0-or-later.
- **SemIf** (formerly OpenJev) by TheoLeeCJ, MIT: <https://github.com/TheoLeeCJ/SemIf-OpenJev>. The installer
  installs it into the runtime; it is not bundled.
- **Qwen3.5-4B** by the Qwen team (Alibaba), Apache-2.0: <https://huggingface.co/Qwen/Qwen3.5-4B>. The installer
  downloads it from Hugging Face; it is not redistributed.
- **PyTorch** (BSD-3-Clause), **Hugging Face Transformers** and **huggingface_hub** (Apache-2.0) and **uv** (MIT or
  Apache-2.0) are installed into the runtime.
- `tests/fixtures/jev_nodes-0.1.0.zip` is the release of the Jev Nodes prototype (same author, GPL-3.0-or-later).

Not affiliated with TypeSafe or Jev. The model server is Jev-compatible only in the shape of its requests
(`POST /v1/systemone`).
