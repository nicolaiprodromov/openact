# OpenAct user guide

Everything about using OpenAct, in depth. For the overview, see the [README](../README.md).

**Contents**

- [Requirements](#requirements)
- [Install](#install)
  - [The model runtime](#the-model-runtime)
- [The six nodes](#the-six-nodes)
  - [Working with the nodes](#working-with-the-nodes)
- [CSV files](#csv-files)
  - [Two ways to give a node its CSV](#two-ways-to-give-a-node-its-csv)
  - [Formats](#formats)
  - [Overrides](#overrides)
  - [Templates](#templates)
  - [Tips](#tips)
- [Render farms and headless scripts](#render-farms-and-headless-scripts)
- [Privacy](#privacy)
- [Limitations and known issues](#limitations-and-known-issues)
- [Future work](#future-work)

## Requirements

- Blender 4.5 or newer. OpenAct 1.0 is tested on 4.5, 5.0 and 5.1.
- Windows 10 or 11 with an NVIDIA GPU (CUDA) with 10 GB of VRAM or more; the model uses about 9 GB. Linux with an
  NVIDIA GPU is best effort and untested.
- 15 GB of free disk space for the one-time install of the model runtime, which takes about 13 GB.

Only authoring needs the GPU and the runtime. Saved files open, play and render without them, and without the add-on,
on Blender 4.5 or newer.

## Install

1. Download `openact-1.0.0.zip` from the GitHub release, or build it from this repository:
   `blender --command extension build --source-dir openact --output-dir dist`.
2. In Blender: *Edit > Preferences > Get Extensions > Install from Disk...* and pick the zip.
3. Open a Geometry Node editor, press **N** and go to the **OpenAct** tab.
   - If it says *Model not installed*, click **Install Model** (below), or **Locate Existing Runtime...** to use a
     runtime already on this computer.
4. Add nodes with **Shift A > OpenAct**.

The model server starts when a node first needs an answer (the *Start Automatically* preference) and loads the model
onto the GPU in about 12-40 s. The sidebar's Model box shows its status, with *Start Model* / *Stop Model*, *Open
Log*, and *Copy Diagnostics*, which copies a report for bug reports without your prompts, CSV text or paths. Another
Blender on the same port uses the running server; there the button reads *Detach*.

### The model runtime

**Install Model** downloads about 12 GB once: uv, Python 3.12, PyTorch for CUDA 12.8, SemIf and the model. It
installs them into `%LOCALAPPDATA%\OpenAct` (on Linux `$XDG_DATA_HOME/OpenAct`, or `~/.local/share/OpenAct`). It needs
*Preferences > System > Network > Allow Online Access*. uv, SemIf and the Python packages are pinned and hash-checked,
and the model is fetched at a fixed revision. The Model box shows the progress, and *Open Log* opens `install.log`.

For a runtime in another folder, use *Locate Existing Runtime...* or the *Runtime Folder* preference. *Open Runtime
Folder* shows the folder; deleting it uninstalls the model. Uninstalling OpenAct leaves it in place.

## The six nodes

OpenAct adds six Geometry Nodes that make judgements. A question, a prompt, a list of items or a table of rules goes
in; numbers, selections, picks and labels come out. The nodes read their words (typed on the node, set on the
modifier panel, or kept in CSV files) outside evaluation, ask a model on your GPU in the background, and store its
answers in the node as a table that Geometry Nodes looks up, per element where needed. The model runs only when the
words change, never during evaluation, playback or rendering. The answers are saved in the .blend, so a file opens,
plays and renders without the model, and without the add-on, on Blender 4.5 or newer.

The model never generates text. It reads the probability of each allowed answer from one forward pass. That is the
method of **SemIf** (formerly OpenJev, MIT), running **Qwen3.5-4B** (Apache-2.0) on your GPU.

| Node | Judges | Answer | Words in | Outputs |
|---|---|---|---|---|
| **Ask** | the whole situation | Yes/No or Amount | Question, Context, Answers (optional) | **Answer** 0-1, **Yes**, **Confidence** |
| **Pick** | the whole situation | one entry of a list | Prompt, Context, the list | **Index**, **Item**, **Confidence**; per element **Selection** and **Probability**; the picked **Instance** (Collection) or **Material** (Materials) |
| **Dials** | the whole situation | per setting: Yes/No, Amount or One of | Prompt, Settings | one output per setting; **Confidence**, the lowest of theirs |
| **Match** | each item of a list | Yes/No or Amount | Question, Context, the list, Answers (optional) | per element **Probability**, **Selection** (Probability ≥ *Threshold*), **Confidence** |
| **Classify** | each item of a list | one of several labels | Question (optional), Context, the list, Labels | per element **Label** (its row), **Confidence**, the Labels CSV's payload |
| **Rules** | each measured situation | One of, Yes/No or Amount | Question, Context, Factors, Outcomes, and one field input per factor | per element: One of gives **Outcome** (its row) and the Outcomes CSV's payload, Yes/No **Probability** and **Yes**, Amount **Amount**; and **Confidence** |

Every node also has hidden inputs: **Overrides** (with *Overrides CSV*), a CSV that locks answers
([below](#overrides)), and **Key**. Ctrl H shows them.

### Working with the nodes

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
  so far. The *Balance Option Order for New Nodes* preference sets it for new nodes.
- **Key.** Objects whose words are identical but whose Collection or Object differ can't be told apart: the node
  says only one answer can show. Show **Key** (Ctrl H), expose it on the modifier and give each object its own
  value. The model never sees it.
- **Sidebar** (N > OpenAct): the Model box; for the active node, its status, a summary per object, the table of
  answers with confidence and source (*Show Question* shows exactly what a row asks the model), **Ask Again** and
  **Write Overrides CSV**; and the *Files* panel: each CSV the node reads, with its columns and problems, *Open File*
  and *New CSV from Template*. The sidebar never edits answers: to change one, change its words, or lock it in the
  Overrides CSV.
- **Animated words.** OpenAct asks nothing during playback or rendering. When words change over the frame range (a
  keyframed dropdown, say), run **Prepare for Playback** in the sidebar first: it answers every set of words the
  frames show and pins those answers in the file. *Unpin* in the node's panel releases them.

## CSV files

Answer sets, item lists and Dials settings can be typed inline or kept in CSV files next to the .blend; Rules factors
and overrides are always CSV files. OpenAct parses every column itself. Text columns (names, `when` descriptions, `if`
restrictions) go to the model. The numeric columns of a Labels CSV (Classify) or a One-of Outcomes CSV (Rules) are
*payload*: each becomes an output socket, and its values are baked into the node's table. The model never sees the
numbers, and a saved file needs no CSV to render.

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
| `when` (or `description`) | When this answer applies, or what the item is (see [Tips](#tips)) |
| `if` | Rules Outcomes only: the cells that may choose this outcome (below) |
| payload columns | Classify's Labels and One-of Rules Outcomes: one output each, read at the chosen row; at most 16 |

A payload column holds only integers (an Int output) or only numbers (a Float output), and its header becomes the
output's name. An empty cell, a decimal comma, `inf` or `nan` makes it text, with a warning. Other text columns are
ignored. In 1.0.0 an Items CSV's numeric columns give no outputs, though the *Files* panel lists them as payload
columns.

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

### Overrides

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
to every object that uses the node; for overrides per object, give each object its own file through the modifier
(show the Overrides input with Ctrl H).

### Templates

*New CSV from Template* (the sidebar's *Files* panel) copies a node's starter files, from
[`openact/templates/`](../openact/templates), next to the saved .blend, never over an existing file (`items.csv`
becomes `items 2.csv`), and puts each copy's path in its input when that input is empty and unlinked.

| Node | Starter files |
|---|---|
| Rules | `life_factors.csv` and `life_outcomes.csv`: Conway's Game of Life |
| Classify | `items.csv`, and `labels.csv`: material families with roughness and metallic |
| Pick, Match | `items.csv`: household objects with descriptions |
| Dials | `weather_dials.csv`: rain, wind speed, time of day, street lights and fog |

### Tips

- Give every option and item a `when`: when it applies, or what it is. Descriptions make the answers much more
  reliable than bare names.
- Keep a question to 16 options or fewer. A longer list is asked as a tournament of smaller questions, which is less
  reliable: split it into two levels instead, e.g. a group first, then an item within the group.

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
hundredths of a second until its nodes are answered, and saves, as `drive()` in
[`tests/_harness.py`](../tests/_harness.py) does. The test suites set the variable (`tests/_harness.py`).

## Privacy

After the one-time install, OpenAct talks only to its model server on 127.0.0.1: nothing leaves your computer. A
saved .blend holds its words, the item and outcome names, and the answers, so sharing the file shares them.

## Limitations and known issues

- **Blender's Import CSV cache affects your own Import CSV nodes.** Blender reads each Import CSV path once per
  session. If one of your own Import CSV nodes (not one plugged into an OpenAct node) reads a file that an OpenAct
  node also reads, it keeps the old rows after you edit the file, while OpenAct's outputs update. Use the OpenAct
  node's payload outputs instead, or restart Blender. The sidebar notes each such node.
- **Blender 4.5 or newer only.** Older versions lack nodes OpenAct builds on.
- **NVIDIA GPUs (CUDA) only.** The installer refuses to install without one, and there is no CPU backend yet (see
  [Future work](#future-work)). macOS is not supported for authoring.
- **At most 16 sets of words per node.** A tree used by more than 16 objects with different words is an error on the
  node: split the tree.
- **Rare crash on Blender 4.5 and 5.0.** With OpenAct enabled, a Blender bug can crash Blender when the node editor's
  viewer shortcuts (Ctrl+1…9 and 1…9), Quick Effects or some other add-ons' tools run. It is rare, and Blender 5.1 is
  not affected.

## Future work

Neither of these is built or scheduled. The [roadmap](../README.md#roadmap) shows where they sit.

- **A CPU backend** through llama.cpp (a GGUF Q4_K_M model of about 3 GB), for machines without an NVIDIA GPU.
- **A high-accuracy model**: SemIf's 27B "bridge" (EXL3, 5 bits per weight). On SemIf's benchmark of 144 authored
  decisions its balanced accuracy is 0.958, against 0.813 for the 4B model OpenAct uses. It needs a 24 GB GPU.
