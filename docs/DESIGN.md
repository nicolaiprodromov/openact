# OpenAct v1 — Decision Nodes for Geometry Nodes

| Field | Value |
|---|---|
| Author | _&lt;author&gt;_ |
| Reviewers | _&lt;reviewers&gt;_ |
| Date | 2026-09-26 |
| Status | Draft |
| Supersedes | `jev_nodes` v0.1.0 prototype (`X:\interzis\CHUMP_KIT\jev_nodes\`) |
| Target | OpenAct 1.0.0 on Blender 4.5 LTS to 5.1; NVIDIA GPU first |
| Audience | Senior engineers who know the prototype; the agent implementing the PR Plan |

## Table of Contents

1. [Overview](#1-overview)
2. [Background & Motivation](#2-background--motivation)
3. [Goals & Non-Goals](#3-goals--non-goals)
4. [Principles](#4-principles)
5. [Architecture](#5-architecture): runtime split, request pipeline, per-context caching, threading
6. [The Six Nodes](#6-the-six-nodes): Ask, Pick, Match, Classify, Rules, Dials (sockets, behaviour, examples)
7. [CSV Formats](#7-csv-formats): schemas, text/numeric split, overrides and locks, validation, change detection
8. [Table Lookup in Geometry Nodes](#8-table-lookup-in-geometry-nodes): Index Switch vs hidden table geometry, thresholds, node graphs
9. [Rules Engine Details](#9-rules-engine-details): factors, bands, state index math, Amount interpolation, dithering, exact-fact restrictions
10. [Engine & Server Changes](#10-engine--server-changes): static evaluator generalization, batching, debias cost, tournaments
11. [Minimum Blender Version](#11-minimum-blender-version)
12. [Migration from jev_nodes v0.1.0](#12-migration-from-jev_nodes-v010)
13. [API / Interface Changes](#13-api--interface-changes)
14. [Data Model](#14-data-model)
15. [Alternatives Considered](#15-alternatives-considered)
16. [Security & Privacy](#16-security--privacy)
17. [Observability](#17-observability)
18. [Testing Strategy & Demo Scenes](#18-testing-strategy--demo-scenes)
19. [Risks](#19-risks)
20. [Rollout Plan](#20-rollout-plan)
21. [Open Questions](#21-open-questions)
22. [References](#22-references)
- [Key Decisions](#key-decisions)
- [PR Plan](#pr-plan)

---

## 1. Overview

OpenAct turns the working `jev_nodes` v0.1.0 prototype into a product: six Geometry Nodes that
make judgments with a local "System One" decision model. The model reads text and returns
calibrated probabilities over a fixed set of typed answers in one forward pass. This is SemIf
letter-logit scoring on Qwen3.5-4B, and no text is ever generated. Geometry Nodes cannot run
Python while it evaluates. So every OpenAct node is a `GeometryNodeCustomGroup` whose hidden
inner node group holds the model's decisions as a precomputed **table**. The add-on reads the
node's words outside evaluation, asks the model in the background and rewrites the table.
Geometry Nodes then applies the table **per element** through an ID or state field. Words go in
and Geometry Nodes data comes out. The model runs only when the words change. It never runs
during evaluation, playback or rendering, and the answers are saved in the .blend.

v1 keeps what already works in the prototype:
- the static input evaluator;
- per-context answers keyed by the exact input strings;
- the runtime split: a separate Python environment behind a Jev-compatible API on 127.0.0.1;
- answers saved in the file.

It adds the following:
1. **Four new nodes and new Pick outputs.** Match, Classify, Rules and Dials are new. Pick gains
   Selection and Probability fields.
2. **Per-element tables read through an ID field.** Every per-element column lives in a hidden
   table object read with `Sample Index`. `Index Switch` is used only for single values (§8.4).
3. **Rules, answer lists and labels written as CSV files.** The node takes each file through a
   path socket, so the file stays a valid Import CSV input for the artist's own nodes. Python
   parses every column:
   - **Text columns** (names, "when" conditions, exact-fact restrictions, descriptions) are what
     the model sees.
   - **Numeric payload columns** (e.g. `alive`, `scale`, `instance`) are baked into the node's
     table and come out as output sockets at the chosen row. They are never sent to the model.

   The inner group does not use Import CSV, because Import CSV caches by path and never re-reads
   an edited file (verified in 4.5.2 and 5.1.1, §7.1).
4. **Caching per cell and batched requests.** Many questions that share one state go into one
   request.
5. **Artist overrides written in an Overrides CSV.** Locked cells are never asked again.
6. **The rename to OpenAct**, with an upgrade path for old files and the 13 GB runtime already
   installed.

The minimum Blender version rises from 4.2 to **4.5 LTS** (§11).

### 1.1 The six nodes at a glance

| Node | Judges | Answer types | Words in (read statically) | Per-element input (field) | Outputs |
|---|---|---|---|---|---|
| **Ask** (exists) | the whole situation | Yes/No, Amount | Question, Context, Answers (optional), Overrides | none | Answer, Yes, Confidence (single values) |
| **Pick** (extend) | the whole situation | one entry of a list | Prompt, Context, Items, Overrides | ID | Index, Item, Confidence; **Selection**, **Probability** (fields); Instance / Material / CSV columns, depending on the list source |
| **Match** (new) | each named item | Yes/No, Amount | Question, Context, Items, Answers (optional), Overrides | ID | Probability, Selection, Confidence (fields) |
| **Classify** (new) | each named item | One-of (labels) | Question, Context, Items, Labels, Overrides | ID | Label, Confidence (fields); numeric columns of the Labels CSV |
| **Rules** (new) | each measured situation | Yes/No, Amount, One-of | Question, Context, Factors (CSV), Outcomes, Overrides | one socket per factor (Float band / Int category / Bool flag) | Outcome or Probability+Yes or Amount; Confidence; numeric columns of the Outcomes CSV |
| **Dials** (new) | the whole situation | per setting: Yes/No, Amount, One-of | Prompt, Settings, Overrides | none | one output per setting, Confidence |

Every node also has a hidden `Key` word. It is compared by the match chain to tell contexts apart,
but it is never sent to the model (§6.0).

**Items** comes from one shared set of list sources, chosen with a dropdown on the node:
- **Text or CSV**: a String socket holding inline items or a `.csv` path;
- **Collection**: its children, in natural-sort order;
- **Materials**: an object's material slots;
- **Attributes**: attribute names.

Entry *i* of any list is addressed by ID *i*. That index matches Collection Info child order,
material index and the CSV data row. Answers, Labels, Outcomes and Settings share one format:
**"answer + when it applies"**. Inline, each entry is `name: when`. In a CSV, the columns are
`answer,when[,if][,numeric payload…]`.

### 1.2 Architecture in one picture

```mermaid
flowchart LR
  subgraph Blender["Blender process: pure Python, bpy on the main thread only"]
    GN["Geometry Nodes tree<br/>OpenAct node = custom group"] -->|"words: strings, paths,<br/>collections, objects"| EV["Static evaluator<br/>evaluator.py"]
    LS["List and CSV readers<br/>lists.py, csvfmt.py"] --> PL["Pipeline: cells, cache,<br/>batching, queue<br/>pipeline.py"]
    EV --> PL
    PL -->|"decided rows"| IN["Inner group builder<br/>inner.py, tables.py"]
    IN -->|"hidden table object (per element)<br/>+ Index Switch (single values)"| GN
  end
  CSV[("CSV files next to the .blend")] -->|"text columns (model)<br/>numeric columns (payload)"| LS
  LS -->|"payload values, baked"| IN
  PL <-->|"POST /v1/systemone<br/>127.0.0.1 + launch token"| SRV["openact_server.py<br/>own venv: torch + SemIf<br/>Qwen3.5-4B"]
```

### 1.3 Numbers the design is built around

All figures were measured on an RTX 3090 with bf16 weights and order debiasing on (§2.2).

| Quantity | Value |
|---|---|
| Model load | 12–40 s (cold disk) |
| VRAM | 8.4 GB |
| One decision without debiasing | ~69 ms |
| 5-option choice with debiasing | ~180 ms |
| New Ask/Pick answer in the add-on | 0.2–0.6 s |
| 255-option tournament | ~2.7 s |
| Table fill: yes/no cells | ~24 cells/s (216 cells in 8.9 s) |
| Table fill: 7-option cells | ~14 cells/s (125 cells in 8.9 s) |
| Game of Life table (18 cells) | 0.5–2.7 s, exact when framed as "outcome + when" |
| Model calls during evaluation, playback and rendering | none, by construction |

The rest of the document specifies:
- the sockets of each node (§6);
- the CSV formats (§7);
- the lookup graphs (§8);
- the Rules maths (§9);
- the engine and server changes (§10);
- the ordered PR Plan (bottom).

## 2. Background & Motivation

### 2.1 The prototype today (`jev_nodes` v0.1.0)

The prototype lives in `X:\interzis\CHUMP_KIT\jev_nodes\`. It has no git repository and is built
with `blender --command extension build` into `dist/jev_nodes-0.1.0.zip`. Every test passes on
Blender 4.2 and 5.1.

| File | What it does today | What happens to it in OpenAct |
|---|---|---|
| `jev_nodes/__init__.py` | Registers `JevPreferences`, then `ops`, `nodes`, `ui`, then `engine.register()` and `server.refresh()` | Package becomes `openact`; same registration order |
| `jev_nodes/prefs.py` | Preferences: `runtime_dir`, `port` (18765), `device`, `gpu`, `auto_start`, `debias`. `runtime_installed()` checks the venv and every shard in `model.safetensors.index.json` (result cached by mtime) | Becomes `OpenActPreferences`; adds legacy-runtime detection, backend choice and the "unsure below" setting |
| `jev_nodes/server.py` | Runs the subprocess with a `--parent-pid` watchdog. Uses `ThreadPoolExecutor(max_workers=4)` for all HTTP; `post()`/`pump()` hand results to the main thread; status machine `unknown/not_installed/stopped/starting/ready/error/installing`. The uv installer pins torch 2.10.0 (cu128 or cpu), SemIf commit `23cf1f39…` and model revision `851bf6e8…`, and honours `bpy.app.online_access` | Kept. Adds a launch token and one priority-ordered decision worker; installers are split per backend |
| `jev_nodes/engine.py` | Static evaluator (`eval_input`, `eval_output`, `_HANDLERS`, `_group_input` incl. modifier values and `*_use_attribute`). Contexts (`find_uses`, `Ctx`). Node specs (`AskSpec`, `PickSpec`). Inner builder (`_Builder.match_chain`, `_Builder.lookup`). Caches (`MEMORY`, `PENDING`, `FAILED`, `STATUS`). Passes (`run_pass`, `_process`, `tick` every 0.1 s, `DEBOUNCE` 0.25 s, `SAFETY_PASS` 2 s, paused during playback and render jobs). Ownership fix-up `ensure_owned` | Split into `evaluator.py`, `engine.py`, `pipeline.py`, `store.py`, `inner.py`, `tables.py` and `specs/` |
| `jev_nodes/nodes.py` | `GeometryNodeJevAsk` ("Ask Jev", mode `YES_NO`/`AMOUNT`) and `GeometryNodeJevPick` ("Jev Pick"). Both use the `_JevNode` mixin with hidden `uid` and `entries` JSON. `copy()` only clears `uid`. Add menu *Shift A > Jev* | Six `GeometryNodeOpenAct*` classes plus two hidden legacy aliases used only for migration |
| `jev_nodes/ops.py`, `jev_nodes/ui.py` | Install, start, stop, re-ask, open log, open runtime folder. Sidebar tab "Jev"; preferences UI | Renamed to `openact.*`. The sidebar adds table inspection, CSV templates and "Upgrade old nodes" |
| `jev_nodes/natsort.py` | Port of `BLI_strcasecmp_natural` (unsigned bytes) | Unchanged |
| `jev_nodes/runtime/jev_server.py` | `POST /v1/systemone`, `GET /health`, `GET /v1/models`, `POST /shutdown`. SemIf `score`/`score_shared` in chunks of 32 rows that share one state; forward+reverse order debiasing; tournament for 17–255 options; LRU cache (4096) keyed on the **whole request** | Becomes `openact_server.py`. Adds a per-question cache, token check, a backend interface and `/v1/capabilities` |
| `tests/e2e_blender.py`, `tests/evaluator_blender.py` | Real-model end-to-end test (phases build / noaddon / reopen); evaluator checked against Geometry Nodes | Kept and extended; new suites for CSV, lookup, simulation zone, migration and a fake backend |

**How one answer flows today:**
1. `run_pass()` lists the nodes (`iter_jev_nodes`) and calls `ensure_owned()`.
2. It maps every tree to the modifier contexts that use it (`find_uses`, up to 64 per tree).
3. For each context, `spec.request()` evaluates the node's strings and builds a `Req`. A `Req`
   has two parts: `gn`, the raw input strings that Geometry Nodes can compare, and `key`, a
   sha1 of the normalised meaning.
4. `_process()` keeps at most 16 `entries` per node (`MAX_ENTRIES`).
5. For every missing key, `_submit()` sends one HTTP request.
6. `build_inner()` runs `tree.nodes.clear()` and rebuilds a Compare(`A_STR`/`B_STR`) chain.
   That chain feeds a Switch(INT), which drives one Index Switch per output.

Geometry Nodes then picks the entry whose strings equal the live inputs. This lets one tree
serve many objects with different prompts, and switching between them is instant.

### 2.2 Measured behaviour of the model (the design's foundation)

- **Framing decides accuracy.** When each option says "outcome + when it applies", the model
  produced exact 18-cell tables for four rule sets, with zero uncertain cells: Conway B3/S23,
  HighLife B36/S23, Seeds B2/S and Day & Night B3678/S34678. A bare yes/no ("is the cell alive
  next step?") with the rules as context got Conway wrong (it answered B3/S2).
- **Code handles what's certain.** A casually worded rule set had 3 wrong cells when every clause
  was offered. It was exact when code offered only the clauses that match the exactly known fact
  (the cell's current state).
- **Descriptions matter.** In a biome table (27 cells, 7 plants), bare plant names gave mean
  confidence 0.32, used only 4 outcomes and collapsed onto one factor. Habitat descriptions gave
  0.61, used all 7 outcomes and respected all three factors.
- **Amount answers are smooth and monotonic.** Snow cover went from 0.08 (low and hot) to 0.89
  (high and cold). Interpolating between band centres is therefore safe.
- **Confidence tracks errors.** The only wrong run had 4 uncertain cells, clustered around the
  mistakes. Flagging low-confidence cells is a useful review signal.
- **Weak at arithmetic, counting and numeric precision; strong at semantic matching.** Numbers
  must be turned into words by code. A band name or an exact category goes to the model, never a
  raw 0.73.

### 2.3 Pain points and gaps

1. **Single values only.** The README says: "Fields and geometry can't be used as inputs: the
   question is asked once per node." Nothing can judge each instance, material or cell state.
2. **Unstructured answers.** Pick items are bare names, which gives low confidence (above). There
   is no way to give conditions, restrictions or payload values, and no CSV.
3. **No overrides.** An artist can correct a wrong answer only by rewording the prompt.
4. **One request per node-context, and a server cache keyed on whole requests.** A 216-cell table
   would be either 216 round trips or one request whose cache entry any single edit invalidates.
5. **No scheduling.** The executor is FIFO and shared with `/health` probes. It has no
   priorities, no chunking, no cancellation of superseded work and no progress reporting.
6. **Rebuild cost.** `build_inner()` rebuilds every node on every change. Setting `node.location`
   alone costs about 6 ms per node in 5.1 (measured in a sibling project). A 255-row table with 4
   outputs would stall the UI.
7. **Blender 4.2 limits.** 4.2 lacks Import CSV, Import Text, `GeometrySet`, and the Object and
   Collection input nodes. Without the add-on, 4.2 evaluates saved OpenAct nodes as zeros.
8. **Naming.** "Jev" is TypeSafe's trademark for its hosted model. SemIf renamed itself from
   "OpenJev" for that reason, so the product cannot be called Jev Nodes.
9. **Status is a single line per node.** There is no notion of partially filled tables, unsure
   cells, locked cells or an ETA.

### 2.4 Why now

The project will be released as open source with a promo video. The audience is Geometry Nodes
artists in general, not one user's projects. The nodes therefore have to be universal building
blocks: one verb per node, the same list sources and answer format everywhere, and CSV files that
the tree itself consumes. A later "agents" design (think slow, act fast, e.g. a low-poly
OpenStreetMap town) will build on v1's pieces:
- the background request queue;
- per-element tables;
- the ability to read per-element state from evaluated geometry.

v1 must ship those pieces without precluding that design.

## 3. Goals & Non-Goals

### 3.1 Goals

**G1. Six nodes, one verb each.** Ask, Pick, Match, Classify, Rules and Dials cover a grid of
scope (whole situation / each named item / each measured situation) by answer type (Yes/No,
Amount, One-of). They share the same list sources, the same "answer + when it applies" format
and the same confidence semantics (§4, §6).

**G2. Decide per meaning, apply per element.** Decisions are tables inside the hidden inner
group. Geometry Nodes reads them per element through an ID or state field. This works inside
simulation and repeat zones and needs no model at evaluation time (§8).

**G3. Author everything in Geometry Nodes and in files.** Inputs are sockets and CSV files. Each
file is given to the node through a path socket and stays valid input for Import CSV.
- Python reads the text columns for the model.
- Numeric payload columns are baked into the node's table and exposed as output sockets.
- Editing only numeric columns never re-asks the model: it triggers a values-only update (§7).

**G4. The model drafts, the artist decides.** Every table can be inspected in the sidebar. It can
be exported to an Overrides CSV, and locked rows win over the model and are never asked again
(§7).

**G5. Confidence everywhere.** Every answer carries a confidence in 0–1. Cells below the
"Unsure below" preference (default 0.30) are flagged in the node status line and the sidebar.

**G6. Interactive speed with nothing at render time.** Answers are saved in the .blend.
Reopening needs no model. Files evaluate without the add-on on Blender 4.5+.

**G7. Keep the runtime split.** The model server runs in its own environment behind the
localhost Jev-compatible API, and Blender stays pure Python. Backends are swappable behind that
API. **Only NVIDIA/CUDA is built in v1.** A CPU backend (llama.cpp) is future work, outside this
plan.

**G8. Rename to OpenAct and migrate cleanly.** New package id, bl_idnames, UI strings and
runtime folder. The installed 13 GB runtime is reused in place. Old files are upgraded with
their stored answers intact (§12).

**G9. Tested and demoable.** Automated suites cover the evaluator, CSV parsing, table lookup
(including main-thread cost budgets), simulation-zone use, migration, a fake-model pipeline and
real-model end to end, on Blender 4.5, 5.0 and 5.1. Demo scenes for the promo video are built by script (§18).

**G10. Don't preclude agents.** v1 ships a job queue with priorities and cancellation,
per-element tables that can be rewritten without rebuilding node graphs (table objects), and
reading of evaluated geometry through `GeometrySet`.

### 3.2 Quantified targets (RTX 3090, bf16, order balancing on, which is the default; warm model, idle queue)

| Metric | Target | Basis |
|---|---|---|
| New Ask / Pick / Dials (≤ 8 settings) answer visible | ≤ 0.6 s p95 after the 0.25 s debounce | measured 0.2–0.6 s |
| Rules, 18 cells, 2 allowed outcomes per cell (Game of Life) | ≤ 3 s | measured 0.5–2.7 s |
| Match / Rules, 50 yes/no cells | ≤ 2.5 s | ~24 cells/s |
| Rules, 216 yes/no cells | ≤ 10 s | measured 8.9 s |
| Classify / Rules, 125 cells × 7 options | ≤ 10 s | measured 8.9 s |
| First partial result on a large table | ≤ 2.5 s (tables fill in chunks of 32 questions) | §10 |
| Re-ask after editing one CSV text cell | only the affected cells are sent | per-cell cache, §5 |
| Re-ask after editing only numeric CSV columns | 0 requests | text/numeric split, §7 |
| Model calls during evaluation, playback, render | 0 | principle 4.b |
| Reopen with no model | 0 requests, identical evaluated output | e2e `reopen` phase |
| Game of Life tables (Conway, HighLife, Seeds, Day & Night) | 18/18 exact, 0 unsure, each | measured |
| Biome Classify/Rules (27 cells, 7 described plants) | mean confidence ≥ 0.55, all 7 outcomes used | measured 0.61 |
| Largest table | 4096 cells and ≤ 8192 scored rows per node context | fill time ≈ 3–5 min worst case |
| Per-element lookup cost, 1M elements | ≤ 5 ms above a no-lookup baseline, up to 8 interpolation corners (review measurement: within ~1 ms on 4.5.2 and 5.1.1) | §8.7 |
| Main-thread cost per pass | ≤ 5 ms per unchanged node; ≤ 50 ms per idle pass with a maximum-size Rules node used by 16 contexts | §5.3 |
| Graph edits on the main thread | `inner.build()` ≤ 100 ms for the demo shapes (≤ 250 ms for the interpolated Rules demo) and ≤ 1 s at each derived worst shape (§8.4), run only on shape or materialised-set changes, at most one per edit; `update_values()` (entry-level Index Switches only) ≤ 30 ms when one entry's values change, ≤ 150 ms when all 16 entries' values change at once; a table-object write ≤ 5 ms | §8.4, §8.8 |
| Add-on upgrade (new templates), toggling a node's *Balance Option Order*, or upgrade of a v0.1.0 file | 0 requests until the words change | §5.4, §10.3 |
| Prepare for Playback over 250 frames | < 1 s on the F-curve path (channelbag F-curves, no scene evaluation). The fallback for drivers, NLA, tweak mode or blended actions costs frames × one scene evaluation, with progress and Esc | §5.5 |
| Blender versions | 4.5 LTS, 5.0, 5.1 | §11 |

### 3.3 Non-Goals (v1)

- **NG1. Agents** (think slow, act fast), decisions arriving asynchronously during playback and a
  record mode for deterministic renders. These belong to a separate design document. v1 keeps
  passes paused during playback and rendering exactly as today.
- **NG2. The "Show" route.** No describing actual elements from geometry (per-element attribute
  values, images) and no vision encoder. The Qwen3.5-4B checkpoint has one, but SemIf loads only
  the text model. v1 has only the "Name" route (Pick/Match/Classify judge named entries) and the
  "Bucket" route (Rules judges banded states). Neither reads geometry values.
- **NG3. A hosted or cloud backend** such as TypeSafe's Jev. The API stays shape-compatible, so
  one could be added later, but none ships.
- **NG4. Text generation,** free-form answers or chat. Every output is a probability over a
  fixed answer set.
- **NG5. Arithmetic by the model.** Counting, thresholds and band assignment happen in nodes and
  code. The model sees band names and descriptions only.
- **NG6. Per-element prompts from continuous values.** Every distinct value would need a model
  call. Rules bands exist so that a table has a bounded number of cells.
- **NG7. Blender 4.2 support** (§11).
- **NG8. Editing tables in the UI.** The sidebar is inspection only. Overrides live in CSV files
  (principle 4.f with the "everything in Geometry Nodes and files" constraint).
- **NG9. Any backend other than NVIDIA/CUDA** (user decision).
  - A CPU backend (llama.cpp, GGUF Q4_K_M) is **future work, not in this plan**. So is Apple
    Silicon.
  - AMD/Intel GPUs and multi-GPU serving are out of scope.
  - The CPU path of torch that the prototype's installer could fall back to is **removed** in v1
    (§10.5).
- **NG10. Model training, fine-tuning or retrieval.** The model and SemIf revision are pinned.

**v1 support matrix.**

| Platform | Status |
|---|---|
| Windows 10/11 with an NVIDIA GPU of ≥ 10 GB VRAM (12 GB recommended; the model uses 8.4 GB) | **Supported and tested.** Every gate runs here (§18) |
| Linux with an NVIDIA GPU of ≥ 10 GB | Best effort, untested. The installer and paths exist, but no gate runs |
| macOS, and any machine without an NVIDIA GPU | Unsupported for authoring; a CPU backend is future work (NG9). Files still **render** there, because answers are saved; only authoring needs the model |

### 3.4 Hard constraints (verified facts the design must respect)

- Geometry Nodes cannot run Python during evaluation, and Python cannot read socket values
  inside a tree during evaluation. Words come from **static evaluation** of the node tree only.
- In a custom node's `copy()`, never reassign `node_tree`. In 4.2 the pointer is not user-counted
  and the original's group is dropped on save. `copy()` clears `uid`, and the next pass
  (`ensure_owned`) gives the copy its own inner group and table object.
- Look up sockets by **identifier** (`A_STR`, `B_STR`, `A_INT`, …), never by display name.
- Import CSV (4.5+) keeps **only numeric columns** (INT or FLOAT point attributes) and silently
  drops text columns. There is one point per data row.
- **Verified in headless Blender 4.5.2 and 5.1.1** (identical results in both):
  - **Import CSV caches by path and never re-reads an edited file.** The node still returned the
    old rows after all three of these: rewriting the file followed by `object.update_tag()`,
    re-assigning the same Path string, and re-setting the Delimiter to the same value. OpenAct
    therefore never relies on Import CSV noticing edits (§7.1).
  - **Index Switch supports every data-block and value type we need**, including `MATERIAL`,
    `OBJECT`, `COLLECTION`, `IMAGE`, `GEOMETRY`, `MENU`, `STRING` and `RGBA`.
  - **`default_input = 'INDEX'` works on an Int interface socket.** The options are `VALUE`,
    `INDEX`, `ID_OR_INDEX`, `NORMAL`, `POSITION`, `INSTANCE_TRANSFORM`, `HANDLE_LEFT` and
    `HANDLE_RIGHT`. With `hide_value = True`, an unconnected ID input on a custom group node
    evaluated to the element index `[0, 1, 2, 3]`.
  - **Custom group nodes follow edits to their inner tree's interface immediately.** New inputs
    and outputs appear on the node at once, and existing links stay valid, so dynamic sockets are
    feasible.
- Collection Info (Separate Children) orders children with `BLI_strcasecmp_natural`
  (`natsort.py`). Join Strings joins the most recently connected link first. Import Text converts
  CRLF to LF, keeps a lone CR and a BOM, and returns "" for invalid UTF-8 or a NUL byte.
- `GeometrySet.from_evaluated_object` (4.5+) must be kept referenced. Chaining `.mesh` on the
  temporary raises `ReferenceError`. A hidden object in an excluded collection still evaluates
  when Object Info references it. So does an object linked to **no** collection whose only user is
  an Object Info socket default: verified by the review on 4.2–5.1, and live updates on 4.5.2 and
  5.1.1 (§8.5).
- Extension rules: check `bpy.app.online_access` before any download; permission strings are at
  most 64 characters; tags come from Blender's allowed list.

## 4. Principles

Every node, CSV format and code path follows these rules. PR reviews should cite them by letter.
Principles a–g were agreed with the user. Principle h restates the user's hard constraint.

### 4.a Words in, Geometry Nodes data out
The model reads words and returns probabilities over named answers. It never receives or produces
raw numbers to compute with. Numbers reach the model only as words that code has chosen:
- a band name and description ("cold: freezing to chilly") for an Amount factor;
- a level label ("exactly three live neighbours") for an Int category;
- a flag label for a Bool factor.

The rest of the maths runs as Geometry Nodes inside the inner group or in the artist's own tree:
- counting (neighbours), thresholds (Match *Threshold*), band assignment;
- interpolation and dithering.

Numeric payload (scale, instance index, material index) is authored data. It travels from the
CSV into the node's table and out of a socket, and the model never sees it.

Python only does exact, deterministic work: parsing, hashing, table layout. The one number it
derives from a model probability is the confidence (4.e).

*Enforced by:*
- Specs build questions from strings only (§10).
- Rules banding is a node graph (§9).
- Numeric CSV columns are never sent to the model (§7).

### 4.b Decide per meaning, apply per element
There is one model decision per distinct meaning, called a **cell**. Each decision is stored as a
row of a table that belongs to the node's hidden inner group `.OpenAct <Node> <uid>`. Per-element
columns are stored in the group's hidden table object (§8). Geometry Nodes applies the table per element through an **ID**
field (Pick, Match, Classify) or through **factor** fields (Rules). This is an O(1) lookup per
element, so evaluation cost does not depend on the model.

The model runs only after the words change: a debounced pass, as in the prototype's `tick()`. It
never runs during evaluation, during playback or while a render job runs (`_playing()`). Answers
persist in three places, all saved in the .blend:
- the node's `entries` JSON;
- the inner group;
- the table object, which holds every per-element column, whatever the table's size.

Neutral fallback outputs (§6.0) keep a tree valid before the first answer and without the model.

### 4.c Options are "outcome + when it applies"
Every answer set is a list of `name` + `when`, where `when` is a condition or description. This
covers Answers, Labels, Outcomes, each setting's answers in Dials, and Items when they are offered
as options. The model is asked which option's *condition applies*, not which name sounds right.
Bare names still work. The status line nudges "add 'when' descriptions for better answers" when
both of these hold:
- more than half of the options lack a description;
- the table's mean confidence is below 0.4.

Measured effect: biome mean confidence rose from 0.32 to 0.61, and Game of Life became exact.

### 4.d Code handles what's certain
Exact facts are resolved by code and never asked:
1. **`if` restrictions (Rules).** An outcome is offered only in cells whose category, flag or band
   levels satisfy its `if` expression.
   - One allowed outcome: the cell is **decided by code** (p = 1, confidence 1, no request).
   - No allowed outcome: the cell is an error, shown in the status line.
2. **Invalid IDs, empty material slots and unknown category values** map to fallbacks inside the
   Geometry Nodes graph.
3. **Locked overrides** (4.f) replace the model outright.
4. **A single-option question** is answered without scoring. The server already does this for
   `len(options) == 1`, and the pipeline short-circuits it so no request is sent.

### 4.e Every answer has a confidence; low confidence is flagged
Every row carries a confidence *c* in the range 0–1:

| Answer type | Confidence | Source |
|---|---|---|
| Yes/No | `c = abs(2p − 1)` | as in the prototype's `AskSpec.parse` |
| One-of | `c = p1 − p2` (top minus runner-up of the tournament-merged distribution, order-balanced when the node's toggle is on, §10.3) | server `_confidence()` |
| Amount | `c = clamp(1 − σ / σu(L), 0, 1)`, where σ is the standard deviation of the level distribution and `σu(L) = sqrt((L² − 1) / 12)` is the spread of a uniform answer over L levels | **new**; computed in Blender from the `probabilities` the server already returns |
| Decided by code, or locked | `c = 1` | shown as "decided" or "locked", never as "unsure" |

The Amount formula replaces the prototype's top-minus-runner-up because that measure misreads
ordinal answers that sit between two neighbouring levels:
- `[0, .45, .55, 0, 0]` scores 0.10 with top-minus-runner-up but 0.65 with σ;
- a split `[.5, 0, 0, 0, .5]` correctly scores 0 (σ = 2 exceeds σu = 1.41);
- a uniform answer scores exactly 0 for **any** number of levels.

The review showed that an earlier normalisation, `σmax = (L − 1)/2`, gave a uniform answer 0.29 at
five levels and 0.32–0.42 at six or more, so a know-nothing answer escaped the flag.

**One scale for one threshold.** All three formulas map a know-nothing answer to 0: `p = 0.5`, a
uniform choice, or a uniform scale. All three map a certain answer to 1. A single threshold
therefore means "close to no information" for every answer type.
- **User decision:** the default is 0.30, adjustable in the preferences (`unsure_below`), and tuned
  later from beta feedback.
- The calibration follow-up (§21 F3) records error rates per answer type. If they diverge, per-type
  default thresholds become hidden constants behind the one preference.
- `tests/rules_math_test.py` pins the uniform and near-uniform cases for L = 2…16.

A row with *c* below the **Unsure below** preference (default 0.30) is flagged in three places:
the node's status line ("· 3 unsure"), marked rows in the sidebar, and `unsure=1` in an exported
Overrides CSV. Confidence is also a node output (a single value or a field), so artists can show
it or gate on it in Geometry Nodes.

### 4.f The model drafts, the artist decides
Tables are visible: the sidebar inspects them, and they can be exported to the node's
**Overrides** CSV. They are also overridable:
- A row with `lock=1` in that file replaces the model's answer.
- A locked row is never asked again and survives unrelated edits.
- A lock whose row key no longer exists is reported as *orphaned*, never dropped silently.

The sidebar never edits a table (NG8). The file is the only override channel.

### 4.g The same list sources and answer types across nodes
- **List sources.** Pick, Match and Classify share one list-source dropdown (Text or CSV,
  Collection, Materials, Attributes), with identical semantics and ID addressing.
- **Answer sets.** Ask, Match, Classify, Rules and Dials share one answer-set format.
- **Answer types.** Yes/No, Amount and One-of have the same output conventions and fallbacks on
  every node.
- **Confidence.** One definition (4.e) applies everywhere.

Each of these is implemented once, in `lists.py`, `csvfmt.py` and `specs/base.py`. No node has a
private format.

### 4.h Nodes and files are the interface
The user's constraint: "everything to do with this happens in geo nodes; this is not a classical
add-on". Everything an artist authors lives in one of three places:
- **sockets**, typed on the node or driven from the modifier panel through Group Input;
- **node dropdowns**;
- **files the tree can read** (CSV, text).

The sidebar and preferences are secondary. They cover install/start/stop of the model, status,
inspection, starter templates and the one-time upgrade of old nodes. No per-file state lives
outside the .blend and its CSV files.

## 5. Architecture

### 5.1 Components

Package `openact/`. It replaces `jev_nodes/` and keeps the prototype's flat style. `engine.py`
(940 lines) is split along its existing seams.

| Module | Responsibility | Main entry points | Comes from |
|---|---|---|---|
| `__init__.py` | Registers prefs → ops → nodes → ui → `engine.register()`, then `server.refresh()` | `register()`, `unregister()` | `jev_nodes/__init__.py` |
| `prefs.py` | `OpenActPreferences`. Runtime root resolution: explicit `runtime_dir` → `…/OpenAct` → legacy `…/JevNodes` (§12). Backend enum, `unsure_below` | `get_prefs()`, `runtime_root()`, `runtime_installed()` | `prefs.py` |
| `nodes.py` | The six node classes on an `_OpenActNode` mixin; two hidden legacy aliases; Add menu | `GeometryNodeOpenAct*` | `nodes.py` |
| `engine.py` | Passes, handlers, timer, file watcher, per-node status, ownership fix-up | `run_pass()`, `tick()`, `schedule()`, `ensure_owned()` | `engine.py` (passes) |
| `evaluator.py` | Static evaluator and contexts, including tracing a plugged-in Import CSV node to its Path and Delimiter (§6.0) | `eval_input()`, `find_uses()`, `trace_csv_source()`, `Ctx`, `Unresolvable`, `Pending`, `Hint` | `engine.py` (evaluator) |
| `lists.py` | The four list sources → `ItemList([Item(name, when, row)])` | `resolve_items(node, ctx)`, `parse_inline()` | `parse_items`, `collection_items` |
| `csvfmt.py` | Pure-Python CSV reader (no `bpy`): header conventions, numeric/text split mirroring Import CSV, validation, `text_hash` | `read_csv(path) -> CsvDoc` | new |
| `specs/base.py`, `specs/{ask,pick,match,classify,rules,dials}.py` | Per node: wanted interface, `plan()` → `Job`, answers → rows, output spec for the builder | `Spec.interface()`, `Spec.plan()`, `Spec.rows()` | `AskSpec`, `PickSpec` |
| `pipeline.py` | `Cell`/`Job` model, per-cell `MEMORY`/`PENDING`/`FAILED`, priority queue, chunking, request build and parse | `submit()`, `cancel()`, `cached()` | `_submit`, `_on_answer` |
| `store.py` | Versioned `entries` JSON (schema 2, §14), override merge, legacy entries | `load(node)`, `save(node, store, live_gns=None)` (§13.3) | `get_entries`, `set_entries` |
| `inner.py` | Inner-group construction: interface sync, match chain, entry-level Index Switches, the per-element sampled lookup on the table object, Rules banding/interpolation/dither graph; value-only updates of entry-level values | `sync_interface()`, `build()`, `update_values()` | `_Builder`, `build_inner`, `create_inner` |
| `tables.py` | Hidden table objects (mesh of loose vertices plus attributes), linked to no collection and referenced only by Object Info (§8.5) | `ensure_table()`, `write_rows()`, `copy_for()` | new |
| `geometry.py` | Attribute names from an object's **original** data (the Attributes list source, §6.0); safe `GeometrySet` access (4.5+) for the agents hook, which keeps the set referenced and never chains on temporaries | `attribute_names(obj)`, `read_attributes(obj, names)` | new |
| `server.py` | Runtime lifecycle, installer, HTTP client with the launch token, decision worker thread | `start()`, `stop()`, `install()`, `Worker` | `server.py` |
| `migrate.py` | `jev_nodes` → OpenAct upgrade of nodes, inner groups and entries | `find_legacy()`, `upgrade_all()` | new |
| `ops.py`, `ui.py` | `openact.*` operators; sidebar tab "OpenAct"; preferences UI | — | `ops.py`, `ui.py` |
| `natsort.py` | `BLI_strcasecmp_natural` port | `sort_key` | unchanged |
| `runtime/openact_server.py` | Local model server: Engine (debias, tournament, cache) plus pluggable backends | HTTP endpoints (§5.2) | `runtime/jev_server.py` |
| `templates/*.csv` | Starter CSVs written by the sidebar's "New CSV from Template" | — | new |

### 5.2 Runtime split

```mermaid
flowchart TB
  subgraph B["Blender process (pure Python)"]
    direction TB
    M["Main thread<br/>timer tick 0.1 s, handlers, operators, draw<br/>all bpy access"]
    W["Decision worker 'openact-decide'<br/>1 thread, 1 request in flight"]
    X["Executor, 2 threads<br/>/health probes, start polling, installer"]
    M -->|"pipeline.submit(chunks)"| W
    W -->|"server.post(callback)"| M
    X -->|"server.post(status)"| M
  end
  subgraph R["Runtime process: runtime venv, Python 3.12"]
    direction TB
    H["ThreadingHTTPServer on 127.0.0.1:18765<br/>token check"]
    E["Engine: validate, debias,<br/>tournament, per-question LRU"]
    K["Backend: semif-torch (v1) or fake (tests)<br/>future: llama.cpp (CPU)"]
    G["GPU lock<br/>(one scoring call at a time)"]
    H --> E --> G --> K
  end
  W -->|"POST /v1/systemone"| H
  X -->|"GET /health, POST /shutdown"| H
  H -.->|"parent-pid watchdog: exits when Blender dies"| M
```

**Process model.** The prototype's model is unchanged:
- `server.start()` launches `venv/python -u runtime/openact_server.py`, with the script taken
  from the add-on folder;
- arguments: `--port <port> --backend semif --device <auto|cuda> [--gpu N] --parent-pid
  <blender pid> --token-file <state>/server-<port>.token`. v1 has no CPU device option at all:
  only NVIDIA/CUDA is built (NG9).
  - **In v1, `auto` means CUDA only.** The prototype's `Engine.load` resolved `auto` to `"cpu"`
    whenever `torch.cuda.is_available()` was False, and switched bf16 to fp32. `openact_server.py`
    instead reports `status: "error"` with "CUDA is not available to PyTorch (is the NVIDIA driver
    too old for cu128?)". The add-on shows that message.
  - It never loads the model on the CPU silently, which would take about 16 GB of RAM and miss
    every target (R3).
  - The fake backend's `--fake-no-cuda` flag lets `server_test.py` and `pipeline_blender.py` cover
    this path;
- environment: `HF_HOME`, `HF_HUB_OFFLINE=1`, `OPENACT_TOKEN`, and optionally
  `OPENACT_STATE_DIR`.
  - `OPENACT_STATE_DIR` holds **writable state**: `server.log`, `probe.json`,
    `server-<port>.token` and `logs/`. It defaults to the runtime folder.
  - Parallel build workers point it at their own temporary folder, so the shared runtime at
    `%LOCALAPPDATA%\JevNodes` stays read-only (§20.6).

`OPENACT_TOKEN` is new. It holds a 32-byte random launch token and is passed in the environment,
never in argv, which other local users can read.

**Blender stays pure Python.** It never imports torch. All HTTP goes through `urllib` with an
empty `ProxyHandler`, as in the existing `server._opener()`.

**API.** The request shape stays Jev-compatible, so hosted Jev or another compatible model could
be plugged in later (NG3).

| Endpoint | Change |
|---|---|
| `POST /v1/systemone` | Unchanged request and response shape: `state`, `questions` map of `{type: noul/choice/score, instructions, criteria}`, `options.debias` |
| `GET /health` | `server: "openact-semif"`. Adds counters: questions, scored rows, cache hits, mean ms per row |
| `GET /v1/models` | Unchanged |
| `GET /v1/capabilities` | **New**: `{backend, slots: 16, max_options: 255, max_levels: 16, batch_rows: 32, debias: true, question_cache: true}`. The pipeline sizes its chunks from this |
| `POST /shutdown` | Now requires the token |

`POST /v1/systemone` and `POST /shutdown` require the header `X-OpenAct-Token`. The server also
rejects requests that carry an `Origin` header or a `Host` other than `127.0.0.1:<port>` or
`localhost:<port>` (§16).

**Backends** are swappable behind one small interface in the server. The `Engine` code keeps
`_distributions`, `_plan`, `decide`, debiasing and the tournament, all backend-agnostic:

```python
class Backend(Protocol):
    name: str            # "semif-torch" (v1), "fake" (tests), later "llamacpp", "mlx"
    slots: int           # options scored in one row: 16 for SemIf letters A..P
    model_id: str        # e.g. "Qwen/Qwen3.5-4B@851bf6e8", reported by /health for provenance
    def load(self) -> None: ...
    def score_rows(self, rows: list[dict]) -> list[dict]:
        """SemIf rows that share one state -> [{"option_ids", "probabilities", "input_tokens"}]."""
```

- `semif-torch` wraps today's `load_causal_model` and `score`/`score_shared` calls, in chunks of
  `BATCH_ROWS = 32`.
- `fake` returns deterministic distributions derived from `sha256(state, question, option ids)`.
  It can optionally follow a JSON answer script. The whole Blender pipeline can therefore be
  tested without a GPU (§18).

**Status machine.** Unchanged (`unknown/not_installed/stopped/starting/ready/error/installing`),
with one new message: if `/health` on the port reports `server == "jev-nodes-semif"`, the status
becomes *error* with "An old Jev Nodes model server is running on this port: disable the Jev Nodes
add-on or choose another port."

### 5.3 Request pipeline

The prototype had one `Req` per node-context and one HTTP request per `Req`. v1 generalises this
into a **Job** per (node, context) that owns one **row** per table row. Each row is resolved from
one of four sources, in this order:
1. a **lock** from the Overrides CSV;
2. a **code decision** (4.d);
3. a cached model answer, looked up by **cell key**;
4. a **new question** sent to the model.

```python
@dataclass(frozen=True)
class Cell:                      # one model question; the unit of caching and batching
    key: str                     # sha1 of [kind, state, instructions, options, debias] (§5.4)
    state: str                   # shared situation text; cells of one state batch together
    question: dict               # Jev question: {"type": "noul"|"choice"|"score", "instructions", "criteria"}
    cost: int                    # scored rows: (2 if debias else 1) * (groups + (1 if options > 16 else 0))

@dataclass(frozen=True)
class Row:                       # one table row as Geometry Nodes will read it
    name: str                    # row identity: item name, level-name tuple ("alive|3") or setting name
    wkey: str                    # row words key: what the artist said for this row, before templating (§5.4)
    source: str                  # "lock" | "code" | "model" | "pending" | "stale" | "error"
    values: dict                 # e.g. {"p": 0.93} | {"label": 4} | {"amount": .62} + {"c": confidence}
    ask: str | None              # ask key the answer was produced with (templates, debias, backend), §5.4
    cell: Cell | None            # None for lock/code rows

@dataclass
class Job:
    uid: str                     # node uid
    gn: tuple[str, ...]          # raw word strings compared inside the inner group (match chain)
    words_key: str               # sha1 of gn, CSV text hashes, answer type, shape and every row's wkey (§5.4)
    shape: "TableShape"          # rows, output columns, factor layout (Rules), payload columns
    rows: list[Row]
    priority: int                # 0 interactive, 1 active node's table, 2 other tables (§5.5)
    generation: int              # bumped whenever the node's words change; stale chunks are dropped
    answer_type: str             # YES_NO | AMOUNT | ONE_OF, stored with the entry (§14.2)
    debias: bool                 # the node's balance_order: its requests carry it as options.debias (§10.3)
```

Every spec implements three pure functions. Only `plan()` sees `bpy`, and then only through the
evaluator, `lists` and `csvfmt`:

```python
class Spec:
    def interface(self, node) -> list[SocketDef]: ...     # sockets wanted for the current dropdowns/CSVs
    def plan(self, node, ctx) -> Job: ...                  # raises Hint / Pending / Unresolvable
    def rows(self, job, answers: dict[str, dict]) -> list[Row]: ...  # server answers -> table rows
```

```mermaid
sequenceDiagram
  autonumber
  participant A as Artist
  participant T as tick + handlers (main)
  participant P as run_pass (main)
  participant Q as pipeline queue
  participant W as decision worker
  participant S as openact_server
  A->>T: edits a socket or saves a CSV
  T->>T: schedule(), debounce 0.25 s (file watcher polls stat at 2 Hz)
  T->>P: run_pass() unless playing or rendering
  P->>P: ensure_owned(), find_uses(), spec.plan(node, ctx) per context
  P->>P: resolve rows: lock, code, MEMORY[cell.key], stored entry
  P->>Q: submit(job, missing cells, priority, generation)
  P->>P: store.save(node, store, live_gns) + tables.write_rows() (partial, stale-while-refilling)
  W->>Q: pop next chunk (one state, at most 32 questions / 64 rows)
  W->>S: POST /v1/systemone (X-OpenAct-Token)
  S-->>W: answers (probabilities, confidence)
  W->>T: server.post(on_chunk, chunk, answers)
  T->>P: MEMORY filled, schedule(immediate=True)
  P->>P: rows rewritten, table-object write (entry-level values: update_values)
  Note over P,S: Geometry Nodes re-evaluates through the depsgraph, the model is never involved
```

**Pass steps** (main thread; `engine.run_pass()`):
1. `iter_nodes()` yields every `GeometryNodeOpenAct*` node and every legacy alias in all
   non-inner Geometry Node trees.
2. `ensure_owned()` does the prototype's work: unique `uid`, one owned inner group per node. It
   now also gives each node its own table object when the node uses one (§8).
3. `find_uses()` maps every tree to its `Ctx` list. The limit stays at 64 contexts per tree.
4. For each context, the words are evaluated statically (§10) and hashed into a **fingerprint**:
   - the raw word strings;
   - `answer_type` and `list_source`;
   - `(path, delimiter, st_mtime_ns, st_size)` of every referenced CSV, whether typed as a path or
     traced from a linked Import CSV node (§7.7);
   - a list-source fingerprint: collection child names, material slot names, or attribute names;
   - the node's `balance_order` toggle (§10.3);
   - the file trust state (§16.1), which decides the files `plan()` may read. A pass works it out
     once at its start, and resolves each file's `realpath` once, not per fingerprint.

   `engine.PLANS[(uid, ctx key)]` holds `(fingerprint, job)`. An unchanged fingerprint reuses the
   cached Job, so the cost is the evaluator plus one hash. Only a changed fingerprint runs
   `spec.plan(node, ctx)`, which resolves the list, answer set and CSV documents and builds the
   Job. Nodes are marked dirty by socket updates, the file watcher and depsgraph updates.
   `SAFETY_PASS` only recomputes fingerprints (§10.6). Budget: ≤ 5 ms per unchanged node and ≤ 50 ms
   per idle pass (§3.2). A changed 4096-cell context costs about 120 ms of planning, once: two
   sha1-over-JSON keys per cell (its `wkey` and its cell key), measured by the PR-08 spike.
   Evaluation or planning can stop with one of:
   - `Hint`: shows a hint, as today.
   - `Pending`: another OpenAct node upstream has no answer yet.
   - `Unresolvable`: shows an error naming the socket.
5. **Rows are resolved** by the validity rule (§5.4):
   - A stored entry with the same `gn` **and** the same `words_key` is reused as it is, whatever
     its ask keys. This costs 0 requests. If a referenced CSV's `num_hash` changed, only its payload
     columns are re-baked (§7.8).
   - Otherwise every row is resolved in this order:
     1. lock;
     2. code;
     3. a row with the same `wkey` in the stored entry for this `gn`, or in the entry this context
        showed last (`LAST_SHOWN`, §14.5);
     4. `MEMORY[cell.key]`.
   - Everything still missing goes to `pipeline.submit()` in one call per Job.
6. **Entries** are updated in `store`: at most 16 **warm or cold** stored entries per node
   (`MAX_ENTRIES`), least recently used evicted. Live and pinned entries are exempt from that limit
   and never evicted (§8.1). The
   **materialised** set, i.e. the entries present in the Geometry Nodes graph and table object, is
   chosen as in §8.1, with the hard cap of 16. Then one of three things happens:
   - the table **shape** or the materialised set changed: `inner.build()`, at most once per pass;
   - only per-element values changed: `tables.write_rows()` (`foreach_set`, at most once per
     0.5 s during a fill);
   - only entry-level values changed (Ask and Pick single values): `inner.update_values()`. Dials
     values live in its one-row-per-entry table object, so they go through
     `tables.write_rows()` like per-element columns (§8.4).
7. The node's status is computed (§17): hint, waiting, filling *n/N*, complete (with unsure and
   locked counts), or error.

**Chaining.** The single-value outputs of Ask, Pick and Dials can feed another node's words. This
works exactly as in the prototype: the evaluator reads them from `MEMORY` or the stored entry, or
raises `Pending`. Field outputs (Probability, Label, Outcome and so on) can never be words. The
evaluator raises `Unresolvable("… is a field; words must be single values")`.

### 5.4 Per-context caching and versioning

| Layer | Where | Keyed by | Lifetime | Invalidated by |
|---|---|---|---|---|
| Stored entries | node `entries` JSON (saved). Only the **materialised** entries (§8.1) are mirrored in the inner group and table object | per context: `gn` (raw word strings) → `words_key` | the .blend | a `words_key` change for the same `gn` (the words' meaning changed; unchanged rows carry over by `wkey`), *Ask Again*, eviction beyond 16 entries. **Never** by template changes, a node's order-balancing toggle, or backend changes |
| Session cell cache | `pipeline.MEMORY`, LRU of 100 000 cells (~40 MB) | `cell.key` | Blender session | *Ask Again* (drops the node's cell keys) |
| In flight | `pipeline.PENDING` | `cell.key` → set of `(uid, generation)` | until answered | `cancel(uid)` drops queued chunks, never the in-flight one |
| Failures | `pipeline.FAILED` | `cell.key` → message | until the words change or *Ask Again* | — (no automatic retry, as today) |
| Server cache | `openact_server`, LRU of 65 536 **questions** (was 4096 whole requests) | sha256 of `[state, question, debias]` | server process | restart |

**Keys.** There are two families, which keep **what the artist said** separate from **how OpenAct
asked**. Both follow the prototype's `_key()` (sha1 over canonical JSON):

```python
# What the artist said: decides whether a stored answer is still valid
wkey      = _key("row", spec.name, answer_type, shared_words, row_facts, offered_options)
words_key = _key("words", gn, sorted(csv_text_hashes), answer_type, shape.digest(),
                 [row.wkey for row in rows], lock_digest)
# How OpenAct asked: decides only what gets cached, and labels rows
ask_key   = _key("ask", PROMPT_VERSION, debias, backend_model_id)   # debias = the node's balance_order
cell_key  = _key("cell", kind, state, instructions, options_in_order, debias)   # MEMORY and server cache
```

- **The inputs to `wkey` are the words before templating:**
  - `shared_words` is the Context and Question text;
  - `row_facts` is the item's name and `when`, or the row's factor level names and whens;
  - `offered_options` is the names and whens of the options offered to that row, in order,
    after `if` restrictions.
- **Validity rule.** A stored entry is reused while its `words_key` matches, **whatever its ask
  keys**. When the words change, each new row takes the answer of an old row with the same `wkey`
  (§5.3 step 5), so only rows whose own words changed are asked again.
- **The ask key never invalidates anything.** A row whose `ask` differs from the current ask key
  is only labelled "asked with older settings" in the sidebar. *Ask Again* (or a change of that
  row's words) refreshes it. This covers three cases:
  - an add-on upgrade with new templates: 0 requests;
  - toggling a node's *Balance Option Order* (`balance_order`, §10.3): 0 requests, with new asks
    using the node's current setting;
  - opening the file with another backend: 0 requests.

  The file keeps the answers it was saved with (G6).
- **`cell_key` is only a cache key.** It holds the exact text sent (`state`, `instructions`, the
  option ids and description strings, `debias`), so `MEMORY` and the server never return an answer
  to a different question. Restricted Rules cells therefore get different cell keys from
  unrestricted ones.
- **Provenance.** `PROMPT_VERSION` and model provenance are stored per entry (`asks` list, `model`)
  and shown in the sidebar (§14.2).
- **Tests pin the rule:**
  - template bump → 0 requests;
  - toggling a node's *Balance Option Order* → 0 requests until the words change;
  - upgrade of a v0.1.0 file → 0 requests.

  They live in `store_test`, `pipeline_blender` and `migrate_blender` (§18).
- Placing each cell's facts in `instructions` and the shared situation in `state` (§10) gives
  every table exactly one state. Batching is maximal, and editing one factor level's `when` only
  changes the cells that include that level.

**CSV versions.** `csvfmt.read_csv()` returns a `CsvDoc` with two hashes:
- `text_hash`: header order, row count and the normalised text columns (`answer`, `when`, `if`,
  descriptions and so on);
- `num_hash`: the numeric columns.

Cell keys only ever contain text, so **editing numeric columns never re-asks the model** (G3).

The file watcher in `engine.tick()` works as follows:
- At most twice a second it calls `os.stat` on every CSV that a planned Job referenced and
  compares `(st_mtime_ns, st_size)`. Missing and unreadable files are errors.
- On a change it re-reads the file. If `text_hash` changed, it schedules a pass, and the affected
  cells are re-asked.
- If only `num_hash` changed, it re-bakes the payload columns: `tables.write_rows()` for
  per-element payload, and `inner.update_values()` for Pick's entry-level payload. No request is
  sent and no graph is rebuilt.
  Import CSV is never involved, because it caches by path (§7.1).
- Files larger than 1 MB, or with more than 4096 data rows, are rejected with an error before
  parsing.

**Partial tables (stale-while-refilling).** A table fills in chunks. After each chunk the pass
rewrites the table, so it always holds the best answer known so far:
- Rows answered by a lock, by code, by carry-over (same `wkey`) or from the cache have their final
  values. Example: when you edit the `when` of one outcome, every row that doesn't offer that
  outcome carries over unchanged.
- A row that is not answered yet takes the value of the row with the **same row name** in the
  entry this context showed last (`LAST_SHOWN`, §14.5), marked `stale`. Example: the rows that do
  offer the edited outcome keep showing their old answers until the new ones arrive.
  - After reopening a file, `LAST_SHOWN` is **seeded on the first pass** from the stored entry
    whose `gn` equals the context's words. A CSV edited while Blender was closed therefore also
    shows stale values while it refills, not fallbacks.
- If no previous value exists, the row gets the neutral fallback (§6.0) and is marked `pending`.
- The status reads "Filling 12/50 · ~2 s". The sidebar shows stale and pending rows greyed out.

```mermaid
stateDiagram-v2
  [*] --> Planned: spec.plan(node, ctx)
  Planned --> Complete: every row is lock, code or cached
  Planned --> Filling: missing cells submitted
  Filling --> Filling: chunk answered, partial table written
  Filling --> Complete: last chunk answered
  Filling --> Superseded: words changed (generation bumped, queued chunks dropped)
  Filling --> Failed: HTTP 422/500, or model not ready
  Failed --> Planned: words change, or Ask Again
  Complete --> Planned: words change, CSV text change, or Ask Again
  Superseded --> [*]
```

**Table size limits** (per node context):

| Limit | Value | Behaviour |
|---|---|---|
| Items (Pick, Match, Classify), options (One-of) | 255 | error above; 17–255 options run as a tournament |
| Amount levels | 2–16 | error outside |
| Cells (Rules: product of level counts) | 4096 | error above: "Rules has 5 400 cells (limit 4 096): use fewer bands" |
| Estimated scored rows (sum of `cell.cost`) | 8192 | error above; the status shows an ETA from the measured rows/s above 1024 rows |
| Stored entries per node | 16 warm or cold entries (`MAX_ENTRIES`), plus every live and pinned entry, which are exempt | least recently used warm or cold entries evicted, also by the 2 MB soft cap; live and pinned entries never evicted, and kept even above 2 MB with a size note; the 4 MB per-entry hard cap and the 12-pin limit bound the total (§8.1, §14.7) |
| Materialised entries per node | at most 16 (`MATERIALISED_MAX`): every live context, then at most 12 pinned, then warm entries while the total is below 4 | more than 16 distinct live word sets: an error, and the extra contexts show neutral values. If live plus pinned grows past 16 after pinning, the pins whose word tuples start latest in the frame range go inactive (pin order, §5.5), with a status message. The rest stay stored in JSON only (§8.1) |

The ETA uses the server's measured rate: `/health` reports `ms_per_row`, with defaults of 21 ms
per two-option row and 36 ms per seven-option row, from the 3090 figures (24 and 14 cells/s with
debias).

### 5.5 Threading and scheduling

| Thread | Owns | Never does |
|---|---|---|
| **Main** (timer `tick` every 0.1 s, handlers, operators, draw) | All `bpy` access: passes, evaluator, CSV reads (at most 1 MB, synchronous), `store`, `inner`, `tables`, status, redraw | Network I/O, waiting on the server |
| **Decision worker** `openact-decide` (one `threading.Thread`, daemon) | Pops chunks from the priority queue, sends one HTTP request at a time, and posts results with `server.post()`. Reads two `threading.Event`s that `tick()` publishes on the main thread: `render_running` and `playing` | Touching `bpy`, including `bpy.app.is_job_running`; more than one request in flight |
| **Executor** (`ThreadPoolExecutor(max_workers=2)`, was 4) | `/health` probes, start-up polling, installer steps and uv/pip subprocesses | Decision requests (moved to the worker) |
| Server: HTTP threads, `load` thread, watchdog, `Engine.gpu` lock | Scoring, one call at a time | — |

**Why one request in flight.** The server already serialises scoring on `Engine.gpu`, so parallel
requests only queue on the server where the add-on cannot reorder them. Keeping the queue on the
Blender side, with one request outstanding, lets a new interactive question overtake a large table
fill after at most one chunk:
- about 1.3 s for 32 yes/no questions;
- at most about 2.5 s for choices.

**Priority classes.** Chunks are popped in class order. Within a class, jobs are served round-robin
in submission order, so one 4096-cell table cannot starve a 50-cell one.

| Class | Work | Target |
|---|---|---|
| P0 | Ask, Pick, Dials, and any job with at most 8 missing cells | ≤ 0.6 s on an idle queue (§3.2) |
| P1 | Tables of the active node in the focused node editor | first partial result ≤ 2.5 s |
| P2 | Every other table | throughput |

**Chunking.**
- A chunk holds missing cells of **one state** (one job): at most 32 questions and at most 64
  scored rows. With debias that is two `score_shared` batches of 32 rows.
- A question with more than 16 options is costed as `2 × (groups + 1)` rows. A 255-option
  tournament costs about 34 rows and travels alone or with cheap questions.
- Chunk limits come from `/v1/capabilities` when it is available. Otherwise the defaults above
  apply.

**Cancellation.**
- A node's generation is bumped, and `pipeline.cancel(uid, below=generation)` removes its queued
  chunks, when:
  - a context's words change: a new fingerprint, or a text edit of a CSV it reads (§7.8);
  - a context goes away while the node stays: its object is deleted, its modifier is removed,
    renamed or pointed at another group, or a group node on its path is muted. The next pass
    queues the missing cells of the contexts left again, at the new generation.
- A deleted node's queued chunks all go (`cancel(uid, below=inf)`), and so does every queued chunk
  when another file is opened or the add-on unregisters (`cancel_all()`).
- The in-flight chunk always completes. Its answers land in `MEMORY`, where they remain useful:
  undo, or switching back to the previous words.

**Playback, rendering and undo.**
- **Playback.** Passes stay paused (`_playing()`, unchanged), and so does the CSV file watcher's
  pass. The worker may finish queued chunks, but tables are written only after playback stops
  (NG1).
  - **Rule:** words that change *during* playback (a keyframed or driven modifier input, such as a
    rule-set Menu Switch, or a CSV edited while playing) must have been answered beforehand.
  - **Two mechanisms keep simulations alive:**
    1. **On a node used by exactly one live context**, a no-match falls back to that context's
       last-shown entry (`k_default`, §8.2), not to the neutral fallbacks. A single-object sim,
       such as the hero demo, keeps its own last rules instead of reading `alive = −1`. On shared
       trees a no-match gives neutral values, so no object ever runs on another's rules.
    2. **Prepare for Playback** (`openact.prepare_playback`) collects every distinct word tuple
       each context will see over the frame range, answers them, and **pins** those entries
       (§8.1). It saves with `store.save(node, store)` (`live_gns=None`), which evicts nothing
       (§13.3).
  - **How Prepare for Playback reads future words without evaluating the scene:**
    1. The evaluator records the **read set** of each context: every modifier input and
       node-socket default a word depended on.
    2. **F-curve path (default).** Only numeric and menu values can be keyframed; String socket
       values cannot.
       - **Finding the F-curves.** `Action.fcurves` no longer exists in 5.x (review, round 3).
         F-curves are found through the owning ID's `animation_data.action` and `action_slot`,
         with `bpy_extras.anim_utils.action_get_channelbag_for_slot(action, slot)`. That call
         exists on 4.5, 5.0 and 5.1, so no `compat.py` switch is needed.
         - The owning ID is the object for modifier inputs (`modifiers["…"]["Socket_n"]`) and the
           node tree for socket defaults.
       - **Evaluating.** For each frame in `frame_start…frame_end`, every animated value in the read
         set is replaced by `fcurve.evaluate(frame)`. F-curve modifiers apply, because `evaluate()`
         includes them. The words are then re-evaluated with those overrides, purely in Python,
         with no scene evaluation.
       - **Cost:** 250 frames × a few words is well under 1 s.
    3. **Fallback when the active action's F-curves don't tell the whole story.** It falls back to
       `scene.frame_set()` stepping if, for any ID in the read set:
       - a read value is **driven**;
       - the ID has **NLA tracks**;
       - the action is in **tweak mode** (`animation_data.use_tweak_mode`);
       - the action is **blended** (`action_blend_type != 'REPLACE'` or `action_influence < 1`).

       The fallback runs as a **modal operator** with progress in the status bar and **Esc to
       cancel**. Cost is frames × one scene evaluation, e.g. 250 × 50 ms ≈ 12 s, shown up front as
       an estimate. Cancelling pins nothing.
    4. **Afterwards.** The current frame is restored. Pinning changes the inner group, which
       invalidates unbaked simulation caches. Baked simulations are flagged "re-bake: rules changed
       for frames …".
  - **Limits and overflow.**
    - At most 12 pinned entries, and live plus pinned at most 16 (§8.1).
    - If the range yields more distinct word tuples, it **refuses without asking anything**: "40
      distinct word sets over frames 1–250; the limit is 12".
    - A new run replaces the previous pins, so pins expire only then, or through `openact.unpin`
      (the sidebar's per-entry *Unpin* and *Unpin All*).
    - Pinned entries are saved with `pinned: true` and their `frames` (§14.2), so they survive save
      and reload.
    - **Pin order is deterministic: by the first frame at which each word tuple occurs**, with ties
      broken by tuple order. It is used when live plus pinned later exceeds 16 (§8.1): the pins
      with the earliest first frames stay materialised, and those that start latest go inactive.
      Every pin from one run shares the same pin time, so time could not order them.
- **Rendering.** While `render_running` is set, the worker starts no new chunk, so the model does
  not compete with Cycles or EEVEE for the GPU. `tick()` sets the event from
  `bpy.app.is_job_running('RENDER')` on the main thread, and the worker never calls `bpy`. The
  in-flight chunk completes.
- **Undo and redo.** Entries JSON, inner groups and table objects are ordinary data inside undo
  steps. `on_undo_redo` schedules an immediate pass, and `MEMORY` makes re-applying answers free.

**Headless and tests.** With `bpy.app.background` set, as in a render farm's `blender -b -f`,
OpenAct renders the saved answers and does not re-plan or re-bake from files (§7.8, R23). Headless
**authoring** opts back in, through `OPENACT_BACKGROUND_PASSES=1` or in process through
`openact.engine.allow_background_passes(True)`; `demos/build_demos.py` uses the latter. The test
suites then drive `engine.tick()` in a loop, as `tests/e2e_blender.py::drive()` does today.

### 5.6 Hooks kept open for the agents design (NG1)

v1 ships these as internal APIs. They are not user features, but the later agents document builds
on them:
- **`pipeline.submit(job, priority, on_rows=callback)`** accepts jobs from producers other than
  nodes. A future background agent loop can queue decisions and receive rows asynchronously.
- **Table objects** (`tables.write_rows()`) can be rewritten without rebuilding any node graph. A
  write is a `foreach_set` on attributes followed by a depsgraph tag. Decisions can therefore land
  while Geometry Nodes plays; v1 simply doesn't do that yet.
- **`geometry.read_attributes(obj, names)`** wraps `GeometrySet.from_evaluated_object` safely: it
  keeps the GeometrySet referenced and never chains `.mesh` on a temporary. v1 itself does not use
  it: the Attributes source reads original data (§6.0). The agents loop will use it to read
  per-element state.
- The server answers many questions per request and **caches per question**, which is the
  traffic pattern an agent loop produces.

## 6. The Six Nodes

### 6.0 Conventions shared by all nodes

| Node | `bl_idname` | Label | Add menu group | Icon | Inner group |
|---|---|---|---|---|---|
| Ask | `GeometryNodeOpenActAsk` | Ask | Whole situation | `QUESTION` | `.OpenAct Ask <uid>` |
| Pick | `GeometryNodeOpenActPick` | Pick | Whole situation | `RESTRICT_SELECT_OFF` | `.OpenAct Pick <uid>` |
| Dials | `GeometryNodeOpenActDials` | Dials | Whole situation | `OPTIONS` | `.OpenAct Dials <uid>` |
| Match | `GeometryNodeOpenActMatch` | Match | Each item | `CHECKMARK` | `.OpenAct Match <uid>` |
| Classify | `GeometryNodeOpenActClassify` | Classify | Each item | `SORTALPHA` | `.OpenAct Classify <uid>` |
| Rules | `GeometryNodeOpenActRules` | Rules | Each situation | `PRESET` | `.OpenAct Rules <uid>` |

**Class and menu.**
- All six nodes subclass `bpy.types.GeometryNodeCustomGroup` through the `_OpenActNode` mixin.
  The mixin keeps the prototype's behaviour: hidden `uid` and `entries`, `init()` creates the
  inner group, `copy()` only clears `uid`, `free()` only drops status, and `update()` and
  `socket_value_update()` call `engine.schedule()`.
- Default width is 240 and minimum 160. `poll()` rejects inner trees (name prefix `.OpenAct `).
- Add menu: *Shift A > OpenAct*, with the three labelled groups above.

**Node body.**
- `draw_buttons` shows the dropdowns (`answer_type`, `list_source`) and one status line, e.g.:
  - `✓ 18 cells · 0 unsure · 2 locked`
  - `⏳ Filling 12/50 · ~2 s`
  - `ⓘ Type a question`
  - `⚠ life_outcomes.csv row 3: 'when' is empty`
- `draw_buttons_ext` (the Node tab and the OpenAct sidebar) shows:
  - the full status and one summary per context;
  - a table preview (§17);
  - *Ask Again*, *Write Overrides CSV* and *Open File*.

**Dropdowns.** Changing one runs `inner.sync_interface()` and `engine.schedule(immediate=True)`.
- `answer_type`: `YES_NO` "Yes / No", `AMOUNT` "Amount", `ONE_OF` "One of". Each node offers a
  subset.
- `list_source`: `TEXT` "Text or CSV", `COLLECTION`, `MATERIALS`, `ATTRIBUTES`.

**Checkbox: *Balance Option Order*** (`balance_order`, every node; user decision). It is on by
default, seeded from the `debias` preference when the node is created.
- On, each question is scored with its options in both orders (2× cost, §10.3).
- Off, it is scored once, for work that doesn't need the double check.
- Toggling it sends no requests and keeps stored answers (they belong to the ask key, KD13). New
  asks, and *Ask Again*, use the node's current setting.

**Input kinds.**

| Kind | Read by | Examples |
|---|---|---|
| **word** | the static evaluator (§10). May be typed, or linked from Group Input (modifier panel), String, Join Strings, Menu Switch, Import Text, another OpenAct node's single-value output… | Question, Prompt, Context, Items, Collection, Object, Answers, Labels, Outcomes, Factors, Settings, Overrides, and `Key` (hidden by default; compared, never sent to the model, §6.0) |
| **field** | Geometry Nodes only, per element; never statically evaluated | ID, Rules factor sockets |
| **GN value** | Geometry Nodes only; single value or field; Python ignores it | Threshold (Match), Dither and Seed (Rules) |

**ID input.**
- An `Int` input with `default_input = 'INDEX'` and `hide_value = True`. Unlinked, it means
  "Index".
- Entry *i* is addressed by ID *i*. IDs outside `0…N−1` give the fallbacks below.
- **Verified** in 4.5.2 and 5.1.1: an unconnected ID on the custom group node evaluates to the
  element index.
- `ID_OR_INDEX` also exists, but it is deliberately **not** used. An ID means "entry number".
  Points made by Distribute Points on Faces carry a large random stable `id`, which is not an
  entry number. Rules dithering, however, uses Random Value's own implicit ID-or-index, which
  stays stable per element (§9).

**Fallback values.** They are used before the first answer, for pending rows without a stale
value, for invalid IDs and for rows in error:

| Type | Float | Bool | Int (Index, Label, Outcome, One-of setting) | String | Geometry | Material |
|---|---|---|---|---|---|---|
| Fallback | 0.0 | False | −1 | "" | empty | none |

**List sources** (Pick, Match, Classify; `lists.resolve_items()`):

| `list_source` | Socket | Entries (ID order) | Pick's extra output |
|---|---|---|---|
| Text or CSV | `Items` (String) | Inline entries, or the data rows of a `.csv` | one output per numeric CSV column |
| Collection | `Collection` (Collection) | Child objects and child collections mixed, sorted with `natsort.sort_key`: the Collection Info (Separate Children) instance order | `Instance` (Geometry) |
| Materials | `Object` (Object) | The object's material slots in slot order, so ID = material index. Empty slots keep their index but are never offered as options | `Material` (Material) |
| Attributes | `Object` (Object; empty means the modifier's own object) | Attribute names of the object's **original data** (`obj.data.attributes` for meshes, curves and point clouds, via `geometry.attribute_names()`), natsorted, excluding names that start with "." and names this tree itself writes (a static scan of Store Named Attribute *Name* inputs, nested groups included). Never the evaluated geometry, which contains this node's own output | none; `Item` carries the name |

**Answer-set format.** Answers, Labels, Outcomes and each setting's answers use one format
(`lists.parse_inline()`, `csvfmt`):
- **Inline text.** Split on newlines if there are any, else on `;` if present, else on `,`. The
  prototype's rule is extended with `;` because Blender's single-line socket field cannot hold a
  newline, while `when` texts contain commas.
  - Within an entry, the first `": "` separates `name` from `when`. For example:
    `yes: it is cold, wet or windy; no: warm and calm`.
- **CSV path.** A single-line value whose stripped text ends in `.csv` (any case). It is resolved
  with `bpy.path.abspath(path, library=tree.library)`, so it is relative to the .blend, or to the
  library for linked groups.
  - Canonical columns: `answer`, `when`, `if`, plus numeric payload columns.
  - `name`, `item`, `label` and `outcome` are accepted as synonyms of `answer`, and `description`
    of `when` (§7).
- `if` restrictions and payload columns exist only in CSV form.

**Payload outputs.** For One-of answers, every numeric column of the CSV that supplies the chosen
row becomes an extra output named after its header:
- the Items CSV for Pick, looked up at `Index`;
- the Labels CSV for Classify, looked up at `Label`;
- the Outcomes CSV for Rules in One-of mode, looked up at `Outcome`.

An INT column becomes an Int output and a FLOAT column a Float output. Python parses the values
(`csvfmt`) and **bakes** them into the node's table as extra columns: table-object attributes for
per-element outputs, or entry-level Index Switch items for Pick's single-value payload (§8). Geometry Nodes then reads them through the same per-element lookup
as the answer.

The inner group contains no Import CSV node, because Import CSV caches by path (§7.1). Reserved
names (`answer`, `when`, `if`, `lock` and their synonyms) never become outputs.

**Overrides.** Every node (Ask, Pick, Match, Classify, Rules and Dials) has an optional
`Overrides` word input: a path to a CSV whose rows are keyed by row name, or by the question or
prompt for Ask and Pick. Rows with `lock=1` win over the model (4.f, §7).

**Dynamic sockets.** `inner.sync_interface(node, spec.interface(node))` diffs the inner group's
interface and never clears it:
- Items are matched by `(in_out, name, socket_type)`, so their identifiers, and the links to them,
  survive.
- Obsolete items are removed, and order is fixed with `interface.move()`.
- Tooltips come from `item.description`, as in the prototype's `create_inner`.
- **Socket shapes come from `structure_type`.** Allowed values differ per version (review
  measurement, round 2, on 4.5.2, 5.0.0 and 5.1.1):
  - **4.5:** input sockets accept only `AUTO` and `SINGLE`; the feature sits behind the
    experimental flag `use_socket_structure_type`.
  - **5.0 and 5.1:** `FIELD` is also accepted for Int, Float and Bool sockets, but not for String.

  So `sync_interface()` does this, with the version switch in `compat.py` (§11):
  - Word inputs, all String, are `SINGLE` on every version. It is harmless, and draws the right
    shape.
  - ID and factor inputs are `FIELD` on 5.0+ and `AUTO` on 4.5.
  - Threshold, Dither and Seed stay `AUTO`.
- **Blender does not flag a string field linked into a `SINGLE` String input.** On all three
  versions such a link stays `is_valid = True`; only a float field into a `SINGLE` Float input is
  flagged, on 5.1. The evaluator's `Unresolvable("… is a field; words must be single values")` is
  therefore the **real guard** against fields linked into words (§10.1).
- **Verified** in 4.5.2 and 5.1.1: custom group nodes follow inner-interface edits immediately,
  and existing links stay valid. A regression test in §18 guards this on 4.5, 5.0 and 5.1.

**Why Attributes reads original data.** The evaluated object is the result of the whole modifier
stack, including the modifier that contains this node. Its attribute list would therefore depend
on the node's own answer, and could re-ask or even oscillate. Attributes created by *earlier*
modifiers are not listed. To pick among those, point `Object` at another object that has them, or
type the names (Text or CSV source).

**How a node gets its CSV: a typed path, or Blender's own Import CSV node** (user decision).
- **Sockets.** Every CSV-capable String input has a companion **Geometry** input named
  `<Input> CSV`, placed directly below it:

  | Node | CSV inputs |
  |---|---|
  | Pick, Match, Classify (Text or CSV source) | `Items` + `Items CSV` |
  | Ask, Match | `Answers` + `Answers CSV` |
  | Classify | `Labels` + `Labels CSV` |
  | Rules | `Factors` + `Factors CSV`; `Outcomes` + `Outcomes CSV` |
  | Dials | `Settings` + `Settings CSV` |
  | all six nodes | `Overrides` + `Overrides CSV`, hidden by default like `Key` |

  The artist either **(a)** types a path into the String input, or links it or exposes it on the
  modifier, or **(b)** plugs an **Import CSV** node's output into the companion socket.
- **What OpenAct reads in case (b).**
  - OpenAct follows the link to the Import CSV node and reads its **Path** and **Delimiter**
    inputs with the static evaluator. Those inputs are found by socket identifier; §18 pins the
    identifiers. A path exposed on the modifier through Group Input therefore works too.
  - It then parses the file itself in Python: text columns kept, edits always seen (§7.8).
  - The Import CSV geometry is **never used**. Inside the inner group the companion socket is left
    unconnected, so Geometry Nodes' lazy evaluation never runs the Import CSV node for OpenAct.
  - Blender's Import CSV path cache therefore doesn't affect OpenAct.
- **Precedence.** If the companion socket is linked, **the link wins**. A non-empty String on the
  same input is then ignored, with the warning "Outcomes: using the linked Import CSV; the typed
  path is ignored". An explicit wire is the most visible intent.
- **Links traced** (`evaluator.trace_csv_source()`, §10.1):
  - Reroute nodes;
  - muted links, which count as unlinked;
  - nested group boundaries: into a group node's output through its Group Output, and out through
    a Group Input to the parent node's socket, as `_group_input` does for words;
  - selector nodes whose selector resolves statically: Switch, Index Switch and Menu Switch of
    type Geometry. A modifier dropdown can therefore choose among several Import CSV nodes.
- **Errors**, shown on the node and naming the socket:
  - the trace ends at anything other than `GeometryNodeImportCSV`, e.g. a Points or Grid node:
    "Outcomes CSV: connect an Import CSV node (found 'Grid')";
  - a node that modifies the point cloud sits in between (Transform Geometry, Set Position, Join
    Geometry, Store Named Attribute, …): "Outcomes CSV: connect the Import CSV node directly
    ('Transform Geometry' is in between)";
  - the Import CSV node's Path is empty: a hint;
  - a selector that cannot be resolved statically: `Unresolvable`, as for words.
- **Delimiter.** In case (b) the Import CSV node's Delimiter is passed to `csvfmt.read_csv(path,
  delimiter=…)`, and header sniffing (§7.2) is skipped. In case (a) the sniffing applies.
- **Same rules otherwise.** The traced path gets the same trust confinement (§16.1) and file
  watcher (§7.8) as a typed path. It resolves as its Import CSV node resolves it (§7.7): `//` is
  relative to the file that holds the node's tree, and any other relative path to Blender's
  working folder.
- **Limitation: the match chain can't see a traced path.** It is not a String input of the node
  (§8.2). Contexts that need *different* files must therefore either use route (a), exposing the
  path String on the modifier, or give each context a different `Key` (above). Otherwise they get
  "only one answer can show".

**Shape agreement across contexts.** Words may differ per context, but the interface exists once
per node.
- **What every live context must agree on:**
  - `list_source` and `answer_type`;
  - for Rules, factor names, kinds and level counts, **and every per-factor value baked into the
    graph as a constant**: custom band edges and band centres (§9.2, §9.6), a contiguous category's
    offset `u_0`, a non-contiguous category's value map, and a flag's value → level assignment. Only
    the level *names* and *whens* may differ, because they are words;
  - for Dials, setting names and types;
  - payload headers and types.

  Row counts may differ, because each entry's rows are addressed through its own base and count
  (§8.3).
- **What happens otherwise.** The node shows an error naming the conflicting contexts, e.g.
  "Rules: 'Cube' and 'Plane' use Factors CSVs with different factors". It keeps its last good
  table and does not rebuild.
- **While a context has no job.** A context has no job while it waits for another node, has an
  error of its own, or shows a hint while its words still name a stored entry (an Import CSV node
  without a Path, an empty collection: a path traced to Import CSV and a datablock are not words,
  §8.2). It still shows its last table, so the node keeps its shape: sockets, links and, for
  Rules, the baked layout. Other contexts are answered if their jobs fit that shape. A job that
  would change it waits, asking and storing nothing, until that context plans again or goes away;
  then it is answered, or gets the error above. Meanwhile the status shows that context's error,
  wait or hint. A hint whose words name no stored entry (e.g. a new object with an empty Prompt)
  holds nothing.
- **What may differ, because it is per-entry data** (in the table object, or selected by `k`):
  - option and level texts, and payload **values**;
  - Dials `min`/`max` (§8.6);
  - Pick's picked material and instance (§8.6).
- **Collection and Object sockets** cannot be compared by the match chain (§8.2). Two contexts
  whose strings are identical but whose collection or object differs therefore get the error "only
  one answer can show". **Remedy: the `Key` word.**
  - Every node has an optional String input `Key`, **hidden by default** (Ctrl H shows it).
  - The match chain compares it, but it is **never sent to the model** and is not part of any
    `wkey` (§5.4).
  - Expose it on the modifier (Group Input → Key) and type a different value per object. The
    contexts become distinguishable without changing any answer or splitting the cell cache.
  - Changing a Key re-uses cached cells (0 requests).
  - Do **not** put such a discriminator into Context, because the model would see it.

### 6.1 Ask: the whole situation, Yes/No or Amount

Property `answer_type` ∈ {`YES_NO` (default), `AMOUNT`}. It replaces the prototype's `mode`
enum; migration copies the value.

| Socket | Kind | Type | Default | Meaning |
|---|---|---|---|---|
| Question | word | String | "" | "Is it night?" / "How weathered should the stone look?". Empty → hint "Type a question" |
| Context | word | String | "" | Facts the question is about. Empty → state "No additional context." (prototype `NO_CONTEXT`) |
| Answers | word | String | "" | Optional answer set (§6.0). Yes/No: entries named `yes` and `no` whose `when` texts become the noul criteria. Amount: 2–16 ordered levels, lowest first. Empty → `Yes.`/`No.`, or the prototype's `AMOUNT_LEVELS` (Not at all, Slightly, Moderately, Very, Extremely) |
| Overrides | word | String | "" | Optional Overrides CSV with one row per question, keyed by `question` (§7.6). A locked row replaces the model's answer for that question |
| **Answer** | output | Float, single | 0.0 | Yes/No: P(yes). Amount: `score / (L − 1)`, the probability-weighted level position |
| **Yes** | output | Bool, single | False | `Answer ≥ 0.5` |
| **Confidence** | output | Float, single | 0.0 | 4.e |

**Behaviour.**
- There is one question per context, forming a one-row table. It runs at priority P0.
- The inner group is the prototype's: a match chain over (Question, Context, Answers, Overrides,
  Key) (§8.2) feeding
  Switch(INT), then one Index Switch per output.
- The prototype's `AskSpec` moves unchanged into `specs/ask.py`. `Answers` extends it, and the
  new Amount confidence (4.e) replaces top-minus-runner-up.

**Request** (one chunk):

```json
{"state": "A snowy mountain village at night, warm windows glowing.",
 "questions": {"answer": {"type": "noul", "instructions": "Is it night?",
   "criteria": {"true": "the scene takes place between sunset and sunrise",
                "false": "daylight, including dawn after sunrise"}}},
 "options": {"debias": true}}
```

**Examples.**
- **Night switch.** Put the scene description on the modifier panel (Group Input → Context) and
  set Question to "Is it night?". Yes → Switch between day and night lighting geometry. One tree
  on many objects gives one answer per object, as in the prototype's e2e test.
- **Weathering amount.** Amount mode, "How weathered should the stone look?", Context "An
  abandoned 18th-century lighthouse". Answer (about 0.8) → Mix factor of an edge-wear mask.
- **Custom rubric.** Amount mode, Answers `none: bare ground; dusting: a thin white layer; patchy:
  snow in patches; thick: most ground covered; buried: deep snow everywhere`. Answer drives snow
  displacement.

### 6.2 Pick: the whole situation, one entry of a list

Property `list_source` (default `TEXT`). There is no `answer_type`: Pick is always One-of.

| Socket | Kind | Type | Default | Meaning |
|---|---|---|---|---|
| Prompt | word | String | "" | What to pick: "something to sit on" |
| Context | word | String | "" | **New.** Optional situation: "a beach bar at noon". Empty → the prototype's `PICK_STATE` |
| Items / Collection / Object | word | per §6.0 | — | The list, swapped by `list_source`. Inline items may carry descriptions (`Pine: grows in cold places`) |
| ID | field | Int | Index | Which entry the per-element outputs describe |
| Overrides | word | String | "" | Optional Overrides CSV with one row per prompt, keyed by `prompt` (§7.6). A locked row forces the picked item by name; its probability becomes 1 and the others 0 |
| **Index** | output | Int, single | −1 | Position of the picked entry (natsort order for collections) |
| **Item** | output | String, single | "" | Name of the picked entry |
| **Confidence** | output | Float, single | 0.0 | 4.e (top minus runner-up) |
| **Selection** | output | Bool, field | False | `ID == Index` |
| **Probability** | output | Float, field | 0.0 | P(entry `ID`) from the stored distribution; 0 outside `0…N−1` |
| Instance | output | Geometry | empty | `list_source = COLLECTION`: the picked child, via Collection Info → Delete Geometry(Index ≠ picked), as in the prototype |
| Material | output | Material | none | `list_source = MATERIALS`: the picked slot's material, via Index Switch of type `MATERIAL` (verified) |
| *payload columns* | output | Int/Float, single | fallback | `list_source = TEXT` with a `.csv`: baked numeric columns, looked up at Index |

**Behaviour.**
- There is one `choice` question per context, with at most 255 options (a tournament above 16).
  The instruction is the prototype's: "Which item best fits: `<prompt>`?".
- Option descriptions come from `when`, formatted as `"<name>: <when>"`.
- Single-value outputs come from the match chain. Selection and Probability read a per-row table
  of the stored probabilities through ID. That table lives in the node's table object (§8). The
  object is created only while Selection or Probability is linked; without it, those outputs give
  their fallbacks.
- Pick adds no model cost over v0.1.0: the probabilities were already returned and stored.
- Pick's answer can be forced through its Overrides CSV (§7.6). An earlier draft left Ask and Pick
  without overrides, which contradicted principle 4.f.

**Examples.**
- **Furniture for a scene.** Collection "Props", Prompt "something to sit on", Context from the
  modifier. Instance → Instance on Points.
  - Collection Info (Separate Children) → Scale Instances by `Probability` (ID = Index) shows how
    well every prop fits.
  - `Selection` → Delete Geometry keeps only the pick.
- **Material by name.** Materials of a "Palette" object, Prompt "rusty metal". Material → Set
  Material.
- **Attribute routing.** Attributes source, Prompt "the attribute that measures wetness". Item →
  Named Attribute *Name*, so a tree adapts to whatever attributes the object's mesh data carries,
  e.g. painted or imported attributes. The list comes from the original data, never from this
  modifier's output (§6.0).
- **Data-driven pick.** Items `//props.csv` with columns `answer,when,instance,scale`. Payload
  outputs `instance` and `scale` → Instance on Points *Instance Index* and *Scale*.

### 6.3 Match: each named item, Yes/No or Amount

Properties `answer_type` ∈ {`YES_NO` (default), `AMOUNT`} and `list_source` (default `TEXT`).

| Socket | Kind | Type | Default | Meaning |
|---|---|---|---|---|
| Question | word | String | "" | Asked about every item: "Would you find it in a kitchen?" / Amount: "How heavy is it?" |
| Context | word | String | "" | The shared situation |
| Items / Collection / Object | word | per §6.0 | — | The items to judge |
| Answers | word | String | "" | Optional; same meaning as on Ask |
| ID | field | Int | Index | Which item each element is |
| Threshold | GN value | Float (0–1) | 0.5 | Cut-off for Selection. Applied by a Compare node in the inner group; may itself be a field |
| Overrides | word | String | "" | Overrides CSV keyed by item name (§7) |
| **Probability** | output | Float, field | 0.0 | Yes/No: P(yes) for item `ID`. Amount: its amount |
| **Selection** | output | Bool, field | False | `Probability ≥ Threshold` |
| **Confidence** | output | Float, field | 0.0 | 4.e |

**Behaviour.**
- There is one question per item: `noul` for Yes/No, `score` for Amount. All items share one
  state (Context), so a table of N items goes out in `ceil(N / 32)` chunks. N items cost 2N scored
  rows with debias: 40 items take about 1.7 s at 24 cells/s.
- The item's name and `when` go into the question's instructions; the exact template is in §10.
- Empty material slots become fallback rows. They are never asked and never flagged.

**Examples.**
- **Kitchen clutter.**
  1. Collection "Props" (40 objects), Question "Would you find it in a kitchen?".
  2. Collection Info (Separate Children, Reset Children) → Delete Geometry on the Instance domain
     with Selection = NOT `Match.Selection`. ID stays Index.
  3. Scatter the survivors with Instance on Points.
- **Metal faces.** Materials of the building object, Question "Is this material metal?", ID =
  *Material Index* (face domain). Selection → Store Named Attribute `metal` (Bool, Face) → the
  shader reads it.
- **Weight.** Amount mode, "How heavy is it?". Probability → Scale Instances, or a `mass`
  attribute for a simulation. The CSV route (`//props.csv` with a `when` column) adds
  descriptions that sharpen the answers (4.c).

### 6.4 Classify: each named item, one of several labels

Property `list_source` (default `TEXT`). There is no `answer_type`: Classify is always One-of.

| Socket | Kind | Type | Default | Meaning |
|---|---|---|---|---|
| Question | word | String | "" | Optional instruction. Empty → "Which label fits this item best?" |
| Context | word | String | "" | The shared situation |
| Items / Collection / Object | word | per §6.0 | — | The items to classify |
| Labels | word | String | "" | **Required** answer set, 1–255 labels: inline `name: when` or a CSV `answer,when[,payload…]`. A single label is decided by code for every item (§7.3, 4.d). Empty → hint "Add labels" |
| ID | field | Int | Index | Which item each element is |
| Overrides | word | String | "" | Overrides CSV keyed by item name (§7) |
| **Label** | output | Int, field | −1 | Row of the chosen label for item `ID` |
| **Confidence** | output | Float, field | 0.0 | 4.e (top minus runner-up) |
| *payload columns* | output | Int/Float, field | fallback | Baked numeric columns of the Labels CSV, looked up at Label |

**Behaviour.**
- There is one `choice` question per item, over all labels, sharing one state. More than 16
  labels become a tournament per item, with cost `2 × (groups + 1)` rows. The 8192-row limit
  (§5.4) therefore stops, for example, 255 labels × 250 items before any request is sent.
- Measured scale: 27 items × 7 labels fill in about 2 s (14 cells/s).

**Examples.**
- **Rooms.** Items = collection "Props". Labels `kitchen: cooking, food and dishes; bathroom:
  washing and hygiene; office: work and paper; garden: plants and outdoor tools`. Label → Index
  Switch of room anchor positions → Set Position of the instances.
- **Material families with payload.**
  1. Materials of an imported model, with ID = *Material Index*.
  2. Labels `//material_families.csv` with columns `answer,when,roughness,metallic`.
  3. The payload fields `roughness` and `metallic` → Store Named Attribute → shader Attribute
     nodes.

  The model decides the family; the CSV holds the numbers (4.a).
- **Town zoning.**
  - Items `//building_types.csv`: bakery, school, church, factory, house… each with a `when`.
  - ID = Named Attribute `building_type` (Int, Instance domain).
  - Labels `loud: traffic, machines or crowds; busy: people come and go all day; quiet: few
    visitors`.
  - Label → crowd density per building.

### 6.5 Rules: each measured situation

Property `answer_type` ∈ {`ONE_OF` (default), `YES_NO`, `AMOUNT`}. The sockets have a fixed head,
one input per factor declared in the Factors CSV, and a tail that exists only in One-of mode.

| Socket | Kind | Type | Default | Meaning |
|---|---|---|---|---|
| Question | word | String | "" | "What happens to this cell in the next generation?" |
| Context | word | String | "" | Shared situation: "A cellular automaton on a square grid; every cell has eight neighbours." |
| Factors | word | String (`.csv` path) | "" | **Required.** Factor declarations (below). Empty → hint "Set a Factors CSV (sidebar: New CSV from Template)" |
| Outcomes | word | String | "" | Answer set: inline or CSV `answer,when[,if][,payload…]`. **Required** for One-of. Optional for Yes/No (`yes`/`no` whens) and Amount (ordered levels; default the five `AMOUNT_LEVELS`) |
| Overrides | word | String | "" | Overrides CSV keyed by level names, one column per factor (§7) |
| *one per factor* | field | Float / Int / Bool | 0 / 0 / False | Named after the factor, in CSV order (table below) |
| Dither | GN value | Float (0–1) | 0.0 | One-of only: random band-edge blending (§9) |
| Seed | GN value | Int | 0 | One-of only: seed for Dither |
| **Outcome** | output | Int, field | −1 | One-of: row of the chosen outcome |
| **Probability**, **Yes** | output | Float / Bool, field | 0.0 / False | Yes/No: P(yes), interpolated over Amount factors; Yes = `Probability ≥ 0.5` |
| **Amount** | output | Float, field | 0.0 | Amount: `score / (L − 1)`, interpolated over Amount factors |
| **Confidence** | output | Float, field | 0.0 | 4.e, interpolated like the value (One-of: the chosen cell's) |
| *payload columns* | output | Int/Float, field | fallback | One-of: baked numeric Outcomes columns, looked up at Outcome |

**Factor kinds.** Declared in the Factors CSV with columns `factor,kind,level,when,value,from,to`.
- Columns are matched by header name, in any order.
- There is one row per level, and the level order is the row order.
- `value` is required for `category` and `flag`. `from`/`to` are optional and apply to `amount`
  only.

| `kind` | Socket | Levels | Element → level (a node graph, §9) |
|---|---|---|---|
| `amount` | Float field, 0–1 | 2–16 named bands; optional `from`/`to` edges, equal width by default | band containing the clamped value; values interpolate between band centres for Yes/No and Amount |
| `category` | Int field | 2–64 levels, each with an integer `value` | exact match on `value`; an unknown value → fallback row |
| `flag` | Bool field | exactly 2 levels, `value` 0 and 1 | False → the `value 0` level, True → the `value 1` level |

**Behaviour** (the maths is in §9):
- **Cells.** There is one cell per combination of levels: `cells = Π nⱼ ≤ 4096`. The flat index
  is row-major in factor order: `cell = Σ levelⱼ · strideⱼ`, where `strideⱼ = Π_{k>j} n_k`.
- **The question for one cell.** All cells share the state (Context plus fixed framing). The
  instructions hold the Question plus the cell's facts (`factor: level (when)`). The options are
  the outcomes whose `if` the cell satisfies.
  - One outcome allowed: decided by code.
  - None allowed: the cell is an error, e.g. "no outcome applies to state=dead, neighbours=3".
- **Yes/No and Amount.** Values interpolate multilinearly between the band centres of up to 3
  Amount factors (at most 8 corners). With more Amount factors the node uses the nearest band and
  warns. Category and flag factors are always exact.
- **One-of.** Uses the nearest band. With `Dither > 0`, each Amount coordinate is jittered by up
  to ±Dither/2 band widths using `Random Value` (seeded by *Seed* and the element's implicit ID),
  so outcomes mix near band edges instead of switching along a hard line.
- **Simulation and repeat zones.** Lookup works inside them because the inner group is a pure
  function of its field inputs (§8, tested in §18).

**Example 1: Game of Life from CSV** (the video's opening demo; exact tables are required).

`//life_factors.csv`:
```
factor,kind,level,value,when
state,flag,dead,0,The cell is empty.
state,flag,alive,1,The cell holds a living cell.
neighbours,category,0,0,No live neighbours.
neighbours,category,1,1,Exactly one live neighbour.
neighbours,category,2,2,Exactly two live neighbours.
neighbours,category,3,3,Exactly three live neighbours.
neighbours,category,4,4,Exactly four live neighbours.
neighbours,category,5,5,Exactly five live neighbours.
neighbours,category,6,6,Exactly six live neighbours.
neighbours,category,7,7,Exactly seven live neighbours.
neighbours,category,8,8,Eight live neighbours; every surrounding cell is alive.
```

`//life_outcomes.csv`. Here `answer`, `when` and `if` are text columns that the model sees. `alive`
is a numeric payload column: Python bakes it into the table, it becomes the node's `alive` output,
and the model never sees it:
```
answer,when,if,alive
Birth,A dead cell with exactly three live neighbours comes to life.,state=dead,1
Stays empty,A dead cell with any other number of live neighbours stays empty.,state=dead,0
Survives,A live cell with two or three live neighbours stays alive.,state=alive,1
Dies,A live cell with fewer than two or more than three live neighbours dies.,state=alive,0
```

**How the demo runs.**
- The table has 2 × 9 = 18 cells, each offering 2 outcomes (4.d). That is 36 scored rows, about
  0.8 s.
- Inside a Simulation Zone over grid points:
  1. `state` = Named Attribute `alive` > 0.
  2. `neighbours` = the sum of 8 `Sample Index` lookups at the grid offsets (counting stays in
     nodes, 4.a). The boundary is **toroidal**:
     - `row = floor(i / W)`, `col = i − row·W`;
     - each neighbour's index is `((row + dr) mod H)·W + ((col + dc) mod W)`, using Integer Math
       `FLOORED_MODULO`.

     The ±1 offsets therefore never wrap into the next row, and the ±W offsets never fall off the
     grid. The Python reference in `tests/lookup_blender.py` uses the same wrap.
  3. Rules → payload output `alive` → Store Named Attribute `alive`.
- **Switching rule sets.** Point Outcomes at `//highlife_outcomes.csv` (Birth: "…exactly three or
  six…") while playback is stopped. The table refills in about 1 s and the simulation continues
  with the new rules.

**Example 2: biome scatter.**
- **Factors.** `temperature` (cold/mild/hot), `moisture` (dry/moderate/wet) and `altitude`
  (low/mid/high), all `amount`, give 27 cells.
- **Outcomes CSV.** `answer,when,instance,scale`, with 7 plants described by habitat, e.g. "Pine:
  grows in cold or mild places, and is the tree of high altitudes".
- **Inputs.** Terrain attributes normalised with Map Range feed the factor sockets. Measured
  quality: mean confidence 0.61, all 7 outcomes used.
- **Outputs.** Outcome → payload `instance` → Instance on Points (*Pick Instance*) over the plant
  collection, and `scale` → Scale. With `Dither` 0.3, plant zones blend instead of forming
  borders.

**Example 3: snow cover.** Same factors, `answer_type = AMOUNT`, Question "How much snow covers the
ground?". Amount is smooth and monotonic (measured 0.08 for low and hot up to 0.89 for high and
cold), with no stair steps between bands. It drives the snow mix factor directly.

### 6.6 Dials: one prompt, many named settings

Dials has no dropdown: each setting declares its own answer type in the Settings list. Its outputs
are dynamic, one per setting, in list order.

| Socket | Kind | Type | Default | Meaning |
|---|---|---|---|---|
| Prompt | word | String | "" | The situation every setting is judged against: "A stormy night in a fishing village" |
| Settings | word | String | "" | **Required.** Inline or CSV (below). Empty → hint "Add settings" |
| Overrides | word | String | "" | Overrides CSV keyed by setting name (§7) |
| *one per setting* | output | Bool (`yes/no`), Float (`amount`), Int (`one-of`), single | False / 0.0 / −1 | Named after the setting |
| **Confidence** | output | Float, single | 0.0 | The **lowest** confidence among the settings. Each setting's own confidence is shown in the sidebar and in the exported Overrides CSV |

**Settings CSV** (`//weather_dials.csv`). The columns are `setting,type,question,answers,min,max`:
- `setting` is unique and becomes the output name.
- `type` is `yes/no`, `amount` or `one-of`.
- `answers` uses the inline answer-set format (§6.0).
  - Required for `one-of`: 2–255 options.
  - Optional for `yes/no` (`yes`/`no` whens) and for `amount` (ordered levels).
- `min`/`max` apply to `amount` only and default to 0 and 1.

```
setting,type,question,answers,min,max
Rain,yes/no,Is it raining?,,,
Wind speed,amount,How windy is it?,calm: still air; breeze: leaves move; strong: branches sway; gale: trees bend; storm: things break,0,30
Time of day,one-of,What time of day is it?,dawn: first light; day: the sun is high; dusk: evening glow; night: dark sky,,
Street lights,yes/no,Are the street lights on?,yes: it is dark or foggy; no: there is enough daylight,,
Fog,amount,How foggy is it?,,,
```

**Inline Settings** support only `yes/no` and `amount`. There is one entry per setting, in the
form `name (type): question`, for example `Rain (yes/no): Is it raining?; Wind (amount): How
windy is it?`. A `one-of` setting needs the CSV, because its options would collide with the `;`
separator. Inline Settings share the CSV's limit of 32 settings (§7.5), because their graph costs
the same.

**Behaviour.**
- Every setting is one question. All of them share the Prompt as state and travel in **one**
  request at priority P0. Five settings with debias are about 10–12 scored rows, around 0.3–0.6 s.
  More than 16 options on a `one-of` setting run as a tournament.
- `min`/`max` are applied by a **Map Range** node inside the inner group, so the mapping is a node
  (4.a).
  - They are **per-entry data**: each setting's *To Min* and *To Max* come from the node's
    one-row-per-entry table object (columns `min_<s>`, `max_<s>`), read at `k − 1` (§8.4, §8.6,
    §14.2). Contexts may therefore use Settings CSVs with different ranges, as long as the setting
    names and types agree (§6.0).
  - Editing `min`/`max` triggers a values-only update (`tables.write_rows()`) and never a model
    call.
  - `Sample Index` with a single-value index yields a single value, verified by the PR-08 spike
    (§8.4). Index Switches on `k` with at most 16 settings remain a contingency only.
- Outputs are single values, read from the node's one-row-per-entry table object at `k − 1`
  (§8.4), with `k` from the match chain over (Prompt, Settings, Overrides, Key) (§8.2). One tree on
  many objects gives each object its own dial values.
- The output sockets are synchronised by name and type (§6.0). Renaming a setting or changing its
  type replaces its socket, so links to it are dropped. The node warns once: "Setting 'Wind'
  renamed or retyped: reconnect its output".

**Example: one prompt drives a whole environment.** The modifier panel exposes Prompt (Group
Input → Prompt). With the CSV above:
- `Rain` → Switch that enables a rain point system.
- `Wind speed` (0–30) → the strength of a noise-based sway on trees.
- `Time of day` → Index Switch over four sun rotations.
- `Street lights` → Switch that instances emissive bulbs.
- `Fog` → the density of a volume cube.

Typing "a calm summer morning" instead re-answers all five dials in one request (under a second),
and switching back is instant (cached).

## 7. CSV Formats

### 7.1 The text/numeric split, and why the inner group never uses Import CSV

Two verified facts drive this section:
- Import CSV (4.5+) keeps only numeric columns and silently drops text columns.
- Import CSV **caches by path and never re-reads an edited file** (4.5.2 and 5.1.1, §3.4).

**Decision.** `csvfmt.read_csv()` parses **every column** of every CSV a node references, and
sorts each column into one of two classes:

| Column class | Examples | Goes to | Editing it |
|---|---|---|---|
| **text** (reserved names, or any non-numeric column) | `answer`, `when`, `if`, `factor`, `level`, `setting`, `question` | the model; part of cell keys (§5.4) | re-asks only the affected cells |
| **numeric payload** (every cell an integer → INT, every cell a number → FLOAT) | `alive`, `scale`, `instance`, `roughness` | **baked** into the node's table as extra columns; one output socket each (§6.0) | values-only update, 0 requests |

Two ways to deliver payload to Geometry Nodes were compared. **(A) bake the parsed values into
the table** was chosen over **(B) point an inner Import CSV at a content-addressed copy**
(`<cache>/<sha1>.csv`, so that every edit gets a new path):

| Criterion | (A) Bake (chosen) | (B) Content-addressed copy |
|---|---|---|
| Send the .blend to a render farm or a colleague | Self-contained: the values are in the file | The inner Import CSV points at this machine's cache. Payload is **silently empty** elsewhere unless the cache ships too |
| Unsaved .blend | Works | Needs an absolute cache folder |
| Extra files on disk | None | One copy per edit, which needs garbage collection |
| Lookup machinery | The same table-object lookup as the answers (§8) | A second path (Import CSV → Named Attribute → Sample Index), plus a row mapping that must match Blender's parser exactly |
| Without the add-on (4.5+) | Evaluates: it is plain data | Evaluates only if the cache file exists |
| Blender memory | Unchanged | Each edit leaves another cached copy in memory for the session |
| Reads "the CSV in Geometry Nodes" | Through the node's path socket; values arrive baked | Literally through Import CSV |

(B) is the literal reading of "Geometry Nodes reads the numeric columns", but it breaks the
file's portability and adds a second lookup path, and it is the only option that can fail
silently. (A) keeps the user's authoring model intact:
- rules and payload live in CSV files;
- the files are given to the node inside Geometry Nodes, either as a path or by **plugging an
  Import CSV node into the OpenAct node** (user decision, §6.0);
- the model never sees numbers (4.a).

When an Import CSV node is plugged in, OpenAct only reads that node's Path and Delimiter and
parses the file itself. The Import CSV geometry and Blender's path cache are never involved.
OpenAct's outputs are current as soon as the file watcher sees an edit (§7.8).

**Artists' own Import CSV nodes on the same file.**
- **Rows line up.** OpenAct row *r* (the value of `Outcome`, `Label` or `Index`) is Import CSV
  point *r*. Numeric columns keep their header names and INT/FLOAT types. The parity probe in §18
  asserts this on fixture files, so `Import CSV → Sample Index (Index = Outcome)` returns the same
  values as OpenAct's payload output **on first read**.
- **But Blender caches.** Blender keeps its first read of each path for the whole session, so
  after an edit the artist's Import CSV node still returns the old rows while OpenAct's outputs
  update.
  - This is documented in the README.
  - It is surfaced in the sidebar: when the static evaluator finds a `GeometryNodeImportCSV`
    whose Path resolves to a file an OpenAct node uses, the node's panel shows "Import CSV '<node>'
    reads the same file; Blender won't re-read it after edits in this session. Use this node's
    payload outputs, or restart Blender."
  - The recommended pattern is to use the payload outputs.
- **Not relied upon.** An equivalent spelling of the path (e.g. `//./life.csv`) may bypass the
  cache if Blender keys it by the raw string. This is unverified.
  - The CSV-probe suite tests it (follow-up §21 F1).
  - The helper `openact.refresh_import_csv` ships **only if** the trick works on both 4.5 and 5.1.
  - This concerns only the artist's *own* Import CSV nodes. An Import CSV node plugged into an
    OpenAct node is never evaluated for OpenAct.

### 7.2 Dialect and parsing rules (`csvfmt.py`, pure Python)

`csvfmt` imports only the standard library (`csv`, `io`, `re`, `hashlib`, `os`). Its unit tests
therefore also run under any plain CPython ≥ 3.11 (§18).

1. **Bytes.** Files larger than 1 MB are an error. The text is decoded as strict UTF-8, and one
   leading BOM is stripped. A NUL byte or invalid UTF-8 is an error. This differs from Import
   Text's silent "": an empty rule set must never look like a valid one.
2. **Rows.** `csv.reader(io.StringIO(text, newline=""))` accepts CRLF, LF and CR.
   - **Delimiter.** `,` by default. If the header line has no `,` but has `;` or a tab, that
     delimiter is used: Excel in decimal-comma locales exports `;`. Artists who also use their own
     Import CSV must set its Delimiter to match.
   - **Quoting.** Cells are quoted with `"`, and `""` escapes a quote. Quoted cells may contain
     the delimiter; Import CSV handles this, as verified with quoted text containing commas. A
     line break inside a quoted cell is a warning, because Import CSV parity is unverified.
3. **Header.** The first non-blank row is the header.
   - Names are trimmed. Reserved names are matched case-insensitively. Payload names keep their
     exact case, because they become socket names.
   - An empty header name, or two names equal ignoring case, is an error.
4. **Data rows.**
   - Blank rows are skipped. A blank row *between* data rows is a warning, because messages use
     physical line numbers and Import CSV parity is pinned by the probe.
   - A row longer than the header is an error. A shorter row is padded with empty cells.
   - Cells are trimmed.
5. **Limits.** At most 4096 data rows and 64 columns. Text cells may hold up to 2000 characters,
   which keeps prompts inside SemIf's context; the server answers 422 on over-long prompts.
6. **Numeric classification** applies to non-reserved columns and is intended to mirror Import
   CSV's typing. The rules below are **provisional** until the §18 parity probe
   (`tests/csv_probe_blender.py`) has run on 4.5, 5.0 and 5.1. The probe's recorded behaviour
   then becomes the specification:
   - **INT**: every cell matches `[+-]?[0-9]+` and fits in an int32.
   - **FLOAT**: every cell matches `[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?`, and the
     column is not INT.
   - **Otherwise text.** An empty cell, `inf`, `nan`, `true` or a decimal comma (`1,5`) makes the
     column text. If all other cells were numeric, a warning names the rows, e.g. "column 'scale'
     has an empty cell in row 7; it is treated as text and gives no output".

```python
@dataclass(frozen=True)
class CsvDoc:
    path: str                                # absolute, os.path.normpath
    stamp: tuple[int, int]                   # (st_mtime_ns, st_size) when read
    header: tuple[str, ...]                  # trimmed, as written
    rows: tuple[tuple[str, ...], ...]        # data rows, trimmed, padded to header width
    lines: tuple[int, ...]                   # physical line number of each data row (for messages)
    text_cols: tuple[str, ...]               # reserved + non-numeric columns
    num_cols: dict[str, str]                 # payload column -> "INT" | "FLOAT" (header order)
    numbers: dict[str, tuple[float | int, ...]]
    text_hash: str                           # sha1: header order, row count, normalised text columns
    num_hash: str                            # sha1: num_cols + numbers
    problems: tuple["Problem", ...]          # (severity, line, column, message), §7.9
    delimiter: str = ","                     # the one the rows were split with: given, or sniffed
    bom: bool = False                        # starts with a UTF-8 BOM; the Overrides export keeps it (§7.6)
```

### 7.3 Answer-set CSV: Answers, Labels, Outcomes and Items

One schema serves Ask/Match Answers, Classify Labels, Rules Outcomes, and Items lists (Text or
CSV source):

| Column | Class | Required | Meaning |
|---|---|---|---|
| `answer` (synonyms `name`, `item`, `label`, `outcome`) | text | yes | Option or item name, shown to the model and in the UI. At most 200 characters; unique ignoring case |
| `when` (synonym `description`) | text | recommended | The condition under which this answer applies, or a description of the item (4.c) |
| `if` | text | no | Exact-fact restriction, Rules Outcomes only (§9.4). On other nodes it is ignored with a warning |
| any other **numeric** column | payload | no | One output socket named after the header, INT or FLOAT (§6.0) |
| any other **text** column | ignored | no | Artist's notes. Reported once as info: "column 'notes' is text and is not used" |

- **Row order is the answer index.** Row *r* is `Outcome`/`Label`/`Index` = *r*, and Import CSV
  point *r*.
- **Reserved in this schema** (case-insensitive, never payload): the names above, plus the
  Overrides columns `lock`, `unsure`, `confidence`, `model_answer` and `source`.
- **Payload limits.** At most 16 payload columns. A payload header that equals a fixed output of
  the node, ignoring case (`Index`, `Item`, `Confidence`, `Selection`, `Probability`, `Instance`,
  `Material`, `Label`, `Outcome`, `Yes`, `Amount`), is an error: "rename column 'Outcome'".
- **Rows by answer type:**
  - Yes/No sets have exactly the rows `yes` and `no`, in any order.
  - Amount sets have 2–16 rows, lowest level first.
  - One-of sets have 1–255 rows. A single row is decided by code (4.d).

`//material_families.csv` (Classify Labels). Here `answer` and `when` are text, `roughness` is a
FLOAT payload and `metallic` an INT payload:
```
answer,when,roughness,metallic
metal,"steel, iron, copper, brass or aluminium surfaces",0.35,1
wood,"planks, bark and furniture timber",0.6,0
stone,"rock, brick, concrete or marble",0.8,0
fabric,"cloth, carpet and upholstery",0.9,0
```

### 7.4 Factors CSV (Rules)

Only Python reads this file; Geometry Nodes never does. Columns may therefore mix numbers and
blanks freely.

| Column | Required | Meaning |
|---|---|---|
| `factor` | yes | Factor name, which becomes the input socket name. At most 63 characters and unique. It must not equal a fixed socket (`Question`, `Context`, `Factors`, `Outcomes`, `Overrides`, `Dither`, `Seed`), nor an Overrides CSV column (`answer`, `lock`, `model_answer`, `confidence`, `unsure`, `source`): factor names are the Overrides CSV's key columns (§7.6) |
| `kind` | on the factor's first row; later rows empty or equal | `amount`, `category` or `flag` |
| `level` | yes | Level name shown to the model; unique within the factor, ignoring case |
| `when` (synonym `description`) | recommended | What the level means: "freezing to chilly: snow is common" |
| `value` | `category` and `flag` | The integer the socket carries for this level; unique within the factor. A `flag` has exactly the values 0 and 1 |
| `from`, `to` | `amount`, optional | Band edges within 0–1 |

**Validation rules** (each violation is an error that names the line):
- A factor's rows are contiguous. Factor order is the order of their first rows, and level order
  is row order.
- `amount` has 2–16 levels. Its band edges are all-or-nothing **per factor**:
  - either no `from`/`to` at all, giving equal-width bands (band *b* covers `[b/n, (b+1)/n)`, the
    last band includes 1);
  - or both on every row, with `from₀ = 0`, `to_{n−1} = 1`, `to_b = from_{b+1}` and strictly
    increasing edges.
- `category` has 2–64 levels and `flag` exactly 2.
- A `category` factor's values are either contiguous (`u_i = u_0 + i`, any integers) or all lie
  within 0–4095. Other value sets are an error, because non-contiguous values map through the
  table object (§9.2).
- There are 1–8 factors, and the product of their level counts is at most 4096 (§5.4).

`//biome_factors.csv`. `temperature` has custom edges; the other two factors use equal-width bands:
```
factor,kind,level,when,from,to
temperature,amount,cold,"freezing to chilly: snow is common",0,0.3
temperature,amount,mild,"cool to warm: temperate seasons",0.3,0.7
temperature,amount,hot,"hot all year: deserts and tropics",0.7,1
moisture,amount,dry,"little rain: sand, rock and dry grass",,
moisture,amount,moderate,"regular rain",,
moisture,amount,wet,"very rainy: swamps and rainforest",,
altitude,amount,low,"lowlands and coasts",,
altitude,amount,mid,"hills",,
altitude,amount,high,"mountains and high slopes",,
```

### 7.5 Settings CSV (Dials)

| Column | Required | Meaning |
|---|---|---|
| `setting` | yes | Output socket name; unique ignoring case, at most 63 characters. It must not be `Confidence` |
| `type` | yes | `yes/no` (synonyms `yes_no`, `bool`), `amount`, or `one-of` (synonyms `one_of`, `choice`) |
| `question` | yes | The instruction for this setting |
| `answers` | yes for `one-of`, optional otherwise | Inline answer set (§6.0): `;`-separated `name: when` entries. One-of: 2–255 options (a tournament above 16). Amount: 2–16 ordered levels. Yes/No: `yes`/`no` whens |
| `min`, `max` | optional, `amount` only | Numbers, default 0 and 1. `min > max` inverts the mapping. Python writes them into the Map Range (§6.6) |

A Dials node has at most 32 settings, from a Settings CSV or inline (§6.6): the limit at which its
graph still builds within its budget (§8.4). A Settings CSV is not a payload source: Dials outputs
are the answers themselves (§6.6).

### 7.6 Overrides CSV (all six nodes)

This file is the only override channel (4.f). The artist writes it, or starts from the sidebar's
**Write Overrides CSV** (`openact.write_overrides`).

| Node | Key columns | `answer` accepts |
|---|---|---|
| Ask | `question` (the Question text, one row per question used by any context) | Yes/No: `yes`, `no` or a probability 0–1. Amount: a number 0–1 or a level name |
| Pick | `prompt` (the Prompt text, one row per prompt) | An item name from the list. A locked row gives that item probability 1 |
| Match | `item` | Yes/No: `yes`, `no` or a probability 0–1. Amount: a number 0–1 or a level name |
| Classify | `item` | A label name from Labels |
| Rules | one column per factor, named after the factor, holding level names | One-of: an outcome name. Yes/No: `yes`, `no` or 0–1. Amount: 0–1 or a level name |
| Dials | `setting` | By type: `yes`/`no`; amount **in output units** (`min`…`max`, mapped back to 0–1 by code); one-of: an option name |

**Columns.**
- **`lock`** is the control column. `1` (also `yes`, `true` or `x`) locks the row; anything else
  leaves it unlocked.
- **Informational columns**, written by export and ignored on read: `model_answer`, `confidence`,
  `unsure` and `source` (`model` / `code` / `lock` / `stale` / `pending`).
- **Matching.** Key cells are trimmed and compared ignoring case. A Rules file must have every
  factor column; if one is missing, the whole file is an error. Unknown columns are kept by export.

**Scope.** One file applies to every context of the node. For per-object overrides, drive the
`Overrides` socket from Group Input so that each object points at its own file; the path is a word.

**A valid locked row:**
- has source `lock` and confidence 1.0;
- is never asked;
- is used everywhere, including as a corner of Rules interpolation;
- is part of the words key (through `lock_digest`, §5.4), so editing a lock updates the table
  immediately with 0 requests.

**Invalid rows:**
- A key that matches no row is an **orphaned lock**: a warning, and the row is ignored.
- An answer that names nothing valid is an error on that row. The lock is ignored and the model's
  answer stays, flagged in the sidebar.

**Export** (`openact.write_overrides`). The context selected in the sidebar (`context_index`, the
first by default) picks the file. The contexts it chooses from are the node's contexts that plan
(`ops.export_uses()`), including one whose Overrides CSV is missing, has errors or is blocked. Such
a context comes after the others, which the sidebar lists in the same order, so a selection there
exports the context it shows. Every
context of the node that uses that file, with the same key columns, adds its rows, so exporting any
of them writes the same file. A per-object file (**Scope**) gets only its object's rows.
1. **Target.** The selected context's Overrides value. If it is empty and unlinked, a file dialog
   proposes `//<tree>_<node>_overrides.csv` and writes the chosen path into the socket's default
   value. The dialog marks an existing file.
   - The path resolves as a read of the socket does (§7.7). In an untrusted file the trust rule
     (§16.1) confines writes as it does reads: "Blocked writing <path>: outside this file's folder.
     Allow this file (or trust its folder) to write it".
   - A missing file is created. An empty one is written as a new one.
   - An existing file is refused, and left as it was, when:
     - an OpenAct node in the open .blend reads it as another input: "kitchen_items.csv is the
       Items CSV of 'Match': choose another file";
     - it has errors, including a missing factor column or a key locked twice: its own message,
       then "Fix it, or write another file";
     - it isn't an Overrides CSV for this node. It has none of the node's key columns ("contacts.csv
       is not an Overrides CSV for this node (no 'item' column): choose another file"), or it has no
       `lock` column, which an Overrides CSV needs to lock anything. That also refuses the input
       files (§7.3–7.5) that no open node reads, whose columns can include a key column (`item`,
       `question`, `setting`): "inventory.csv is not an Overrides CSV (no 'lock' column): choose
       another file".
2. **Merge.** No row of the file is lost: other contexts' rows, rows no table uses any more
   (orphaned locks, renamed items or levels) and rows whose key is there twice all stay. For each
   key in the tables, the export uses one of the file's rows with that key: the locked one (the row
   the lock reads), else the first. A locked row is kept verbatim, including user columns. An
   unlocked row gets the new answer and keeps its user columns.
3. **Write** one row per distinct key in those contexts' tables, in context order, then table
   order, each with its own context's answer. The file's other rows follow, in file order.
   - A key used twice, such as a material in two slots or a question that two contexts ask, is
     written once; its lock applies to every use.
   - Columns: the key columns, `answer`, `lock`, `model_answer`, `confidence`, `unsure`, `source`,
     then the file's other columns.
   - `answer` is the artist's value on a locked row. Otherwise it is the table's answer in the form
     its lock reads: Yes/No as the probability (`0.9`, not `yes`); an Amount as a number, in full
     (`1.0`) when its short form is a level name (levels may be named `1` to `5`); Pick and One-of
     as the chosen name. Dials writes a yes/no setting as `yes`/`no` (its output is a Bool and its
     lock takes only those), and an amount in output units.
4. **Atomically.** Write `<path>.tmp`, then `os.replace()`. A failed write removes the `.tmp` and
   leaves the file as it was. The `.tmp` is created new, so a planted link is never followed; one
   left by an interrupted export is removed first.
   - An existing file keeps its delimiter and its UTF-8 BOM, if it has one. A new file takes the
     Import CSV node's Delimiter, else `,`.
   - A read-only file is refused: "Can't write <path>: the file is read-only".
   - On `PermissionError` the message is "Close the file in the other program (e.g. Excel) and try
     again."
5. **Never automatically.** OpenAct writes user files only from this operator and from *New CSV
   from Template*.

`//life_overrides.csv`, after exporting and then locking one cell by hand (HighLife's birth on six):
```
state,neighbours,answer,lock,model_answer,confidence,unsure,source
dead,3,Birth,0,Birth,0.98,0,model
dead,6,Birth,1,Stays empty,0.95,0,lock
alive,2,Survives,0,Survives,1.00,0,model
```
(15 more rows.)

### 7.7 Path resolution and inline detection

- **Two routes to a CSV** (user decision, §6.0). `lists.csv_source(node, input, ctx)` picks the
  route in this order:
  1. **Linked Import CSV:** if the companion `<Input> CSV` socket is linked,
     `evaluator.trace_csv_source()` returns the Import CSV node's Path and Delimiter. It **wins**
     over a typed path, which is then ignored with a warning.
  2. **Typed path:** otherwise the String input is used.
- **Inline or path.** `lists.is_csv_path(text)` is true when the stripped value is a single line
  ending in `.csv`, in any case. Anything else is inline text (§6.0).
- **Resolution.**
  - A typed path resolves as an Import node's Path (next item), with `//` implied: `//`, like any
    relative path, is relative to the .blend, or to the library file for linked node groups.
  - An Import CSV node's Path, like an Import Text Path, resolves as Geometry Nodes resolves it
    (`evaluator.import_path()`), so OpenAct reads the file that node reads. `//` is relative to the
    file that holds the Import node's tree: the .blend, or the library of a linked group. Any other
    relative path is relative to Blender's working folder.
  - A relative path in an unsaved file gives the hint "Save the .blend: '//' paths are relative to
    it" (for an Import node, a `//` path). Absolute paths always work.
- **Identity.** The file watcher and the parse cache key files by
  `os.path.normcase(os.path.realpath(p))`. Each file is parsed once per change, and the result is
  shared by every node that uses it.
- **Keys.** The raw path string is part of `gn`, because the match chain compares word strings.
  It is **never** part of a cell key. So:
  - moving a CSV and updating the path does not re-ask;
  - two files with identical text share cached answers.
- **Where paths come from.** Paths are words, so every static-evaluator source works:
  - typed on the node;
  - the modifier panel through Group Input;
  - Join Strings (folder + file name);
  - a **Menu Switch** of type String. For example, a modifier dropdown "Conway / HighLife / Seeds
    / Day & Night" selects the Outcomes file. That is the rule-set switcher in the Game of Life
    demo (§18);
  - an **Import CSV node's Path input**, when that node is plugged into the companion socket. Its
    Path can itself be typed, exposed on the modifier, or chosen by a Menu Switch. A Menu Switch of
    type *Geometry* choosing among several Import CSV nodes also works.
- **Linked data.** Nodes in linked libraries are read-only. Their tables come from the library
  file, and export is disabled for them.
- **Trust.** In a file that is not trusted (§16.1), every file read is confined to the .blend's own
  folder tree. This covers the Import Text word source as well as CSVs. `realpath` is compared,
  so a symlink or `..` cannot escape. A read outside that tree is blocked, and the node shows
  "Blocked reading <path>: outside this file's folder. Allow this file (or trust its folder) to
  read it". The
  blocked text is never read, hashed or stored.

### 7.8 Change detection

| Event | Detected by | Action |
|---|---|---|
| Text columns edited | watcher: `os.stat` `(st_mtime_ns, st_size)` changed → re-read → `text_hash` differs | Pass. Only cells whose keys changed are re-asked; other rows stay stale-while-refilling (§5.4) |
| Only numeric columns edited | `num_hash` differs, `text_hash` equal | Values-only update (re-bake payload); 0 requests |
| Saved without real changes | both hashes equal | Nothing |
| Overrides edited | same watcher | Words keys change (through `lock_digest`) → values-only update; 0 requests for locked rows |
| File missing or unreadable | `stat`/`open` fails | Node error; that context keeps its **last good table** (its entry untouched); the node's other contexts are still answered while they fit the node's shape (§6.0) |
| Path word changed | socket update / depsgraph → pass | Re-plan. The same content elsewhere hits the cache |
| Edited while Blender was closed | the first pass after `load_post` re-reads every referenced file (stamps are not trusted across sessions) | As above |

The watcher (`engine.WATCH: {path_id: (stamp, CsvDoc)}`) behaves as follows:
- It polls from `tick()` at most every 0.5 s, and only files referenced by the last pass.
- Each `os.stat` costs about 20 µs. Above 32 files it checks 32 per tick, round-robin.
- A parse error keeps the previous `CsvDoc`, so a half-saved file never blanks a table.
- A file it can stat but not read (held open by another program) is reported when its stamp
  changes, not on every poll. Once it is let go, the 2 s safety pass reads it again.
- **Background and render mode.** When `bpy.app.background` is set, or a render job runs, OpenAct
  neither re-plans nor re-bakes from files. A farm renders exactly the saved answers and payload.
  If a referenced CSV's `text_hash` or `num_hash` differs from the stored ones, it prints once per
  file: "CSV <path> differs from the saved table (text/numbers); rendering the saved answers" (R23).
  - **Headless authoring opts back in.** This covers the test suites, `demos/build_demos.py`, and
    any script that sets prompts headlessly and saves. There are two ways:
    - the environment variable `OPENACT_BACKGROUND_PASSES=1`, set before Blender starts;
    - **in process**, `openact.engine.allow_background_passes(True)`, a public Python function
      that a script calls after enabling the add-on. `build_demos.py` calls it.
  - The README documents both. Without either, a headless session renders saved answers only.

### 7.9 Validation and error reporting

`Problem(severity, file, line, column, message)` has three severities:
- **error**: that context keeps its last good table, and nothing that depends on the file is sent;
  the node's other contexts are still answered while they fit its shape (§6.0). The node's status
  shows the first error with file and line.
- **warning**: the file is used. `⚠` plus the first warning appears in the status line once the
  table is complete and there are no errors.
- **info**: shown in the sidebar only.

The sidebar lists every problem (at most 50 per file), each with *Open File*.

| Problem | Severity | Message (example) |
|---|---|---|
| Missing file | error | "Can't find //life_outcomes.csv" |
| Over a limit (1 MB, 4096 rows, 64 columns, 2000 characters per cell) | error | "life.csv has 5 000 rows (limit 4 096)" |
| Not UTF-8, or a NUL byte | error | "life.csv is not UTF-8 text (byte 1 203)" |
| Duplicate or empty header | error | "Two columns are named 'when'" |
| No name column | error | "No 'answer' column (or name, item, label, outcome)" |
| Empty or duplicate answer | error | "Line 5: 'Birth' appears twice" |
| Row longer than the header | error | "Line 7 has 5 cells; the header has 4" |
| Yes/No set without both `yes` and `no` | error | "A Yes/No answer set needs rows 'yes' and 'no'" |
| `if` syntax, or unknown factor/level | error | "Line 4: 'if' names unknown level 'alvie' of factor 'state'" |
| A cell with no allowed outcome | error | "No outcome applies to state=dead, neighbours=3" |
| Invalid band edges | error | "temperature: 'to' of cold (0.3) must equal 'from' of mild (0.35)" |
| Payload name clash, or more than 16 payload columns | error | "Rename column 'Outcome': it clashes with a node output" |
| Outcome never offered | warning | "'Dies' is never offered: its 'if' matches no cell" |
| Number-like column with blanks or decimal commas | warning | "Column 'scale' has an empty cell in line 7; treated as text, no output" |
| `if` outside Rules, or blank line between rows, or quoted line break | warning | "'if' is only used by Rules; ignored" |
| Text column that nothing uses | info | "Column 'notes' is text and is not used" |

### 7.10 What the sidebar shows (inspection only)

- **Files.** For each file a node uses: path, data rows, text columns, payload columns with
  INT/FLOAT types, last read time, problems, and the Import CSV cache note (§7.1) when it applies.
- **Table.** One view per context: rows, answer, confidence, source, with unsure rows flagged
  (§17).
- **Buttons.** *Open File*, *Write Overrides CSV* (§7.6) and *New CSV from Template*.
- **Templates.** `openact/templates/` contains:
  - `life_factors.csv` and `life_outcomes.csv` (Game of Life);
  - `labels.csv`, `items.csv`, `weather_dials.csv`.

  The template button copies the right file(s) next to the .blend, never overwriting (`name.csv`,
  `name 2.csv`, …). It then writes the path into the node's socket default value if that socket
  is unlinked and empty.
- **Never an editor.** The sidebar does not change CSV content or tables (NG8). Every change goes
  through a file or a socket.

## 8. Table Lookup in Geometry Nodes

### 8.1 Table model

**Stored and materialised entries.**
- **Stored** entries live in the node's `entries` JSON. They make switch-back and undo free of
  model calls.
  - At most 16 **warm and cold** entries are kept, evicted least recently used.
  - **Live and pinned entries are exempt from that limit, and are never evicted, not even by the
    2 MB soft cap.** The soft cap evicts only warm and cold entries. Live plus pinned entries are
    kept even above 2 MB, with a size note, bounded by the 12-pin limit and the 4 MB per-entry
    hard cap (§14.7).
- **Materialised** entries are the ones present in the Geometry Nodes graph (match chain and
  table object). There is **one hard cap: `MATERIALISED_MAX = 16` per node, pinned included**. They
  are, in order:
  1. every entry whose `gn` equals the words of a **live** context;
  2. then **pinned** entries: those answered by *Prepare for Playback* for animated words (§5.5).
     There are at most 12, stored with `pinned: true` (§14.2), in pin order: by the first frame of
     each word tuple;
  3. then **warm** entries (the most recently live, not live now), only while the total is below
     4, and only if they fit the row budget (at most 16 × 4096 rows).
- **Above the cap.** More than 16 distinct live word sets on one node is an error: "20 objects use
  this tree with different words; the limit is 16: split the tree". The first 16 contexts, in
  context order, stay materialised. The others show neutral values and name the limit in their
  status.

  Prepare for Playback refuses when live plus pinned would exceed 16 (§5.5).
  - **More objects after pinning.** Live plus pinned can still exceed 16 later, when more objects
    start using the tree. Then **live entries win**:
    - pins are materialised in **pin order**: by the first frame at which each word tuple occurs,
      ties broken by tuple order (§5.5). The pins that start latest in the frame range are the
      ones left out;
    - those pins stay stored with `pinned: true`, never evicted;
    - the node's status says "3 pinned entries are inactive: this tree is now used by more objects
      (live + pinned > 16). Re-run Prepare for Playback, or split the tree".

    If the live entries alone exceed 16, the over-cap error above applies.
  - **Why a cap.** Every materialised entry costs graph nodes and items (§8.2, §8.4). The PR-08
    spike confirmed that the §8.4 budgets hold at 16, so the cap did not drop to 8.

  Warm entries let a recently used prompt, or a keyframed Menu Switch choice, switch back inside
  Geometry Nodes without waiting for a pass. Switching to a stored entry that is not materialised
  costs one pass and one rebuild (≤ 0.35 s), and never a model call.
- **Consequence.** Table size and graph size depend on the live contexts, not on edit history.
  Typing ten successive Questions on an 18-row Rules node keeps at most 4 materialised entries, and
  each edit causes at most one rebuild (tested in §18).

The per-element data of the materialised entries is a **flat table** stored in the node's table
object (§8.5). Entries are concatenated in materialised order (*e* = 1…E, E ≤ 16). Entry *e* starts at `base_e = Σ_{e'<e} N_{e'}`, and there are `F = Σ N_e` flat
rows. Row *r* of entry *e* sits at `base_e + r`. Columns depend on the node:

| Node | Per-row columns (read per element) | Entry-level values (one per context, single values) |
|---|---|---|
| Ask | — | Answer (F), Yes (B), Confidence (F) |
| Pick | `prob` (F) for Probability | Index (I), Item (S), Confidence (F), Material (MAT), payload (I/F) |
| Match | `p` (F), `c` (F) | — |
| Classify | `label` (I), `c` (F), payload (I/F) | — |
| Rules, One-of | `outcome` (I), `c` (F), payload (I/F) | — |
| Rules, Yes/No or Amount | `p` or `amount` (F), `c` (F), `ok` (F: 1 answered, 0 fallback) | — |
| Dials | — | one per setting (B/F/I), Confidence (F) |

**Payload is denormalised.** Every row carries the payload values of *its* chosen answer, so a
single per-element lookup returns the answer and its payload together. A numeric CSV edit rewrites
those columns: a values-only update (§7.8).

### 8.2 Selecting the context: the match chain

This is the prototype's `_Builder.match_chain`, unchanged in principle:
- **Joined key.**
  - One `Join Strings` node joins **every position of the node's fixed word tuple** with the
    delimiter `"\x1f"` (the ASCII unit separator).
    - A present socket feeds its Group Input.
    - A position whose socket is absent under the current `list_source` is fed by a **constant
      empty String node**, so the live key always has the same arity as the stored
      `"\x1f".join(gn)`.
    - Join Strings keeps empty items, orders most recently linked first, and preserves `"\x1f"`
      across save and reload (verified by the review on 4.5.2, 5.0.0 and 5.1.1). Python therefore
      links the items in reverse tuple order, following the `multi_input_sort_id` rule (§3.4).
  - Each **materialised** entry (§8.1) then needs one `Compare` (STRING, EQUAL) of that joined
    string with the entry's stored `"\x1f".join(gn)` (`B_STR`), and one `Switch` (INT).
  - The chain yields `k ∈ {0…E}`, with about **2 nodes per entry** instead of 10 (one Compare per
    word, the ANDs and one Switch). When no materialised entry matches the live words, the chain
    yields `k_default` (see "No match" below). 0 remains the "never answered" value.
  - A word containing `"\x1f"` makes that node fall back to one Compare per word, with an AND
    chain. The node notes it in its status, and the lookup tests cover it.
- **Compared words.** `spec.words(node)` returns a **fixed-length tuple per node type**,
  independent of `list_source`. It lists every String word the node can have. A position whose
  socket is absent under the current `list_source` (Items under Collection, Materials or
  Attributes) holds `""` in `gn`, and is joined as a constant `""` on the live side (above). Tuple
  lengths therefore never change, and a stored entry's `gn` stays comparable across dropdown
  changes and upgrades (§14.4).
  - Ask: Question, Context, Answers, Overrides, Key.
  - Pick: Prompt, Context, Items, Overrides, Key.
  - Match: Question, Context, Items, Answers, Overrides, Key.
  - Classify: Question, Context, Items, Labels, Overrides, Key.
  - Rules: Question, Context, Factors, Outcomes, Overrides, Key.
  - Dials: Prompt, Settings, Overrides, Key.

  `Key` is compared but never sent to the model, and is excluded from every `wkey` (§6.0).
- Collection and Object sockets cannot be compared in Geometry Nodes. The prototype's rule stays:
  if two contexts share every string but not the collection or object, the node reports an error
  ("only one answer can show"). §6.0 gives the shape-agreement rule and the `Key` remedy.
- **No match: `k_default`, single-context nodes only.** When the live words match no materialised
  entry, the Switch chain's final value is `k_default`. The inner group is shared by every context
  that uses the tree, so `k_default` is one value for the whole node. The pass therefore sets it
  as follows (`inner.update_values()`):
  - **Exactly one live context, and the node's state is neither hint nor error:** `k_default` is
    that context's last-shown entry (`LAST_SHOWN`, §14.5).
    - Between an edit and the next pass (at most about 0.35 s), the outputs keep that object's own
      last answers, marked stale, instead of flashing to fallbacks.
    - During playback, unprepared animated words keep the object's own last rules instead of
      breaking a simulation (§5.5).
  - **Two or more live contexts:** `k_default = 0`, so a no-match gives the neutral fallbacks, as
    in the prototype. While OpenAct is running, one object can never show another object's answers.
    (Without the add-on, `k_default` stays as saved; see R6.) *Prepare for Playback*
    (§5.5) remains the tool for animated words on shared trees.
  - **A hint or error state** (a cleared Question, "only one answer can show", a file error):
    `k_default = 0`, so errors are never hidden behind old answers.
  - **Never answered:** `k_default = 0`.
  - The pass then stores the new entry with stale-while-refilling rows (§5.4).
  - Owned by PR-11. Its lookup tests: two contexts with A's words unmatched show neutral values,
    not B's answers; a single context shows its own last answers; a cleared Question gives neutral
    outputs.

### 8.3 Per-element index

For Pick, Match and Classify:
```
k     = match_chain(words)                      # on no match: k_default (§8.2); 0 = neutral or never answered
base  = IndexSwitch(INT)[k]    items [0, base_1 … base_E]
count = IndexSwitch(INT)[k]    items [0, N_1 … N_E]
valid = (ID >= 0) AND (ID < count)              # Compare INT GREATER_EQUAL / LESS_THAN, Boolean Math AND
flat  = base + ID                               # Integer Math ADD
```

For Rules, `ID` is replaced by the factor graph's `cell`, and `valid` becomes the AND of every
factor's validity (§9).
- **Integer Math** (`FunctionNodeIntegerMath`) is available in 4.5.
- A float `Math` node would also be exact, because table indices stay below 2^16, far under 2^24.

**Output columns.** Every per-element column *c* is read from the node's table object (§8.4):

`v = Sample Index(Object Info(table).Geometry, Named Attribute("oa_<c>"), flat, POINT)`, then
`col_c = Switch(type_c)[valid](fallback_c, v)`.

There is one Object Info per node. Each lookup (column × interpolation corner) adds one Named
Attribute and one Sample Index.

**Derived outputs** are computed in Geometry Nodes (4.a):
- Match `Selection = valid AND (p ≥ Threshold)`.
- Pick `Selection = (ID == Index) AND (Index ≥ 0)`.
- Rules `Yes = Probability ≥ 0.5`.

```mermaid
flowchart LR
  GI["Group Input<br/>words, ID field, Threshold"] --> MC["Match chain<br/>Join Strings, Compare STRING, Switch INT"]
  MC -->|"k"| BS["Index Switch INT<br/>base per entry"]
  MC -->|"k"| CN["Index Switch INT<br/>row count per entry"]
  GI -->|"ID"| VA["valid = 0 ≤ ID < count"]
  CN --> VA
  BS --> FL["flat = base + ID"]
  GI -->|"ID"| FL
  subgraph SA["Per-element lookup (table object)"]
    OI["Object Info<br/>.OpenAct Table uid"] --> SX["Sample Index POINT<br/>Named Attribute oa_col"]
    SX --> SW["Switch valid ? v : fallback"]
  end
  FL --> SX
  VA --> SW
  SW --> GO["Group Output<br/>Probability, Selection, Confidence …"]
  GI -->|"Threshold"| TH["Selection =<br/>valid AND p ≥ Threshold"]
  SW -->|"p"| TH
  VA --> TH
  TH --> GO
```

### 8.4 Where values live: the table object for per-element data, Index Switch for single values

**Rule.** Every **per-element** column goes through the node's table object (§8.3, §8.5).
`Index Switch` is used only for small, bounded item lists:
- entry-level single values selected by `k`: Ask's outputs; Pick's Index, Item, Confidence,
  Material and payload; per-entry base and count. Each has E + 1 items, usually 2–5, and at most
  17;
- Rules band centres (at most 16 per Amount factor).

**Dials is the exception to watch.** At its 32-setting limit (§7.5), one Index Switch per setting
plus per-entry `min`/`max` switches would mean about 100 Index Switches of 17 items. Writing a
new entry's 32–96 values at about 0.9 ms per socket would take 30–85 ms, at or above the ≤ 30 ms
`update_values()` budget for one entry.
- Dials therefore stores its values in a **one-row-per-entry table object** (columns: each
  setting's value, `min`, `max`), read with `Sample Index` at `k − 1` behind a `Switch` on
  `k > 0`. Map Range then applies `min`/`max`, and a value update is a `foreach_set`.
- This needs `Sample Index` with a single-value Index to yield a **single value**, so Dials outputs
  can still feed single-value sockets.
  - **Verified by the review (round 3) on 4.5.2, 5.0.0 and 5.1.1.** A custom group node sampling a
    table object's INT attribute at a single-value index fed Mesh Grid's *Vertices X* through a
    valid link, and the grid evaluated correctly; 5.x infers the output as `SINGLE`.
  - The PR-08 spike re-checked it on 4.5.2 and 5.1.1 (`tests/lookup_blender.py` keeps it as a
    regression guard). The fallback (Index Switches, 16 settings) remains a contingency only.

Non-contiguous category value maps also go through the table object (§9.2).

**Why.** The review measured both sides in headless 4.5.2 and 5.1.1, in a tree used by a modifier:

| Operation | 14-node tree | 214-node tree | 814-node tree |
|---|---|---|---|
| Create 12 Index Switches × 65 items | 68 ms | ~900 ms | 2.7–2.9 s |
| Set their 780 item defaults | 77–81 ms | ~680 ms | 1.8–2.0 s |
| One `node.location` write, per node | 0.11 ms | ~0.9 ms | 2.3–2.6 ms |
| `tree.nodes.foreach_set("location", …)` for every node | ≈ 0.04 ms in total | | |
| Per-element lookup on 1M points (Index Switch with 65 or 257 items; Object Info + Sample Index with 65 or 4096 rows) | within ~1 ms of a no-lookup baseline (9–12 ms) | | |

- **Graph edits are the expensive side.** Every RNA write gets slower as the tree grows, even in a
  tree used by no modifier, so building a detached tree and swapping it in does not help.
- **Per-element lookup is nearly free on both paths.**
- **So per-element data belongs in the table object.** A value update is one `foreach_set`
  (≤ 5 ms), and the graph stays constant during fills. There is no threshold for history or row
  count to cross, and there is only one per-element path to keep correct.
- **`inner.build()` sets every location in one `foreach_set` call.**

**Graph-size budget: worst shapes derived from the documented limits.** Every shape uses 16
materialised entries, the cap in §8.1. The joined-key match chain costs about 2 nodes per entry,
so about 34 nodes (§8.2).

| Node type | Worst shape under the limits | Approx. nodes |
|---|---|---|
| Rules | Custom-edge compares are maximised by three 16-level Amount factors (16³ = 4096 cells, the cell limit), giving 45 Compare + 45 Add. Plus 8 interpolation corners × 3 columns = 24 lookups (48 nodes), weights and blend, and 16 payload lookups in One-of mode (not combined with interpolation) | ≈ 290 |
| Dials | 32 Amount settings, 8 nodes each: its value, `min` and `max` lookups from the Dials table object (§8.4), a Map Range and the `k > 0` Switch; plus the match chain | ≈ 300 |
| Pick, with payload | 16 payload columns and Material on `k` (17 items each), plus the per-element `prob` lookup | ≈ 70 |
| Match / Classify | a fixed graph, plus the chain; Classify with 16 payload lookups | ≈ 60–100 |

- The review measured a synthetic 379-node Rules-like group at about 1.2 s to build, roughly
  3.2 ms per node at that size. Each new node, link and property write costs a tree update
  proportional to the tree's size, so build time grows with the square of the node count.
- **Budgets:**
  - `inner.build()` ≤ 100 ms for the demo shapes, and ≤ 1 s at each derived worst shape. The
    interpolated Rules demo (snow) is allowed ≤ 250 ms: its 8-corner blend is a fixed graph of about
    170 nodes, whatever the table's size;
  - `update_values()` (entry-level Index Switches only) ≤ 30 ms when one entry's values change,
    and ≤ 150 ms when all 16 entries' values change at once (a numeric CSV edit re-baking the
    payload of every Pick context);
  - `tables.write_rows()` ≤ 5 ms.
- `inner.build()` runs only when the **shape or the materialised set** changes (§8.8), at most once
  per pass and never per chunk. A new prompt on a live context changes the materialised set, so it
  costs one rebuild.
- The **PR-08 contract spike** built each shape on 4.5 and 5.1 before the contracts froze (headless
  4.5.2 and 5.1.1, in a tree used by a modifier; stand-in graphs for the nodes that did not exist
  yet, built through the real `inner.build()`). The cap stays at 16; Dials' limit dropped from 64
  settings to 32, and the budgets above were split where the measurements required it:

  | Shape | Nodes | 4.5.2 | 5.1.1 |
  |---|---|---|---|
  | Ask / Pick, the prototype's scene (2 entries) | 13 / 17 | 2 / 3 ms | 2 / 3 ms |
  | Classify (materials), Rules One-of (biome), Dials (weather), 1 entry | 24 / 45 / 51 | 4 / 14 / 19–20 ms | 4 / 12 / 16–18 ms |
  | Rules Amount, interpolated (snow), 1 entry | 169 | 166–168 ms | 115–163 ms |
  | Ask / Pick, 16 entries | 69 / 73 | 75–76 / 61–83 ms | 54–75 / 60–83 ms |
  | Pick with payload / Classify with payload, 16 entries | 68 / 96 | 200–273 / 93–94 ms | 197–202 / 79–80 ms |
  | Rules worst / Dials at 32 settings, 16 entries | 289 / 297 | 580–590 / 540–545 ms | 583–595 / 520–531 ms |
  | Dials at 64 settings, 16 entries (the former limit) | 553 | 1.99 s | 1.87 s |
  | `update_values()`, one entry / all 16 entries, Pick with payload | 20 / 320 sockets | 7–9 / 100–129 ms | 7 / 97–101 ms |
  | `update_values()`, all 16 entries, Ask / Pick | 48 / 48 sockets | 13–14 / 11–14 ms | 10–13 / 11–15 ms |
  | Planning a 4096-cell context (§5.3; Python 3.11 / 3.13) | — | 118–121 ms | 120–123 ms |

  These are three runs on an otherwise idle machine. With another project's 8-thread Blender job
  running, the 16-entry Rules and Dials builds rose to 0.71–0.91 s, still within budget, and planning
  to about 0.2–0.27 s.
- `tests/lookup_blender.py --bench` enforces these budgets as merge gates: `tests/run_all.ps1` runs
  it on every Blender version (§18.1). It times each measurement as the fastest of several runs and
  follows the timing rule of §18.2, because the gate runs while other work uses the machine: it
  prints every measurement against its budget, warns above the budget, and fails only above twice
  the budget.

| Typical node | Table object (per materialised entry) | Index Switch |
|---|---|---|
| Ask | none | entry-level values only |
| Dials | a one-row-per-entry table: each setting's value, `min`, `max` and `conf` (§8.4), read at row `k − 1` behind a `Switch` on `k > 0` | none: no base or count is needed |
| Pick | one `prob` column, only while Selection or Probability is linked | Index, Item, Confidence, Material, payload |
| Match, 40 props | 40 rows × (`p`, `c`) | base and count |
| Rules: Game of Life, 18 cells | 18 rows × (`outcome`, `c`, `alive`) | base and count |
| Rules: snow, Amount, 8 corners | 27 rows × (`amount`, `c`, `ok`); 24 lookups | base, count and band centres |

The per-element data is not visible inside the inner group; the sidebar's table view shows it
(§17.2). Files still evaluate without the add-on, because the table object is plain data (§8.5).

### 8.5 Hidden table objects (`tables.py`)

- **Object.** `.OpenAct Table <uid>`, with mesh data of the same name. It has one loose vertex per
  flat row (vertex *i* = flat row *i*) and one POINT attribute per column:
  - `oa_p`, `oa_c`, `oa_ok`, `oa_label`, `oa_outcome`, `oa_amount`, `oa_prob`;
  - payload as `oa_pl_<k>`, where *k* is the payload column's position. The header → *k* map is
    stored in the entry's `shape`, so any header text is safe.
  - Types are `FLOAT`, `INT` or `BOOLEAN`. Positions are unused, so all vertices sit at the
    origin.
- **No collection: the table object is linked to no collection at all.** Its only user is the
  inner group's Object Info socket default.
  - The review verified on 4.2–5.1 that such an object still evaluates through Object Info
    (vertex count and `oa_*` attributes arrive).
  - It verified on 4.5.2 and 5.1.1 that later `foreach_set` + `mesh.update()` writes reach the
    next evaluation.
  - The object is never rendered, selected or shown in a scene's outliner, and there is nothing to
    re-exclude or re-link on each pass.
  - `tests/interface_blender.py` checks, on 4.5, 5.0 and 5.1, that it survives save and reload,
    *Purge Unused Data*, and rendering (F12 and `blender -b -f`).
  - **Contingency:** if a future version fails that test, link table objects into a collection
    excluded in every view layer. That is also verified behaviour (§3.4), and it would be a
    `tables.py`-only change.
- **Reference.** The inner group's `Object Info` (As Instance off, Original space) holds the object
  as its socket default value. That makes the object a user-counted ID, saved with the file.
  Appending or linking the node group brings it along with no extra step.
- **Writes.** `tables.write_rows(node, flat_rows)`:
  1. **Row count unchanged:** for each column, `mesh.attributes[name].data.foreach_set("value",
     values)`.
  2. **Growing:** `mesh.vertices.add(F − current)`, which keeps and resizes existing POINT
     attributes (verified), then step 1.
  3. **Shrinking:** `mesh.clear_geometry()` removes every custom attribute (verified on 4.2–5.1:
     only `position` remains). So `mesh.vertices.add(F)`, then re-create each column from `shape`
     with `mesh.attributes.new(name, type, 'POINT')`, then step 1.
  4. Finish with `mesh.update()`.

  The depsgraph re-evaluates the users. Writing 16 × 4096 rows × 6 columns takes a few
  milliseconds. The lookup test runs a grow → shrink → grow sequence and checks every column after
  each step.
- **Copies.** When `ensure_owned()` gives a copied node its own inner group, `tables.copy_for()`
  also runs:
  - it copies the object and its mesh as `.OpenAct Table <new uid>`;
  - it re-points the copied inner group's Object Info.

  `copy()` itself still only clears `uid` (§3.4).
- **Garbage.** Tables are identified by **reference, never by name**. Each table object carries the
  custom property `openact_uid`. Each pass collects the set of objects referenced by the Object
  Info node of every live node's inner group. A table object is removed (`bpy.data.objects.remove`,
  with its mesh) only if it has `openact_uid` **and** no live inner group references it.
  - Appending renames duplicates to `….001`. A name-based check could therefore remove a live
    table; a reference-based check cannot.
  - Even without that step, an orphaned table loses its last user once its inner group is gone,
    and Blender drops it on save. A Pick whose
  Selection and Probability outputs are both unlinked releases its table object the same way
  (§6.2).
- **Without the add-on.** The table object and the inner group are plain data, so Blender 4.5+
  evaluates them in files opened without OpenAct. The e2e `noaddon` phase covers this (§18).

### 8.6 Single-value outputs

Entry-level outputs use the prototype's `_Builder.lookup`: one Index Switch per output, indexed by
`k`, with items `[fallback, v_1 … v_E]`.

| Output | Graph |
|---|---|
| Ask: Answer, Yes, Confidence | Index Switch (FLOAT / BOOLEAN / FLOAT) on `k` |
| Pick: Index, Item, Confidence, payload | Index Switch (INT / STRING / FLOAT / INT or FLOAT) on `k` |
| Pick: Instance | As in the prototype: Collection Info (Separate Children, Reset Children) → Delete Geometry (INSTANCE, `Index ≠ picked`) |
| Pick: Material | Index Switch (**MATERIAL**, verified) on `k`, with items `[None, m_1 … m_E]`, where `m_e` is entry *e*'s **picked** material. It is stored by name in `single` (§14.2) and resolved with `bpy.data.materials.get()` at build time; a missing material gives None and a warning. Contexts on different objects therefore each get their own material |
| Dials: each setting | `Sample Index` on the node's **one-row-per-entry table object** at `k − 1`, behind a `Switch` on `k > 0`, with fallback (§8.4). Amount settings then go through `Map Range` (FLOAT, Linear, no clamp, From 0–1), whose *To Min* and *To Max* are sampled from the same row (`min_<s>`, `max_<s>`, per-entry data, §6.0). Single-value sampling was verified by the PR-08 spike; Index Switches on `k` with at most 16 settings remain a contingency only (§8.4) |
| Dials: Confidence | Sampled from the same table row (`conf`): the minimum confidence of each entry |

These outputs are **single values**: their graphs use no field input. They therefore connect to
single-value sockets such as a Grid's *Vertices X* or Mesh Line's *Count*, just as in the
prototype.

### 8.7 Fields, zones and performance

- **Field inference.** An output is a field exactly when it depends on the ID or on factor inputs
  (Pick Selection and Probability; every Match, Classify and Rules output). Blender infers this
  from the inner graph, so no flags are needed.
- **Zones.** The inner group is stateless: a pure function of its inputs. Object Info re-reads the
  table object on every evaluation, so lookup works inside Simulation and Repeat zones. The Game of
  Life acceptance test runs 32 frames inside a Simulation Zone and compares every cell with a
  Python reference (§18).
- **Budgets** (from §3.2), enforced by `tests/lookup_blender.py --bench`, which `tests/run_all.ps1`
  runs as a merge gate (§18.1):
  - **Per-element lookup, 1M points:** ≤ 5 ms above a no-lookup baseline, for up to 4096 rows and
    8 interpolation corners. The review measured within ~1 ms.
  - **Main thread, the expensive side:**
    - `inner.build()` ≤ 100 ms for the demo shapes (≤ 250 ms for the interpolated Rules demo) and
      ≤ 1 s at the documented worst shape;
    - `update_values()` ≤ 30 ms when one entry's values change, ≤ 150 ms when all 16 entries'
      values change at once (§8.4);
    - `tables.write_rows()` ≤ 5 ms for 16 × 4096 rows.

  The bench follows the timing rule of §18.2: it prints every measurement against its budget, warns
  above the budget and fails only above twice the budget, so other work on the machine cannot fail
  the gate. If a measurement fails, the PR that introduced the cost must fix it before merge.
- **No model at evaluation.** Evaluation touches only Index Switch, Compare, Switch, Integer
  Math, Map Range, Random Value, Object Info, Named Attribute and Sample Index. No Python runs
  (4.b).

### 8.8 Rebuild or values-only update

| Change | What happens |
|---|---|
| Answers arrive, locks or numeric payload edited, confidence changes | **Values only.** `tables.write_rows()` writes the table object: per-element columns, and Dials' one-row-per-entry values, `min` and `max` (§8.4), at most once per 0.5 s during a fill. `inner.update_values()` sets changed entry-level Index Switch defaults (Ask, Pick) |
| `k_default` changes (§8.2): a new last-shown entry, a change in the number of live contexts, or entering or leaving a hint or error state | **Values only.** `update_values()` sets the final Switch value of the match chain |
| Row count of an entry changes (items added or removed) | **Values only.** The table object's vertex count changes, and base/count items are updated; the graph does not change |
| The materialised set changes (a live context's words changed, or a warm entry is dropped, §8.1) | **Rebuild** of the match chain and entry-level switches: at most one per edit |
| Answer type, list source, factor layout, or the payload column set changes | **Rebuild**, after `inner.sync_interface()` |

A rebuild keeps the prototype's approach (`tree.nodes.clear()`, then build), plus the batched
location write. The interface is never cleared: `sync_interface()` diffs it (§6.0). Links from the
outer tree therefore survive every rebuild.

## 9. Rules Engine Details

### 9.1 Overview

The work is split between Python (planning) and nodes (applying):

| Python, when words change (`specs/rules.py`) | Geometry Nodes, per element (`inner.py` builds it) |
|---|---|
| 1. Parse the Factors CSV (§7.4) and the Outcomes answer set (§7.3) | 1. Map each factor socket's value to a level index (§9.2) |
| 2. Enumerate cells in row-major order (§9.3) | 2. Combine the levels into a cell index (§9.3) |
| 3. Evaluate `if` restrictions and decide single-option cells by code (§9.4) | 3. Look up the cell's row (§8): outcome / p / amount, confidence, ok, payload |
| 4. Build one question per remaining cell, all sharing one state (§9.5) | 4. Yes/No and Amount: blend corner rows smoothly (§9.6). One-of: optionally dither band coordinates (§9.7) |
| 5. Write rows into the table (§8); refill when words change | No Python, no model (4.b) |

```mermaid
flowchart LR
  FS["Factor sockets<br/>Float / Int / Bool fields"] --> LV["Level index per factor<br/>band / category map / flag"]
  DS["Dither, Seed<br/>Random Value per Amount factor"] -.->|"One-of only"| LV
  LV --> CI["cell = Σ b_j · stride_j<br/>valid = AND of valid_j"]
  MC["Match chain → k → base"] --> FL["flat = base + cell"]
  CI --> FL
  LV -->|"Yes/No, Amount: b0_j, t_j"| CO["2^Ka corners, weights w_s"]
  CO --> FL
  FL --> LK["Lookup in the table object<br/>outcome or p / amount, c, ok, payload"]
  LK -->|"One-of"| GO["Group Output"]
  LK -->|"Yes/No, Amount"| BL["Renormalised blend"] --> GO
```

### 9.2 Level index per factor (a node graph)

`x = clamp(v, 0, 1)` for Amount factors. `b` is the level index, and `valid` is true unless noted
otherwise.

| Kind | Graph | Nodes |
|---|---|---|
| `amount`, equal width, *n* bands | `b = min(floor(x · n), n − 1)` | Clamp → Math MULTIPLY → Float to Integer (FLOOR) → Integer Math `MINIMUM` |
| `amount`, custom edges `e_1 < … < e_{n−1}` (the interior `from` values) | `b = Σ_{i=1}^{n−1} [x ≥ e_i]` | *n* − 1 Compare (FLOAT, GREATER_EQUAL) → Integer Math `ADD` chain (at most 15). The Bool → Int step is Blender's implicit socket conversion, not a node |
| `category`, contiguous values `u_i = u_0 + i` | `b = u − u_0`, `valid = 0 ≤ b < n` | Integer Math SUBTRACT, 2 Compare, Boolean AND |
| `category`, non-contiguous values within 0–4095 (§7.4) | `b = Sample Index(table, oa_map_j, map_base_j + u)` if `0 ≤ u ≤ u_max`, otherwise −1; `valid = b ≥ 0`. The map rows (level index, or −1 for unused values) are stored in the table object after all cell rows. `map_base_j` is **computed at build time** from the materialised layout, not stored. Because factor values must agree across contexts (§6.0), one map serves every entry | 1 Sample Index + 1 Named Attribute, plus 3 small nodes, whatever *n* is |
| `flag` | `b = f ? level(value 1) : level(value 0)` | Switch (INT) |

- A category value that matches no level makes the cell invalid, which gives the fallback outputs
  (4.d).
- An Amount value outside 0–1 is clamped into the first or last band. That is intended: authors
  normalise inputs with Map Range (4.a).

### 9.3 Cell index

Factors *j* = 0…K−1 are taken in CSV order, with *n_j* levels. The strides are row-major, so the
last factor varies fastest:

```
stride[K-1] = 1;   stride[j] = stride[j+1] * n[j+1]
cell  = Σ_j b[j] * stride[j]          # Integer Math MULTIPLY_ADD chain
valid = AND_j valid[j]
cell  = valid ? cell : -1
```

- Python enumerates rows with `itertools.product(*(range(n) for n in n_levels))`, which is the same
  order. Table row *r* is therefore cell *r*.
- A pure-Python unit test asserts `ravel(product_row(r)) == r` for all *r*. The GN lookup test
  asserts that the node graph produces the same index for every combination (§18).

**Game of Life.** `state` (flag, *n* = 2: dead = 0, alive = 1) and `neighbours` (category, *n* = 9,
values 0–8, contiguous) give strides (9, 1). So `cell = 9 · b_state + b_neighbours`. A live cell
with 3 neighbours is cell 9 + 3 = 12. Row 12 is (alive, 3), whose outcome is *Survives*.

**Biome.** `temperature`, `moisture` and `altitude` have 3 levels each, so the strides are
(9, 3, 1) and there are 27 cells. (hot, wet, low) = 2·9 + 2·3 + 0 = cell 24.

### 9.4 Exact-fact restrictions (`if`) and cells decided by code

Grammar of the Outcomes CSV `if` column (`specs/rules.py::parse_if`):

```
if      := ""  |  clause ( ";" clause )*        # every clause must hold (AND)
clause  := factor ( "=" | "!=" ) levels
levels  := level ( "|" level )*                 # any listed level (OR)
```

- Names are trimmed and matched ignoring case.
- A `level` is a level name. For `category` and `flag` factors it may also be the level's
  `value`, so `neighbours=2|3` and `state=1` both work.
- An `amount` factor may be restricted by band name (`temperature=cold|mild`). The test uses the
  cell's band.
- Unknown factors or levels, and syntax errors, are errors that name the line (§7.9). An outcome
  that no cell allows is a warning.

When planning, Python evaluates every cell against every outcome (at most 4096 × 255 checks,
well under 50 ms) and gets each cell's **allowed set** *A*:

| Answer type | \|A\| = 0 | \|A\| = 1 | \|A\| ≥ 2 |
|---|---|---|---|
| One-of | error row (Outcome −1); status "No outcome applies to state=dead, neighbours=3" (first 3 cells listed) | **decided by code**: that outcome, c = 1, source `code`, no request | `choice` question over *A* only |
| Yes/No (`if` on the `yes`/`no` rows) | error row | code: p = 1 for `yes`, p = 0 for `no` | `noul` question |
| Amount | `if` is ignored, with a warning ("'if' is not used for Amount answers") | — | `score` question over all levels |

This is principle 4.d made concrete. In the measured casual rule set, offering only the clauses
that matched the known current state turned 3 wrong cells into an exact table.

### 9.5 The question for one cell

All cells of a Rules context share **one state**. Each cell's facts go into its instructions.
That gives maximal batching (§5.5), and editing one level's `when` changes only the keys of the
cells that include it (§5.4). Templates live in `rules_core.py`, which `specs/rules.py` builds on
(`PROMPT_VERSION = 2`).

**State** (shared by every cell):
```
{Context, or "No additional context."}
Each question describes one situation with these facts:
- state: dead | alive
- neighbours: 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8
Each option is an outcome and when it applies.
Decide only by the conditions the options state, not by outside knowledge.
Choose the outcome whose condition this situation meets.
```

- **The last three lines** (the conditions lines) are added only in One-of mode, and only when
  at least one outcome has a `when`. They make the outcomes' conditions decide the answer, not
  what the model knows about the subject (4.c). With `PROMPT_VERSION = 1`, which lacked them, the
  model followed Conway's rules over the Seeds outcomes: a dead cell with two neighbours stayed
  empty instead of being born. It was also unsure on three Day & Night cells.
- Yes/No and Amount states end after the facts, and so do One-of states whose outcomes are bare
  names: those answers need what the model knows.

**Instructions** (one per cell, here cell 12):
```
Situation:
- state: alive (The cell holds a living cell.)
- neighbours: 3 (Exactly three live neighbours.)
What happens to this cell in the next generation?
```

**Options** (criteria), keyed by outcome row index and limited to the allowed set:
```json
{"2": "Survives: A live cell with two or three live neighbours stays alive.",
 "3": "Dies: A live cell with fewer than two or more than three live neighbours dies."}
```

- **One-of** sends `type: "choice"`, with each option written as `"<answer>: <when>"`. This is
  the "outcome + when it applies" framing that produced the exact Game of Life, HighLife, Seeds
  and Day & Night tables (4.c). A bare name is sent as the name alone.
- **Yes/No** sends `type: "noul"` with `criteria = {"true": <yes when>, "false": <no when>}`, or
  `Yes.`/`No.` when the whens are absent.
- **Amount** sends `type: "score"`, with the levels as `"<name>: <when>"`, lowest first.
- **Empty Question.**
  - One-of uses "Which outcome applies in this situation?".
  - Yes/No and Amount show the hint "Type a question", because a scale without a question is
    meaningless.
- **Acceptance.** Template changes must keep all four Game of Life rule sets at 18/18 exact with 0
  unsure cells in the e2e suite (§18). Measured with `PROMPT_VERSION = 2`: all four at 18/18 with
  0 unsure; the lowest confidence is 0.44 (Day & Night, a live cell with three neighbours).
  `PROMPT_VERSION` is stored per entry but never forces a re-ask (§5.4).

**Cost.** Each cell with 2–16 options is 2 scored rows with debias. A cell with *m* > 16 options is
a tournament of `2 × (ceil(m / 16) + 1)` rows.

### 9.6 Smooth interpolation (Yes/No and Amount)

The first **K_a ≤ 3** Amount factors, in CSV order, interpolate between band centres. Any further
Amount factors use their nearest band, with the warning "interpolating over 3 Amount factors;
'<name>' uses its nearest band".

For each interpolated factor *j* with *n* bands:

| | Equal-width bands | Custom edges |
|---|---|---|
| Band centres | `m_b = (b + ½) / n` | `m_b = (from_b + to_b) / 2` |
| Lower corner `b0` | `u = clamp(x·n − ½, 0, n − 1)`, `b0 = min(floor(u), n − 2)` | `b0 = clamp(Σ_{b=1}^{n−2} [x ≥ m_b], 0, n − 2)` |
| Weight `t ∈ [0, 1]` | `t = u − b0` | `t = clamp((x − m_b0) / (m_{b0+1} − m_b0), 0, 1)`, with the centres read from an Index Switch (FLOAT) |

Outside the outermost centres, `t` clamps. The outer half-bands are therefore flat: there is no
extrapolation.

**Blend** over the corners `s ∈ {0,1}^{K_a}` (at most 8):
```
cell_s = cell_exact + Σ_j (b0_j + s_j) * stride_j      # cell_exact: category, flag and nearest-band factors
w_s    = Π_j (t_j if s_j else 1 - t_j)
value  = Σ_s w_s * ok_s * v_s  /  Σ_s w_s * ok_s       # v = p (Yes/No) or amount; confidence blended the same way
valid  = Σ_s w_s * ok_s > 1e-6                          # otherwise: fallback outputs
```

- `ok_s` is 1 for answered, locked and code rows, and 0 for pending or error rows. The
  renormalisation stops a missing corner from dragging values towards 0 while a table fills.
- `Yes = Probability ≥ 0.5` is computed after the blend.

**Why multilinear is safe.**
- The result equals the table exactly at every band centre, and it is continuous everywhere.
- Between two centres it moves monotonically whenever the table does.
- Measured Amount answers are smooth and monotonic: snow cover runs from 0.08 (low and hot) to
  0.89 (high and cold). Interpolation therefore removes stair steps without inventing values.

**Graph cost.** Each corner needs an Integer Math cell index and three lookups (value, c, ok),
plus weight Math nodes. Eight corners make 24 Sample Index lookups on the table object. Their
per-element cost is negligible (§8.7), and they count towards the graph-size budget (§8.4).

### 9.7 Dithering (One-of only)

One-of answers are discrete, so averaging outcome indices would be meaningless. Instead, `Dither`
(0–1) blends outcomes stochastically near band edges. For each Amount factor *j*:

```
r_j  = Random Value(FLOAT, 0..1, ID = implicit id-or-index, Seed = Seed + 7919 * j)
w_j  = width of the element's undithered band      # 1/n, or to_b - from_b from an Index Switch (FLOAT)
x'_j = clamp(x_j + (r_j - 0.5) * Dither * w_j, 0, 1)
b_j  = band of x'_j (§9.2)
```

- **Which elements change band.** Only elements within `Dither/2` band widths of an edge can
  switch. The chance of taking the neighbouring band rises linearly from 0 at that distance to ½
  at the edge. The result is a blend zone of width `Dither × w_j` centred on each edge, with the
  correct average proportions.
- **Stability.** Random Value's implicit ID uses the `id` attribute when it exists and the index
  otherwise. Dithered choices are therefore stable across frames inside Simulation Zones.
- **Exact factors never dither.** Categories and flags are exact facts (4.d).
- **No rebuilds.** The Random Value branch is built whenever the node is in One-of mode, and at
  `Dither = 0` it is multiplied away. Changing Dither or Seed never rebuilds anything. They are GN
  values, not words, so they never trigger a pass either.

### 9.8 Limits, costs and worked figures

| Quantity | Limit | On violation |
|---|---|---|
| Factors | 1–8 | error |
| Levels: `amount` / `category` / `flag` | 2–16 / 2–64 / exactly 2 | error |
| Cells (product of level counts) | ≤ 4096 | error, before any request |
| Interpolated Amount factors | ≤ 3 | nearest band beyond 3, with a warning |
| Outcomes | ≤ 255 (a tournament above 16 allowed options) | error |
| Estimated scored rows per context | ≤ 8192; ETA shown above 1024 | error |

| Table (RTX 3090, order balancing on, the default) | Cells | Options per cell | Scored rows | Fill time |
|---|---|---|---|---|
| Game of Life with `if` restrictions | 18 | 2 | 36 | ≈ 0.8 s (measured 0.5–2.7 s) |
| Game of Life without `if` | 18 | 4 | 36 (longer rows) | ≈ 1 s; accuracy risk (§2.2) |
| Biome, One-of | 27 | 7 | 54 | ≈ 2 s (14 cells/s) |
| Snow, Amount with 5 levels | 27 | 5 | 54 | ≈ 1.6 s |
| 4 factors × 5 bands, Yes/No | 625 | 2 | 1250 | ≈ 26 s; ETA shown |
| Largest allowed | 4096 | 2 | 8192 | ≈ 3 min |

The fill times use the measured rates: 21 ms per two-option row, 36 ms per seven-option row, and
about 30 ms per five-level score row.

## 10. Engine & Server Changes

### 10.1 Static evaluator (`evaluator.py`)

The evaluator moves out of `engine.py` **without behaviour changes**:
- `eval_input`, `eval_output`, `_group_input` (modifier values, `*_use_attribute`, nested groups
  through `Ctx.push`/`pop`) and `_group_output`;
- muted nodes and links, and Join Strings ordered by `multi_input_sort_id` descending;
- Import Text normalisation;
- `find_uses` and `Ctx` (at most 64 contexts per tree, at most 32 nesting levels).

`tests/evaluator_blender.py` must keep passing with only its import renamed.

**Changes:**

| Change | Why | Detail |
|---|---|---|
| `_jev_output` → `_openact_output` | Six node types plus the legacy aliases | **Single-value** outputs resolve from `MEMORY` or the stored entry via `spec.output(entry, out.identifier)`, else raise `Pending`. They are: Ask Answer/Yes/Confidence; Pick Index/Item/Confidence/payload; each Dials setting and its Confidence. **Field** outputs raise `Unresolvable("'Match' output 'Probability' is a field; words must be single values")`. Geometry and Material outputs raise `Unresolvable`, as today |
| New handler: Menu Switch (`GeometryNodeMenuSwitch`) | Modifier dropdowns choose rule sets and presets (§7.7) | Algorithm, from the review's measured storage (4.2–5.1). `NodeEnumItem` exposes only `name` and `description`. 1) **A socket value is matched by name**: a Menu socket's `default_value` is the item name. 2) **A modifier value is an int identifier** (for a Menu reaching the modifier through Group Input). It is mapped to a name through `mod.id_properties_ui(key).as_dict()["items"]`, whose tuples end with the int, e.g. `('A','A','',0,0)`. The int is **never** used as a position in `enum_items`, because identifiers stay fixed when items are removed or reordered. 3) **`-1`** (a fresh modifier; the UI default is −1 on 4.4+ and 0 on 4.2/4.3) means the group interface socket's default item; the evaluator test verifies this. 4) The name is matched against `node.enum_items`, and that item's input socket is evaluated. 5) An unknown name or value raises `Unresolvable` naming the menu. The evaluator test pins each step against Geometry Nodes ground truth on 4.5, 5.0 and 5.1, including deleted and reordered items and a Menu inside a nested group |
| New handler: Special Characters (`FunctionNodeInputSpecialCharacters`) | Multi-line Items or Labels built with Join Strings | "Line Break" → `"\n"`, "Tab" → `"\t"` |
| New handler: String Length (`FunctionNodeStringLength`) | Int words, e.g. an Index Switch index | `len(text)` in code points; parity checked by the evaluator test |
| Per-pass memo | Six node types often read the same upstream words | Cache keyed by `(socket.as_pointer(), ctx key)`, cleared at the start of each pass. `ctx key` is `(obj pointer, modifier name, group-node pointers)` |
| Breadcrumb errors | The artist must see *which* socket fails | `Unresolvable` carries up to 3 hops: "Context ← Join Strings ← Named Attribute: fields can't be words" |
| Only word sockets are read | ID, factor sockets, Threshold, Dither and Seed belong to Geometry Nodes | `spec.words(node)` plus the list-source sockets are the only inputs the evaluator reads. Everything else is ignored, never reported |
| Linked libraries | Linked data is read-only | Evaluation works. `engine` skips writes and shows "Linked from <library>: answers come from that file" |
| New: `trace_csv_source(node, csv_socket, ctx) -> CsvSource` | The Import CSV hand-off (user decision, §6.0) | Walks **geometry** links upstream from a node's `<Input> CSV` socket: Reroutes; nested group boundaries (Group Output into a group, Group Input out to the parent node's socket); Switch / Index Switch / Menu Switch of type Geometry, whose selector is evaluated with `eval_input`. It must end at a `GeometryNodeImportCSV`, whose Path and Delimiter inputs (found by identifier) are evaluated with `eval_input`, so modifier values work. It returns `CsvSource(path, delimiter, via=[node names])`. Any other node raises `Unresolvable` with "connect an Import CSV node", or "connect the Import CSV node directly" for a modifying node in between. The trust rule applies to the resulting path (§7.7) |

**Word sources supported in v1:**
- Group Input: modifier values, nested groups and the tree's interface defaults.
- The String, Integer, Boolean, Value, Object and Collection input nodes.
- Join Strings, Value to String, Replace String, Slice String, String Length, Special Characters.
- Switch, Index Switch, Menu Switch.
- Import Text, Reroute, and muted nodes and links.
- The single-value outputs of OpenAct nodes.

Anything that needs geometry or fields (Named Attribute, Sample Index, Import CSV, …) raises
`Unresolvable` naming the node, which is the prototype's rule. The one intentional
data-derived word source is the **Attributes** list source (§6.0). It reads attribute *names*,
never values, from the object's **original** data through `geometry.attribute_names()`. It never
reads the evaluated geometry, which would include this node's own output, and it stays outside the
evaluator.

The Import Text handler obeys the file trust rule (§7.7, §16.1): in an untrusted file it reads
only inside the .blend's folder tree.

### 10.2 Specs and prompt templates (`specs/`)

```python
class Spec:                                        # specs/base.py
    name: str                                      # "ask" | "pick" | "match" | "classify" | "rules" | "dials"
    idname: str                                    # "GeometryNodeOpenActAsk", ...
    PROMPT_VERSION = 1                             # stored per entry as "pv"; part of the ask key, never of a words key (§5.4)
    def words(self, node) -> list[str]: ...        # String word sockets, in match-chain order (§8.2)
    def interface(self, node) -> list[SocketDef]: ...
    def plan(self, node, ctx) -> Job: ...          # raises Hint / Pending / Unresolvable
    def rows(self, job, answers) -> list[Row]: ...
    def output(self, entry, identifier): ...       # single-value outputs, for chaining (§10.1)
    def summary(self, entry) -> str: ...           # status line and sidebar
```

`specs/base.py` also provides the shared helpers that make principle 4.g real:
- `answer_set(node, ctx, socket, kind)` → `AnswerSet(options=[Option(name, when, if_expr,
  payload)], doc)`;
- `item_list(node, ctx)` → `ItemList` (via `lists.resolve_items`);
- `option_text(o)` returns `"name: when"`, or `name` when there is no `when`;
- `confidence(kind, distribution)` implements the 4.e formulas.

**Templates.** `{…}` marks a substitution. `NO_CONTEXT = "No additional context."` and
`PICK_STATE = "Choose one of the listed items."` are kept from the prototype. The criteria keys
are row indices as strings, as in the prototype's Pick.

| Node | `state` (shared per job) | `instructions` (per question) | Type and criteria |
|---|---|---|---|
| Ask | Context or `NO_CONTEXT` | `{question}` | `noul` with `{"true": yes.when, "false": no.when}`, or `score` with the levels `"name: when"` |
| Pick | Context or `PICK_STATE` | `Which item best fits: {prompt without trailing "?"}?` | `choice` over the items, each `option_text` |
| Match | Context or `NO_CONTEXT` | `Item: {name}` + `" — {when}"` if present + `"\n{question}"` | `noul` or `score`, as Ask |
| Classify | Context or `NO_CONTEXT` | `Item: {name}` + `" — {when}"` + `"\n{question or 'Which label fits this item best?'}"` | `choice` over the labels |
| Rules | as in §9.5 | as in §9.5 | as in §9.5 |
| Dials | Prompt (required) | the setting's `question` | `yes/no` → `noul`; `amount` → `score` (default `AMOUNT_LEVELS`); `one-of` → `choice` |

- Every table (Match, Classify, Rules, Dials) has **one** state, so its questions share the SemIf
  prefix and batch in `score_shared` (§5.5).
- `summary()` examples:
  - Ask: `Yes (93% yes)`
  - Pick: `armchair (71%)`
  - Match: `12 of 40 yes · 2 unsure`
  - Classify: `27 items · 7 labels used`
  - Rules: `18 cells · 0 unsure · 2 locked`
  - Dials: `5 settings · lowest confidence 0.62`

### 10.3 Batching, order debiasing and tournaments

**One request per chunk** (`pipeline.build_request(chunk)`, chunk limits in §5.5):

```json
{"model": "semif-qwen3.5-4b",
 "state": "A cellular automaton on a square grid; every cell has eight neighbours.\nEach question describes one situation with these facts: ...",
 "questions": {
   "c3f9a1d27b0e4f15": {"type": "choice", "instructions": "Situation:\n- state: alive (...)\n- neighbours: 3 (...)\nWhat happens to this cell in the next generation?",
                        "criteria": {"2": "Survives: ...", "3": "Dies: ..."}},
   "7b20e4aa91c35d08": {"type": "choice", "instructions": "...", "criteria": {"...": "..."}}},
 "options": {"debias": true}}
```

- **Question ids** are the first 16 hex digits of the cell key. On a collision within one
  request, the full key is used.
- **Parsing.** `noul` gives `p`. `choice` gives `probabilities` and `choice`. `score` gives
  `probabilities` and `score`.
- **Confidence** is always recomputed in Blender with the 4.e formulas. The server's `confidence`
  field is only logged, so every backend yields the same semantics.
- **Short circuit.** Single-option questions are answered in Blender (4.d) and never sent.
- **Failure isolation.** On a 422, the server names the failing question (§10.4). If it cannot,
  the worker bisects the chunk. Isolating one bad question in 32 while still answering the good
  halves takes up to about 10 extra requests (both halves at each of 5 levels). Only the offending
  cell
  goes to `FAILED`, for example "prompt too long"; the rest are answered.

**Order debiasing** is unchanged on the server (`_distributions`). Each question is scored with its
options in the given order and reversed, and the two distributions are averaged.
- It costs exactly 2× the scored rows.
- **It is a per-node toggle** (user decision): the node property `balance_order`, shown on every
  node as *Balance Option Order*.
  - A new node takes its value from the global preference `debias` ("Balance option order for
    new nodes", default **on**). Changing the preference later leaves existing nodes as they are.
  - Some work doesn't need the double check. Turning the toggle off halves the scored rows for
    that node only.
  - Small models favour early options, and every exact Game of Life table was measured with order
    balancing on, which is why on is the default.
- **The node's current setting is part of the ask key** and of every **cell key** (the caches),
  but not of any **words key** (§5.4, KD13). So:
  - toggling it sends 0 requests and never discards stored answers, which are labelled "asked
    with older settings";
  - new asks, after a change of words, use the node's current setting;
  - *Ask Again* re-asks every row with it.

  Each chunk is one node's job, so its request carries that node's `options.debias`. This is
  tested in `pipeline_blender.py` (§18).
- Measured throughput with order balancing: about 24 yes/no cells/s and about 14 seven-option
  cells/s. Without it, roughly double.
  - Measuring accuracy with and without it is a follow-up, not a blocker (§21 F2).

**Tournaments** (more than 16 options) keep the prototype's server algorithm:
1. Balanced groups: sizes differ by at most 1, and every group has at least 8 options.
2. A final between the group winners.
3. `P(item) = P(item | group) × P(group winner wins the final)`.
4. Choice = argmax of the merged distribution; confidence = top minus runner-up of that
   distribution.

**Cost.** `debias × (groups + 1)` rows. 255 options → 16 groups + 1 final → 34 rows ≈ 2.7 s
(measured).

**Known limitation.** A strong runner-up inside the winning group keeps only its in-group share,
so tournament confidence is conservative, and the unsure flag surfaces this. When a Classify, Match
or Rules question has more than 16 options, every item or cell pays the tournament cost. The
8192-row budget (§5.4) bounds this before any request is sent.

### 10.4 Server changes (`runtime/openact_server.py`, from `jev_server.py`)

| Area | `jev_server.py` today | `openact_server.py` |
|---|---|---|
| Identity | `SERVER_NAME = "jev-nodes-semif"`, 0.1.0 | `"openact-semif"`, 1.0.0; handler `OpenActSemIf/1.0.0` |
| Model code | inside `Engine.load` / `Engine._score_rows` | `SemIfTorchBackend(Backend)` and `FakeBackend(Backend)`; flags `--backend {semif,fake}` and `--fake-script <json>` |
| Cache | LRU of 4096 **whole requests**, key `[state, questions, debias]` | LRU of 65 536 **questions**, key `sha256([state, question, debias])`. `decide()` serves cached questions first and scores only the rest |
| Auth | none | `OPENACT_TOKEN` → `X-OpenAct-Token` required on every POST (`hmac.compare_digest`). The server **refuses to start** without `OPENACT_TOKEN`, except with `--backend fake` (tests only). Reject any `Origin` header and any `Host` outside the allow-list (403, §16) |
| Endpoints | `/v1/systemone`, `/health`, `/v1/models`, `/shutdown` | adds `GET /v1/capabilities` (§5.2) |
| `/health` | status, model, device, requests, … | adds `backend`, `model_id`, `questions`, `rows_scored`, `cache_hits`, `ms_per_row` (EMA over the last 64 scoring calls) |
| Limits | `MAX_BODY` 4 MB | adds at most 256 questions per request (422 above) |
| While loading | `do_POST` blocks on `ENGINE.loaded.wait(timeout=300)`, returning 503 only after a failed load or 300 s. The client timeout is 180 s | **Returns 503 immediately** with `{"status": "loading"}` while `status == "loading"`, and `{"status": "error", "error": …}` after a failed load. The worker never blocks behind a cold load |
| 422 body | `{"error"}` | `{"error", "question": qid}`, so the client can isolate the bad question |

**Unchanged:**
- binding to 127.0.0.1, `ThreadingHTTPServer`, and background load with a warm-up row;
- the `--parent-pid` watchdog (`OpenProcess`/`WaitForSingleObject` on Windows);
- `CUDA_VISIBLE_DEVICES` from `--gpu`, SemIf `score`/`score_shared` in chunks of 32 rows, and the
  debias and tournament code.

**The per-question cache wraps today's `decide()` body**, renamed `_decide_uncached()`:

```python
def decide(self, request):
    state, questions, debias = self._validate(request)          # today's checks + 256-question limit
    answers, todo = {}, {}
    for qid, question in questions.items():
        key = _sha256_json([state, question, debias])
        hit = self.cache.get(key)                               # LRU, 65 536 entries, lock-protected
        if hit is not None:
            answers[qid] = hit
        else:
            todo[qid] = (key, question)
    if todo:
        fresh = self._decide_uncached(state, {q: v[1] for q, v in todo.items()}, debias)
        for qid, (key, _question) in todo.items():
            self.cache.put(key, fresh[qid])
            answers[qid] = fresh[qid]
    return {"model": MODEL_NAME, "answers": answers, "debias": debias,
            "cached_questions": len(questions) - len(todo), "usage": ..., "latency_ms": ...}
```

### 10.5 Blender-side client and installer (`server.py`, `prefs.py`)

**Decision worker.** `server.Worker`, a single daemon thread, replaces `ask()` on the executor
(§5.5).
- It pops the highest-priority chunk, sends it with the token header and a 180 s timeout, and
  posts `(chunk, answers)` back to the main thread.
- An `offline` result keeps the prototype's `_lost_connection()`.
- A 503 with `status: "loading"` puts the chunk back at the head of the queue. The worker then
  polls `GET /health` every 0.5 s until `ready`. P0 work can overtake and cancellation acts during
  that time, because nothing is blocked in flight.
- A 503 with `status: "error"` (failed load) is a **failure**. The chunk's cells go to `FAILED` with
  the server's message, and the status shows the error. It is never requeued forever.

**Launch token and sharing a server.** `start()` generates `secrets.token_hex(32)`, passes it as
`OPENACT_TOKEN`, and has `<runtime>/server-<port>.token` written: the server writes it
(`--token-file`) once it holds the port, so a Blender whose server lost a start race for the port
never overwrites the winner's file. The server binds its port exclusively (no `SO_REUSEADDR` on
Windows, where it would let a second server listen on the same port), and removes the file on a
clean exit if it still holds its token.
- That folder (`OPENACT_STATE_DIR` when set, §5.2) is per-user by default (`%LOCALAPPDATA%`). An
  explicitly chosen one passes on its own permissions, and the preferences warn when the folder
  that holds the token lies outside the user's profile (§16.1). On POSIX the file gets mode 0600.
- **Whose server it is.** `/health` reports `parent_pid`, the Blender that started the server
  (`--parent-pid`; a uv launcher's own pid differs from the server's). A server is this Blender's
  own when that is its pid; any other `openact-semif` is **another Blender's**.
- A second Blender that finds another Blender's healthy `openact-semif` on the port reads the
  token file and attaches, so two Blender sessions share **one** 8.4 GB model instead of loading
  two. It reads the file again at each `/health` probe and on a 403, and tries a token that
  changed since.
- **A lost start race.** A server that can't bind its port exits with its own code (4), which is
  no crash (§17.6): its Blender attaches at once to the server that holds the port, or, when
  nothing OpenAct can use answers there, says the port can't be used.
- If the token file is missing or wrong, the status says "A model server started elsewhere is
  running on port 18765 without a readable token: stop it or choose another port."
- A request that a server refuses (403) after this Blender's token changed, such as one sent to
  a server that crashed and taken by the one auto-start started in its place, is asked again with
  the new token.
- **Stop Model only detaches from another Blender's server:** this Blender stops using it and
  forgets its token, and it runs on for the Blender that started it. The Model box says "Shared:
  started by another Blender", and the Stop button reads *Detach* there. Detach lasts: health probes
  don't attach and auto-start waits, until *Start Model*, or until that server goes away. A server
  of this Blender's own is always shut down by *Stop Model*, and a start forgets the health of a
  server that has gone, so its own new server never looks shared.
- The watchdog still ties the server's life to the Blender that launched it. Attached sessions
  see *stopped* when it exits, and can start their own.

**Installer.** Pre-checks run before anything is downloaded:
- free disk space of at least 15 GB;
- an NVIDIA GPU found by `nvidia-smi --query-gpu=name,memory.total`. With no NVIDIA GPU, the
  install is **refused**: "OpenAct 1.0 needs an NVIDIA GPU". It no longer falls back to CPU torch,
  and v1 has **no CPU install path at all**. Only NVIDIA/CUDA is built (user decision); a CPU
  backend is future work;
- a warning below 10 GB of VRAM.

Then the same pinned steps as today:
1. uv 0.8.17 (the pinned, hash-checked copy; a `uv` on PATH is ignored);
2. a managed Python 3.12 venv (`UV_PYTHON_PREFERENCE=only-managed`);
3. torch 2.10.0 (cu128);
4. SemIf at commit `23cf1f39…`;
5. Qwen3.5-4B at revision `851bf6e8…` via `snapshot_download`.

It still honours `bpy.app.online_access`. It adds:
- **SHA-256 checks** of the downloaded uv archive (`UV_SHA256[asset]`) and of the SemIf archive.
  SemIf is downloaded at install time from the **pinned GitHub commit archive**, checked by
  SHA-256 (user decision); bundling may come later;
- installation from **hash-locked requirement files** (`--require-hashes`, §16.2);
- the install-log header "OpenAct runtime install";
- runtime-root resolution as in §12.2.

**Backends (preference `backend`).**
- `SEMIF_CUDA` is the default in v1.
- `FAKE` is hidden and selected with the environment variable `OPENACT_BACKEND=fake`, for tests
  and CI.
- **Future work, not in this plan:** a CPU backend, `LLAMACPP_CPU` (llama.cpp, GGUF Q4_K_M,
  about 3 GB), with its own installer profile. v1 builds only NVIDIA/CUDA (user decision).

The pipeline adapts to each backend through `/v1/capabilities`: `slots`, `batch_rows` and
`max_options`.

### 10.6 Pass engine (`engine.py`)

Kept from the prototype:
- the 0.1 s `tick()` timer and the 0.25 s debounce;
- the depsgraph / `load_post` / undo / redo handlers;
- `SAFETY_PASS` (2 s), which now only **recomputes fingerprints** and re-plans nothing unless a
  fingerprint changed (§5.3 step 4);
- pausing during playback and render jobs;
- skipping passes in files with no OpenAct nodes (`_has_nodes`, `_group_count`).

New:
- `iter_nodes()` covers the six node types plus the legacy aliases (§12.3), which are shown as
  frozen.
- The file watcher runs in `tick()` (§7.8).
- `run_pass()` follows §5.3, including the plan cache (`PLANS`), per-node dirty flags and the
  materialised-set choice (§8.1).
- The file trust state is worked out once per pass (§5.3); `load_post` records which file was
  opened (§16.1).
- `ensure_owned()` handles table objects (§8.5), and garbage collection of orphaned table objects
  runs once per pass.
- Linked trees are evaluated but never written.
- `STATUS[uid]` becomes a small record (icon, text, counts, ETA) that the node body and the sidebar
  both read (§17).

## 11. Minimum Blender Version

**Decision: Blender 4.5 LTS.** The manifest sets `blender_version_min = "4.5.0"`, with no maximum.
v1 is tested on 4.5.x, 5.0.x and 5.1.x. The prototype supported 4.2. OpenAct does not.

| Capability | 4.2 LTS | 4.5 LTS+ | What depends on it |
|---|---|---|---|
| Import CSV node | absent | present | The product story, "rules as CSV that Geometry Nodes reads"; artists' own Import CSV next to OpenAct; the parity probe (§18) |
| Import Text node | absent | present | Items lists from text files (evaluator handler) |
| Object and Collection input nodes | absent | present | Word sources for the Collection, Materials and Attributes list sources |
| `GeometrySet.from_evaluated_object` | absent | present | The agents hook (§5.6). The Attributes list source reads original data instead (§6.0) |
| Undefined custom group nodes evaluate their stored inner tree | no (zeros) | yes | G6: files render without the add-on, on a farm or on a colleague's machine |
| Integer Math node | absent (added in 4.3) | present | Index and cell math (§8.3, §9.3) |
| `default_input = 'INDEX'`, Index Switch `MATERIAL`, live interface edits on custom groups | not verified | **verified** (4.5.2, 5.1.1) | ID sockets, Pick Material, dynamic sockets |

**Why not keep 4.2.**
- **Support has ended.** 4.2 LTS was released in July 2024. Its two-year LTS window closed before
  this design's date, so no Blender support remains.
- **The core promise fails there.** Without the add-on, a saved OpenAct file on 4.2 evaluates to
  zeros, not answers.
- **Two code paths.** Keeping 4.2 would need a Python-only CSV path with no in-tree equivalent, no
  Attributes source, and a second lookup graph without Integer Math. The test matrix would double
  for a Blender with no support.

**What 4.2 users can do.** Keep using `jev_nodes` v0.1.0 (`dist/jev_nodes-0.1.0.zip`), which is
unchanged, or upgrade Blender. The OpenAct runtime folder and the old one can coexist (§12.2).

**Enforcement.**
- The extension platform refuses installation below 4.5.0 because of the manifest.
- A legacy install from a folder, as the tests do with `addon_utils.enable`, is guarded in
  `register()`. On older versions it raises `RuntimeError("OpenAct needs Blender 4.5 or newer")`
  before touching any data.
- `bl_info` is kept, with `"blender": (4, 5, 0)`, because `addon_utils.enable()` in the test
  scripts loads the package as a legacy add-on.

**Version-specific code.** Every node type and API used exists in 4.5. The one known difference is
`structure_type`, whose `FIELD` value is 5.0+ only (§6.0). `compat.py` therefore exists from PR-08
and holds that switch, keyed on `bpy.app.version`. Any further 5.x difference goes there too, and
the PR that adds it must test both versions.

## 12. Migration from jev_nodes v0.1.0

### 12.1 Rename map

| Item | `jev_nodes` v0.1.0 | OpenAct 1.0.0 |
|---|---|---|
| Extension id / Python package | `jev_nodes` | `openact` |
| Name / tagline | Jev Nodes / "Ask a local decision model from Geometry Nodes" | OpenAct / "Decision nodes for Geometry Nodes, run by a local model" |
| Node bl_idnames | `GeometryNodeJevAsk`, `GeometryNodeJevPick` | `GeometryNodeOpenActAsk`, `…Pick`, `…Match`, `…Classify`, `…Rules`, `…Dials` |
| Node labels | "Ask Jev", "Jev Pick" | "Ask", "Pick" (plus the four new nodes) |
| Inner group prefix | `.Jev ` (`INNER_PREFIX`) | `.OpenAct ` |
| Add menu / sidebar tab | *Shift A > Jev* / "Jev" | *Shift A > OpenAct* / "OpenAct" |
| Operators | `jev.install_runtime`, `jev.server_start`, `jev.server_stop`, `jev.reask`, `jev.open_log`, `jev.open_runtime_folder` | the same verbs under `openact.*`, plus new operators (§13.4) |
| Preferences class | `JevPreferences` | `OpenActPreferences`. Settings are keyed by add-on id, so on first registration OpenAct imports `runtime_dir`, `port`, `device`, `gpu`, `auto_start` and `debias` from the `jev_nodes` preferences when they are available (§12.2) |
| Ask property | `mode` (`YES_NO`/`AMOUNT`) | `answer_type` (same identifiers) |
| `entries` JSON | schema 1: a list | schema 2: an object (§14.2) |
| Server script / id | `runtime/jev_server.py` / `jev-nodes-semif` | `runtime/openact_server.py` / `openact-semif` |
| Runtime folder | `%LOCALAPPDATA%\JevNodes` | `%LOCALAPPDATA%\OpenAct`; the legacy folder is reused in place |
| Port | 18765 | 18765 (an old server there is detected, §5.2) |
| Thread names | `jev`, `jev-poll`, `jev-install` | `openact-*` |
| Docs and credits | README "Jev Nodes"; "Not affiliated with TypeSafe or Jev" | README "OpenAct"; the non-affiliation line is kept, and "Jev" appears only in credits and API-compatibility notes |

### 12.2 Reusing the installed runtime (13 GB)

`prefs.runtime_root()` resolves the folder in this order:
1. an explicit `runtime_dir`;
2. `<base>/OpenAct`, if `runtime_installed()`;
3. `<base>/JevNodes`, if `runtime_installed()` (**legacy, used in place**);
4. `<base>/OpenAct`, the target of a new install.

`<base>` is `%LOCALAPPDATA%` on Windows, and `$XDG_DATA_HOME` or `~/.local/share` on Linux (best
effort; `default_runtime_root()`). macOS is not a v1 target (NG9).

**The legacy folder is never moved or renamed.** The venv's `pyvenv.cfg` holds an absolute `home`,
and uv's console-script shims embed absolute paths, so moving would break the environment.

**What the installed legacy runtime actually looks like** (review, read-only, on this machine):
- `pyvenv.cfg` has `home = C:\Users\Nic\AppData\Roaming\uv\python\cpython-3.12.11-windows-x86_64-none`.
  The base interpreter lives in the user's **global uv folder**, not inside `JevNodes`.
- There is no `python\`, no `tools\uv.exe` and no `install.log`. There are `bootstrap_env.log` and
  `bootstrap_model.log` instead: the runtime was created by a bootstrap script, not by the
  add-on's installer.
- The packages and model revision match the pins: torch 2.10.0+cu128, semif_phase1 0.1.0,
  huggingface_hub, snapshot `851bf6e8…`.

Consequences:
- **Two checks, and neither blocks the main thread.**
  - **`runtime_installed()` stays a cheap file-system check.** It checks the venv's
    `Scripts\python.exe`, that the `home` directory named in `pyvenv.cfg` exists, and the model
    shards. It runs where the prototype calls it: `server.refresh()`, `tick()` and `start()`.
  - **The import probe runs on the executor thread**, never inside `runtime_installed()`.
    - It runs `python -c "import torch, semif_phase1"`, which takes seconds on a cold disk.
    - It runs once per session at registration and before the first start. Meanwhile the status
      reads "Checking the runtime…".
    - A start requested before the probe finishes, by auto-start or *Start Model*, is **queued, not
      blocking**. The operator returns at once, the main thread never waits, and the start runs
      when the probe posts its result. If the probe fails, the start is cancelled with the probe's
      message.
    - The result is cached on disk in `<runtime>/probe.json`, keyed by the mtimes of `pyvenv.cfg`
      and torch's `dist-info`, so later sessions skip the subprocess.
    - On failure it reports "The runtime's Python at <home> is missing or broken (was it removed
      with `uv python uninstall`?): reinstall the runtime".

  Checking only `Scripts\python.exe` and the model shards would call a broken runtime
  "installed".
- **Custom locations.** If `jev_nodes` preferences are available at first registration, OpenAct
  imports `runtime_dir`, `port`, `device`, `gpu`, `auto_start` and `debias`. Otherwise the sidebar
  offers **Locate Existing Runtime…** (`openact.locate_runtime`, PR-07), a folder picker that sets
  `runtime_dir` and runs the background probe.
  A user with a custom `runtime_dir` is therefore never pushed into a fresh 13 GB install.

Using the folder in place costs nothing. The server script ships inside the add-on (`SERVER_SCRIPT`
is resolved from the package folder), and the venv needs only torch, SemIf and huggingface_hub,
all of which are already installed. `openact_server.py` needs no new packages.

- **Sidebar and preferences** show "Using the Jev Nodes runtime at …\JevNodes (13 GB); Python from
  <home>". An *Open Runtime Folder* button is next to it. Diagnostics (§17.5) include the same two
  paths.
- **Deleting the old runtime.** Users can delete `JevNodes` after installing a fresh `OpenAct`
  runtime. The external uv-managed Python it used may be shared with other tools, so the README
  says to remove it only with `uv python uninstall` and only if nothing else uses it. OpenAct
  itself never deletes anything.
- **Both add-ons enabled.** Two add-ons would register two timers and could start two servers
  (17 GB of VRAM). OpenAct therefore:
  1. detects `jev_nodes` (or `bl_ext.<repo>.jev_nodes`) in `bpy.context.preferences.addons`;
  2. shows a persistent sidebar banner: "Disable Jev Nodes: OpenAct replaces it";
  3. does **not** register the legacy alias classes while it is enabled, because the bl_idnames
     would clash (§12.3).
- **An old server on the port.** If the old add-on started one, the status reports it (§5.2).

### 12.3 Old .blend files

**Legacy aliases.** `nodes.py` registers two hidden classes named `GeometryNodeJevAsk` and
`GeometryNodeJevPick`, so old files load them as *defined* nodes instead of `NodeUndefined`:
- `poll()` returns False, so they never appear in the Add menu.
- They are **frozen**. They keep evaluating their stored inner groups (`.Jev Ask <uid>`,
  `.Jev Pick <uid>`) and are never asked anything.
- Their status line reads "Old Jev node: upgrade in the OpenAct sidebar".
- They are skipped when the `jev_nodes` add-on is enabled (§12.2). In that case, and on Blender
  4.5+ without any add-on, the old nodes still evaluate from their saved inner groups.

**Upgrade.** `openact.upgrade_legacy` is offered by the sidebar banner "This file has 9 old Jev
nodes: Upgrade". After `load_post`, a one-line INFO report points to it. It is explicit, not
automatic, so opening a file never silently rewrites it. For each legacy node, inside a single
undo step:

1. **Create** the new node (`GeometryNodeOpenActAsk` or `…Pick`) in the same tree. Copy `name`,
   `label`, `location`, `width`, `parent` frame, `mute`, `hide` and `color`.
2. **Copy settings.**
   - Ask: `mode` → `answer_type`.
   - Pick: the prototype's precedence is kept, a set Collection winning over the Items text
     (`collection_items(coll) if coll is not None else parse_items(items)`):
     - `list_source = COLLECTION` if the Collection socket is set or linked; its Items text is
       ignored;
     - otherwise `TEXT`.
   - Pick nodes also get the hidden property `items_parse_v0 = True` (below).
   - `balance_order` is seeded from the `debias` preference. The imported value of `jev_nodes`'
     global `debias`, when available, keeps the prototype's behaviour (§10.3, §12.2).
   - `uid` is kept.
3. **Inputs, by socket name.** Copy unlinked default values and re-create links for Question and
   Context (Ask), and for Prompt, Items and Collection (Pick). The new optional inputs (Answers,
   Pick's Context, ID, Overrides) stay at their defaults, so the words are unchanged.
   - **Legacy item parsing.** v1 splits inline items on newlines, else `;`, else `,`, and separates
     `name: when`. The prototype split on newlines, else commas, with no descriptions.
   - While `items_parse_v0` is set, `lists.parse_inline(text, mode="v0")` parses the Items text
     with the prototype's rule, so an item string containing `;` or `": "` keeps its v0.1.0 item
     names, and its words key.
   - The flag is cleared the first time the Items text changes, and the node then uses v1 parsing.
4. **Outputs, by name.** Re-link Answer, Yes and Confidence (Ask), and Index, Item, Confidence and
   Instance (Pick). The names are identical by design.
5. **Answers.**
   - Convert `entries` from schema 1 to schema 2 (§14.4), and save them with
     `store.save(node, store)` (`live_gns=None`, no eviction, §13.3).
   - Rename the old inner group `.Jev Ask <uid>` → `.OpenAct Ask <uid>` and assign it to the new
     node.
   - Run `inner.sync_interface()` and `inner.build()`, which regenerate the group from the entries.
6. **Remove** the legacy node. Its old inner group is now used by the new node, so nothing is
   orphaned.

The operator reports a count, e.g. "Upgraded 9 nodes in 3 node groups; 0 problems". A problem, such
as an unexpected link type, leaves that node as a frozen legacy node and names it.

### 12.4 Answers survive the upgrade

- Converted entries are marked `legacy: true` and keep their v0.1.0 `gn` strings (the raw words).
- The conversion (§14.4) computes each entry's `words_key`, and each Pick row's `wkey`, with v1's
  own definition (§5.4), using the stored words and item names. The live plan of the upgraded node
  produces the same `words_key`, so the entry is reused under the validity rule. The rows' ask key
  is `jev_nodes 0.1.0`, and they are labelled "asked with older settings". **Reopening an upgraded
  file needs no model**, sends 0 requests (asserted by `migrate_blender.py`), and rendering is
  unaffected.
- A legacy Amount entry keeps its stored confidence (the prototype's top-minus-runner-up) until the
  words change or the artist clicks *Ask Again*. After that the 4.e formula applies.
- Pick's legacy entries already stored `probabilities`, so the new Probability and Selection
  fields work immediately after the upgrade.

### 12.5 Compatibility matrix

| File \ runtime | Blender 4.5+ with OpenAct | Blender 4.5+ without any add-on | Blender 4.2 |
|---|---|---|---|
| v0.1.0 file, not upgraded | Evaluates (frozen legacy nodes); upgrade offered | Evaluates stored answers (undefined nodes) | With `jev_nodes`: works as today; without it: zeros |
| v0.1.0 file, upgraded and saved | Full OpenAct | Evaluates stored answers | Not supported (OpenAct needs 4.5) |
| New OpenAct file | Full OpenAct | Evaluates stored answers, including table objects (§8.5) | Not supported |

**Downgrading** an OpenAct file to `jev_nodes` is not supported. v0.1.0 would see undefined nodes,
which Blender 4.5+ still evaluates from their stored answers.

**Test.** `tests/migrate_blender.py` (§18) runs these steps:
1. Build a v0.1.0 file with the prototype package, reusing `tests/e2e_blender.py build` from the
   untouched `jev_nodes` folder.
2. Reopen it with OpenAct and run `openact.upgrade_legacy`.
3. Assert, on 4.5 and 5.1:
   - evaluated outputs equal the v0.1.0 values;
   - 0 requests were sent;
   - inner groups are renamed and owned;
   - undo restores the legacy nodes.

## 13. API / Interface Changes

### 13.1 Node interfaces (artist-facing)

Full socket tables are in §6. This table is the before/after summary:

| Node | v0.1.0 inputs → outputs | OpenAct 1.0 inputs → outputs | Breaking? |
|---|---|---|---|
| Ask | Question, Context → Answer, Yes, Confidence | Question, Context, **Answers**, **Overrides** → Answer, Yes, Confidence | No: new optional inputs. Amount confidence switches to the normalised σ formula after a re-ask (§4.e) |
| Pick | Prompt, Items, Collection → Index, Item, Confidence, Instance | Prompt, **Context**, Items / Collection / **Object** (by `list_source`), **ID**, **Overrides** → Index, Item, Confidence, **Selection**, **Probability**, Instance / **Material** / payload | Mostly additive. Items and Collection are no longer shown together: the dropdown picks one, and the upgrade sets it (§12.3) |
| Match | — | Question, Context, Items/…, Answers, ID, Threshold, Overrides → Probability, Selection, Confidence | new |
| Classify | — | Question, Context, Items/…, Labels, ID, Overrides → Label, Confidence, payload | new |
| Rules | — | Question, Context, Factors, Outcomes, Overrides, *factor sockets*, Dither, Seed → Outcome / Probability + Yes / Amount, Confidence, payload | new |
| Dials | — | Prompt, Settings, Overrides → *one output per setting*, Confidence | new |

**Every node also has the hidden input `Key`** (String, word; compared by the match chain, never
sent to the model, §6.0). It is part of the 1.x interface.

**Stability promise from 1.0:**
- Socket **names** and **types** in these tables, `Key` included, do not change within 1.x,
  because links and migrations depend on them.
- Dynamic sockets (factors, settings, payload) take their names from the CSV (§6.0).
- New optional inputs may be appended.

### 13.2 HTTP API (Blender ↔ model server)

**Request: unchanged shape**, so hosted Jev or another compatible model can be plugged in later
(NG3):

```http
POST /v1/systemone HTTP/1.1
Host: 127.0.0.1:18765
Content-Type: application/json
X-OpenAct-Token: 5f0c…e1a9                      <- new, required

{"model": "semif-qwen3.5-4b", "state": "...", "questions": {"<qid>": {"type": "noul|choice|score",
 "instructions": "...", "criteria": ...}}, "options": {"debias": true}}
```

**Response: unchanged**, with two added fields (additive, so clients built for the old response
still work):

```json
{"model": "semif-qwen3.5-4b",
 "answers": {"<qid>": {"type": "choice", "choice": "2", "probabilities": {"2": 0.97, "3": 0.03}, "confidence": 0.94}},
 "usage": {"input_tokens": 1834, "output_tokens": 0}, "latency_ms": 212.4, "debias": true,
 "cached": false, "cached_questions": 5}
```

`cached_questions` is new. `cached` means every question was served from the cache.

**`GET /v1/capabilities`** (new):

```json
{"server": "openact-semif", "version": "1.0.0", "backend": "semif-torch",
 "model_id": "Qwen/Qwen3.5-4B@851bf6e8", "slots": 16, "max_options": 255, "max_levels": 16,
 "max_questions": 256, "batch_rows": 32, "debias": true, "question_cache": true}
```

| Code | When | Client reaction (`server.Worker`) |
|---|---|---|
| 200 | answered | fill `MEMORY`, post to the main thread |
| 403 | missing or wrong token, `Origin` present, bad `Host` | status *error* "A model server started elsewhere is running on port N without a readable token: stop it or choose another port" (§10.5) while `/health` reports that server's pid; the cells stay unanswered (not `FAILED`) and are asked again once it has gone. A refused *Stop Model* (`/shutdown`) is reported the same way. A request sent with a token that has changed since, or whose token another Blender's token file no longer holds, is asked again |
| 413 | body over 4 MB | should never happen, because chunks stay far below it; log, then split the chunk |
| 422 | invalid question; `question` names it | mark that cell `FAILED`; resend the rest of the chunk (§10.3) |
| 503 `loading` | model still loading (returned immediately, no blocking) | requeue at the head, poll `/health` until `ready` |
| 503 `error` | the model failed to load | cells → `FAILED` with the message; status *error*; no requeue |
| 500 | internal error | mark the chunk's cells `FAILED` with the message; *Ask Again* retries |
| connection refused | server down | `_lost_connection()` → status *stopped*; cells stay queued and auto-start applies |

`GET /health`, `GET /v1/models` and `POST /shutdown` keep their paths. `/shutdown` now needs the
token. `/health` gains counters (§10.4) and stays readable without a token, so it can serve as the
probe that distinguishes "ours", "old Jev Nodes" and "another program".

### 13.3 Internal Python interfaces (contracts between PRs)

These signatures are the seams the PR Plan builds on. A PR may add parameters with defaults, but
must not change these.

```python
# csvfmt.py (pure Python)
def read_csv(path: str) -> CsvDoc: ...                          # §7.2; never raises for content problems
def parse_answer_set(doc: CsvDoc, kind: str) -> AnswerSet: ...  # §7.3; kind: "yes_no" | "amount" | "one_of" | "items"
def parse_factors(doc: CsvDoc) -> list[Factor]: ...             # §7.4
def parse_settings(doc: CsvDoc) -> list[Setting]: ...           # §7.5
def parse_overrides(doc: CsvDoc, keys: list[str]) -> dict[str, Lock]: ...   # §7.6

# lists.py
def resolve_items(node, ctx) -> ItemList: ...                   # §6.0; raises Hint / Unresolvable
def parse_inline(text: str) -> list[tuple[str, str]]: ...       # newline, then ";", then "," (§6.0)
def is_csv_path(text: str) -> bool: ...
def csv_source(node, input_name: str, ctx) -> CsvSource | None: ...   # the linked "<input> CSV" wins, else the String path (§6.0)

# evaluator.py
def trace_csv_source(node, csv_socket, ctx) -> CsvSource: ...   # follows geometry links to an Import CSV node (§10.1)

# pipeline.py
def submit(job: Job, missing: list[Cell], priority: int, on_rows=None) -> None: ...
def cancel(uid: str, below: int) -> None: ...                   # drop queued chunks of older generations
def cached(cell_key: str) -> dict | None: ...                   # MEMORY lookup

# store.py
def load(node) -> NodeStore: ...                                # schema 2; converts schema 1 on read (§14.4)
def save(node, store: NodeStore, live_gns: frozenset[tuple[str, ...]] | None = None) -> None: ...
    # live_gns: the gn tuples of the node's live contexts, computed by the pass (find_uses + evaluator).
    # "Live" is not saved in entries, so eviction (16 warm/cold, 2 MB soft cap) needs it:
    # it never removes an entry whose gn is in live_gns, nor one with pinned: true (§8.1, §14.7).
    # None (the default) means "live set unknown": save() then evicts NOTHING.
    # Only engine.run_pass() passes a live set. Every other writer saves with None and leaves
    # eviction to the next pass: openact.reask, openact.unpin, Prepare for Playback,
    # openact.upgrade_legacy, and the schema 1 → 2 conversion on load.

# inner.py / tables.py
def sync_interface(node, wanted: list[SocketDef]) -> None: ...  # diff, never clear (§6.0)
def build(node, store: NodeStore) -> None: ...                  # full rebuild (§8.8)
def update_values(node, store: NodeStore) -> bool: ...          # False if the shape changed, so build() is needed
def ensure_table(node) -> bpy.types.Object: ...
def write_rows(node, flat_columns: dict[str, list]) -> None: ...

# geometry.py
def attribute_names(obj, exclude: set[str] = ()) -> list[str]: ...  # original data only (obj.data.attributes), natsorted, without "." names or exclude
```

### 13.4 Operators and preferences

| Operator | v0.1.0 | OpenAct |
|---|---|---|
| Install / start / stop model | `jev.install_runtime`, `jev.server_start`, `jev.server_stop` | `openact.install_runtime`, `openact.server_start`, `openact.server_stop` (same behaviour; install adds the uv SHA-256 check) |
| Ask again | `jev.reask` (forgets the node's answers, clears **all** `FAILED`) | `openact.reask`: forgets only this node's cell keys and failures; locked and code rows are kept. Saves with `live_gns=None`, so no eviction; the next pass evicts (§13.3) |
| Logs and folders | `jev.open_log`, `jev.open_runtime_folder` | `openact.open_log`, `openact.open_runtime_folder` |
| New | — | `openact.write_overrides` (§7.6), `openact.new_csv_from_template` (§7.10), `openact.open_file`, `openact.upgrade_legacy` (§12.3), `openact.copy_diagnostics` (§17), `openact.allow_file` ("Allow this file": adds this .blend's path to `trusted_files`, §16.1), `openact.trust_folder` ("Trust this folder": adds the folder, and so its subfolders, to `trusted_folders`), `openact.prepare_playback` (answers and pins animated words; modal only on the driver fallback, §5.5), `openact.unpin` (one entry, or all, §5.5), `openact.show_question` (shows the exact request of a table row without calling the model, §17.2), `openact.locate_runtime` (picks an existing runtime folder and runs the probe, §12.2; PR-07). Conditional: `openact.refresh_import_csv`, which ships only if follow-up F1 shows the trick works on 4.5 and 5.1 (§7.1) |

| Preference | v0.1.0 | OpenAct |
|---|---|---|
| `runtime_dir`, `port` (18765), `gpu`, `auto_start` | yes | kept, same meaning. Imported from the `jev_nodes` preferences on first registration when available (§12.2) |
| `debias` | global on/off | **"Balance option order for new nodes"**, default on. It only seeds each new node's own `balance_order` toggle; existing nodes keep theirs (§10.3). Imported from `jev_nodes` when available |
| `device` | `auto`/`cuda`/`cpu` | `auto`/`cuda` only, and `auto` **means CUDA only**: no run-time CPU fallback. Without CUDA the server reports an error with a driver hint (§5.2). There is no `cpu` option in v1 (NG9) |
| `backend` | — | `SEMIF_CUDA` (v1); `FAKE` via the environment only. `LLAMACPP_CPU` is future work, not in this plan |
| `unsure_below` | — | Float, default **0.30**, adjustable here, and tuned later from beta feedback (user decision; 4.e, §21 F3) |
| `log_requests` | — | Bool, default off. Writes JSONL of requests and responses to `<runtime>/logs/` (§17) |
| `trusted_folders` | — | Folders (and their subfolders) whose .blend files OpenAct trusts for file reads outside their own folder tree (§16.1). Edited in the preferences; *Trust this folder* adds to it |
| `trusted_files` | — | Individual .blend paths trusted the same way. *Allow this file* adds to it, which is the safer default for a Downloads folder |

### 13.5 Extension manifest (`openact/blender_manifest.toml`)

```toml
schema_version = "1.0.0"
id = "openact"                                    # was "jev_nodes"
version = "1.0.0"
name = "OpenAct"
tagline = "Decision nodes for Geometry Nodes, run by a local model"      # 55 chars
maintainer = "Nic"
type = "add-on"
blender_version_min = "4.5.0"                     # was "4.2.0" (§11)
license = ["SPDX:GPL-3.0-or-later"]
tags = ["Geometry Nodes", "Node"]                 # from Blender's allowed list, unchanged

[permissions]
network = "Download the model runtime once, then talk to it on localhost"   # 61 chars
files = "Store the model runtime; read text and CSV files; write CSVs"      # 60 chars
clipboard = "Copy a diagnostics report for bug reports"                   # 41 chars

[build]
paths_exclude_pattern = ["__pycache__/", "*.pyc", "/.git/", "*.zip"]
```

- All three permission strings stay under the 64-character limit and have no trailing period.
- `clipboard` covers `openact.copy_diagnostics` (§17.5) and the *Copy* button of *Show Question*
  (§17.2). Platform review compares declared permissions with behaviour, and `extension validate`
  checks the manifest (PR-23, PR-26).
- `files` names what OpenAct actually touches:
  - it **reads text and CSV files**: CSVs, and the Import Text word source;
  - it **writes CSVs**, but only from *Write Overrides CSV* and *New CSV from Template*, into files
    the user chose. Otherwise it writes only the runtime folder.
- The build command is unchanged: `blender --command extension build`, run from `openact/`,
  producing `dist/openact-1.0.0.zip`.

## 14. Data Model

### 14.1 What is stored where

```mermaid
classDiagram
  class OpenActNode {
    uid : str
    entries : JSON schema 2
    answer_type : enum
    list_source : enum
  }
  class Entry {
    words_key : str
    asks : list of str
    pinned : bool
    gn : list of str
    pv : int
    model : str
    t : float
    shape : Shape
    rows : columnar rows
    single : dict
    csv : path to hashes
    legacy : bool
  }
  class InnerGroup {
    name : .OpenAct Node uid
    matchChain
    indexSwitchTables
  }
  class TableObject {
    name : .OpenAct Table uid
    vertices : one per flat row
    attributes : oa_ prefixed columns
  }
  class CsvFile {
    path
    textColumns
    numericColumns
  }
  OpenActNode "1" --> "0..16" Entry : entries JSON
  OpenActNode "1" --> "1" InnerGroup : node_tree
  InnerGroup "1" --> "0..1" TableObject : Object Info
  Entry "1" --> "0..*" CsvFile : text and num hashes
```

- **The node's `entries` JSON is the single source of truth.** The inner group and the table object
  are *derived*: `inner.build(node, store.load(node))` can regenerate both at any time. Ownership
  repair, upgrades and "shape changed" rebuilds all rely on this.
- **CSV files are inputs**, never stored in the .blend. Entries keep only their hashes, which is
  enough to tell whether the baked text and payload are current (§7.8).

### 14.2 Node properties and the `entries` JSON (schema 2)

| RNA property | Type | Nodes | Notes |
|---|---|---|---|
| `uid` | `StringProperty(options={'HIDDEN'})` | all | 12 hex characters (`uuid4().hex[:12]`); cleared by `copy()` (§3.4) |
| `entries` | `StringProperty(options={'HIDDEN'})` | all | schema-2 JSON (below). A dynamic-length string, as in the prototype |
| `answer_type` | `EnumProperty` | Ask, Match, Rules | the update callback runs `sync_interface()` and `schedule(immediate=True)` |
| `balance_order` | `BoolProperty` (*Balance Option Order*) | all | per-node order balancing (§10.3). `init()` seeds it from the `debias` preference. It is part of the ask key, so toggling it sends 0 requests and discards nothing (KD13, KD26) |
| `list_source` | `EnumProperty` | Pick, Match, Classify | same |

**Example: the Game of Life entry.** Rows are stored **columnar**: compact, and ready for
`foreach_set`.

```json
{"schema": 2,
 "entries": [{
   "words_key": "5e1c0b…", "asks": ["b77a…"], "pv": 1, "t": 1790000123.4, "legacy": false,
   "pinned": false,
   "gn": ["What happens to this cell in the next generation?", "A cellular automaton…",
          "//life_factors.csv", "//life_outcomes.csv", "", ""],
   "model": "semif-torch Qwen/Qwen3.5-4B@851bf6e8 debias",
   "kind": "rules", "answer_type": "ONE_OF",
   "shape": {"rows": 18,
             "columns": {"outcome": "INT", "c": "FLOAT"},
             "payload": [["alive", "INT"]],
             "outcomes": ["Birth", "Stays empty", "Survives", "Dies"],
             "factors": [
               {"name": "state", "kind": "flag", "levels": ["dead", "alive"], "values": [0, 1]},
               {"name": "neighbours", "kind": "category", "levels": ["0", "1", "…", "8"],
                "values": [0, 1, 2, 3, 4, 5, 6, 7, 8], "contiguous": true}]},
   "rows": {"name": ["dead|0", "dead|1", "dead|2", "dead|3", "…"],
            "wkey": ["a1f3…", "0c9d…", "77e0…", "5b21…", "…"],
            "ask": [0, 0, 0, 0, "…"],
            "src": "mmmmmmmmmmmmmmmmmm",
            "v": {"outcome": [1, 1, 1, 0, "…"], "c": [0.99, 0.98, 0.97, 0.96, "…"]},
            "payload": {"alive": [0, 0, 0, 1, "…"]},
            "top": [[[1, 0.995], [0, 0.005]], "…"]},
   "single": {},
   "csv": {"//life_factors.csv": ["a41f…", "0000…"], "//life_outcomes.csv": ["9c2e…", "77b1…"]}}]}
```

| Field | Meaning |
|---|---|
| `schema` | always 2. Schema 1 (the v0.1.0 list) is converted on read (§14.4) |
| `words_key` | what the artist said (§5.4). The entry is reused while it matches, whatever its ask keys. On a mismatch for the same `gn`, a new entry is built and rows with an unchanged `wkey` carry over |
| `pinned`, `frames` | `pinned: true` for entries answered by *Prepare for Playback* (§5.5), with `frames: [[start, end], …]`, the frame ranges where those words occur. Pinned entries are saved, never evicted, materialised before warm ones (§8.1), and count towards the cap. They expire only on the next Prepare for Playback run or through `openact.unpin` |
| `asks` | the ask keys (templates, debias, backend) that produced this entry's rows; `rows.ask` indexes into it. A key other than the current one only labels rows "asked with older settings" |
| `gn` | raw word strings, in `spec.words()` order, compared by the match chain (§8.2) |
| `pv`, `model` | template version and model provenance. Shown in the sidebar, never part of a key |
| `t` | last-use time, for LRU eviction beyond 16 entries |
| `kind`, `answer_type` | spec name and answer type when stored. On mismatch after a dropdown change, the entry is dropped |
| `shape` | **everything `inner.build()` needs, so no CSV is required to rebuild:** row count; column types; the payload `[header, type]` list (index *k* → `oa_pl_<k>`); option or outcome names. For Rules, each factor's `name`, `kind`, `levels`, plus: category/flag `values` with `contiguous` (non-contiguous maps are laid out at build time, §9.2); amount `edges` and `centres`. For Dials, `settings`: name, type, and this entry's `min`/`max` |
| `rows.wkey`, `rows.ask` | each row's words key (§5.4), and the index into `asks` of the settings it was asked with |
| `rows.name` | row identity: item name, `level\|level` for Rules, or setting name. It drives stale-while-refilling (§5.4) and Overrides matching (§7.6) |
| `rows.src` | one character per row: `m` model, `c` code, `l` lock, `s` stale, `p` pending, `e` error |
| `rows.v` | answer columns: `p`, `amount`, `label`, `outcome`, `prob`, `c`, plus `ok` for Rules Yes/No and Amount |
| `rows.payload` | baked numeric CSV columns, denormalised per row (§8.1) |
| `rows.top` | optional top 3 `(answer row, probability)` per row, for the sidebar. Stored only when the table has at most 1024 rows |
| `single` | entry-level values of Ask, Pick and Dials, e.g. `{"Answer": 0.93, "Yes": true, "Confidence": 0.86}`. Pick also stores `"Material": "<picked material name>"`, resolved to the ID at build time (§8.6) |
| `csv` | per referenced file, `[text_hash, num_hash]` when the entry was written |
| `legacy` | true for entries converted from v0.1.0 (§12.4) |

### 14.3 Derived data in the .blend

| Data-block | Content | Rebuilt from |
|---|---|---|
| `.OpenAct <Node> <uid>` (GeometryNodeTree, hidden) | Interface (fixed and dynamic sockets); match chain over the materialised entries; entry-level Index Switches (base, count, Ask and Pick single values, band centres); Object Info + Sample Index for per-element columns and for Dials' one-row-per-entry values; Rules factor/interpolation/dither graph; Map Range for Dials | `entries` **alone**: `shape` holds everything `build()` needs (§14.2), so no CSV file is required |
| `.OpenAct Table <uid>` (Object + Mesh) | One loose vertex per flat row of the materialised entries, then the category value-map rows (§9.2); POINT attributes `oa_*` | `entries` (the materialised entries' `rows` and `shape`) |
| (no collection) | Table objects are linked to no collection. Their only user is the Object Info default (§8.5) | — |

Rebuilding derived data must never change evaluated outputs. The lookup test compares outputs
before and after `inner.build()` on the same entries, once with the CSV files present and once
with every referenced CSV file **deleted** (§18).

### 14.4 Schema 1 → 2 conversion (`store.load`, used by the upgrade)

Schema 1 is the v0.1.0 list `[{"key", "gn", "a", "t"}]`. `store.load` converts it as follows:

| v0.1.0 `a` | Schema-2 entry |
|---|---|
| Ask `{"mode", "value", "confidence", "levels"?}` | `kind "ask"`, `answer_type = mode`, `single = {"Answer": value, "Yes": value ≥ 0.5, "Confidence": confidence}`, no rows |
| Pick `{"index", "item", "confidence", "probabilities", "items", "collection"}` | `kind "pick"`, `single = {"Index", "Item", "Confidence"}`. Rows: `name = items`, `v.prob = probabilities`, `src = "m" × N` |

In both cases `gn` is re-ordered to v1's fixed-length `spec.words()` tuple (§8.2):
- Ask `(Question, Context)` becomes `(Question, Context, "", "", "")`. The `""` are the new empty
  Answers, Overrides and Key words.
- Pick `(Prompt, Items)` becomes `(Prompt, "", Items, "", "")`. The `""` are the new empty Context,
  Overrides and Key words.
  - For a collection Pick, the collection applied because it was set, not because the Items text
    was empty: the prototype ignored any Items text then (§12.3). The conversion therefore writes
    `""` at the Items position. Under `COLLECTION` the absent Items socket also contributes `""`,
    so the converted tuple matches the live words exactly.
  - Item names are re-derived with legacy parsing (`items_parse_v0`, §12.3), so the words key
    matches too.

Every converted entry gets:
- `legacy = true`, `pv = 0` and `model = "jev_nodes 0.1.0"`;
- `asks = ["jev_nodes 0.1.0"]`, so its rows are labelled "asked with older settings";
- a `words_key`, and for Pick a `wkey` per item row, computed with **v1's own definition** (§5.4).

The words-key inputs come from the stored raw words, the answer type and the stored item names
(for a collection Pick, the prototype stored the child names in `items`). The upgraded node's live
plan therefore produces the same `words_key`, and the entry is reused with 0 requests (§12.4). The
prototype's old `key` is dropped.

Conversion is pure Python. `tests/store_test.py` covers it with the exact JSON the prototype
writes (`set_entries`), and asserts that the converted `words_key` equals the one `plan()`
computes for the same words.

### 14.5 Session-only state (never saved)

| State | Owner | Reset on |
|---|---|---|
| `MEMORY` (cell answers, LRU 100 000), `PENDING`, `FAILED` | `pipeline.py` | Blender restart. `load_post` clears `FAILED`, as today; *Ask Again* clears a node's keys |
| `STATUS` (per-node icon, text, counts, ETA) | `engine.py` | `load_post` |
| `PLANS[(uid, ctx key)] → (fingerprint, job)`, the plan cache (§5.3) | `engine.py` | a changed fingerprint; a context that disappears, unless the pass planned it (an evaluator read also reaches contexts `find_uses()` doesn't list, through a group node in a tree no modifier uses); `load_post` |
| `LAST_SHOWN[(uid, ctx key)] → words_key` of the entry that context showed last (stale-while-refilling, carry-over, and `k_default` for single-context nodes, §5.3–§5.4, §8.2). **Seeded on the first pass after load** from the stored entry whose `gn` equals that context's words, so rows re-asked after a CSV edit made while Blender was closed show stale values, not fallbacks | `engine.py` | `load_post` (then re-seeded); a context that disappears |
| `TRUST`: which file was opened; `PASS`: whether the open file is trusted, worked out once per pass (§16.1) | `trust.py` | `load_post` and enabling OpenAct (`TRUST`); every pass (`PASS`) |
| `WATCH` (file stamps and `CsvDoc` cache), the evaluator memo and the file identities (`csvfmt.IDENTITIES`) | `engine.py`, `evaluator.py`, `csvfmt.py` | `load_post` (watch); every pass (memos) |
| Worker queue, server `STATE` | `server.py` | add-on unregister (`shutdown_owned`) |

### 14.6 Files on disk

```
%LOCALAPPDATA%\OpenAct\                      (or the legacy JevNodes folder, §12.2)
  venv\                                      Python 3.12: torch 2.10.0, semif-phase1, huggingface_hub (~4.4 GB)
  python\                                    uv-managed CPython 3.12
  hf\hub\models--Qwen--Qwen3.5-4B\snapshots\851bf6e8…\   model weights (~8.7 GB)
  tools\uv.exe                               uv 0.8.17 (SHA-256 checked)
  server.log                                 server output, overwritten on each start
  install.log                                appended on each install
  server-18765.token                         launch token of the running server (one per port)
  logs\requests-YYYYMMDD.jsonl               only with log_requests on (§17)
<project folder next to the .blend>
  *.csv                                      the artist's Factors, Outcomes, Labels, Items, Settings and Overrides files
```

This is the layout **OpenAct's installer** creates, with its managed Python kept inside the runtime
folder. The **legacy `JevNodes` runtime on this machine** differs (§12.2):
- it has `venv\` and `hf\`, plus `bootstrap_env.log` and `bootstrap_model.log`;
- it has no `python\`, no `tools\` and no `install.log`;
- its venv's `home` points at the user's global uv Python
  (`%APPDATA%\uv\python\cpython-3.12.11-…`).

The start-up probe checks that external interpreter (§12.2).

### 14.7 Storage estimates and caps

| Case | `entries` JSON | Inner group | Table object |
|---|---|---|---|
| Ask (one context) | < 1 KB | about 10 nodes | — |
| Dials, 5 settings (one context) | < 1 KB | about 35–50 nodes | 1 vertex × (5 values + 2 × amount settings + `conf`), a few hundred bytes |
| Game of Life, 18 rows, 1 payload | ≈ 1.5 KB | about 40 nodes | 18 vertices × 3 columns, < 1 KB |
| Pick, 255 items (Probability field linked) | ≈ 5 KB | about 20 nodes | 255 vertices, about 4 KB |
| Rules, 4096 rows, 2 payload columns, one context | ≈ 250 KB (no `top` above 1024 rows) | about 60 nodes | 4096 vertices, about 150 KB |
| Worst case: **16 warm or cold entries, plus live (at most 16) and pinned (at most 12) entries**, × 4096 rows; materialised up to the cap of 16, pinned included (usually 4 or fewer, §8.1) | warm/cold bounded by the 2 MB soft cap; live and pinned kept above it with a size note; each entry ≤ 4 MB (below). Realistic worst case about 3 MB (12 pins of a 4096-cell table) | at most about 300 nodes for the derived worst shapes (§8.4) | typically 4 × 4096 = 16 384 vertices, about 0.6 MB; 65 536 and about 2.4 MB at the cap |

- **Soft cap.** If a node's `entries` exceeds 2 MB, the least recently used **warm and cold**
  entries are evicted until it fits. The status notes "2 old contexts dropped to keep the file
  small".
  - **Precedence:** live and pinned entries are **never** evicted by the soft cap (§8.1).
    - "Pinned" is saved in each entry (`pinned: true`).
    - "Live" is not: the pass passes it to `store.save(node, store, live_gns)` (§13.3) as the `gn`
      tuples of its current contexts.
    - Every other writer (*Ask Again*, *Unpin*, Prepare for Playback, the legacy upgrade) saves
      with `live_gns=None`, which evicts nothing; the next pass evicts with the real live set.
  - If live plus pinned entries alone exceed 2 MB, they are **kept**. The status notes the size,
    e.g. "entries use 3.1 MB: live and pinned contexts are kept; pin fewer word sets to shrink the
    file".
  - They are still bounded: at most 12 pins, each entry within the 4 MB hard cap below. A realistic
    worst case is 12 pins of a 4096-cell Rules table, about 3 MB.
  - Prepare for Playback shows the projected size before pinning ("pinning 12 entries adds about
    3 MB to the file").
  - Pins are an explicit artist action for playback, so silently evicting them would break the
    playback they exist for. Refusing them by size would make Prepare for Playback unusable on
    large Rules tables.
- **Hard cap.** A single entry above 4 MB is an error. Only an absurd table could reach this,
  because the 4096-row limit bounds it first.
- **Undo memory.** JSON changes land in undo steps with the node-tree data-block. During a fill the
  JSON is written at most every 2 s and once at completion, while the table object carries the
  live values (`write_rows()` at most every 0.5 s). `update_values()` writes only changed
  entry-level sockets.

## 15. Alternatives Considered

### 15.1 One generic "Decide" node with modes, or one node per verb (chosen)

| | One generic node (scope × answer-type dropdowns) | Six verb nodes (chosen) |
|---|---|---|
| Discoverability | One Shift A entry, and the artist must learn a 3 × 3 mode matrix | The menu entry says what it does: "Match: each item, yes/no or amount" |
| Sockets | The union of all sockets, most hidden per mode. Tooltips must describe every mode | Each node shows only its own sockets. Dynamic sockets are limited to Rules factors, Dials settings and payload |
| Code | One large spec with branches per mode | Six small specs over shared helpers (`specs/base.py`, `lists.py`, `csvfmt.py`) |
| Migration | Ask and Pick become modes of something new | Ask and Pick keep their identity and output names (§12) |
| Future | Every new idea grows the matrix | New verbs are new nodes (the agents design may add its own) |

**Decision.** One node per verb. Consistency (4.g) comes from shared code and formats, not from
cramming everything into a single node.

### 15.2 Live model calls per element, or decision tables (chosen)

**Live calls** would ask the model about every element (point, instance, face) from its own
values.
- **Impossible during evaluation.** Geometry Nodes has no Python hook while it evaluates, so the
  answers would have to be precomputed anyway, one per element.
- **Too slow.** At the measured ~24 cells/s, 1M elements take about 11.6 hours. Every change to the
  geometry would re-trigger it, and playback would be impossible.
- **Poor inputs.** The model would see raw numbers, where it is weakest (4.a), and results would
  drift between frames.

**Tables** are bounded by *meaning* (at most 4096 cells), not by element count:
- an O(1) lookup per element;
- saved in the .blend and deterministic;
- safe on a render farm.

The cost is quantisation of continuous inputs, which interpolation (Yes/No, Amount) and dithering
(One-of) remove visually (§9.6, §9.7). **Decision:** tables.

### 15.3 Import CSV at evaluation time, or baked payload (chosen)

The full comparison is in §7.1. An inner Import CSV cannot see edits, because Blender caches by
path (verified). The workaround, content-addressed copies, makes the .blend depend on a
machine-local cache, and a render farm would get **silently empty** payload. Baking keeps the file
self-contained, and the payload uses the same lookup as the answers. **Decision:** bake.

### 15.4 The model inside Blender's Python, or a separate runtime server (chosen)

| | In-process (torch inside Blender's Python) | Separate server (chosen, as in the prototype) |
|---|---|---|
| Install | pip into Blender's bundled Python 3.11: ~3 GB of CUDA wheels in the Blender folder, often under Program Files (admin), lost on every Blender update | Own venv with managed Python 3.12, per user, survives Blender updates; downloaded once (13 GB) |
| Crashes | A CUDA or torch fault kills Blender and the unsaved work | A server fault becomes a status message; Blender survives |
| Responsiveness | Model load (12–40 s) and scoring compete with the UI for the GIL | Fully asynchronous over localhost HTTP |
| VRAM | Held for the whole Blender session | *Stop Model* frees 8.4 GB, and several Blender sessions share one server (§10.5) |
| Backends | Tied to Blender's Python ABI | Anything that speaks `/v1/systemone`: a llama.cpp CPU backend is possible future work, and so is hosted Jev |
| Latency | No HTTP | About 1–2 ms per chunk on localhost, negligible next to ~69 ms per decision |
| Extension policy | Bundling torch is incompatible with a pure-Python extension | The add-on stays pure Python |

**Decision.** Keep the runtime split (decision 7).

### 15.5 Overrides edited in the sidebar, or an Overrides CSV (chosen)

- **Sidebar editing** (click a cell, pick an answer) is quick, but it is classical add-on UI. It
  contradicts 4.h, keeps hidden state in the node, cannot be diffed or reviewed, and cannot vary
  per object.
- **The Overrides CSV** is a file the artist owns: diffable and spreadsheet-friendly, and it can
  vary per object through Group Input. The cost is an export step (*Write Overrides CSV*).

**Decision.** CSV. The sidebar stays inspection only (NG8).

### 15.6 Where tables live: JSON plus derived groups (chosen), or alternatives

| Option | Why not chosen |
|---|---|
| `CollectionProperty` of PropertyGroups (one per row) | 4096 rows × several columns of RNA items; slow writes, heavy undo steps, verbose code |
| A Text data-block per node | Extra IDs that artists can edit by hand, which competes with the CSV override channel; weaker copy semantics |
| Always a table object, even for Ask | A single code path, but extra objects for every node and Object Info indirection for single values. The inner group is no longer self-contained |
| Always inline Index Switch | Hundreds to thousands of sockets for large tables, and every value update edits the graph. Measured: setting 780 item defaults costs 77–81 ms in a 14-node tree and about 680 ms in a 214-node tree (§8.4) |
| Inline up to 64 rows, table object above (the previous draft) | The path flips as a side effect of row count or history, two paths must be kept in parity, and inline value updates still edit the graph |

**Decision.** The `entries` JSON is the source of truth (§14). **Every per-element column lives in
the table object. Index Switch holds only entry-level single values** (§8.4). This is the middle
option: no objects for single-value nodes (Ask, Dials), and a constant graph with `foreach_set`
updates for everything per element.

**Where the lookup data lives: measured** (the review's numbers, 4.5.2 and 5.1.1):

| Option | Build | Value update | Per-element lookup, 1M points |
|---|---|---|---|
| Inline Index Switch columns (12 × 65 items) | 68 ms in a 14-node tree; about 900 ms in a 214-node tree | 77–81 ms / about 680 ms (sets 780 sockets) | within about 1 ms of baseline |
| Table object for per-element columns (**chosen**) | 1 Object Info plus 2 nodes per lookup: a constant graph | one `foreach_set` per column; a few ms for 16 × 4096 × 6 values | within about 1 ms of baseline (65 or 4096 rows) |

**Where the rows are stored: the source of truth.**

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| **JSON string on the node** (chosen) | Holds text (row names, `wkey`s, option names) and numbers in one versioned schema; copies with the node; undo-safe; human-inspectable; easy to migrate | A large table's JSON is up to about 250 KB per entry, and rewriting it after every chunk costs undo memory | Chosen, with writes **throttled**: during a fill the JSON is written at most every 2 s and once at completion. The table object carries the live values in between (§14.7) |
| Nested ID properties on the node (`node["oa"] = {"gn": [...], "rows": {"name": [...], "p": [...]}}`) | String lists, float lists and nested dicts all round-trip on custom nodes (review, round 2, on 4.5–5.1). A throttled write could replace one list instead of re-serialising the whole blob | Needs `bpy`, so `store.load`/`save` and the schema 1 → 2 conversion could no longer run as pure-Python tests under CPython (§18). Migration becomes an in-place mutation of a live IDProperty tree instead of one pure `dict → dict` transform. The write saving is small: `json.dumps` of a 250 KB entry takes about 1–3 ms, at most every 2 s | Rejected on those grounds, not for lack of string support. Revisit if profiling ever shows JSON writes mattering |
| Table-object attributes as the truth for large tables | No large JSON rewrites | Numbers only, so the text still needs JSON. The truth would split between two data-blocks, and a lost object would lose answers. Rebuild-from-entries (§14.1) would no longer hold | Rejected; the table object stays derived |

### 15.7 Declaring Rules factors

| Option | Why not chosen |
|---|---|
| Header conventions in the Outcomes CSV (e.g. `if:state` columns, band names in headers) | Cramped, with no room for level descriptions (the key accuracy lever, 4.c). Mixes model-facing text with payload |
| A list editor in the node's sidebar | Classical add-on UI (4.h); the data is not in a file |
| **A separate Factors CSV** (chosen) | One row per level with its `when`. Read by Python only, so mixed blanks are fine. Validated line by line (§7.4) |

## 16. Security & Privacy

### 16.1 Threat model

**Trust boundaries:**
- the Blender process, which runs the user's add-ons;
- the local model server, a separate process bound to 127.0.0.1;
- files that .blend documents point at;
- the one-time installer's downloads.

**Attackers considered:**
- a web page open in the user's browser;
- another local process or user;
- a malicious .blend or CSV received from someone else;
- a compromised download.

| Asset | Threat | Mitigation |
|---|---|---|
| Model server (GPU time, 8.4 GB VRAM) | A web page POSTs to `127.0.0.1:18765` (CSRF, e.g. `/shutdown` as a denial of service) | Every POST needs `X-OpenAct-Token`. Browsers cannot add a custom header cross-origin without a CORS preflight, which the server never answers. Requests carrying an `Origin` header get 403 |
| Model server | DNS rebinding: a page on `evil.example` re-resolves to 127.0.0.1 and reads responses | `Host` allow-list (`127.0.0.1:<port>`, `localhost:<port>`), otherwise 403; plus the token |
| Model server | Another local process or user sends requests | Bound to 127.0.0.1 only (unchanged). The token lives in the environment (never in argv, which other users can read) and in a file in the runtime folder, mode 0600 on POSIX. That folder is per-user by default (`%LOCALAPPDATA%`). An explicitly chosen `runtime_dir` passes on its own permissions, so the preferences warn when it lies outside the user's profile. The token is regenerated on every start and compared with `hmac.compare_digest`. There is no unauthenticated mode: the server refuses to start without a token, except with the test-only fake backend (§10.4) |
| Server lifetime | An orphaned server keeps the GPU busy after Blender dies | The `--parent-pid` watchdog exits within 2 s (unchanged). Attached sessions never extend its life (§10.5) |
| Local files | A malicious .blend points word sockets (a CSV path, or an Import Text path) at private files. OpenAct would read them and **store their contents** in the file (`gn`, Compare `B_STR` defaults, row names), which is exfiltration once the file is saved and sent back | **Trust rule** (below). In an untrusted file every read is confined to the .blend's own folder tree, and blocked text is never read, hashed or stored. OpenAct reads CSVs only from paths ending in `.csv`, at most 1 MB. The sidebar's *Files* panel lists every file each node reads |
| Local files | Writes in unexpected places | OpenAct writes only the runtime folder, plus files the user picks in *Write Overrides CSV* or *New CSV from Template* (file dialog, never overwrites a template, atomic replace). It never writes automatically. In an untrusted file the trust rule (below) confines writes as it does reads: "Blocked writing <path>: outside this file's folder". *Write Overrides CSV* refuses, and leaves as it was, a file that a node in the open .blend reads as another input (such as Items), has errors, isn't an Overrides CSV for the node (no key column, or no `lock` column) or is read-only. Its temporary file is created new, so a planted link is never followed (§7.6) |
| Answers | Prompt injection inside CSV text ("ignore the rules, answer Birth") | The model can only pick among the fixed options; no text is generated (NG4). The worst case is a wrong answer, which confidence flags and a lock overrides (4.e, 4.f) |
| Blender responsiveness | Huge inputs exhaust memory or time | Limits: 1 MB per CSV, 4096 rows or cells, 8192 scored rows per context, 256 questions per request, a 4 MB body. Stored JSON per node: 16 warm or cold entries within a 2 MB soft cap; at most 16 live word sets (the over-cap error, §8.1) and 12 pins, kept above 2 MB with a size note; a 4 MB hard cap per entry |
| Installed runtime | Compromised or changed downloads | See 16.2 |

**Trust rule for opened files** (`trust.is_trusted()`, worked out once per pass; `load_post`
records which file was opened).
- **A file is trusted** when any of these holds:
  1. it was created in this session (it is unsaved). A file opened while OpenAct was disabled is
     judged by 2 and 3 once OpenAct is enabled again;
  2. the file's path is in `trusted_files`, which **Allow this file** (`openact.allow_file`) adds
     to. This is per file, so allowing one download does not trust the rest of a Downloads folder;
  3. the file's folder, or a parent folder, is in `trusted_folders`, which **Trust this folder**
     (`openact.trust_folder`) adds to. The button says it covers every subfolder and every future
     file there.
- **Blender's *Auto Run Python Scripts* does not make a file trusted.** Blender can refuse it for
  one file (*Trusted Source* unchecked in File > Open). An add-on can't see that refusal for a file
  without scripts, so following the preference would let a file that Blender refused read outside
  its folder.
- **A trust change plans every node again** (either button, or an edit of the trusted lists): the
  trust state is part of each plan's fingerprint (§5.3).
- **In an untrusted file:**
  - Every file read, for CSVs and for the Import Text word source, is confined to the .blend's own
    folder tree (§7.7). Files shipped next to a downloaded demo therefore still work, while nothing
    outside can be read into the file. A path outside the tree is blocked without being touched,
    so a UNC path contacts no server. A path inside is also compared by `realpath`, so a symlink
    cannot escape.
  - Stored answers keep evaluating as usual; Geometry Nodes needs no Python.
  - A blocked read makes the node show the blocked path, and the sidebar shows the banner with
    *Allow this file* and *Trust this folder*.
- **Residual risk: appended and linked node groups.** They run under the *host* file's trust, as
  with Blender's own auto-run. A trusted file that appends a crafted group from elsewhere therefore
  lets that group's word paths read outside the host's folder. The sidebar's *Files* panel still
  lists every file each node reads, and linked groups are read-only (§7.7).
  - Nothing is read, hashed or stored for a blocked path.
- **Test.** `tests/pipeline_blender.py` opens an untrusted file with an absolute Import Text path
  and a CSV path outside its folder. It asserts that neither file is read, that nothing from them
  appears in `entries` or the inner group, and that a CSV next to the file *is* read. After
  `openact.allow_file`, both are read, and so are a Rules node's CSVs in the next pass.

### 16.2 Supply chain (installer)

1. Every download needs an explicit click, a confirmation showing size and path, and
   `bpy.app.online_access`. This is unchanged.
2. **uv 0.8.17** is fetched from GitHub releases into `<runtime>/tools/`, and checked against a
   pinned SHA-256 per platform asset. That check is new. A `uv` on PATH is **ignored**: the
   prototype preferred `shutil.which("uv")`, and this machine has two, so "pinned" would not hold.
3. **torch 2.10.0** comes from the official PyTorch index (`cu128` or `cpu`) over HTTPS.
4. **SemIf** is pinned to commit `23cf1f39…`. **Qwen3.5-4B** is pinned to revision `851bf6e8…`,
   whose Hugging Face snapshots are content-addressed by commit.
5. **Hash-locked environment (new).** Today SemIf's transitive dependencies are not locked.
   `runtime/requirements-cu128.lock` (the only lock; v1 has no CPU install path) is generated with
   `uv pip compile --generate-hashes` and installed with `--require-hashes`.
   - **SemIf supply** (user decision): it is downloaded at install time from the pinned GitHub
     commit archive and checked against a recorded SHA-256, because GitHub archive bytes are not
     guaranteed stable.
   - A mismatch stops the install and tells the user to update OpenAct.
   - Bundling a wheel may come later.
6. **At runtime the server runs offline** (`HF_HUB_OFFLINE=1`). After installation OpenAct makes no
   network calls except to localhost.

### 16.3 Privacy

- **Local only.** Prompts, CSV text, answers and geometry never leave the machine. There is no
  telemetry, analytics or update check.
- **What the .blend carries.** A .blend stores its words: raw strings in `gn`, row names from CSV
  files, and the answers. Sharing a file shares them, like any node tree; the README says so.
  Cell keys are one-way hashes.
- **Logs stay local and minimal by default.**
  - `server.log` holds status lines, timings and error messages.
  - `log_requests` (off by default) writes full requests and responses to `<runtime>/logs/`. Its
    tooltip says exactly that.
  - `openact.copy_diagnostics` copies versions, GPU, status, counters and the last errors
    (as node type, severity, file name and line, §17.5). It never includes CSV contents or prompts.

### 16.4 Licences and naming

- **Licences.**
  - The add-on is GPL-3.0-or-later.
  - SemIf (MIT) is installed into the runtime at install time, not bundled.
  - Qwen3.5-4B (Apache-2.0) is downloaded by the user from Hugging Face, not redistributed.
  - PyTorch is BSD-3-Clause.

  Credits appear in the README and in the preferences, as in the prototype's `draw_preferences`.
- **Naming.** "Jev" is TypeSafe's trademark. The product is OpenAct. "Jev-compatible API" appears
  only as a factual statement of request compatibility, and "Not affiliated with TypeSafe or Jev"
  is kept.

## 17. Observability

OpenAct is a local desktop tool with no telemetry (§16.3). "Observability" therefore means:
- what the artist sees on the node and in the sidebar;
- what the logs and `/health` show;
- what a bug report can include.

### 17.1 Node status record

`engine.STATUS[uid]` holds one record per node. Both `draw_buttons` (one line) and the sidebar
(full detail) read it:

```python
@dataclass
class NodeStatus:
    state: str          # "hint" | "paused" | "starting" | "waiting" | "filling" | "complete" | "warning" | "error"
    text: str           # one line, e.g. "Filling 12/50 · ~2 s"
    rows: int; answered: int; pending: int; stale: int
    unsure: int; locked: int; code: int; errors: int
    eta_s: float | None # remaining scored rows × /health ms_per_row
    contexts: int       # live contexts; above 16 the node shows the over-cap error (§8.1)
    problems: list      # Problem records from csvfmt, the evaluator and the pipeline (§7.9)
```

| State | Icon | Example line |
|---|---|---|
| hint | `INFO` | "Type a question" / "Set a Factors CSV (sidebar: New CSV from Template)" |
| paused | `PAUSE` | "Model server is stopped (OpenAct tab)" (auto-start off) |
| starting | `SORTTIME` | "Starting the model…" / "Installing the model… 3/5 PyTorch" |
| waiting | `SORTTIME` | "Waiting for another OpenAct node…" |
| filling | `SORTTIME` | "Filling 12/50 · ~2 s" (stale rows show old values meanwhile, §5.4) |
| complete | `CHECKMARK` | "18 cells · 0 unsure · 2 locked" |
| warning | `ERROR` (triangle) | "18 cells · 1 warning: column 'scale' has an empty cell (line 7)" |
| error | `CANCEL` | "life_outcomes.csv:4 'if' names unknown level 'alvie' of factor 'state'" |

The glyphs in §6.0's examples (✓ ⏳ ⓘ ⚠) stand for these Blender icons. The prototype's `ERROR`
icon for hard errors becomes `CANCEL`, so that warnings and errors look different.

### 17.2 Sidebar (OpenAct tab, inspection only)

- **Model box.**
  - Status, backend, device name, `model_id` and load time.
  - Queue: chunks per priority class and cells in flight.
  - Throughput: `ms_per_row`, and the cache hit rate `cache_hits / questions`.
  - Buttons: Install / Start / Stop, and *Open Log*.
- **Active node.**
  - The status record, and one summary line per context: the object and modifier label, as in the
    prototype's `answers_for`.
  - **A context with no table to show** says "Asking..." while the pass asks one of its cells (sent,
    or sent once the model that is starting is ready), else "Not asked: see the status above": the
    model is stopped, not installed or in error, the context waits or disagrees (§6.0), or its cell
    failed. Once a row has an answer (a model or stale one), a context being filled shows its table so
    far (§5.4), and its line sums it up. A context that waits or disagrees shows its stored entry,
    also a table so far, while that entry's words key is its plan's; a Dials context shows its last
    good table, each row matched to the plan's by wkey.
  - **Table view**, 50 rows per page. Paging uses a `WindowManager` IntProperty, which is UI state,
    not file data. Each row shows its name, answer, a confidence bar and a source icon:
    - `QUESTION`: unsure;
    - `LOCKED`: lock;
    - `CHECKBOX_HLT`: decided by code;
    - `CANCEL`: in error: a Rules cell no outcome applies to, or a cell whose ask failed (Show
      Question says so, and Ask Again asks it again);
    - greyed out: stale or pending.
  - Pick also shows its top 5 items with probabilities, as today.
  - **Show Question** (`openact.show_question`) on the selected row rebuilds that row's request
    from the cached plan: the `state`, `instructions` and options exactly as sent, or as they
    would be sent, plus the ask key it was answered with. It calls no model.
    - The text opens in a popup with a *Copy* button, which is what the manifest's `clipboard`
      permission covers.
    - Wording is the main quality lever (R1), and this lets artists see how their `when` texts
      reach the model without turning on `log_requests`.
- **Files panel** (§7.10) and the **problems** list (up to 50 per file), each with *Open File*.
- **Banners.** Shown for:
  - model not installed;
  - "Checking the runtime…" and runtime-probe failures, with *Locate Existing Runtime…* (§12.2);
  - an untrusted file whose reads were blocked, with *Allow this file* / *Trust this folder*
    (§16.1);
  - `jev_nodes` still enabled (§12.2);
  - old Jev nodes in the file (§12.3);
  - a port conflict;
  - the Import CSV cache note (§7.1);
  - more than 16 distinct live word sets on a node (§8.1).

### 17.3 Server metrics (`GET /health`)

`status`, `backend`, `model_id`, `device`, `load_seconds`, `requests`, `questions`,
`rows_scored`, `cache_hits`, and `ms_per_row` (EMA over the last 64 scoring calls).
- The Blender side polls every 5 s while nodes exist, as `tick()` does today.
- `ms_per_row` feeds every ETA.
- If `ms_per_row` exceeds 3× its 3090 default (21 ms) for a minute while cells are being asked,
  the Model box shows "Running slower than expected: is the GPU busy (rendering, another app)?".
  While nothing is asked the clock stops: an idle server's `ms_per_row` is its last fill's.

### 17.4 Logs

| Log | Content | Retention |
|---|---|---|
| `<runtime>/server.log` | Start line (backend, device, load time); one line per request, e.g. `12:03:44 req q=18 rows=36 812ms cached=0/18 debias`; tracebacks on 500 | Overwritten on each start |
| `<runtime>/install.log` | Every installer command and its output | Appended |
| Blender console | `[OpenAct]` prefix; tracebacks for internal errors only (per-node errors go to the status record) | Session |
| `<runtime>/logs/requests-YYYYMMDD.jsonl` | Only when `log_requests` is on (it takes effect at once). One JSON line per request with `t`, `status` (the HTTP code), `state`, `questions`, `answers`, `ms`, `rows`. On POSIX the folder is 0700 and the file 0600 | Local only; the user deletes it |

### 17.5 Diagnostics for bug reports

`openact.copy_diagnostics` copies a plain-text report to the clipboard. It contains:
- OpenAct, Blender and OS versions;
- GPU and device;
- backend, `model_id` and revision;
- the runtime root, marked when it is the legacy folder;
- server status and `/health` counters;
- a node count per type, with rows per table, materialised entries and inner-group node counts;
- queue state;
- the last 10 problems, each as node type, severity, file name and line (never its text).

It contains **no prompts and no CSV contents** (§16.3): problems keep no text, the runtime root
shows the home folder as `~`, and the server's message has every quoted segment as `'…'` and
every path cut to its file name.

### 17.6 Alerts (local)

There is no remote alerting. The "alerts" are user-facing messages that name the fix:
- the status line and the sidebar banners above;
- **crash-loop protection:** if the server this Blender started exits unexpectedly (an exit code
  other than 0, which a `/shutdown` gives) 3 times within 10 minutes, auto-start pauses: the status
  becomes an error, so the Model box and the nodes show "The model server keeps stopping: see the
  log"; *Start Model* re-enables it;
- **a filling table that stalls** (no answered chunk for 60 s while the server is `ready`) shows
  "No answers for 60 s: see the log" as its node's warning line, and its queued chunks stay queued.

### 17.7 Test and benchmark output

Tests keep the prototype's line protocol: `PASS …` / `FAIL …` / `RESULT OK|FAILED: …`, then
`os._exit` with the matching code. They add:
- `TIMING <table> rows=<n> <s>s <rows/s>` lines, compared against §3.2;
- `BENCH <path> <elements> <ms>` lines from `tests/lookup_blender.py --bench`;
- `WARN …` lines for a timing above its budget but within twice it (§18.2), which the runner counts
  in its summary.

A later run can diff these lines to spot performance regressions.

## 18. Testing Strategy & Demo Scenes

### 18.1 Matrix and harness

- **Blender.** 4.5.x, 5.0.x and 5.1.x, at `C:\Program Files\Blender Foundation\Blender
  4.5|5.0|5.1\blender.exe`, always run as `-b --factory-startup --python <suite> -- <args>`, as
  today.
- **Plain CPython ≥ 3.11.** The pure suites (`csvfmt`, `store`, Rules maths) import no `bpy`, so
  they also run under plain CPython in seconds. They must pass under both **3.11** and **3.13**:
  Blender 4.5 and 5.0 bundle Python 3.11, Blender 5.1 bundles 3.13, and this machine has 3.12 on
  PATH.
- **Runner.** `tests/run_all.ps1 [-Model] [-Versions 4.5,5.0,5.1] [-Threads 4]` runs the pure suites once and
  every Blender suite on every installed version, `tests/lookup_blender.py` with `--bench`, so the
  §8.4 budgets gate every merge under the timing rule of §18.2. The real-model e2e runs only with
  `-Model`, because it needs the runtime and the GPU. Every suite runs at BelowNormal priority with
  `OMP_NUM_THREADS`/`MKL_NUM_THREADS` at `-Threads`, and every Blender with `-t <Threads>`, so a run
  leaves most of a shared machine free. The runner ends with a summary table, which counts each
  run's warnings, and a non-zero exit code if anything failed.
- **Fake backend.** Suites that need a server set `OPENACT_BACKEND=fake`. The add-on then starts
  `openact_server.py --backend fake [--fake-script tests/fixtures/<name>.json]`. That path imports
  no torch, so `OPENACT_SERVER_PYTHON` may point at any Python 3.10 or newer, Blender's bundled one
  included. The pipeline therefore runs on machines without the 13 GB runtime.
- **Fixtures** in `tests/fixtures/`:
  - `life_*.csv`, `highlife_outcomes.csv`, `seeds_outcomes.csv`, `daynight_outcomes.csv`;
  - `biome_*.csv`, `material_families.csv`, `weather_dials.csv`;
  - one broken CSV per §7.9 message;
  - `jev_v010_entries.json`, the exact JSON the prototype's `set_entries` writes;
  - fake-answer scripts.

### 18.2 Suites

| Suite | Runs in | Model | What it checks |
|---|---|---|---|
| `tests/csv_test.py` | CPython + Blender | none | `csvfmt` dialect (BOM, CRLF and CR, `;` and tab delimiters, quoting); INT/FLOAT/text classification; limits; every §7.9 message by fixture; the `parse_answer_set` / `parse_factors` / `parse_settings` / `parse_overrides` parsers; the `if` grammar; `lists.parse_inline` (newline, then `;`, then `,`); `text_hash` unchanged by numeric edits and `num_hash` unchanged by text edits |
| **`tests/store_test.py`** | CPython + Blender | none | **Schema 1 → 2 conversion** of the prototype's exact JSON (Ask Yes/No and Amount, Pick from text and from a collection); `gn` re-ordering (`""` inserted for Answers and Context); `legacy = true`, `pv = 0`; the converted `words_key` and `wkey`s equal what `plan()` computes for the same words (§14.4); a changed `PROMPT_VERSION` or `debias` changes the ask key but no words key; save/load round trip; 2 MB soft-cap eviction removes only warm and cold entries and **keeps live and pinned entries**, even when live plus pinned alone exceed 2 MB (with the size note) |
| `tests/rules_math_test.py` | CPython + Blender | none | The 4.e confidence formulas: uniform answers give exactly 0 for L = 2…16, point masses give 1, near-uniform answers stay below 0.30, and the documented examples (0.65; the bimodal split gives 0). `itertools.product` order equals `ravel`; allowed sets and code-decided cells; the reference interpolation (corners, weights, renormalisation); dithering probabilities, analytic versus 100 000 samples within ±1 % |
| **`tests/evaluator_blender.py`** (extended) | Blender | none | Every existing case, plus **Menu Switch**: a value typed on the socket and a value coming from a modifier's Group Input menu, for String, Int and Float, covering both storage forms (§10.1). Also Special Characters, String Length, OpenAct single-value outputs used as words, field outputs rejected with the breadcrumb message, and the per-pass memo returning identical values. All are checked against Geometry Nodes ground truth with Compare nodes, the existing technique. **`trace_csv_source()`** (added by PR-13): a direct link; a Path exposed on the modifier; Reroute, nested-group and Geometry Menu Switch pass-through; the Import CSV Path/Delimiter socket identifiers pinned on 4.5, 5.0 and 5.1; both error messages |
| `tests/csv_probe_blender.py` | Blender | none | **Import CSV parity** per fixture: point count equals `csvfmt` data rows; kept attributes equal `csvfmt` numeric columns with the same INT/FLOAT types and values. It records Blender's behaviour for blank lines, quoted line breaks, BOM and `;` in `tests/fixtures/import_csv_behaviour.json`. It re-asserts the path-cache fact (an edited file still returns old rows), so a future Blender change is noticed |
| `tests/interface_blender.py` | Blender | none | Guards the verified facts: interface edits reach custom group nodes and links survive; `default_input='INDEX'` with `hide_value` gives `[0, 1, 2, 3]`; Index Switch `MATERIAL` works; `structure_type` as measured per version: `SINGLE` on String words everywhere, `FIELD` on ID and factors only on 5.0+, `AUTO` on 4.5; `FIELD` on a String socket is rejected on 5.x, and setting it is never attempted. A string field linked into a word stays `is_valid = True`, pinning the fact that the evaluator's `Unresolvable` is the guard; `copy()` plus `ensure_owned()` gives every copy its own inner group and table object, including when the whole tree is duplicated; an unlinked table object survives save/reload, *Purge Unused Data*, F12 and `blender -b -f` (§8.5) |
| `tests/lookup_blender.py` | Blender | none | Synthetic tables written through `store` and `inner`, with no server. Checks: invalid IDs give fallbacks; 3 contexts through the match chain; payload columns; the Rules cell index for **every** combination against Python, including non-contiguous category maps; interpolation against the `rules_math` reference (max error 1e−5); dither fraction within ±2 %. **A rebuild leaves outputs unchanged**: evaluate, run `inner.build()` on the same entries, evaluate again and compare every attribute, **once with the CSV files deleted**. A values-only update keeps the node count. **Materialised set:** 10 successive Question edits on an 18-row Rules node keep at most 4 materialised entries, with at most one rebuild per edit; switching back to a stored but not materialised prompt sends 0 requests. **Shape agreement:** two contexts with different Outcomes CSVs that share a payload set work; conflicting Factors CSVs give the error and keep the last good table; per-entry Dials `min`/`max` and Pick materials. **Game of Life in a Simulation Zone**: 32 frames against a Python B3/S23 reference, both on a toroidal grid (§6.5), plus a Repeat Zone variant. **Joined key (§8.2):** most-recent-first link order; an absent position joined as a constant `""` matches its stored entry (a Collection-source Pick, an upgraded collection Pick); a word containing `"\x1f"` uses the per-word fallback. **Dials (§8.4):** a Dials output feeds a single-value socket (Grid *Vertices X*), and, after a new prompt's one rebuild, the answers arriving update the values through `write_rows()` with no further rebuild. **Pins:** when live plus pinned exceeds 16, the pins whose word tuples start latest in the frame range go inactive (pin order, §5.5). **No-match fallback (§8.2):** with one live context, unmatched words show that context's own last answers; with two contexts, A's unmatched words show neutral values, never B's answers; a cleared Question (a hint) shows neutral values. `--bench` enforces the §3.2 and §8.4 budgets: per-element lookup, `inner.build()`, `update_values()`, `tables.write_rows()`, under the timing rule below |
| `tests/pipeline_blender.py` | Blender + fake server | fake | Token required (403 without); `Origin`/`Host` rejection; chunks of at most 32 questions / 64 rows; a P0 job overtakes a P2 fill; cancellation drops queued chunks; per-question cache (`cached_questions`); 422 isolation; stale-while-refilling; locks send 0 requests; a numeric-only CSV edit sends 0 requests and updates the payload; a text edit re-asks only the affected cells; a missing file keeps the table; playback and render pauses. **Entry validity (§5.4):** a template bump sends 0 requests; toggling a node's *Balance Option Order* sends 0 requests until the words change, and later requests carry that node's `options.debias`; after a text edit, only rows whose `wkey` changed are asked. **Plan cache:** a TIMING line shows ≤ 5 ms per unchanged node and ≤ 50 ms per idle pass with a maximum-size Rules node used by 16 contexts. **Attributes:** a Pick (Attributes source) whose Item feeds a Store Named Attribute settles in one pass with one request. **Trust (§16.1):** an untrusted file with an absolute Import Text or CSV path outside its folder reads and stores nothing, while a CSV next to it is read |
| `tests/support_blender.py` | Blender + fake server | fake | `blender --command extension validate` with the `clipboard` permission; the templates parse with their sockets' schemas, and *New CSV from Template* never overwrites (`name 2.csv`) and fills only an empty, unlinked input; the diagnostics text has versions and counters but no prompt or CSV text; the JSONL request log only while `log_requests` is on; `unsure_below` moves the unsure counts, icons and exported column; **two Blender processes share one server through the token file** (the suite starts the second Blender, whose Stop Model only detaches), a second server can't take the port, a lost start race attaches at once and is no crash, and a wrong token file gives the §10.5 message; diagnostics redact quoted names and paths; time-scaled: 3 fake-server exits within 10 minutes pause auto-start, a stalling `--fake-script` raises the stalled-fill alert, and a faked `ms_per_row` the slow-server note, only while cells are asked |
| **`tests/migrate_blender.py`** | Blender 4.5 and 5.1 | none | §12.5: a v0.1.0 file built with the untouched `jev_nodes` → `openact.upgrade_legacy` → identical outputs, 0 requests, inner groups renamed and owned, links kept, and undo restores the old nodes. It covers the prototype's **collection Pick** (the e2e scene's `lamp` node), which must **match its converted entry through the joined key**, with the absent Items position joined as `""` (§8.2), a Pick whose Items text contains `;` and `": "` (legacy parsing keeps its names), and a Pick with both Items text and a Collection (collection wins, its answers kept). It also imports custom `jev_nodes` preferences (§12.2) |
| `tests/e2e_blender.py` (extended) | Blender + real model | real | The existing `build` / `noaddon` / `reopen` phases, now covering all six nodes. **Game of Life acceptance:** Conway, HighLife, Seeds and Day & Night, each 18/18 exact with 0 unsure. Biome: mean confidence ≥ 0.55 with all 7 outcomes used. Sanity checks for Match, Classify and Dials; Pick's Probability field; `noaddon` with table objects and entry-level Index Switches; `reopen` sends 0 requests. TIMING lines are compared with §3.2 |

**Timing checks.** They print always and fail only above 2× the §3.2 target, so a busy GPU does
not make the suite flaky. The `--bench` budgets (§8.4) follow the same rule, because the gate runs
while other work uses the machine: every measurement is printed against its budget, one above its
budget prints a `WARN` line, and only one above twice its budget fails. Accuracy checks (the Game
of Life tables and the biome thresholds) are hard failures.

### 18.3 Gates

| When | Must pass |
|---|---|
| Every PR | Pure suites, evaluator, interface and lookup suites on Blender 4.5 and 5.1, the lookup suite with `--bench` (the §8.4 budgets, under the timing rule of §18.2), as `tests/run_all.ps1` runs them. None needs a model; together they take a few minutes |
| PRs that touch `server.py`, `pipeline.py` or `openact_server.py` | Also `tests/pipeline_blender.py` (fake backend) |
| PRs that touch specs, templates or scoring | Also `tests/e2e_blender.py` with the real model, on the RTX 3090 machine |
| PRs that touch migration or storage | Also `tests/store_test.py` and `tests/migrate_blender.py` |
| Release | Every suite on 4.5, 5.0 and 5.1; e2e on 4.5 and 5.1; demo scenes rebuilt and rendered **without** the add-on |

### 18.4 Demo scenes (for the promo video)

`demos/build_demos.py` builds every demo headlessly:
- `blender -b --factory-startup --python demos/build_demos.py -- <out folder> [--model]`. The
  script enables the add-on, then calls `openact.engine.allow_background_passes(True)` (§7.8), so
  passes run headlessly and the tables fill.
- It writes `demos/<name>.blend` next to that demo's CSV files, which are copied from
  `tests/fixtures/` and `openact/templates/`.
- With `--model` it fills every table with the real model and checks the acceptance criteria below.
  The saved files then carry their answers, so they open, play and render **without the model and
  without the add-on** on Blender 4.5+ (G6).

| Demo | Nodes | What it shows | Acceptance |
|---|---|---|---|
| `life.blend`: **Life from a CSV** (hero) | Rules (flag `state`, category `neighbours`) inside a Simulation Zone over a 64 × 64 grid. Neighbours are counted with 8 Sample Index offsets on a **toroidal** grid (Integer Math `FLOORED_MODULO` on row and column, §6.5). The payload output `alive` feeds Store Named Attribute | A modifier dropdown (Menu Switch → Outcomes path, §7.7) switches Conway / HighLife / Seeds / Day & Night. The sidebar shows the exact 18-cell table. The CSV opens side by side, and editing Birth to "three or six" turns Conway into HighLife. One cell is locked in `life_overrides.csv` | All four tables 18/18 exact, 0 unsure; playback at full speed with 0 requests; matches the Python reference for 32 frames |
| `biome.blend`: **Biome** | Rules One-of (temperature, moisture, altitude; 27 cells) → payload `instance` and `scale` → Instance on Points; a second Rules in Amount mode for snow cover | Plants follow habitat descriptions, with Dither 0.3 blending the zones. Snow grows smoothly with altitude and cold, with no stair steps. An A/B of bare names against descriptions shows the confidence difference in the sidebar | Mean confidence ≥ 0.55 with all 7 plants used. Snow is monotonic along a test transect |
| `kitchen.blend`: **Clutter** | Match "Would you find it in a kitchen?" (Yes/No) and "How heavy is it?" (Amount) on a 40-prop collection; Classify into rooms (Labels) | Per-instance Selection with Delete Geometry; weight → Scale; room labels → anchors | Spot checks: e.g. pan = yes, sofa = no |
| `materials.blend`: **Material families** | Classify materials (ID = Material Index), with `material_families.csv` payload `roughness` and `metallic` given by **plugging an Import CSV node into `Labels CSV`** (the hand-off, §6.0); Match "metal?" per face | The model decides the family and the CSV supplies the numbers (4.a). Editing a roughness value updates instantly with 0 requests | The payload edit sends 0 requests (pipeline counter) |
| `weather.blend`: **Weather dials** | Dials with `weather_dials.csv`: Rain, Wind speed (0–30), Time of day, Street lights, Fog | One prompt on the modifier drives the whole environment. Switching "a stormy night in a fishing village" ↔ "a calm summer morning" re-answers five dials in one request | ≤ 0.6 s per new prompt; switching back is instant (cached) |
| `ask_pick.blend`: **Ask and Pick** | The prototype's e2e scene, **rebuilt by the script with OpenAct nodes**, not produced by the legacy upgrade: Ask (night switch) on two objects with different modifier prompts; Pick from a props collection | One tree gives different answers per object. Pick's Probability field drives a "fitness" heat map over Collection Info children | The prototype's e2e values: A night > 0.9, B night < 0.1, armchair for A, bookshelf for B |

**Video beats (about 90 s).**
1. Title card: "Words in, Geometry Nodes data out".
2. Life from a CSV. **While playback is stopped**, switch the dropdown (the table refills in about
   1 s), press play, then stop and edit Birth in the CSV and play again. For a continuous shot
   through all four rule sets, keyframe the dropdown and run *Prepare for Playback* first (§5.5).
   Without it, unanswered words would show the last good rules, never a dead grid.
3. The biome grows from habitat descriptions.
4. Weather dials from one sentence.
5. The clutter and material demos as quick cuts.
6. Close: "Runs on your GPU. Answers are saved in the file. Renders without the model."

Every shot is taken from the saved demo files, so the video is reproducible.

## 19. Risks

Severity is the impact if the risk happens. Likelihood is judged for v1 as designed.

| # | Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|---|
| R1 | **Accuracy on artists' own rules and prompts** is worse than on the measured, well-framed rule sets: wrong cells in real scenes | High | Medium | Templates enforce "outcome + when" (4.c); `if` restrictions (4.d); confidence flags and the sidebar table (4.e); locks (4.f); good starter templates; README "Writing good whens"; e2e acceptance on four rule sets and the biome (§18) |
| R2 | **Hardware barrier.** 8.4 GB of VRAM on an NVIDIA GPU excludes many artists | High | High | Saved answers make files work everywhere; only *authoring* needs the model. Sessions share one server (§10.5). The installer refuses up front, before any download, on machines without an NVIDIA GPU and warns below 10 GB of VRAM (§10.5). The support matrix is explicit (§3.3). A CPU backend (llama.cpp, GGUF ~3 GB) is future work, outside this plan (user decision) |
| R3 | **Install fragility.** 13 GB download; antivirus, proxies, full disks, drivers too old for cu128; a legacy runtime that depends on an external uv Python | Medium | Medium | Resumable, idempotent steps (as today); hash-locked environment and pinned uv, ignoring PATH (§16.2); a new **free-space pre-check** (`shutil.disk_usage` ≥ 15 GB) and a **GPU/VRAM pre-check** (§10.5); *Open Log*; the legacy runtime is reused in place, with a background probe of its interpreter (§12.2). A driver too old for cu128, which passes the `nvidia-smi` check, is caught at load time: `auto` means CUDA only, so the server reports an error with a driver hint instead of silently loading on the CPU (§5.2) |
| R4 | **Blender API drift** in 5.x: custom group nodes, interface API, Index Switch items, evaluation of undefined nodes | Medium | Medium | Interface, lookup, evaluator and `noaddon` suites on 4.5, 5.0 and 5.1 at every release (§18.3); `compat.py` switches (§11) |
| R5 | **Table-object lifecycle bugs.** A table is lost on append or link, garbage collection removes a live table, or Blender stops evaluating objects linked to no collection, giving silently wrong outputs | High | Low | `entries` JSON is the source of truth, and derived data can be rebuilt from it without the CSV files (§14.1, §14.3). Garbage collection never removes a table whose uid is live. Copy, append, delete, save/reload, *Purge Unused Data* and rendering are tested on 4.5, 5.0 and 5.1 (§18). The excluded-collection fallback is a `tables.py`-only change (§8.5) |
| R6 | **Static evaluator mismatch.** Python computes a word differently from Geometry Nodes, so the match chain finds no entry | Medium | Low | Every handler is checked against Geometry Nodes ground truth (§18), and unsupported nodes raise explicitly. On shared trees (two or more live contexts) a mismatch shows neutral fallback outputs, never another object's answer. On a single-context node it shows that context's own last answer (`k_default`, §8.2), which masks the mismatch but is never another object's data. The ground-truth tests are the real guard there. **Residual case without the add-on (G6):** no pass runs, so `k_default` keeps its saved value. If a tree saved with one live context is later used by a second object with different words in a Blender without OpenAct, that object shows the first object's last answers. The next pass with OpenAct installed resets it (two live contexts → 0) |
| R7 | **Performance on the main thread.** Graph edits get slower as a tree grows (measured, §8.4), and planning repeats on every pass | Medium | Medium | Per-element data lives in the table object, so value updates are `foreach_set` and never graph edits (§8.4). Materialised entries are bounded (§8.1). Rebuilds happen only on shape changes, at most once per pass. Plan cache and fingerprints (§5.3). Batched `foreach_set` locations. `--bench` budgets for `build()`, `update_values()`, `write_rows()` and pass time are merge gates (§3.2, §8.7) |
| R8 | **Perceived hangs.** Large tables take minutes and users think Blender froze | Low | Medium | ETA, partial tables written after every chunk, priority classes, and size limits with clear messages (§5.4, §17) |
| R9 | **Template regressions.** A prompt change breaks the exact tables | Medium | Medium | Template PRs must pass the real-model e2e acceptance (§18.3). `PROMPT_VERSION` never invalidates stored answers (§5.4) |
| R10 | **Import CSV cache confusion.** Artists read the same CSV with their own Import CSV node and see stale rows after edits | Low | High | Payload outputs are the recommended path; a sidebar note appears when the same file is detected; README (§7.1) |
| R11 | **Local server abuse** (CSRF, DNS rebinding, other local users) | Medium | Low | Token, `Origin` and `Host` checks, localhost binding, per-user token file (§16.1) |
| R12 | **Supply chain drift.** The SemIf archive or the Hugging Face revision changes or disappears | Medium | Low | Recorded hashes and a hash-locked environment stop a bad install; existing installs are unaffected. SemIf comes from the pinned commit archive with a SHA-256 check (user decision); bundling may come later |
| R13 | **Name conflict.** "OpenAct" is taken or trademarked | Medium | Unknown | User decision: keep "OpenAct" for now, with no trademark gate before release; the user may rebrand later. Before 1.0 a rename changes only constants (id, bl_idname prefix, inner prefix, runtime folder); after 1.0 it would need another migration like §12 |
| R14 | **Threading bug.** `bpy` is touched off the main thread and Blender crashes | High | Low | Strict rule from the prototype: all `bpy` access on the main thread through `post()`/`pump()`. The worker holds only plain data. The fake-backend pipeline suite runs under load. Review checklist item |
| R15 | **GPU contention** with Cycles, EEVEE or other applications: slowdowns or out-of-memory | Medium | Medium | The worker pauses during render jobs (§5.5); *Stop Model* frees VRAM; a slow-server warning (§17.3); OOM surfaces as a server error with the log |
| R16 | **Weaker tournaments** above 16 options: conservative confidence and occasional misses | Low | Medium | Unsure flags; README recommends 16 labels or fewer per question, or a two-level split; locks |
| R17 | **Undo and file-size growth** from large `entries` JSON | Low | Low | The 2 MB soft cap bounds warm and cold entries. **Deliberate exception:** live and pinned entries are kept above 2 MB with a size note, bounded by the 16-live-word-set over-cap error, the 12-pin limit and the 4 MB per-entry hard cap (realistic worst case about 3 MB: 12 pins of a 4096-cell table). Prepare for Playback shows the projected size first. During fills the JSON is written at most every 2 s and once at completion (§14.7) |
| R18 | **Scope creep towards agents** delays v1 | Medium | Medium | NG1. v1 ships only the hooks (§5.6), and the PR Plan contains no playback-time updates |
| R19 | **Main-thread planning and graph-edit cost at the documented limits** (measured: RNA writes slow down as trees grow; planning 4096 cells takes about 120 ms, and a 64-setting Dials graph took 1.9 s to build, so Dials is limited to 32 settings, §8.4) | Medium | Medium | Per-element data in the table object (§8.4); bounded materialised entries (§8.1); plan cache and fingerprints (§5.3); the PR-08 contract spike; `--bench` and TIMING budgets as merge gates (§3.2, §8.7) |
| R20 | **Mass re-asks** after an add-on upgrade (new templates), toggling a node's *Balance Option Order*, or a backend switch, silently changing saved scenes | High | Low (by design) | Words keys decide validity and ask keys only label rows (§5.4). A template bump, a toggle change and an upgrade each send 0 requests, asserted in `store_test`, `pipeline_blender` and `migrate_blender` |
| R21 | **Self-referential inputs**: a word source that reads the node's own output (evaluated geometry) re-asks or oscillates | Medium | Low (by design) | Attributes reads original data only and skips names the tree writes (§6.0). A test checks that the tree settles in one pass with one request |
| R22 | **Exfiltration through an untrusted .blend**: OpenAct reads a private file named in a crafted file and stores its text in the saved file | High | Low | Trust rule: untrusted files may read only inside their own folder tree, and blocked text is never read or stored (§16.1). Pipeline test |
| R23 | **CSV drift on another machine with the add-on**: different numeric CSV values on a farm or a colleague's machine re-bake silently, so renders differ from what the artist saw | Medium | Medium | In background mode (`bpy.app.background`) and during render jobs, OpenAct neither re-plans nor re-bakes from files. It keeps the saved values and prints "CSV <path> differs from the saved table (text/numbers); rendering the saved answers" (§7.8). Interactive sessions re-bake as usual and show the same note once in the status |

## 20. Rollout Plan

### 20.1 Where development happens

- **Repository** (user decision): **https://github.com/nicolaiprodromov/openact**.
  - It was created by the user; it is private, empty and has no default branch yet.
  - The gh CLI on this machine is logged in as `nicolaiprodromov` with the `repo` and `workflow`
    scopes.
  - **Local clone:** `X:\interzis\CHUMP_KIT\openact\`.
  - **PR-01** clones the repository, creates `main`, scaffolds, commits this design document as
    `docs/DESIGN.md`, and pushes.
  - The clone holds the package, `tests/`, `demos/`, `docs/` and `dist/`:
  ```
  openact/
    openact/     (the add-on package)
    tests/
    demos/
    docs/DESIGN.md
    dist/
    README.md
  ```
- **The prototype folder `X:\interzis\CHUMP_KIT\jev_nodes\` stays untouched.**
  `tests/migrate_blender.py` builds v0.1.0 files from it (§12.5), and it remains the fallback for
  Blender 4.2 users (§11).
- **One PR per PR Plan entry**, built by parallel worker subagents under an orchestrator (§20.6)
  and merged into `main` in dependency order. Each merge must pass its gates (§18.3). A build step
  turns any commit into `dist/openact-<version>.zip`.

### 20.2 Stages

| Stage | Audience | What ships | Exit criteria |
|---|---|---|---|
| 0. Development | the orchestrator and its worker subagents on this machine (two RTX 3090, Blender 4.5 / 5.0 / 5.1; §20.6) | PRs merged wave by wave, in dependency order | Every gate green; §3.2 targets met on this machine |
| 1. Internal alpha | the user | `dist/openact-1.0.0-alpha.N.zip`; the demo files; the user's existing Jev files upgraded | All demo acceptance checks (§18.4); no data lost on upgrade; the user signs off on the nodes and the sidebar |
| 2. Private beta | 5–10 Geometry Nodes artists on **Windows 10/11** with NVIDIA GPUs of 10 GB or more (the v1 support matrix, §3.3); Linux testers welcome as best effort | A GitHub pre-release zip, README and known-issues list | No open crash or data-loss bugs; at least 90 % of testers install successfully; the Game of Life acceptance reproduces on their machines; feedback collected on `unsure_below` and on the CSV formats |
| 3. Release 1.0 | the repository's audience (user decision) | A **GitHub release** in `nicolaiprodromov/openact`, plus the promo video. The repository is private for now, and making it public is the user's call at release time. A Blender Extensions submission is considered **after** 1.0 | All release gates (§18.3); release checklist (§20.5) |
| 4. Future work (not in this plan) | — | A CPU backend (llama.cpp); a possible high-accuracy model (README note only); the separate agents design document | Separate plans |

### 20.3 Switches and flags

There are no remote flags, because OpenAct makes no network calls after install.

| Switch | Kind | Purpose |
|---|---|---|
| `OPENACT_BACKEND=fake`, `OPENACT_SERVER_PYTHON=<python>` | environment | Tests and CI without a GPU or the runtime (§18.1) |
| `OPENACT_BACKGROUND_PASSES=1`, or in process `openact.engine.allow_background_passes(True)` | environment / Python API | Lets headless authoring (test suites, `demos/build_demos.py`, user scripts) run passes. Without either, background mode renders only the saved answers (§7.8) |
| `OPENACT_BENCH_SHAPE=demo\|worst` | environment | Selects the shape that `tests/lookup_blender.py --bench` builds for the main-thread budgets (§8.4) |
| `debias` (the default for new nodes), `auto_start`, `unsure_below` (0.30), `log_requests` | preferences | Behaviour and diagnostics (§13.4) |
| `balance_order` | node property | Per-node order balancing (§10.3) |
| `OPENACT_STATE_DIR` | environment | Writable state folder (logs, probe cache, token files). Build workers set it so the shared runtime stays read-only (§5.2, §20.6) |
| `backend` enum | preference | A new backend appears only when its runtime profile is installed **and** its PR's acceptance suite passed. Until then it is hidden |

### 20.4 Rollback

- **Uninstall OpenAct.** Every file keeps evaluating on Blender 4.5+ from its stored answers
  (§12.5). The runtime folder stays until the user deletes it.
- **Back to v0.1.0.** Install `dist/jev_nodes-0.1.0.zip`. Files that were never upgraded work
  exactly as before. Upgraded files show undefined nodes, which still evaluate on 4.5+.
- **Upgrades are explicit and undoable.** They happen in one undo step, and Blender's `.blend1`
  save versions keep the previous file on disk.
- **Bad template or model release.** Stored answers never depend on `PROMPT_VERSION` or model
  provenance (§5.4), so downgrading the add-on loses no answers.
- **Server trouble.** *Stop Model*; answers stay, and rendering never needs the server.

### 20.5 Release checklist

1. Bump `version` in the manifest and in `bl_info`, and update the README changelog.
2. Run `blender --command extension validate` and `blender --command extension build`.
3. Run every suite on 4.5, 5.0 and 5.1 with `tests/run_all.ps1 -Model` (§18.3).
4. Rebuild the demos with `--model`, then open and render each one **without** the add-on on 4.5.
5. README covers:
   - installation and the runtime (13 GB, reuse of `JevNodes`);
   - the six nodes;
   - the CSV formats and templates;
   - the two ways to give a node its CSV: a typed path, or a plugged-in Import CSV node (§6.0);
   - the limitations: Blender's path cache for the artist's *own* Import CSV nodes, 4.5+ only,
     NVIDIA/CUDA only;
   - **future work**:
     - a CPU backend (llama.cpp);
     - a possible high-accuracy model: SemIf's 27B "bridge" (EXL3, 5 bpw; 0.958 against 0.813
       balanced accuracy on SemIf's benchmark; needs a 24 GB GPU). It is not built and not in this
       plan (user decision);
   - credits and licences;
   - the Jev non-affiliation line.
6. Tag the commit and create the **GitHub release** in `nicolaiprodromov/openact` with
   `gh release create`, attaching the zip and the release notes. The repository stays private until
   the user decides otherwise. A Blender Extensions submission is considered after 1.0.

### 20.6 Build orchestration (parallel subagents)

The PR Plan is built **locally on this machine by parallel worker subagents, coordinated by a main
agent, the orchestrator** (user decision).

**Workers.**
- There is **one worker subagent per PR**. It works in its **own git worktree and branch**
  (`git worktree add ..\openact-wt\pr-NN -b pr-NN main`), created from the latest local `main` that
  already contains all of that PR's dependencies.
- A worker implements exactly its PR section.
- It runs that PR's *Tests to merge* in its worktree and reports the diff and test output to the
  orchestrator.
- It never merges, and it never touches another worktree.

**Orchestrator loop.**
1. **Dispatch** every PR whose dependencies are all merged. These are the waves below, derived from
   the dependency table.
2. **Review** each finished PR: its diff against the PR section, and its test output against its
   *Tests to merge*. Send it back to the worker with notes if anything is missing.
3. **Merge** into local `main` in dependency order (`git merge --no-ff pr-NN`).
4. After **each** merge, **re-run the full suite** (`tests/run_all.ps1`, with `-Model` when a GPU
   lease is free).
5. On a failure, revert that merge and return the PR to its worker.
6. **Push**, using the one GitHub flow: *the orchestrator merges locally and pushes `main`*.
   - Workers push their `pr-NN` branch.
   - The orchestrator opens a GitHub pull request for the record with `gh pr create`.
   - After the local merge passes the full suite, it pushes `main`. GitHub then marks the pull
     request merged, because its commits are in `main`.
   - `origin/main` is therefore always green.
7. Remove the finished worktree, then dispatch the next ready PRs.

**Shared-resource rules.**

| Resource | Rule |
|---|---|
| GPUs: two RTX 3090 | At most **two concurrent real-model test runs**, one per GPU. Each gets a lease from the orchestrator (`--gpu 0` or `--gpu 1`) and its **own server port** (18765 on GPU 0, 18767 on GPU 1) |
| Installed runtime at `%LOCALAPPDATA%\JevNodes` | Shared **read-only**. Workers never modify or reinstall it, and set `OPENACT_STATE_DIR` to a per-worker temporary folder for logs, probe cache and token files (§5.2). Installer tests use a temporary runtime folder with mocked downloads |
| Headless Blender processes | At most **4** concurrent, across all workers. The machine has 128 GB RAM |
| Prototype folder `X:\interzis\CHUMP_KIT\jev_nodes\` | **Read-only**; `migrate_blender.py` copies from it into a temporary folder |
| GPUs shared with the user's other jobs | Tests must not assume exclusive GPU use. Before starting a server they check free VRAM (≥ 9 GB) and wait if it is short. Real-model timeouts are generous, and timing checks fail only above 2× target (§18.2), so slower runs pass |

**Schedule: parallel waves.** The S/M/L/XL sizes remain relative effort (§PR Plan). A wave contains
every PR whose dependencies are in earlier waves:

| Wave | PRs |
|---|---|
| W1 | PR-01 |
| W2 | PR-02 |
| W3 | PR-03, PR-04, PR-05, PR-06 |
| W4 | PR-07, PR-08 |
| W5 | PR-09 |
| W6 | PR-10, PR-11 |
| W7 | PR-12, PR-13 |
| W8 | PR-14, PR-15 |
| W9 | PR-16, PR-17, PR-18, PR-20 |
| W10 | PR-19 |
| W11 | PR-21 |
| W12 | PR-22 |
| W13 | PR-23, PR-24 |
| W14 | PR-25 |
| W15 | PR-26 |

- **Critical path (by size):** PR-01 → PR-02 → PR-05 → PR-08 → PR-09 → PR-11 → PR-13 → PR-15 (or
  PR-14) → PR-18 → PR-19 → PR-21 → PR-22 → PR-23 → PR-25 → PR-26.
  - That is about 24 relative units, against about 37 units of total effort.
  - Parallel workers can therefore shorten the build to roughly the critical path plus review and
    merge overhead.
- No wave has more than four PRs, so the 4-process Blender cap is never exceeded by the number of
  workers alone.
- A PR can start as soon as its own dependencies are merged, even before the rest of its wave.

## 21. Open Questions

**No open decisions remain.** The user decided every earlier question:
- the name (keep OpenAct);
- CSV hand-off (a typed path, or a plugged-in Import CSV node);
- the repository (`nicolaiprodromov/openact`);
- the unsure threshold (0.30, adjustable);
- NVIDIA/CUDA only;
- order balancing as a per-node toggle;
- the 27B model (README only);
- SemIf supply (the pinned archive with SHA-256);
- local builds by parallel subagents;
- the GitHub release, with an Extensions submission after 1.0.

What remains are **measurement-driven follow-ups**. They are not decisions: each is measured
during or after the build, and none blocks implementation.

**F1. Does the Import CSV refresh trick work?** Blender caches Import CSV by path (§7.1). An
equivalent spelling, such as `//./life.csv` for `//life.csv`, might make the artist's *own* Import
CSV nodes re-read an edited file.
- **Measure:** `tests/csv_probe_blender.py` checks it on 4.5 and 5.1, in the CSV PR (PR-14).
- **Outcome:** `openact.refresh_import_csv` ships only if it re-reads on both versions. Otherwise
  the README says "use the payload outputs, or restart Blender".
- This does not affect OpenAct nodes, which parse files themselves even when an Import CSV node is
  plugged in (§6.0).

**F2. Accuracy with and without order balancing.** The toggle is decided: per node, default on
(§10.3). What is not yet measured is how much accuracy it buys.
- **Measure:** the Game of Life (four rule sets) and biome acceptance suites run with the toggle
  on and off.
- **Outcome:** documented guidance on when turning it off is safe. The default stays on unless the
  data says otherwise.

**F3. Calibrating the unsure threshold.** The default of 0.30, adjustable in the preferences, is
decided (§4.e).
- **Measure:** beta data, **separately per answer type**: the share of flagged cells that were
  actually wrong, against the share of wrong cells that were not flagged. All formulas share the
  endpoints 0 = no information and 1 = certain.
- **Outcome:** if the types diverge, hidden per-type defaults go behind the one preference, in a
  later minor release. The threshold affects UI flags only, never outputs.

## 22. References

**Prototype (read in full for this design; no git repository):**
- `X:\interzis\CHUMP_KIT\jev_nodes\jev_nodes\__init__.py`: registration
- `X:\interzis\CHUMP_KIT\jev_nodes\jev_nodes\prefs.py`: preferences, runtime paths, `runtime_installed()`
- `X:\interzis\CHUMP_KIT\jev_nodes\jev_nodes\server.py`: lifecycle, installer, HTTP client, main-thread queue
- `X:\interzis\CHUMP_KIT\jev_nodes\jev_nodes\engine.py`: static evaluator, contexts, specs, inner builder, passes
- `X:\interzis\CHUMP_KIT\jev_nodes\jev_nodes\nodes.py`: `GeometryNodeJevAsk`, `GeometryNodeJevPick`, Add menu
- `X:\interzis\CHUMP_KIT\jev_nodes\jev_nodes\ops.py`, `ui.py`: operators, sidebar, preferences UI
- `X:\interzis\CHUMP_KIT\jev_nodes\jev_nodes\natsort.py`: `BLI_strcasecmp_natural` port
- `X:\interzis\CHUMP_KIT\jev_nodes\jev_nodes\runtime\jev_server.py`: the local Jev-compatible server
- `X:\interzis\CHUMP_KIT\jev_nodes\jev_nodes\blender_manifest.toml`: extension manifest
- `X:\interzis\CHUMP_KIT\jev_nodes\tests\e2e_blender.py`, `tests\evaluator_blender.py`: existing tests
- `X:\interzis\CHUMP_KIT\jev_nodes\README.md`, `dist\jev_nodes-0.1.0.zip`: docs and build
- `%LOCALAPPDATA%\JevNodes`: the installed runtime (venv 4.4 GB, model 8.7 GB)

**Measurements used** (from the task brief; RTX 3090, bf16, Blender 4.5.2 and 5.1.1, 2026-09):
- model load 12–40 s and 8.4 GB of VRAM;
- decision latencies and table-fill throughput;
- Game of Life, HighLife, Seeds and Day & Night exactness with "outcome + when";
- biome confidence of 0.32 against 0.61;
- the Import CSV behaviours: numeric columns only, and the path cache;
- `GeometrySet` access, and the evaluation of hidden objects in excluded collections;
- `default_input`, Index Switch types, interface propagation, and the `copy()` pointer issue;
- the cost of `node.location` (about 6 ms per node in 5.1), measured in the sibling RB Solver v4
  project (`X:\interzis\CHUMP_KIT\RIGID BODY\v4_src\`).

**External:**
- SemIf (formerly OpenJev), MIT, pinned commit `23cf1f39fc9534fe81437200959b6dfc7106e45a`:
  https://github.com/TheoLeeCJ/SemIf-OpenJev
- Qwen3.5-4B, Apache-2.0, pinned revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`:
  https://huggingface.co/Qwen/Qwen3.5-4B
- TypeSafe Jev API: the `POST /v1/systemone` request shape and the Noul / Choice / Score
  primitives mirrored by `openact_server.py`. OpenAct is not affiliated with TypeSafe.
- uv 0.8.17 (installer tool): https://github.com/astral-sh/uv
- PyTorch 2.10.0 wheels (cu128 index): https://download.pytorch.org/whl/cu128
- Blender Python API (`GeometryNodeCustomGroup`, `NodeTreeInterface`, `GeometrySet`, handlers,
  timers): https://docs.blender.org/api/current/
- Blender Manual (Geometry Nodes: Import CSV, Import Text, Index Switch, Menu Switch, Simulation
  and Repeat zones; Extensions: manifest and permissions): https://docs.blender.org/manual/en/latest/
- Blender LTS release policy: https://www.blender.org/download/lts/
- Blender source, `BLI_strcasecmp_natural` in `source/blender/blenlib/intern/string.cc` (ported
  in `natsort.py`)
- RFC 4180, the CSV format: https://www.rfc-editor.org/rfc/rfc4180
- Python `csv` module (used by `csvfmt.py`): https://docs.python.org/3/library/csv.html

## Key Decisions

| # | Decision | Rationale | Where |
|---|---|---|---|
| KD1 | The product is **OpenAct**: package and extension id `openact`, bl_idnames `GeometryNodeOpenAct*`, inner prefix `.OpenAct `. "Jev" appears only in credits and API-compatibility notes | "Jev" is TypeSafe's trademark, and SemIf renamed itself from "OpenJev" for the same reason. **User decision:** keep "OpenAct" for now, with no trademark gate before release; the user may rebrand later. Before 1.0 a rename changes only constants (id, bl_idname prefix, inner prefix, runtime folder) | §12.1, R13 |
| KD2 | **Six nodes, one per verb**: Ask, Pick, Match, Classify, Rules, Dials. They share list sources, one answer-set format, the answer types and one confidence definition | Discoverable, and each node shows only its own sockets. Consistency comes from shared code (`specs/base.py`, `lists.py`, `csvfmt.py`), not from one mode-laden node | §6, §15.1 |
| KD3 | **Decide per meaning, apply per element.** Decisions are table rows in the node's hidden inner group, looked up per element through an ID or factor fields. The model never runs during evaluation, playback or rendering | Geometry Nodes cannot run Python while it evaluates. Tables are bounded by meaning (at most 4096 cells), saved, deterministic and safe on a render farm | §4.b, §8, §15.2 |
| KD4 | Every answer set is **"answer + when it applies"**. Exact facts are handled by code: `if` restrictions, and single-option cells decided without the model | Measured: descriptions raised confidence from 0.32 to 0.61, and "outcome + when" made four Game of Life rule sets exact. Offering only the applicable clauses fixed 3 wrong cells | §4.c, §4.d, §9.4 |
| KD5 | **CSV authoring with a text/numeric split.** Python parses every column, whether the file comes from a typed path or from a plugged-in Import CSV node (KD25). Text columns go to the model and into cell keys. Numeric payload columns are **baked** into the table and exposed as output sockets. The inner group never uses Import CSV's output | Import CSV drops text columns and caches by path (verified). Baking keeps the .blend self-contained, and numeric edits never re-ask the model | §7.1, §15.3 |
| KD6 | Rules factors are declared in a **separate Factors CSV**. Artist overrides live in an **Overrides CSV** with a `lock` column. The sidebar is inspection only | "Everything happens in Geometry Nodes and files" (4.h). Files are diffable and can vary per object through Group Input | §7.4, §7.6, §15.5, §15.7 |
| KD7 | **Every per-element column lives in the node's hidden table object** (Object Info + Sample Index). Index Switch holds only entry-level single values. Only **materialised** entries enter the graph: live contexts, then at most 12 pinned, then warm entries up to 4, with a hard cap of 16. More live word sets than that is an error. The **stored** entries live in `entries` JSON, the single source of truth, whose `shape` suffices to rebuild without CSVs. Live and pinned entries are never evicted | Measured: graph edits cost much more as trees grow, while per-element lookup is nearly free on both paths. Value updates become `foreach_set` with a constant graph. Table size depends on live contexts, not edit history. Derived data can always be rebuilt | §8.1, §8.4, §8.5, §14.1 |
| KD8 | **Rules maths runs in nodes:** row-major cell index; bands, categories and flags mapped by node graphs; multilinear interpolation between band centres for Yes/No and Amount (at most 3 Amount factors, renormalised by `ok`); random dithering near band edges for One-of | Words in, data out (4.a). Amount answers are smooth and monotonic (measured), so interpolation removes stair steps. Dithering blends discrete outcomes honestly | §9 |
| KD9 | **Confidence everywhere:** `abs(2p−1)` for Yes/No, top minus runner-up for One-of, and `clamp(1 − σ/σu(L), 0, 1)` for Amount (new; σu is the spread of a uniform answer). All three map "no information" to 0 and "certain" to 1. Rows below "Unsure below" (0.30) are flagged | Low confidence tracked the measured errors. Top-minus-runner-up misreads ordinal answers. Anchoring all types at the same endpoints makes one threshold meaningful. **User decision:** 0.30 by default, adjustable in the preferences, tuned later with beta feedback (follow-up F3) | §4.e, §21 F3 |
| KD10 | **Minimum Blender 4.5 LTS**; tested on 4.5, 5.0 and 5.1 | Import CSV, Import Text, `GeometrySet`, the input nodes, Integer Math and evaluation without the add-on all require 4.5. 4.2's LTS window has closed | §11 |
| KD11 | **Keep the runtime split.** The model server runs in its own venv behind the Jev-compatible localhost API, with a backend interface (SemIf torch, fake), a per-question cache, `/v1/capabilities`, and a launch token plus `Origin`/`Host` checks | Blender stays pure Python and crash-safe. Backends are swappable, a hosted Jev stays possible, and CSRF and DNS rebinding are blocked | §5.2, §10.4, §15.4, §16 |
| KD12 | **The cell is the unit of caching and batching.** Each table has one state; chunks hold at most 32 questions / 64 rows (one node's job, so one request carries that node's order-balancing setting); one request is in flight; priority classes P0/P1/P2; stale-while-refilling. Order balancing is a **per-node toggle, default on** (KD26) | Maximal `score_shared` batching, and interactive questions overtake long fills after at most one chunk. Partial tables never blank out | §5.3–§5.5, §10.3 |
| KD13 | **Two key families.** Words keys (`wkey` per row, `words_key` per entry) cover only what the artist said, and decide whether a stored answer is valid. The ask key (templates, the node's *Balance Option Order* setting, backend) and the cell key (the exact text sent) only label rows and key the caches | Template upgrades, per-node order-balancing toggles, backend switches and upgrades of v0.1.0 files send 0 requests. After a words edit, only rows whose own words changed are asked again. Files keep the answers they were saved with | §5.4, §14.2 |
| KD14 | **Dynamic sockets** come from diffing the inner group's interface. The ID input uses `default_input = 'INDEX'`, not `ID_OR_INDEX` | Both behaviours are verified in 4.5.2 and 5.1.1. An ID means "entry number", and Distribute Points gives points large random ids that are not entry numbers | §6.0 |
| KD15 | **Migration:** reuse the 13 GB `JevNodes` runtime in place; old nodes load as frozen legacy aliases; an explicit, undoable upgrade converts entries from schema 1 to schema 2 and keeps the answers | No re-download, no silent file rewrites, and upgraded files need no model to reopen | §12, §14.4 |
| KD16 | **Supply chain:** pinned versions, SHA-256 checks for uv and SemIf, a hash-locked environment, a ≥ 15 GB free-space check, and `online_access` respected | A 13 GB install must be reproducible, and it must fail loudly rather than end up half-working | §16.2, R3 |
| KD17 | Development happens in the user's GitHub repository **https://github.com/nicolaiprodromov/openact** (private), cloned to `X:\interzis\CHUMP_KIT\openact\`. PR-01 clones it, creates `main`, scaffolds, commits this design as `docs/DESIGN.md`, and pushes. The prototype folder stays untouched and read-only. 1.0 is released as a GitHub release there; making the repository public is the user's call at release time, and an Extensions submission is considered after 1.0 | User decisions. The migration tests build v0.1.0 files from the untouched prototype, which also remains the fallback for Blender 4.2 users | §20.1, §20.2, §20.5 |
| KD18 | **Only NVIDIA/CUDA is built** (user decision). A CPU backend (llama.cpp) is future work, outside this plan, and so is Apple Silicon. A high-accuracy 27B model is mentioned in the README only | Built and measured on RTX 3090s. Any future backend must pass the same acceptance suites before being offered | §3.3, NG9, §20.5 |
| KD19 | **Agents are deferred.** v1 ships only hooks: `pipeline.submit(..., on_rows=…)`, table objects writable without rebuilds, and `geometry.read_attributes` | Keeps v1 focused (NG1) without precluding "think slow, act fast" | §5.6 |
| KD20 | **Shape agreement across contexts.** All live contexts of a node must agree on the interface shape (list source, answer type, factors, settings, payload headers), **and on every value baked into the graph** (band edges and centres, category offsets and value maps, flag assignments). Anything else (texts, payload values, Dials ranges, Pick's picked material) is per-entry data selected by `k` | The interface exists once per node. A clear error is safer than a silent "first context wins", and per-entry values keep per-object variation possible | §6.0, §8.6 |
| KD21 | **Trust model for opened files.** Untrusted files may read only inside their own folder tree. Trust comes from a per-file *Allow this file* (`trusted_files`) or an explicit *Trust this folder* (`trusted_folders`), and a file created in this session is trusted. Blender's *Auto Run Python Scripts* does not count: a file that reads outside its folder needs one *Allow this file* click. Appended groups inherit the host file's trust, which is a documented residual risk | OpenAct stores the words it reads in the saved file, so a crafted file could otherwise exfiltrate local text. Blender can refuse Auto Run for one file (*Trusted Source* unchecked), and an add-on can't see that refusal. Demos with their CSVs still work, and allowing one download does not trust a whole Downloads folder | §16.1 |
| KD22 | **v1 platform matrix.** Windows 10/11 with an NVIDIA GPU of ≥ 10 GB is supported and tested. Linux with NVIDIA is best effort. There is **no CPU path** in v1: the installer refuses without an NVIDIA GPU, and at run time `auto` means CUDA only. macOS is unsupported for authoring (future work) | Every gate runs on the one tested platform. A 13 GB install that cannot meet any target is refused up front, and a model that cannot use CUDA reports an error instead of crawling on the CPU | §3.3, §5.2, §10.5 |
| KD23 | **Main-thread discipline.** A plan cache with fingerprints; rebuilds only on shape or materialised-set changes; table-object writes throttled; budgets as merge gates | Graph edits and planning are the expensive side (measured); an unchanged scene must cost almost nothing per pass | §5.3, §8.4, §8.8 |
| KD24 | **Playback safety.** On a node used by exactly one live context, and not in a hint or error state, a no-match falls back to that context's last-shown entry. Shared trees fall back to neutral values. *Prepare for Playback* pins the entries animated words need | Passes stay paused during playback (NG1), yet a single-object sim must not break. On shared trees, one object must never silently show another's answers while OpenAct runs. Without the add-on, `k_default` stays as saved (R6) | §5.5, §8.1, §8.2 |
| KD25 | **Import CSV hand-off** (user decision). A node gets its CSV from a typed path (String input) **or** from Blender's own Import CSV node plugged into the companion `<Input> CSV` Geometry socket. OpenAct traces the link (Reroutes, nested groups, Geometry selectors with static selectors) to the Import CSV node, reads its Path and Delimiter statically, and parses the file itself. A link wins over a typed path. Any other node, or a modifying node in between, is an error ("connect the Import CSV node directly") | Artists can wire CSVs the Blender way, while OpenAct keeps the text columns and always sees edits. The Import CSV geometry and path cache are never used. Limitation: the match chain can't see a traced path, so per-object files use route (a) or `Key` | §6.0, §7.7, §10.1 |
| KD26 | **Per-node order balancing** (user decision). *Balance Option Order* (`balance_order`) is a node checkbox, default on, seeded from the global `debias` preference. It belongs to the ask key: toggling it discards nothing and sends 0 requests; new asks, and *Ask Again*, use the node's current setting | Some work doesn't need the double check, which costs 2× scored rows. Keeping it in the ask key preserves answers (KD13). Measuring accuracy with and without it is follow-up F2 | §6.0, §10.3, §14.2 |
| KD27 | **Build orchestration** (user decision). The PR Plan is built locally by parallel worker subagents, one per PR, each in its own git worktree and branch, under an orchestrator that dispatches waves, reviews against *Tests to merge*, merges in dependency order, re-runs `tests/run_all.ps1`, and pushes `main`. Shared-resource rules: two GPU leases (one real-model run per RTX 3090, own port), a read-only runtime (`OPENACT_STATE_DIR` per worker), at most 4 headless Blenders, a read-only prototype, and tests tolerant of shared GPUs | Parallel waves shorten the build to about the critical path, while `origin/main` stays green and shared resources are never corrupted | §20.6 |

## PR Plan

**Conventions.**
- All work happens in the local clone `X:\interzis\CHUMP_KIT\openact\` of
  **https://github.com/nicolaiprodromov/openact** (private; §20.1), whose layout is:
  ```
  openact/
    openact/     (the add-on package)
    tests/
    demos/
    docs/DESIGN.md
    dist/
  ```
  The prototype folder `X:\interzis\CHUMP_KIT\jev_nodes\` is never modified.
- **Build orchestration (§20.6):** one worker subagent per PR, in its own worktree and branch.
  The orchestrator dispatches the waves listed in §20.6, reviews each PR against its *Tests to
  merge*, merges locally in dependency order, re-runs `tests/run_all.ps1` after each merge, and
  pushes `main`.
- PR IDs give a valid topological order: every dependency has a lower ID. "Parallel with" lists
  PRs that can be developed at the same time once their own dependencies are merged.
- **Tests to merge** must pass on Blender 4.5 and 5.1, via `tests/run_all.ps1` from PR-02 on. They
  follow the gates in §18.3. No PR may break a suite that already passes.
- **Real-model** tests run on this machine (two RTX 3090, at most one run per GPU under the
  orchestrator's leases, §20.6) against the installed runtime, found through the legacy `JevNodes`
  fallback (§12.2), which is read-only for workers.
- Two suites complement §18.2: `tests/server_test.py` (PR-06) and `tests/installer_blender.py`
  (PR-07).

**Milestones.**

| Milestone | PRs | Result |
|---|---|---|
| M1 Foundation | PR-01 … PR-09 | The renamed prototype running on the new architecture (evaluator, store, server v1, installer, inner builder, specs, cell pipeline) |
| M2 Tables and data | PR-10 … PR-15 | Decision worker, per-element lookup through table objects, list sources with path resolution and the trust rule, CSV watcher, answer sets |
| M3 New nodes | PR-16 … PR-21 | Match, Classify, Rules I and II, Dials, Overrides |
| M4 Product 1.0 | PR-22 … PR-26 | Sidebar and status, templates and diagnostics, migration, demos, **release 1.0.0** |

**Future work (not in this plan):** a CPU backend (llama.cpp, GGUF Q4_K_M) and a possible
high-accuracy 27B model. Both are noted in the README by PR-26. Apple Silicon is not planned.

**Sizes** are **relative effort**, including the PR's tests: **S** = ½, **M** = 1, **L** = 2,
**XL** = 3 units. The schedule is the parallel waves in §20.6, not a calendar. An L or XL PR that
overruns splits along the seam in its *Split seam* column: part *a* merges first, part *b* after
it, with the same dependencies.

| ID | Title | Size | Depends on | Parallel with | Split seam (L/XL) |
|---|---|---|---|---|---|
| PR-01 | Repository, rename and scaffold | M | — | — | — |
| PR-02 | Test harness and runner | S | PR-01 | — | — |
| PR-03 | CSV reader and inline-list parsing | M | PR-02 | PR-04, PR-05, PR-06 | — |
| PR-04 | Node store, schema 2 | S | PR-02 | PR-03, PR-05, PR-06 | — |
| PR-05 | Evaluator module and new word sources | M | PR-02 | PR-03, PR-04, PR-06 | — |
| PR-06 | Model server v1 (server side; token checked when set) | L | PR-02 | PR-03, PR-04, PR-05 | a: backends, fake backend (`--record`, load flags), per-question cache · b: capabilities, health counters, limits, 422 ids, auth checks |
| PR-07 | Installer hardening, runtime probe, `locate_runtime` | M | PR-06 | PR-08 … PR-21 | — |
| PR-08 | Inner builder module, interface sync, contract spike | M | PR-04, PR-05 | PR-07 | — |
| PR-09 | Spec framework and cell pipeline model | L | PR-08 | PR-07 | a: specs, key families, validity rule (pure tests) · b: pass on Jobs, plan cache, dirty flags |
| PR-10 | Decision worker, client token, priorities and chunking | L | PR-06, PR-09 | PR-11 | a: worker, priority queue, chunking, cancellation · b: client token, 503/422 handling, pause events |
| PR-11 | Table objects, per-element lookup, `k_default`, Pick fields | XL | PR-09 | PR-10 | a: table object, sampled lookup, `write_rows()` · b: materialised set and cap, `k_default`, Pick fields |
| PR-12 | Table-object lifecycle: copies, append, garbage, save/reload | M | PR-11 | PR-13, PR-14, PR-15 | — |
| PR-13 | List sources, path resolution, Import CSV hand-off, trust rule, per-item planning | XL | PR-03, PR-10, PR-11 | PR-12 | a: list sources, path resolution, Import CSV hand-off (`trace_csv_source`, companion sockets), `CsvDoc` cache, per-item helper · b: trust rule, Attributes (original data), shape agreement |
| PR-14 | CSV file watcher | M | PR-13 | PR-12, PR-15 | — |
| PR-15 | Answer sets and Ask's Answers input | M | PR-13 | PR-12, PR-14 | — |
| PR-16 | Match node | M | PR-12, PR-14, PR-15 | PR-17, PR-18, PR-19, PR-20 | — |
| PR-17 | Classify node | M | PR-12, PR-14, PR-15 | PR-16, PR-18, PR-19, PR-20 | — |
| PR-18 | Rules I: factors, cells, `if`, One-of | L | PR-12, PR-14, PR-15 | PR-16, PR-17, PR-20 | a: Python side (factors, cells, `if`, templates, limits) · b: inner graph, payload, Game of Life sim test and e2e acceptance |
| PR-19 | Rules II: interpolation and dithering | L | PR-18 | PR-16, PR-17, PR-20 | a: interpolation (Yes/No, Amount) · b: dithering (One-of) |
| PR-20 | Dials node | M | PR-14, PR-15 | PR-16 … PR-19 | — |
| PR-21 | Overrides CSV: locks and export (all six nodes) | M | PR-16, PR-17, PR-19, PR-20 | — | — |
| PR-22 | Node status, sidebar inspection, Show Question | M | PR-07, PR-21 | — | — |
| PR-23 | Templates, diagnostics, logs, alerts, playback prep, unpin, server sharing | XL | PR-22 | PR-24 | a: templates, diagnostics, JSONL log, alerts, server sharing · b: Prepare for Playback (read sets, channelbag F-curve path, NLA/driver fallback), pins, unpin |
| PR-24 | Migration from jev_nodes | M | PR-22 | PR-23 | — |
| PR-25 | Demo scenes | L | PR-23 | — | a: `life` and `biome` · b: the other four demos, and the no-add-on render check |
| PR-26 | **Release 1.0.0** | S | PR-07, PR-24, PR-25 | — | — |

**Schedule: parallel waves, not a calendar** (user decision; §20.6).
- **W1** PR-01 · **W2** PR-02 · **W3** PR-03, PR-04, PR-05, PR-06 · **W4** PR-07, PR-08 · **W5**
  PR-09.
- **W6** PR-10, PR-11 · **W7** PR-12, PR-13 · **W8** PR-14, PR-15 · **W9** PR-16, PR-17, PR-18,
  PR-20.
- **W10** PR-19 · **W11** PR-21 · **W12** PR-22 · **W13** PR-23, PR-24 · **W14** PR-25 · **W15**
  PR-26.

The **critical path** is PR-01 → 02 → 05 → 08 → 09 → 11 → 13 → 15 (or 14) → 18 → 19 → 21 → 22 → 23
→ 25 → 26: about 24 of the 37 total effort units. The orchestrator dispatches each PR as soon as
its dependencies are merged.

```mermaid
flowchart LR
  PR01 --> PR02
  PR02 --> PR03 & PR04 & PR05 & PR06
  PR06 --> PR07
  PR04 --> PR08
  PR05 --> PR08
  PR08 --> PR09
  PR06 --> PR10
  PR09 --> PR10
  PR09 --> PR11
  PR11 --> PR12
  PR03 --> PR13
  PR10 --> PR13
  PR11 --> PR13
  PR13 --> PR14
  PR13 --> PR15
  PR12 --> PR16 & PR17 & PR18
  PR14 --> PR16 & PR17 & PR18 & PR20
  PR15 --> PR16 & PR17 & PR18 & PR20
  PR18 --> PR19
  PR16 --> PR21
  PR17 --> PR21
  PR19 --> PR21
  PR20 --> PR21
  PR21 --> PR22
  PR22 --> PR23 & PR24
  PR23 --> PR25
  PR07 --> PR22
  PR07 --> PR26
  PR24 --> PR26
  PR25 --> PR26
```

### PR-01: Repository, rename and scaffold
- **Files/components:**
  - **Clone** `https://github.com/nicolaiprodromov/openact` (private, empty, no default branch)
    into `X:\interzis\CHUMP_KIT\openact\` with `gh repo clone nicolaiprodromov/openact`. Then
    **create `main`**: the first commit is on `main`, which becomes the default branch.
  - `.gitignore` (`__pycache__/`, `*.pyc`, `dist/`, `*.blend1`) and a `README.md` skeleton.
  - **`docs/DESIGN.md`**: this design document, committed verbatim.
  - `openact/openact/`, copied from `jev_nodes/jev_nodes/` and renamed: `__init__.py`,
    `blender_manifest.toml`, `prefs.py`, `server.py`, `engine.py`, `nodes.py`, `ops.py`, `ui.py`,
    `natsort.py`, `runtime/openact_server.py`.
  - `tests/e2e_blender.py` and `tests/evaluator_blender.py`, copied and renamed.
- **Depends on:** none. **Parallel with:** none.
- **Changes:**
  - Apply the rename map (§12.1): package and manifest (§13.5); `bl_info` `(4, 5, 0)`; bl_idnames
    `GeometryNodeOpenActAsk` / `GeometryNodeOpenActPick`; labels "Ask" and "Pick"; `INNER_PREFIX`
    `.OpenAct `; menu *Shift A > OpenAct*; tab "OpenAct"; `openact.*` operators;
    `OpenActPreferences`; `openact-*` thread names; `SERVER_ID` / `SERVER_NAME` `openact-semif`.
  - `register()` refuses to run on Blender below 4.5 (§11).
  - `prefs.runtime_root()` gets the legacy fallback: explicit folder → `OpenAct` → `JevNodes`
    (§12.2).
  - The status reports an old `jev-nodes-semif` server on the port (§5.2).
  - Otherwise the behaviour is identical to v0.1.0: two nodes, the same answers.
  - **Push** `main` to origin (`git push -u origin main`). Every later PR branches from it (§20.6).
- **Tests to merge:**
  - `origin/main` exists and contains `docs/DESIGN.md` (`gh repo view` shows `main` as the default
    branch).
  - `tests/evaluator_blender.py` on 4.5 and 5.1.
  - `tests/e2e_blender.py` phases `build`, `noaddon` and `reopen` on 4.5 and 5.1, with the real
    model through the legacy runtime.
  - `blender --command extension validate` passes.

### PR-02: Test harness and runner
- **Files/components:**
  - `tests/_harness.py`: `check()` and the PASS/FAIL/RESULT protocol, `drive()`, `sock()`,
    `enable_addon()`, `store()`;
  - `tests/run_all.ps1`;
  - the `tests/fixtures/` folder;
  - the existing suites, switched to the harness.
- **Depends on:** PR-01. **Parallel with:** none.
- **Changes:** `run_all.ps1 [-Model] [-Versions 4.5,5.0,5.1]` (§18.1):
  1. Finds `C:\Program Files\Blender Foundation\Blender <v>\blender.exe`.
  2. Runs the `*_test.py` suites under every CPython ≥ 3.11 it finds (PATH, `py -3.11`,
     `py -3.13`), and always inside Blender 4.5 (Python 3.11) and 5.1 (Python 3.13).
  3. Runs the `*_blender.py` suites per version with `-b --factory-startup --python`.
  4. Runs e2e only with `-Model` (`build` → `noaddon` → `reopen` in a temporary folder).
  5. Parses the `RESULT` lines, prints a summary table and exits non-zero on any failure.
- **Tests to merge:** `tests/run_all.ps1 -Model -Versions 4.5,5.1` is green (evaluator and e2e).

### PR-03: CSV reader and inline-list parsing
- **Files/components:**
  - `openact/csvfmt.py` (new, pure Python);
  - `openact/lists.py` (new: `parse_inline`, `is_csv_path`; `bpy` imported lazily inside later
    functions);
  - `tests/csv_test.py`;
  - `tests/fixtures/*.csv`:
    - `life_factors`, `life_outcomes`, `highlife_outcomes`, `seeds_outcomes`, `daynight_outcomes`;
    - `biome_factors`, `biome_outcomes`, `material_families`, `weather_dials`, `props`;
    - one broken file per §7.9 message.
- **Depends on:** PR-02. **Parallel with:** PR-04, PR-05, PR-06.
- **Changes:**
  - The §7.2 dialect, numeric classification, limits, `CsvDoc`, `Problem`, and `text_hash` /
    `num_hash`.
  - The parsers `parse_answer_set`, `parse_factors`, `parse_settings` and `parse_overrides`
    (§7.3–§7.6, §13.3). The semantic checks of Rules `if` arrive with PR-18.
  - The inline-list rules (§6.0): newline, then `;`, then `,`; `name: when`; `.csv` detection.
- **Tests to merge:** `tests/csv_test.py` under plain CPython ≥ 3.11, and inside Blender 4.5
  (Python 3.11) and 5.1 (Python 3.13).

### PR-04: Node store, schema 2
- **Files/components:**
  - `openact/store.py`;
  - `tests/store_test.py`;
  - `tests/fixtures/jev_v010_entries.json`, captured from the prototype's `set_entries` for Ask
    Yes/No, Ask Amount, Pick from text and Pick from a collection.
- **Depends on:** PR-02. **Parallel with:** PR-03, PR-05, PR-06.
- **Changes:**
  - `NodeStore` and its entry dataclasses, with columnar rows (§14.2), plus `load(node)` /
    `save(node, store, live_gns=None)`.
  - Schema 1 → 2 conversion, including `gn` re-ordering, `legacy = true` and `pv = 0` (§14.4).
  - The 2 MB soft cap (evicts least recently used **warm and cold** entries only; keeps live and
    pinned entries, even above 2 MB, with a size note) and the 4 MB per-entry hard cap (§14.7).
  - `save(node, store, live_gns=None)` (§13.3): live entries are identified by the `gn` tuples the
    caller passes, because "live" is not saved. **`None`, the default, means "unknown" and evicts
    nothing**, so operators that edit entries can never evict live ones by accident. PR-08 wires
    the pass, the only caller that passes a live set, to fill `live_gns` from its current contexts.
  - The store is not wired into the engine yet (PR-08).
- **Tests to merge:** `tests/store_test.py` under CPython and inside Blender 4.5 and 5.1. It
  includes the precedence case: a store whose live and pinned entries alone exceed 2 MB keeps all
  of them, evicts every warm and cold entry, and reports the size. The test passes the live set
  **explicitly** as `live_gns`, because the store is not wired to the pass until PR-08. A second
  case saves the same over-limit store with `live_gns=None` and asserts that **nothing** is
  evicted.

### PR-05: Evaluator module and new word sources
- **Files/components:**
  - `openact/evaluator.py` (new; moved out of `engine.py`);
  - `openact/engine.py` (imports it);
  - `tests/evaluator_blender.py` (extended).
- **Depends on:** PR-02. **Parallel with:** PR-03, PR-04, PR-06.
- **Changes:** §10.1.
  - Move `eval_input`, `eval_output`, `_group_input`, `_group_output`, the handlers, `Ctx`,
    `find_uses` and the exceptions without behaviour change.
  - Add handlers for Menu Switch (both storage forms), Special Characters and String Length.
  - Add the per-pass memo, breadcrumb `Unresolvable` messages, rejection of field outputs, and a
    read-only path for linked libraries.
- **Tests to merge:**
  - `tests/evaluator_blender.py` on 4.5, 5.0 and 5.1: every existing case, plus Menu Switch:
    - socket value by name;
    - modifier int mapped through `id_properties_ui`;
    - `-1` falling back to the interface default;
    - deleted and reordered items;
    - a Menu inside a nested group;
    - String, Int and Float types.

    Also Special Characters, String Length, breadcrumbs and the memo.
  - e2e `build` and `reopen` stay green.

### PR-06: Model server v1 (server side; token checked when set)
- **Files/components:**
  - `openact/runtime/openact_server.py`;
  - `openact/server.py` (only support for `OPENACT_BACKEND=fake` and `OPENACT_SERVER_PYTHON`);
  - `tests/server_test.py` (new, CPython; starts the server with `--backend fake` on a free port).
- **Depends on:** PR-02. **Parallel with:** PR-03, PR-04, PR-05.
- **Changes:**
  - **Server** (§10.4): the `Backend` protocol with `SemIfTorchBackend` and `FakeBackend`
    (`--backend`, `--fake-script`, `--record <path.jsonl>` which appends every received request
    with its answers so tests can inspect exactly what was sent, and `--fake-load-seconds` /
    `--fake-load-fail` to simulate cold loads); the per-question LRU (65 536) wrapping
    `_decide_uncached()`; an immediate 503 `loading` / `error` instead of blocking (§10.4);
    `GET /v1/capabilities`; new `/health` counters (`questions`, `rows_scored`, `cache_hits`,
    `ms_per_row`); a 256-question limit; 422 responses naming the question; token plus `Origin` and
    `Host` checks (§16.1).
  - **Transitional seam, PR-06 only:** the server enforces the token only when it was launched
    with `OPENACT_TOKEN`. The client does not send one until PR-10, so the existing Ask/Pick e2e
    keeps working in this PR. `Origin` and `Host` checks are always on.
    - This open mode **does not survive PR-10**. From PR-10 on, the server **refuses to start**
      without `OPENACT_TOKEN`, except with `--backend fake` (tests only), as §10.4 and §16.1
      require.
  - Debias and tournament code are unchanged. `options.debias` is honoured **per request**, which
    the per-node toggle relies on (§10.3).
  - **`OPENACT_STATE_DIR`** (§5.2): `server.log`, token files and `logs/` are written there,
    defaulting to the runtime folder. Parallel build workers point it at a temporary folder, so the
    shared runtime stays read-only (§20.6).
- **Tests to merge:**
  - `tests/server_test.py`:
    - 403 with no token or a wrong token, with an `Origin` header, and with a foreign `Host`;
    - `cached_questions` on a repeat;
    - the capabilities fields;
    - 422 for 257 questions, and for an invalid question (naming it);
    - a 40-option tournament returns 40 probabilities that sum to 1;
    - with `--fake-load-seconds 5`, a POST during loading gets an immediate 503 `loading`, and a
      simulated failed load gives 503 `error`.
  - `server_test.py` also checks that a server launched with `OPENACT_TOKEN` rejects a missing
    token. In PR-06 only, one launched without it accepts requests; PR-10 replaces that check
    (see there).
  - e2e with the real model (the client sends no token yet).

### PR-07: Installer hardening, runtime probe, `locate_runtime`
- **Files/components:**
  - `openact/server.py` (`install()`);
  - `openact/runtime/requirements-cu128.lock`, the only lock (v1 builds NVIDIA/CUDA only);
  - `tools/make_locks.ps1` (runs `uv pip compile --generate-hashes`);
  - `tests/installer_blender.py` (new).
- **Depends on:** PR-06. **Parallel with:** PR-08 … PR-21 (PR-22 depends on it).
- **Changes:** §10.5, §16.2 and R3.
  - A **free-space pre-check** (at least 15 GB, `shutil.disk_usage`) before any download.
  - A **GPU/VRAM pre-check** (`nvidia-smi`). With no NVIDIA GPU the install is refused, because
    there is no CPU install path in v1. Below 10 GB of VRAM it warns (§10.5, §3.3).
  - uv always comes from the pinned download, hash-checked in `<runtime>/tools/`. A `uv` on PATH
    is ignored.
  - `runtime_installed()` stays a cheap file-system check: the venv's Python, the `home`
    directory and the model shards.
  - The `import torch, semif_phase1` probe runs **on the executor thread**, with the status
    "Checking the runtime…", and is cached in `<runtime>/probe.json` (§12.2).
  - `openact.locate_runtime` (moved here from PR-22 so it lives with the probe it runs): a folder
    picker that sets `runtime_dir` and runs the background probe.
  - SHA-256 checks: uv (`UV_SHA256[asset]`), and the SemIf archive downloaded at install time
    from the pinned GitHub commit (user decision; bundling may come later).
  - torch, SemIf and their dependencies installed with `--require-hashes` from
    `requirements-cu128.lock`.
  - The install-log header "OpenAct runtime install".
  - `install(dry_run=True)`, which logs every command without running it. The existing resumable
    steps are kept.
- **Tests to merge:**
  - `tests/installer_blender.py`:
    - the dry run prints the expected cu128 commands, runs in a **temporary runtime folder with
      mocked downloads**, and never touches the installed `JevNodes` runtime (§20.6);
    - a hash mismatch aborts with a clear status;
    - too little disk space (mocked `disk_usage`) refuses before downloading;
    - with `online_access` off, the install refuses;
    - a mocked machine without `nvidia-smi` is refused before downloading, and a mocked 8 GB GPU
      gets the warning;
    - a `uv` on PATH is not used;
    - the file-system check reports a legacy runtime whose venv `home` is missing, using a copied
      `pyvenv.cfg` pointing at a non-existent path;
    - the import probe runs off the main thread: `register()` returns in < 50 ms while a
      deliberately slow fake probe sleeps 5 s, and the second session reads `probe.json` without
      starting a subprocess.
  - Once, manually: a real install into a temporary runtime folder on this machine, logged in the
    PR description.

### PR-08: Inner builder module and interface sync
- **Files/components:**
  - `openact/inner.py` (new: `SocketDef`, `_Builder`, match chain, entry-level `lookup`, `build()`,
    `update_values()`, `create_inner()`, `sync_interface()`);
  - `openact/compat.py` (new: `structure_type()`, §11);
  - `openact/engine.py` (uses `inner` and `store`);
  - `openact/nodes.py` (`init()` and `copy()` go through `inner`);
  - `tests/interface_blender.py` (new).
- **Depends on:** PR-04, PR-05. **Parallel with:** PR-07.
- **Changes:**
  - Extract the builder without behaviour change.
  - `sync_interface()` diffs by `(in_out, name, socket_type)`, never clears the interface, orders
    sockets with `interface.move()`, and sets `structure_type` through `compat.py`: `SINGLE` for
    words on every version, `FIELD` for ID and factors on 5.0+ only (§6.0).
  - `SocketDef.hidden`: sockets declared hidden, such as `Key`, get `hide = True` on the node when
    they are created. Once a user reveals one with Ctrl H, it is never hidden again.
  - **Contract spike, done first in this PR and before PR-09 freezes the contracts.** Measure
    `inner.build()` and `update_values()` at the demo and worst shapes, and the plan-and-hash cost
    of a 4096-cell context on Blender 4.5 (Python 3.11) and 5.1 (Python 3.13). If a §3.2 or §8.4
    budget is missed, adjust the budget or the design **before** PR-09. The numbers are recorded
    in the PR description.
  - All locations are written in one batched `tree.nodes.foreach_set("location", …)`.
  - `entries` is stored as schema 2 through `store.py`; schema 1 is converted on read. The pass
    calls `store.save(node, store, live_gns)` with the `gn` tuples of its current contexts, so
    eviction never removes a live entry (§13.3). It is the **only** caller that passes a live set.
    Operators that edit entries save with the default `live_gns=None`, which evicts nothing.
  - `ensure_owned()` keeps its semantics.
- **Tests to merge:**
  - `tests/interface_blender.py`:
    - interface edits reach the custom group node and links survive;
    - `default_input='INDEX'` with `hide_value` gives `[0, 1, 2, 3]`;
    - Index Switch `MATERIAL` works;
    - `copy()` plus `ensure_owned()` gives every copy its own inner group, including when the
      whole tree is duplicated.
  - `tests/lookup_blender.py --bench` (created here with the build benchmark): `inner.build()`
    and `update_values()` within the §8.4 budgets for the Ask and Pick shapes.
  - Evaluator, `store_test` and every e2e phase.

### PR-09: Spec framework and cell pipeline model
- **Files/components:**
  - `openact/specs/__init__.py`, `specs/base.py`, `specs/ask.py`, `specs/pick.py`;
  - `openact/pipeline.py` (new);
  - `openact/engine.py`;
  - `tests/rules_math_test.py` (new; PR-18 and PR-19 extend it).
- **Depends on:** PR-08. **Parallel with:** PR-07.
- **Changes:**
  - `Spec` (§10.2), using `inner.SocketDef` (PR-08), plus the helpers `option_text`, `state_text`
    and `confidence()` (all three 4.e formulas).
  - `AskSpec` and `PickSpec` move into `specs/`. Ask's `mode` becomes `answer_type`, and Amount
    confidence switches to the σ formula.
  - **The `Key` word** (§6.0): every spec declares a hidden String `Key` as the last position of its
    fixed word tuple (§8.2). `Spec.plan()` never puts it into `state`, `instructions` or any
    `wkey`. A store test asserts that changing only `Key` leaves every `wkey` unchanged.
  - **The per-node *Balance Option Order* toggle** (user decision, §6.0, §10.3):
    - the `balance_order` BoolProperty on every node, seeded in `init()` from the `debias`
      preference;
    - it feeds the ask key, the cell keys and the fingerprint, never a words key.
  - `pipeline.py`: `Cell`, `Row`, `Job`, `TableShape`.
  - **The two key families (§5.4):** `wkey` and `words_key` (what the artist said), and
    `ask_key` and `cell_key` (how OpenAct asked).
  - **The validity rule:** reuse by `words_key`, carry rows over by `wkey`, label rows with an old
    ask key.
  - Per-cell `MEMORY` (LRU 100 000), `PENDING` and `FAILED`. Requests are still sent one per job on
    the existing executor.
  - `engine.run_pass()` works on Jobs (§5.3, steps 1–7), with the plan cache `PLANS`, fingerprints,
    per-node dirty flags, `LAST_SHOWN`, and `SAFETY_PASS` reduced to a fingerprint check (§10.6).
  - `openact.reask` forgets only that node's cell keys and re-asks every row with the current
    settings. It saves with `store.save(node, store)` (`live_gns=None`, no eviction, §13.3).
- **Tests to merge:**
  - evaluator;
  - `store_test`, including: a changed `PROMPT_VERSION` or `debias` changes the ask key but no
    words key;
  - `tests/rules_math_test.py` (confidence values for fixed distributions);
  - every e2e phase, which must still meet the prototype's value checks;
  - an e2e `reopen` run with a bumped `PROMPT_VERSION` (test switch) sends 0 requests.

### PR-10: Decision worker, priorities and chunking
- **Files/components:**
  - `openact/server.py`:
    - the `Worker` thread; the executor reduced to 2 threads;
    - **the client token**: `secrets.token_hex(32)` passed as `OPENACT_TOKEN`, the
      `X-OpenAct-Token` header in `http()`, and a status message on 403. This turns enforcement
      on **permanently**: `openact_server.py` now refuses to start without `OPENACT_TOKEN` unless
      `--backend fake` is given. `server_test.py` asserts this, replacing PR-06's open-mode
      check;
    - 503 `loading` / `error` handling (§10.5);
  - `openact/pipeline.py` (`submit`, `cancel`, `cached`, chunk builder, stale-while-refilling);
  - `openact/engine.py` (generation bump, render pause, `STATUS` record skeleton);
  - `tests/pipeline_blender.py` (new; fake backend).
- **Depends on:** PR-06, PR-09. **Parallel with:** PR-11.
- **Changes:** §5.5 and §10.3.
  - Priority classes P0, P1 and P2, round-robin within a class.
  - Chunks of at most 32 questions and 64 rows, sized from `/v1/capabilities`; one request in
    flight.
  - Cancellation by generation; a 503 puts the chunk back at the head of the queue.
  - 422 isolation, by the server's `question` id or else by bisecting the chunk.
  - No new chunk starts during a render job. Results reach the main thread through `server.post()`.
- **Tests to merge:**
  - `tests/pipeline_blender.py`:
    - the 403 path;
    - the chunk limits;
    - a P0 job overtakes a P2 fill;
    - cancellation drops queued chunks;
    - `cached_questions`;
    - 422 isolation;
    - stale-while-refilling;
    - the playback and render pauses, driven by the `playing` and `render_running` events that
      `tick()` publishes. The worker never touches `bpy`;
    - during a simulated 5 s cold load, a P0 job queued after a P2 chunk is answered first once
      the server is ready, and a 503 `error` marks cells `FAILED` without requeueing;
    - every request carries the token, and a server started by this Blender rejects requests
      without it;
    - **entry validity (§5.4):**
      - a template bump sends 0 requests;
      - **toggling a node's *Balance Option Order* sends 0 requests** and keeps its stored answers;
      - after a text edit only rows whose `wkey` changed are asked, and their request carries the
        node's current `options.debias`;
      - *Ask Again* re-asks with the node's current setting;
      - two nodes with different toggles in one scene send requests with their own `debias`
        values.

      All of these are checked through the fake backend's `--record` log;
    - **plan cache:** TIMING ≤ 5 ms per unchanged node and ≤ 50 ms per idle pass with a
      maximum-size Rules-like synthetic plan used by 16 contexts.
  - e2e stays green.

### PR-11: Table objects, per-element lookup, `k_default`, Pick fields
- **Files/components:**
  - `openact/tables.py` (new: `ensure_table()`, `write_rows()`);
  - `openact/inner.py` (the sampled lookup, base/count switches, materialised set);
  - `openact/specs/pick.py` (ID, Selection, Probability, Context);
  - `openact/engine.py` (rebuild or values-only, §8.8);
  - `tests/lookup_blender.py` (extended).
- **Depends on:** PR-09. **Parallel with:** PR-10.
- **Changes:** §8.1–§8.4, §8.8.
  - The **table object** `.OpenAct Table <uid>`: a mesh with `oa_*` POINT attributes, written by
    `write_rows()` (`foreach_set`, at most every 0.5 s during a fill).
  - Every per-element column is read through Object Info + Sample Index, with the §6.0 fallbacks.
  - **Materialised set:** live entries, then pinned, then warm up to 4 in total, with the hard cap
    `MATERIALISED_MAX = 16` (§8.1), confirmed by the PR-08 spike (§8.4).
  - **`k_default`** (§8.2): set only when exactly one context is live and the node is not in a hint
    or error state, otherwise 0.
  - `update_values()` touches only entry-level Index Switches.
  - Pick gains:
    - an `ID` input (Int, `default_input = 'INDEX'`, `hide_value`);
    - the `Probability` field, from the per-row `prob` column;
    - `Selection = (ID == Index) AND (Index ≥ 0)`;
    - the optional `Context` input.
- **Tests to merge:**
  - `tests/lookup_blender.py`, on synthetic tables:
    - invalid IDs give fallbacks;
    - 3 contexts through the match chain;
    - **a rebuild leaves outputs unchanged**, rebuilding from `entries` alone;
    - a values-only update keeps the node count;
    - 10 successive Question edits keep at most 4 materialised entries, with at most one rebuild
      per edit; switching back to a stored prompt sends 0 requests;
    - `write_rows()` grow → shrink → grow keeps every column correct, re-creating attributes
      after `clear_geometry()` (§8.5);
    - **joined key** (§8.2):
      - items are joined most recently linked first, and Python's reverse linking reproduces the
        tuple order;
      - an absent position fed by a constant `""` gives the same arity as the stored
        `"\x1f".join(gn)`, so the entry matches;
      - a word containing `"\x1f"` switches that node to per-word Compares and still matches;
    - **no-match fallback:**
      - two live contexts, with A's words unmatched: A shows neutral values, never B's answers;
      - one live context: it shows its own last answers;
      - a cleared Question (a hint): neutral outputs;
    - `--bench`: per-element lookup ≤ 5 ms above baseline at 1M points; `write_rows()` ≤ 5 ms;
      `inner.build()` within the §8.4 budget.
  - The interface suite.
  - e2e: on the prototype scene, the Probability of the picked item equals its stored top
    probability.

### PR-12: Table-object lifecycle: copies, append, garbage, save/reload
- **Files/components:**
  - `openact/tables.py` (`copy_for()`, garbage collection);
  - `openact/engine.py` (`ensure_owned()` copies tables; Pick releases its table when Selection
    and Probability are unlinked);
  - `tests/lookup_blender.py`, `tests/interface_blender.py` and `tests/e2e_blender.py`, extended.
- **Depends on:** PR-11. **Parallel with:** PR-13, PR-14, PR-15.
- **Changes:** §8.5.
  - Copies use `copy_for()`. Orphaned tables are removed by garbage collection.
  - Table objects are linked to no collection.
  - A row-count change needs no graph edit.
- **Tests to merge:**
  - **lookup:** 4096-row and 16 × 4096-row tables stay within the `--bench` budgets.
  - **interface** (4.5, 5.0 and 5.1):
    - a copied node gets its own table object;
    - appending into a new file brings the table along, with no extra step;
    - deleting the node removes its table;
    - an unlinked table object survives save/reload and *Purge Unused Data*, and renders with F12
      and with `blender -b -f`.
  - **e2e:** `noaddon` with a 100-item text Pick whose Probability field is linked, so it uses a
    table object.

### PR-13: List sources, path resolution, Import CSV hand-off, trust rule and per-item planning
- **Files/components:**
  - `openact/lists.py` (`resolve_items()`, path resolution, `csv_source()`);
  - `openact/evaluator.py` (`trace_csv_source()`, §10.1);
  - `openact/specs/base.py` (`SocketDef` companions: every CSV-capable String input declares its
    `<Input> CSV` Geometry socket, §6.0; later node PRs inherit it);
  - `openact/csvfmt.py` (the document cache keyed by path identity);
  - `openact/geometry.py` (new: `attribute_names()`, `read_attributes()`);
  - `openact/specs/base.py` (`per_item_cells()`, the per-item planning helper shared by Match and
    Classify);
  - `openact/specs/pick.py` and `openact/nodes.py` (the `list_source` dropdown and dynamic
    sockets);
  - `openact/engine.py` (`TRUST`); `openact/ops.py` (`openact.allow_file`);
    `openact/prefs.py` (`trusted_folders`); `openact/evaluator.py` (Import Text obeys the trust
    rule);
  - `tests/pipeline_blender.py` and `tests/lookup_blender.py`, extended.
- **Depends on:** PR-03, PR-10, PR-11. **Parallel with:** PR-12.
- **Changes:** the §6.0 list sources, §7.7 path resolution and §16.1 trust.
  - Text or CSV, with Items CSV payload outputs baked as entry-level values.
  - Path resolution: `bpy.path.abspath(library=…)`, a hint for relative paths in unsaved files,
    and one `CsvDoc` per file identity.
  - **Import CSV hand-off** (user decision, §6.0, §7.7):
    - the companion `<Input> CSV` sockets;
    - `trace_csv_source()` through Reroutes, nested groups and Geometry Switch / Index Switch /
      Menu Switch;
    - the Import CSV node's Path and Delimiter evaluated statically;
    - the link winning over a typed path, with a warning;
    - the two error messages.
  - **Trust rule:** in an untrusted file, reads are confined to the .blend's folder tree, and
    blocked text is never read or stored. `openact.allow_file` lifts the restriction.
  - Collection, in natsort order.
  - Materials: empty slots keep their index but are never offered. The `Material` output is the
    entry's picked material, by name, through Index Switch `MATERIAL` on `k` (§8.6).
  - Attributes: names from the object's **original data**, excluding "." names and names this
    tree writes (§6.0).
  - The shape-agreement check across contexts (§6.0).
  - Switching `list_source` syncs the sockets.
- **Tests to merge:**
  - **pipeline (fake backend):**
    - a collection item's index equals its Collection Info order;
    - Materials with an empty slot;
    - a **Collection-source Pick matches its stored entry**: the absent Items position is joined
      as a constant `""` (§8.2);
    - **the `Key` remedy:** two contexts that differ only by collection give "only one answer can
      show" until each gets a different `Key`. Then both are distinguishable, with 0 extra
      requests, because `Key` is excluded from `wkey`s (§6.0);
    - **Import CSV hand-off:**
      - a **direct link** from an Import CSV node reads its file;
      - a **path exposed on the modifier** feeding the Import CSV node's Path resolves per context;
      - **precedence**: with both a typed path and a link, the link wins and the warning appears;
      - Reroutes, a nested group and a Geometry Menu Switch pass through;
      - a `;` Delimiter on the Import CSV node is respected;
      - **errors**: a non-Import-CSV source (e.g. Grid) gives "connect an Import CSV node", a
        Transform Geometry in between gives "connect the Import CSV node directly", and an empty
        Import CSV Path gives a hint;
      - an edit to the file is picked up although Blender's own Import CSV would still show the
        old rows;
    - the Attributes name list comes from original data. A Pick (Attributes) whose Item feeds a
      Store Named Attribute settles in one pass with one request;
    - Items CSV payload outputs carry the CSV values;
    - **trust:** an untrusted file with an absolute Import Text or CSV path outside its folder
      reads and stores nothing, while a CSV next to it is read; after `openact.allow_file`, both
      are read;
    - two contexts on different objects that share a tree and differ only by collection give the
      "only one answer can show" error.
  - **lookup:** the per-entry Material output.
  - **e2e:** a collection Pick and its Instance output.

### PR-14: CSV file watcher
- **Files/components:**
  - `openact/engine.py` (`WATCH`; marking dependent nodes dirty for the plan cache);
  - `tests/csv_probe_blender.py` (new);
  - `tests/pipeline_blender.py`.
- **Depends on:** PR-13 (path resolution and the document cache; PR-10 through PR-13).
  **Parallel with:** PR-12, PR-15.
- **Changes:** §7.8.
  - `stat` polling at 2 Hz, round-robin over 32 files.
  - A text change runs a pass; a numeric-only change re-bakes values.
  - A missing file keeps the last good table.
  - Problems go into the status record.
  - Import CSV nodes reading the same file are flagged for the sidebar (PR-22).
- **Tests to merge:**
  - `tests/csv_probe_blender.py`:
    - Import CSV parity per fixture: point count, kept columns, INT/FLOAT types;
    - writes `tests/fixtures/import_csv_behaviour.json`;
    - re-asserts the path-cache fact;
    - the equivalent-spelling probe, follow-up §21 F1. The helper `openact.refresh_import_csv`
      ships only if the trick re-reads on both 4.5 and 5.1.
  - pipeline:
    - a numeric-only edit sends 0 requests and updates the payload;
    - a text edit re-asks only the affected cells;
    - a missing file keeps the table.

### PR-15: Answer sets and Ask's Answers input
- **Files/components:**
  - `openact/specs/base.py` (`answer_set()`);
  - `openact/specs/ask.py` (the `Answers` input);
  - `tests/pipeline_blender.py`, `tests/csv_test.py` and `tests/e2e_blender.py`, extended.
- **Depends on:** PR-13, which provides CSV path resolution, the document cache and the trust rule
  (PR-03 and PR-10 come through PR-13). **Parallel with:** PR-12, PR-14.
- **Changes:** §6.1, §7.3 and 4.d.
  - Inline and CSV answer sets for Yes/No (`yes`/`no` whens → noul criteria), Amount (2–16
    ordered levels) and One-of.
  - A single-option set is decided by code and never sent.
- **Tests to merge:**
  - **pipeline**, reading the fake backend's `--record` log (PR-06):
    - the criteria sent match the answer set;
    - a single-option set sends no request;
    - custom levels map to `score / (L − 1)`.
  - **csv_test:** the answer-set fixtures.
  - **e2e:** Ask in Amount mode with a custom 5-level snow rubric; snowy mountain > tropical beach.

### PR-16: Match node
- **Files/components:**
  - `openact/specs/match.py` (new);
  - `openact/nodes.py` (`GeometryNodeOpenActMatch`; its entry in the Add menu's "Each item" group,
    created by whichever of PR-16 or PR-17 merges first);
  - `openact/inner.py` (the Threshold compare);
  - `tests/lookup_blender.py`, `tests/pipeline_blender.py` and `tests/e2e_blender.py`.
- **Depends on:** PR-12, PR-14, PR-15. **Parallel with:** PR-17, PR-18, PR-19, PR-20.
- **Changes:** §6.3.
  - Sockets: Question, Context, the list source, Answers, ID and Threshold. The Overrides socket
    arrives with PR-21.
  - Outputs: Probability, `Selection = valid AND (p ≥ Threshold)`, Confidence.
  - One question per item, using the §10.2 template, with one state per job.
  - Empty material slots become fallback rows.
- **Tests to merge:**
  - **lookup:** a synthetic Match table, with Threshold driven by a field.
  - **pipeline:** 40 items go out as 2 chunks sharing one state, with the exact template text.
  - **e2e:** spot checks: pan = yes and sofa = no for "Would you find it in a kitchen?"; in Amount
    mode, "How heavy is it?" ranks anvil above feather.

### PR-17: Classify node
- **Files/components:**
  - `openact/specs/classify.py` (new);
  - `openact/nodes.py` (`GeometryNodeOpenActClassify`);
  - `openact/inner.py` (label plus baked payload columns per row);
  - tests as for PR-16.
- **Depends on:** PR-12, PR-14, PR-15. Classify no longer waits for Match: the shared per-item
  planning helper now lands in PR-13. **Parallel with:** PR-16, PR-18, PR-19, PR-20.
- **Changes:** §6.4.
  - Labels is required.
  - The Labels CSV payload is denormalised per row (§8.1).
  - More than 16 labels run as a tournament, and the 8192-row estimate is enforced before anything
    is sent (§5.4).
- **Tests to merge:**
  - **lookup:** Label and payload per element; an invalid ID gives −1 and the fallbacks.
  - **pipeline:** 255 labels × 250 items (34 rows × 250 = 8 500 > 8 192) is refused by the
    row-estimate limit before sending; the tournament cost estimate
    for 20 labels is correct.
  - **e2e:** material families: "steel" → metal, "oak planks" → wood.

### PR-18: Rules I: factors, cells, `if`, One-of
- **Files/components:**
  - `openact/specs/rules.py` (new: factors, cell enumeration, `parse_if`, allowed sets, cells
    decided by code, the §9.5 templates, limits);
  - `openact/nodes.py` (`GeometryNodeOpenActRules`);
  - `openact/inner.py` (dynamic factor sockets, the §9.2 level-index graphs, the §9.3 cell index,
    Outcome/Confidence plus baked payload);
  - `tests/csv_test.py`, `tests/rules_math_test.py`, `tests/lookup_blender.py`,
    `tests/pipeline_blender.py` and `tests/e2e_blender.py`.
- **Depends on:** PR-12, PR-14, PR-15. **Parallel with:** PR-16, PR-17, PR-20.
- **Changes:** §6.5 (One-of mode only) and §9.1–§9.5. Yes/No, Amount, Dither and Seed arrive with
  PR-19.
- **Tests to merge:**
  - **csv_test:** the `if` grammar and the factor validation messages.
  - **rules_math_test:** `product` order equals `ravel`; allowed sets; cells decided by code.
  - **lookup:**
    - the cell index for every combination, including a non-contiguous category map in the table
      object (§9.2);
    - **Game of Life in a Simulation Zone**: 32 frames against a Python B3/S23 reference, both
      using the toroidal wrap of §6.5;
    - a Repeat Zone variant;
    - **a rebuild with every CSV file deleted** leaves outputs unchanged (`shape` suffices,
      §14.2);
    - **shape agreement:** two contexts whose Outcomes CSVs share a payload set work; conflicting
      Factors CSVs give the error and keep the last good table, including two Factors CSVs that
      differ **only in their band edges** (§6.0).
  - **pipeline:**
    - restricted criteria are sent;
    - cells decided by code send nothing;
    - a cell with no allowed outcome becomes an error row.
  - **e2e acceptance:** Conway, HighLife, Seeds and Day & Night, each **18/18 exact with 0
    unsure**.

### PR-19: Rules II: interpolation and dithering
- **Files/components:**
  - `openact/specs/rules.py` (Yes/No and Amount questions, the `ok` column);
  - `openact/inner.py` (corners, weights and the renormalised blend; the Dither/Seed graph; the
    nearest-band warning);
  - `tests/rules_math_test.py`, `tests/lookup_blender.py` and `tests/e2e_blender.py`.
- **Depends on:** PR-18. **Parallel with:** PR-16, PR-17, PR-20.
- **Changes:** §9.6 and §9.7.
  - `answer_type` YES_NO and AMOUNT, interpolating over at most 3 Amount factors.
  - Dither and Seed sockets, present only in One-of mode.
- **Tests to merge:**
  - **rules_math_test:** the interpolation reference; dithering probabilities within ±1 % of
    100 000 samples.
  - **lookup:**
    - interpolation within 1e−5 of the reference;
    - renormalisation when corners are pending;
    - the dither fraction within ±2 %;
    - 8 corners (24 lookups) stay within the graph-size and `inner.build()` budgets (§8.4);
    - `--bench` for 8 corners stays ≤ 5 ms above the no-lookup baseline per 1M elements (§3.2).
  - **e2e:** snow (Amount) is monotonic along a transect; biome (One-of) has mean confidence ≥
    0.55 and uses all 7 outcomes.

### PR-20: Dials node
- **Files/components:**
  - `openact/specs/dials.py` (new);
  - `openact/nodes.py` (`GeometryNodeOpenActDials`);
  - `openact/tables.py` (the Dials **one-row-per-entry table object**: each setting's value,
    `min_<s>`, `max_<s>` and `conf`, §8.4);
  - `openact/inner.py` (per-setting `Sample Index` at row `k − 1` behind a `Switch` on `k > 0`;
    Map Range whose *To Min*/*To Max* are sampled from the same row);
  - `tests/csv_test.py`, `tests/pipeline_blender.py`, `tests/lookup_blender.py` and
    `tests/e2e_blender.py`.
- **Depends on:** PR-14, PR-15. **Parallel with:** PR-16 … PR-19.
- **Changes:** §6.6, §8.6.
  - Settings, inline (Yes/No and Amount only) or from a CSV, at most 32 either way (§7.5;
    `csvfmt.MAX_SETTINGS`, lowered from 64 by the PR-08 spike).
  - `min`/`max` are per-entry data stored in `shape.settings` (§14.2). Contexts may use different
    ranges, as long as the setting names and types agree (§6.0).
  - Dynamic outputs, synced by name and type.
  - Confidence is the lowest among the settings.
  - All settings go in one P0 request.
  - A warning when a setting is renamed or retyped.
- **Tests to merge:**
  - **csv_test:** the settings parser, including the inline form; 32 settings are accepted and 33
    give the limit error, inline and from a CSV.
  - **pipeline:**
    - 5 settings travel in 1 request;
    - editing `min`/`max` is a values-only update with 0 requests;
    - renaming a setting warns.
  - **lookup:**
    - two contexts with different `min`/`max` each get their own mapped values;
    - different setting names give the shape-agreement error;
    - a Dials output feeds a **single-value socket** (Grid *Vertices X*) through a valid link;
    - a new prompt costs its one rebuild, because the materialised set changes (§8.8). **After that
      rebuild**, the answers arriving update the Dials values through `tables.write_rows()`, and the
      node count is unchanged: the answers themselves cause no rebuild.
  - **e2e (weather):** a new prompt is answered in ≤ 0.6 s; switching back is instant; the storm
    prompt gives Rain = true.

### PR-21: Overrides CSV: locks and export
- **Files/components:**
  - `openact/specs/base.py` (lock resolution);
  - `openact/specs/{ask,pick,match,classify,rules,dials}.py` (the `Overrides` socket and per-node
    key columns: `question` for Ask, `prompt` for Pick, §7.6);
  - `openact/ops.py` (`openact.write_overrides`, with a `context_index` property that defaults to
    the first live context; PR-22's sidebar selector passes the chosen one);
  - `openact/engine.py` (the watcher also covers Overrides files);
  - `tests/csv_test.py`, `tests/pipeline_blender.py` and `tests/lookup_blender.py`.
- **Depends on:** PR-16, PR-17, PR-19, PR-20. **Parallel with:** none.
- **Changes:** §7.6.
  - A valid lock: source `lock`, c = 1, never asked, and part of the words key (through
    `lock_digest`).
  - A lock whose key matches no row is warned about as orphaned.
  - An invalid answer becomes an error on that row, and the model's answer stays.
  - Dials amounts are written in output units.
  - Export: a proposed target that fills an empty socket; a merge that keeps locked rows and user
    columns; an atomic `os.replace`; the `PermissionError` message.
- **Tests to merge:**
  - **csv_test:** the `parse_overrides` fixtures.
  - **pipeline:**
    - locks send 0 requests;
    - editing a lock is a values-only update;
    - the orphaned-lock warning;
    - the invalid-answer error;
    - the export merge keeps locked rows and user columns;
    - a locked target file reports the `PermissionError` message.
  - **lookup:** a locked corner takes part in interpolation.

### PR-22: Node status, sidebar inspection, Show Question
- **Files/components:**
  - `openact/engine.py` (the `NodeStatus` record);
  - `openact/nodes.py` (the node body line; the Add menu groups);
  - `openact/ui.py`: the Model box; the Active node panel with contexts and a paged table view; the
    Files panel with the Import CSV note; the problems list; banners, including the trust banner
    (*Allow this file* / *Trust this folder*) and the runtime banner ("Checking the runtime…",
    probe failures, *Locate Existing Runtime…*). Pure data functions (`status_rows()`,
    `table_page()`) are kept separate so they can be tested headless;
  - `openact/ops.py`:
    - `openact.open_file`, used by the Files panel and problems list, moved here from PR-23;
    - `openact.show_question`: the exact request of the selected row, built from the cached plan,
      with no model call (§17.2);
    - the *Locate Existing Runtime…* button calls `openact.locate_runtime` from PR-07;
  - `tests/pipeline_blender.py`.
- **Depends on:** PR-21, and PR-07 for the runtime banner and `locate_runtime`. **Parallel with:**
  none.
- **Changes:** §17.1–§17.3 and §7.10.
  - Icons: `CHECKMARK`, `SORTTIME`, `INFO`, `PAUSE`, `ERROR` (warnings) and `CANCEL` (errors).
  - ETA from `ms_per_row`.
  - Banners for: model not installed, `jev_nodes` enabled, a port conflict, the Import CSV note, and
    a hook for PR-24's legacy-node banner.
  - The table page number lives in a `WindowManager` property.
- **Tests to merge:**
  - **pipeline**, over a fake fill:
    - waiting → filling *n/N* with an ETA → complete;
    - the warning and error lines;
    - `table_page()` rows, icons and paging;
    - the banner conditions;
    - `show_question` returns exactly the `state`, `instructions` and options that the fake
      backend's `--record` log shows for that row.
  - A manual screenshot checklist in the PR, on 4.5 and 5.1.

### PR-23: Templates, diagnostics, logs, alerts, playback prep, unpin, server sharing
- **Files/components:**
  - `openact/templates/` (`life_factors.csv`, `life_outcomes.csv`, `labels.csv`, `items.csv`,
    `weather_dials.csv`);
  - `openact/ops.py` (`openact.new_csv_from_template`, `openact.copy_diagnostics`;
    `openact.open_file` already landed in PR-22);
  - `openact/prefs.py` (`log_requests`, `unsure_below`);
  - `openact/server.py`:
    - the JSONL request log and crash-loop protection;
    - **server sharing**: `<runtime>/server-<port>.token` (mode 0600 on POSIX), and a second Blender
      attaching through it (§10.5). Moved here from PR-10 to keep PR-10 within one session;
  - `openact/engine.py` (stalled-fill and slow-server alerts; **Prepare for Playback** and entry
    pinning, §5.5 and §8.1);
  - `openact/evaluator.py` (the **read set** and value overrides used by the F-curve path, §5.5);
  - `openact/ops.py` (`openact.prepare_playback`, modal only for the driver fallback;
    `openact.unpin`, per entry or all). Both save with `live_gns=None` (no eviction, §13.3), and
    the next pass applies the limits;
  - `openact/ui.py` (the *Unpin* / *Unpin All* buttons);
  - `openact/blender_manifest.toml` (the `clipboard` permission, §13.5);
  - `tests/pipeline_blender.py`.
- **Depends on:** PR-22. **Parallel with:** PR-24.
- **Changes:** §7.10, §16.3, §17.4–§17.6, plus the `clipboard` permission and Prepare for
  Playback (§5.5):
  - the F-curve path through `animation_data.action` + `action_slot` and
    `anim_utils.action_get_channelbag_for_slot()` (4.5–5.1);
  - the `frame_set` fallback (modal, progress, Esc) for drivers, NLA tracks, tweak mode and
    blended actions;
  - the current frame restored afterwards;
  - pins saved as `pinned: true` with `frames`, at most 12, and live plus pinned at most 16;
  - refusal on overflow with nothing asked;
  - `openact.unpin`.
- **Tests to merge:**
  - `blender --command extension validate` passes with the `clipboard` permission.
  - pipeline:
    - a keyframed Menu Switch cycling through three Outcomes CSVs: after Prepare for Playback, 0
      requests and no fallback outputs during playback;
    - over a **250-frame range**, the F-curve path finishes in < 1 s with no scene evaluation;
    - a driven input uses the modal fallback, which can be cancelled with nothing pinned, and the
      current frame is restored;
    - an **NLA-animated** input (a strip on an NLA track) takes the fallback path, and pins the
      right word tuples;
    - on 4.5 **and** 5.1, a keyframed Int modifier input is read through the channelbag API
      (`Action.fcurves` is absent on 5.x), and `evaluate()` returns the keyed values;
    - 13 distinct word sets are refused with the limit message, and nothing is asked;
    - pins survive save and reload (`pinned: true`), and `openact.unpin` removes them;
    - **pin order**: after pinning 12 word sets, adding objects until live plus pinned exceeds 16
      makes the pins whose tuples start latest in the frame range go inactive, the same ones on
      every run;
    - pins whose total size exceeds 2 MB are kept, the size note appears, and Prepare for Playback
      showed the projected size first (§14.7).
  - `tests/support_blender.py` (new, so it doesn't meet the playback work in pipeline):
    - two Blender processes share one server through the token file, and the second attaches
      without loading a second model; its Stop Model only detaches;
    - a template copy never overwrites (it creates `name 2.csv`) and fills an empty, unlinked
      socket;
    - the diagnostics text has versions and counters but no prompt or CSV text;
    - the JSONL log appears only when `log_requests` is on;
    - 3 fake-server exits within 10 minutes pause auto-start;
    - a stalling fake script raises the stalled-fill alert, time-scaled in the test;
    - a fake `ms_per_row` raises the slow-server note.

### PR-24: Migration from jev_nodes
- **Files/components:**
  - `openact/nodes.py`: legacy aliases `GeometryNodeJevAsk` / `GeometryNodeJevPick`, frozen, with
    `poll()` False, and not registered while `jev_nodes` is enabled;
  - `openact/migrate.py` (new: `find_legacy()`, `upgrade_all()`). It saves converted entries with
    `live_gns=None` (no eviction, §13.3);
  - `openact/ops.py` (`openact.upgrade_legacy`, one undo step, with a report);
  - `openact/ui.py` (the legacy and `jev_nodes` banners);
  - `openact/engine.py` (frozen legacy nodes in `iter_nodes()`; the `load_post` INFO report);
  - `openact/prefs.py` (a one-time import of `jev_nodes` preferences, §12.1–§12.2);
  - `openact/lists.py` (`parse_inline(mode="v0")` and the `items_parse_v0` node flag, §12.3);
  - `tests/migrate_blender.py` (new).
- **Depends on:** PR-22. **Parallel with:** PR-23.
- **Changes:** §12.2–§12.4, including the six upgrade steps and keeping answers through the schema
  1 → 2 conversion.
  - The conversion computes v1 `words_key`s with fixed-length word tuples (§8.2, §14.4), so the
    upgraded nodes' live plans reuse the converted entries.
  - Pick keeps the prototype's precedence (a set Collection over the Items text, §12.3) and legacy
    item parsing.
- **Tests to merge:**
  - `tests/migrate_blender.py` on 4.5 and 5.1: a v0.1.0 file built with the untouched prototype
    is upgraded, then:
    - outputs are identical;
    - 0 requests are sent;
    - inner groups are renamed and owned;
    - links are kept;
    - undo restores the old nodes.
  - `store_test` and e2e `reopen`.

### PR-25: Demo scenes
- **Files/components:**
  - `demos/build_demos.py`;
  - `demos/*.csv`, copied from the fixtures and templates;
  - a "Demos" section in the README.
- **Depends on:** PR-23. **Parallel with:** none.
- **Changes:** the six scenes of §18.4:
  1. Life from a CSV, with the Menu Switch rule-set dropdown and one locked cell;
  2. Biome;
  3. Clutter;
  4. Material families;
  5. Weather dials;
  6. Ask and Pick, rebuilt directly with OpenAct nodes, so PR-25 does not depend on the migration
     PR-24.

  `--model` fills the tables and runs each acceptance check. The saved files carry their answers.
- **Tests to merge:**
  - `blender -b --factory-startup --python demos/build_demos.py -- <tmp> --model` passes every
    acceptance check in §18.4.
  - After the build, **every demo table is complete**: no `pending` or `stale` rows. The "0
    requests" checks therefore cannot pass trivially on empty tables.
  - Each saved demo opens and evaluates on 4.5 **without** the add-on, with identical evaluated
    attributes.

### PR-26: Release 1.0.0
- **Files/components:**
  - `README.md` (complete, with a changelog);
  - `openact/blender_manifest.toml` and `bl_info` (`1.0.0`);
  - `dist/openact-1.0.0.zip`.
- **Depends on:** PR-07, PR-24, PR-25. **Parallel with:** none.
- **Changes:** the §20.5 checklist.
  - **The README covers:**
    - installation and the reuse of `JevNodes`;
    - the six nodes;
    - the CSV formats and templates, and both ways to give a node its CSV (a path, or a plugged-in
      Import CSV node);
    - the limitations: the path cache for the artist's own Import CSV nodes, 4.5+ only, NVIDIA/CUDA
      only;
    - a **"Future work"** section: a CPU backend (llama.cpp), and a possible high-accuracy model
      (SemIf's 27B "bridge"). Neither is built (user decisions);
    - credits and licences, and the non-affiliation line.
  - **The release:** a GitHub release created with `gh release create` in
    `nicolaiprodromov/openact`. The repository stays private until the user decides to make it
    public. A Blender Extensions submission is considered after 1.0.
- **Tests to merge:**
  - `tests/run_all.ps1 -Model -Versions 4.5,5.0,5.1`, every suite green;
  - `blender --command extension validate` and `build`;
  - demos rebuilt with `--model`;
  - the GitHub release exists with the zip attached (`gh release view v1.0.0`).
- **This PR completes OpenAct v1.0.**

---

**Future work (not in this plan).**
- **A CPU backend (llama.cpp, GGUF Q4_K_M, about 3 GB).** It would plug in behind the same
  `Backend` interface and `/v1/capabilities`. It would need its own installer profile, and would
  have to pass the same acceptance suites before being offered.
- **A possible high-accuracy model** (SemIf's 27B "bridge"): a README mention only.
- **Apple Silicon** is not planned beyond a one-line mention.

None of these has a PR here (user decisions).

**Total: 26 PRs.** PR-01 to PR-26 lead up to and include the v1.0.0 release.