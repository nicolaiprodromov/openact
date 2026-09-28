# OpenAct

Decision nodes for Geometry Nodes, run by a local model.

> **Status: in development.** This is the start of OpenAct v1 (design: [docs/DESIGN.md](docs/DESIGN.md)).
> For now the add-on is the renamed `jev_nodes` v0.1.0 prototype with its two nodes, Ask and Pick.

| Node | In | Out |
|---|---|---|
| **Ask** | a question (+ optional context) | **Answer** 0-1, **Yes** (bool), **Confidence** |
| **Pick** | a prompt + a list of items (or a collection) | **Index**, **Item** (string), **Confidence**, **Instance** |

The model never generates text. It reads the probability of each allowed answer from one forward
pass. That is the method of **SemIf** (formerly OpenJev, MIT), running **Qwen3.5-4B** (Apache-2.0)
on your GPU. Nothing leaves your computer.

## Requirements

- Blender 4.5 or newer.
- An NVIDIA GPU with about 9 GB of free VRAM.

## Known issues

- **Blender 4.5 and 5.0 (Python 3.11):** Blender can crash when an operator runs under a context
  override while one of OpenAct's background threads briefly runs Python (a Blender bug). OpenAct
  makes no such call, but Blender does (the node editor's viewer shortcuts Ctrl+1…9 and 1…9, Quick
  Effects), and so may other add-ons. It is rare. Blender 5.1 is not affected.

## Install

1. Build the extension: `blender --command extension build --source-dir openact --output-dir dist`.
2. In Blender: *Edit > Preferences > Get Extensions > Install from Disk...* and pick
   `dist/openact-0.1.0.zip`.
3. Open a Geometry Node editor, press **N** and go to the **OpenAct** tab.
   - If it says *Model not installed*, click **Install Model**. It downloads about 12 GB (Python,
     PyTorch, SemIf, the model) into `%LOCALAPPDATA%\OpenAct`, one time.
   - A runtime that Jev Nodes already installed in `%LOCALAPPDATA%\JevNodes` is found and used in
     place, so nothing is downloaded.
4. Add nodes with **Shift A > OpenAct**.

## Repository

- `openact/`: the add-on (extension) package.
- `tests/`: headless Blender tests.
- `demos/`: the demo scenes' build script and CSV files.
- `docs/DESIGN.md`: the v1 design.

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

## Credits

SemIf / OpenJev by TheoLeeCJ (MIT). Qwen3.5-4B by the Qwen team (Apache-2.0). PyTorch, Hugging Face
Transformers. The add-on is GPL-3.0-or-later. Not affiliated with TypeSafe or Jev.
