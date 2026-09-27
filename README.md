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

## Credits

SemIf / OpenJev by TheoLeeCJ (MIT). Qwen3.5-4B by the Qwen team (Apache-2.0). PyTorch, Hugging Face
Transformers. The add-on is GPL-3.0-or-later. Not affiliated with TypeSafe or Jev.
