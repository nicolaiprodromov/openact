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
blender -b --factory-startup --python tests/evaluator_blender.py
blender -b --factory-startup --python tests/legacy_blender.py
blender -b --factory-startup --python tests/e2e_blender.py -- build   <file.blend> [port] [gpu]
blender -b --factory-startup --python tests/e2e_blender.py -- noaddon <file.blend>
blender -b --factory-startup --python tests/e2e_blender.py -- reopen  <file.blend>
```

The e2e phases drive the add-on with the real model.

## Credits

SemIf / OpenJev by TheoLeeCJ (MIT). Qwen3.5-4B by the Qwen team (Apache-2.0). PyTorch, Hugging Face
Transformers. The add-on is GPL-3.0-or-later. Not affiliated with TypeSafe or Jev.
