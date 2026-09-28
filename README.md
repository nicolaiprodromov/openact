<h1 align="center">OpenAct</h1>

<h3 align="center">Judgement nodes for Blender Geometry Nodes</h3>

<p align="center">One forward pass. No text. Just decisions.</p>

<p align="center">
  <img src="docs/assets/banner.svg" width="100%" alt="A Rules node in the Geometry Nodes editor, asked 'Which plant grows here?', reads the temperature, moisture and altitude of every point of a terrain and decides which plant grows there: pines below the mountain's rock and snow, birches on its cold slopes, heather on the cold dry hills, olives on mild dry ground, oaks in the mild wet lowland, palms in the hot wet lowland and cacti in the desert.">
</p>

<p align="center">
  <b>Six Geometry Nodes that make judgements:</b> Ask, Pick, Dials, Match, Classify and Rules.<br>
  Words go in: a question, a prompt, a list or a table of rules. Decisions come out:<br>
  numbers, selections, picks and labels, per element when you need them.<br>
  A local model scores every allowed answer in one forward pass. It never writes text.
</p>


<p align="center">
  <a href="https://www.blender.org/download/"><img alt="Blender 4.5+" src="https://img.shields.io/badge/Blender-4.5%2B-E87D0D?style=flat-square&logo=blender&logoColor=white&labelColor=1B1F27"></a>
  <a href="CHANGELOG.md"><img alt="Version 1.0.0" src="https://img.shields.io/badge/version-1.0.0-34E0A1?style=flat-square&labelColor=1B1F27"></a>
  <a href="LICENSE"><img alt="Licence: GPL-3.0-or-later" src="https://img.shields.io/badge/licence-GPL--3.0--or--later-8B8FFF?style=flat-square&labelColor=1B1F27"></a>
  <a href="https://github.com/TheoLeeCJ/SemIf-OpenJev"><img alt="SemIf on Qwen3.5-4B" src="https://img.shields.io/badge/SemIf-Qwen3.5--4B-22C3EE?style=flat-square&logo=huggingface&logoColor=white&labelColor=1B1F27"></a>
  <a href="docs/GUIDE.md#privacy"><img alt="Runs 100% locally" src="https://img.shields.io/badge/runs-100%25%20locally-0B9E70?style=flat-square&labelColor=1B1F27"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#the-six-nodes">The nodes</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="docs/GUIDE.md">User guide</a> ·
  <a href="#roadmap">Roadmap</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

<br>

## Why OpenAct

- **Fast.** A chat model answers by writing text, one token at a time. OpenAct never writes: it scores every allowed
  answer in a single forward pass, and batches many questions into one request. On an RTX 3090, a new answer shows up
  in 0.2 to 0.6 seconds, and 216 yes/no cells fill in under 9 seconds. Reading the answers back costs about a
  millisecond per million elements.
- **Decisions, not text.** Every result is a number, a switch, an index or a label, ready to wire into your tree,
  with nothing to parse.
- **Runs on your GPU.** Qwen3.5-4B runs on your own NVIDIA card. After the one-time install, OpenAct talks only to
  `127.0.0.1`: nothing leaves your computer.
- **Asks only when words change.** Never during evaluation, playback or rendering. Answers arrive in the background,
  and only the rows whose words changed are asked again.
- **Saved in the .blend.** Answers live in the file, so it opens, plays and renders without the model, and even
  without the add-on. Render farms need nothing extra.
- **Written in CSV.** Keep rules, labels and settings in CSV files next to the .blend. Numbers beside labels and
  outcomes become output sockets, and saved edits show up while Blender runs.
- **You have the last word.** Every answer comes with a confidence, and unsure ones are flagged. Lock any answer in an
  Overrides CSV, and it is never asked again.

## Quick start

**You need** Blender 4.5 or newer, Windows 10 or 11, an NVIDIA GPU with 10 GB of VRAM, and 15 GB of free disk space
for the one-time model install. Linux is best effort. Opening, playing and rendering saved files needs none of
this, not even the add-on.

1. **Download** `openact-1.0.0.zip` from the [latest release](https://github.com/nicolaiprodromov/openact/releases/latest),
   or [build it from source](CONTRIBUTING.md#set-up-a-checkout).
2. **Install** it in Blender: *Edit › Preferences › Get Extensions › Install from Disk…*
3. **Get the model.** In a Geometry Node editor, press <kbd>N</kbd>, open the **OpenAct** tab and click
   **Install Model**. It downloads about 12 GB once, pinned and hash-checked.
4. **Add a node** with <kbd>Shift</kbd> <kbd>A</kbd> › **OpenAct** and type a question. The model loads on first
   use, in 12 to 40 seconds; after that, a new answer takes well under a second.

## The six nodes

**Ask**, **Pick** and **Dials** judge the whole situation. **Match** and **Classify** judge each item of a list.
**Rules** judges each measured situation. Add them with <kbd>Shift</kbd> <kbd>A</kbd> › **OpenAct**. On the cards,
a round socket carries one value and a diamond carries a value per element; the chips show what the example words
produce.

<table>
  <tr>
    <td width="440"><img src="docs/assets/nodes/ask.svg" width="440" alt="The Ask node, asked 'Is it night?' about a snowy village at night: Answer 0.97, Yes True, Confidence 0.94."></td>
    <td valign="middle">
      <h3>Ask</h3>
      <p>One question about the whole scene, answered <b>yes / no</b> or as an <b>amount</b> from 0 to 1.
      Use it to switch a branch, light a window or set a strength.</p>
      <p><sub>EXAMPLE</sub><br><i>"Is it night?"</i> → <b>Answer</b> 0.97, <b>Yes</b> on.</p>
    </td>
  </tr>
  <tr>
    <td width="440"><img src="docs/assets/nodes/pick.svg" width="440" alt="The Pick node, prompted 'something to sit on' over a Furniture collection: Index 0, Item Armchair, the Armchair as an instance."></td>
    <td valign="middle">
      <h3>Pick</h3>
      <p>Chooses the one entry of a list that best fits a prompt. The list is typed, read from a CSV, or taken
      from a collection, an object's materials or its attribute names.</p>
      <p><sub>EXAMPLE</sub><br><i>"something to sit on"</i> from a Furniture collection → the <b>Armchair</b>, as an
      instance.</p>
    </td>
  </tr>
  <tr>
    <td width="440"><img src="docs/assets/nodes/dials.svg" width="440" alt="The Dials node, prompted 'A stormy night in a fishing village': Rain True, Wind speed 24.6, Time of day night, Street lights True, Fog 0.35."></td>
    <td valign="middle">
      <h3>Dials</h3>
      <p>One prompt sets many named settings at once: switches, amounts in any range, or one of several options.
      Each setting becomes its own output, and all of them are answered in one request.</p>
      <p><sub>EXAMPLE</sub><br><i>"A stormy night in a fishing village"</i> → rain on, wind 24.6, night, street
      lights on.</p>
    </td>
  </tr>
  <tr>
    <td width="440"><img src="docs/assets/nodes/match.svg" width="440" alt="The Match node, asking 'Would you find it in a kitchen?' about every prop of a collection: a probability, a selection and a confidence per element."></td>
    <td valign="middle">
      <h3>Match</h3>
      <p>Asks the same question about every item of a list. Each element reads its item's probability, and
      <b>Selection</b> marks the items at or above the threshold.</p>
      <p><sub>EXAMPLE</sub><br><i>"Would you find it in a kitchen?"</i> over 40 props → the pans and kettles stay,
      the sofa goes.</p>
    </td>
  </tr>
  <tr>
    <td width="440"><img src="docs/assets/nodes/classify.svg" width="440" alt="The Classify node, labelling an object's material slots from a Labels CSV plugged in through Import CSV: a label, a confidence, and the roughness and metallic of each label, per element."></td>
    <td valign="middle">
      <h3>Classify</h3>
      <p>Gives every item one of several labels. The numeric columns of the Labels CSV, like roughness or metallic,
      come out as outputs at the chosen label.</p>
      <p><sub>EXAMPLE</sub><br>Each material slot → <b>metal</b>, <b>wood</b>, <b>stone</b> or <b>fabric</b>, with
      its roughness and metallic.</p>
    </td>
  </tr>
  <tr>
    <td width="440"><img src="docs/assets/nodes/rules.svg" width="440" alt="The Rules node, running Conway's Game of Life from life_factors.csv and life_outcomes.csv: each cell's outcome, confidence and alive flag, read through the state and neighbours inputs."></td>
    <td valign="middle">
      <h3>Rules</h3>
      <p>Decides a table of situations. Factors such as amount bands, categories and flags span a grid of cells.
      The model decides each cell once, and every element reads its own cell through the factor inputs.</p>
      <p><sub>EXAMPLE</sub><br>Conway's Game of Life, written as four outcomes, each with a sentence that says when
      it happens.</p>
    </td>
  </tr>
</table>

The [user guide](docs/GUIDE.md#the-six-nodes) covers every input, output and option.

## How it works

1. **You write words.** Type them on the node, expose them on the modifier panel, or keep them in CSV files.
2. **OpenAct asks outside evaluation.** When words change, it sends only the new questions to its model server on
   `127.0.0.1`. There, [SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev) scores each allowed answer with
   [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) in a single forward pass. No text is generated.
3. **The answers become a table inside the node.** Geometry Nodes looks them up per element, through an ID or the
   Rules factor inputs, as fast as any other lookup.
4. **The table is saved with the file.** Anyone can open, play and render it on Blender 4.5 or newer, with or
   without OpenAct.

## Demos

Six scenes ship in [`demos/`](demos), answered and saved. Open one and press play: they need neither the model nor
the add-on. With OpenAct installed, the sidebar shows their tables, and editing their prompts or CSVs asks the model
again.

| Scene | What it shows |
|:---|:---|
| `life.blend` | Conway's Game of Life from a CSV, with four rule sets to switch between (Rules) |
| `biome.blend` | Plants chosen from habitat descriptions, and snow laid by altitude (Rules) |
| `kitchen.blend` | 40 props kept or dropped for a kitchen, scaled by weight and placed by room (Match, Classify) |
| `materials.blend` | Material slots sorted into families, with each family's roughness and metallic (Classify, Match) |
| `weather.blend` | One prompt sets rain, wind, time of day, street lights and fog (Dials) |
| `ask_pick.blend` | "Is it night?" and a furniture pick, with a different prompt on each object (Ask, Pick) |

## Documentation

- **[User guide](docs/GUIDE.md):** install and the model runtime, the nodes in depth, CSV formats, overrides,
  templates, render farms, and known issues.
- **[Changelog](CHANGELOG.md):** what changed in each release.
- **[Contributing](CONTRIBUTING.md):** tests, demos, code guidelines and releases.
- **[Design](docs/DESIGN.md):** the v1 design, with the reasons behind every decision.
- **[Branding](branding/README.md):** the logo, colours and type.

## Roadmap

<p align="center">
  <img src="docs/assets/roadmap.svg" width="100%" alt="Roadmap. Shipped in 1.0: six judgement nodes; rules, labels and payloads from CSV; answers saved in the .blend. Next: lists read from geometry inside the node tree; the Blender Extensions platform. Later: a CPU backend, so no NVIDIA GPU is needed; a high-accuracy mode with SemIf's 27B model. Exploring: agents whose decisions arrive during playback; Apple Silicon.">
</p>

Ideas and requests are welcome: [open an issue](https://github.com/nicolaiprodromov/openact/issues).

## Credits and licence

OpenAct is free software, licensed under [GPL-3.0-or-later](LICENSE), which the extension zip carries too. It
builds on:

- **[SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev)** (formerly OpenJev) by TheoLeeCJ, MIT: the one-pass decision
  method. The installer installs it into the runtime; it is not bundled.
- **[Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)** by the Qwen team (Alibaba), Apache-2.0: the model. The
  installer downloads it from Hugging Face; it is not redistributed.
- **PyTorch** (BSD-3-Clause), **Hugging Face Transformers** and **huggingface_hub** (Apache-2.0) and **uv** (MIT or
  Apache-2.0), installed into the runtime.
- **[Inter](https://rsms.me/inter/)** (SIL Open Font License 1.1), the typeface of Blender's interface, set in the
  artwork.

Not affiliated with TypeSafe or Jev, its hosted model. The model server is Jev-compatible only in the shape of its
requests (`POST /v1/systemone`).

## Disclaimer

OpenAct is partly vibe-coded with AI. I review all of the code by hand and test it thoroughly, including with the
headless test suites that run on Blender 4.5, 5.0 and 5.1.

<br>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="branding/openact-logo-dark.svg">
    <img src="branding/openact-logo-light.svg" height="44" alt="OpenAct">
  </picture>
</p>
