# Changelog

All notable changes to OpenAct. Versions follow the extension manifest (`openact/blender_manifest.toml`).

## Unreleased

- The demos are removed.
- *Prepare for Playback* is removed, with its pins, *Unpin* and *Unpin All*. An entry that 1.0 pinned is kept in the
  file, but may now be dropped like any other stored answer to keep the file small.
- Fix: a Pick whose answer is below *Unsure Below* counts as unsure in its status.
- Every node starts with a **Geometry** input and output, which pass the geometry through unchanged, and a
  **Trigger** input. Files saved by 1.0 gain them on their first pass, keeping their links. Rules factors can't be
  called Geometry or Trigger, and payload columns and Dials settings can't be called Geometry.
- Nodes show an info marker that carries their words: each one reports its words and whether its Trigger is on
  through a Warning node, on every evaluation that uses it.
- No limit on the sets of words a node answers: a tree used by any number of objects with different words shows each
  its own answer, and every stored answer switches back instantly, as nodes find answers by a hash of their words in
  their table. The error "... the limit is 16: split the tree" is gone; two sets of words whose hashes collide, which
  is rare, get an error asking for another Key. Files saved by 1.0 are rebuilt on their first pass.
- A node keeps up to 1024 stored sets of answers besides the ones in use (was 16), still within 2 MB per node.
- Words can come from any nodes that make a single String, such as the frame through Value to String, a Repeat zone
  or a simulation: OpenAct reads the words each evaluation reports. A For Each Element zone gives each element its
  own words and answers. A node no evaluation uses reads its words as before, and words only Geometry Nodes computes
  wait for its first evaluation. In a file that isn't trusted, words an Import Text node reads from outside the
  file's folder are blocked, as before.
- A node asks the model only while its **Trigger** is on, for the words it reported with the Trigger on, and never
  asks just because its words changed. Link a Boolean node set to True to ask whenever something is missing. Files
  saved by 1.0 keep their answers and ask nothing until a Trigger is linked. *Ask Again* still asks once, whatever
  the Trigger.
- A node with missing answers that nothing asks says why: *Link a Trigger to ask*, *Trigger off* or *Not
  evaluated*, with how many answers are missing. The sidebar shows the active node's Trigger at the last evaluation
  and how many sets of words the evaluation reported.
- Nodes ask during playback: while the timeline plays, a node whose Trigger is on asks for the words each frame
  brings that it has no answer for, and each answer shows a frame or so after it arrives, with no graph rebuilt and a
  few milliseconds of Blender's time per frame at most. Meanwhile the node keeps showing what it showed last. The
  answers are stored once playback stops, or when the file is saved; simulations continue without resetting. Renders
  still never ask: bake simulations before final renders.

## 1.0.0

The first release of OpenAct.

- Six nodes: **Ask**, **Pick** and **Dials** judge the whole situation, **Match** and **Classify** each item of a
  list, and **Rules** each measured situation. Lists come from text, CSV files, collections, material slots or
  attribute names.
- Answers apply per element, through an ID or the Rules factor inputs, from a table stored in the node. Saved files,
  per-element tables included, evaluate and render without the add-on.
- CSV authoring: answer sets written as "answer + when", Factors and Settings files, and payload columns that become
  outputs. A node takes a typed path or a plugged-in Import CSV node, and sees edits while Blender runs.
- An Overrides CSV with locks on all six nodes, and *Write Overrides CSV*.
- A confidence for every answer, unsure flags (*Unsure Below*), and *Balance Option Order* on each node.
- The sidebar shows each node's status, tables, questions and file problems, and offers starter templates and *Copy
  Diagnostics*; the *Log Requests* preference keeps a request log.
- *Prepare for Playback* for animated words.
- Several Blender sessions can share one model server.
- The installer checks the disk and the GPU, and can locate an existing runtime.
- Background renders use the saved answers; headless scripts can opt in to asking.
- Blender 4.5 or newer.
- Six demo scenes.
