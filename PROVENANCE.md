# Provenance, Licenses, and Attribution

This repository is a task layer. It contains no connectome, no body model, and
no physics engine: it fetches a simulator checkout at setup time and drives it.
This file records what that simulator is built from, what each piece is licensed
under, and which of those facts were verified directly.

## What this repository contains

Everything under `fishing/`, `web/src/`, `tests/`, `tools/`, `run.py`, and this
file is original work in this repository, MIT licensed (see `LICENSE`). Nothing
in `vendor/` is part of this repository; `vendor/` is gitignored, and so is the
response cache derived from it.

The viewer's dependencies (`three`, `vite`, `typescript`) are installed from npm
into `web/node_modules`, which is gitignored. They are not vendored or
redistributed here either.

## The simulator this layer drives

**Embodied Fly Lab** - <https://github.com/statsleelab/embodied-fly-lab>

- **No LICENSE file was present in the upstream repository when this layer was
  written** (verified: the repository root contains no `LICENSE`, `LICENCE`, or
  `COPYING` file). Under default copyright that means no redistribution rights
  are granted.
- Consequence for this repository: **the simulator is never vendored here.**
  `tools/fetch_sim.py` clones your own checkout into `vendor/`, which
  `.gitignore` excludes. No upstream code, and none of its roughly 95 MB of
  connectome arrays, is committed to this repository or redistributed by it.
- **The response cache is not committed either.** `results/dn-cache.json` is
  recorded output of the upstream network, so it is derived from upstream data
  and from the GPL-2.0-or-later connectivity files upstream builds on.
  `.gitignore` excludes it; `python3 run.py cache` regenerates it from your own
  checkout.
- This layer also never modifies upstream files. It imports `brain-core.mjs` and
  reads `data/` and `assets/` as they are. The ablations in
  `fishing/brain-host.mjs` permute *copies* of the arrays this layer loads off
  disk before handing them to upstream's unmodified `BrainEngine`; no upstream
  source file is touched by them.
- If you intend to do anything beyond running this locally, ask the upstream
  author to add a license first.

## Upstream components, as recorded in the simulator's own PROVENANCE.md

The simulator's `PROVENANCE.md` documents its sources. The verification state
below is stated for each claim rather than assumed.

| Component | Used for | License | Verified |
| --- | --- | --- | --- |
| [MuJoCo](https://github.com/google-deepmind/mujoco) (Google DeepMind) | physics runtime; not used by this layer's replay viewer, and used by upstream's own browser arena | Apache-2.0 | Yes, `LICENSE` in the MuJoCo repository is the Apache License 2.0 |
| [FlyGym / NeuroMechFly v2](https://github.com/NeLy-EPFL/flygym) (NeLy, EPFL) | fly body, CPG parameters, recorded leg trajectories | Apache-2.0 | Yes, the published `flygym` 1.2.1 package declares `License: Apache-2.0` and bundles the Apache 2.0 text |
| [Three.js](https://github.com/mrdoob/three.js) | rendering in `web/`, installed from npm | MIT | Yes, the `three` npm package declares `"license": "MIT"` |
| [Vite](https://github.com/vitejs/vite) | the viewer's build, installed from npm | MIT | Yes, the `vite` npm package declares `"license": "MIT"` |
| [TypeScript](https://github.com/microsoft/TypeScript) | the viewer's typechecking, installed from npm | Apache-2.0 | Yes, the `typescript` npm package declares `"license": "Apache-2.0"` |
| [FlyWire](https://flywire.ai/) FAFB v783 connectome | the network that is simulated | Reported as CC BY 4.0, with citation required | **Unverified here.** `codex.flywire.ai` was unreachable from the environment the sibling repository was written in, and this layer inherits that claim rather than rechecking it. Consult the FlyWire/Codex terms before any reuse beyond running this locally |
| [Eon fly-brain](https://github.com/eonsystemspbc/fly-brain) | connectivity and completeness files | GPL-2.0-or-later, per the simulator's PROVENANCE.md | Taken from the simulator's PROVENANCE.md; not independently rechecked |
| [Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model) | LIF constants, reference sugar population | MIT, per the simulator's PROVENANCE.md | Taken from the simulator's PROVENANCE.md; not independently rechecked |

The GPL-2.0-or-later component is another reason this repository does not
redistribute the data directory or anything recorded from it: mixing either into
an MIT-licensed repository would misstate its terms.

## Required citations

FlyWire asks that work using the dataset cite the source papers. If you publish
anything derived from a run of this layer, cite:

- Dorkenwald, S. *et al.* (2024). Neuronal wiring diagram of an adult brain.
  *Nature*. doi:10.1038/s41586-024-07558-y
- Schlegel, P. *et al.* (2024). Whole-brain annotation and multi-connectome cell
  typing of *Drosophila*. *Nature*. doi:10.1038/s41586-024-07686-5
- Matsliah, A. *et al.* (2024). Neuronal parts list and wiring diagram for a
  visual system. *Nature*.
- Zheng, Z. *et al.* (2018). A complete electron microscopy volume of the brain
  of adult *Drosophila melanogaster*. *Cell*.

For the body and the sensorimotor framework:

- Wang-Chen, S. *et al.* NeuroMechFly v2 / FlyGym: simulating embodied
  sensorimotor control in adult *Drosophila*.

For the physics runtime:

- Todorov, E., Erez, T., Tassa, Y. MuJoCo: A physics engine for model-based
  control.

The FlyWire annotations repository additionally asks that the release matching
your citations be used, and names Berg *et al.* (2025) for its 3.x annotations.

## What this layer adds, and what that means

The fishing task, the bite schedule, the stimulus encoding, the hook window, the
re-cast, all scoring, every baseline, the learning-curve plotting, the replay
format and the whole viewer are written here and are all scripted. The readout
weights are fitted here by REINFORCE against recorded simulator output.

**The connectome is frozen.** No weight inside the simulator is trained, read
back and rewritten, or altered in any way. The trained object is a linear map
from eight of upstream's readout populations to one binary action, and it lives
entirely in `fishing/readout.mjs` and `results/readout.json`.

No claim of biological validity is made anywhere in this repository. The
simulator's own README states that it is a research prototype rather than a
validated digital animal, that its visual input is a looming-population proxy,
and that its sensor and descending-neuron transforms are designed interfaces.
Every number this layer reports inherits those limits, and the scripted
scaffolding around them adds more. **A fly does not learn to fish here.** A
policy-gradient fit on a small readout learns to press a button when a
particular population fires, and the fish are a drawing.
