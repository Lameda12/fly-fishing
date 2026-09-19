# Fly Fishing

A fruit fly connectome sits on a dock with a rod. Fish take the bait at random
times. Something has to decide when to strike.

That something is **not the fly**. This repository is a task layer on top of
[Embodied Fly Lab](https://github.com/statsleelab/embodied-fly-lab), which
couples a whole-brain FlyWire v783 spiking network to a NeuroMechFly body in
MuJoCo. This layer does not contain, modify, or reimplement any of that. It
fetches a checkout, sends a pulse down one of the simulator's existing sensory
channels, records what its descending neurons do about it, and then fits
**seventeen numbers** that turn those rates into "hook" or "wait".

## Frozen versus trained

This is the whole claim, so it goes first.

**Frozen.** The connectome. All 138,639 neurons, all 15,091,983 weighted edges,
the LIF constants, upstream's stimulus channels and its readout populations.
Nothing inside the simulator is trained, fine-tuned, adapted, or written to. The
same network, reset from the same seed, produces the same response on episode 1
and episode 600.

**Trained.** A linear readout. Seventeen weights mapping eight descending and
motor readout rates (this window and the last) plus a bias to one number, pushed
through a sigmoid, sampled as hook-or-wait, fitted by REINFORCE. They live in
`fishing/readout.mjs` and are written to `results/readout.json`.

**The fly does not learn to fish.** No biological learning happens here, and
none is claimed. A policy-gradient fit learns that a particular population
firing hard means a fish is on the line, in the same sense that a logistic
regression on a thermometer learns what a fever is. The fish are a drawing.

## Demo

<!-- DEMO GIF SLOT -->
<!--
  Drop the recording in here, replacing the line below:
      ![Trained readout against the random control](docs/demo.gif)
  Suggested capture:
    1. the viewer at 1x, through two or three catches side by side
    2. the random panel snapping a line, for the red flash
    3. results/learning-curve.svg
-->

_Demo recording not captured yet. See the slot above._

## Run it

Needs Python 3.9+, Node 18+, npm, and about 200 MB of disk for the simulator
checkout. Nothing here calls out to a network service at run time.

```bash
python3 run.py
```

That one command fetches the simulator if it is missing, records the
descending-neuron response cache if it is missing, converts the fly body, trains
the readout, records a pair of episodes, reports the baselines, and serves the
viewer.

The steps are also available separately:

```bash
python3 run.py fetch                                # only fetch/verify the simulator
python3 run.py cache                                # re-record the response cache (slow)
python3 run.py train -- --seed 7 --episodes 1200    # only train
python3 run.py record                               # only re-record the replay pair
python3 run.py report                               # only re-run the baselines table
python3 run.py glb                                  # only convert the fly body
python3 run.py serve                                # only serve the viewer
python3 run.py build                                # build the static site
python3 run.py test                                 # the scripted layer's tests
python3 tools/fetch_sim.py --from ../embodied-fly-lab   # link an existing checkout
```

Every run is seeded. The same `--seed` reproduces the same readout, the same
learning curve and the same recorded episodes, because the network is reset from
that seed and the scripted layer draws from one seeded RNG.

**Only the cache step is slow.** It is the only thing that runs the whole-brain
network, and it takes about 40 minutes on a laptop. Training on the cache takes
under a minute, so re-training with different hyperparameters is cheap.

### Tests

```bash
node tests/task.test.mjs      # the schedule, the hook window, the re-cast, scoring
node tests/readout.test.mjs   # features, the REINFORCE update, every baseline
node tests/cache.test.mjs     # the splice, its guard rails, the ablations
```

These need no simulator checkout: they cover the scripted layer, which is where
the bugs that would silently corrupt a result live.

## The task

A bobber floats 13 units out. Things take the bait at random times, and half of
them are not fish.

| Rule | Value | Scripted or simulated |
| --- | --- | --- |
| Episode length | 60 s of brain time | scripted |
| Decision window | 50 ms | scripted |
| **Bite** (a real fish) | 150 Hz pulse on `loomHz` for 250 ms | scripted stimulus, simulated response |
| **Decoy** (a nibble) | 60 Hz pulse on `loomHz` for 250 ms | scripted stimulus, simulated response |
| Share of events that are decoys | 50% | scripted |
| Gap between events | exponential, mean 5 s, minimum 3 s | scripted |
| Hook within 500 ms of a bite | fish caught, **+1** | scripted |
| Hook on a decoy, or on nothing | line snapped, **-0.5** | scripted |
| Bite nobody hooks | **0** | scripted |
| After any hook | 1.5 s re-cast, no decisions | scripted |

Event times and spacing are redrawn every episode from an exponential with a
minimum, so there is no rhythm for a fixed-interval policy to lock onto.

**The bobber dips on a decoy too**, which is the whole point of having them. A
policy that watches the water can tell that *something* happened; only a policy
that watches the fly can tell *what*. That is what the fixed-delay-after-dip
baseline measures, and it is the score to beat.

The re-cast is what makes the reward function non-degenerate. Without it, a
coin-flip policy would hook 600 times in an episode and the scoring would be
about hook volume rather than hook timing.

## Why the raw rates, and why the looming channel

Two measurements decided the design, and both are worth stating because either
one going the other way would have made the task unwinnable.

**The bite rides on `loomHz`.** Upstream exposes seven stimulus channels. A
dose-response sweep of each against every readout population gave this for
looming input, as the raw `escape_giant_fiber` rate:

| `loomHz` in | 0 | 5 | 10 | 20 | 35 | 60 | 100 | 150 | 220 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `escape_giant_fiber` out (Hz) | 0.0 | 45.5 | 67.4 | 110.1 | 134.7 | 160.0 | 177.4 | 199.0 | 218.1 |

Monotone across the whole range. Pulse amplitude survives into the readout,
which is what the decoy discrimination runs on: a 150 Hz bite peaks the giant
fiber near 199 Hz and a 60 Hz decoy near 160 Hz. A 39 Hz gap, against whatever
trial-to-trial spread the network has, is the entire signal available.

**The readout reads the raw rates, not `frame.motor`.** Upstream's normalized
motor fields divide by 35 and clamp to 1, so `motor.escape` pins at 1.0 for any
looming input above about 5 Hz and carries no amplitude information at all. The
raw rate in the table above is the same signal before that clamp. Feeding the
policy `frame.motor` would have thrown the measurement away; the sibling
repository's note that the giant fiber "saturates above roughly 30 Hz" is a
property of the clamp, not of the population.

The eight features are upstream's own readout populations: `forward_odn1`,
`walk_dnp09`, `turn_left`, `turn_right`, `reverse_mdn`, `groom_adn1`,
`escape_giant_fiber`, `feed_mn9`. The readout gets each of them for the current
window and the previous one, so a linear policy can see an onset rather than
only a level, plus a bias. Seventeen weights: 8 + 8 + 1.

A constant 40 Hz `hungerHz` walking drive runs underneath everything, so the
baseline is a fly that is doing something rather than a silent network. That is
why not hooking is a skill: `walk_dnp09` sits near 40 Hz the whole episode and
the policy has to learn to ignore it.

## How training works, and why the cache is not a shortcut

One 15 ms brain step costs 106-140 ms of wall clock on the machine this was
written on, which is about 0.11x real time. A 60 s episode is therefore about
eight minutes, and REINFORCE wants hundreds of them. Training against the live
simulator is days of compute.

So training runs against a recorded cache of descending-neuron responses. **That
cache is exact, not an approximation,** and the reason is structural:

> Hooking has no sensory consequence. The bobber, the fish and the schedule do
> not depend on what the readout decides, so the descending-activity trace an
> episode produces is independent of the policy watching it.

This is not true of most control problems, and it is the whole justification. The
cache replays real network output; it does not model it. Every number in
`results/dn-cache.json` came out of upstream's engine.

What the cache does is record four long baseline traces (70 s each, under the
walking drive alone) and sixteen bite responses (2.8 s each, from pulse onset),
then build an episode by laying responses onto a baseline at the scheduled
times. The one assumption that introduces is that a response is over before the
next bite starts. The task enforces a 3 s minimum gap against a 2.8 s recorded
response for exactly that reason, and `fishing/cache.mjs` reports the splice's
own error bar: how far each feature still is from baseline where the splice
hands back.

`python3 run.py cache` prints that residual per feature at the end of the run,
and `results/baselines.json` carries it. On the smoke run used to shake the
pipeline out, `escape_giant_fiber` (the feature the policy actually uses)
returned to exactly its baseline of 0 Hz before the splice point, and the
largest residual anywhere was `walk_dnp09` at -2.6 Hz against a 39.5 Hz
baseline.

### The update

Per decision window: `p = sigmoid(w . x)`, action sampled Bernoulli, and

```
grad = mean over decisions of (action - p) * x * advantage
```

with the advantage being the discounted reward-to-go minus a running-mean
baseline, divided by its own standard deviation. The discount (gamma 0.9, about
a one-second horizon at 50 ms windows) is not for delayed reward, since hooking
pays immediately; it is there so the policy feels the cost of the 1.5 s re-cast
a hook commits it to.

## Results

**Not filled in yet.** The full response cache takes about 40 minutes to record
and this commit is the code that records it. Run

```bash
python3 run.py            # records the cache, trains, records, reports
```

and `python3 run.py report` prints the table below straight from
`results/baselines.json`:

| Policy | Catch rate | Snapped lines / min | Hook precision | Mean reward |
| --- | ---: | ---: | ---: | ---: |
| Oracle (hooks on true bites) | | | | |
| Trained readout (greedy) | | | | |
| Fixed interval, every 5 s | | | | |
| Random control (rate matched) | | | | |

Training also writes `results/learning-curve.svg`, which plots catch rate and
false-hook rate against training episode with the oracle and the random control
drawn in as reference lines.

On a one-trace smoke cache the readout reached the oracle's 100% catch rate with
zero snapped lines, against 8.2% for the rate-matched random control, and the
weights it settled on were the ones you would hope for: strongly positive on
`escape_giant_fiber` for both the current and the lagged window, negative on
`walk_dnp09` (the background walking drive), and a large negative bias. **That
smoke run is not a result** and is reported here only to say what the pipeline
does; a one-realization cache has no trial-to-trial variability in it, so the
task it poses is easier than the real one. The numbers that go in the table
above come from the full cache.

## What is in this commit, and what is not

Present:

- bites and decoys, the task, the scoring, the re-cast
- the response cache, the splice, and its measured residual per event type
- the REINFORCE readout, checkpoints, and the learning curve
- four baselines: oracle, fixed delay after the dip, fixed interval, and a
  rate-matched random control
- the real NeuroMechFly body, converted to GLB, with a labelled placeholder
  when the GLB has not been generated
- the replay viewer, two recordings side by side

Not yet, and each is a named next step rather than a silent omission:

- **Live mode.** A WebSocket from the Python side, with a documented schema.
  Replay mode is all that ships here, which is also what lets `web/` deploy as a
  static site with no backend.
- **A close-up camera and the webm export.** The viewer has orbit controls and a
  shared transport; the follow camera and the one-click MediaRecorder capture
  are not built.
- **The ablation results.** Both ablations (`--ablation weight-shuffle`,
  `--ablation input-shuffle`) are implemented in `fishing/brain-host.mjs` and
  unit tested, and each needs its own recorded cache. No ablation number is
  claimed until those runs exist.
- **A live-sim validation column.** The results below are measured on spliced
  cache episodes. Running whole episodes against the live network to confirm the
  splice end to end is the check that has not been done yet.

## The fly model

`tools/build_fly_glb.py` converts the NeuroMechFly body out of the fetched
checkout into one glTF binary:

```bash
python3 tools/build_fly_glb.py     # reads vendor/, writes web/public/fly.glb
```

It parses `assets/model/fly.xml` (68 bodies, 66 hinges, 69 mesh geoms), loads
the 39 binary STLs, applies each declared mesh scale (thirty of the meshes are
mirrored, so their winding is flipped back), welds vertices, recomputes smooth
normals, and writes a node tree that mirrors the MJCF body tree. No third-party
Python package is involved.

The rig goes in the glTF's `extras` rather than a skin, because this is a
rigid-body tree and not a skinned mesh. Each hinge records the node it turns,
its axis, its neutral angle and the control index that drives it, so the viewer
can do forward kinematics without a physics engine. The converter derives qpos
addresses by walking the tree and **checks that derivation against the actuator
table for all 42 actuated joints** before trusting it for the 24 passive ones.

**`web/public/fly.glb` is gitignored.** It is geometry converted from a
repository that ships no LICENSE, so it is generated locally and never
committed, exactly like the response cache. The viewer treats its absence as
normal: without it you get a labelled placeholder capsule, and the HUD says
which body is on screen. A static deploy that wants the real body has to run the
converter as part of its build, from its own checkout.

**The legs are at the model's neutral pose.** Upstream's CPG could drive them
and the rig carries everything needed for that, but a fly sitting on a dock
holding a rod is not walking, and animating a gait it is not performing would be
drawing behaviour rather than showing it. The idle sway is scripted and labelled.

## The viewer

`web/` is a Vite + TypeScript app. `three` comes from npm; there is no CDN and no
external network call at build or run time.

```bash
cd web && npm install
npm run dev      # replay mode on localhost
npm run build    # static site in web/dist
```

Replay mode needs no backend at all, so `web/dist` deploys to Vercel (or any
static host) as it is. Set the project root to `web/`, the build command to
`npm run build`, and the output directory to `dist`.

Two recordings play side by side off one episode clock, because both were made
from the same bite schedule. The difference between the panels is the policy and
nothing else.

### The replay format

`fishing/record.mjs` writes it; `web/src/types.ts` is its machine-readable
definition. Schema version 1:

```jsonc
{
  "schemaVersion": 2,
  "kind": "fly-fishing-replay",
  "policy": "readoutGreedy",        // or "random"
  "label": "Trained readout",
  "seed": 3109511962,
  "windowMs": 50,                   // one frame is one decision window
  "episodeMs": 60000,
  "summary": { "bites": 7, "decoys": 6, "caught": 7, "decoysHooked": 1, "snapped": 1, "...": "..." },
  "events":   [{ "tMs": 3856, "type": "bite" }],    // or "decoy"
  "outcomes": [{ "tMs": 3950, "type": "catch" },
               { "tMs": 9910, "type": "snap", "onDecoy": true }],
  "frames": {                       // columnar, one entry per window
    "escapeHz": [0, 107.4, 159.8],  // raw escape_giant_fiber, simulated
    "walkHz":   [39.5, 38.1, 40.2], // raw walk_dnp09, simulated
    "pHook":    [0.004, 0.03, 0.45],// the readout's P(hook), null if it has none
    "bobber":   [0, 0.4, 1]         // dip, 0 floating to 1 under; scripted
  },
  "provenance": { "simulated": ["..."], "scripted": ["..."], "frozen": "...", "trained": "..." }
}
```

`onDecoy` separates the two kinds of snapped line: falling for a decoy is the
discrimination failing, and snapping on empty water is not.

The viewer samples that 20 Hz grid and interpolates the bobber between windows,
so it renders smoothly at whatever frame rate the browser gives it.

## Scripted versus simulated

**Simulated** (upstream's whole-brain LIF network, unmodified, driven through its
own stimulus channels):

- every descending and motor readout rate the policy sees or the HUD prints
- the response to the bite pulse, from the looming visual proxy through to the
  giant-fiber population

**Scripted** (this repository, plain code, no neural component anywhere in it):

- the fishing task: the schedule, the hook window, the re-cast, all scoring
- the mapping from "a fish bit" to a 150 Hz pulse on `loomHz`
- every baseline, including the oracle and the rate-matched random control
- the bobber track, the dock, the water, the fly's position, and the placeholder
  body in the viewer
- the splice that assembles an episode from recorded responses

**Trained** (this repository): seventeen readout weights, by REINFORCE.

`results/readout.json` and every recording restate this split in their own
`frozen` / `trained` / `provenance` fields, so a stray copy of either file cannot
be mistaken for a measurement.

## No biological validity

Nothing here measures a fruit fly.

The simulator's own README is explicit that it is an interactive research
prototype and not a biologically validated digital animal: its visual input is a
looming-population proxy rather than a photoreceptor model, and its sensor and
descending-neuron transforms are designed interfaces. Every number this layer
produces inherits all of that, and then adds a scripted game and a fitted
readout on top.

A catch rate here is a property of this code. It says nothing about
*Drosophila*, about connectomics, or about what a real fly would do.

## Credits and licenses

This layer is MIT licensed (`LICENSE`). It bundles nothing.

**The upstream simulator ships no LICENSE file**, so no redistribution rights are
granted for it, and this repository therefore never vendors it: the fetch script
clones your own checkout into a gitignored `vendor/`, and no upstream code or
connectome data is committed here. The response cache recorded from it is
gitignored for the same reason.

| Component | License | Verified |
| --- | --- | --- |
| [Embodied Fly Lab](https://github.com/statsleelab/embodied-fly-lab) | no LICENSE file present; fetched, never redistributed | Yes, checked directly |
| [MuJoCo](https://github.com/google-deepmind/mujoco) | Apache-2.0 | Yes |
| [FlyGym / NeuroMechFly v2](https://github.com/NeLy-EPFL/flygym) | Apache-2.0 | Yes |
| [Three.js](https://github.com/mrdoob/three.js) | MIT | Yes, the npm package declares it |
| [Vite](https://github.com/vitejs/vite) | MIT | Yes, the npm package declares it |
| [TypeScript](https://github.com/microsoft/TypeScript) | Apache-2.0 | Yes, the npm package declares it |
| [FlyWire](https://flywire.ai/) FAFB v783 connectome | reported CC BY 4.0, citation required | **No**, see below |
| [Eon fly-brain](https://github.com/eonsystemspbc/fly-brain) | GPL-2.0-or-later | Inherited from upstream's PROVENANCE.md |
| [Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model) | MIT | Inherited from upstream's PROVENANCE.md |

Work using the connectome should cite Dorkenwald *et al.* (2024), Schlegel
*et al.* (2024), Matsliah *et al.* (2024), and Zheng *et al.* (2018); the body
and framework, Wang-Chen *et al.* (NeuroMechFly v2); the physics runtime,
Todorov *et al.* (MuJoCo).

Full attribution, verification state for each license claim, and full citations
are in [PROVENANCE.md](PROVENANCE.md). One claim there is explicitly marked
unverified: the FlyWire terms page was unreachable from the environment the
sibling repository was written in, so its license is recorded as reported rather
than as checked.

This repository is a sibling of
[fly-nomad-games](https://github.com/Lameda12/fly-nomad-games) and follows its
structure, its fetch-never-vendor rule, and its habit of saying which half of a
number is scripted.
