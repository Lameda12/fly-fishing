// Headless host for the upstream whole-brain LIF engine.
//
// The browser build reaches BrainEngine through brain-worker.js and fetch(). The
// fishing layer needs the same engine from Node, so this module reads the same
// binary CSR arrays off disk and hands them to the same unmodified
// vendor/embodied-fly-lab/brain-core.mjs. Nothing here reimplements neural
// computation: every spike below comes out of upstream's engine.
//
// The ablations live here too, and they are the reason this file owns the
// arrays rather than letting upstream load them. `weight-shuffle` permutes the
// weight array and `input-shuffle` permutes the population index lists *before*
// the engine is constructed from them. Upstream's brain-core.mjs is imported as
// it is in every case; what changes is the connectome handed to it.

import { readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { createRng } from "./rng.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));
export const REPO_ROOT = path.resolve(here, "..");
export const SIM_ROOT = path.join(REPO_ROOT, "vendor", "embodied-fly-lab");

const TYPED_ARRAYS = {
  uint8: Uint8Array,
  int16: Int16Array,
  uint32: Uint32Array,
  float32: Float32Array,
};

// Stimulus channel names accepted by upstream's BrainEngine.step(). Kept here so
// the task can be checked against the real interface instead of silently
// misspelling a channel into a no-op.
export const STIMULUS_CHANNELS = Object.freeze([
  "odorLeftHz",
  "odorRightHz",
  "sugarLeftHz",
  "sugarRightHz",
  "touchHz",
  "loomHz",
  "hungerHz",
]);

/**
 * The eight readout populations upstream exposes, in the order this layer feeds
 * them to the policy. These are upstream's own names; the feature vector is
 * `frame.readoutRates` read in this order.
 *
 * The *raw* rates are used, not the normalized `frame.motor` fields. That is a
 * deliberate and load-bearing choice: motor.escape is readoutRates
 * .escape_giant_fiber / 35 clamped to 1, so it pins at 1.0 for any looming input
 * above about 5 Hz and carries no amplitude information at all. The raw rate
 * stays monotone across the whole input range (see README, "Why the raw rates"),
 * which is what makes a strong bite distinguishable from a weak nibble.
 */
export const READOUT_FEATURES = Object.freeze([
  "forward_odn1",
  "walk_dnp09",
  "turn_left",
  "turn_right",
  "reverse_mdn",
  "groom_adn1",
  "escape_giant_fiber",
  "feed_mn9",
]);

export const ABLATIONS = Object.freeze(["none", "weight-shuffle", "input-shuffle"]);

export class MissingSimulatorError extends Error {
  constructor() {
    super(`No simulator checkout at ${SIM_ROOT}\n` + "Run: python3 tools/fetch_sim.py");
    this.name = "MissingSimulatorError";
  }
}

export function assertSimulatorPresent() {
  if (!existsSync(path.join(SIM_ROOT, "brain-core.mjs"))) throw new MissingSimulatorError();
  if (!existsSync(path.join(SIM_ROOT, "data", "manifest.json"))) throw new MissingSimulatorError();
}

async function loadArray(dataDir, descriptor) {
  const bytes = await readFile(path.join(dataDir, descriptor.file));
  const Constructor = TYPED_ARRAYS[descriptor.dtype];
  if (!Constructor) throw new Error(`unsupported dtype ${descriptor.dtype}`);
  const array = new Constructor(bytes.buffer, bytes.byteOffset, descriptor.length);
  if (array.length !== descriptor.length) throw new Error(`${descriptor.file} length mismatch`);
  return array;
}

export function validateStimulus(stimulus) {
  for (const key of Object.keys(stimulus)) {
    if (!STIMULUS_CHANNELS.includes(key)) {
      throw new Error(
        `unknown stimulus channel "${key}"; expected one of ${STIMULUS_CHANNELS.join(", ")}`,
      );
    }
  }
  return stimulus;
}

/** Fisher-Yates over a copy, from a seeded RNG, so an ablation is reproducible. */
export function shuffled(values, rng) {
  const out = values.slice();
  for (let i = out.length - 1; i > 0; i--) {
    const j = rng.int(i + 1);
    const swap = out[i];
    out[i] = out[j];
    out[j] = swap;
  }
  return out;
}

/**
 * Ablation 1: shuffle the connectome weights.
 *
 * Every edge keeps its source and its target, so the graph is untouched; only
 * which weight sits on which edge changes. The multiset of weights is therefore
 * identical, which means total drive into the network is preserved and the
 * comparison isolates *wiring* rather than overall excitability. Done on a copy:
 * the array read off disk is never mutated.
 */
export function shuffleWeights(weights, seed) {
  const rng = createRng(seed);
  const out = weights.slice();
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rng.next() * (i + 1));
    const swap = out[i];
    out[i] = out[j];
    out[j] = swap;
  }
  return out;
}

/**
 * Ablation 2: randomize the input mapping.
 *
 * Upstream's brain-core.mjs looks its stimulus and readout populations up in
 * `manifest.populations`. Replacing those index lists with random neurons of the
 * same count sends the bite pulse into arbitrary cells and reads the decision
 * out of arbitrary cells, while the network, its weights and the population
 * sizes all stay exactly as they were. The manifest is rebuilt as a copy.
 */
export function shuffleInputMapping(manifest, seed, neuronCount) {
  const rng = createRng(seed);
  const populations = {};
  for (const [name, indices] of Object.entries(manifest.populations)) {
    const picked = new Set();
    while (picked.size < indices.length) picked.add(rng.int(neuronCount));
    populations[name] = [...picked].sort((a, b) => a - b);
  }
  return { ...manifest, populations };
}

/**
 * Load the upstream connectome and wrap its engine with the small amount of
 * bookkeeping the task needs: a seeded reset, a settle helper, and a windowed
 * run that reports the readout rates once per decision window.
 *
 * `ablation` is one of ABLATIONS and is applied to the arrays/manifest before
 * the engine sees them. `ablationSeed` makes each ablation reproducible.
 */
export async function createBrainHost({
  seed = 0x5eed1234,
  ablation = "none",
  ablationSeed = 0xab1a7e,
  onStatus,
} = {}) {
  if (!ABLATIONS.includes(ablation)) {
    throw new Error(`unknown ablation "${ablation}"; expected one of ${ABLATIONS.join(", ")}`);
  }
  assertSimulatorPresent();
  const { BrainEngine } = await import(path.join(SIM_ROOT, "brain-core.mjs"));
  const dataDir = path.join(SIM_ROOT, "data");
  onStatus?.("Reading FlyWire whole-brain metadata");
  let manifest = JSON.parse(await readFile(path.join(dataDir, "manifest.json"), "utf8"));
  onStatus?.(
    `Loading CSR connectivity (${manifest.edge_count.toLocaleString("en-US")} weighted edges)`,
  );
  const [offsets, targets, rawWeights, groups] = await Promise.all([
    loadArray(dataDir, manifest.arrays.offsets),
    loadArray(dataDir, manifest.arrays.targets),
    loadArray(dataDir, manifest.arrays.weights),
    loadArray(dataDir, manifest.arrays.groups),
  ]);

  let weights = rawWeights;
  if (ablation === "weight-shuffle") {
    onStatus?.("Ablation: shuffling connectome weights across edges");
    weights = shuffleWeights(rawWeights, ablationSeed);
  } else if (ablation === "input-shuffle") {
    onStatus?.("Ablation: randomizing stimulus and readout population membership");
    manifest = shuffleInputMapping(manifest, ablationSeed, manifest.neuron_count);
  }

  onStatus?.("Initializing LIF state arrays");
  const engine = new BrainEngine(manifest, { offsets, targets, weights, groups });
  engine.reset(seed);

  return {
    manifest,
    engine,
    seed,
    ablation,
    ablationSeed: ablation === "none" ? null : ablationSeed,
    neuronCount: manifest.neuron_count,
    edgeCount: manifest.edge_count,
    synapseCount: manifest.synapse_count,

    reset(nextSeed = seed) {
      engine.reset(nextSeed >>> 0);
    },

    /** One upstream brain step. Returns upstream's frame unchanged. */
    step(durationMs, stimulus = {}) {
      return engine.step(durationMs, validateStimulus(stimulus));
    },

    /**
     * Settle the network under a holding stimulus so a measurement does not
     * start on the transient from a cold reset.
     */
    settle(durationMs, stimulus = {}, resolutionMs = 25) {
      validateStimulus(stimulus);
      let elapsed = 0;
      let frame = null;
      while (elapsed < durationMs) {
        const slice = Math.min(resolutionMs, durationMs - elapsed);
        frame = engine.step(slice, stimulus);
        elapsed += slice;
      }
      return frame;
    },

    /**
     * Step in fixed windows, asking `stimulusAt(startMs)` for each window's
     * input and handing each window's frame to `onWindow(features, frame, ms)`.
     * One window is one decision: the task, the cache recorder and the live
     * runner all march on the same grid so a cached feature row and a live one
     * mean the same thing.
     */
    runWindows({ windowMs, windows, stimulusAt, onWindow }) {
      for (let index = 0; index < windows; index++) {
        const startMs = index * windowMs;
        const frame = engine.step(windowMs, validateStimulus(stimulusAt(startMs, index)));
        onWindow(featuresOf(frame), frame, startMs, index);
      }
    },
  };
}

/** The feature row a decision is made from: upstream's raw rates, in order. */
export function featuresOf(frame) {
  return READOUT_FEATURES.map((name) => frame.readoutRates[name] ?? 0);
}
