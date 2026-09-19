// The descending-neuron response cache, and the splice that turns it into an
// episode.
//
// Why a cache at all: one 15 ms brain step costs 106-140 ms of wall clock on
// this machine, so a 60 s episode is about eight minutes. REINFORCE wants
// hundreds of episodes. Training against the live simulator is days of compute.
//
// Why the cache is exact rather than an approximation: hooking has no sensory
// consequence. The bobber, the fish and the schedule do not depend on what the
// readout decides, so the descending-activity trace an episode produces is
// independent of the policy that watches it. That is not true of most control
// problems and it is what makes this legitimate: the cache replays real network
// output, it does not model it. Every number in it came out of upstream's
// engine.
//
// What the splice does assume: that the response to one bite is over before the
// next one starts, so a bite response can be laid onto a baseline trace without
// the two interacting. The task enforces a 3 s minimum gap against a 2.8 s
// recorded response for exactly this reason, and `spliceResidual` below reports
// how far from baseline the response still is where the splice ends, so the
// assumption is measured rather than asserted. tools/validate-cache runs whole
// episodes live and compares them against the spliced version.

import { READOUT_FEATURES, featuresOf } from "./brain-host.mjs";
import { createRng, deriveSeed } from "./rng.mjs";
import { TASK } from "./task.mjs";
import { STIMULI } from "./voyage.mjs";

/**
 * The recorded sections: `baseline` is the background, and one per stimulus the
 * voyage can present. `bite` and `decoy` are in that table too, so a cache
 * recorded for the single-stage task is a subset of a voyage cache rather than
 * a different thing, and one can be merged into the other.
 */
export const EVENT_SECTIONS = Object.freeze(Object.keys(STIMULI));
export const SECTIONS = Object.freeze(["baseline", ...EVENT_SECTIONS]);

export const CACHE_SCHEMA_VERSION = 2;

/** How long a bite response is recorded for, from pulse onset. */
export const RESPONSE_MS = 2_800;
/** Settling time before any recording, so nothing measures a cold reset. */
export const SETTLE_MS = 500;
/**
 * Baseline traces are recorded longer than an episode so a spliced episode can
 * start at a random offset inside one instead of always at the same sample.
 */
export const BASELINE_MS = TASK.episodeMs + 10_000;

const round2 = (value) => Math.round(value * 100) / 100;

/**
 * Record the cache from the live simulator.
 *
 * Baseline traces: the background walking drive alone, one long trace per
 * network seed. Bite traces: settle under the same background, then the pulse,
 * recorded from pulse onset.
 */
export async function buildCache({
  host,
  task = TASK,
  seed = 0x5eed1234,
  baselineTraces = 4,
  eventTraces = 16,
  sections = SECTIONS,
  onProgress,
} = {}) {
  for (const section of sections) {
    if (!SECTIONS.includes(section)) {
      throw new Error(`unknown cache section "${section}"; expected ${SECTIONS.join(", ")}`);
    }
  }
  const windowMs = task.windowMs;
  const background = { hungerHz: task.backgroundHungerHz };

  const record = (windows, stimulusAt) => {
    const rows = [];
    for (let i = 0; i < windows; i++) {
      const frame = host.step(windowMs, stimulusAt(i * windowMs));
      rows.push(featuresOf(frame).map(round2));
    }
    return rows;
  };

  const recorded = {};

  if (sections.includes("baseline")) {
    const baselineWindows = Math.round(BASELINE_MS / windowMs);
    const baseline = { settleMs: SETTLE_MS, durationMs: BASELINE_MS, seeds: [], traces: [] };
    for (let i = 0; i < baselineTraces; i++) {
      const traceSeed = deriveSeed(seed, "baseline", i);
      onProgress?.({ stage: "baseline", index: i, of: baselineTraces });
      host.reset(traceSeed);
      host.settle(SETTLE_MS, background);
      baseline.seeds.push(traceSeed);
      baseline.traces.push(record(baselineWindows, () => background));
    }
    recorded.baseline = baseline;
  }

  const responseWindows = Math.round(RESPONSE_MS / windowMs);
  for (const type of EVENT_SECTIONS) {
    if (!sections.includes(type)) continue;
    const pulse = STIMULI[type];
    const section = {
      settleMs: SETTLE_MS,
      durationMs: RESPONSE_MS,
      stimulus: pulse,
      seeds: [],
      traces: [],
    };
    for (let i = 0; i < eventTraces; i++) {
      const traceSeed = deriveSeed(seed, type, i);
      onProgress?.({ stage: type, index: i, of: eventTraces });
      host.reset(traceSeed);
      host.settle(SETTLE_MS, background);
      section.seeds.push(traceSeed);
      section.traces.push(
        record(responseWindows, (tMs) =>
          tMs < task.biteDurationMs ? { ...background, ...pulse } : background,
        ),
      );
    }
    recorded[type] = section;
  }

  return {
    schemaVersion: CACHE_SCHEMA_VERSION,
    generatedAt: new Date().toISOString(),
    windowMs,
    featureNames: [...READOUT_FEATURES],
    ablation: host.ablation,
    ablationSeed: host.ablationSeed,
    seed,
    simulator: {
      source: "statsleelab/embodied-fly-lab",
      dataset: host.manifest.dataset,
      neurons: host.neuronCount,
      edges: host.edgeCount,
    },
    sections: Object.keys(recorded),
    task: {
      backgroundHungerHz: task.backgroundHungerHz,
      biteDurationMs: task.biteDurationMs,
      stimuli: Object.fromEntries(
        Object.entries(STIMULI).filter(([name]) => recorded[name]),
      ),
    },
    ...recorded,
    provenance:
      "Every row is raw readout output from the upstream whole-brain LIF network, " +
      "recorded through its own stimulus channels. Nothing here is interpolated or modelled.",
  };
}

/**
 * Build one episode's per-window feature rows by laying recorded bite responses
 * onto a recorded baseline trace.
 *
 * The rows this returns are the same eight raw readout rates, in the same order,
 * that the live runner reads off a frame. Downstream code cannot tell the
 * difference, which is the point.
 */
export function spliceEpisode({
  cache,
  events,
  seed,
  task = TASK,
  episodeMs = TASK.episodeMs,
}) {
  if (cache.schemaVersion !== CACHE_SCHEMA_VERSION) {
    throw new Error(`cache schema ${cache.schemaVersion}, expected ${CACHE_SCHEMA_VERSION}`);
  }
  if (cache.windowMs !== task.windowMs) {
    throw new Error(`cache window ${cache.windowMs} ms, task window ${task.windowMs} ms`);
  }
  const rng = createRng(seed);
  const windows = Math.round(episodeMs / task.windowMs);

  const trace = cache.baseline.traces[rng.int(cache.baseline.traces.length)];
  const slack = trace.length - windows;
  if (slack < 0) throw new Error("baseline trace is shorter than an episode");
  const offset = rng.int(slack + 1);
  const rows = trace.slice(offset, offset + windows).map((row) => row.slice());

  for (const event of events) {
    const section = cache[event.type];
    if (!section) {
      throw new Error(
        `the cache has no "${event.type}" responses; re-record it with: ` +
          `node fishing/build-cache.mjs --sections ${event.type} --merge-into <cache>`,
      );
    }
    const response = section.traces[rng.int(section.traces.length)];
    const start = Math.round(event.atMs / task.windowMs);
    for (let i = 0; i < response.length; i++) {
      const target = start + i;
      if (target >= windows) break;
      rows[target] = response[i].slice();
    }
  }
  return rows;
}

/**
 * How far a recorded response is from baseline at the moment the splice hands
 * back to the baseline trace, per feature, in Hz.
 *
 * This is the splice's own error bar. A large residual on a feature would mean
 * the response is being cut off mid-transient and the episode after an event is
 * not what the live simulator would produce.
 */
export function spliceResidual(cache, type = "bite") {
  const tail = (trace) => trace[trace.length - 1];
  const meanOf = (traces, pick) =>
    READOUT_FEATURES.map((_, f) => traces.reduce((sum, t) => sum + pick(t)[f], 0) / traces.length);

  // Compare the last window of each response against the mean baseline level,
  // which is what the splice replaces it with.
  const biteTail = meanOf(cache[type].traces, tail);
  const baselineMean = READOUT_FEATURES.map((_, f) => {
    let total = 0;
    let count = 0;
    for (const trace of cache.baseline.traces) {
      for (const row of trace) {
        total += row[f];
        count++;
      }
    }
    return total / count;
  });

  return READOUT_FEATURES.map((name, f) => ({
    feature: name,
    baselineHz: round2(baselineMean[f]),
    responseTailHz: round2(biteTail[f]),
    residualHz: round2(biteTail[f] - baselineMean[f]),
  }));
}

/**
 * How much bite-versus-decoy information each readout population actually
 * carries, measured on the recorded responses.
 *
 * For each feature: the peak rate reached within each recorded response, summarized
 * across realizations, plus the baseline level, plus d' = |mean difference| /
 * pooled standard deviation. d' is the honest way to say "how separable"; a
 * large one means a threshold on that feature alone would do it.
 *
 * This is the measurement that says whether an ablation destroyed the signal or
 * left it intact, which is the whole question an ablation is asked.
 */
export function discriminability(cache) {
  const summarize = (values) => {
    const mean = values.reduce((total, value) => total + value, 0) / values.length;
    const variance =
      values.reduce((total, value) => total + (value - mean) ** 2, 0) / values.length;
    return { mean, sd: Math.sqrt(variance) };
  };
  const peaks = (section, feature) =>
    section.traces.map((trace) => Math.max(...trace.map((row) => row[feature])));

  return READOUT_FEATURES.map((name, feature) => {
    const bite = summarize(peaks(cache.bite, feature));
    const decoy = cache.decoy ? summarize(peaks(cache.decoy, feature)) : null;
    const baseline = summarize(cache.baseline.traces.flat().map((row) => row[feature]));
    const pooled = decoy ? Math.sqrt((bite.sd ** 2 + decoy.sd ** 2) / 2) : 0;
    return {
      feature: name,
      bitePeakHz: round2(bite.mean),
      bitePeakSd: round2(bite.sd),
      decoyPeakHz: decoy ? round2(decoy.mean) : null,
      decoyPeakSd: decoy ? round2(decoy.sd) : null,
      baselineHz: round2(baseline.mean),
      baselineSd: round2(baseline.sd),
      // Null when the two are identical to the last decimal, which for the
      // silent populations means there is nothing to separate rather than a
      // separation of zero standard deviations.
      dPrime: decoy && pooled > 0 ? round2(Math.abs(bite.mean - decoy.mean) / pooled) : null,
    };
  });
}
