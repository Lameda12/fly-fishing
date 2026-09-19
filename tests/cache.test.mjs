import assert from "node:assert/strict";

import { READOUT_FEATURES, shuffleInputMapping, shuffleWeights } from "../fishing/brain-host.mjs";
import {
  BASELINE_MS,
  CACHE_SCHEMA_VERSION,
  RESPONSE_MS,
  spliceEpisode,
  spliceResidual,
} from "../fishing/cache.mjs";
import { bobberTrack } from "../fishing/record.mjs";
import { TASK, buildSchedule } from "../fishing/task.mjs";

const WINDOWS = Math.round(TASK.episodeMs / TASK.windowMs);

/**
 * A stand-in cache with recognizable rows: baseline windows ramp from 1 upward
 * and bite responses are all 9s, so a splice can be checked by reading values.
 */
function fakeCache({ baselineTraces = 2, biteTraces = 3 } = {}) {
  const baselineWindows = Math.round(BASELINE_MS / TASK.windowMs);
  const responseWindows = Math.round(RESPONSE_MS / TASK.windowMs);
  return {
    schemaVersion: CACHE_SCHEMA_VERSION,
    windowMs: TASK.windowMs,
    featureNames: [...READOUT_FEATURES],
    ablation: "none",
    baseline: {
      seeds: [],
      // Each baseline window carries its own value, so a splice that picked a
      // different trace or a different offset is visible in the result.
      traces: Array.from({ length: baselineTraces }, (_, t) =>
        Array.from({ length: baselineWindows }, (_, w) =>
          READOUT_FEATURES.map(() => 1 + t + w / 10_000),
        ),
      ),
    },
    // Bites read 9, decoys read 5, so a spliced episode says which is which.
    bite: {
      seeds: [],
      traces: Array.from({ length: biteTraces }, () =>
        Array.from({ length: responseWindows }, () => READOUT_FEATURES.map(() => 9)),
      ),
    },
    decoy: {
      seeds: [],
      traces: Array.from({ length: biteTraces }, () =>
        Array.from({ length: responseWindows }, () => READOUT_FEATURES.map(() => 5)),
      ),
    },
  };
}

// --- the splice -------------------------------------------------------------
{
  const cache = fakeCache();
  const events = buildSchedule(42);
  const rows = spliceEpisode({ cache, events, seed: 42 });

  assert.equal(rows.length, WINDOWS, "one row per decision window");
  assert.ok(
    rows.every((row) => row.length === READOUT_FEATURES.length),
    "every row is one rate per readout population",
  );

  // Each event's own response lands exactly where that event is.
  assert.ok(
    events.some((event) => event.type === "decoy") && events.some((event) => event.type === "bite"),
    "this fixture schedule should contain both kinds of event",
  );
  for (const event of events) {
    const index = Math.round(event.atMs / TASK.windowMs);
    const expected = event.type === "bite" ? 9 : 5;
    assert.equal(rows[index][0], expected, `wrong response at the ${event.type} at ${event.atMs} ms`);
  }
  // A window a long way from any bite is still baseline.
  const quiet = Math.round((TASK.leadInMs - TASK.windowMs) / TASK.windowMs);
  assert.notEqual(rows[quiet][0], 9, "the lead-in should be baseline");
}

// Splicing is a pure function of (cache, events, seed).
{
  const cache = fakeCache();
  const events = buildSchedule(7);
  const digest = (rows) => rows.map((row) => row[0]).join(",");
  assert.equal(
    digest(spliceEpisode({ cache, events, seed: 7 })),
    digest(spliceEpisode({ cache, events, seed: 7 })),
    "the same seed must splice the same episode",
  );
  assert.notEqual(
    digest(spliceEpisode({ cache, events, seed: 7 })),
    digest(spliceEpisode({ cache, events, seed: 8 })),
    "a different seed should pick a different trace or offset",
  );
}

// The splice must refuse a cache it cannot interpret rather than quietly
// producing rows on the wrong grid.
{
  const cache = fakeCache();
  assert.throws(
    () => spliceEpisode({ cache: { ...cache, schemaVersion: 99 }, events: [], seed: 1 }),
    /cache schema/,
  );
  assert.throws(
    () => spliceEpisode({ cache: { ...cache, windowMs: 17 }, events: [], seed: 1 }),
    /cache window/,
  );
  assert.throws(
    () =>
      spliceEpisode({
        cache: { ...cache, baseline: { traces: [[[0, 0, 0, 0, 0, 0, 0, 0]]] } },
        events: [],
        seed: 1,
      }),
    /shorter than an episode/,
  );

  // A cache recorded before decoys existed must say so rather than splicing a
  // bite response in where a decoy belongs.
  const { decoy, ...withoutDecoys } = cache;
  void decoy;
  assert.throws(
    () =>
      spliceEpisode({
        cache: withoutDecoys,
        events: [{ atMs: 2000, type: "decoy" }],
        seed: 1,
      }),
    /no "decoy" responses/,
  );
}

// --- the residual is reported per feature -----------------------------------
{
  const rows = spliceResidual(fakeCache({ baselineTraces: 1 }), "bite");
  assert.equal(rows.length, READOUT_FEATURES.length);
  assert.deepEqual(rows.map((row) => row.feature), [...READOUT_FEATURES]);
  // The fixture's baseline ramps 1 + w/10000 over its windows, so its mean is
  // just above 1 and the residual is the response tail minus that.
  assert.ok(rows[0].baselineHz > 1 && rows[0].baselineHz < 1.2, "baseline mean");
  assert.equal(rows[0].responseTailHz, 9);
  assert.ok(
    Math.abs(rows[0].residualHz - (9 - rows[0].baselineHz)) < 0.02,
    "residual is tail minus baseline",
  );
}

// --- the bobber track -------------------------------------------------------
{
  const events = [{ atMs: 1000, type: "bite" }];
  const track = bobberTrack(events, WINDOWS);
  assert.equal(track.length, WINDOWS);
  assert.equal(track[0], 0, "the bobber floats before the bite");
  assert.ok(Math.max(...track) > 0.9, "a real bite pulls it right under");

  // A decoy dips the bobber too, or the dip-reacting baseline would be an
  // oracle in disguise and the discrimination task would be given away.
  const decoyTrack = bobberTrack([{ atMs: 1000, type: "decoy" }], WINDOWS);
  assert.ok(Math.max(...decoyTrack) > 0.2, "a decoy must visibly dip the bobber");
  assert.ok(
    Math.max(...decoyTrack) < Math.max(...track),
    "and dip it less than a real bite does",
  );
  assert.ok(
    track[Math.round(1000 / TASK.windowMs)] < track[Math.round(1150 / TASK.windowMs)],
    "the dip should deepen after onset",
  );
  assert.equal(track.at(-1), 0, "and come back up");
}

// --- the ablations ----------------------------------------------------------
{
  const weights = Int16Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
  const shuffled = shuffleWeights(weights, 99);
  assert.notDeepEqual([...shuffled], [...weights], "the ablation has to change something");
  assert.deepEqual(
    [...shuffled].sort((a, b) => a - b),
    [...weights].sort((a, b) => a - b),
    "shuffling weights must preserve the multiset, so total drive is unchanged",
  );
  assert.deepEqual([...weights], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "the source array is not mutated");
  assert.deepEqual([...shuffleWeights(weights, 99)], [...shuffled], "reproducible from its seed");
}
{
  const manifest = {
    neuron_count: 1000,
    populations: { looming_visual_proxy: [1, 2, 3], escape_giant_fiber: [4, 5] },
  };
  const ablated = shuffleInputMapping(manifest, 5, 1000);
  for (const [name, indices] of Object.entries(manifest.populations)) {
    assert.equal(ablated.populations[name].length, indices.length, `${name} keeps its size`);
    assert.equal(new Set(ablated.populations[name]).size, indices.length, "no duplicate neurons");
    assert.ok(
      ablated.populations[name].every((index) => index >= 0 && index < 1000),
      "indices stay in range",
    );
  }
  assert.deepEqual(manifest.populations.looming_visual_proxy, [1, 2, 3], "the manifest is copied");
  assert.deepEqual(shuffleInputMapping(manifest, 5, 1000), ablated, "reproducible from its seed");
}

console.log("cache: all assertions passed");
