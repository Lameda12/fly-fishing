import assert from "node:assert/strict";

import { READOUT_FEATURES } from "../fishing/brain-host.mjs";
import {
  FEATURE_COUNT,
  FEATURE_NAMES,
  RATE_SCALE,
  buildFeatures,
  createPolicy,
  createReadout,
  createTrainer,
  rewardsToGo,
  sigmoid,
} from "../fishing/readout.mjs";
import { TASK, buildSchedule } from "../fishing/task.mjs";

// --- the feature row --------------------------------------------------------
{
  assert.equal(FEATURE_COUNT, READOUT_FEATURES.length * 2 + 1);
  assert.equal(FEATURE_NAMES.at(-1), "bias");
  assert.equal(new Set(FEATURE_NAMES).size, FEATURE_COUNT, "feature names must be unique");

  const now = [1, 2, 3, 4, 5, 6, 7, 8];
  const before = [8, 7, 6, 5, 4, 3, 2, 1];
  const row = buildFeatures(now, before);
  assert.equal(row.length, FEATURE_COUNT);
  assert.equal(row[0], 1 / RATE_SCALE);
  assert.equal(row[7], 8 / RATE_SCALE);
  assert.equal(row[8], 8 / RATE_SCALE, "the lagged block follows the current one");
  assert.equal(row.at(-1), 1, "bias");
}

// --- sigmoid does not overflow ---------------------------------------------
assert.equal(sigmoid(0), 0.5);
assert.ok(sigmoid(1e6) > 0.999 && Number.isFinite(sigmoid(1e6)));
assert.ok(sigmoid(-1e6) < 0.001 && Number.isFinite(sigmoid(-1e6)));

// --- an untrained readout mostly waits -------------------------------------
{
  const readout = createReadout();
  const quiet = buildFeatures([0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]);
  assert.ok(readout.probability(quiet) < 0.1, "a fresh readout should not spam hooks");
  assert.equal(readout.weights.length, FEATURE_COUNT);
  assert.deepEqual(readout.toJSON().featureNames, [...FEATURE_NAMES]);
}
assert.throws(() => createReadout({ weights: [1, 2, 3] }), /needs 17 weights/);

// --- reward-to-go -----------------------------------------------------------
{
  assert.deepEqual(rewardsToGo([0, 0, 1], 1), [1, 1, 1]);
  assert.deepEqual(rewardsToGo([1, 0, 0], 0.5), [1, 0, 0]);
  const discounted = rewardsToGo([0, 1], 0.5);
  assert.equal(discounted[0], 0.5);
  assert.equal(discounted[1], 1);
}

// --- REINFORCE moves the weights the right way ------------------------------
// One feature is on whenever hooking pays. A correct policy gradient has to
// raise that feature's weight when hooking on it was rewarded, and lower it
// when the same action was punished.
{
  const hot = buildFeatures([0, 0, 0, 0, 0, 0, 100, 0], [0, 0, 0, 0, 0, 0, 0, 0]);
  const escapeIndex = READOUT_FEATURES.indexOf("escape_giant_fiber");

  const rewarded = createReadout();
  createTrainer({ readout: rewarded, learningRate: 1, gamma: 0 }).update([
    { features: hot, action: 1, probability: 0.5, reward: 1 },
    { features: hot, action: 0, probability: 0.5, reward: -1 },
  ]);
  assert.ok(
    rewarded.weights[escapeIndex] > 0,
    "hooking on the informative feature paid, so its weight should rise",
  );

  const punished = createReadout();
  createTrainer({ readout: punished, learningRate: 1, gamma: 0 }).update([
    { features: hot, action: 1, probability: 0.5, reward: -1 },
    { features: hot, action: 0, probability: 0.5, reward: 1 },
  ]);
  assert.ok(
    punished.weights[escapeIndex] < 0,
    "hooking on it cost, so its weight should fall",
  );
}

// An episode where nothing happened must not move the weights at all.
{
  const readout = createReadout();
  const before = [...readout.weights];
  const trainer = createTrainer({ readout });
  trainer.update([]);
  assert.deepEqual([...readout.weights], before, "no decisions, no update");
}

// --- the policies -----------------------------------------------------------
{
  const events = buildSchedule(7);
  const bites = events.filter((event) => event.type === "bite");
  assert.ok(bites.length && bites.length < events.length, "the fixture needs both kinds");

  const fireTimes = (policy) => {
    const fired = [];
    for (let t = 0; t < TASK.episodeMs; t += TASK.windowMs) if (policy.act(t)) fired.push(t);
    return fired;
  };

  // The oracle hooks once per real fish and never on a decoy.
  const oracle = fireTimes(createPolicy("oracle", { task: TASK, events }));
  assert.equal(oracle.length, bites.length, "the oracle hooks once per bite");
  for (let i = 0; i < bites.length; i++) {
    const delay = oracle[i] - bites[i].atMs;
    assert.ok(delay >= 0 && delay < TASK.hookWindowMs, `oracle hook ${delay} ms after the bite`);
  }

  // The dip-delay baseline reacts to the bobber, so it hooks on everything.
  const dip = fireTimes(createPolicy("dipDelay", { task: TASK, events }));
  assert.equal(dip.length, events.length, "dip-delay hooks on every fish event");
  assert.ok(dip.length > oracle.length, "which is more often than the oracle does");

  // Rate-matched random: its expected hook count is the number of catchable fish.
  const random = createPolicy("random", { task: TASK, events, seed: 3 });
  assert.ok(Math.abs(random.hookProbability * 1200 - bites.length) < 1e-9);

  const fixed = createPolicy("fixedInterval", { task: TASK, events, intervalMs: 5000 });
  let hooks = 0;
  for (let t = 0; t < TASK.episodeMs; t += TASK.windowMs) hooks += fixed.act(t);
  assert.equal(hooks, 11, "hooks at 5 s, 10 s ... 55 s of a 60 s episode");

  assert.throws(() => createPolicy("readout", { task: TASK }), /needs a readout/);
  assert.throws(() => createPolicy("nonsense", { task: TASK }), /unknown policy/);
}

// A greedy readout is deterministic; a sampled one is not.
{
  const readout = createReadout({ weights: new Array(FEATURE_COUNT).fill(0) });
  const greedy = createPolicy("readoutGreedy", { task: TASK, readout });
  const features = buildFeatures([0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]);
  assert.equal(greedy.act(0, features), 0, "p = 0.5 is not greater than 0.5");
  assert.equal(greedy.probability(features), 0.5);
}

console.log("readout: all assertions passed");
