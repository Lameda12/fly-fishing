import assert from "node:assert/strict";

import { READOUT_FEATURES, STIMULUS_CHANNELS } from "../fishing/brain-host.mjs";
import { BASELINE_MS, CACHE_SCHEMA_VERSION, RESPONSE_MS } from "../fishing/cache.mjs";
import { TASK } from "../fishing/task.mjs";
import {
  STAGES,
  STAGE_STARTS,
  STIMULI,
  VOYAGE_MS,
  buildStageEvents,
  createVoyage,
  stageAt,
} from "../fishing/voyage.mjs";
import { restingRates } from "../fishing/cache.mjs";
import {
  FEATURE_COUNT,
  buildFeatures,
  createPolicy,
  createStagedReadout,
} from "../fishing/readout.mjs";
import { runVoyage, voyageOraclePolicy, voyageReflexPolicy } from "../fishing/voyage-episode.mjs";

// --- the stage table --------------------------------------------------------
{
  assert.equal(VOYAGE_MS, TASK.episodeMs, "the voyage must fill exactly one episode");
  assert.deepEqual(
    STAGES.map((stage) => stage.id),
    ["prep", "fish", "row", "cook", "eat"],
    "prep, fish, row back, cook, eat",
  );
  assert.deepEqual(STAGE_STARTS, [0, 10_000, 35_000, 40_000, 50_000]);

  for (const stage of STAGES) {
    if (!stage.decision) {
      assert.equal(stage.good, null, `${stage.id} is transit and scores nothing`);
      continue;
    }
    assert.ok(STIMULI[stage.good], `${stage.id}.good must be a recorded stimulus`);
    if (stage.bad !== null) {
      assert.ok(STIMULI[stage.bad], `${stage.id}.bad must be a recorded stimulus`);
      assert.notEqual(stage.good, stage.bad, `${stage.id} needs two distinguishable events`);
    }
  }
  // Exactly one stage is transit, and it is the row back.
  assert.deepEqual(
    STAGES.filter((stage) => !stage.decision).map((stage) => stage.id),
    ["row"],
  );
}

// Every stimulus must address channels the simulator actually has.
{
  for (const [name, pulse] of Object.entries(STIMULI)) {
    for (const channel of Object.keys(pulse)) {
      assert.ok(STIMULUS_CHANNELS.includes(channel), `${name} uses unknown channel ${channel}`);
    }
    assert.ok(Object.keys(pulse).length > 0, `${name} must drive something`);
  }
  // The four decision stages must each use a different channel, or the shared
  // readout could not tell which stage it is in from the rates alone.
  const channelOf = (stimulus) => Object.keys(STIMULI[stimulus])[0].replace(/(Left|Right)Hz$/, "");
  const used = STAGES.filter((s) => s.decision).map((s) => channelOf(s.good));
  assert.equal(new Set(used).size, used.length, `stages share a channel: ${used.join(", ")}`);
}

// --- stageAt ----------------------------------------------------------------
{
  assert.equal(stageAt(0).id, "prep");
  assert.equal(stageAt(9_999).id, "prep");
  assert.equal(stageAt(10_000).id, "fish");
  assert.equal(stageAt(34_999).id, "fish");
  assert.equal(stageAt(35_000).id, "row");
  assert.equal(stageAt(40_000).id, "cook");
  assert.equal(stageAt(50_000).id, "eat");
  assert.equal(stageAt(59_999).id, "eat");
}

// --- stage events -----------------------------------------------------------
{
  const fish = STAGES.find((stage) => stage.id === "fish");
  const events = buildStageEvents(fish, 1234);
  assert.ok(events.length >= 2, "the fishing stage should present several chances");
  for (const event of events) {
    assert.equal(event.stage, "fish");
    assert.ok(event.atMs >= STAGE_STARTS[1], "inside its own stage");
    assert.ok(event.atMs < STAGE_STARTS[1] + fish.durationMs, "and not past the end");
    assert.ok([fish.good, fish.bad].includes(event.stimulus));
  }
  assert.deepEqual(buildStageEvents(fish, 1234), events, "a seed reproduces its schedule");

  // The count cap is how the loop closes: it is what a thin catch imposes on
  // the stages downstream of it.
  const cook = STAGES.find((stage) => stage.id === "cook");
  const uncapped = buildStageEvents(cook, 55).length;
  assert.ok(uncapped >= 2, `the cook stage should have room for several, got ${uncapped}`);
  assert.equal(buildStageEvents(cook, 55, { count: 1 }).length, 1, "one fish, one thing to cook");
  assert.equal(buildStageEvents(fish, 1234, { count: 0 }).length, 0, "nothing caught, nothing to do");

  // The transit stage never presents anything, whatever it is asked for.
  const row = STAGES.find((stage) => stage.id === "row");
  assert.deepEqual(buildStageEvents(row, 1234, { count: 99 }), []);

  // A stage with no bad type only ever presents its good one.
  assert.ok(
    buildStageEvents(cook, 7).every((event) => event.stimulus === cook.good),
    "cooking has no weaker version to confuse it with",
  );
}

// --- scoring ----------------------------------------------------------------
// Acting on the rewarding event pays; acting on the other one costs.
{
  const voyage = createVoyage([
    { atMs: 1_000, stage: "prep", stimulus: "bait-firm", rewarding: true },
    { atMs: 5_000, stage: "prep", stimulus: "bait-slip", rewarding: false },
  ]);
  assert.equal(voyage.step(1_100, 1).outcome, "good");
  // The re-cast runs to 1100 + 50 + 1500, so by 5100 the fly is free to act
  // again, and acting on the slipping bait is a mistake rather than a no-op.
  assert.ok(!voyage.canDecide(2_600), "still re-gripping just before the re-cast ends");
  assert.ok(voyage.canDecide(2_650), "free again at the end of it");
  assert.equal(voyage.step(5_100, 1).outcome, "bad");
  const summary = voyage.finish();
  assert.equal(summary.perStage.prep.correct, 1);
  assert.equal(summary.perStage.prep.wrong, 1);
  assert.equal(summary.perStage.prep.chances, 1, "only the firm bait was a chance");
  assert.equal(summary.totalReward, TASK.rewardCatch + TASK.rewardSnap);
}
{
  const voyage = createVoyage([
    { atMs: 1_000, stage: "prep", stimulus: "bait-slip", rewarding: false },
  ]);
  const result = voyage.step(1_100, 1);
  assert.equal(result.outcome, "bad");
  assert.equal(result.onEvent, "bait-slip", "the mistake names what it fell for");
  assert.equal(voyage.finish().perStage.prep.wrong, 1);
}
// Acting in the transit stage is not even a decision.
{
  const voyage = createVoyage([]);
  const result = voyage.step(36_000, 1);
  assert.equal(result.decision, false);
  assert.equal(result.outcome, "transit");
  assert.equal(result.reward, 0);
  assert.equal(voyage.totalReward, 0);
}
// Events added after the fact are counted.
{
  const voyage = createVoyage([]);
  voyage.addEvents([{ atMs: 41_000, stage: "cook", stimulus: "pan-ready", rewarding: true }]);
  assert.equal(voyage.step(41_100, 1).outcome, "good");
  assert.equal(voyage.finish().perStage.cook.chances, 1);
}

// --- the loop actually closes ----------------------------------------------
/** A cache whose every stimulus reads as a distinct constant. */
function fakeCache() {
  const baselineWindows = Math.round(BASELINE_MS / TASK.windowMs);
  const responseWindows = Math.round(RESPONSE_MS / TASK.windowMs);
  const flat = (value) =>
    Array.from({ length: responseWindows }, () => READOUT_FEATURES.map(() => value));
  const cache = {
    schemaVersion: CACHE_SCHEMA_VERSION,
    windowMs: TASK.windowMs,
    featureNames: [...READOUT_FEATURES],
    ablation: "none",
    baseline: {
      traces: [Array.from({ length: baselineWindows }, () => READOUT_FEATURES.map(() => 1))],
    },
  };
  Object.keys(STIMULI).forEach((name, i) => {
    cache[name] = { traces: [flat(10 + i), flat(11 + i)] };
  });
  return cache;
}

{
  const cache = fakeCache();
  const oracle = runVoyage({
    cache,
    seed: 4242,
    policy: ({ events }) => voyageOraclePolicy({ scorerEvents: events }),
  });
  const landed = oracle.summary.perStage.fish.correct;
  assert.ok(landed > 0, "the oracle should land fish");
  assert.equal(oracle.summary.perStage.fish.wrong, 0, "and never take a decoy");
  // The catch caps what the later stages may present. The stage's own length
  // caps it too, so this is an upper bound rather than an equality.
  assert.ok(
    oracle.summary.perStage.cook.chances <= landed,
    `cooked ${oracle.summary.perStage.cook.chances} with only ${landed} landed`,
  );
  assert.ok(oracle.summary.perStage.eat.chances <= landed, "and no more to eat than was cooked");
  assert.ok(oracle.summary.perStage.cook.chances > 0, "something was landed, so something cooks");
}

// Catch nothing and there is nothing to cook or eat: the loop is a loop.
{
  const cache = fakeCache();
  const idle = runVoyage({ cache, seed: 4242, policy: { kind: "idle", act: () => 0 } });
  assert.equal(idle.summary.perStage.fish.correct, 0);
  assert.equal(idle.summary.perStage.cook.chances, 0, "an empty net means an empty pan");
  assert.equal(idle.summary.perStage.eat.chances, 0);
  assert.equal(idle.summary.totalReward, 0, "doing nothing is free, it just earns nothing");
}

// The reflex baseline takes every chance and every trap, so it scores worse
// than the oracle while acting more often.
{
  const cache = fakeCache();
  const reflex = runVoyage({
    cache,
    seed: 4242,
    policy: ({ events }) => voyageReflexPolicy({ scorerEvents: events }),
  });
  const oracle = runVoyage({
    cache,
    seed: 4242,
    policy: ({ events }) => voyageOraclePolicy({ scorerEvents: events }),
  });
  assert.ok(reflex.summary.wrong > 0, "the reflex must fall for the traps");
  assert.ok(
    reflex.summary.totalReward < oracle.summary.totalReward,
    `reflex ${reflex.summary.totalReward} should score under oracle ${oracle.summary.totalReward}`,
  );
}

// Every decision carries the stage it was made in, so per-stage credit works.
{
  const cache = fakeCache();
  const run = runVoyage({
    cache,
    seed: 99,
    policy: ({ events }) => voyageOraclePolicy({ scorerEvents: events }),
  });
  assert.ok(run.decisions.length > 0);
  assert.ok(
    run.decisions.every((decision) => ["prep", "fish", "cook", "eat"].includes(decision.stage)),
    "no decision may be attributed to the transit stage",
  );
  assert.equal(
    run.rows.length,
    Math.round(VOYAGE_MS / TASK.windowMs),
    "one feature row per window of the voyage",
  );
}

// --- the per-stage window ---------------------------------------------------
//
// Cooking gets a longer window than the rest because the network answers odor
// tonically rather than phasically; see the stage table. The point of the test
// is that only cooking is special and that the longer window actually reaches
// the scorer, not that 2500 is the right number.
{
  const cook = STAGES.find((stage) => stage.id === "cook");
  assert.equal(cook.windowMs, 2_500);
  for (const stage of STAGES.filter((s) => s.decision && s.id !== "cook")) {
    assert.equal(stage.windowMs, undefined, `${stage.id} should inherit the task window`);
  }

  const scorer = createVoyage(
    [{ atMs: 41_000, stage: "cook", stimulus: "pan-ready", rewarding: true }],
    { task: TASK },
  );
  // Well past the 500 ms the other stages get, and still inside cooking's own.
  assert.equal(scorer.step(42_000, 1).outcome, "good", "cooking pays late in its window");

  const late = createVoyage(
    [{ atMs: 41_000, stage: "cook", stimulus: "pan-ready", rewarding: true }],
    { task: TASK },
  );
  assert.equal(late.step(43_600, 1).outcome, "bad", "and stops paying once it closes");
}

// --- resting rates and centred features -------------------------------------
{
  const cache = fakeCache();
  const resting = restingRates(cache);
  assert.equal(resting.length, READOUT_FEATURES.length);

  const rates = resting.map((rate) => rate + 10);
  const centred = buildFeatures(rates, resting, resting);
  for (let i = 0; i < READOUT_FEATURES.length; i++) {
    assert.ok(Math.abs(centred[i] - 0.1) < 1e-9, "a rate 10 Hz over rest reads as 0.1");
    assert.ok(Math.abs(centred[READOUT_FEATURES.length + i]) < 1e-9, "rest itself reads as 0");
  }
  // Raw is still available, and is what the single-stage task is trained on.
  const raw = buildFeatures(rates, resting, null);
  assert.ok(Math.abs(raw[0] - rates[0] / 100) < 1e-9);
  assert.equal(centred[centred.length - 1], 1, "the bias stays 1 either way");
}

// --- per-stage readout heads ------------------------------------------------
{
  const stageIds = STAGES.filter((stage) => stage.decision).map((stage) => stage.id);
  const staged = createStagedReadout({ stageIds });
  assert.equal(staged.head("cook").weights.length, FEATURE_COUNT);
  assert.throws(() => staged.head("row"), /no readout head/, "transit has no head");

  // The heads are independent: moving one must not move another.
  staged.head("cook").weights[0] = 99;
  assert.equal(staged.head("eat").weights[0], 0);

  const json = staged.toJSON();
  assert.equal(json.perStage, true);
  assert.deepEqual(Object.keys(json.weights), stageIds);

  // A staged policy dispatches on the stage it is handed, and nothing else.
  const zeros = new Array(READOUT_FEATURES.length).fill(0);
  const features = buildFeatures(zeros, zeros, null);
  staged.head("cook").weights.fill(0);
  staged.head("cook").weights[staged.head("cook").weights.length - 1] = 40;
  const greedy = createPolicy("stagedReadoutGreedy", { task: TASK, staged });
  assert.equal(greedy.act(0, features, "cook"), 1, "a saturated head acts");
  assert.equal(greedy.act(0, features, "eat"), 0, "its neighbour is untouched");
}

// --- the voyage hands the stage to the policy -------------------------------
{
  const cache = fakeCache();
  const seen = new Set();
  runVoyage({
    cache,
    seed: 7,
    collect: true,
    policy: {
      act: (_tMs, _features, stage) => {
        seen.add(stage);
        return 0;
      },
    },
  });
  assert.deepEqual([...seen].sort(), ["cook", "eat", "fish", "prep"].sort());
  assert.ok(!seen.has("row"), "transit never asks the policy");
}

console.log("voyage: all assertions passed");
