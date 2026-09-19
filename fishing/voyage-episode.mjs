// Run one voyage: five stages, one shared readout, and a loop that closes.
//
// The single-stage task splices a whole episode up front, because its schedule
// is fixed before anything happens. A voyage cannot do that: how much there is
// to cook depends on how many fish were landed, and that depends on what the
// readout did in the fishing stage. So the episode is spliced **as it is
// played** — prep and fishing first, then the cook and eat stages once the net
// has been counted.
//
// That does not weaken the argument for why the cache is exact. The claim was
// always about the response, not the schedule: the network's answer to a given
// stimulus does not depend on what the readout decided, so replaying a recorded
// answer is replaying real output. What is now policy-dependent is *which*
// stimuli occur, and those are spliced from the same real recordings the moment
// they are decided. Nothing is interpolated and nothing is modelled.

import { assertUsableCache, baselineRows, overlayResponse } from "./cache.mjs";
import { buildFeatures } from "./readout.mjs";
import { createRng, deriveSeed } from "./rng.mjs";
import { TASK } from "./task.mjs";
import {
  STAGES,
  STAGE_STARTS,
  VOYAGE_MS,
  buildStageEvents,
  createVoyage,
  stageAt,
} from "./voyage.mjs";

const ZERO_RATES = Object.freeze([0, 0, 0, 0, 0, 0, 0, 0]);

const stageIndex = (id) => STAGES.findIndex((stage) => stage.id === id);
const windowOf = (tMs, task) => Math.round(tMs / task.windowMs);

/**
 * Play one voyage against a cache.
 *
 * `policy` may be a policy object, or a factory taking `{ events }` where
 * `events()` returns the schedule so far. The baselines need the factory form,
 * because the schedule they react to does not fully exist until the fishing
 * stage has been played.
 *
 * Returns the decisions (for REINFORCE), the per-stage summary, and optionally
 * a per-window trace for the recorder.
 */
export function runVoyage({ cache, seed, policy: policySpec, task = TASK, collect = false }) {
  assertUsableCache(cache, task);
  const rng = createRng(deriveSeed(seed, "voyage-splice"));
  const windows = Math.round(VOYAGE_MS / task.windowMs);

  // One recorded stretch of background for the whole voyage; responses are laid
  // onto it as each stage's events become known.
  const rows = baselineRows(cache, rng, windows);

  const prep = STAGES[stageIndex("prep")];
  const fish = STAGES[stageIndex("fish")];
  const cook = STAGES[stageIndex("cook")];
  const eat = STAGES[stageIndex("eat")];

  const scorer = createVoyage([], { task });
  const policy =
    typeof policySpec === "function" ? policySpec({ events: () => scorer.events }) : policySpec;

  /** Schedule a stage's events and splice their recorded responses in. */
  const present = (stage, count) => {
    const events = buildStageEvents(stage, deriveSeed(seed, "stage", stage.id), { task, count });
    scorer.addEvents(events);
    for (const event of events) {
      overlayResponse(rows, cache, event.stimulus, windowOf(event.atMs, task), rng);
    }
    return events;
  };

  present(prep, Infinity);
  present(fish, Infinity);

  const decisions = [];
  const trace = collect ? [] : null;
  let previousRates = ZERO_RATES;
  let laterStagesScheduled = false;

  for (let index = 0; index < windows; index++) {
    const tMs = index * task.windowMs;

    // The moment the fishing stage is over, the catch is known, so the stages
    // that depend on it can be scheduled and spliced. Everything after this
    // point in the episode reads rows that were written just now.
    if (!laterStagesScheduled && tMs >= STAGE_STARTS[stageIndex("cook")]) {
      const landed = scorer.perStage.fish.correct;
      present(cook, landed);
      present(eat, landed);
      laterStagesScheduled = true;
    }

    const rates = rows[index] ?? ZERO_RATES;
    const features = buildFeatures(rates, previousRates);
    previousRates = rates;

    const stage = stageAt(tMs);
    const open = stage.decision && scorer.canDecide(tMs);
    const action = open ? policy.act(tMs, features) : 0;
    const result = scorer.step(tMs, action);

    if (result.decision) {
      decisions.push({
        features,
        action: result.action,
        probability: policy.probability ? policy.probability(features) : result.action,
        reward: result.reward,
        stage: result.stage,
      });
    }
    if (collect) {
      trace.push({
        tMs,
        stage: stage.id,
        rates,
        probability: policy.probability ? policy.probability(features) : null,
        action: result.action,
        outcome: result.outcome,
      });
    }
  }

  return { summary: scorer.finish(), decisions, trace, events: scorer.events, rows };
}

/**
 * An oracle for the voyage: acts on every rewarding event and nothing else.
 *
 * It needs the event list, which for the later stages does not exist until the
 * fishing stage has been played, so it reads the scorer's growing list rather
 * than being handed a fixed schedule. That is also why it cannot be built with
 * `createPolicy` like the single-stage baselines.
 */
export function voyageOraclePolicy({ scorerEvents, task = TASK, delayMs = 150 }) {
  return {
    kind: "voyageOracle",
    act: (tMs) =>
      scorerEvents().some(
        (event) =>
          event.rewarding &&
          tMs >= event.atMs + delayMs &&
          tMs < event.atMs + delayMs + task.windowMs,
      )
        ? 1
        : 0,
  };
}

/**
 * The "reacts to the world, not to the fly" baseline, per stage: something
 * happened, so act. It cannot tell a firm bait from a slipping one, a bite from
 * a decoy, or a good morsel from a burnt one, so it takes every chance and every
 * trap.
 */
export function voyageReflexPolicy({ scorerEvents, task = TASK, delayMs = 150 }) {
  return {
    kind: "voyageReflex",
    act: (tMs) =>
      scorerEvents().some(
        (event) =>
          tMs >= event.atMs + delayMs && tMs < event.atMs + delayMs + task.windowMs,
      )
        ? 1
        : 0,
  };
}
