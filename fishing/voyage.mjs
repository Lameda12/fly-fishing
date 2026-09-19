// The voyage: one trip out in the boat, as five stages.
//
//   prep   bait the hook          touchHz   -> groom_adn1
//   fish   catch fish             loomHz    -> escape_giant_fiber
//   row    row back to the jetty  (nothing) -- no decision, see below
//   cook   take the pan off       odorHz    -> turn_left / turn_right
//   eat    swallow or spit out    sugarHz   -> feed_mn9
//
// Why these channels. A dose-response sweep of every stimulus channel upstream
// exposes, against every readout population it exposes, came out close to
// one-hot: looming drives the giant fiber and nothing else, sugar drives MN9
// and nothing else, touch drives the grooming readout and nothing else, and
// odor drives the turn readouts and nothing else. So a stage's identity is
// legible from the descending rates alone, and one shared readout can serve all
// four decision stages. That is a convenience the network happened to hand us,
// not a hard problem solved, and the README says so.
//
// Two honest limits, both forced by measurement rather than chosen:
//
// **Rowing back has no decision.** Steering the body would mean closing a loop
// through the turn readouts, and that interface is the one upstream's own
// roadmap lists for calibration; the sibling repository measured a constant
// -0.6 offset on it that made chemotaxis arc instead of close. Rather than
// invent a decision the simulator cannot support, the row stage is transit:
// the boat moves, nothing is scored, and the README says it is scenery.
//
// **Cooking is a detection, not a judgement of doneness.** Odor saturates
// almost immediately: 5 Hz of input already puts turn_left at 30 Hz and 220 Hz
// only reaches 35 Hz. There is no graded "how done is it" signal to read, so
// the cook stage asks when the smell arrives, not how strong it is. The graded
// discriminations live in the three stages whose channels are actually graded.
//
// This module imports nothing from Node.

import { createRng } from "./rng.mjs";
import { TASK } from "./task.mjs";

/**
 * Every stimulus the voyage can present, as the channel settings that produce
 * it. The cache records one response set per key, and the splice looks events
 * up here, so adding an event type is a table entry plus a recording.
 *
 * Amplitudes are picked off the measured dose-response so that the two
 * outcomes of each graded stage land far apart on their population:
 *
 *   bait-firm 220 Hz touch -> groom_adn1 ~117    bait-slip 60 Hz -> ~75
 *   bite      150 Hz loom  -> escape_gf   ~199   decoy     60 Hz -> ~160
 *   morsel-good 100 Hz sugar -> feed_mn9  ~105   morsel-burnt 25 Hz -> ~50
 */
export const STIMULI = Object.freeze({
  // Sugar and odor are driven on both sides: the sibling repository measured
  // that MN9 does not respond to one-sided sugar, matching upstream's own
  // sensor model, and the pan is in front of the fly rather than beside it.
  "bait-firm": { touchHz: 220 },
  "bait-slip": { touchHz: 60 },
  bite: { loomHz: 150 },
  decoy: { loomHz: 60 },
  "pan-ready": { odorLeftHz: 145, odorRightHz: 145 },
  "morsel-good": { sugarLeftHz: 100, sugarRightHz: 100 },
  "morsel-burnt": { sugarLeftHz: 25, sugarRightHz: 25 },
});

export const STIMULUS_NAMES = Object.freeze(Object.keys(STIMULI));

/**
 * The stages, in order. Durations sum to the 60 s episode the rest of the
 * repository already uses, so the recorded baseline traces are reused as they
 * are rather than re-recorded.
 *
 * `reward` is the stage's own name for a correct act; `wrong` is what acting on
 * the other event type costs. Every stage uses the same +1 / -0.5 / 0 shape so
 * that one shared readout gets consistent credit across all four.
 */
export const STAGES = Object.freeze([
  {
    id: "prep",
    name: "Bait the hook",
    durationMs: 10_000,
    decision: true,
    /** Acting on `good` pays; acting on `bad` costs; letting either go is 0. */
    good: "bait-firm",
    bad: "bait-slip",
    act: "grip",
    meanGapMs: 2_600,
  },
  {
    id: "fish",
    name: "Fishing",
    durationMs: 25_000,
    decision: true,
    good: "bite",
    bad: "decoy",
    act: "hook",
    meanGapMs: 4_200,
  },
  {
    id: "row",
    name: "Row back",
    durationMs: 5_000,
    decision: false,
    good: null,
    bad: null,
    act: null,
    meanGapMs: 0,
  },
  {
    id: "cook",
    name: "Cook the catch",
    durationMs: 10_000,
    decision: true,
    good: "pan-ready",
    // Detection, not doneness: odor saturates, so there is no weaker version of
    // this smell for the readout to tell apart. Acting with no smell present is
    // the mistake, and that is scored the same as any other false act.
    bad: null,
    act: "lift the pan",
    meanGapMs: 3_400,
  },
  {
    id: "eat",
    name: "Eat",
    durationMs: 10_000,
    decision: true,
    good: "morsel-good",
    bad: "morsel-burnt",
    act: "swallow",
    meanGapMs: 2_800,
  },
]);

export const VOYAGE_MS = STAGES.reduce((total, stage) => total + stage.durationMs, 0);

/** Where each stage starts, in episode time. */
export const STAGE_STARTS = Object.freeze(
  STAGES.reduce(
    (starts, stage) => [...starts, starts[starts.length - 1] + stage.durationMs],
    [0],
  ).slice(0, STAGES.length),
);

export function stageAt(tMs) {
  for (let i = STAGES.length - 1; i >= 0; i--) {
    if (tMs >= STAGE_STARTS[i]) return STAGES[i];
  }
  return STAGES[0];
}

/**
 * The events for one stage, drawn the same way the single-stage task draws its
 * bites: exponential gaps with a minimum, so nothing is on a rhythm.
 *
 * `count` caps how many events the stage may present. The cook and eat stages
 * pass the number of fish actually caught, which is how the loop closes: an
 * empty net means nothing to cook and nothing to eat.
 */
export function buildStageEvents(stage, seed, { task = TASK, count = Infinity } = {}) {
  if (!stage.decision || count <= 0) return [];
  const rng = createRng(seed);
  const start = STAGE_STARTS[STAGES.indexOf(stage)];
  const events = [];
  let at = start + task.leadInMs / 2 + rng.range(0, stage.meanGapMs / 2);
  const end = start + stage.durationMs - task.hookWindowMs;

  while (at < end && events.length < count) {
    // A stage with no `bad` type presents only its good one.
    const bad = stage.bad !== null && rng.next() < task.decoyShare;
    events.push({
      atMs: Math.round(at),
      stage: stage.id,
      stimulus: bad ? stage.bad : stage.good,
      rewarding: !bad,
    });
    at += task.minBiteGapMs / 2 + rng.exponential(stage.meanGapMs);
  }
  return events;
}

/**
 * The scorer for one voyage.
 *
 * Structurally the same state machine as the single-stage task: act inside an
 * event's window, and it pays if that event was the rewarding one. What differs
 * is that the events carry a stage and a stimulus, and the tally is kept per
 * stage as well as overall, because "it learned to fish but not to eat" is the
 * interesting failure and a single number would hide it.
 */
export function createVoyage(events, { task = TASK } = {}) {
  const claimed = new Set();
  let recastUntilMs = -1;
  let ret = 0;

  const perStage = {};
  for (const stage of STAGES) {
    if (!stage.decision) continue;
    perStage[stage.id] = { chances: 0, correct: 0, wrong: 0, missed: 0, acts: 0 };
  }
  for (const event of events) {
    if (perStage[event.stage] && event.rewarding) perStage[event.stage].chances++;
  }

  /** The unclaimed event whose window covers `tMs`, rewarding or not. */
  const liveEventAt = (tMs) => {
    for (let i = 0; i < events.length; i++) {
      if (claimed.has(i)) continue;
      const event = events[i];
      if (tMs >= event.atMs && tMs < event.atMs + task.hookWindowMs) return i;
    }
    return -1;
  };

  return {
    perStage,
    get totalReward() {
      return ret;
    },
    canDecide(tMs) {
      return tMs >= recastUntilMs;
    },
    step(tMs, action) {
      const stage = stageAt(tMs);
      if (!stage.decision) {
        return { decision: false, action: 0, reward: 0, outcome: "transit", stage: stage.id };
      }
      if (tMs < recastUntilMs) {
        return { decision: false, action: 0, reward: 0, outcome: "recast", stage: stage.id };
      }
      if (!action) {
        return { decision: true, action: 0, reward: 0, outcome: "wait", stage: stage.id };
      }

      recastUntilMs = tMs + task.windowMs + task.recastMs;
      const index = liveEventAt(tMs);
      const tally = perStage[stage.id];
      tally.acts++;

      if (index >= 0 && events[index].rewarding) {
        claimed.add(index);
        tally.correct++;
        ret += task.rewardCatch;
        return {
          decision: true,
          action: 1,
          reward: task.rewardCatch,
          outcome: "good",
          stage: stage.id,
        };
      }
      if (index >= 0) claimed.add(index);
      tally.wrong++;
      ret += task.rewardSnap;
      return {
        decision: true,
        action: 1,
        reward: task.rewardSnap,
        outcome: "bad",
        stage: stage.id,
        // Whether the mistake was acting on the wrong thing or on nothing at all.
        onEvent: index >= 0 ? events[index].stimulus : null,
      };
    },
    finish() {
      for (const stage of STAGES) {
        if (!stage.decision) continue;
        const tally = perStage[stage.id];
        tally.missed = tally.chances - tally.correct;
        tally.successRate = tally.chances ? tally.correct / tally.chances : null;
        tally.precision = tally.acts ? tally.correct / tally.acts : null;
      }
      const chances = Object.values(perStage).reduce((total, s) => total + s.chances, 0);
      const correct = Object.values(perStage).reduce((total, s) => total + s.correct, 0);
      const wrong = Object.values(perStage).reduce((total, s) => total + s.wrong, 0);
      return {
        perStage,
        chances,
        correct,
        wrong,
        totalReward: Number(ret.toFixed(4)),
        successRate: chances ? correct / chances : 0,
        fishCaught: perStage.fish.correct,
      };
    },
  };
}
