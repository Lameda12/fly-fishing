// The fishing task: the bobber, the fish, the decision windows and the scoring.
//
// Every line of this file is scripted. It decides when a fish bites, turns that
// bite into a pulse on one of the simulator's existing stimulus channels, and
// scores what the readout does about it. The fly's response to the pulse is the
// only simulated part, and it does not live here.
//
// This module imports nothing from Node so the tests, the headless runner and
// the recorder all share one copy of the rules.

import { createRng } from "./rng.mjs";

/**
 * The task constants. All scripted; none of them is a measurement.
 *
 * The bite pulse rides on `loomHz`, the simulator's looming-visual-proxy
 * channel. That channel was chosen against a measured dose-response rather than
 * by taste: the raw escape_giant_fiber rate is monotone in looming input across
 * the whole range (0 Hz in -> 0 Hz out, 35 -> 135, 150 -> 199, 220 -> 218), so
 * pulse amplitude survives into the readout. See the README.
 */
export const TASK = Object.freeze({
  episodeMs: 60_000,
  /** One decision window. Ten of them fit inside the 500 ms hook window. */
  windowMs: 50,
  /** Background walking drive, so the baseline is a fly that is doing something. */
  backgroundHungerHz: 40,

  /** A real bite: a strong pulse on the looming channel. */
  biteLoomHz: 150,
  biteDurationMs: 250,

  /** Hooking this long after a bite onset still lands the fish. */
  hookWindowMs: 500,

  /** Re-casting the line after any hook. No decision is taken while it runs. */
  recastMs: 1_500,

  /** Mean gap between bites, before the minimum-spacing clamp. */
  meanBiteGapMs: 5_000,
  /**
   * Minimum gap between bites. Two things depend on it: the response to one
   * bite is over before the next arrives (which is what lets an episode be
   * spliced from cached responses), and a fixed-interval policy cannot lock
   * onto the rhythm because there isn't one.
   */
  minBiteGapMs: 3_000,
  /** No bite in the first second, so every episode opens on a clean baseline. */
  leadInMs: 1_000,

  rewardCatch: 1,
  rewardSnap: -0.5,
  rewardMiss: 0,
});

export const WINDOWS_PER_EPISODE = Math.round(TASK.episodeMs / TASK.windowMs);

/**
 * Draw one episode's fish events.
 *
 * Gaps are exponential with a minimum, so neither the spacing nor the count is
 * fixed across episodes. A policy that hooks on a schedule cannot win, because
 * there is no schedule to learn.
 */
export function buildSchedule(seed, { episodeMs = TASK.episodeMs, task = TASK } = {}) {
  const rng = createRng(seed);
  const events = [];
  let at = task.leadInMs + rng.range(0, task.minBiteGapMs);
  while (at < episodeMs - task.hookWindowMs) {
    events.push({ atMs: Math.round(at), type: "bite" });
    at += task.minBiteGapMs + rng.exponential(task.meanBiteGapMs - task.minBiteGapMs);
  }
  return events;
}

/** Looming drive in Hz at `tMs`, from the episode's events. Scripted. */
export function loomAt(events, tMs, task = TASK) {
  let hz = 0;
  for (const event of events) {
    if (tMs >= event.atMs && tMs < event.atMs + task.biteDurationMs) {
      hz = Math.max(hz, task.biteLoomHz);
    }
  }
  return hz;
}

/** The full stimulus for one window, in upstream's channel names. */
export function stimulusAt(events, tMs, task = TASK) {
  return {
    loomHz: loomAt(events, tMs, task),
    hungerHz: task.backgroundHungerHz,
  };
}

/**
 * The scorer. One instance runs one episode: feed it a window index and the
 * action the policy chose, and it returns the reward and what happened.
 *
 * It is a plain state machine over the schedule so the trained policy, every
 * baseline and the replay recorder are all scored by exactly the same code.
 */
export function createEpisode(events, { task = TASK, episodeMs = TASK.episodeMs } = {}) {
  const claimed = new Set();
  let recastUntilMs = -1;
  const tally = { bites: events.length, caught: 0, snapped: 0, missed: 0, hooks: 0, decisions: 0 };
  let ret = 0;

  /** The unclaimed bite whose hook window covers `tMs`, if any. */
  const liveBiteAt = (tMs) => {
    for (let i = 0; i < events.length; i++) {
      const event = events[i];
      if (claimed.has(i)) continue;
      if (tMs >= event.atMs && tMs < event.atMs + task.hookWindowMs) return i;
    }
    return -1;
  };

  return {
    tally,
    get totalReward() {
      return ret;
    },
    /** True when the line is in the water and the policy may act. */
    canDecide(tMs) {
      return tMs >= recastUntilMs;
    },
    /**
     * Apply one window. `action` is 1 for hook, 0 for wait. During a re-cast the
     * action is ignored and the window is not a decision, so it contributes no
     * gradient and no reward.
     */
    step(tMs, action) {
      if (tMs < recastUntilMs) {
        return { decision: false, action: 0, reward: 0, outcome: "recast" };
      }
      tally.decisions++;
      if (!action) return { decision: true, action: 0, reward: 0, outcome: "wait" };

      tally.hooks++;
      recastUntilMs = tMs + task.windowMs + task.recastMs;
      const index = liveBiteAt(tMs);
      if (index >= 0) {
        claimed.add(index);
        tally.caught++;
        ret += task.rewardCatch;
        return { decision: true, action: 1, reward: task.rewardCatch, outcome: "catch" };
      }
      tally.snapped++;
      ret += task.rewardSnap;
      return { decision: true, action: 1, reward: task.rewardSnap, outcome: "snap" };
    },
    /** Call once the last window is done: bites nobody hooked are misses. */
    finish() {
      tally.missed = tally.bites - tally.caught;
      return {
        ...tally,
        totalReward: Number(ret.toFixed(4)),
        catchRate: tally.bites ? tally.caught / tally.bites : 0,
        // Snapped lines per minute of episode: a rate the baselines can be
        // compared on even when they hook wildly different numbers of times.
        falseHooksPerMinute: (tally.snapped * 60_000) / episodeMs,
        hookPrecision: tally.hooks ? tally.caught / tally.hooks : 0,
      };
    },
  };
}
