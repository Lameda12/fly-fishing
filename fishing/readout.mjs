// The trained part. This is the only thing in the repository that learns.
//
// The connectome is frozen: no weight inside the simulator is ever written to.
// What trains is a linear readout from the eight descending/motor readout rates
// upstream exposes to a single hook-or-wait decision, fitted by REINFORCE.
// Seventeen numbers, listed in the checkpoint file.
//
// No Node imports: the tests and the recorder share this module.

import { READOUT_FEATURES } from "./brain-host.mjs";
import { createRng } from "./rng.mjs";

/** Rates are divided by this before they reach the weights. Scripted. */
export const RATE_SCALE = 100;

/**
 * The feature row: the eight readout rates in this window, the same eight in the
 * previous window, and a bias. The lag is what lets a linear policy see an
 * onset instead of only a level, which matters because a bite is a transient.
 */
export const FEATURE_NAMES = Object.freeze([
  ...READOUT_FEATURES.map((name) => `${name}_t`),
  ...READOUT_FEATURES.map((name) => `${name}_t-1`),
  "bias",
]);

export const FEATURE_COUNT = FEATURE_NAMES.length;

export function buildFeatures(rates, previousRates) {
  const row = new Array(FEATURE_COUNT);
  for (let i = 0; i < READOUT_FEATURES.length; i++) {
    row[i] = rates[i] / RATE_SCALE;
    row[READOUT_FEATURES.length + i] = previousRates[i] / RATE_SCALE;
  }
  row[FEATURE_COUNT - 1] = 1;
  return row;
}

export const sigmoid = (x) => 1 / (1 + Math.exp(-Math.max(-40, Math.min(40, x))));

/**
 * The readout itself.
 *
 * `bias` starts negative so an untrained readout mostly waits. Starting it at
 * zero means a coin flip every 50 ms, which burns the whole episode on re-casts
 * and gives the gradient almost nothing to work with.
 */
export function createReadout({ weights = null, initialBias = -3 } = {}) {
  const w = weights ? Float64Array.from(weights) : new Float64Array(FEATURE_COUNT);
  if (!weights) w[FEATURE_COUNT - 1] = initialBias;
  if (w.length !== FEATURE_COUNT) {
    throw new Error(`readout needs ${FEATURE_COUNT} weights, got ${w.length}`);
  }
  return {
    weights: w,
    /** P(hook) for one feature row. */
    probability(features) {
      let sum = 0;
      for (let i = 0; i < FEATURE_COUNT; i++) sum += w[i] * features[i];
      return sigmoid(sum);
    },
    toJSON() {
      return {
        featureNames: [...FEATURE_NAMES],
        rateScale: RATE_SCALE,
        weights: [...w],
      };
    },
  };
}

/**
 * Discounted reward-to-go over the decision windows of one episode.
 *
 * The rewards here are immediate: hooking pays or costs at once, and waiting
 * never pays later. The discount is not for delayed reward, it is for the
 * re-cast: a hook blocks the next 1.5 s of decisions, and the reward-to-go is
 * what makes the policy feel that cost. gamma 0.9 over 50 ms windows is roughly
 * a one-second horizon, which covers a re-cast.
 */
export function rewardsToGo(rewards, gamma = 0.9) {
  const out = new Array(rewards.length);
  let running = 0;
  for (let i = rewards.length - 1; i >= 0; i--) {
    running = rewards[i] + gamma * running;
    out[i] = running;
  }
  return out;
}

/**
 * One REINFORCE update from one episode's decisions.
 *
 * grad = mean over decisions of (action - p) * features * advantage, which is
 * the score-function estimator for a Bernoulli policy. The advantage is the
 * reward-to-go minus a running mean baseline, then scaled by its own standard
 * deviation so the step size does not depend on how good the episode was in
 * absolute terms.
 */
export function createTrainer({
  readout,
  learningRate = 0.5,
  gamma = 0.9,
  baselineDecay = 0.9,
} = {}) {
  let baseline = 0;
  let seen = 0;
  const grad = new Float64Array(FEATURE_COUNT);

  return {
    get baseline() {
      return baseline;
    },
    /** `decisions` is [{ features, action, probability, reward }, ...]. */
    update(decisions) {
      if (!decisions.length) return { gradNorm: 0, meanAdvantage: 0 };
      const returns = rewardsToGo(decisions.map((d) => d.reward), gamma);

      const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
      seen++;
      // A cold baseline would make the first episode's advantage the whole
      // return, so warm it up on the first episode instead of decaying into it.
      baseline = seen === 1 ? mean : baselineDecay * baseline + (1 - baselineDecay) * mean;

      const advantages = returns.map((g) => g - baseline);
      const variance =
        advantages.reduce((total, a) => total + a * a, 0) / advantages.length;
      const scale = Math.sqrt(variance) || 1;

      grad.fill(0);
      for (let i = 0; i < decisions.length; i++) {
        const { features, action, probability } = decisions[i];
        const coefficient = (action - probability) * (advantages[i] / scale);
        for (let f = 0; f < FEATURE_COUNT; f++) grad[f] += coefficient * features[f];
      }
      let gradNorm = 0;
      for (let f = 0; f < FEATURE_COUNT; f++) {
        grad[f] /= decisions.length;
        gradNorm += grad[f] * grad[f];
        readout.weights[f] += learningRate * grad[f];
      }
      return {
        gradNorm: Math.sqrt(gradNorm),
        meanAdvantage: advantages.reduce((a, b) => a + b, 0) / advantages.length,
      };
    },
  };
}

/**
 * The policies an episode can be run with. Every one of them, the trained
 * readout included, is asked the same question once per decision window and
 * answers 0 or 1, so the scoring code does not know which is which.
 *
 * `dipDelay` and `oracle` are given the event schedule directly. They are
 * baselines, not competitors: the point of handing them ground truth is to say
 * what the ceiling is and what a policy that reacts to the bobber alone gets.
 */
export function createPolicy(kind, options = {}) {
  const { seed = 1, task, events = [], readout = null } = options;
  const rng = createRng(seed);

  switch (kind) {
    /**
     * Uniform random, with its hook probability set so that, before the re-cast
     * takes windows away from it, its expected hook count equals the number of
     * bites. A coin flip would be a straw man; a rate-matched random policy is
     * the strongest version of "hooks without looking", which is what a control
     * should be. (The re-cast then costs it some of those windows, so it ends up
     * hooking somewhat less often than the oracle does.)
     */
    case "random": {
      const windows = Math.round(task.episodeMs / task.windowMs);
      const p = options.hookProbability ?? (events.length || 1) / windows;
      return { kind, hookProbability: p, act: () => (rng.next() < p ? 1 : 0) };
    }

    /** Hooks every `intervalMs`, blind. */
    case "fixedInterval": {
      const intervalMs = options.intervalMs ?? task.meanBiteGapMs;
      let nextMs = intervalMs;
      return {
        kind,
        intervalMs,
        act: (tMs) => {
          if (tMs < nextMs) return 0;
          nextMs += intervalMs;
          return 1;
        },
      };
    }

    /**
     * Hooks a fixed delay after the bobber dips. The bobber dips on every fish
     * event, so this policy knows *that* something happened and not *what*. With
     * one bite type and no decoys it is identical to the oracle by construction;
     * it only becomes a real baseline once decoys dip the bobber too.
     */
    case "dipDelay": {
      const delayMs = options.delayMs ?? 150;
      return {
        kind,
        delayMs,
        act: (tMs) =>
          events.some(
            (event) => tMs >= event.atMs + delayMs && tMs < event.atMs + delayMs + task.windowMs,
          )
            ? 1
            : 0,
      };
    }

    /** Hooks only on real bites. The ceiling. */
    case "oracle": {
      const delayMs = options.delayMs ?? 150;
      return {
        kind,
        act: (tMs) =>
          events.some(
            (event) =>
              event.type === "bite" &&
              tMs >= event.atMs + delayMs &&
              tMs < event.atMs + delayMs + task.windowMs,
          )
            ? 1
            : 0,
      };
    }

    /** The trained readout. The only policy that looks at the fly. */
    case "readout": {
      if (!readout) throw new Error("the readout policy needs a readout");
      return {
        kind,
        act: (_tMs, features) => (rng.next() < readout.probability(features) ? 1 : 0),
        probability: (features) => readout.probability(features),
      };
    }

    /** The trained readout with sampling off: hooks when P(hook) > 0.5. */
    case "readoutGreedy": {
      if (!readout) throw new Error("the readout policy needs a readout");
      return {
        kind,
        act: (_tMs, features) => (readout.probability(features) > 0.5 ? 1 : 0),
        probability: (features) => readout.probability(features),
      };
    }

    default:
      throw new Error(`unknown policy "${kind}"`);
  }
}
