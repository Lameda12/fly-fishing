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

/**
 * Build one feature row, optionally centred on the network's resting rates.
 *
 * `resting` is the mean rate of each readout population with no stimulus
 * present, measured from the cache's baseline traces. Passing it subtracts
 * that rate, so a population sitting at its resting level contributes exactly
 * zero rather than a small constant. Passing null leaves the rates raw, which
 * is what the single-stage fishing task was trained with.
 *
 * **Centring is not cosmetic here, and it is Adam that makes it matter.** Adam
 * normalises each weight's step by that weight's own gradient magnitude, so
 * what reaches a weight is very close to the *sign* of its gradient and not
 * the size. That is the property that made the single-stage task trainable at
 * all (see `createTrainer`), and it has a sharp edge. A population with a
 * nonzero resting rate is slightly positive in every empty window, so every
 * act that lands in an empty window — nine out of ten of them, in a task where
 * about 2% of windows contain anything — pushes its weight down by a tiny
 * amount with a consistent sign, and Adam scales that tiny amount up to a full
 * step. The rare correct act pushes back with a much larger gradient and gets
 * scaled down to the same full step. The constant loses to nothing and wins on
 * volume.
 *
 * Measured, under the 40 Hz background drive: `reverse_mdn`, `groom_adn1`,
 * `escape_giant_fiber` and `feed_mn9` rest at exactly 0.00 Hz; `walk_dnp09`
 * rests at 40.11, `forward_odn1` at 2.16, `turn_right` at 2.10 and
 * `turn_left` at 0.24. Uncentred, the only stages of the voyage that ever
 * trained were the two whose discriminating population is one of the four
 * exact zeros: baiting on `groom_adn1` and fishing on `escape_giant_fiber`.
 * Cooking, whose only tell is `turn_left` at 34.7 Hz against a resting 0.24,
 * drove that weight to -2.3 when the sign it needed was positive.
 */
export function buildFeatures(rates, previousRates, resting = null) {
  const row = new Array(FEATURE_COUNT);
  for (let i = 0; i < READOUT_FEATURES.length; i++) {
    const offset = resting ? resting[i] : 0;
    row[i] = (rates[i] - offset) / RATE_SCALE;
    row[READOUT_FEATURES.length + i] = (previousRates[i] - offset) / RATE_SCALE;
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
    /** The pre-sigmoid score for one feature row. */
    logit(features) {
      let sum = 0;
      for (let i = 0; i < FEATURE_COUNT; i++) sum += w[i] * features[i];
      return sum;
    },
    /** P(hook) for one feature row. */
    probability(features) {
      return sigmoid(this.logit(features));
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
 * One readout per stage of the voyage, all reading the same eight rates.
 *
 * The shared-readout version of the voyage learns baiting and fishing and
 * never learns cooking or eating; `fishing/train-voyage.mjs` reports both
 * arms and the README gives the measurement. The short version is that a
 * single weight vector that solves two stages drives P(act) in the other two
 * to 1e-4, and a policy gradient cannot climb out of that: the act is never
 * sampled, so the gradient that would raise it is never formed. Every way of
 * forcing the act back in costs more than it buys, because a voyage has about
 * 1100 decision windows and roughly 30 of them contain anything.
 *
 * Splitting the head per stage removes the coupling instead of fighting it.
 * Each head sees only its own stage's decisions, starts at the same wait-by
 * -default bias, and is trained by exactly the code path that reached 100% on
 * the single-stage fishing task.
 *
 * **What this costs in honesty: the stage index is scripted.** It comes from
 * the voyage's clock, not from the fly; nothing here decodes which stage it is
 * in from the descending rates. The heads are told. The shared-readout arm is
 * the one that is not told, and it is reported next to this one for exactly
 * that reason.
 */
export function createStagedReadout({ stageIds, weights = null, initialBias = -3 } = {}) {
  const heads = Object.fromEntries(
    stageIds.map((id) => [id, createReadout({ weights: weights?.[id] ?? null, initialBias })]),
  );
  return {
    stageIds: [...stageIds],
    heads,
    head(stage) {
      const head = heads[stage];
      if (!head) throw new Error(`no readout head for stage "${stage}"`);
      return head;
    },
    logit(features, stage) {
      return this.head(stage).logit(features);
    },
    probability(features, stage) {
      return this.head(stage).probability(features);
    },
    toJSON() {
      return {
        featureNames: [...FEATURE_NAMES],
        rateScale: RATE_SCALE,
        perStage: true,
        weights: Object.fromEntries(stageIds.map((id) => [id, [...heads[id].weights]])),
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
 *
 * **The step is Adam, and that is load-bearing rather than fashionable.** The
 * features here are wildly unequal in how often they are on: the bias is 1 in
 * every window, while `escape_giant_fiber` is nonzero in roughly a tenth of
 * them, around a fish event. Under a plain SGD step the bias therefore collects
 * about ten times the gradient the informative feature does, drives itself to
 * about -7 within a couple of hundred episodes, and takes P(hook) to ~0.001
 * everywhere. Exploration stops and nothing is ever learned again: measured on
 * this cache, 600 episodes of plain SGD ended at a 0% catch rate with the right
 * signs on every weight and no magnitude on any of them. Adam gives each weight
 * a step scaled by its own gradient history, so a feature that is only
 * occasionally active still moves.
 */
export function createTrainer({
  readout,
  learningRate = 0.05,
  gamma = 0.9,
  baselineDecay = 0.9,
  beta1 = 0.9,
  beta2 = 0.999,
  epsilon = 1e-8,
  /**
   * Standardize the advantage within each `decision.group` instead of across
   * the whole episode.
   *
   * The voyage needs this. Its stages are not equally represented: baiting and
   * fishing offer more chances than cooking and eating, so pooled advantages
   * let the two busy stages own the gradient. Trained without it, the readout
   * reaches 100% on baiting and fishing and exactly 0% on cooking and eating,
   * in every seed; a supervised fit on the same features gets all four, so
   * that gap is the optimizer satisficing rather than the features being
   * insufficient. Grouping also stops reward-to-go leaking across a stage
   * boundary, which it otherwise does for the last second of each stage.
   */
  groupAdvantages = false,
} = {}) {
  let baseline = 0;
  let seen = 0;
  let steps = 0;
  const grad = new Float64Array(FEATURE_COUNT);
  const moment1 = new Float64Array(FEATURE_COUNT);
  const moment2 = new Float64Array(FEATURE_COUNT);

  return {
    get baseline() {
      return baseline;
    },
    /** `decisions` is [{ features, action, probability, reward, group? }, ...]. */
    update(decisions) {
      if (!decisions.length) return { gradNorm: 0, meanAdvantage: 0 };

      const advantages = new Array(decisions.length);
      const scales = new Array(decisions.length);

      if (!groupAdvantages) {
        // The single-stage path, unchanged: reward-to-go over the whole episode
        // against a running-mean baseline, scaled by its own spread.
        const returns = rewardsToGo(decisions.map((d) => d.reward), gamma);
        const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
        seen++;
        // A cold baseline would make the first episode's advantage the whole
        // return, so warm it up on the first episode rather than decaying into it.
        baseline = seen === 1 ? mean : baselineDecay * baseline + (1 - baselineDecay) * mean;
        const centred = returns.map((g) => g - baseline);
        const variance = centred.reduce((total, a) => total + a * a, 0) / centred.length;
        const scale = Math.sqrt(variance) || 1;
        for (let i = 0; i < decisions.length; i++) {
          advantages[i] = centred[i];
          scales[i] = scale;
        }
      } else {
        // One slice per contiguous run of the same group. Each gets its own
        // reward-to-go, its own baseline and its own scale, so a stage with few
        // chances still contributes a gradient of comparable size to a busy one,
        // and no return leaks across a stage boundary.
        let start = 0;
        let weightedMean = 0;
        for (let i = 1; i <= decisions.length; i++) {
          if (i < decisions.length && decisions[i].group === decisions[start].group) continue;
          const returns = rewardsToGo(
            decisions.slice(start, i).map((d) => d.reward),
            gamma,
          );
          const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
          weightedMean += (mean * (i - start)) / decisions.length;
          const centred = returns.map((g) => g - mean);
          const variance = centred.reduce((total, a) => total + a * a, 0) / centred.length;
          const scale = Math.sqrt(variance) || 1;
          for (let k = start; k < i; k++) {
            advantages[k] = centred[k - start];
            scales[k] = scale;
          }
          start = i;
        }
        seen++;
        baseline =
          seen === 1 ? weightedMean : baselineDecay * baseline + (1 - baselineDecay) * weightedMean;
      }

      grad.fill(0);
      for (let i = 0; i < decisions.length; i++) {
        const { features, action, probability } = decisions[i];
        // The Bernoulli score, d log pi / d z. `probability` is whatever the
        // policy actually sampled from, which is why the exploration floor is
        // applied there rather than here: the estimator stays on-policy.
        const logGrad = action - probability;
        const coefficient = logGrad * (advantages[i] / scales[i]);
        for (let f = 0; f < FEATURE_COUNT; f++) grad[f] += coefficient * features[f];
      }
      steps++;
      const correction1 = 1 - beta1 ** steps;
      const correction2 = 1 - beta2 ** steps;
      let gradNorm = 0;
      for (let f = 0; f < FEATURE_COUNT; f++) {
        grad[f] /= decisions.length;
        gradNorm += grad[f] * grad[f];
        moment1[f] = beta1 * moment1[f] + (1 - beta1) * grad[f];
        moment2[f] = beta2 * moment2[f] + (1 - beta2) * grad[f] * grad[f];
        const step =
          (moment1[f] / correction1) / (Math.sqrt(moment2[f] / correction2) + epsilon);
        // Ascent: the gradient above is of the expected return, not of a loss.
        readout.weights[f] += learningRate * step;
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
  const { seed = 1, task, events = [], readout = null, staged = null } = options;
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
      // Matched to the number of catchable fish, which is the oracle's hook
      // count, not to the number of times the bobber moved.
      const bites = events.filter((event) => event.type === "bite").length;
      const p = options.hookProbability ?? (bites || 1) / windows;
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
     * event, so this policy knows *that* something happened and not *what*. It is
     * the hard baseline: it catches every fish and falls for every decoy, which
     * is exactly the score to beat by telling the two apart.
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

    /**
     * The trained readout. The only policy that looks at the fly.
     *
     * `epsilon` is an exploration floor: neither answer is ever sampled with
     * probability below `epsilon / 2`, however certain the weights become.
     * The voyage needs one. Trained without it, the readout solves baiting and
     * fishing using `reverse_mdn` and `forward_odn1`, both of which are exactly
     * zero in the cooking and eating stages; P(act) there falls to 0.0001, no
     * act is ever sampled in 1500 voyages, and those two stages can never be
     * learned. A supervised fit on the same seventeen features gets all four
     * stages right, so the barrier is exploration, not representation.
     *
     * **The floor is enforced on the logit, and that detail is the whole
     * fix.** The obvious construction is a mixture, sampling from
     * `q = (1 - e) * p + e / 2`. Two versions of that were tried and both
     * failed. Scoring against `q` (the unbiased, textbook choice, gradient
     * `(1 - e) * p * (1 - p) / q`) reintroduces the exact factor the
     * exploration was meant to escape: with `p` at 1e-4 the numerator is 1e-4,
     * so however often the coin forces an act, the update it produces is
     * scaled straight back down to the size the saturated policy would have
     * given anyway. 1500 voyages of it left cooking and eating at 0.0%.
     * Scoring against `p` instead keeps the update full size but is biased,
     * and the bias is a constant push toward acting applied in every one of
     * the ~1170 windows per voyage, almost all of which are empty and punish
     * acting. That one was worse than doing nothing: 0.0% on every stage.
     *
     * Clamping the logit to `+/-ln((1 - e/2) / (e/2))` gives the same floor
     * with neither problem. The policy that is sampled from is the policy the
     * score is taken against, so `E[action - p]` is zero at every fixed weight
     * vector and the estimator adds no push in either direction; and the score
     * of a forced act is a full `1 - p`, because `p` at the clamp is
     * `epsilon / 2` rather than 1e-4. The gradient is not passed through the
     * clamp, which makes this a straight-through estimator and therefore not
     * exactly the gradient of `sigmoid(z)` outside the clamp; it has the right
     * sign everywhere, and it is an optimizer choice, not a claim about the
     * fly. Greedy evaluation does not clamp, and does not need to: clamping
     * preserves the sign of the logit, so it never changes a decision.
     */
    case "readout": {
      if (!readout) throw new Error("the readout policy needs a readout");
      const epsilon = options.epsilon ?? 0;
      const limit = epsilon > 0 ? Math.log((1 - epsilon / 2) / (epsilon / 2)) : Infinity;
      const floored = (features) =>
        sigmoid(Math.max(-limit, Math.min(limit, readout.logit(features))));
      return {
        kind,
        epsilon,
        act: (_tMs, features) => (rng.next() < floored(features) ? 1 : 0),
        probability: floored,
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

    /**
     * The per-stage heads, sampling. `stage` arrives from the episode loop,
     * which reads it off the voyage clock; see `createStagedReadout`.
     */
    case "stagedReadout": {
      if (!staged) throw new Error("the stagedReadout policy needs a staged readout");
      return {
        kind,
        act: (_tMs, features, stage) =>
          rng.next() < staged.probability(features, stage) ? 1 : 0,
        probability: (features, stage) => staged.probability(features, stage),
      };
    }

    /** The per-stage heads with sampling off. */
    case "stagedReadoutGreedy": {
      if (!staged) throw new Error("the stagedReadout policy needs a staged readout");
      return {
        kind,
        act: (_tMs, features, stage) => (staged.probability(features, stage) > 0.5 ? 1 : 0),
        probability: (features, stage) => staged.probability(features, stage),
      };
    }

    default:
      throw new Error(`unknown policy "${kind}"`);
  }
}
