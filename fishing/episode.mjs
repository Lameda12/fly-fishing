// One episode: march the decision windows, ask the policy, score the result.
//
// The same function runs a spliced-cache episode and a live-simulator episode.
// The only difference is where `featureAt(index)` gets its row from, which is
// the whole point of the cache being real network output rather than a model of
// it: nothing below this line knows or cares which it is.

import { TASK, createEpisode } from "./task.mjs";
import { buildFeatures } from "./readout.mjs";

const ZERO_RATES = Object.freeze([0, 0, 0, 0, 0, 0, 0, 0]);

/**
 * @param featureAt  (index) => the eight raw readout rates for that window
 * @param policy     from createPolicy()
 * @param events     the episode's fish events
 * @param collect    true to keep a per-window trace for the replay recorder
 */
export function runEpisode({
  featureAt,
  policy,
  events,
  task = TASK,
  episodeMs = TASK.episodeMs,
  collect = false,
}) {
  const windows = Math.round(episodeMs / task.windowMs);
  const scorer = createEpisode(events, { task, episodeMs });
  const decisions = [];
  const trace = collect ? [] : null;

  let previousRates = ZERO_RATES;
  for (let index = 0; index < windows; index++) {
    const tMs = index * task.windowMs;
    const rates = featureAt(index);
    const features = buildFeatures(rates, previousRates);
    previousRates = rates;

    const open = scorer.canDecide(tMs);
    const action = open ? policy.act(tMs, features) : 0;
    const result = scorer.step(tMs, action);

    if (result.decision) {
      decisions.push({
        features,
        action: result.action,
        // A deterministic baseline has no probability of its own; the gradient
        // only ever reads this for the readout policy, which does.
        probability: policy.probability ? policy.probability(features) : result.action,
        reward: result.reward,
      });
    }
    if (collect) {
      trace.push({
        tMs,
        rates,
        probability: policy.probability ? policy.probability(features) : null,
        action: result.action,
        outcome: result.outcome,
      });
    }
  }

  return { summary: scorer.finish(), decisions, trace };
}

/** Feature source backed by a spliced cache episode. */
export function fromRows(rows) {
  return (index) => rows[index] ?? ZERO_RATES;
}
