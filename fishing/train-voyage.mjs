#!/usr/bin/env node
// Train the voyage readout across all four decision stages.
//
// Frozen: the connectome. Trained: a linear readout from the eight descending
// rates to one act-or-wait decision, in one of two shapes.
//
//   --heads shared      one 17-weight readout for all four stages. It sees the
//                       eight rates and nothing else, and has to work out from
//                       those alone whether it is holding a bait, watching a
//                       bobber, standing over a pan, or chewing.
//   --heads per-stage   four 17-weight readouts, one per stage, each trained
//                       only on its own stage's decisions. The stage index is
//                       read off the voyage clock, so this arm is *told* which
//                       stage it is in; the shared arm is not.
//
// Both are worth running and the README reports both, because the shared arm
// fails in a specific and legible way: it learns baiting and fishing, drives
// P(act) in the other two stages to 1e-4, and then cannot climb back out.
//
// Per-stage scores are the point of the output. A single number would hide the
// interesting failure, which is learning one stage and not another.

import path from "node:path";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { parseArgs } from "node:util";

import { REPO_ROOT } from "./brain-host.mjs";
import { learningCurveSvg } from "./plot.mjs";
import {
  FEATURE_COUNT,
  createPolicy,
  createReadout,
  createStagedReadout,
  createTrainer,
} from "./readout.mjs";
import { deriveSeed } from "./rng.mjs";
import { TASK } from "./task.mjs";
import { STAGES } from "./voyage.mjs";
import { runVoyage, voyageOraclePolicy, voyageReflexPolicy } from "./voyage-episode.mjs";

const DECISION_STAGES = STAGES.filter((stage) => stage.decision);

const OPTIONS = {
  seed: { type: "string", default: "1592594996" },
  episodes: { type: "string", default: "4000" },
  "eval-every": { type: "string", default: "50" },
  "eval-episodes": { type: "string", default: "24" },
  "learning-rate": { type: "string", default: "0.02" },
  gamma: { type: "string", default: "0.9" },
  epsilon: { type: "string", default: "0" },
  heads: { type: "string", default: "per-stage" },
  seeds: { type: "string", default: "5" },
  cache: { type: "string" },
  out: { type: "string" },
  quiet: { type: "boolean", default: false },
  help: { type: "boolean", default: false },
};

const USAGE = `Usage: node fishing/train-voyage.mjs [options]

  --seed N             run seed (default 1592594996)
  --episodes N         training voyages per seed (default 4000; eating needs
                       about 2200 of them before its weight beats the bias)
  --eval-every N       evaluate every N voyages (default 50)
  --eval-episodes N    held-out voyages per evaluation (default 24)
  --learning-rate F    Adam step size (default 0.02)
  --gamma F            reward-to-go discount (default 0.9)
  --epsilon F          exploration floor during training (default 0; see the
                       comment on the readout policy for why raising it does
                       not rescue the shared head, and does break it)
  --heads MODE         per-stage (default) or shared
  --seeds N            training seeds (default 5); training is bimodal, so one
                       seed reports a coin flip rather than a result
  --cache PATH         default results/dn-cache.json
  --out DIR            default results/
  --quiet
  --help
`;

const pct = (value) => (value === null ? "  n/a" : `${(value * 100).toFixed(1)}%`);

export function voyageEvalSeeds(runSeed, count) {
  return Array.from({ length: count }, (_, i) => deriveSeed(runSeed, "voyage-eval", i));
}

/** Average a policy's per-stage outcome over a fixed set of voyages. */
export function evaluateVoyage({ cache, seeds, makePolicy, task = TASK }) {
  const totals = { totalReward: 0, successRate: 0, fishCaught: 0 };
  const stages = Object.fromEntries(
    DECISION_STAGES.map((stage) => [stage.id, { successRate: 0, wrong: 0, chances: 0, seen: 0 }]),
  );

  for (const seed of seeds) {
    const { summary } = runVoyage({ cache, seed, policy: makePolicy({ seed }), task });
    totals.totalReward += summary.totalReward / seeds.length;
    totals.successRate += summary.successRate / seeds.length;
    totals.fishCaught += summary.fishCaught / seeds.length;
    for (const stage of DECISION_STAGES) {
      const tally = summary.perStage[stage.id];
      stages[stage.id].wrong += tally.wrong / seeds.length;
      stages[stage.id].chances += tally.chances / seeds.length;
      // A stage with no chances this voyage cannot contribute a rate; averaging
      // over the voyages that had one keeps an empty pan from reading as 0%.
      if (tally.successRate !== null) {
        stages[stage.id].successRate += tally.successRate;
        stages[stage.id].seen++;
      }
    }
  }
  for (const stage of DECISION_STAGES) {
    const entry = stages[stage.id];
    entry.successRate = entry.seen ? entry.successRate / entry.seen : null;
  }
  return { ...totals, stages, episodes: seeds.length };
}

/**
 * Build the trainable part and the two policies that read it.
 *
 * `shared` is one readout for the whole voyage, updated from every decision
 * with per-stage advantage normalisation so the busiest two stages cannot own
 * the gradient. `per-stage` is one readout per stage; each is updated only
 * from its own stage's decisions, by the same ungrouped reward-to-go path that
 * trained the single-stage fishing task.
 */
function buildArm({ heads, stageIds, learningRate, gamma, task }) {
  if (heads === "shared") {
    const readout = createReadout();
    const trainer = createTrainer({ readout, learningRate, gamma, groupAdvantages: true });
    return {
      checkpoint: readout,
      behaviour: (epsilon, seed) => createPolicy("readout", { readout, seed, epsilon }),
      greedy: () => createPolicy("readoutGreedy", { task, readout }),
      update: (decisions) => trainer.update(decisions.map((d) => ({ ...d, group: d.stage }))),
      // One head cannot hold a stage out: every window it saw belongs to it.
      holdsOutDormantStages: false,
    };
  }
  if (heads !== "per-stage") throw new Error(`unknown --heads mode "${heads}"`);

  const staged = createStagedReadout({ stageIds });
  const trainers = Object.fromEntries(
    stageIds.map((id) => [id, createTrainer({ readout: staged.head(id), learningRate, gamma })]),
  );
  return {
    checkpoint: staged,
    behaviour: (_epsilon, seed) => createPolicy("stagedReadout", { staged, seed }),
    greedy: () => createPolicy("stagedReadoutGreedy", { task, staged }),
    /**
     * Update each head from its own stage, skipping the stages that had no
     * chance this voyage.
     *
     * **That skip is the difference between this arm working and not.** The
     * voyage is a chain: there is nothing to cook unless a fish was caught, so
     * `runVoyage` schedules zero cooking and eating events until the fishing
     * head starts landing them, which takes about 200 voyages. Without the
     * skip, those 200 voyages are not neutral for the cooking head. It still
     * sees its 200 windows per voyage, acting in any of them is still a
     * mistake, and it dutifully learns the only lesson available: never act.
     * By the time the first pan appears the head is at a bias near -6 and
     * P(act) near 0.001, and it never recovers. Measured: cooking and eating
     * sit at 0.0% for 1500 voyages with the skip removed, on every seed and
     * every step size tried.
     *
     * A voyage in which a stage never happened is not a hard sample of that
     * stage, it is an absence of the stage, so the head is held out of it.
     * This is a curriculum fix to a non-stationary task, and it is the reason
     * the shared-readout arm cannot be repaired the same way: with one head
     * there is nothing to hold out.
     *
     * Dormant means no events, not no rewarding events. A voyage where every
     * bite turned out to be a decoy did happen, and the decoys it refused are
     * exactly the samples that teach the difference; holding those out was
     * tried and is strictly worse.
     */
    update: (decisions, events) => {
      const present = new Set(events.map((event) => event.stage));
      for (const id of stageIds) {
        if (!present.has(id)) continue;
        const slice = decisions.filter((d) => d.stage === id);
        if (slice.length) trainers[id].update(slice);
      }
    },
    holdsOutDormantStages: true,
  };
}

export function trainVoyageReadout({
  cache,
  runSeed,
  episodes,
  heldOut,
  evalEvery = 50,
  learningRate = 0.02,
  gamma = 0.9,
  heads = "per-stage",
  epsilon = 0,
  task = TASK,
  onEval,
}) {
  const arm = buildArm({
    heads,
    stageIds: DECISION_STAGES.map((stage) => stage.id),
    learningRate,
    gamma,
    task,
  });
  const points = [];

  const measure = () => evaluateVoyage({ cache, seeds: heldOut, makePolicy: arm.greedy, task });

  for (let episode = 1; episode <= episodes; episode++) {
    const seed = deriveSeed(runSeed, "voyage-train", episode);
    const { decisions, events } = runVoyage({
      cache,
      seed,
      policy: arm.behaviour(epsilon, deriveSeed(seed, "act")),
      task,
    });
    arm.update(decisions, events);

    if (episode % evalEvery === 0 || episode === 1) {
      const result = measure();
      points.push({
        episode,
        catchRate: Number(result.successRate.toFixed(4)),
        meanReward: Number(result.totalReward.toFixed(3)),
        ...Object.fromEntries(
          DECISION_STAGES.map((stage) => [
            `${stage.id}Rate`,
            result.stages[stage.id].successRate === null
              ? null
              : Number(result.stages[stage.id].successRate.toFixed(4)),
          ]),
        ),
      });
      onEval?.(episode, result);
    }
  }
  return { readout: arm.checkpoint, greedy: arm.greedy, points, final: measure() };
}

async function main() {
  let values;
  try {
    ({ values } = parseArgs({ options: OPTIONS, strict: true }));
  } catch (error) {
    process.stderr.write(`${error.message}\n\n${USAGE}`);
    return 2;
  }
  if (values.help) {
    process.stdout.write(USAGE);
    return 0;
  }

  const say = values.quiet ? () => {} : (text) => console.log(text);
  const runSeed = Number.parseInt(values.seed, 10) >>> 0;
  const episodes = Number.parseInt(values.episodes, 10);
  const evalEvery = Number.parseInt(values["eval-every"], 10);
  const evalCount = Number.parseInt(values["eval-episodes"], 10);
  const seedCount = Number.parseInt(values.seeds, 10);
  const cachePath = values.cache
    ? path.resolve(values.cache)
    : path.join(REPO_ROOT, "results", "dn-cache.json");
  const outDir = values.out ? path.resolve(values.out) : path.join(REPO_ROOT, "results");

  let cache;
  try {
    cache = JSON.parse(await readFile(cachePath, "utf8"));
  } catch {
    process.stderr.write(`No response cache at ${cachePath}\nRun: python3 run.py cache\n`);
    return 1;
  }
  const missing = [...new Set(DECISION_STAGES.flatMap((s) => [s.good, s.bad]).filter(Boolean))]
    .filter((stimulus) => !cache[stimulus]);
  if (missing.length) {
    process.stderr.write(
      `The cache is missing the voyage stimuli: ${missing.join(", ")}\n` +
        `Record them with:\n  node fishing/build-cache.mjs --sections ${missing.join(",")} ` +
        `--merge-into ${path.relative(process.cwd(), cachePath)}\n`,
    );
    return 1;
  }

  const trainSeeds = Array.from({ length: seedCount }, (_, i) =>
    i === 0 ? runSeed : deriveSeed(runSeed, "voyage-arm", i),
  );
  const scoreSeeds = voyageEvalSeeds(runSeed, 1000);

  // The two baselines need the growing event list, so they are passed as
  // factories; the readout does not, because it only ever looks at the fly.
  const oracle = evaluateVoyage({
    cache,
    seeds: scoreSeeds,
    makePolicy: () => ({ events }) => voyageOraclePolicy({ scorerEvents: events }),
  });
  const reflex = evaluateVoyage({
    cache,
    seeds: scoreSeeds,
    makePolicy: () => ({ events }) => voyageReflexPolicy({ scorerEvents: events }),
  });
  const random = evaluateVoyage({
    cache,
    seeds: scoreSeeds,
    makePolicy: ({ seed }) => createPolicy("random", { task: TASK, events: [], seed, hookProbability: 0.012 }),
  });

  say("");
  say(
    values.heads === "shared"
      ? `  training one shared readout across ${DECISION_STAGES.length} stages, seed ${runSeed}`
      : `  training ${DECISION_STAGES.length} per-stage readout heads, seed ${runSeed}`,
  );
  say(`  cache: ${path.relative(process.cwd(), cachePath)} (ablation ${cache.ablation})`);
  say(
    `  oracle ${pct(oracle.successRate)} overall, reflex ${pct(reflex.successRate)},` +
      ` random ${pct(random.successRate)}`,
  );
  say("");

  const runs = [];
  let best = null;
  let bestReward = -Infinity;
  for (const trainSeed of trainSeeds) {
    const heldOut = voyageEvalSeeds(trainSeed, evalCount);
    const { readout, greedy, points } = trainVoyageReadout({
      cache,
      runSeed: trainSeed,
      episodes,
      heldOut,
      evalEvery,
      learningRate: Number.parseFloat(values["learning-rate"]),
      gamma: Number.parseFloat(values.gamma),
      epsilon: Number.parseFloat(values.epsilon),
      heads: values.heads,
      onEval:
        trainSeeds.length === 1 || trainSeed === runSeed
          ? (episode, result) =>
              say(
                `  episode ${String(episode).padStart(5)}  overall ${pct(result.successRate).padStart(6)}  ` +
                  DECISION_STAGES.map(
                    (stage) => `${stage.id} ${pct(result.stages[stage.id].successRate).padStart(6)}`,
                  ).join("  ") +
                  `  reward ${result.totalReward.toFixed(2).padStart(6)}`,
              )
          : undefined,
    });
    const scored = evaluateVoyage({ cache, seeds: scoreSeeds, makePolicy: greedy });
    const converged = scored.successRate > 0.5;
    runs.push({
      trainSeed,
      converged,
      successRate: Number(scored.successRate.toFixed(4)),
      meanReward: Number(scored.totalReward.toFixed(3)),
      fishCaught: Number(scored.fishCaught.toFixed(2)),
      stages: Object.fromEntries(
        DECISION_STAGES.map((stage) => [
          stage.id,
          scored.stages[stage.id].successRate === null
            ? null
            : Number(scored.stages[stage.id].successRate.toFixed(4)),
        ]),
      ),
      points,
    });
    if (scored.totalReward > bestReward) {
      bestReward = scored.totalReward;
      best = { readout, points, scored };
    }
    say(
      `  seed ${String(trainSeed).padStart(10)}  overall ${pct(scored.successRate).padStart(6)}` +
        `  reward ${scored.totalReward.toFixed(2).padStart(6)}${converged ? "" : "   (collapsed)"}`,
    );
  }

  const survivors = runs.filter((run) => run.converged);
  await mkdir(outDir, { recursive: true });

  const report = {
    schemaVersion: 1,
    generatedAt: new Date().toISOString(),
    runSeed,
    episodes,
    heads: values.heads,
    trainSeeds,
    evalEpisodes: scoreSeeds.length,
    convergedCount: survivors.length,
    runs: runs.map(({ points, ...rest }) => rest),
    baselines: {
      oracle: {
        successRate: Number(oracle.successRate.toFixed(4)),
        meanReward: Number(oracle.totalReward.toFixed(3)),
        stages: Object.fromEntries(
          DECISION_STAGES.map((s) => [s.id, Number((oracle.stages[s.id].successRate ?? 0).toFixed(4))]),
        ),
      },
      reflex: {
        successRate: Number(reflex.successRate.toFixed(4)),
        meanReward: Number(reflex.totalReward.toFixed(3)),
        stages: Object.fromEntries(
          DECISION_STAGES.map((s) => [s.id, Number((reflex.stages[s.id].successRate ?? 0).toFixed(4))]),
        ),
      },
      random: {
        successRate: Number(random.successRate.toFixed(4)),
        meanReward: Number(random.totalReward.toFixed(3)),
      },
    },
    readout: best ? best.readout.toJSON() : null,
    frozen:
      "The connectome. Nothing inside the simulator is trained, and the fly does " +
      "not learn to fish, cook or eat.",
    trained: `${FEATURE_COUNT} linear readout weights, shared across all four decision stages.`,
  };
  // The shared arm is the comparison, not the deliverable, so it writes beside
  // the per-stage result rather than over it.
  const suffix = values.heads === "shared" ? "-shared" : "";
  const reportPath = path.join(outDir, `voyage${suffix}.json`);
  await writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");

  if (best) {
    const svgPath = path.join(outDir, `voyage-curve${suffix}.svg`);
    await writeFile(
      svgPath,
      learningCurveSvg({
        title: `Voyage training, seed ${runSeed}`,
        subtitle:
          "The connectome is frozen. One 17-weight readout serves all four decision stages.",
        points: best.points,
        reference: { oracleCatchRate: oracle.successRate, randomCatchRate: random.successRate },
      }),
      "utf8",
    );
    say("");
    say(`  wrote ${path.relative(process.cwd(), reportPath)}`);
    say(`        ${path.relative(process.cwd(), svgPath)}`);
  }

  say("");
  say(`  converged ${survivors.length}/${trainSeeds.length}`);
  if (best) {
    say("  best run, per stage:");
    for (const stage of DECISION_STAGES) {
      say(
        `    ${stage.name.padEnd(18)} ${pct(best.scored.stages[stage.id].successRate).padStart(6)}` +
          `   (${stage.act})`,
      );
    }
  }
  say("");
  return 0;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  process.exitCode = await main();
}
