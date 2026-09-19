#!/usr/bin/env node
// Train one shared readout across all four decision stages of the voyage.
//
// Frozen: the connectome. Trained: the same 17 weights as the single-stage
// task, except that now they have to mean four different things depending on
// which sensory pattern is present. There is no stage input and no per-stage
// head: the readout sees eight descending rates and nothing else, and has to
// work out from those alone whether it is holding a bait, watching a bobber,
// standing over a pan, or chewing.
//
// Per-stage scores are the point of the output. A single number would hide the
// interesting failure, which is learning one stage and not another.

import path from "node:path";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { parseArgs } from "node:util";

import { REPO_ROOT } from "./brain-host.mjs";
import { learningCurveSvg } from "./plot.mjs";
import { FEATURE_COUNT, createPolicy, createReadout, createTrainer } from "./readout.mjs";
import { deriveSeed } from "./rng.mjs";
import { TASK } from "./task.mjs";
import { STAGES } from "./voyage.mjs";
import { runVoyage, voyageOraclePolicy, voyageReflexPolicy } from "./voyage-episode.mjs";

const DECISION_STAGES = STAGES.filter((stage) => stage.decision);

const OPTIONS = {
  seed: { type: "string", default: "1592594996" },
  episodes: { type: "string", default: "1500" },
  "eval-every": { type: "string", default: "50" },
  "eval-episodes": { type: "string", default: "24" },
  "learning-rate": { type: "string", default: "0.05" },
  gamma: { type: "string", default: "0.9" },
  seeds: { type: "string", default: "5" },
  cache: { type: "string" },
  out: { type: "string" },
  quiet: { type: "boolean", default: false },
  help: { type: "boolean", default: false },
};

const USAGE = `Usage: node fishing/train-voyage.mjs [options]

  --seed N             run seed (default 1592594996)
  --episodes N         training voyages per seed (default 1500)
  --eval-every N       evaluate every N voyages (default 50)
  --eval-episodes N    held-out voyages per evaluation (default 24)
  --learning-rate F    Adam step size (default 0.05)
  --gamma F            reward-to-go discount (default 0.9)
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

export function trainVoyageReadout({
  cache,
  runSeed,
  episodes,
  heldOut,
  evalEvery = 50,
  learningRate = 0.05,
  gamma = 0.9,
  task = TASK,
  onEval,
}) {
  const readout = createReadout();
  const trainer = createTrainer({ readout, learningRate, gamma });
  const points = [];

  const measure = () =>
    evaluateVoyage({
      cache,
      seeds: heldOut,
      makePolicy: () => createPolicy("readoutGreedy", { task, readout }),
      task,
    });

  for (let episode = 1; episode <= episodes; episode++) {
    const seed = deriveSeed(runSeed, "voyage-train", episode);
    const { decisions } = runVoyage({
      cache,
      seed,
      policy: createPolicy("readout", { readout, seed: deriveSeed(seed, "act") }),
      task,
    });
    trainer.update(decisions);

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
  return { readout, points, final: measure() };
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
  say(`  training one shared readout across ${DECISION_STAGES.length} stages, seed ${runSeed}`);
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
    const { readout, points } = trainVoyageReadout({
      cache,
      runSeed: trainSeed,
      episodes,
      heldOut,
      evalEvery,
      learningRate: Number.parseFloat(values["learning-rate"]),
      gamma: Number.parseFloat(values.gamma),
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
    const scored = evaluateVoyage({
      cache,
      seeds: scoreSeeds,
      makePolicy: () => createPolicy("readoutGreedy", { task: TASK, readout }),
    });
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
  const reportPath = path.join(outDir, "voyage.json");
  await writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");

  if (best) {
    const svgPath = path.join(outDir, "voyage-curve.svg");
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
