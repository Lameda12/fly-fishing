#!/usr/bin/env node
// Train the readout with REINFORCE against the cached descending responses.
//
// Frozen: the connectome, its weights, the LIF constants, upstream's stimulus
// and readout interfaces. Trained: seventeen numbers in fishing/readout.mjs.
//
// Each training episode draws a fresh bite schedule, splices its feature rows
// out of the cache, samples an action per decision window, and takes one policy
// gradient step. Every `--eval-every` episodes the readout is evaluated greedily
// on a fixed held-out set of schedules, which is what the learning curve plots.

import path from "node:path";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { parseArgs } from "node:util";

import { REPO_ROOT } from "./brain-host.mjs";
import { spliceEpisode } from "./cache.mjs";
import { fromRows, runEpisode } from "./episode.mjs";
import { learningCurveSvg } from "./plot.mjs";
import { FEATURE_COUNT, createPolicy, createReadout, createTrainer } from "./readout.mjs";
import { deriveSeed } from "./rng.mjs";
import { TASK, buildSchedule } from "./task.mjs";

const OPTIONS = {
  seed: { type: "string", default: "1592594996" },
  episodes: { type: "string", default: "600" },
  "eval-every": { type: "string", default: "20" },
  "eval-episodes": { type: "string", default: "24" },
  "learning-rate": { type: "string", default: "0.5" },
  gamma: { type: "string", default: "0.9" },
  cache: { type: "string" },
  out: { type: "string" },
  quiet: { type: "boolean", default: false },
  help: { type: "boolean", default: false },
};

const USAGE = `Usage: node fishing/train.mjs [options]

  --seed N             run seed (default 1592594996); the same seed reproduces the run
  --episodes N         training episodes (default 600)
  --eval-every N       evaluate every N episodes (default 20)
  --eval-episodes N    held-out episodes per evaluation (default 24)
  --learning-rate F    REINFORCE step size (default 0.5)
  --gamma F            reward-to-go discount (default 0.9)
  --cache PATH         default results/dn-cache.json
  --out DIR            default results/
  --quiet
  --help
`;

/** Episodes whose schedules are never trained on. */
export function evalSeeds(runSeed, count) {
  return Array.from({ length: count }, (_, i) => deriveSeed(runSeed, "eval", i));
}

/**
 * Run a policy over a set of cached episodes and average the outcome. Used for
 * the learning curve and for the baselines table.
 */
export function evaluate({ cache, seeds, makePolicy, task = TASK }) {
  const totals = { catchRate: 0, falseHooksPerMinute: 0, totalReward: 0, hookPrecision: 0 };
  const perEpisode = [];
  for (const seed of seeds) {
    const events = buildSchedule(seed, { task });
    const rows = spliceEpisode({ cache, events, seed, task });
    const { summary } = runEpisode({
      featureAt: fromRows(rows),
      policy: makePolicy({ seed, events }),
      events,
      task,
    });
    perEpisode.push(summary);
    for (const key of Object.keys(totals)) totals[key] += summary[key];
  }
  const mean = {};
  for (const key of Object.keys(totals)) mean[key] = totals[key] / seeds.length;
  return { ...mean, episodes: seeds.length, perEpisode };
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
  const cachePath = values.cache
    ? path.resolve(values.cache)
    : path.join(REPO_ROOT, "results", "dn-cache.json");
  const outDir = values.out ? path.resolve(values.out) : path.join(REPO_ROOT, "results");

  let cache;
  try {
    cache = JSON.parse(await readFile(cachePath, "utf8"));
  } catch {
    process.stderr.write(
      `No response cache at ${cachePath}\nRun: node fishing/build-cache.mjs\n`,
    );
    return 1;
  }

  const readout = createReadout();
  const trainer = createTrainer({
    readout,
    learningRate: Number.parseFloat(values["learning-rate"]),
    gamma: Number.parseFloat(values.gamma),
  });

  const heldOut = evalSeeds(runSeed, evalCount);
  const reference = {
    oracle: evaluate({
      cache,
      seeds: heldOut,
      makePolicy: ({ events }) => createPolicy("oracle", { task: TASK, events }),
    }),
    random: evaluate({
      cache,
      seeds: heldOut,
      makePolicy: ({ seed, events }) => createPolicy("random", { task: TASK, events, seed }),
    }),
  };

  say("");
  say(`  training the readout, seed ${runSeed}`);
  say(`  cache: ${path.relative(process.cwd(), cachePath)} (ablation ${cache.ablation})`);
  say(
    `  held out ${evalCount} episodes; oracle catches ${(reference.oracle.catchRate * 100).toFixed(1)}%,` +
      ` random control catches ${(reference.random.catchRate * 100).toFixed(1)}%`,
  );
  say("");

  const points = [];
  const checkpoints = [];
  for (let episode = 1; episode <= episodes; episode++) {
    const seed = deriveSeed(runSeed, "train", episode);
    const events = buildSchedule(seed, { task: TASK });
    const rows = spliceEpisode({ cache, events, seed, task: TASK });
    const { decisions } = runEpisode({
      featureAt: fromRows(rows),
      policy: createPolicy("readout", { readout, seed: deriveSeed(seed, "act") }),
      events,
      task: TASK,
    });
    trainer.update(decisions);

    if (episode % evalEvery === 0 || episode === 1) {
      const result = evaluate({
        cache,
        seeds: heldOut,
        makePolicy: () => createPolicy("readoutGreedy", { task: TASK, readout }),
      });
      points.push({
        episode,
        catchRate: Number(result.catchRate.toFixed(4)),
        falseHooksPerMinute: Number(result.falseHooksPerMinute.toFixed(3)),
        hookPrecision: Number(result.hookPrecision.toFixed(4)),
        meanReward: Number(result.totalReward.toFixed(3)),
      });
      checkpoints.push({ episode, weights: [...readout.weights] });
      say(
        `  episode ${String(episode).padStart(4)}   catch ${(result.catchRate * 100).toFixed(1).padStart(5)}%` +
          `   false hooks ${result.falseHooksPerMinute.toFixed(2).padStart(5)}/min` +
          `   reward ${result.totalReward.toFixed(2).padStart(6)}`,
      );
    }
  }

  const final = evaluate({
    cache,
    seeds: heldOut,
    makePolicy: () => createPolicy("readoutGreedy", { task: TASK, readout }),
  });

  await mkdir(outDir, { recursive: true });
  const checkpointPath = path.join(outDir, "readout.json");
  await writeFile(
    checkpointPath,
    `${JSON.stringify(
      {
        schemaVersion: 1,
        generatedAt: new Date().toISOString(),
        runSeed,
        episodes,
        learningRate: Number.parseFloat(values["learning-rate"]),
        gamma: Number.parseFloat(values.gamma),
        cache: { ablation: cache.ablation, seed: cache.seed, generatedAt: cache.generatedAt },
        readout: readout.toJSON(),
        heldOut: {
          episodes: evalCount,
          catchRate: Number(final.catchRate.toFixed(4)),
          falseHooksPerMinute: Number(final.falseHooksPerMinute.toFixed(3)),
          hookPrecision: Number(final.hookPrecision.toFixed(4)),
          meanReward: Number(final.totalReward.toFixed(3)),
        },
        frozen:
          "The connectome, its weights, the LIF constants and upstream's stimulus and " +
          "readout interfaces. Nothing inside the simulator is trained, and no claim is " +
          "made that the fly learns anything.",
        trained: `${readout.toJSON().weights.length} linear readout weights, by REINFORCE.`,
      },
      null,
      2,
    )}\n`,
    "utf8",
  );

  const curvePath = path.join(outDir, "learning-curve.json");
  await writeFile(
    curvePath,
    `${JSON.stringify(
      {
        schemaVersion: 1,
        runSeed,
        evalEpisodes: evalCount,
        ablation: cache.ablation,
        reference: {
          oracleCatchRate: Number(reference.oracle.catchRate.toFixed(4)),
          randomCatchRate: Number(reference.random.catchRate.toFixed(4)),
          randomFalseHooksPerMinute: Number(reference.random.falseHooksPerMinute.toFixed(3)),
        },
        points,
        checkpoints,
      },
      null,
      2,
    )}\n`,
    "utf8",
  );

  const svgPath = path.join(outDir, "learning-curve.svg");
  await writeFile(
    svgPath,
    learningCurveSvg({
      title: `Readout training, seed ${runSeed}${cache.ablation === "none" ? "" : ` (${cache.ablation})`}`,
      subtitle:
        "The connectome is frozen. What moves on these axes is a " +
        `${FEATURE_COUNT}-weight linear readout.`,
      points,
      reference: {
        oracleCatchRate: reference.oracle.catchRate,
        randomCatchRate: reference.random.catchRate,
        randomFalseHooksPerMinute: reference.random.falseHooksPerMinute,
      },
    }),
    "utf8",
  );

  say("");
  say(`  final: catch ${(final.catchRate * 100).toFixed(1)}%, ` +
      `false hooks ${final.falseHooksPerMinute.toFixed(2)}/min, ` +
      `mean reward ${final.totalReward.toFixed(2)}`);
  say(`  wrote ${path.relative(process.cwd(), checkpointPath)}`);
  say(`        ${path.relative(process.cwd(), curvePath)}`);
  say(`        ${path.relative(process.cwd(), svgPath)}`);
  say("");
  return 0;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  process.exitCode = await main();
}
