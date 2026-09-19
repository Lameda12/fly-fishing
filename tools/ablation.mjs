#!/usr/bin/env node
// The ablation: does the wiring matter, or is the readout doing all the work?
//
// Three caches, each recorded from the same simulator through the same stimulus
// channels, differing only in what was done to the connectome before upstream's
// unmodified BrainEngine was constructed from it:
//
//   none            the FlyWire connectome as it is
//   weight-shuffle  every edge keeps its source and target; only which weight
//                   sits on which edge is permuted. The multiset of weights,
//                   and so the total drive available, is identical.
//   input-shuffle   the graph and its weights are untouched; the stimulus and
//                   readout populations are re-drawn as random neurons of the
//                   same count, so the pulse goes into arbitrary cells and the
//                   decision is read out of arbitrary cells.
//
// A readout is fitted to each from scratch under identical hyperparameters,
// seeds and episode schedules, and all three are scored on the same evaluation
// episodes. Two numbers matter per row: what the readout scores, and the d' the
// recorded responses carry before any readout sees them. The second is what
// separates "the wiring mattered" from "the readout found whatever was there".

import path from "node:path";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { parseArgs } from "node:util";

import { ABLATIONS, REPO_ROOT } from "../fishing/brain-host.mjs";
import { discriminability } from "../fishing/cache.mjs";
import { createPolicy, FEATURE_NAMES } from "../fishing/readout.mjs";
import { deriveSeed } from "../fishing/rng.mjs";
import { TASK } from "../fishing/task.mjs";
import { evalSeeds, evaluate, trainReadout } from "../fishing/train.mjs";

const OPTIONS = {
  seed: { type: "string", default: "1592594996" },
  episodes: { type: "string", default: "600" },
  "eval-episodes": { type: "string", default: "1000" },
  seeds: { type: "string", default: "5" },
  out: { type: "string" },
  quiet: { type: "boolean", default: false },
  help: { type: "boolean", default: false },
};

const USAGE = `Usage: node tools/ablation.mjs [options]

  --seed N             run seed (default 1592594996), shared by every arm
  --episodes N         training episodes per arm (default 600)
  --eval-episodes N    evaluation episodes, shared by every arm (default 1000)
  --seeds N            training seeds per arm (default 5). REINFORCE on this task
                       either finds the signal or collapses to never acting, so a
                       single seed reports a coin flip rather than a result.
  --out PATH           default results/ablation.json
  --quiet
  --help

Needs one cache per arm. Record them with:
  node fishing/build-cache.mjs
  node fishing/build-cache.mjs --ablation weight-shuffle
  node fishing/build-cache.mjs --ablation input-shuffle
`;

function cachePath(ablation) {
  const name = ablation === "none" ? "dn-cache.json" : `dn-cache-${ablation}.json`;
  return path.join(REPO_ROOT, "results", name);
}

const pct = (value) => `${(value * 100).toFixed(1)}%`;

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

  const say = values.quiet ? () => {} : (text) => console.error(text);
  const runSeed = Number.parseInt(values.seed, 10) >>> 0;
  const episodes = Number.parseInt(values.episodes, 10);
  const evalCount = Number.parseInt(values["eval-episodes"], 10);
  const seedCount = Number.parseInt(values.seeds, 10);
  const out = values.out ? path.resolve(values.out) : path.join(REPO_ROOT, "results", "ablation.json");

  // Every arm trains from the same list of seeds and is scored on the same
  // episodes, so nothing but the connectome differs between them.
  const trainSeeds = Array.from({ length: seedCount }, (_, i) =>
    i === 0 ? runSeed : deriveSeed(runSeed, "arm-seed", i),
  );
  const scoreSeeds = Array.from({ length: evalCount }, (_, i) =>
    deriveSeed(runSeed, "report", i),
  );
  /** A run that never learned to act at all, which is the failure mode here. */
  const converged = (result) => result.catchRate > 0.5;

  const arms = [];
  for (const ablation of ABLATIONS) {
    const file = cachePath(ablation);
    let cache;
    try {
      cache = JSON.parse(await readFile(file, "utf8"));
    } catch {
      process.stderr.write(
        `No cache for "${ablation}" at ${file}\n` +
          `Record it with: node fishing/build-cache.mjs${ablation === "none" ? "" : ` --ablation ${ablation}`}\n`,
      );
      return 1;
    }
    if (cache.ablation !== ablation) {
      process.stderr.write(`${file} says ablation "${cache.ablation}", expected "${ablation}"\n`);
      return 1;
    }

    say(`\n  === ${ablation} ===`);
    const separation = discriminability(cache);
    const escape = separation.find((row) => row.feature === "escape_giant_fiber");
    const best = separation.reduce(
      (top, row) => ((row.dPrime ?? 0) > (top?.dPrime ?? 0) ? row : top),
      null,
    );
    say(
      `  recorded separation: escape_giant_fiber d' ${escape.dPrime ?? "n/a"}` +
        ` (bite ${escape.bitePeakHz} Hz, decoy ${escape.decoyPeakHz} Hz);` +
        ` best feature ${best?.feature ?? "none"} d' ${best?.dPrime ?? "n/a"}`,
    );

    const started = Date.now();
    const runs = [];
    let bestReadout = null;
    let bestReward = -Infinity;
    for (const trainSeed of trainSeeds) {
      const heldOut = evalSeeds(trainSeed, 24);
      const { readout, points } = trainReadout({
        cache,
        runSeed: trainSeed,
        episodes,
        heldOut,
        evalEvery: 1e9,
      });
      const result = evaluate({
        cache,
        seeds: scoreSeeds,
        makePolicy: () => createPolicy("readoutGreedy", { task: TASK, readout }),
      });
      runs.push({
        trainSeed,
        converged: converged(result),
        catchRate: Number(result.catchRate.toFixed(4)),
        decoyHookRate: Number(result.decoyHookRate.toFixed(4)),
        meanReward: Number(result.totalReward.toFixed(3)),
        curve: points,
      });
      if (result.totalReward > bestReward) {
        bestReward = result.totalReward;
        bestReadout = readout;
      }
      say(
        `  seed ${String(trainSeed).padStart(10)}  catch ${pct(result.catchRate).padStart(6)}` +
          `  decoys ${pct(result.decoyHookRate).padStart(6)}` +
          `  reward ${result.totalReward.toFixed(2).padStart(6)}` +
          `${converged(result) ? "" : "   (collapsed to never acting)"}`,
      );
    }

    const readout = bestReadout;
    const survivors = runs.filter((run) => run.converged);
    const mean = (pick) =>
      survivors.length ? survivors.reduce((t, r) => t + pick(r), 0) / survivors.length : 0;
    const scored = {
      catchRate: mean((r) => r.catchRate),
      decoyHookRate: mean((r) => r.decoyHookRate),
      falseHooksPerMinute: 0,
      hookPrecision: 0,
      totalReward: mean((r) => r.meanReward),
    };
    // The oracle is a property of the schedule, not of the network, so it is the
    // same in every arm. Re-scoring it per arm is the check that that is true.
    const oracle = evaluate({
      cache,
      seeds: scoreSeeds,
      makePolicy: ({ events }) => createPolicy("oracle", { task: TASK, events }),
    });

    arms.push({
      ablation,
      cache: { seed: cache.seed, ablationSeed: cache.ablationSeed, generatedAt: cache.generatedAt },
      separation,
      bestDPrime: best?.dPrime ?? null,
      escapeDPrime: escape.dPrime,
      seeds: trainSeeds,
      runs,
      convergedCount: survivors.length,
      // Averaged over the seeds that converged, so a converged run's score is
      // not dragged down by a collapsed one; the count says how often that was.
      trained: {
        catchRate: Number(scored.catchRate.toFixed(4)),
        decoyHookRate: Number(scored.decoyHookRate.toFixed(4)),
        meanReward: Number(scored.totalReward.toFixed(3)),
      },
      oracle: {
        catchRate: Number(oracle.catchRate.toFixed(4)),
        meanReward: Number(oracle.totalReward.toFixed(3)),
      },
      weights: FEATURE_NAMES.map((name, i) => ({
        feature: name,
        weight: Number(readout.weights[i].toFixed(4)),
      })),
      wallSeconds: Number(((Date.now() - started) / 1000).toFixed(1)),
    });
  }

  const report = {
    schemaVersion: 1,
    generatedAt: new Date().toISOString(),
    runSeed,
    trainEpisodes: episodes,
    evalEpisodes: evalCount,
    trainSeeds,
    arms,
    provenance: {
      frozen:
        "The connectome is frozen in every arm. The ablations permute copies of " +
        "the arrays and of the population index lists before upstream's unmodified " +
        "BrainEngine is constructed from them; no upstream source file is touched.",
      trained: `${FEATURE_NAMES.length} linear readout weights per arm, by REINFORCE.`,
      identical:
        "All three arms share the run seed, the training schedules, the held-out " +
        "schedules, the evaluation episodes and every hyperparameter.",
      disclaimer:
        "These are properties of this code and its hand-designed sensory interfaces. " +
        "They are not biological measurements.",
    },
  };

  await mkdir(path.dirname(out), { recursive: true });
  await writeFile(out, `${JSON.stringify(report, null, 2)}\n`, "utf8");

  const label = {
    none: "Intact connectome",
    "weight-shuffle": "Weights shuffled across edges",
    "input-shuffle": "Stimulus and readout populations randomized",
  };

  const lines = [];
  lines.push(
    `All three arms: the same ${seedCount} training seeds, the same schedules, and the` +
      ` same ${evalCount} evaluation episodes, ${episodes} training episodes per seed.`,
  );
  lines.push("");
  lines.push(
    "| Connectome | escape_giant_fiber d' | Best feature d' | Runs that converged | Catch rate | Decoys hooked | Mean reward |",
  );
  lines.push("| --- | ---: | ---: | ---: | ---: | ---: | ---: |");
  for (const arm of arms) {
    const converged = `${arm.convergedCount}/${seedCount}`;
    const cells = arm.convergedCount
      ? `${pct(arm.trained.catchRate)} | ${pct(arm.trained.decoyHookRate)} | ${arm.trained.meanReward.toFixed(2)}`
      : "- | - | -";
    lines.push(
      `| ${label[arm.ablation]} | ${arm.escapeDPrime ?? "-"} | ${arm.bestDPrime ?? "-"} | ` +
        `${converged} | ${cells} |`,
    );
  }
  lines.push("");
  lines.push(
    "Catch rate, decoys hooked and mean reward are averaged over the seeds that converged." +
      " A run is counted as converged if it learned to act at all; the failure mode here is" +
      " collapsing to never acting, which scores exactly zero.",
  );
  lines.push("");
  lines.push("Peak response per arm, bite against decoy, on the two populations that carry the signal:");
  lines.push("");
  lines.push("| Connectome | `escape_giant_fiber` bite | decoy | `reverse_mdn` bite | decoy |");
  lines.push("| --- | ---: | ---: | ---: | ---: |");
  for (const arm of arms) {
    const e = arm.separation.find((row) => row.feature === "escape_giant_fiber");
    const r = arm.separation.find((row) => row.feature === "reverse_mdn");
    lines.push(
      `| ${label[arm.ablation]} | ${e.bitePeakHz} ± ${e.bitePeakSd} | ${e.decoyPeakHz} ± ${e.decoyPeakSd} | ` +
        `${r.bitePeakHz} ± ${r.bitePeakSd} | ${r.decoyPeakHz} ± ${r.decoyPeakSd} |`,
    );
  }

  process.stdout.write(`${lines.join("\n")}\n`);
  say(`\n  wrote ${path.relative(process.cwd(), out)}`);
  return 0;
}

process.exitCode = await main();
