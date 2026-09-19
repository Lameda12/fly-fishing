#!/usr/bin/env node
// Evaluate every policy on one identical set of episodes and print the results
// table as markdown, so the README's numbers are generated rather than typed.
//
// Also writes results/baselines.json, which is the machine-readable version and
// carries its own provenance block.

import path from "node:path";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { parseArgs } from "node:util";

import { REPO_ROOT } from "../fishing/brain-host.mjs";
import { EVENT_SECTIONS, spliceResidual } from "../fishing/cache.mjs";
import { createPolicy, createReadout, FEATURE_NAMES } from "../fishing/readout.mjs";
import { deriveSeed } from "../fishing/rng.mjs";
import { TASK } from "../fishing/task.mjs";
import { evaluate } from "../fishing/train.mjs";

const OPTIONS = {
  seed: { type: "string", default: "1592594996" },
  episodes: { type: "string", default: "1000" },
  cache: { type: "string" },
  readout: { type: "string" },
  out: { type: "string" },
  help: { type: "boolean", default: false },
};

const USAGE = `Usage: node tools/report.mjs [options]

  --seed N        evaluation seed (default 1592594996)
  --episodes N    evaluation episodes, shared by every policy (default 1000)
  --cache PATH    default results/dn-cache.json
  --readout PATH  default results/readout.json
  --out PATH      default results/baselines.json
  --help
`;

/** Every policy is scored on this same list of episodes. */
function reportSeeds(seed, count) {
  return Array.from({ length: count }, (_, i) => deriveSeed(seed, "report", i));
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

  const seed = Number.parseInt(values.seed, 10) >>> 0;
  const episodes = Number.parseInt(values.episodes, 10);
  const cachePath = values.cache
    ? path.resolve(values.cache)
    : path.join(REPO_ROOT, "results", "dn-cache.json");
  const readoutPath = values.readout
    ? path.resolve(values.readout)
    : path.join(REPO_ROOT, "results", "readout.json");
  const out = values.out ? path.resolve(values.out) : path.join(REPO_ROOT, "results", "baselines.json");

  const cache = JSON.parse(await readFile(cachePath, "utf8"));
  const checkpoint = JSON.parse(await readFile(readoutPath, "utf8"));
  const readout = createReadout({ weights: checkpoint.readout.weights });
  const seeds = reportSeeds(seed, episodes);

  const policies = [
    [
      "Oracle (hooks on true bites)",
      "oracle",
      ({ events }) => createPolicy("oracle", { task: TASK, events }),
    ],
    [
      "Trained readout (greedy)",
      "readoutGreedy",
      () => createPolicy("readoutGreedy", { task: TASK, readout }),
    ],
    [
      "Fixed delay after the bobber dips",
      "dipDelay",
      ({ events }) => createPolicy("dipDelay", { task: TASK, events }),
    ],
    [
      "Fixed interval, every 5 s",
      "fixedInterval",
      ({ events }) => createPolicy("fixedInterval", { task: TASK, events, intervalMs: 5000 }),
    ],
    [
      "Random control (rate matched)",
      "random",
      ({ seed: episodeSeed, events }) =>
        createPolicy("random", { task: TASK, events, seed: episodeSeed }),
    ],
  ];

  const rows = policies.map(([label, kind, makePolicy]) => {
    const result = evaluate({ cache, seeds, makePolicy, task: TASK });
    return {
      label,
      policy: kind,
      catchRate: Number(result.catchRate.toFixed(4)),
      decoyHookRate: Number(result.decoyHookRate.toFixed(4)),
      falseHooksPerMinute: Number(result.falseHooksPerMinute.toFixed(3)),
      hookPrecision: Number(result.hookPrecision.toFixed(4)),
      meanReward: Number(result.totalReward.toFixed(3)),
    };
  });

  const report = {
    schemaVersion: 1,
    generatedAt: new Date().toISOString(),
    seed,
    episodes,
    evaluatedOn: "spliced cache episodes",
    cache: { ablation: cache.ablation, seed: cache.seed, generatedAt: cache.generatedAt },
    readout: { runSeed: checkpoint.runSeed, episodes: checkpoint.episodes },
    rows,
    spliceResidual: Object.fromEntries(
      EVENT_SECTIONS.filter((type) => cache[type]).map((type) => [type, spliceResidual(cache, type)]),
    ),
    weights: FEATURE_NAMES.map((name, i) => ({
      feature: name,
      weight: Number(checkpoint.readout.weights[i].toFixed(4)),
    })),
    provenance: {
      frozen: "The connectome. Nothing inside the simulator is trained.",
      trained: `${FEATURE_NAMES.length} linear readout weights, by REINFORCE.`,
      scripted: "The task, the schedule, all scoring, and every baseline in this table.",
      simulated: "Every descending and motor readout rate the policies are scored from.",
      disclaimer:
        "These are properties of this code and its hand-designed sensory interfaces. " +
        "They are not biological measurements and the fly does not learn to fish.",
    },
  };

  await mkdir(path.dirname(out), { recursive: true });
  await writeFile(out, `${JSON.stringify(report, null, 2)}\n`, "utf8");

  // --- markdown, for pasting into the README -------------------------------
  const lines = [];
  lines.push(`All four policies on the same ${episodes} evaluation episodes, seed ${seed}:`);
  lines.push("");
  lines.push(
    "| Policy | Catch rate | Decoys hooked | Snapped lines / min | Hook precision | Mean reward |",
  );
  lines.push("| --- | ---: | ---: | ---: | ---: | ---: |");
  for (const row of rows) {
    lines.push(
      `| ${row.label} | ${pct(row.catchRate)} | ${pct(row.decoyHookRate)} | ` +
        `${row.falseHooksPerMinute.toFixed(2)} | ` +
        `${pct(row.hookPrecision)} | ${row.meanReward.toFixed(2)} |`,
    );
  }
  lines.push("");
  for (const [type, residual] of Object.entries(report.spliceResidual)) {
    lines.push(`Splice residual for a ${type}, response tail minus baseline:`);
    lines.push("");
    lines.push("| Feature | Baseline (Hz) | Response tail (Hz) | Residual (Hz) |");
    lines.push("| --- | ---: | ---: | ---: |");
    for (const row of residual) {
      lines.push(
        `| \`${row.feature}\` | ${row.baselineHz} | ${row.responseTailHz} | ${row.residualHz} |`,
      );
    }
    lines.push("");
  }
  lines.push("");
  lines.push("Readout weights:");
  lines.push("");
  lines.push("| Feature | Weight |");
  lines.push("| --- | ---: |");
  for (const row of report.weights) {
    if (Math.abs(row.weight) < 0.1) continue;
    lines.push(`| \`${row.feature}\` | ${row.weight.toFixed(2)} |`);
  }
  lines.push("");
  lines.push(
    `_(weights under 0.1 in magnitude omitted; all ${FEATURE_NAMES.length} are in results/baselines.json)_`,
  );

  process.stdout.write(`${lines.join("\n")}\n`);
  process.stderr.write(`\nwrote ${path.relative(process.cwd(), out)}\n`);
  return 0;
}

process.exitCode = await main();
