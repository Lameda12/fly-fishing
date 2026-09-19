#!/usr/bin/env node
// Record the descending-neuron response cache from the live simulator.
//
// This is the only step that costs real compute. Everything downstream -
// training, the baselines, the recordings - reads the file it writes.

import path from "node:path";
import { mkdir, writeFile } from "node:fs/promises";
import { parseArgs } from "node:util";

import { REPO_ROOT, createBrainHost } from "./brain-host.mjs";
import { BASELINE_MS, RESPONSE_MS, SETTLE_MS, buildCache, spliceResidual } from "./cache.mjs";
import { TASK } from "./task.mjs";

const OPTIONS = {
  seed: { type: "string", default: "1592594996" },
  ablation: { type: "string", default: "none" },
  "baseline-traces": { type: "string", default: "4" },
  "bite-traces": { type: "string", default: "16" },
  out: { type: "string" },
  help: { type: "boolean", default: false },
};

const USAGE = `Usage: node fishing/build-cache.mjs [options]

  --seed N              cache seed (default 1592594996)
  --ablation KIND       none | weight-shuffle | input-shuffle (default none)
  --baseline-traces N   long background traces to record (default 4)
  --bite-traces N       bite responses to record (default 16)
  --out PATH            default results/dn-cache.json, or dn-cache-<ablation>.json
  --help
`;

function defaultOut(ablation) {
  const name = ablation === "none" ? "dn-cache.json" : `dn-cache-${ablation}.json`;
  return path.join(REPO_ROOT, "results", name);
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

  const seed = Number.parseInt(values.seed, 10) >>> 0;
  const baselineTraces = Number.parseInt(values["baseline-traces"], 10);
  const biteTraces = Number.parseInt(values["bite-traces"], 10);
  const out = values.out ? path.resolve(values.out) : defaultOut(values.ablation);

  const biologicalSeconds =
    (baselineTraces * (SETTLE_MS + BASELINE_MS) + biteTraces * (SETTLE_MS + RESPONSE_MS)) / 1000;
  console.log(`  cache seed ${seed}, ablation ${values.ablation}`);
  console.log(
    `  recording ${baselineTraces} baseline traces and ${biteTraces} bite responses` +
      ` (${biologicalSeconds.toFixed(0)} s of brain time)`,
  );

  const started = Date.now();
  const host = await createBrainHost({
    seed,
    ablation: values.ablation,
    onStatus: (message) => console.log(`  . ${message}`),
  });
  console.log(
    `  ready: ${host.neuronCount.toLocaleString("en-US")} neurons,` +
      ` ${host.edgeCount.toLocaleString("en-US")} weighted edges`,
  );

  const cache = await buildCache({
    host,
    task: TASK,
    seed,
    baselineTraces,
    biteTraces,
    onProgress: ({ stage, index, of }) => {
      const elapsed = ((Date.now() - started) / 1000).toFixed(0);
      console.log(`  . ${stage} trace ${index + 1}/${of}  (${elapsed}s elapsed)`);
    },
  });
  cache.wallSeconds = Number(((Date.now() - started) / 1000).toFixed(1));

  await mkdir(path.dirname(out), { recursive: true });
  await writeFile(out, `${JSON.stringify(cache)}\n`, "utf8");

  console.log("");
  console.log("  splice residual (response tail minus baseline, Hz):");
  for (const row of spliceResidual(cache)) {
    console.log(
      `    ${row.feature.padEnd(20)} baseline ${String(row.baselineHz).padStart(7)}` +
        `  tail ${String(row.responseTailHz).padStart(7)}` +
        `  residual ${String(row.residualHz).padStart(7)}`,
    );
  }
  console.log("");
  console.log(`  wrote ${path.relative(process.cwd(), out)} in ${cache.wallSeconds}s`);
  return 0;
}

process.exitCode = await main();
