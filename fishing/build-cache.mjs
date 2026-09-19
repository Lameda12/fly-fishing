#!/usr/bin/env node
// Record the descending-neuron response cache from the live simulator.
//
// This is the only step that costs real compute. Everything downstream -
// training, the baselines, the recordings - reads the file it writes.

import path from "node:path";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { parseArgs } from "node:util";

import { REPO_ROOT, createBrainHost } from "./brain-host.mjs";
import {
  BASELINE_MS,
  EVENT_SECTIONS,
  RESPONSE_MS,
  SECTIONS,
  SETTLE_MS,
  buildCache,
  spliceResidual,
} from "./cache.mjs";
import { TASK } from "./task.mjs";

const OPTIONS = {
  seed: { type: "string", default: "1592594996" },
  ablation: { type: "string", default: "none" },
  "baseline-traces": { type: "string", default: "4" },
  "event-traces": { type: "string", default: "16" },
  sections: { type: "string" },
  "merge-into": { type: "string" },
  out: { type: "string" },
  help: { type: "boolean", default: false },
};

const USAGE = `Usage: node fishing/build-cache.mjs [options]

  --seed N              cache seed (default 1592594996)
  --ablation KIND       none | weight-shuffle | input-shuffle (default none)
  --baseline-traces N   long background traces to record (default 4)
  --event-traces N      responses per event type (default 16)
  --sections a,b        only record these: ${SECTIONS.join(", ")} (default: all)
  --merge-into PATH     add the recorded sections to an existing cache instead of
                        starting a new one. The baselines are the slow part, so
                        this is how a new event type is added without redoing them.
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
  const eventTraces = Number.parseInt(values["event-traces"], 10);
  const sections = values.sections
    ? values.sections.split(",").map((name) => name.trim())
    : [...SECTIONS];
  for (const section of sections) {
    if (!SECTIONS.includes(section)) {
      process.stderr.write(`unknown section "${section}"; expected ${SECTIONS.join(", ")}\n`);
      return 2;
    }
  }

  const mergeInto = values["merge-into"] ? path.resolve(values["merge-into"]) : null;
  const out = values.out
    ? path.resolve(values.out)
    : mergeInto ?? defaultOut(values.ablation);

  let existing = null;
  if (mergeInto) {
    try {
      existing = JSON.parse(await readFile(mergeInto, "utf8"));
    } catch {
      process.stderr.write(`No cache to merge into at ${mergeInto}\n`);
      return 1;
    }
    if (existing.ablation !== values.ablation) {
      process.stderr.write(
        `refusing to merge: ${mergeInto} is ablation "${existing.ablation}", ` +
          `this run is "${values.ablation}"\n`,
      );
      return 1;
    }
    if (existing.seed !== seed) {
      process.stderr.write(
        `refusing to merge: ${mergeInto} was recorded from seed ${existing.seed}, ` +
          `this run uses ${seed}\n`,
      );
      return 1;
    }
  }

  const eventSectionCount = sections.filter((name) => EVENT_SECTIONS.includes(name)).length;
  const biologicalSeconds =
    ((sections.includes("baseline") ? baselineTraces * (SETTLE_MS + BASELINE_MS) : 0) +
      eventSectionCount * eventTraces * (SETTLE_MS + RESPONSE_MS)) /
    1000;
  console.log(`  cache seed ${seed}, ablation ${values.ablation}`);
  console.log(
    `  recording ${sections.join(", ")}` +
      ` (${biologicalSeconds.toFixed(0)} s of brain time)` +
      (mergeInto ? `, merging into ${path.relative(process.cwd(), mergeInto)}` : ""),
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

  const recorded = await buildCache({
    host,
    task: TASK,
    seed,
    baselineTraces,
    eventTraces,
    sections,
    onProgress: ({ stage, index, of }) => {
      const elapsed = ((Date.now() - started) / 1000).toFixed(0);
      console.log(`  . ${stage} trace ${index + 1}/${of}  (${elapsed}s elapsed)`);
    },
  });
  recorded.wallSeconds = Number(((Date.now() - started) / 1000).toFixed(1));

  // A merge keeps whatever the existing cache already holds and replaces only
  // the sections this run recorded, so adding an event type costs one event
  // type's worth of compute rather than a whole cache.
  const cache = existing ? { ...existing, ...recorded } : recorded;
  // Derived from what is actually in the file rather than from a field, so a
  // merge into a cache written by an older build still lists its sections right.
  cache.sections = SECTIONS.filter((name) => cache[name]);

  await mkdir(path.dirname(out), { recursive: true });
  await writeFile(out, `${JSON.stringify(cache)}\n`, "utf8");

  for (const type of EVENT_SECTIONS) {
    if (!cache[type] || !cache.baseline) continue;
    console.log("");
    console.log(`  splice residual for "${type}" (response tail minus baseline, Hz):`);
    for (const row of spliceResidual(cache, type)) {
      console.log(
        `    ${row.feature.padEnd(20)} baseline ${String(row.baselineHz).padStart(7)}` +
          `  tail ${String(row.responseTailHz).padStart(7)}` +
          `  residual ${String(row.residualHz).padStart(7)}`,
      );
    }
  }
  console.log("");
  console.log(
    `  wrote ${path.relative(process.cwd(), out)}` +
      ` (sections: ${cache.sections.join(', ')}) in ${recorded.wallSeconds}s`,
  );
  return 0;
}

process.exitCode = await main();
