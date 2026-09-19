#!/usr/bin/env node
// Record episodes to the compact replay files the viewer plays.
//
// A recording is self-contained: the viewer needs no backend to play one, which
// is what lets web/ deploy as a static site.

import path from "node:path";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { parseArgs } from "node:util";

import { READOUT_FEATURES, REPO_ROOT } from "./brain-host.mjs";
import { spliceEpisode } from "./cache.mjs";
import { fromRows, runEpisode } from "./episode.mjs";
import { FEATURE_COUNT, createPolicy, createReadout } from "./readout.mjs";
import { deriveSeed } from "./rng.mjs";
import { TASK, buildSchedule } from "./task.mjs";
import { evalSeeds } from "./train.mjs";

export const REPLAY_SCHEMA_VERSION = 2;

// Looked up rather than hardcoded, so reordering READOUT_FEATURES cannot
// silently send the wrong column into a recording.
const ESCAPE_INDEX = READOUT_FEATURES.indexOf("escape_giant_fiber");
const WALK_INDEX = READOUT_FEATURES.indexOf("walk_dnp09");

const round1 = (value) => Math.round(value * 10) / 10;
const round3 = (value) => Math.round(value * 1000) / 1000;

/**
 * The bobber's vertical dip, 0 (floating) to 1 (pulled under), per window.
 *
 * Entirely scripted: it is a drawn envelope around each fish event, not
 * simulator output. It exists so the viewer has something to animate and so a
 * dip-reacting baseline has a cue to react to.
 */
export function bobberTrack(events, windows, task = TASK) {
  const track = new Array(windows).fill(0);
  for (const event of events) {
    const start = Math.round(event.atMs / task.windowMs);
    const length = Math.round((task.biteDurationMs + 350) / task.windowMs);
    const strength = event.type === "bite" ? 1 : 0.45;
    for (let i = 0; i < length; i++) {
      const index = start + i;
      if (index >= windows) break;
      const phase = i / length;
      // Fast pull under, slower bob back up.
      const shape = phase < 0.25 ? phase / 0.25 : Math.max(0, 1 - (phase - 0.25) / 0.75);
      track[index] = Math.max(track[index], strength * shape);
    }
  }
  return track;
}

export function buildReplay({ cache, seed, policyKind, label, readout = null, task = TASK }) {
  const events = buildSchedule(seed, { task });
  const rows = spliceEpisode({ cache, events, seed, task });
  const policy =
    policyKind === "readoutGreedy"
      ? createPolicy("readoutGreedy", { task, readout })
      : createPolicy(policyKind, { task, events, seed: deriveSeed(seed, "act") });

  const { summary, trace } = runEpisode({
    featureAt: fromRows(rows),
    policy,
    events,
    task,
    collect: true,
  });

  const withinAnyDecoy = (tMs) =>
    events.some(
      (event) =>
        event.type === "decoy" && tMs >= event.atMs && tMs < event.atMs + task.hookWindowMs,
    );
  const outcomes = trace
    .filter((frame) => frame.outcome === "catch" || frame.outcome === "snap")
    .map((frame) => ({
      tMs: frame.tMs,
      type: frame.outcome,
      // A snap on a decoy and a snap on empty water are both snapped lines, but
      // only one of them is the discrimination failing.
      ...(frame.outcome === "snap" ? { onDecoy: withinAnyDecoy(frame.tMs) } : {}),
    }));

  return {
    schemaVersion: REPLAY_SCHEMA_VERSION,
    kind: "fly-fishing-replay",
    generatedAt: new Date().toISOString(),
    policy: policyKind,
    label,
    seed,
    windowMs: task.windowMs,
    episodeMs: task.episodeMs,
    task: {
      hookWindowMs: task.hookWindowMs,
      recastMs: task.recastMs,
      biteLoomHz: task.biteLoomHz,
      decoyLoomHz: task.decoyLoomHz,
      rewardCatch: task.rewardCatch,
      rewardSnap: task.rewardSnap,
    },
    simulator: cache.simulator,
    summary: {
      bites: summary.bites,
      caught: summary.caught,
      snapped: summary.snapped,
      decoys: summary.decoys,
      decoysHooked: summary.decoysHooked,
      missed: summary.missed,
      totalReward: summary.totalReward,
      catchRate: round3(summary.catchRate),
      decoyHookRate: round3(summary.decoyHookRate),
      falseHooksPerMinute: round3(summary.falseHooksPerMinute),
    },
    events: events.map((event) => ({ tMs: event.atMs, type: event.type })),
    outcomes,
    // Columnar, one entry per decision window, so the file stays small.
    frames: {
      escapeHz: trace.map((frame) => round1(frame.rates[ESCAPE_INDEX])),
      walkHz: trace.map((frame) => round1(frame.rates[WALK_INDEX])),
      pHook: trace.map((frame) => (frame.probability === null ? null : round3(frame.probability))),
      bobber: bobberTrack(events, trace.length, task).map(round3),
    },
    provenance: {
      simulated: [
        "escapeHz and walkHz: raw readout rates from the upstream whole-brain LIF network",
      ],
      scripted: [
        "the bite schedule, the bobber track, the hook window, the re-cast, and all scoring",
        "the dock, the water and the fly's position in the viewer",
      ],
      frozen: "the connectome",
      trained: `a ${FEATURE_COUNT}-weight linear readout, by REINFORCE`,
      disclaimer:
        "This is a property of this code and its hand-designed sensory interfaces. It is " +
        "not a biological measurement and the fly does not learn anything.",
    },
  };
}

const OPTIONS = {
  seed: { type: "string", default: "1592594996" },
  cache: { type: "string" },
  readout: { type: "string" },
  out: { type: "string" },
  help: { type: "boolean", default: false },
};

const USAGE = `Usage: node fishing/record.mjs [options]

  --seed N       run seed (default 1592594996); picks the recorded episode
  --cache PATH   default results/dn-cache.json
  --readout PATH default results/readout.json
  --out DIR      default web/public/recordings
  --help
`;

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

  const runSeed = Number.parseInt(values.seed, 10) >>> 0;
  const cachePath = values.cache
    ? path.resolve(values.cache)
    : path.join(REPO_ROOT, "results", "dn-cache.json");
  const readoutPath = values.readout
    ? path.resolve(values.readout)
    : path.join(REPO_ROOT, "results", "readout.json");
  const outDir = values.out ? path.resolve(values.out) : path.join(REPO_ROOT, "web", "public", "recordings");

  let cache;
  try {
    cache = JSON.parse(await readFile(cachePath, "utf8"));
  } catch {
    process.stderr.write(`No response cache at ${cachePath}\nRun: node fishing/build-cache.mjs\n`);
    return 1;
  }
  let checkpoint;
  try {
    checkpoint = JSON.parse(await readFile(readoutPath, "utf8"));
  } catch {
    process.stderr.write(`No readout at ${readoutPath}\nRun: node fishing/train.mjs\n`);
    return 1;
  }
  const readout = createReadout({ weights: checkpoint.readout.weights });

  // The recorded episode is a held-out one, so the demo shows the readout on a
  // schedule it was never trained against.
  const seed = evalSeeds(runSeed, 1)[0];

  await mkdir(outDir, { recursive: true });
  const written = [];
  for (const [file, kind, label] of [
    ["trained.json", "readoutGreedy", "Trained readout"],
    ["random.json", "random", "Random control"],
  ]) {
    const replay = buildReplay({ cache, seed, policyKind: kind, label, readout });
    const target = path.join(outDir, file);
    await writeFile(target, `${JSON.stringify(replay)}\n`, "utf8");
    written.push({ target, replay });
  }

  const index = {
    schemaVersion: REPLAY_SCHEMA_VERSION,
    generatedAt: new Date().toISOString(),
    seed,
    recordings: written.map(({ target, replay }) => ({
      file: path.basename(target),
      label: replay.label,
      policy: replay.policy,
      summary: replay.summary,
    })),
  };
  await writeFile(path.join(outDir, "index.json"), `${JSON.stringify(index, null, 2)}\n`, "utf8");

  for (const { target, replay } of written) {
    console.log(
      `  ${path.relative(process.cwd(), target).padEnd(46)}` +
        ` ${replay.summary.caught}/${replay.summary.bites} caught,` +
        ` ${replay.summary.decoysHooked}/${replay.summary.decoys} decoys hooked,` +
        ` ${replay.summary.snapped} snapped,` +
        ` reward ${replay.summary.totalReward}`,
    );
  }
  return 0;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  process.exitCode = await main();
}
