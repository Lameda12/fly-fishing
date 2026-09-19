#!/usr/bin/env node
// Record voyages to the replay files the viewer plays.
//
// Same format as the single-stage fishing recording, plus a `stages` array.
// The viewer uses that to place the boat, light the fire and name the stage;
// a recording without it is a single-stage fishing episode and plays as before.

import path from "node:path";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { parseArgs } from "node:util";

import { READOUT_FEATURES, REPO_ROOT } from "./brain-host.mjs";
import { createPolicy, createReadout, FEATURE_COUNT } from "./readout.mjs";
import { TASK } from "./task.mjs";
import { STAGES, STAGE_STARTS } from "./voyage.mjs";
import { runVoyage, voyageReflexPolicy } from "./voyage-episode.mjs";
import { voyageEvalSeeds } from "./train-voyage.mjs";

export const REPLAY_SCHEMA_VERSION = 2;

const ESCAPE = READOUT_FEATURES.indexOf("escape_giant_fiber");
const WALK = READOUT_FEATURES.indexOf("walk_dnp09");
const round1 = (value) => Math.round(value * 10) / 10;
const round3 = (value) => Math.round(value * 1000) / 1000;

/**
 * The bobber's dip, per window.
 *
 * Only the fishing stage has a bobber, so only its events move it. The other
 * stages have their own tells in the scene and do not need one. Scripted.
 */
function voyageBobberTrack(events, windows, task = TASK) {
  const track = new Array(windows).fill(0);
  for (const event of events) {
    if (event.stage !== "fish") continue;
    const start = Math.round(event.atMs / task.windowMs);
    const length = Math.round((task.biteDurationMs + 350) / task.windowMs);
    const strength = event.rewarding ? 1 : 0.45;
    for (let i = 0; i < length; i++) {
      const index = start + i;
      if (index >= windows) break;
      const phase = i / length;
      const shape = phase < 0.25 ? phase / 0.25 : Math.max(0, 1 - (phase - 0.25) / 0.75);
      track[index] = Math.max(track[index], strength * shape);
    }
  }
  return track;
}

export function buildVoyageReplay({ cache, seed, policySpec, label, policyName, task = TASK }) {
  const { summary, trace, events } = runVoyage({
    cache,
    seed,
    policy: policySpec,
    task,
    collect: true,
  });

  const outcomes = trace
    .filter((frame) => frame.outcome === "good" || frame.outcome === "bad")
    .map((frame) => ({
      tMs: frame.tMs,
      // The viewer knows "catch" and "snap"; a voyage's good and bad acts are
      // the same two things under stage-specific names.
      type: frame.outcome === "good" ? "catch" : "snap",
      stage: frame.stage,
    }));

  const decisionStages = STAGES.filter((stage) => stage.decision);
  const totalChances = decisionStages.reduce(
    (total, stage) => total + summary.perStage[stage.id].chances,
    0,
  );

  return {
    schemaVersion: REPLAY_SCHEMA_VERSION,
    kind: "fly-fishing-replay",
    generatedAt: new Date().toISOString(),
    policy: policyName,
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
    stages: STAGES.map((stage, i) => ({
      id: stage.id,
      name: stage.name,
      startMs: STAGE_STARTS[i],
      durationMs: stage.durationMs,
      decision: stage.decision,
    })),
    // The viewer's counters are named for the fishing task; on a voyage they
    // count correct acts and mistakes across every stage.
    summary: {
      bites: totalChances,
      caught: summary.correct,
      snapped: summary.wrong,
      decoys: decisionStages.reduce(
        (total, stage) =>
          total + (summary.perStage[stage.id].acts - summary.perStage[stage.id].correct),
        0,
      ),
      decoysHooked: summary.wrong,
      missed: totalChances - summary.correct,
      totalReward: summary.totalReward,
      catchRate: round3(summary.successRate),
      decoyHookRate: 0,
      falseHooksPerMinute: round3((summary.wrong * 60_000) / task.episodeMs),
      fishCaught: summary.fishCaught,
      perStage: Object.fromEntries(
        decisionStages.map((stage) => [
          stage.id,
          {
            chances: summary.perStage[stage.id].chances,
            correct: summary.perStage[stage.id].correct,
            wrong: summary.perStage[stage.id].wrong,
          },
        ]),
      ),
    },
    events: events.map((event) => ({
      tMs: event.atMs,
      type: event.stimulus,
      stage: event.stage,
      rewarding: event.rewarding,
    })),
    outcomes,
    frames: {
      escapeHz: trace.map((frame) => round1(frame.rates[ESCAPE])),
      walkHz: trace.map((frame) => round1(frame.rates[WALK])),
      pHook: trace.map((frame) => (frame.probability === null ? null : round3(frame.probability))),
      bobber: voyageBobberTrack(events, trace.length, task).map(round3),
    },
    provenance: {
      simulated: [
        "escapeHz and walkHz: raw readout rates from the upstream whole-brain LIF network",
      ],
      scripted: [
        "the five stages, every schedule, the bobber track, and all scoring",
        "the boat, the jetty, the fire, the water, and where the fly stands",
      ],
      frozen: "the connectome",
      trained: `a ${FEATURE_COUNT}-weight linear readout, shared by all four decision stages`,
      disclaimer:
        "This is a property of this code and its hand-designed sensory interfaces. The fly " +
        "does not learn to fish, cook or eat.",
    },
  };
}

const OPTIONS = {
  seed: { type: "string", default: "1592594996" },
  episode: { type: "string", default: "0" },
  cache: { type: "string" },
  readout: { type: "string" },
  out: { type: "string" },
  help: { type: "boolean", default: false },
};

const USAGE = `Usage: node fishing/record-voyage.mjs [options]

  --seed N       run seed (default 1592594996)
  --episode N    which held-out voyage to record (default 0)
  --cache PATH   default results/dn-cache.json
  --readout PATH default results/voyage.json
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
  const episodeIndex = Number.parseInt(values.episode, 10);
  const cachePath = values.cache
    ? path.resolve(values.cache)
    : path.join(REPO_ROOT, "results", "dn-cache.json");
  const readoutPath = values.readout
    ? path.resolve(values.readout)
    : path.join(REPO_ROOT, "results", "voyage.json");
  const outDir = values.out
    ? path.resolve(values.out)
    : path.join(REPO_ROOT, "web", "public", "recordings");

  let cache;
  try {
    cache = JSON.parse(await readFile(cachePath, "utf8"));
  } catch {
    process.stderr.write(`No response cache at ${cachePath}\nRun: python3 run.py cache\n`);
    return 1;
  }
  let checkpoint;
  try {
    checkpoint = JSON.parse(await readFile(readoutPath, "utf8"));
  } catch {
    process.stderr.write(`No voyage readout at ${readoutPath}\nRun: python3 run.py voyage\n`);
    return 1;
  }
  if (!checkpoint.readout) {
    process.stderr.write(
      `${path.relative(process.cwd(), readoutPath)} has no trained readout: every seed collapsed.\n`,
    );
    return 1;
  }
  const readout = createReadout({ weights: checkpoint.readout.weights });

  const seed = voyageEvalSeeds(runSeed, episodeIndex + 1)[episodeIndex];
  await mkdir(outDir, { recursive: true });

  const written = [];
  for (const [file, policyName, label, policySpec] of [
    [
      "trained.json",
      "readoutGreedy",
      "Trained readout",
      createPolicy("readoutGreedy", { task: TASK, readout }),
    ],
    [
      "reflex.json",
      "reflex",
      "Reflex baseline",
      ({ events }) => voyageReflexPolicy({ scorerEvents: events }),
    ],
  ]) {
    const replay = buildVoyageReplay({ cache, seed, policySpec, label, policyName });
    const target = path.join(outDir, file);
    await writeFile(target, `${JSON.stringify(replay)}\n`, "utf8");
    written.push({ target, replay });
  }

  await writeFile(
    path.join(outDir, "index.json"),
    `${JSON.stringify(
      {
        schemaVersion: REPLAY_SCHEMA_VERSION,
        generatedAt: new Date().toISOString(),
        seed,
        kind: "voyage",
        recordings: written.map(({ target, replay }) => ({
          file: path.basename(target),
          label: replay.label,
          policy: replay.policy,
          summary: replay.summary,
        })),
      },
      null,
      2,
    )}\n`,
    "utf8",
  );

  for (const { target, replay } of written) {
    const perStage = STAGES.filter((stage) => stage.decision)
      .map((stage) => {
        const tally = replay.summary.perStage[stage.id];
        return `${stage.id} ${tally.correct}/${tally.chances}`;
      })
      .join("  ");
    console.log(
      `  ${path.relative(process.cwd(), target).padEnd(44)} ${perStage}   reward ${replay.summary.totalReward}`,
    );
  }
  return 0;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  process.exitCode = await main();
}
