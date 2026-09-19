#!/usr/bin/env node
// Live mode: run the task against the live network and stream it to the viewer.
//
// This is the same task, the same scoring and the same readout as everything
// else; the only difference is where the feature rows come from. Replay and the
// training cache both hand `runEpisode` recorded rows. Here each row is a frame
// that just came out of upstream's engine.
//
// **It runs at about a ninth of real time.** One 50 ms decision window costs
// roughly 400 ms of wall clock, so a 60 s episode takes about eight minutes of
// watching. That is the honest cost of driving 138,639 neurons per window, and
// it is why training uses a cache. The stream therefore carries brain time as
// well as wall time, and the viewer shows brain time, so what you see is the
// fly's clock rather than the machine's.

import path from "node:path";
import { readFile } from "node:fs/promises";
import { parseArgs } from "node:util";

import { REPO_ROOT, createBrainHost, featuresOf } from "./brain-host.mjs";
import { buildFeatures, createPolicy, createReadout } from "./readout.mjs";
import { bobberTrack } from "./record.mjs";
import { deriveSeed } from "./rng.mjs";
import { TASK, buildSchedule, createEpisode, stimulusAt } from "./task.mjs";
import { createWebSocketServer } from "./ws-server.mjs";

export const LIVE_SCHEMA_VERSION = 1;

const ESCAPE = 6;
const WALK = 1;
const round1 = (value) => Math.round(value * 10) / 10;
const round3 = (value) => Math.round(value * 1000) / 1000;

const OPTIONS = {
  seed: { type: "string", default: "1592594996" },
  port: { type: "string", default: "8765" },
  episode: { type: "string", default: "4" },
  policy: { type: "string", default: "readoutGreedy" },
  readout: { type: "string" },
  ablation: { type: "string", default: "none" },
  loop: { type: "boolean", default: false },
  help: { type: "boolean", default: false },
};

const USAGE = `Usage: node fishing/live.mjs [options]

  --seed N        run seed (default 1592594996)
  --episode N     which held-out episode to run (default 4, the recorded one)
  --policy KIND   readoutGreedy (default), oracle, dipDelay, random
  --readout PATH  default results/readout.json
  --ablation KIND none | weight-shuffle | input-shuffle (default none)
  --port N        WebSocket port (default 8765)
  --loop          start a new episode when one ends
  --help

Runs at roughly a ninth of real time: a 60 s episode takes about eight minutes.
`;

/**
 * Run one episode against the live network, calling `onFrame` per decision
 * window. Exported so a validation run can use it without a socket.
 *
 * `host.step` is synchronous CPU work, hundreds of milliseconds of it per
 * window, so the loop yields to the event loop between windows. Without that
 * yield nothing else in the process ever runs: the WebSocket server never
 * accepts a connection and no frame is ever flushed, which is exactly what
 * happened the first time this was wired up.
 */
export async function runLiveEpisode({
  host,
  policy,
  events,
  task = TASK,
  onFrame,
  onEvent,
  shouldStop,
}) {
  const windows = Math.round(task.episodeMs / task.windowMs);
  const scorer = createEpisode(events, { task });
  const bobber = bobberTrack(events, windows, task);
  const rows = [];
  let previousRates = [0, 0, 0, 0, 0, 0, 0, 0];
  let nextEvent = 0;

  for (let index = 0; index < windows; index++) {
    if (shouldStop?.()) return { stopped: true, summary: scorer.finish(), rows };
    // Let the server accept connections and flush the previous frame.
    await new Promise((resolve) => setImmediate(resolve));
    const tMs = index * task.windowMs;

    while (nextEvent < events.length && events[nextEvent].atMs <= tMs) {
      onEvent?.(events[nextEvent]);
      nextEvent++;
    }

    // The one line that differs from replay: this steps the real network.
    const frame = host.step(task.windowMs, stimulusAt(events, tMs, task));
    const rates = featuresOf(frame);
    rows.push(rates);
    const features = buildFeatures(rates, previousRates);
    previousRates = rates;

    const open = scorer.canDecide(tMs);
    const action = open ? policy.act(tMs, features) : 0;
    const result = scorer.step(tMs, action);

    onFrame?.({
      tMs,
      rates,
      features,
      probability: policy.probability ? policy.probability(features) : null,
      bobber: bobber[index] ?? 0,
      result,
      tally: scorer.tally,
      computeMs: frame.computeMs,
    });
  }
  return { stopped: false, summary: scorer.finish(), rows };
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

  const runSeed = Number.parseInt(values.seed, 10) >>> 0;
  const port = Number.parseInt(values.port, 10);
  const episodeIndex = Number.parseInt(values.episode, 10);
  const readoutPath = values.readout
    ? path.resolve(values.readout)
    : path.join(REPO_ROOT, "results", "readout.json");

  let readout = null;
  if (values.policy === "readoutGreedy" || values.policy === "readout") {
    try {
      const checkpoint = JSON.parse(await readFile(readoutPath, "utf8"));
      readout = createReadout({ weights: checkpoint.readout.weights });
    } catch {
      process.stderr.write(`No readout at ${readoutPath}\nRun: node fishing/train.mjs\n`);
      return 1;
    }
  }

  console.log(`  loading the simulator (ablation ${values.ablation})`);
  const host = await createBrainHost({
    seed: runSeed,
    ablation: values.ablation,
    onStatus: (message) => console.log(`  . ${message}`),
  });
  console.log(
    `  ready: ${host.neuronCount.toLocaleString("en-US")} neurons,` +
      ` ${host.edgeCount.toLocaleString("en-US")} weighted edges`,
  );

  let stopping = false;
  // The episode in progress, so a viewer that connects part-way through is not
  // left waiting for the next one to start before it can render anything.
  let currentEpisode = null;
  const server = createWebSocketServer({
    port,
    onStatus: (message) => console.log(`  . ${message}`),
    onConnect: (client) => {
      client.send(hello());
      if (currentEpisode) client.send(currentEpisode);
    },
  });

  function hello() {
    return {
      type: "hello",
      schemaVersion: LIVE_SCHEMA_VERSION,
      kind: "fly-fishing-live",
      policy: values.policy,
      seed: runSeed,
      windowMs: TASK.windowMs,
      episodeMs: TASK.episodeMs,
      ablation: values.ablation,
      task: {
        hookWindowMs: TASK.hookWindowMs,
        recastMs: TASK.recastMs,
        biteLoomHz: TASK.biteLoomHz,
        decoyLoomHz: TASK.decoyLoomHz,
        rewardCatch: TASK.rewardCatch,
        rewardSnap: TASK.rewardSnap,
      },
      simulator: {
        source: "statsleelab/embodied-fly-lab",
        dataset: host.manifest.dataset,
        neurons: host.neuronCount,
        edges: host.edgeCount,
      },
      provenance: {
        frozen: "the connectome",
        trained: "a 17-weight linear readout, by REINFORCE",
        note:
          "Every escapeHz and walkHz in this stream is a rate that came out of the " +
          "network in the preceding 50 ms of brain time. The bobber, the schedule " +
          "and the scoring are scripted.",
      },
    };
  }

  let address;
  try {
    address = await server.listen();
  } catch (error) {
    process.stderr.write(`${error.message}\n`);
    return 1;
  }
  console.log("");
  console.log(`  live stream: ws://127.0.0.1:${address.port}`);
  console.log(`  open the viewer and switch it to live, or: npm run dev -- --open`);
  console.log("  about a ninth of real time: a 60 s episode takes roughly eight minutes.");
  console.log("  Ctrl+C to stop.");
  console.log("");

  const stop = () => {
    stopping = true;
  };
  process.on("SIGINT", stop);
  process.on("SIGTERM", stop);

  let episode = episodeIndex;
  do {
    const seed = deriveSeed(runSeed, "eval", episode);
    const events = buildSchedule(seed, { task: TASK });
    host.reset(seed);
    host.settle(500, { hungerHz: TASK.backgroundHungerHz });

    currentEpisode = {
      type: "episode-start",
      episode,
      seed,
      events: events.map((event) => ({ tMs: event.atMs, type: event.type })),
    };
    server.broadcast(currentEpisode);
    console.log(
      `  episode ${episode} (seed ${seed}): ` +
        `${events.filter((e) => e.type === "bite").length} bites, ` +
        `${events.filter((e) => e.type === "decoy").length} decoys`,
    );

    const policy =
      values.policy === "readoutGreedy"
        ? createPolicy("readoutGreedy", { task: TASK, readout })
        : createPolicy(values.policy, { task: TASK, events, seed: deriveSeed(seed, "act") });

    const started = Date.now();
    const { summary, stopped } = await runLiveEpisode({
      host,
      policy,
      events,
      shouldStop: () => stopping,
      onEvent: (event) =>
        server.broadcast({ type: "event", tMs: event.atMs, event: event.type }),
      onFrame: (frame) => {
        server.broadcast({
          type: "frame",
          tMs: frame.tMs,
          escapeHz: round1(frame.rates[ESCAPE]),
          walkHz: round1(frame.rates[WALK]),
          pHook: frame.probability === null ? null : round3(frame.probability),
          bobber: round3(frame.bobber),
          outcome: frame.result.outcome,
          caught: frame.tally.caught,
          snapped: frame.tally.snapped,
          decoysHooked: frame.tally.decoysHooked,
          computeMs: round1(frame.computeMs),
        });
        if (frame.result.outcome === "catch" || frame.result.outcome === "snap") {
          server.broadcast({ type: "outcome", tMs: frame.tMs, outcome: frame.result.outcome });
        }
      },
    });

    const wall = ((Date.now() - started) / 1000).toFixed(0);
    server.broadcast({ type: "episode-end", episode, summary, wallSeconds: Number(wall) });
    console.log(
      `  episode ${episode} ${stopped ? "stopped" : "done"}: ` +
        `${summary.caught}/${summary.bites} caught, ` +
        `${summary.decoysHooked}/${summary.decoys} decoys hooked, ` +
        `reward ${summary.totalReward} (${wall}s wall)`,
    );
    episode++;
  } while (values.loop && !stopping);

  server.close();
  return 0;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  process.exitCode = await main();
}
