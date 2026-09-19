// Two modes over one pond.
//
// Replay plays recorded episodes side by side with no backend, which is what
// lets this deploy as a static site. Live connects to fishing/live.mjs and
// renders the network as it runs.
//
// Live mode reuses the replay Player by building a Replay incrementally as
// frames arrive: the frame columns grow, the outcome list grows, and the
// transport clock follows the last frame received rather than wall time. That
// keeps one rendering path for both modes, so what live shows and what a
// recording shows cannot drift apart.

import "./style.css";
import { connectLive, type LiveEpisodeStart, type LiveFrame, type LiveHello } from "./live";
import { Player } from "./player";
import { assertReplay, type Replay, type ReplayIndex } from "./types";

const BASE = `${import.meta.env.BASE_URL}recordings/`;
const DEFAULT_LIVE_URL = "ws://127.0.0.1:8765";

const byId = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

async function loadJson<T>(file: string): Promise<T> {
  const response = await fetch(`${BASE}${file}`, { cache: "no-store" });
  if (!response.ok) throw new Error(`${file}: ${response.status} ${response.statusText}`);
  return (await response.json()) as T;
}

function fail(message: string, hint: string): void {
  byId("panels").innerHTML = `
    <div class="fatal">
      <h1>Nothing to show</h1>
      <p>${message}</p>
      <pre>${hint}</pre>
    </div>`;
}

/** An empty Replay that live frames are appended to. */
function liveReplay(hello: LiveHello, start: LiveEpisodeStart): Replay {
  return {
    schemaVersion: 2,
    kind: "fly-fishing-replay",
    generatedAt: new Date().toISOString(),
    policy: hello.policy,
    label: `Live: ${hello.policy}${hello.ablation === "none" ? "" : ` (${hello.ablation})`}`,
    seed: start.seed,
    windowMs: hello.windowMs,
    episodeMs: hello.episodeMs,
    task: hello.task,
    simulator: hello.simulator,
    summary: {
      bites: start.events.filter((event) => event.type === "bite").length,
      decoys: start.events.filter((event) => event.type === "decoy").length,
      decoysHooked: 0,
      caught: 0,
      snapped: 0,
      missed: 0,
      totalReward: 0,
      catchRate: 0,
      decoyHookRate: 0,
      falseHooksPerMinute: 0,
    },
    events: start.events,
    outcomes: [],
    frames: { escapeHz: [], walkHz: [], pHook: [], bobber: [] },
    provenance: {
      simulated: [hello.provenance.note],
      scripted: ["the bite schedule, the bobber track, and all scoring"],
      frozen: hello.provenance.frozen,
      trained: hello.provenance.trained,
      disclaimer: "Not a biological measurement.",
    },
  };
}

function appendFrame(replay: Replay, frame: LiveFrame): void {
  const index = Math.round(frame.tMs / replay.windowMs);
  replay.frames.escapeHz[index] = frame.escapeHz;
  replay.frames.walkHz[index] = frame.walkHz;
  replay.frames.pHook[index] = frame.pHook;
  replay.frames.bobber[index] = frame.bobber;
}

// --- replay mode ------------------------------------------------------------

async function startReplay(): Promise<void> {
  let index: ReplayIndex;
  let replays: Replay[];
  try {
    index = await loadJson<ReplayIndex>("index.json");
    replays = await Promise.all(
      index.recordings.map(async (entry) =>
        assertReplay(await loadJson<unknown>(entry.file), entry.file),
      ),
    );
  } catch (error) {
    fail(
      `No recordings: ${error instanceof Error ? error.message : String(error)}`,
      "python3 run.py record",
    );
    return;
  }
  if (!replays.length) {
    fail("The recordings index is empty.", "python3 run.py record");
    return;
  }

  const first = replays[0]!;
  byId("seed").textContent = String(index.seed);
  byId("sim").textContent =
    `${first.simulator.neurons.toLocaleString("en-US")} neurons, ${first.simulator.dataset}`;

  const panels = byId("panels");
  panels.replaceChildren();
  // One recording fills the width; two sit side by side for the comparison.
  panels.classList.toggle("single", replays.length === 1);
  const players = replays.map((replay) => {
    const panel = document.createElement("section");
    panel.className = "panel";
    panels.appendChild(panel);
    return new Player(panel, replay);
  });

  const episodeMs = first.episodeMs;
  let playing = true;
  let clockMs = 0;
  let speed = 1;
  let last = performance.now();

  const playButton = byId<HTMLButtonElement>("play");
  const scrub = byId<HTMLInputElement>("scrub");
  const clockLabel = byId("clock");
  scrub.max = String(episodeMs);
  scrub.disabled = false;
  playButton.disabled = false;
  byId<HTMLButtonElement>("restart").disabled = false;
  for (const button of document.querySelectorAll<HTMLButtonElement>("[data-speed]")) {
    button.disabled = false;
  }

  const setPlaying = (next: boolean) => {
    playing = next;
    playButton.textContent = playing ? "Pause" : "Play";
  };
  playButton.onclick = () => setPlaying(!playing);
  byId("restart").onclick = () => {
    clockMs = 0;
    for (const player of players) player.reset();
    setPlaying(true);
  };
  for (const button of document.querySelectorAll<HTMLButtonElement>("[data-speed]")) {
    button.onclick = () => {
      speed = Number(button.dataset.speed);
      for (const other of document.querySelectorAll<HTMLButtonElement>("[data-speed]")) {
        other.setAttribute("aria-pressed", String(other === button));
      }
    };
  }
  scrub.oninput = () => {
    clockMs = Number(scrub.value);
    for (const player of players) player.seek(clockMs);
  };

  function frame(now: number): void {
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    if (playing) {
      clockMs += dt * 1000 * speed;
      if (clockMs >= episodeMs) {
        clockMs = 0;
        for (const player of players) player.reset();
      }
      scrub.value = String(Math.round(clockMs));
    }
    clockLabel.textContent = `${(clockMs / 1000).toFixed(1)} / ${(episodeMs / 1000).toFixed(0)} s`;
    for (const player of players) player.update(clockMs, dt);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

// --- live mode --------------------------------------------------------------

function startLive(url: string): void {
  const panels = byId("panels");
  panels.replaceChildren();
  panels.classList.add("single");

  const status = document.createElement("p");
  status.className = "live-status";
  status.textContent = "Connecting to the simulator...";
  panels.appendChild(status);

  // Live has no transport: the simulator decides the clock, so pausing or
  // scrubbing would be a lie. Disable them rather than leaving dead buttons.
  for (const id of ["play", "restart"]) {
    byId<HTMLButtonElement>(id).disabled = true;
  }
  for (const button of document.querySelectorAll<HTMLButtonElement>("[data-speed]")) {
    button.disabled = true;
  }
  byId<HTMLInputElement>("scrub").disabled = true;
  byId("clock").textContent = "waiting";

  let hello: LiveHello | null = null;
  let player: Player | null = null;
  let replay: Replay | null = null;
  let panel: HTMLElement | null = null;
  let clockMs = 0;
  let adopted = false;
  let last = performance.now();

  const connection = connectLive(url, {
    onStatus: (state, detail) => {
      if (state === "open" && !player) status.textContent = "Connected. Waiting for an episode...";
      if (state === "connecting") status.textContent = `Connecting to ${url}...`;
      if (state === "closed") status.textContent = `Disconnected from ${url}. Retrying...`;
      if (state === "error") status.textContent = detail ?? `Could not reach ${url}.`;
    },
    onHello: (message) => {
      hello = message;
      byId("seed").textContent = String(message.seed);
      byId("sim").textContent =
        `${message.simulator.neurons.toLocaleString("en-US")} neurons, ${message.simulator.dataset}`;
    },
    onEpisodeStart: (message) => {
      if (!hello) return;
      adopted = false;
      status.remove();
      player?.dispose();
      panel?.remove();
      panel = document.createElement("section");
      panel.className = "panel";
      panels.appendChild(panel);
      replay = liveReplay(hello, message);
      player = new Player(panel, replay);
      clockMs = 0;
    },
    onFrame: (message) => {
      if (!replay) return;
      appendFrame(replay, message);
      clockMs = message.tMs;
      // The first frame after joining carries totals for outcomes this viewer
      // never saw, so adopt them once and count normally from there.
      if (!adopted && player) {
        player.setCounts({
          caught: message.caught,
          snapped: message.snapped,
          decoysHooked: message.decoysHooked,
        });
        adopted = true;
      }
      byId("clock").textContent =
        `${(message.tMs / 1000).toFixed(1)} / ${(replay.episodeMs / 1000).toFixed(0)} s brain time` +
        ` (${message.computeMs.toFixed(0)} ms/window)`;
    },
    onOutcome: (message) => {
      replay?.outcomes.push({ tMs: message.tMs, type: message.outcome });
    },
    onEpisodeEnd: (message) => {
      byId("clock").textContent =
        `episode ${message.episode} done in ${message.wallSeconds}s wall: ` +
        `${message.summary.caught}/${message.summary.bites} caught`;
    },
  });
  window.addEventListener("beforeunload", () => connection.close());

  function frame(now: number): void {
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    // Live never runs ahead of what it has been told, so it renders the last
    // frame received and waits rather than interpolating towards a guess.
    player?.update(clockMs, dt);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

// --- mode switch ------------------------------------------------------------

function main(): void {
  const params = new URLSearchParams(location.search);
  const liveUrl = params.get("live");
  const isLive = liveUrl !== null;

  for (const button of document.querySelectorAll<HTMLButtonElement>("[data-mode]")) {
    button.setAttribute("aria-pressed", String((button.dataset.mode === "live") === isLive));
    button.onclick = () => {
      const next = new URL(location.href);
      if (button.dataset.mode === "live") next.searchParams.set("live", liveUrl || DEFAULT_LIVE_URL);
      else next.searchParams.delete("live");
      location.href = next.toString();
    };
  }

  if (isLive) startLive(liveUrl || DEFAULT_LIVE_URL);
  else void startReplay();
}

main();
