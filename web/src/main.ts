// Replay mode: play recorded episodes side by side, with no backend.
//
// Both panels run off one episode clock because both recordings were made from
// the same bite schedule. That is the comparison: the same fish, the same
// times, two policies.
//
// Live mode (a WebSocket from the Python side) is not in this build.

import "./style.css";
import { Player } from "./player";
import { assertReplay, type Replay, type ReplayIndex } from "./types";

const BASE = `${import.meta.env.BASE_URL}recordings/`;

const byId = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

async function loadJson<T>(file: string): Promise<T> {
  const response = await fetch(`${BASE}${file}`, { cache: "no-store" });
  if (!response.ok) throw new Error(`${file}: ${response.status} ${response.statusText}`);
  return (await response.json()) as T;
}

function fail(message: string, detail?: unknown): void {
  console.error(message, detail);
  byId("app").innerHTML = `
    <div class="fatal">
      <h1>No recordings to play</h1>
      <p>${message}</p>
      <p>Record a pair with:</p>
      <pre>python3 run.py record</pre>
    </div>`;
}

async function main(): Promise<void> {
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
    fail(String(error instanceof Error ? error.message : error), error);
    return;
  }
  if (!replays.length) {
    fail("The recordings index is empty.");
    return;
  }

  const first = replays[0]!;
  const episodeMs = first.episodeMs;
  byId("seed").textContent = String(index.seed);
  byId("sim").textContent =
    `${first.simulator.neurons.toLocaleString("en-US")} neurons, ` +
    `${first.simulator.dataset}`;

  const panels = byId("panels");
  const players = replays.map((replay) => {
    const panel = document.createElement("section");
    panel.className = "panel";
    panels.appendChild(panel);
    return new Player(panel, replay);
  });

  // Transport, shared by every panel.
  let playing = true;
  let clockMs = 0;
  let speed = 1;
  let last = performance.now();

  const playButton = byId<HTMLButtonElement>("play");
  const scrub = byId<HTMLInputElement>("scrub");
  const clockLabel = byId("clock");
  scrub.max = String(episodeMs);

  const setPlaying = (next: boolean) => {
    playing = next;
    playButton.textContent = playing ? "Pause" : "Play";
  };
  playButton.addEventListener("click", () => setPlaying(!playing));
  byId("restart").addEventListener("click", () => {
    clockMs = 0;
    for (const player of players) player.reset();
    setPlaying(true);
  });
  for (const button of document.querySelectorAll<HTMLButtonElement>("[data-speed]")) {
    button.addEventListener("click", () => {
      speed = Number(button.dataset.speed);
      for (const other of document.querySelectorAll<HTMLButtonElement>("[data-speed]")) {
        other.setAttribute("aria-pressed", String(other === button));
      }
    });
  }
  scrub.addEventListener("input", () => {
    clockMs = Number(scrub.value);
    for (const player of players) player.seek(clockMs);
  });

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

void main();
