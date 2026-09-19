// One panel: a pond, a recording, and the HUD over it.
//
// The recording is sampled on its own 50 ms decision grid; the panel renders at
// whatever rate the browser gives it and interpolates the bobber between
// windows, so a 20 Hz recording plays smoothly at 60 fps.

import { recordStream, supportedMimeType } from "./capture";
import { PondScene } from "./scene";
import type { ReplayStage } from "./types";

/**
 * Where the boat is and whether the fire is lit, for a moment in a voyage.
 *
 * Entirely staging: it says where to draw things, and nothing here is read back
 * into the simulation. `progress` is how far through its own stage the clock is.
 */
function staging(stage: ReplayStage["id"], progress: number) {
  const ramp = (t: number) => Math.min(1, Math.max(0, t));
  switch (stage) {
    case "prep":
      return { out: 0, heat: 0, bobberVisible: false };
    case "fish":
      // Row out over the first fifth of the stage, then sit and fish.
      return {
        out: ramp(progress / 0.2),
        heat: 0,
        bobberVisible: progress > 0.15,
        rowing: progress < 0.2 ? 1 : 0,
      };
    case "row":
      return { out: 1 - ramp(progress), heat: 0, bobberVisible: false, rowing: 1 };
    case "cook":
      return { out: 0, heat: ramp(progress / 0.25), bobberVisible: false, atFire: true };
    case "eat":
      return { out: 0, heat: 1, bobberVisible: false, atFire: true };
    default:
      return { out: 1, heat: 0, bobberVisible: true };
  }
}
import type { Replay } from "./types";

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

interface Hud {
  root: HTMLElement;
  tools: HTMLElement;
  caught: HTMLElement;
  snapped: HTMLElement;
  timer: HTMLElement;
  escape: HTMLElement;
  escapeBar: HTMLElement;
  hook: HTMLElement;
  hookBar: HTMLElement;
  decoys: HTMLElement;
  body: HTMLElement;
  stage: HTMLElement;
  flash: HTMLElement;
}

function buildHud(container: HTMLElement, replay: Replay): Hud {
  const root = document.createElement("div");
  root.className = "hud";
  root.innerHTML = `
    <div class="hud-head">
      <strong>${replay.label}</strong>
      <span>${replay.provenance.trained}</span>
      <span data-body class="body-note">loading body</span>
    </div>
    <div class="hud-stage" data-stage></div>
    <div class="hud-counts">
      <div class="count"><b data-caught>0</b><span>caught</span></div>
      <div class="count snap"><b data-snapped>0</b><span>snapped</span></div>
      <div class="count decoy"><b data-decoys>0</b><span>decoys hooked</span></div>
      <div class="count"><b data-timer>0.0s</b><span>episode</span></div>
    </div>
    <div class="hud-meters">
      <label><span>escape_giant_fiber</span><output data-escape>0 Hz</output></label>
      <div class="meter"><span data-escape-bar></span></div>
      <label><span>P(hook)</span><output data-hook>-</output></label>
      <div class="meter hook"><span data-hook-bar></span></div>
    </div>`;
  const tools = document.createElement("div");
  tools.className = "panel-tools";
  tools.innerHTML = `
    <button type="button" class="btn tiny" data-camera="orbit" aria-pressed="true">Orbit</button>
    <button type="button" class="btn tiny" data-camera="closeup" aria-pressed="false">Close-up</button>
    <button type="button" class="btn tiny" data-record>Record 30s</button>`;

  const flash = document.createElement("div");
  flash.className = "snap-flash";
  container.append(root, tools, flash);

  const pick = (name: string) => root.querySelector(`[data-${name}]`) as HTMLElement;
  return {
    root,
    tools,
    caught: pick("caught"),
    snapped: pick("snapped"),
    timer: pick("timer"),
    escape: pick("escape"),
    escapeBar: pick("escape-bar"),
    hook: pick("hook"),
    hookBar: pick("hook-bar"),
    decoys: pick("decoys"),
    body: pick("body"),
    stage: pick("stage"),
    flash,
  };
}

export class Player {
  private readonly pond: PondScene;
  private readonly hud: Hud;
  /** Outcomes already played, so a restart replays them and a seek does not double-fire. */
  private nextOutcome = 0;
  private caught = 0;
  private snapped = 0;
  private decoysHooked = 0;
  private flashLife = 0;
  private elapsedSeconds = 0;
  private recording: { stop(): void; done: Promise<void> } | null = null;

  constructor(
    private readonly container: HTMLElement,
    private readonly replay: Replay,
  ) {
    const stage = document.createElement("div");
    stage.className = "stage";
    container.appendChild(stage);
    this.pond = new PondScene(stage);
    this.hud = buildHud(container, replay);
    this.paint(0);
    // The real body is generated locally and gitignored, so its absence is
    // normal rather than an error; the badge says which one is on screen.
    void this.pond.attachBody(`${import.meta.env.BASE_URL}fly.glb`).then((real) => {
      this.hud.body.textContent = real
        ? "NeuroMechFly body, neutral pose"
        : "placeholder body (run: python3 tools/build_fly_glb.py)";
    });
    this.wireTools();
  }

  reset(): void {
    this.nextOutcome = 0;
    this.caught = 0;
    this.snapped = 0;
    this.decoysHooked = 0;
    this.flashLife = 0;
    this.paint(0);
  }

  /**
   * @param episodeMs where the shared transport is in episode time
   * @param dt        wall seconds since the last frame
   */
  update(episodeMs: number, dt: number): void {
    this.elapsedSeconds += dt;

    const last = Math.max(0, this.frameCount - 1);
    const exact = episodeMs / this.replay.windowMs;
    const index = Math.min(last, Math.max(0, Math.floor(exact)));
    const next = Math.min(last, index + 1);
    const t = exact - Math.floor(exact);
    const dip = lerp(this.replay.frames.bobber[index] ?? 0, this.replay.frames.bobber[next] ?? 0, t);

    // Fire every outcome the clock has passed. Walking a cursor rather than
    // testing a window keeps a dropped frame from swallowing a catch.
    while (
      this.nextOutcome < this.replay.outcomes.length &&
      this.replay.outcomes[this.nextOutcome]!.tMs <= episodeMs
    ) {
      const outcome = this.replay.outcomes[this.nextOutcome]!;
      if (outcome.type === "catch") {
        this.caught++;
        this.pond.splash();
      } else {
        this.snapped++;
        if (outcome.onDecoy) this.decoysHooked++;
        this.flashLife = 1;
      }
      this.nextOutcome++;
    }

    if (this.flashLife > 0) {
      this.flashLife = Math.max(0, this.flashLife - dt * 2.6);
      this.hud.flash.style.opacity = String(this.flashLife * 0.55);
    }

    this.applyStaging(episodeMs);
    this.pond.update(this.elapsedSeconds, dt, dip, this.elapsedSeconds * 7.5);
    this.paint(index);
  }

  private wireTools(): void {
    const cameras = this.hud.tools.querySelectorAll<HTMLButtonElement>("[data-camera]");
    for (const button of cameras) {
      button.addEventListener("click", () => {
        const mode = button.dataset.camera as "orbit" | "closeup";
        this.pond.setCameraMode(mode);
        for (const other of cameras) {
          other.setAttribute("aria-pressed", String(other === button));
        }
      });
    }

    const record = this.hud.tools.querySelector<HTMLButtonElement>("[data-record]")!;
    if (!supportedMimeType()) {
      record.disabled = true;
      record.textContent = "No webm here";
      record.title = "This browser has no MediaRecorder webm encoder";
      return;
    }
    record.addEventListener("click", () => {
      if (this.recording) {
        this.recording.stop();
        return;
      }
      const started = recordStream({
        stream: this.pond.captureStream(60),
        filename: `fly-fishing-${this.replay.policy}-${this.replay.seed}.webm`,
        onTick: (left) => {
          record.textContent = left > 0 ? `Stop (${left}s)` : "Record 30s";
        },
      });
      if (!started) {
        record.disabled = true;
        record.textContent = "No webm here";
        return;
      }
      this.recording = started;
      record.setAttribute("aria-pressed", "true");
      void started.done.then(() => {
        this.recording = null;
        record.textContent = "Record 30s";
        record.setAttribute("aria-pressed", "false");
      });
    });
  }

  /**
   * Place the boat and the fire for wherever the voyage clock is.
   *
   * A single-stage fishing recording carries no stages, and then the boat just
   * sits on the fishing ground with the line out, which is what that task is.
   */
  private applyStaging(episodeMs: number): void {
    const stages = this.replay.stages;
    if (!stages?.length) {
      this.pond.setVoyage({ out: 1, heat: 0, bobberVisible: true });
      this.hud.stage.textContent = "";
      return;
    }
    let current = stages[0]!;
    for (const stage of stages) if (episodeMs >= stage.startMs) current = stage;
    const progress = Math.min(1, (episodeMs - current.startMs) / current.durationMs);
    this.pond.setVoyage(staging(current.id, progress));
    this.hud.stage.textContent = current.name;
    this.hud.stage.classList.toggle("transit", !current.decision);
  }

  /**
   * Adopt the simulator's running totals.
   *
   * Live mode needs this because a viewer that connects part-way through an
   * episode never saw the outcomes that came before it, and counting only what
   * it witnessed would under-report. Every live frame carries the authoritative
   * totals, so they are taken rather than recomputed. Replay has no use for it:
   * there, the outcome list is complete from the start.
   */
  setCounts(counts: { caught: number; snapped: number; decoysHooked: number }): void {
    this.caught = counts.caught;
    this.snapped = counts.snapped;
    this.decoysHooked = counts.decoysHooked;
    // Outcomes already accounted for must not fire their splash a second time.
    this.nextOutcome = this.replay.outcomes.length;
  }

  /**
   * How many frames exist right now.
   *
   * Read rather than cached: a live episode's columns start empty and grow as
   * the simulator produces windows, so a count taken at construction would be
   * zero and every index would clamp to -1.
   */
  private get frameCount(): number {
    return this.replay.frames.bobber.length;
  }

  /** Rewind the outcome cursor, for a restart or a scrub backwards. */
  seek(episodeMs: number): void {
    this.nextOutcome = 0;
    this.caught = 0;
    this.snapped = 0;
    this.decoysHooked = 0;
    for (const outcome of this.replay.outcomes) {
      if (outcome.tMs > episodeMs) break;
      if (outcome.type === "catch") {
        this.caught++;
      } else {
        this.snapped++;
        if (outcome.onDecoy) this.decoysHooked++;
      }
      this.nextOutcome++;
    }
    this.paint(
      Math.min(Math.max(0, this.frameCount - 1), Math.floor(episodeMs / this.replay.windowMs)),
    );
  }

  private paint(index: number): void {
    const escapeHz = this.replay.frames.escapeHz[index] ?? 0;
    const pHook = this.replay.frames.pHook[index];
    this.hud.caught.textContent = String(this.caught);
    this.hud.snapped.textContent = String(this.snapped);
    this.hud.decoys.textContent = `${this.decoysHooked}/${this.replay.summary.decoys}`;
    this.hud.timer.textContent = `${((index * this.replay.windowMs) / 1000).toFixed(1)}s`;
    this.hud.escape.textContent = `${escapeHz.toFixed(0)} Hz`;
    this.hud.escapeBar.style.width = `${Math.min(100, (escapeHz / 220) * 100).toFixed(1)}%`;
    this.hud.hook.textContent = pHook === null || pHook === undefined ? "-" : pHook.toFixed(2);
    this.hud.hookBar.style.width = `${((pHook ?? 0) * 100).toFixed(1)}%`;
  }

  dispose(): void {
    this.recording?.stop();
    this.pond.dispose();
    this.container.replaceChildren();
  }
}
