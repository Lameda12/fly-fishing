// One panel: a pond, a recording, and the HUD over it.
//
// The recording is sampled on its own 50 ms decision grid; the panel renders at
// whatever rate the browser gives it and interpolates the bobber between
// windows, so a 20 Hz recording plays smoothly at 60 fps.

import { PondScene } from "./scene";
import type { Replay } from "./types";

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

interface Hud {
  root: HTMLElement;
  caught: HTMLElement;
  snapped: HTMLElement;
  timer: HTMLElement;
  escape: HTMLElement;
  escapeBar: HTMLElement;
  hook: HTMLElement;
  hookBar: HTMLElement;
  decoys: HTMLElement;
  body: HTMLElement;
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
  const flash = document.createElement("div");
  flash.className = "snap-flash";
  container.append(root, flash);

  const pick = (name: string) => root.querySelector(`[data-${name}]`) as HTMLElement;
  return {
    root,
    caught: pick("caught"),
    snapped: pick("snapped"),
    timer: pick("timer"),
    escape: pick("escape"),
    escapeBar: pick("escape-bar"),
    hook: pick("hook"),
    hookBar: pick("hook-bar"),
    decoys: pick("decoys"),
    body: pick("body"),
    flash,
  };
}

export class Player {
  private readonly pond: PondScene;
  private readonly hud: Hud;
  private readonly frames: number;
  /** Outcomes already played, so a restart replays them and a seek does not double-fire. */
  private nextOutcome = 0;
  private caught = 0;
  private snapped = 0;
  private decoysHooked = 0;
  private flashLife = 0;
  private elapsedSeconds = 0;

  constructor(
    private readonly container: HTMLElement,
    private readonly replay: Replay,
  ) {
    const stage = document.createElement("div");
    stage.className = "stage";
    container.appendChild(stage);
    this.pond = new PondScene(stage);
    this.hud = buildHud(container, replay);
    this.frames = replay.frames.bobber.length;
    this.paint(0);
    // The real body is generated locally and gitignored, so its absence is
    // normal rather than an error; the badge says which one is on screen.
    void this.pond.attachBody(`${import.meta.env.BASE_URL}fly.glb`).then((real) => {
      this.hud.body.textContent = real
        ? "NeuroMechFly body, neutral pose"
        : "placeholder body (run: python3 tools/build_fly_glb.py)";
    });
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

    const exact = episodeMs / this.replay.windowMs;
    const index = Math.min(this.frames - 1, Math.max(0, Math.floor(exact)));
    const next = Math.min(this.frames - 1, index + 1);
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

    this.pond.update(this.elapsedSeconds, dt, dip, this.elapsedSeconds * 7.5);
    this.paint(index);
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
    this.paint(Math.min(this.frames - 1, Math.floor(episodeMs / this.replay.windowMs)));
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
    this.pond.dispose();
    this.container.replaceChildren();
  }
}
