// The replay file schema, version 2.
//
// Written by fishing/record.mjs, read here. The README documents it; this file
// is the machine-readable half of that documentation. A recording is
// self-contained so the viewer needs no backend to play one.

export interface ReplayEvent {
  /** Milliseconds of episode time at which something took the bait. */
  tMs: number;
  /** A real fish, or a decoy nibble that dips the bobber the same way. */
  type: "bite" | "decoy";
}

export interface ReplayOutcome {
  tMs: number;
  type: "catch" | "snap";
  /** On a snap: whether a decoy was what the policy fell for. */
  onDecoy?: boolean;
}

export interface ReplaySummary {
  bites: number;
  decoys: number;
  decoysHooked: number;
  caught: number;
  snapped: number;
  missed: number;
  totalReward: number;
  catchRate: number;
  decoyHookRate: number;
  falseHooksPerMinute: number;
}

export interface ReplayFrames {
  /** Raw escape_giant_fiber rate, Hz, one entry per decision window. Simulated. */
  escapeHz: number[];
  /** Raw walk_dnp09 rate, Hz. Simulated. */
  walkHz: number[];
  /** The readout's P(hook), or null for a policy that has none. */
  pHook: (number | null)[];
  /** Bobber dip, 0 floating to 1 pulled under. Scripted. */
  bobber: number[];
}

/** One stage of a voyage, as the recorder writes it. */
export interface ReplayStage {
  id: "prep" | "fish" | "row" | "cook" | "eat";
  name: string;
  startMs: number;
  durationMs: number;
  /** False for the row back, which is transit and scores nothing. */
  decision: boolean;
}

export interface Replay {
  schemaVersion: 2;
  kind: "fly-fishing-replay";
  generatedAt: string;
  policy: string;
  label: string;
  seed: number;
  /** Milliseconds per recorded frame. One frame is one decision window. */
  windowMs: number;
  episodeMs: number;
  task: {
    hookWindowMs: number;
    recastMs: number;
    biteLoomHz: number;
    decoyLoomHz: number;
    rewardCatch: number;
    rewardSnap: number;
  };
  simulator: { source: string; dataset: string; neurons: number; edges: number };
  summary: ReplaySummary;
  events: ReplayEvent[];
  outcomes: ReplayOutcome[];
  frames: ReplayFrames;
  /** Present on a voyage recording; absent on a single-stage fishing one. */
  stages?: ReplayStage[];
  provenance: {
    simulated: string[];
    scripted: string[];
    frozen: string;
    trained: string;
    disclaimer: string;
  };
}

export interface ReplayIndex {
  schemaVersion: 2;
  generatedAt: string;
  seed: number;
  recordings: { file: string; label: string; policy: string; summary: ReplaySummary }[];
}

export function assertReplay(value: unknown, source: string): Replay {
  const replay = value as Replay;
  if (!replay || replay.kind !== "fly-fishing-replay") {
    throw new Error(`${source} is not a fly-fishing replay`);
  }
  if (replay.schemaVersion !== 2) {
    throw new Error(`${source} is schema ${replay.schemaVersion}, this viewer reads 2`);
  }
  const { escapeHz, bobber } = replay.frames;
  if (escapeHz.length !== bobber.length) {
    throw new Error(`${source} has ragged frame columns`);
  }
  return replay;
}
