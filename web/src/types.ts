// The replay file schema, version 1.
//
// Written by fishing/record.mjs, read here. The README documents it; this file
// is the machine-readable half of that documentation. A recording is
// self-contained so the viewer needs no backend to play one.

export interface ReplayEvent {
  /** Milliseconds of episode time at which the fish took the bait. */
  tMs: number;
  type: "bite";
}

export interface ReplayOutcome {
  tMs: number;
  type: "catch" | "snap";
}

export interface ReplaySummary {
  bites: number;
  caught: number;
  snapped: number;
  missed: number;
  totalReward: number;
  catchRate: number;
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

export interface Replay {
  schemaVersion: 1;
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
    rewardCatch: number;
    rewardSnap: number;
  };
  simulator: { source: string; dataset: string; neurons: number; edges: number };
  summary: ReplaySummary;
  events: ReplayEvent[];
  outcomes: ReplayOutcome[];
  frames: ReplayFrames;
  provenance: {
    simulated: string[];
    scripted: string[];
    frozen: string;
    trained: string;
    disclaimer: string;
  };
}

export interface ReplayIndex {
  schemaVersion: 1;
  generatedAt: string;
  seed: number;
  recordings: { file: string; label: string; policy: string; summary: ReplaySummary }[];
}

export function assertReplay(value: unknown, source: string): Replay {
  const replay = value as Replay;
  if (!replay || replay.kind !== "fly-fishing-replay") {
    throw new Error(`${source} is not a fly-fishing replay`);
  }
  if (replay.schemaVersion !== 1) {
    throw new Error(`${source} is schema ${replay.schemaVersion}, this viewer reads 1`);
  }
  const { escapeHz, bobber } = replay.frames;
  if (escapeHz.length !== bobber.length) {
    throw new Error(`${source} has ragged frame columns`);
  }
  return replay;
}
