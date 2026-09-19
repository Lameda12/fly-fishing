// Live mode: the same pond, fed by a WebSocket instead of a recording.
//
// The stream's schema is documented in the README and mirrored in the types
// below. fishing/live.mjs is the only thing that writes it.
//
// The important difference from replay is the clock. Replay plays 60 s of brain
// time in 60 s of wall time. Live runs the real network, which takes about eight
// minutes of wall time to produce those same 60 s, so this mode never
// interpolates towards a future it has not been told about: it renders the last
// frame it received and waits. What the viewer shows is the fly's clock.

export interface LiveHello {
  type: "hello";
  schemaVersion: 1;
  kind: "fly-fishing-live";
  policy: string;
  seed: number;
  windowMs: number;
  episodeMs: number;
  ablation: string;
  task: {
    hookWindowMs: number;
    recastMs: number;
    biteLoomHz: number;
    decoyLoomHz: number;
    rewardCatch: number;
    rewardSnap: number;
  };
  simulator: { source: string; dataset: string; neurons: number; edges: number };
  provenance: { frozen: string; trained: string; note: string };
}

export interface LiveEpisodeStart {
  type: "episode-start";
  episode: number;
  seed: number;
  events: { tMs: number; type: "bite" | "decoy" }[];
}

export interface LiveFrame {
  type: "frame";
  tMs: number;
  escapeHz: number;
  walkHz: number;
  pHook: number | null;
  bobber: number;
  outcome: "wait" | "catch" | "snap" | "recast";
  caught: number;
  snapped: number;
  decoysHooked: number;
  /** Wall milliseconds the network took for this window. Says how slow live is. */
  computeMs: number;
}

export interface LiveOutcome {
  type: "outcome";
  tMs: number;
  outcome: "catch" | "snap";
}

export interface LiveEpisodeEnd {
  type: "episode-end";
  episode: number;
  wallSeconds: number;
  summary: {
    bites: number;
    decoys: number;
    caught: number;
    snapped: number;
    decoysHooked: number;
    totalReward: number;
  };
}

export interface LiveEvent {
  type: "event";
  tMs: number;
  event: "bite" | "decoy";
}

export type LiveMessage =
  | LiveHello
  | LiveEpisodeStart
  | LiveFrame
  | LiveOutcome
  | LiveEpisodeEnd
  | LiveEvent;

export interface LiveHandlers {
  onHello?(message: LiveHello): void;
  onEpisodeStart?(message: LiveEpisodeStart): void;
  onFrame?(message: LiveFrame): void;
  onOutcome?(message: LiveOutcome): void;
  onEpisodeEnd?(message: LiveEpisodeEnd): void;
  onStatus?(status: "connecting" | "open" | "closed" | "error", detail?: string): void;
}

/**
 * Connect, and keep trying. The simulator takes half a minute to load its
 * connectome before it starts listening, so a viewer opened first should wait
 * rather than fail; and an episode that ends is not a reason to give up either.
 */
export function connectLive(url: string, handlers: LiveHandlers): { close(): void } {
  let socket: WebSocket | null = null;
  let retry = 0;
  let closed = false;
  let timer = 0;

  const open = () => {
    if (closed) return;
    handlers.onStatus?.("connecting");
    socket = new WebSocket(url);

    socket.onopen = () => {
      retry = 0;
      handlers.onStatus?.("open");
    };
    socket.onmessage = (event) => {
      let message: LiveMessage;
      try {
        message = JSON.parse(String(event.data)) as LiveMessage;
      } catch {
        return; // a frame this viewer cannot read is not worth tearing down for
      }
      switch (message.type) {
        case "hello":
          handlers.onHello?.(message);
          break;
        case "episode-start":
          handlers.onEpisodeStart?.(message);
          break;
        case "frame":
          handlers.onFrame?.(message);
          break;
        case "outcome":
          handlers.onOutcome?.(message);
          break;
        case "episode-end":
          handlers.onEpisodeEnd?.(message);
          break;
        default:
          break; // "event" is carried by episode-start too; nothing to do here
      }
    };
    socket.onerror = () => handlers.onStatus?.("error");
    socket.onclose = () => {
      socket = null;
      if (closed) return;
      handlers.onStatus?.("closed");
      // Back off to two seconds, then stay there: this is a localhost socket,
      // and a viewer left open overnight should still reconnect promptly.
      retry = Math.min(retry + 1, 4);
      timer = window.setTimeout(open, 250 * 2 ** retry);
    };
  };

  open();
  return {
    close() {
      closed = true;
      clearTimeout(timer);
      socket?.close();
    },
  };
}
