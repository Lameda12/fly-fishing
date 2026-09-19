// The one-click webm export.
//
// MediaRecorder over the canvas's own captureStream, so what lands in the file
// is exactly what was on screen: no re-render, no second code path that could
// disagree with the live one.
//
// Browser support is uneven, and this is a nice-to-have rather than something
// the viewer depends on, so every entry point here degrades to "not available"
// instead of throwing into the page.

/** Codecs to try, best first. Safari has neither, which is handled. */
const CANDIDATES = [
  "video/webm;codecs=vp9",
  "video/webm;codecs=vp8",
  "video/webm",
];

export function supportedMimeType(): string | null {
  if (typeof MediaRecorder === "undefined") return null;
  for (const type of CANDIDATES) {
    try {
      if (MediaRecorder.isTypeSupported(type)) return type;
    } catch {
      // isTypeSupported itself can throw on some engines; try the next one.
    }
  }
  return null;
}

export interface Recording {
  /** Resolves when the recording has stopped and the file has been offered. */
  done: Promise<void>;
  /** Stop early. Safe to call after it has already stopped. */
  stop(): void;
}

/**
 * Record `stream` for `durationMs` and hand the viewer the file.
 *
 * `onTick` gets the seconds remaining, once a second, so a button can count
 * down without this module knowing anything about the DOM.
 */
export function recordStream({
  stream,
  filename,
  durationMs = 30_000,
  onTick,
}: {
  stream: MediaStream;
  filename: string;
  durationMs?: number;
  onTick?: (secondsLeft: number) => void;
}): Recording | null {
  const mimeType = supportedMimeType();
  if (!mimeType) return null;

  let recorder: MediaRecorder;
  try {
    recorder = new MediaRecorder(stream, { mimeType, videoBitsPerSecond: 6_000_000 });
  } catch {
    return null;
  }

  const chunks: Blob[] = [];
  recorder.ondataavailable = (event) => {
    if (event.data.size) chunks.push(event.data);
  };

  let ticker = 0;
  let stopTimer = 0;
  const clear = () => {
    clearInterval(ticker);
    clearTimeout(stopTimer);
  };

  const done = new Promise<void>((resolve) => {
    recorder.onstop = () => {
      clear();
      onTick?.(0);
      if (chunks.length) save(new Blob(chunks, { type: mimeType }), filename);
      resolve();
    };
    recorder.onerror = () => {
      clear();
      onTick?.(0);
      resolve();
    };
  });

  recorder.start(250);
  const startedAt = performance.now();
  onTick?.(Math.ceil(durationMs / 1000));
  ticker = window.setInterval(() => {
    const left = Math.ceil((durationMs - (performance.now() - startedAt)) / 1000);
    onTick?.(Math.max(0, left));
  }, 250);
  stopTimer = window.setTimeout(() => {
    if (recorder.state !== "inactive") recorder.stop();
  }, durationMs);

  return {
    done,
    stop() {
      if (recorder.state !== "inactive") recorder.stop();
    },
  };
}

function save(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  // Revoking immediately can cancel the download in some browsers.
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
}
