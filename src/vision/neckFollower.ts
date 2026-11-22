/**
 * Main-thread side of neck tracking: captures small video frames at about
 * 15 per second and sends them to src/vision/neckWorker.ts, one in flight.
 * The worker owns all computation; this class owns the worker, the timer and
 * every bitmap it creates.
 */
import type { Quadrilateral } from "./geometry";
import { MIN_REFERENCE_POINTS, type Box, type NeckResult } from "./neckTracker";

/** Width of the greyscale frame the tracker analyses. */
export const NECK_ANALYSIS_WIDTH = 480;
export const NECK_INTERVAL_MS = 66;

export type NeckRequest =
  | {
      type: "reference";
      bitmap: ImageBitmap;
      width: number;
      corners: Quadrilateral;
      exclude: Box[];
    }
  | { type: "frame"; bitmap: ImageBitmap; width: number; exclude: Box[] };
export type NeckReply =
  | { type: "reference"; points: number; ms: number }
  | { type: "result"; result: NeckResult; ms: number }
  | { type: "error"; message: string };

export interface NeckUpdate extends NeckResult {
  /** Main-thread capture time, before bitmap creation and worker processing. */
  capturedAt: number;
  /** Worker time for this frame. */
  ms: number;
}

/** A pose is evidence only while it describes a recent, successfully tracked frame. */
export function neckTrackingUsable(
  update: NeckUpdate | null,
  now: number,
): boolean {
  return (
    !!update &&
    update.status === "tracking" &&
    !!update.corners &&
    now >= update.capturedAt &&
    now - update.capturedAt <= 500
  );
}

export class NeckFollower {
  private worker: Worker | null = null;
  private timer: ReturnType<typeof setInterval> | undefined;
  private busy = false;
  private capturedAt = 0;
  private stopped = false;
  private referenced = false;
  private pending: { corners: Quadrilateral; exclude: Box[] } | null = null;
  /** Last reference sent, resent until a frame is good enough to start from. */
  private requested: { corners: Quadrilateral; exclude: Box[] } | null = null;
  private exclude: Box[] = [];

  constructor(
    private video: HTMLVideoElement,
    private onUpdate: (update: NeckUpdate) => void,
    private onError: (message: string) => void,
  ) {}

  start() {
    try {
      this.worker = new Worker(new URL("./neckWorker.ts", import.meta.url), {
        type: "module",
      });
    } catch (error) {
      this.onError(error instanceof Error ? error.message : String(error));
      return;
    }
    this.worker.onmessage = ({ data }: MessageEvent<NeckReply>) => {
      this.busy = false;
      if (this.stopped) return;
      // Too dark or featureless (camera still exposing): try the next frame.
      if (data.type === "reference" && data.points < MIN_REFERENCE_POINTS) {
        this.referenced = false;
        this.pending ??= this.requested;
      }
      if (data.type === "error") this.onError(data.message);
      else if (data.type === "result")
        this.onUpdate({
          ...data.result,
          capturedAt: this.capturedAt,
          ms: data.ms,
        });
    };
    this.worker.onerror = (event) => {
      this.busy = false;
      this.onError(event.message || "Neck tracking stopped.");
    };
    this.timer = setInterval(() => void this.capture(), NECK_INTERVAL_MS);
  }

  /** Follow from the next frame, where `corners` are known to be right. */
  setReference(corners: Quadrilateral, exclude: Box[]) {
    this.pending = { corners, exclude };
  }

  /** Image areas to ignore (the hands), normalised. */
  setExclude(exclude: Box[]) {
    this.exclude = exclude;
  }

  stop() {
    this.stopped = true;
    clearInterval(this.timer);
    this.worker?.terminate();
    this.worker = null;
  }

  private async capture() {
    const video = this.video;
    if (this.busy || !this.worker || this.stopped) return;
    if (!this.pending && !this.referenced) return;
    // Like the hand model, pause while the window is hidden: nothing is
    // shown, and a lost neck would keep searching (tens of ms per frame).
    if (typeof document !== "undefined" && document.hidden) return;
    if (video.readyState < 2 || !video.videoWidth || !video.videoHeight) return;
    this.busy = true;
    const capturedAt = performance.now();
    const width = NECK_ANALYSIS_WIDTH;
    const height = Math.max(
      16,
      Math.round((width * video.videoHeight) / video.videoWidth),
    );
    let bitmap: ImageBitmap;
    try {
      // Resize options may be ignored (older WebKit); the worker scales anyway.
      bitmap = await createImageBitmap(video, {
        resizeWidth: width,
        resizeHeight: height,
        resizeQuality: "low",
      });
    } catch {
      this.busy = false;
      return;
    }
    if (this.stopped || !this.worker) {
      bitmap.close();
      return;
    }
    const reference = this.pending;
    this.pending = null;
    const request: NeckRequest = reference
      ? { type: "reference", bitmap, width, ...reference }
      : { type: "frame", bitmap, width, exclude: this.exclude };
    if (reference) {
      this.referenced = true;
      this.requested = { ...reference, exclude: this.exclude };
    }
    this.capturedAt = capturedAt;
    this.worker.postMessage(request, [bitmap]);
  }
}
