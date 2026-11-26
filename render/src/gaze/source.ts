/**
 * Gaze sources for the renderer: a WebSocket client for the GZFV stream (fed by
 * scripts/relay.ts from the Python tracker's UDP output) and a pointer fallback for desktop demos.
 */
import { FLAG_VALID, decodeGaze } from './protocol';
import type { GazePacket } from './protocol';

export interface GazeSample {
  /** NDC on the view, x right / y up. */
  ndc: [number, number];
  valid: boolean;
  saccade: boolean;
  errorDeg: number;
  /** Receive time (performance.now(), ms). */
  tMs: number;
  packet?: GazePacket;
}

export interface GazeSource {
  latest(): GazeSample | null;
  close(): void;
}

export class WebSocketGazeSource implements GazeSource {
  private ws: WebSocket;
  private last: GazeSample | null = null;
  dropped = 0;

  constructor(url = 'ws://127.0.0.1:8766') {
    this.ws = new WebSocket(url);
    this.ws.binaryType = 'arraybuffer';
    this.ws.onmessage = (ev) => {
      try {
        const p = decodeGaze(ev.data as ArrayBuffer);
        this.last = { ndc: p.ndc, valid: (p.flags & FLAG_VALID) !== 0, saccade: (p.flags & (1 << 4)) !== 0, errorDeg: p.errorDeg, tMs: performance.now(), packet: p };
      } catch {
        this.dropped++;
      }
    };
  }

  latest(): GazeSample | null {
    return this.last;
  }

  close(): void {
    this.ws.close();
  }
}

/** Mouse/pointer as gaze, for desktop demos without a tracker. */
export class PointerGazeSource implements GazeSource {
  private last: GazeSample | null = null;
  private readonly onMove: (e: PointerEvent) => void;

  constructor(private readonly el: HTMLElement) {
    this.onMove = (e) => {
      const r = el.getBoundingClientRect();
      const x = ((e.clientX - r.left) / r.width) * 2 - 1;
      const y = 1 - ((e.clientY - r.top) / r.height) * 2;
      this.last = { ndc: [x, y], valid: true, saccade: false, errorDeg: 0, tMs: performance.now() };
    };
    el.addEventListener('pointermove', this.onMove);
  }

  latest(): GazeSample | null {
    return this.last;
  }

  close(): void {
    this.el.removeEventListener('pointermove', this.onMove);
  }
}
