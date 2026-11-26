/** Minimal RFC 6455 helpers for the relay: handshake accept key and unmasked server frames. */
import { createHash } from 'node:crypto';

const GUID = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11';

export function acceptKey(key: string): string {
  return createHash('sha1').update(key + GUID).digest('base64');
}

export function encodeFrame(payload: Uint8Array, opcode = 0x2): Uint8Array {
  const n = payload.length;
  const head = n < 126 ? 2 : n < 65536 ? 4 : 10;
  const out = new Uint8Array(head + n);
  out[0] = 0x80 | opcode; // FIN + opcode
  if (n < 126) out[1] = n;
  else if (n < 65536) {
    out[1] = 126;
    out[2] = n >> 8;
    out[3] = n & 0xff;
  } else {
    out[1] = 127;
    const dv = new DataView(out.buffer);
    dv.setBigUint64(2, BigInt(n));
  }
  out.set(payload, head);
  return out;
}
