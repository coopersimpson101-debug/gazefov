/**
 * UDP -> WebSocket relay for the GZFV gaze stream (browsers cannot receive UDP).
 *
 *   npm run relay -- [--udp 5566] [--ws 8766] [--host 127.0.0.1]
 *
 * Every valid 92-byte GZFV datagram is forwarded unchanged as one binary WebSocket message to
 * every connected client. Invalid datagrams (size/CRC/magic) are dropped. Minimal RFC 6455
 * server: handshake + unmasked server->client binary frames; client frames other than close
 * and ping are ignored. Binds to localhost by default.
 */
import { createSocket } from 'node:dgram';
import { createServer } from 'node:http';
import type { Duplex } from 'node:stream';
import { decodeGaze } from '../src/gaze/protocol';
import { acceptKey, encodeFrame } from './wsframe';

const args = process.argv.slice(2);
const opt = (n: string, d: string) => {
  const i = args.indexOf(`--${n}`);
  return i >= 0 && i + 1 < args.length ? args[i + 1] : d;
};
const host = opt('host', '127.0.0.1');
const udpPort = parseInt(opt('udp', '5566'), 10);
const wsPort = parseInt(opt('ws', '8766'), 10);

const clients = new Set<Duplex>();
let forwarded = 0;
let dropped = 0;

const server = createServer((_req, res) => {
  res.writeHead(426, { 'Content-Type': 'text/plain' });
  res.end('WebSocket only\n');
});
server.on('upgrade', (req, socket) => {
  const key = req.headers['sec-websocket-key'];
  if (typeof key !== 'string' || (req.headers.upgrade ?? '').toLowerCase() !== 'websocket') {
    socket.destroy();
    return;
  }
  socket.write(
    'HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n' + `Sec-WebSocket-Accept: ${acceptKey(key)}\r\n\r\n`,
  );
  clients.add(socket);
  socket.on('data', (buf: Buffer) => {
    const op = buf[0] & 0x0f;
    if (op === 0x8) {
      socket.end(encodeFrame(new Uint8Array(0), 0x8));
      clients.delete(socket);
    } else if (op === 0x9) socket.write(encodeFrame(new Uint8Array(0), 0xa));
  });
  socket.on('close', () => clients.delete(socket));
  socket.on('error', () => clients.delete(socket));
});

const udp = createSocket('udp4');
udp.on('message', (msg) => {
  try {
    decodeGaze(new Uint8Array(msg));
  } catch {
    dropped++;
    return;
  }
  const frame = encodeFrame(new Uint8Array(msg), 0x2);
  for (const c of clients) c.write(frame);
  forwarded++;
});
udp.bind(udpPort, host, () => console.log(`GZFV relay: udp://${host}:${udpPort} -> ws://${host}:${wsPort}`));
server.listen(wsPort, host);
setInterval(() => console.log(`forwarded ${forwarded}, dropped ${dropped}, clients ${clients.size}`), 5000).unref();
