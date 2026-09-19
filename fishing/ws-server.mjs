// A minimal WebSocket server, in the standard library.
//
// The rest of this repository has no runtime dependencies, and live mode needs
// one socket that pushes text frames at a browser on localhost. That is a small
// enough slice of RFC 6455 to write out rather than take a dependency for: the
// upgrade handshake, unmasked outbound text frames, and enough inbound parsing
// to honour a close and answer a ping.
//
// What this deliberately does not implement: fragmented outbound messages,
// permessage-deflate, binary frames, and subprotocol negotiation. It serves one
// local viewer, and anything beyond that should use a real library.

import { createHash } from "node:crypto";
import { createServer } from "node:http";

/** The GUID RFC 6455 fixes for the handshake. */
const ACCEPT_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

const OPCODE = { continuation: 0x0, text: 0x1, binary: 0x2, close: 0x8, ping: 0x9, pong: 0xa };

function acceptKey(key) {
  return createHash("sha1").update(key + ACCEPT_GUID).digest("base64");
}

/**
 * Encode one unmasked frame. Server-to-client frames must not be masked, which
 * is the one asymmetry in the protocol worth remembering.
 */
export function encodeFrame(payload, opcode = OPCODE.text) {
  const body = Buffer.isBuffer(payload) ? payload : Buffer.from(String(payload), "utf8");
  const length = body.length;
  let header;
  if (length < 126) {
    header = Buffer.alloc(2);
    header[1] = length;
  } else if (length < 65536) {
    header = Buffer.alloc(4);
    header[1] = 126;
    header.writeUInt16BE(length, 2);
  } else {
    header = Buffer.alloc(10);
    header[1] = 127;
    // Node cannot have a payload over 2^53 anyway, so the high word is zero.
    header.writeUInt32BE(0, 2);
    header.writeUInt32BE(length, 6);
  }
  header[0] = 0x80 | opcode; // FIN, single frame
  return Buffer.concat([header, body]);
}

/**
 * Pull complete frames off a buffer. Returns the frames it could read and
 * whatever bytes are left over, because a socket hands over arbitrary slices.
 */
export function decodeFrames(buffer) {
  const frames = [];
  let offset = 0;
  while (offset + 2 <= buffer.length) {
    const first = buffer[offset];
    const second = buffer[offset + 1];
    const opcode = first & 0x0f;
    const masked = (second & 0x80) !== 0;
    let length = second & 0x7f;
    let cursor = offset + 2;

    if (length === 126) {
      if (cursor + 2 > buffer.length) break;
      length = buffer.readUInt16BE(cursor);
      cursor += 2;
    } else if (length === 127) {
      if (cursor + 8 > buffer.length) break;
      // The high word is ignored: a frame that large is not something this
      // server will ever legitimately receive.
      length = buffer.readUInt32BE(cursor + 4);
      cursor += 8;
    }

    let mask = null;
    if (masked) {
      if (cursor + 4 > buffer.length) break;
      mask = buffer.subarray(cursor, cursor + 4);
      cursor += 4;
    }
    if (cursor + length > buffer.length) break;

    const payload = Buffer.from(buffer.subarray(cursor, cursor + length));
    if (mask) {
      for (let i = 0; i < payload.length; i++) payload[i] ^= mask[i % 4];
    }
    frames.push({ opcode, payload });
    offset = cursor + length;
  }
  return { frames, rest: buffer.subarray(offset) };
}

/**
 * Start a server that upgrades websocket requests and serves nothing else.
 *
 * `onConnect(client)` is called per viewer, with `send(object)` and `close()`.
 * Sends are fire-and-forget: a viewer that has gone away must not stop the
 * simulation, so a write to a dead socket is swallowed.
 */
export function createWebSocketServer({ port = 8765, host = "127.0.0.1", onConnect, onStatus } = {}) {
  const clients = new Set();

  const server = createServer((request, response) => {
    // Anything that is not an upgrade gets a plain answer, so hitting the port
    // in a browser says what it is rather than hanging.
    response.writeHead(426, { "Content-Type": "text/plain", Connection: "close" });
    response.end("This port serves the fly-fishing live WebSocket stream.\n");
  });

  server.on("upgrade", (request, socket) => {
    const key = request.headers["sec-websocket-key"];
    if (request.headers.upgrade?.toLowerCase() !== "websocket" || !key) {
      socket.end("HTTP/1.1 400 Bad Request\r\n\r\n");
      return;
    }
    socket.write(
      "HTTP/1.1 101 Switching Protocols\r\n" +
        "Upgrade: websocket\r\n" +
        "Connection: Upgrade\r\n" +
        `Sec-WebSocket-Accept: ${acceptKey(key)}\r\n\r\n`,
    );
    socket.setNoDelay(true);

    const client = {
      socket,
      send(message) {
        if (socket.destroyed || !socket.writable) return;
        try {
          socket.write(encodeFrame(JSON.stringify(message)));
        } catch {
          // A viewer that vanished mid-write is not the simulation's problem.
        }
      },
      close() {
        if (!socket.destroyed) {
          try {
            socket.write(encodeFrame(Buffer.alloc(0), OPCODE.close));
          } catch {
            /* already gone */
          }
          socket.end();
        }
      },
    };

    let pending = Buffer.alloc(0);
    socket.on("data", (chunk) => {
      pending = Buffer.concat([pending, chunk]);
      const { frames, rest } = decodeFrames(pending);
      pending = rest;
      for (const frame of frames) {
        if (frame.opcode === OPCODE.close) {
          client.close();
        } else if (frame.opcode === OPCODE.ping) {
          try {
            socket.write(encodeFrame(frame.payload, OPCODE.pong));
          } catch {
            /* gone */
          }
        }
      }
    });

    const drop = () => {
      clients.delete(client);
      onStatus?.(`viewer disconnected (${clients.size} connected)`);
    };
    socket.on("close", drop);
    socket.on("error", drop);

    clients.add(client);
    onStatus?.(`viewer connected (${clients.size} connected)`);
    onConnect?.(client);
  });

  return {
    clients,
    /** Send to every connected viewer. */
    broadcast(message) {
      for (const client of clients) client.send(message);
    },
    listen() {
      return new Promise((resolve, reject) => {
        // A port already in use is the likeliest failure here, and the default
        // message does not say what to do about it.
        server.once("error", (error) => {
          if (error.code === "EADDRINUSE") {
            reject(
              new Error(
                `port ${port} is already in use; another live stream is probably ` +
                  "running. Stop it, or pass --port with a different number.",
              ),
            );
          } else {
            reject(error);
          }
        });
        server.listen(port, host, () => resolve(server.address()));
      });
    },
    close() {
      for (const client of clients) client.close();
      server.close();
    },
  };
}
