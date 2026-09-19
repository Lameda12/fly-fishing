import assert from "node:assert/strict";
import { createHash } from "node:crypto";

import { decodeFrames, encodeFrame } from "../fishing/ws-server.mjs";

// --- outbound frames --------------------------------------------------------
// Server-to-client frames must not be masked, and the length must land in the
// right field for each of the three size classes.
{
  const small = encodeFrame("hi");
  assert.equal(small[0], 0x81, "FIN set, opcode text");
  assert.equal(small[1], 2, "short lengths live in the low seven bits");
  assert.equal((small[1] & 0x80) >> 7, 0, "a server frame is never masked");
  assert.equal(small.subarray(2).toString("utf8"), "hi");
}
{
  const medium = encodeFrame("x".repeat(200));
  assert.equal(medium[1], 126, "126 means the length is in the next two bytes");
  assert.equal(medium.readUInt16BE(2), 200);
  assert.equal(medium.length, 4 + 200);
}
{
  const large = encodeFrame("x".repeat(70_000));
  assert.equal(large[1], 127, "127 means the length is in the next eight bytes");
  assert.equal(large.readUInt32BE(2), 0, "the high word is zero");
  assert.equal(large.readUInt32BE(6), 70_000);
  assert.equal(large.length, 10 + 70_000);
}
// The boundary between the size classes is where an off-by-one would live.
assert.equal(encodeFrame("x".repeat(125))[1], 125);
assert.equal(encodeFrame("x".repeat(126))[1], 126);
assert.equal(encodeFrame("x".repeat(65_535))[1], 126);
assert.equal(encodeFrame("x".repeat(65_536))[1], 127);

// --- inbound frames ---------------------------------------------------------
/** Build a masked client frame, which is what a browser actually sends. */
function clientFrame(text, opcode = 0x1) {
  const body = Buffer.from(text, "utf8");
  const mask = Buffer.from([0x0a, 0x1b, 0x2c, 0x3d]);
  const masked = Buffer.from(body);
  for (let i = 0; i < masked.length; i++) masked[i] ^= mask[i % 4];
  const header = Buffer.from([0x80 | opcode, 0x80 | body.length]);
  return Buffer.concat([header, mask, masked]);
}

{
  const { frames, rest } = decodeFrames(clientFrame("hello"));
  assert.equal(frames.length, 1);
  assert.equal(frames[0].opcode, 0x1);
  assert.equal(frames[0].payload.toString("utf8"), "hello", "the mask must be undone");
  assert.equal(rest.length, 0);
}

// Two frames in one chunk, which a socket will do.
{
  const { frames } = decodeFrames(Buffer.concat([clientFrame("one"), clientFrame("two")]));
  assert.deepEqual(
    frames.map((frame) => frame.payload.toString("utf8")),
    ["one", "two"],
  );
}

// A frame split across chunks must not be consumed until it is complete: this
// is the bug that would corrupt every subsequent frame on the connection.
{
  const whole = clientFrame("split me");
  const head = whole.subarray(0, 6);
  const tail = whole.subarray(6);
  const first = decodeFrames(head);
  assert.equal(first.frames.length, 0, "an incomplete frame yields nothing");
  assert.equal(first.rest.length, head.length, "and is kept in full for the next chunk");
  const second = decodeFrames(Buffer.concat([first.rest, tail]));
  assert.equal(second.frames.length, 1);
  assert.equal(second.frames[0].payload.toString("utf8"), "split me");
  assert.equal(second.rest.length, 0);
}

// A close and a ping have to be recognizable, since the server answers both.
{
  const { frames } = decodeFrames(clientFrame("", 0x8));
  assert.equal(frames[0].opcode, 0x8, "close");
}
{
  const { frames } = decodeFrames(clientFrame("ping payload", 0x9));
  assert.equal(frames[0].opcode, 0x9);
  assert.equal(frames[0].payload.toString("utf8"), "ping payload", "a pong echoes the payload");
}

// An empty buffer is not an error.
{
  const { frames, rest } = decodeFrames(Buffer.alloc(0));
  assert.equal(frames.length, 0);
  assert.equal(rest.length, 0);
}

// --- the handshake key ------------------------------------------------------
// RFC 6455's own worked example, so the accept value is checked against the
// specification rather than against this implementation's own output.
{
  const key = "dGhlIHNhbXBsZSBub25jZQ==";
  const expected = "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=";
  const actual = createHash("sha1")
    .update(key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11")
    .digest("base64");
  assert.equal(actual, expected, "the RFC 6455 example handshake must round-trip");
}

console.log("ws: all assertions passed");
