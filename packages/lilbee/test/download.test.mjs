import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { createHash } from "node:crypto";
import { Readable } from "node:stream";
import { DownloadCanceledError, download, isDownloadCanceled, progressLine, progressReporter } from "../lib/download.mjs";
import { LauncherError } from "../lib/errors.mjs";

const MB = 1048576;
const PAYLOAD = Buffer.from("lilbee binary bytes ".repeat(64));
const sha256 = (buf) => createHash("sha256").update(buf).digest("hex");

const tmpDir = () => fs.mkdtempSync(path.join(os.tmpdir(), "lilbee-dl-"));

function release(overrides = {}) {
  return {
    tag: "v9",
    dev: false,
    assetName: "lilbee-macos-arm64",
    variant: "default",
    size: PAYLOAD.length,
    digest: sha256(PAYLOAD),
    url: "https://dl/lilbee-macos-arm64",
    ...overrides,
  };
}

/** Split a buffer into `n` chunks for a streamed body. */
function chunks(buf, n = 4) {
  const size = Math.ceil(buf.length / n);
  return Array.from({ length: n }, (_, i) => buf.subarray(i * size, (i + 1) * size));
}

/** A fetch stub: `bodies` are consumed one per call; each is a web stream, a Node Readable, or a function. */
function fetchWith(bodies, { status = 200, contentLength = undefined } = {}) {
  const calls = [];
  const fetch = async (url, init = {}) => {
    calls.push({ url, init });
    const next = bodies[Math.min(calls.length - 1, bodies.length - 1)];
    const body = typeof next === "function" ? next(init) : next;
    return {
      ok: status < 400,
      status,
      headers: { get: (name) => (name === "content-length" && contentLength !== undefined ? String(contentLength) : null) },
      body,
      json: async () => ({}),
      text: async () => "",
    };
  };
  return { fetch, calls };
}

const webBody = (parts) => Readable.toWeb(Readable.from(parts));

/** A body that delivers `parts` and then never ends, like a dead socket. */
function hangingBody(parts) {
  const body = new Readable({ read() {} });
  for (const part of parts) body.push(part);
  return body;
}

test("download streams to the destination, verifies sha256, and reports done/total per chunk", async () => {
  const dir = tmpDir();
  const dest = path.join(dir, "v9", "lilbee-macos-arm64");
  const progress = [];
  const logs = [];
  const { fetch, calls } = fetchWith([webBody(chunks(PAYLOAD))], { contentLength: PAYLOAD.length });
  await download({ release: release(), dest, fetch, onProgress: (p) => progress.push(p), log: (m) => logs.push(m) });
  assert.ok(Buffer.from(fs.readFileSync(dest)).equals(PAYLOAD));
  assert.equal(fs.statSync(dest).mode & 0o111, 0o111);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, "https://dl/lilbee-macos-arm64");
  assert.equal(progress.length, 5);
  assert.deepEqual(progress[0], { done: 0, total: PAYLOAD.length });
  assert.deepEqual(progress.at(-1), { done: PAYLOAD.length, total: PAYLOAD.length });
  assert.ok(progress.every((p) => p.total === PAYLOAD.length));
  assert.ok(logs.some((l) => /downloading lilbee-macos-arm64/.test(l)));
  assert.ok(fs.readdirSync(path.join(dir, "v9")).every((f) => !f.includes(".download.")));
  fs.rmSync(dir, { recursive: true, force: true });
});

test("progress starts with a zero-byte event when the transfer opens", async () => {
  const dir = tmpDir();
  const events = [];
  await download({ release: release(), dest: path.join(dir, "v9", "bin"), fetch: fetchWith([webBody(chunks(PAYLOAD, 2))]).fetch, onProgress: (p) => events.push(p) });
  assert.deepEqual(events[0], { done: 0, total: PAYLOAD.length });
  assert.equal(events.length, 3);
  assert.equal(events.at(-1).done, PAYLOAD.length);
  fs.rmSync(dir, { recursive: true, force: true });
});

test("total falls back to the release size without Content-Length, and to null without either", async () => {
  const dir = tmpDir();
  const dest = path.join(dir, "v9", "bin");
  let seen = [];
  await download({ release: release(), dest, fetch: fetchWith([webBody(chunks(PAYLOAD))]).fetch, onProgress: (p) => seen.push(p) });
  assert.equal(seen[0].total, PAYLOAD.length);
  seen = [];
  await download({ release: release({ size: 0 }), dest, fetch: fetchWith([webBody(chunks(PAYLOAD))]).fetch, onProgress: (p) => seen.push(p) });
  assert.equal(seen[0].total, null);
  fs.rmSync(dir, { recursive: true, force: true });
});

test("a Node Readable body works as well as a web stream", async () => {
  const dir = tmpDir();
  const dest = path.join(dir, "v9", "bin");
  await download({ release: release(), dest, fetch: fetchWith([Readable.from(chunks(PAYLOAD))]).fetch });
  assert.ok(Buffer.from(fs.readFileSync(dest)).equals(PAYLOAD));
  fs.rmSync(dir, { recursive: true, force: true });
});

test("a digest mismatch is retried once, then rejected with nothing left on disk", async () => {
  const dir = tmpDir();
  const dest = path.join(dir, "v9", "bin");
  const logs = [];
  const { fetch, calls } = fetchWith([() => webBody(chunks(PAYLOAD))]);
  await assert.rejects(
    download({ release: release({ digest: "0".repeat(64) }), dest, fetch, log: (m) => logs.push(m) }),
    (err) => err instanceof LauncherError && err.code === "digest-mismatch" && /sha256 mismatch/.test(err.message)
  );
  assert.equal(calls.length, 2);
  assert.deepEqual(fs.readdirSync(path.join(dir, "v9")), []);
  assert.ok(logs.some((l) => /retrying once/.test(l)));
  fs.rmSync(dir, { recursive: true, force: true });
});

test("a release without a digest lands unverified and says so", async () => {
  const dir = tmpDir();
  const dest = path.join(dir, "v9", "bin");
  const logs = [];
  await download({ release: release({ digest: null }), dest, fetch: fetchWith([webBody(chunks(PAYLOAD))]).fetch, log: (m) => logs.push(m) });
  assert.ok(fs.existsSync(dest));
  assert.ok(logs.some((l) => /no published digest/.test(l)));
  fs.rmSync(dir, { recursive: true, force: true });
});

test("requireDigest refuses a release without a digest before any request", async () => {
  const dir = tmpDir();
  const { fetch, calls } = fetchWith([webBody(chunks(PAYLOAD))]);
  await assert.rejects(
    download({ release: release({ digest: null }), dest: path.join(dir, "v9", "bin"), fetch, requireDigest: true }),
    (err) => err instanceof LauncherError && err.code === "no-digest"
  );
  assert.equal(calls.length, 0);
  fs.rmSync(dir, { recursive: true, force: true });
});

test("the download banner and the non-TTY progress lines read as they did in 0.6.96", async () => {
  const dir = tmpDir();
  const logs = [];
  await download({ release: release(), dest: path.join(dir, "v9", "bin"), fetch: fetchWith([webBody(chunks(PAYLOAD))]).fetch, log: (m) => logs.push(m) });
  assert.equal(logs[0], "lilbee: downloading lilbee-macos-arm64 (0MB) from tobocop2/lilbee@v9. One-time per release; cached after that.");
  assert.equal(progressLine({ done: 3 * MB, total: 0, bytesPerSec: 0 }), "lilbee: downloading… 3MB");
  const lines = [];
  progressReporter((m) => lines.push(m), { isTTY: false })({ done: 5 * MB, total: 10 * MB });
  assert.deepEqual(lines, ["lilbee: downloading… 50% (5MB)"]);
  fs.rmSync(dir, { recursive: true, force: true });
});

test("aborting mid-stream rejects with DownloadCanceledError, removes the temp file, and never retries", async () => {
  const dir = tmpDir();
  const dest = path.join(dir, "v9", "bin");
  const controller = new AbortController();
  const { fetch, calls } = fetchWith([() => hangingBody([chunks(PAYLOAD)[0]])]);
  setTimeout(() => controller.abort(), 20);
  const err = await download({ release: release(), dest, fetch, signal: controller.signal }).then(
    () => assert.fail("must reject"),
    (e) => e
  );
  assert.ok(err instanceof DownloadCanceledError);
  assert.equal(err.name, "AbortError");
  assert.ok(isDownloadCanceled(err));
  assert.equal(calls.length, 1);
  assert.deepEqual(fs.readdirSync(path.join(dir, "v9")), []);
  fs.rmSync(dir, { recursive: true, force: true });
});

test("an already-aborted signal rejects before any request is made", async () => {
  const dir = tmpDir();
  const controller = new AbortController();
  controller.abort();
  const { fetch, calls } = fetchWith([webBody(chunks(PAYLOAD))]);
  await assert.rejects(download({ release: release(), dest: path.join(dir, "v9", "bin"), fetch, signal: controller.signal }), (e) => isDownloadCanceled(e));
  assert.equal(calls.length, 0);
  fs.rmSync(dir, { recursive: true, force: true });
});

test("the abort signal reaches the transport so a web fetch can drop its socket", async () => {
  const dir = tmpDir();
  const controller = new AbortController();
  const { fetch, calls } = fetchWith([webBody(chunks(PAYLOAD))]);
  await download({ release: release(), dest: path.join(dir, "v9", "bin"), fetch, signal: controller.signal });
  assert.ok(calls[0].init.signal instanceof AbortSignal);
  fs.rmSync(dir, { recursive: true, force: true });
});

test("a stalled transfer is retried once, then rejected with a stall message", async () => {
  const dir = tmpDir();
  const dest = path.join(dir, "v9", "bin");
  const logs = [];
  const { fetch, calls } = fetchWith([() => hangingBody([chunks(PAYLOAD)[0]])]);
  await assert.rejects(download({ release: release(), dest, fetch, log: (m) => logs.push(m), idleTimeoutMs: 30 }), /download stalled/);
  assert.equal(calls.length, 2);
  assert.deepEqual(fs.readdirSync(path.join(dir, "v9")), []);
  fs.rmSync(dir, { recursive: true, force: true });
});

test("a stall while waiting for headers is covered by the same clock", async () => {
  const dir = tmpDir();
  const calls = [];
  const fetch = (url, init) =>
    new Promise((_, reject) => {
      calls.push(url);
      init.signal.addEventListener("abort", () => reject(init.signal.reason ?? new Error("aborted")));
    });
  await assert.rejects(download({ release: release(), dest: path.join(dir, "v9", "bin"), fetch, idleTimeoutMs: 30 }), /download stalled/);
  assert.equal(calls.length, 2);
  fs.rmSync(dir, { recursive: true, force: true });
});

test("a transfer error is retried once and the second attempt lands", async () => {
  const dir = tmpDir();
  const dest = path.join(dir, "v9", "bin");
  async function* broken() {
    yield chunks(PAYLOAD)[0];
    throw new Error("ECONNRESET");
  }
  const { fetch, calls } = fetchWith([Readable.from(broken()), webBody(chunks(PAYLOAD))]);
  await download({ release: release(), dest, fetch });
  assert.equal(calls.length, 2);
  assert.ok(Buffer.from(fs.readFileSync(dest)).equals(PAYLOAD));
  fs.rmSync(dir, { recursive: true, force: true });
});

test("an HTTP error on the asset counts as a failed attempt", async () => {
  const dir = tmpDir();
  const { fetch, calls } = fetchWith([null], { status: 502 });
  await assert.rejects(download({ release: release(), dest: path.join(dir, "v9", "bin"), fetch }), /HTTP 502/);
  assert.equal(calls.length, 2);
  fs.rmSync(dir, { recursive: true, force: true });
});

test("a rival process that lands the file first is adopted instead of retried", async () => {
  const dir = tmpDir();
  const dest = path.join(dir, "v9", "bin");
  const logs = [];
  async function* losing() {
    yield chunks(PAYLOAD)[0];
    fs.writeFileSync(dest, "rival");
    throw new Error("ECONNRESET");
  }
  const { fetch, calls } = fetchWith([Readable.from(losing())]);
  await download({ release: release(), dest, fetch, log: (m) => logs.push(m) });
  assert.equal(calls.length, 1);
  assert.equal(fs.readFileSync(dest, "utf8"), "rival");
  assert.ok(logs.some((l) => /another process finished/.test(l)));
  fs.rmSync(dir, { recursive: true, force: true });
});

test("a landed download leaves a live rival's in-progress temp file alone", async () => {
  const dir = tmpDir();
  const dest = path.join(dir, "v9", "lilbee-macos-arm64");
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  const rival = `${dest}.download.${process.ppid}`;
  fs.writeFileSync(rival, "partial");
  fs.writeFileSync(path.join(dir, "v9", "lilbee-macos-x86_64"), "old build");
  await download({ release: release(), dest, fetch: fetchWith([webBody(chunks(PAYLOAD))]).fetch });
  assert.deepEqual(fs.readdirSync(path.join(dir, "v9")).sort(), ["lilbee-macos-arm64", `lilbee-macos-arm64.download.${process.ppid}`]);
  fs.rmSync(dir, { recursive: true, force: true });
});

test("a download whose temp file vanished adopts the binary a rival landed", async () => {
  const dir = tmpDir();
  const dest = path.join(dir, "v9", "bin");
  const logs = [];
  const body = () => { const r = webBody(chunks(PAYLOAD)); fs.mkdirSync(path.dirname(dest), { recursive: true }); return r; };
  const { fetch } = fetchWith([body]);
  // The rival lands dest and removes our temp file while our stream is still open.
  const tmp = `${dest}.download.${process.pid}`;
  const timer = setInterval(() => { if (fs.existsSync(tmp)) { fs.writeFileSync(dest, PAYLOAD); fs.rmSync(tmp, { force: true }); clearInterval(timer); } }, 1);
  await download({ release: release(), dest, fetch, log: (m) => logs.push(m) });
  clearInterval(timer);
  assert.ok(fs.existsSync(dest));
  assert.ok(logs.some((l) => /another process finished/.test(l)));
  fs.rmSync(dir, { recursive: true, force: true });
});

test("a landed download removes the other builds of the same release", async () => {
  const dir = tmpDir();
  const releaseDir = path.join(dir, "v9");
  fs.mkdirSync(releaseDir, { recursive: true });
  fs.writeFileSync(path.join(releaseDir, "lilbee-linux-x86_64"), "old default build");
  // 4194305 is above Linux's pid_max, so no runner can have a process with this pid
  fs.writeFileSync(path.join(releaseDir, "lilbee-linux-x86_64.download.4194305"), "stale temp");
  const dest = path.join(releaseDir, "lilbee-linux-x86_64-cu125");
  await download({ release: release({ assetName: "lilbee-linux-x86_64-cu125" }), dest, fetch: fetchWith([webBody(chunks(PAYLOAD))]).fetch });
  assert.deepEqual(fs.readdirSync(releaseDir), ["lilbee-linux-x86_64-cu125"]);
  fs.rmSync(dir, { recursive: true, force: true });
});

test("a download that cannot fit on the disk is refused before any request", { skip: typeof fs.promises.statfs !== "function" }, async () => {
  const dir = tmpDir();
  const { fetch, calls } = fetchWith([webBody(chunks(PAYLOAD))]);
  const huge = release({ size: 2 ** 52 });
  await assert.rejects(download({ release: huge, dest: path.join(dir, "v9", "bin"), fetch }), /Not enough disk space.*needs about .*free.*is available/);
  assert.equal(calls.length, 0);
  fs.rmSync(dir, { recursive: true, force: true });
});

test("isDownloadCanceled accepts any AbortError and nothing else", () => {
  assert.equal(isDownloadCanceled(new DownloadCanceledError()), true);
  assert.equal(isDownloadCanceled(Object.assign(new Error("x"), { name: "AbortError" })), true);
  assert.equal(isDownloadCanceled(new Error("boom")), false);
  assert.equal(isDownloadCanceled(null), false);
});

// --- progress line rendering (TTY bar) ---

test("progressLine draws an empty, half, and full bar at a fixed width", () => {
  const at = (done) => progressLine({ done, total: 100 * MB, bytesPerSec: 0, barWidth: 10, color: false });
  assert.match(at(0), /│─{10}│ +0%/);
  assert.match(at(50 * MB), /│━{5}─{5}│ +50%/);
  assert.match(at(100 * MB), /│━{10}│ +100%/);
});

test("progressLine marks a partial cell with a half-step edge", () => {
  const line = progressLine({ done: 55 * MB, total: 100 * MB, bytesPerSec: 0, barWidth: 10, color: false });
  assert.match(line, /│━{5}╸─{4}│ +55%/);
});

test("progressLine shows sizes, rate, and eta", () => {
  const line = progressLine({ done: 523 * MB, total: 1246 * MB, bytesPerSec: 87 * MB, barWidth: 10, color: false });
  assert.match(line, /523\/1246MB/);
  assert.match(line, /87\.0MB\/s/);
  assert.match(line, /eta 0:0[89]/); // (1246-523)/87 is about 8.3s
});

test("progressLine clamps overshoot and hides rate/eta when unknown", () => {
  const over = progressLine({ done: 120 * MB, total: 100 * MB, bytesPerSec: 0, barWidth: 10, color: false });
  assert.match(over, /100%/);
  assert.doesNotMatch(over, /eta/);
  assert.doesNotMatch(over, /MB\/s/);
});

test("progressLine without a total reports plain progress", () => {
  const line = progressLine({ done: 42 * MB, total: 0, bytesPerSec: 0, barWidth: 10, color: false });
  assert.match(line, /42MB/);
  assert.doesNotMatch(line, /%/);
});

test("progressLine colors the bar only when asked", () => {
  const colored = progressLine({ done: 50 * MB, total: 100 * MB, bytesPerSec: 0, barWidth: 10, color: true });
  const plain = progressLine({ done: 50 * MB, total: 100 * MB, bytesPerSec: 0, barWidth: 10, color: false });
  assert.match(colored, /\x1b\[/);
  assert.doesNotMatch(plain, /\x1b\[/);
});

// --- progress as an onProgress consumer ---

test("on a TTY the reporter redraws one line in place and ends it with a newline once the total lands", () => {
  const writes = [];
  const stream = { isTTY: true, write: (s) => writes.push(s) };
  const report = progressReporter(() => {}, stream);
  report({ done: 2 * MB, total: 4 * MB });
  report({ done: 4 * MB, total: 4 * MB });
  assert.ok(writes.length >= 2);
  assert.ok(writes.slice(0, -1).every((w) => w.startsWith("\r")));
  assert.match(writes.at(-2), /100% 4\/4MB/);
  assert.equal(writes.at(-1), "\n");
  report.finish();
  assert.equal(writes.at(-1), "\n");
  assert.equal(writes.filter((w) => w === "\n").length, 1);
});

test("on a TTY finish() closes a bar that stopped short, and does nothing when nothing was drawn", () => {
  const writes = [];
  const stream = { isTTY: true, write: (s) => writes.push(s) };
  const report = progressReporter(() => {}, stream);
  report({ done: MB, total: 4 * MB });
  report.finish();
  assert.equal(writes.at(-1), "\n");
  const untouched = [];
  progressReporter(() => {}, { isTTY: true, write: (s) => untouched.push(s) }).finish();
  assert.deepEqual(untouched, []);
});

test("off a TTY the reporter logs a line per 10% and starts over when a retry restarts the count", () => {
  const lines = [];
  const stream = { isTTY: false, write: () => assert.fail("must not draw") };
  const report = progressReporter((m) => lines.push(m), stream);
  for (let i = 1; i <= 10; i += 1) report({ done: i * MB, total: 10 * MB });
  assert.equal(lines.length, 10);
  assert.match(lines[0], /10% \(1MB\)/);
  assert.match(lines.at(-1), /100% \(10MB\)/);
  report({ done: MB, total: 10 * MB });
  assert.equal(lines.length, 11);
  assert.match(lines.at(-1), /10% \(1MB\)/);
  report.finish();
  assert.equal(lines.length, 11);
});

test("stall, HTTP, and disk-space failures carry their LauncherError codes", async () => {
  const dir = tmpDir();
  const stalled = download({ release: release(), dest: path.join(dir, "v9", "a"), fetch: fetchWith([() => hangingBody([])]).fetch, idleTimeoutMs: 5 });
  await assert.rejects(stalled, (err) => err instanceof LauncherError && err.code === "stalled");
  const http = download({ release: release(), dest: path.join(dir, "v9", "b"), fetch: fetchWith([null], { status: 502 }).fetch });
  await assert.rejects(http, (err) => err instanceof LauncherError && err.code === "http" && err.status === 502);
  const space = new LauncherError("no-space", "full", { neededBytes: 10, freeBytes: 1 });
  assert.equal(space.name, "LauncherError");
  assert.deepEqual([space.code, space.neededBytes, space.freeBytes], ["no-space", 10, 1]);
  fs.rmSync(dir, { recursive: true, force: true });
});
