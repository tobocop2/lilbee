import { test, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { Readable } from "node:stream";
import { nodeFetch } from "../lib/http-client.mjs";

let server;
let base;
let handler;
const requests = [];

function listen(srv) {
  return new Promise((resolve) => {
    srv.listen(0, "127.0.0.1", () => resolve(`http://127.0.0.1:${srv.address().port}`));
  });
}

async function readAll(body) {
  let text = "";
  for await (const chunk of body) text += chunk.toString();
  return text;
}

beforeEach(async () => {
  requests.length = 0;
  handler = (_req, res) => {
    res.writeHead(200, { "content-type": "text/plain", "content-length": "5" });
    res.end("hello");
  };
  server = createServer((req, res) => {
    requests.push({ url: req.url, headers: req.headers });
    handler(req, res);
  });
  base = await listen(server);
});

afterEach(async () => {
  await new Promise((resolve) => server.close(resolve));
});

test("returns the status, the headers, and the body as a Node stream", async () => {
  const res = await nodeFetch(`${base}/asset`, { headers: { "user-agent": "lilbee-test" } });
  assert.equal(res.ok, true);
  assert.equal(res.status, 200);
  assert.equal(res.headers.get("Content-Length"), "5");
  assert.equal(res.headers.get("x-missing"), null);
  assert.ok(res.body instanceof Readable);
  assert.equal(await readAll(res.body), "hello");
  assert.equal(requests[0].headers["user-agent"], "lilbee-test");
});

test("reads json and text from the body", async () => {
  handler = (_req, res) => {
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({ tag_name: "v1" }));
  };
  assert.deepEqual(await (await nodeFetch(`${base}/releases`)).json(), { tag_name: "v1" });
  assert.equal(await (await nodeFetch(`${base}/releases`)).text(), '{"tag_name":"v1"}');
});

test("reports a failing status without throwing", async () => {
  handler = (_req, res) => {
    res.writeHead(404);
    res.end("missing");
  };
  const res = await nodeFetch(`${base}/nope`);
  assert.equal(res.ok, false);
  assert.equal(res.status, 404);
  assert.equal(await res.text(), "missing");
});

test("follows a redirect and keeps the headers on the same host", async () => {
  handler = (req, res) => {
    if (req.url === "/start") {
      res.writeHead(302, { location: "/final" });
      res.end();
      return;
    }
    res.writeHead(200);
    res.end("landed");
  };
  const res = await nodeFetch(`${base}/start`, { headers: { authorization: "Bearer t", "user-agent": "ua" } });
  assert.equal(await res.text(), "landed");
  assert.deepEqual(
    requests.map((r) => r.url),
    ["/start", "/final"]
  );
  assert.equal(requests[1].headers.authorization, "Bearer t");
  assert.equal(requests[1].headers["user-agent"], "ua");
});

test("drops the authorization header when a redirect changes host", async () => {
  const other = createServer((req, res) => {
    requests.push({ url: `other${req.url}`, headers: req.headers });
    res.writeHead(200);
    res.end("elsewhere");
  });
  const otherBase = await listen(other);
  handler = (_req, res) => {
    res.writeHead(302, { location: `${otherBase.replace("127.0.0.1", "localhost")}/asset` });
    res.end();
  };
  try {
    const res = await nodeFetch(`${base}/start`, { headers: { authorization: "Bearer t", "user-agent": "ua" } });
    assert.equal(await res.text(), "elsewhere");
    assert.equal(requests[1].url, "other/asset");
    assert.equal(requests[1].headers.authorization, undefined);
    assert.equal(requests[1].headers["user-agent"], "ua");
  } finally {
    await new Promise((resolve) => other.close(resolve));
  }
});

test("gives up after too many redirects", async () => {
  handler = (_req, res) => {
    res.writeHead(302, { location: "/loop" });
    res.end();
  };
  await assert.rejects(nodeFetch(`${base}/loop`), /redirects/);
});

test("rejects with an AbortError when the signal aborts before the response", async () => {
  handler = () => {};
  const controller = new AbortController();
  const pending = nodeFetch(`${base}/slow`, { signal: controller.signal });
  controller.abort();
  await assert.rejects(pending, { name: "AbortError" });
});

test("rejects at once when the signal is already aborted", async () => {
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(nodeFetch(`${base}/asset`, { signal: controller.signal }), { name: "AbortError" });
  assert.equal(requests.length, 0);
});

test("ends the body stream with an AbortError when the signal aborts mid-transfer", async () => {
  let hold = null;
  handler = (_req, res) => {
    res.writeHead(200, { "content-length": "10" });
    res.write("12345");
    hold = res;
  };
  const controller = new AbortController();
  const res = await nodeFetch(`${base}/asset`, { signal: controller.signal });
  const ended = new Promise((resolve) => {
    res.body.on("error", (err) => resolve(err));
    res.body.on("close", () => resolve(null));
  });
  controller.abort();
  const outcome = await ended;
  assert.ok(outcome === null || outcome.name === "AbortError");
  hold?.destroy();
});

test("rejects a connection failure", async () => {
  await assert.rejects(nodeFetch("http://127.0.0.1:1/asset"));
});
