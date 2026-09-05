import { test } from "node:test";
import assert from "node:assert/strict";
import { browserRuntime, launcherFetch } from "../lib/fetch.mjs";
import { nodeFetch } from "../lib/http-client.mjs";

test("a scope with window and document is a browser runtime", () => {
  assert.equal(browserRuntime({ window: {}, document: {} }), true);
  assert.equal(browserRuntime({ window: {} }), false);
  assert.equal(browserRuntime({}), false);
});

test("inside a browser runtime the launcher uses Node's own client", async () => {
  const scope = { window: {}, document: {}, fetch: async () => ({}) };
  assert.equal(await launcherFetch({}, () => {}, scope), nodeFetch);
});

test("outside a browser runtime the launcher uses the scope's fetch", async () => {
  const fetch = async () => ({});
  assert.equal(await launcherFetch({}, () => {}, { fetch }), fetch);
});
