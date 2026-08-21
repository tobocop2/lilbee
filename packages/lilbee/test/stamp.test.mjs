import { test } from "node:test";
import assert from "node:assert/strict";
import { npmVersionFor } from "../tools/stamp-release.mjs";

test("maps a build-suffixed release tag to an npm prerelease", () => {
  assert.equal(npmVersionFor("v0.6.90b423"), "0.6.90-b423");
  assert.equal(npmVersionFor("0.6.90b423"), "0.6.90-b423");
  assert.equal(npmVersionFor("v0.7.0rc1"), "0.7.0-rc1");
});

test("leaves a plain release tag alone", () => {
  assert.equal(npmVersionFor("v0.6.90"), "0.6.90");
});

test("normalizes a separator the tag already carries", () => {
  assert.equal(npmVersionFor("v0.6.90-b423"), "0.6.90-b423");
  assert.equal(npmVersionFor("v0.6.90.b423"), "0.6.90-b423");
});

test("rejects tags npm cannot take", () => {
  assert.throws(() => npmVersionFor("v0.6"), /MAJOR\.MINOR\.PATCH/);
  assert.throws(() => npmVersionFor("nightly"), /MAJOR\.MINOR\.PATCH/);
  assert.throws(() => npmVersionFor("v0.6.90b4_23"), /prerelease/);
});
