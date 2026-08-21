/**
 * Stamp a release tag into the npm launcher package before publishing.
 *
 * Release tags carry a build suffix (`v0.6.90b423`) that npm rejects, so the
 * tag maps to a semver prerelease (`0.6.90-b423`). The launcher still needs the
 * original tag to find the release assets, so it is written to `lilbee.release`
 * verbatim.
 *
 * Usage: node packages/lilbee/tools/stamp-release.mjs v0.6.90b423
 * Prints the npm version on stdout.
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const PKG = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "package.json");

/**
 * Map a release tag to an npm version.
 *
 * @param {string} tag - release tag, with or without the leading "v"
 * @returns {string} npm semver version
 */
export function npmVersionFor(tag) {
  const m = /^v?(\d+\.\d+\.\d+)(.*)$/.exec(String(tag).trim());
  if (!m) throw new Error(`tag "${tag}" is not a MAJOR.MINOR.PATCH release tag`);
  const [, core, suffix] = m;
  const pre = suffix.replace(/^[-._+]+/, "");
  if (pre && !/^[0-9A-Za-z.-]+$/.test(pre)) {
    throw new Error(`tag "${tag}" has a suffix npm cannot use as a prerelease: "${suffix}"`);
  }
  return pre ? `${core}-${pre}` : core;
}

/**
 * Write the version and the pinned release tag into package.json.
 *
 * @param {string} tag - release tag whose assets the launcher downloads
 * @returns {string} the npm version written
 */
export function stamp(tag) {
  const version = npmVersionFor(tag);
  const pkg = JSON.parse(fs.readFileSync(PKG, "utf8"));
  pkg.version = version;
  pkg.lilbee.release = tag.startsWith("v") ? tag : `v${tag}`;
  fs.writeFileSync(PKG, `${JSON.stringify(pkg, null, 2)}\n`);
  return version;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const tag = process.argv[2];
  if (!tag) {
    console.error("usage: node tools/stamp-release.mjs <release-tag>");
    process.exit(2);
  }
  console.log(stamp(tag));
}
