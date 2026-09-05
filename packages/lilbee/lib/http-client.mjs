/**
 * The launcher's fetch contract over Node's own http and https clients, for
 * runtimes whose global fetch is a browser's: an Electron renderer refuses
 * GitHub's cross-host asset redirect under CORS.
 */

import { request as httpRequest } from "node:http";
import { request as httpsRequest } from "node:https";

const MAX_REDIRECTS = 5;
const REDIRECT_STATUSES = new Set([301, 302, 303, 307, 308]);
const AUTHORIZATION_HEADER = "authorization";

function abortError() {
  return new DOMException("The request was aborted.", "AbortError");
}

/** One GET; resolves with the raw response once its headers arrive. */
function get(url, headers, signal) {
  return new Promise((resolve, reject) => {
    const request = url.protocol === "https:" ? httpsRequest : httpRequest;
    let response = null;
    const req = request(url, { method: "GET", headers }, (res) => {
      response = res;
      res.on("close", () => signal?.removeEventListener("abort", onAbort));
      resolve(res);
    });
    // After the headers the body is the caller's stream, so it carries the abort.
    const onAbort = () => (response ?? req).destroy(abortError());
    signal?.addEventListener("abort", onAbort, { once: true });
    req.on("error", reject);
    req.end();
  });
}

function collect(body) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    body.on("data", (chunk) => chunks.push(chunk));
    body.on("error", reject);
    body.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
  });
}

function toResponse(res) {
  const status = res.statusCode ?? 0;
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: {
      get: (name) => {
        const value = res.headers[name.toLowerCase()];
        if (value === undefined) return null;
        return Array.isArray(value) ? value.join(", ") : value;
      },
    },
    body: res,
    text: () => collect(res),
    json: async () => JSON.parse(await collect(res)),
  };
}

function withoutAuthorization(headers) {
  return Object.fromEntries(Object.entries(headers).filter(([name]) => name.toLowerCase() !== AUTHORIZATION_HEADER));
}

/** GET `rawUrl`, following up to five redirects; the authorization header stays on its own host. */
export async function nodeFetch(rawUrl, init = {}) {
  const { signal } = init;
  if (signal?.aborted) throw abortError();
  let url = new URL(rawUrl);
  let headers = { ...(init.headers ?? {}) };
  for (let hop = 0; ; hop++) {
    const res = await get(url, headers, signal);
    const location = res.headers.location;
    if (!REDIRECT_STATUSES.has(res.statusCode ?? 0) || !location) return toResponse(res);
    res.resume();
    if (hop >= MAX_REDIRECTS) throw new Error(`GET ${rawUrl}: too many redirects`);
    const next = new URL(location, url);
    if (next.host !== url.host) headers = withoutAuthorization(headers);
    url = next;
  }
}
