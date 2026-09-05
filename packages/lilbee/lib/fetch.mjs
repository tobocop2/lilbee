/** The launcher's default transport: global fetch, routed through the env proxy when one is set. */

import { nodeFetch } from "./http-client.mjs";

const PROXY_VARS = ["HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"];

/** A browser's fetch is in scope: an Electron renderer, where CORS refuses GitHub's asset redirect. */
export function browserRuntime(scope = globalThis) {
  return typeof scope.window !== "undefined" && typeof scope.document !== "undefined";
}

/**
 * Node's own client inside a browser runtime; otherwise global fetch, with
 * undici's env proxy agent installed first when the environment names a
 * proxy. Node's fetch ignores HTTP(S)_PROXY on its own.
 */
export async function launcherFetch(env = process.env, log = () => {}, scope = globalThis) {
  if (browserRuntime(scope)) return nodeFetch;
  if (PROXY_VARS.some((name) => env[name])) {
    try {
      const { EnvHttpProxyAgent, getGlobalDispatcher, setGlobalDispatcher } = await import("undici");
      if (!(getGlobalDispatcher() instanceof EnvHttpProxyAgent)) setGlobalDispatcher(new EnvHttpProxyAgent());
    } catch {
      log("lilbee: HTTPS_PROXY is set but the proxy agent is unavailable; downloading directly.");
    }
  }
  return scope.fetch;
}
