var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};
var __copyProps = (to, from, except, desc) => {
  if (from && typeof from === "object" || typeof from === "function") {
    for (let key of __getOwnPropNames(from))
      if (!__hasOwnProp.call(to, key) && key !== except)
        __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
  }
  return to;
};
var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);

// src/main.ts
var main_exports = {};
__export(main_exports, {
  default: () => LilbeePlugin
});
module.exports = __toCommonJS(main_exports);
var import_obsidian5 = require("obsidian");

// src/types.ts
var SERVER_STATE = {
  STOPPED: "stopped",
  DOWNLOADING: "downloading",
  STARTING: "starting",
  READY: "ready",
  ERROR: "error"
};
var SERVER_MODE = {
  MANAGED: "managed",
  EXTERNAL: "external"
};
var DEFAULT_SETTINGS = {
  serverUrl: "http://127.0.0.1:7433",
  topK: 5,
  syncMode: "manual",
  syncDebounceMs: 5e3,
  ollamaUrl: "http://127.0.0.1:11434",
  temperature: null,
  top_p: null,
  top_k_sampling: null,
  repeat_penalty: null,
  num_ctx: null,
  seed: null,
  serverMode: "managed",
  serverPort: null,
  lilbeeVersion: ""
};
var SSE_EVENT = {
  TOKEN: "token",
  SOURCES: "sources",
  DONE: "done",
  ERROR: "error",
  PROGRESS: "progress",
  MESSAGE: "message",
  FILE_START: "file_start",
  EXTRACT: "extract",
  EMBED: "embed",
  FILE_DONE: "file_done",
  PULL: "pull"
};
var JSON_HEADERS = { "Content-Type": "application/json" };

// src/api.ts
var DEFAULT_TIMEOUT_MS = 15e3;
var RETRY_COUNT = 2;
var RETRY_BACKOFF_MS = 500;
var LilbeeClient = class {
  constructor(baseUrl) {
    this.baseUrl = baseUrl;
  }
  async assertOk(res) {
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`Server responded ${res.status}: ${text}`);
    }
    return res;
  }
  /**
   * Fetch with automatic retry on network errors and a default timeout.
   * Does NOT retry on HTTP error responses (4xx/5xx) — only on fetch failures
   * (e.g. connection refused, DNS failure, timeout).
   * SSE streams should pass `stream: true` to skip the timeout.
   */
  async fetchWithRetry(url, init, opts) {
    const maxAttempts = RETRY_COUNT + 1;
    let lastError;
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      if (attempt > 0) {
        await new Promise((r) => setTimeout(r, RETRY_BACKOFF_MS * attempt));
      }
      try {
        const fetchInit = { ...init };
        if (opts?.signal) {
          fetchInit.signal = opts.signal;
        } else if (!opts?.stream) {
          const controller = new AbortController();
          fetchInit.signal = controller.signal;
          setTimeout(() => controller.abort(), DEFAULT_TIMEOUT_MS);
        }
        return await this.assertOk(await fetch(url, fetchInit));
      } catch (err) {
        lastError = err;
        if (err instanceof Error && err.message.startsWith("Server responded")) {
          throw err;
        }
        if (err instanceof Error && err.name === "AbortError") {
          throw err;
        }
      }
    }
    throw lastError;
  }
  async health() {
    const res = await this.fetchWithRetry(`${this.baseUrl}/api/health`);
    return res.json();
  }
  async status() {
    const res = await this.fetchWithRetry(`${this.baseUrl}/api/status`);
    return res.json();
  }
  async search(query, topK) {
    const params = new URLSearchParams({ q: query });
    if (topK !== void 0) params.set("top_k", String(topK));
    const res = await this.fetchWithRetry(`${this.baseUrl}/api/search?${params}`);
    return res.json();
  }
  async ask(question, topK) {
    const res = await this.fetchWithRetry(`${this.baseUrl}/api/ask`, {
      method: "POST",
      headers: JSON_HEADERS,
      body: JSON.stringify({ question, top_k: topK ?? 0 })
    });
    return res.json();
  }
  async *askStream(question, topK, signal, options) {
    const body = { question, top_k: topK ?? 0 };
    if (options && Object.keys(options).length > 0) body.options = options;
    const res = await this.fetchWithRetry(
      `${this.baseUrl}/api/ask/stream`,
      {
        method: "POST",
        headers: JSON_HEADERS,
        body: JSON.stringify(body)
      },
      { stream: true, signal }
    );
    yield* this.parseSSE(res);
  }
  async chat(question, history, topK) {
    const res = await this.fetchWithRetry(`${this.baseUrl}/api/chat`, {
      method: "POST",
      headers: JSON_HEADERS,
      body: JSON.stringify({ question, history, top_k: topK ?? 0 })
    });
    return res.json();
  }
  async *chatStream(question, history, topK, signal, options) {
    const body = { question, history, top_k: topK ?? 0 };
    if (options && Object.keys(options).length > 0) body.options = options;
    const res = await this.fetchWithRetry(
      `${this.baseUrl}/api/chat/stream`,
      {
        method: "POST",
        headers: JSON_HEADERS,
        body: JSON.stringify(body)
      },
      { stream: true, signal }
    );
    yield* this.parseSSE(res);
  }
  async *addFiles(paths, force = false, visionModel, signal) {
    const body = { paths, force };
    if (visionModel) body.vision_model = visionModel;
    const res = await this.fetchWithRetry(
      `${this.baseUrl}/api/add`,
      {
        method: "POST",
        headers: JSON_HEADERS,
        body: JSON.stringify(body)
      },
      { stream: true, signal }
    );
    yield* this.parseSSE(res);
  }
  async *syncStream(forceVision = false, signal) {
    const res = await this.fetchWithRetry(
      `${this.baseUrl}/api/sync`,
      {
        method: "POST",
        headers: JSON_HEADERS,
        body: JSON.stringify({ force_vision: forceVision })
      },
      { stream: true, signal }
    );
    yield* this.parseSSE(res);
  }
  async listModels() {
    const res = await this.fetchWithRetry(`${this.baseUrl}/api/models`);
    return res.json();
  }
  async *pullModel(model) {
    const res = await this.fetchWithRetry(
      `${this.baseUrl}/api/models/pull`,
      {
        method: "POST",
        headers: JSON_HEADERS,
        body: JSON.stringify({ model })
      },
      { stream: true }
    );
    yield* this.parseSSE(res);
  }
  async setChatModel(model) {
    const res = await this.fetchWithRetry(`${this.baseUrl}/api/models/chat`, {
      method: "PUT",
      headers: JSON_HEADERS,
      body: JSON.stringify({ model })
    });
    return res.json();
  }
  async setVisionModel(model) {
    const res = await this.fetchWithRetry(`${this.baseUrl}/api/models/vision`, {
      method: "PUT",
      headers: JSON_HEADERS,
      body: JSON.stringify({ model })
    });
    return res.json();
  }
  async *parseSSE(response) {
    if (!response.body) {
      throw new Error("Response body is null");
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let currentEvent = SSE_EVENT.MESSAGE;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();
      for (const line of lines) {
        if (line.startsWith("event:")) {
          currentEvent = (line.startsWith("event: ") ? line.slice(7) : line.slice(6)).trim();
        } else if (line.startsWith("data:")) {
          const raw = line.startsWith("data: ") ? line.slice(6) : line.slice(5);
          try {
            yield { event: currentEvent, data: JSON.parse(raw) };
          } catch {
            yield { event: currentEvent, data: raw };
          }
          currentEvent = SSE_EVENT.MESSAGE;
        }
      }
    }
  }
};
var PARAM_KEY_MAP = {
  temperature: "temperature",
  top_p: "top_p",
  top_k: "top_k",
  repeat_penalty: "repeat_penalty",
  num_ctx: "num_ctx",
  seed: "seed"
};
function parseModelParameters(parameters, modelInfo) {
  const defaults = {};
  for (const line of parameters.split("\n")) {
    const parts = line.trim().split(/\s+/);
    if (parts.length < 2) continue;
    const key = PARAM_KEY_MAP[parts[0]];
    if (key) {
      const num = Number(parts[1]);
      if (!isNaN(num)) defaults[key] = num;
    }
  }
  const ctxKey = Object.keys(modelInfo).find((k) => k.endsWith(".context_length"));
  if (ctxKey && !defaults.num_ctx) {
    const val = Number(modelInfo[ctxKey]);
    if (!isNaN(val) && val > 0) defaults.num_ctx = val;
  }
  return defaults;
}
var OllamaClient = class {
  constructor(baseUrl) {
    this.baseUrl = baseUrl;
  }
  async *pull(model, signal) {
    const res = await fetch(`${this.baseUrl}/api/pull`, {
      method: "POST",
      headers: JSON_HEADERS,
      body: JSON.stringify({ name: model, stream: true }),
      signal
    });
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`Ollama responded ${res.status}: ${text}`);
    }
    yield* this.parseNDJSON(res);
  }
  async show(model) {
    const res = await fetch(`${this.baseUrl}/api/show`, {
      method: "POST",
      headers: JSON_HEADERS,
      body: JSON.stringify({ name: model })
    });
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`Ollama responded ${res.status}: ${text}`);
    }
    const data = await res.json();
    return parseModelParameters(data.parameters ?? "", data.model_info ?? {});
  }
  async delete(model) {
    const res = await fetch(`${this.baseUrl}/api/delete`, {
      method: "DELETE",
      headers: JSON_HEADERS,
      body: JSON.stringify({ name: model })
    });
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`Ollama responded ${res.status}: ${text}`);
    }
  }
  async *parseNDJSON(response) {
    if (!response.body) {
      throw new Error("Response body is null");
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        try {
          yield JSON.parse(trimmed);
        } catch {
        }
      }
    }
    if (buffer.trim()) {
      try {
        yield JSON.parse(buffer.trim());
      } catch {
      }
    }
  }
};

// src/binary-manager.ts
var import_obsidian = require("obsidian");
var import_child_process = require("child_process");
var import_fs = require("fs");
var import_path = require("path");
var import_util = require("util");
var execFileAsync = (0, import_util.promisify)(import_child_process.execFile);
var node = { spawn: import_child_process.spawn, execFile: execFileAsync, existsSync: import_fs.existsSync, mkdirSync: import_fs.mkdirSync, chmodSync: import_fs.chmodSync, writeFileSync: import_fs.writeFileSync, readFileSync: import_fs.readFileSync, unlinkSync: import_fs.unlinkSync, requestUrl: import_obsidian.requestUrl, fetch: globalThis.fetch.bind(globalThis) };
var GITHUB_REPO = "tobocop2/lilbee";
var RELEASES_API = `https://api.github.com/repos/${GITHUB_REPO}/releases/latest`;
function getPlatformAssetName() {
  const platform = process.platform;
  const arch = process.arch;
  if (platform === "darwin" && arch === "arm64") return "lilbee-macos-arm64";
  if (platform === "darwin" && arch === "x64") return "lilbee-macos-x86_64";
  if (platform === "linux" && arch === "x64") return "lilbee-linux-x86_64";
  if (platform === "win32" && arch === "x64") return "lilbee-windows-x86_64.exe";
  throw new Error(`Unsupported platform: ${platform}/${arch}`);
}
async function getLatestRelease() {
  const res = await node.requestUrl({
    url: RELEASES_API,
    headers: { Accept: "application/vnd.github.v3+json" }
  });
  if (res.status >= 400) throw new Error(`GitHub API responded ${res.status}`);
  const data = res.json;
  const assetName = getPlatformAssetName();
  const asset = data.assets.find((a) => a.name === assetName);
  if (!asset) throw new Error(`No asset "${assetName}" in release ${data.tag_name}`);
  return { tag: data.tag_name, assetUrl: asset.browser_download_url };
}
function checkForUpdate(currentVersion, latestTag) {
  return currentVersion !== latestTag && latestTag !== "";
}
var BinaryManager = class {
  binDir;
  constructor(pluginDir) {
    this.binDir = (0, import_path.join)(pluginDir, "bin");
  }
  get binaryPath() {
    const name = process.platform === "win32" ? "lilbee.exe" : "lilbee";
    return (0, import_path.join)(this.binDir, name);
  }
  binaryExists() {
    return node.existsSync(this.binaryPath);
  }
  async ensureBinary(onProgress) {
    if (this.binaryExists()) return this.binaryPath;
    onProgress?.("Fetching latest release info...");
    const release = await getLatestRelease();
    await this.download(release.assetUrl, onProgress);
    return this.binaryPath;
  }
  async download(assetUrl, onProgress) {
    if (!node.existsSync(this.binDir)) {
      node.mkdirSync(this.binDir, { recursive: true });
    }
    onProgress?.("Downloading...", assetUrl);
    const res = await node.requestUrl({ url: assetUrl });
    if (res.status >= 400) throw new Error(`Download failed: ${res.status}`);
    const dest = this.binaryPath;
    node.writeFileSync(dest, Buffer.from(res.arrayBuffer));
    if (process.platform !== "win32") {
      node.chmodSync(dest, 493);
    }
    if (process.platform === "darwin") {
      try {
        await node.execFile("xattr", ["-cr", dest]);
      } catch {
      }
    }
    onProgress?.("Download complete.", assetUrl);
  }
};

// src/server-manager.ts
var HEALTH_POLL_INTERVAL_MS = 1e3;
var HEALTH_POLL_MAX_ATTEMPTS = 60;
var STOP_GRACE_MS = 5e3;
var CRASH_RESTART_DELAY_MS = 3e3;
var MAX_CRASH_RESTARTS = 3;
var PORT_FILE_POLL_INTERVAL_MS = 200;
var PORT_FILE_MAX_ATTEMPTS = 150;
var ServerManager = class _ServerManager {
  opts;
  child = null;
  _state = SERVER_STATE.STOPPED;
  crashCount = 0;
  stopping = false;
  restartTimer = null;
  _actualPort = null;
  _stderrLines = [];
  static MAX_STDERR_LINES = 20;
  constructor(opts) {
    this.opts = opts;
  }
  get lastStderr() {
    return this._stderrLines.join("\n");
  }
  get state() {
    return this._state;
  }
  get serverUrl() {
    const port = this._actualPort ?? this.opts.port;
    return `http://127.0.0.1:${port}`;
  }
  get portFilePath() {
    return `${this.opts.dataDir}/server.port`;
  }
  async waitForPortFile() {
    for (let i = 0; i < PORT_FILE_MAX_ATTEMPTS; i++) {
      if (node.existsSync(this.portFilePath)) {
        const content = node.readFileSync(this.portFilePath, "utf-8").trim();
        const port = parseInt(content, 10);
        if (!isNaN(port) && port > 0 && port <= 65535) {
          this._actualPort = port;
          return;
        }
      }
      await new Promise((r) => setTimeout(r, PORT_FILE_POLL_INTERVAL_MS));
    }
    throw new Error("Port file not found within timeout");
  }
  setState(s) {
    this._state = s;
    this.opts.onStateChange?.(s);
  }
  async start() {
    if (this.child) return;
    this.stopping = false;
    this._actualPort = null;
    this.setState(SERVER_STATE.STARTING);
    const args = [
      "serve",
      "--host",
      "127.0.0.1"
    ];
    if (this.opts.port !== null) {
      args.push("--port", String(this.opts.port));
    }
    args.push("--data-dir", this.opts.dataDir);
    const env = {
      ...process.env,
      OLLAMA_HOST: this.opts.ollamaUrl,
      LILBEE_CORS_ORIGINS: "app://obsidian.md"
    };
    this._stderrLines = [];
    this.child = node.spawn(this.opts.binaryPath, args, {
      env,
      stdio: ["ignore", "ignore", "pipe"],
      detached: false
    });
    if (this.child.stderr) {
      let partial = "";
      this.child.stderr.on("data", (chunk) => {
        partial += chunk.toString();
        const lines = partial.split("\n");
        partial = lines.pop();
        for (const line of lines) {
          if (line.length > 0) {
            this._stderrLines.push(line);
            if (this._stderrLines.length > _ServerManager.MAX_STDERR_LINES) {
              this._stderrLines.shift();
            }
          }
        }
      });
    }
    this.child.on("exit", (_code, _signal) => {
      this.child = null;
      if (!this.stopping && this.crashCount < MAX_CRASH_RESTARTS) {
        this.crashCount++;
        this.setState(SERVER_STATE.ERROR);
        this.restartTimer = setTimeout(() => {
          this.restartTimer = null;
          if (!this.stopping) void this.start();
        }, CRASH_RESTART_DELAY_MS);
      } else if (!this.stopping) {
        this.setState(SERVER_STATE.ERROR);
        this.opts.onRestartsExhausted?.(this.lastStderr);
      }
    });
    this.child.on("error", () => {
      this.child = null;
      this.setState(SERVER_STATE.ERROR);
    });
    try {
      if (this.opts.port === null) {
        await this.waitForPortFile();
      } else {
        this._actualPort = this.opts.port;
      }
      await this.waitForReady();
      this.crashCount = 0;
      this.setState(SERVER_STATE.READY);
    } catch {
      this.setState(SERVER_STATE.ERROR);
    }
  }
  async waitForReady() {
    for (let i = 0; i < HEALTH_POLL_MAX_ATTEMPTS; i++) {
      try {
        const res = await node.fetch(`${this.serverUrl}/api/health`);
        if (res.ok) return;
      } catch {
      }
      await new Promise((r) => setTimeout(r, HEALTH_POLL_INTERVAL_MS));
    }
    throw new Error("Server did not become ready within timeout");
  }
  async stop() {
    this.stopping = true;
    if (this.restartTimer) {
      clearTimeout(this.restartTimer);
      this.restartTimer = null;
    }
    if (!this.child) {
      this.setState(SERVER_STATE.STOPPED);
      return;
    }
    const child = this.child;
    if (process.platform === "win32") {
      try {
        await node.execFile("taskkill", ["/pid", String(child.pid), "/f", "/t"]);
      } catch {
      }
    } else {
      child.kill("SIGTERM");
    }
    const exited = await Promise.race([
      new Promise((resolve) => {
        child.on("exit", () => resolve(true));
      }),
      new Promise((resolve) => {
        setTimeout(() => resolve(false), STOP_GRACE_MS);
      })
    ]);
    if (!exited && this.child) {
      this.child.kill("SIGKILL");
    }
    this.child = null;
    this.setState(SERVER_STATE.STOPPED);
    if (node.existsSync(this.portFilePath)) {
      try {
        node.unlinkSync(this.portFilePath);
      } catch {
      }
    }
  }
  async restart() {
    await this.stop();
    this.crashCount = 0;
    await this.start();
  }
  updateOllamaUrl(url) {
    this.opts.ollamaUrl = url;
  }
  updatePort(port) {
    this.opts.port = port;
    if (port !== null) {
      this._actualPort = port;
    }
  }
};

// src/settings.ts
var import_obsidian2 = require("obsidian");
var CHECK_TIMEOUT_MS = 5e3;
var CLS_MODELS_CONTAINER = "lilbee-models-container";
var SEPARATOR_KEY = "__separator__";
var SEPARATOR_LABEL = "\u2500\u2500 Other... \u2500\u2500";
function deduplicateLatest(models) {
  const bases = new Set(
    models.filter((m) => !m.endsWith(":latest")).map((m) => m.split(":")[0])
  );
  return models.filter((m) => {
    if (!m.endsWith(":latest")) return true;
    return !bases.has(m.split(":")[0]);
  });
}
function buildModelOptions(catalog, type) {
  const options = {};
  if (type === "vision") {
    options[""] = "Disabled";
  }
  const catalogNames = new Set(catalog.catalog.map((m) => m.name));
  for (const model of catalog.catalog) {
    const suffix = model.installed ? "" : " (not installed)";
    options[model.name] = `${model.name}${suffix}`;
  }
  const otherInstalled = deduplicateLatest(
    catalog.installed.filter((name) => !catalogNames.has(name))
  ).sort();
  if (otherInstalled.length > 0) {
    options[SEPARATOR_KEY] = SEPARATOR_LABEL;
    for (const name of otherInstalled) {
      options[name] = name;
    }
  }
  return options;
}
var GEN_DEFAULTS_MAP = {
  temperature: "temperature",
  top_p: "top_p",
  top_k_sampling: "top_k",
  repeat_penalty: "repeat_penalty",
  num_ctx: "num_ctx",
  seed: "seed"
};
var LilbeeSettingTab = class extends import_obsidian2.PluginSettingTab {
  plugin;
  pulling = false;
  genInputs = /* @__PURE__ */ new Map();
  constructor(app, plugin) {
    super(app, plugin);
    this.plugin = plugin;
  }
  display() {
    const { containerEl } = this;
    containerEl.empty();
    this.renderConnectionSettings(containerEl);
    this.renderModelsSection(containerEl);
    this.renderGeneralSettings(containerEl);
    this.renderSyncSettings(containerEl);
    this.renderGenerationSettings(containerEl);
    this.loadModelDefaults();
  }
  renderConnectionSettings(containerEl) {
    new import_obsidian2.Setting(containerEl).setName("Server mode").setDesc("How the lilbee server is managed").addDropdown(
      (dropdown) => dropdown.addOption(SERVER_MODE.MANAGED, "Managed (built-in)").addOption(SERVER_MODE.EXTERNAL, "External (manual)").setValue(this.plugin.settings.serverMode).onChange(async (value) => {
        this.plugin.settings.serverMode = value;
        await this.plugin.saveSettings();
        this.display();
      })
    );
    if (this.plugin.settings.serverMode === SERVER_MODE.MANAGED) {
      this.renderManagedSettings(containerEl);
    } else {
      this.renderExternalSettings(containerEl);
    }
    const ollamaSetting = new import_obsidian2.Setting(containerEl).setName("Ollama URL").setDesc("Address of the Ollama server").addText(
      (text) => text.setPlaceholder("http://127.0.0.1:11434").setValue(this.plugin.settings.ollamaUrl).onChange(async (value) => {
        this.plugin.settings.ollamaUrl = value;
        await this.plugin.saveSettings();
      })
    );
    const ollamaStatusEl = ollamaSetting.settingEl.createEl("span", { cls: "lilbee-health-status" });
    ollamaSetting.addButton(
      (btn) => btn.setButtonText("Test").onClick(async () => {
        await this.checkEndpoint(this.plugin.settings.ollamaUrl, ollamaStatusEl);
      })
    );
    void this.checkEndpoint(this.plugin.settings.ollamaUrl, ollamaStatusEl);
  }
  renderManagedSettings(containerEl) {
    const statusSetting = new import_obsidian2.Setting(containerEl).setName("Server status").setDesc("Current state of the managed lilbee server");
    const statusEl = statusSetting.settingEl.createDiv({ cls: "lilbee-server-status" });
    const dot = statusEl.createDiv({ cls: "lilbee-server-dot" });
    const stateText = statusEl.createEl("span");
    const serverState = this.plugin.serverManager?.state ?? "stopped";
    stateText.textContent = serverState;
    dot.classList.add(`is-${serverState}`);
    const controlSetting = new import_obsidian2.Setting(containerEl).setName("Server controls").setDesc("Start, stop, or restart the managed server");
    if (serverState === "stopped" || serverState === "error") {
      controlSetting.addButton(
        (btn) => btn.setButtonText("Start").onClick(async () => {
          await this.plugin.startManagedServer();
          this.display();
        })
      );
    }
    if (serverState === "ready" || serverState === "starting") {
      controlSetting.addButton(
        (btn) => btn.setButtonText("Stop").onClick(async () => {
          await this.plugin.serverManager?.stop();
          this.display();
        })
      );
    }
    if (serverState === "ready") {
      controlSetting.addButton(
        (btn) => btn.setButtonText("Restart").onClick(async () => {
          await this.plugin.serverManager?.restart();
          this.display();
        })
      );
    }
    new import_obsidian2.Setting(containerEl).setName("Server port").setDesc("Port for the managed server. Leave blank for automatic.").addText(
      (text) => text.setPlaceholder("Auto").setValue(this.plugin.settings.serverPort !== null ? String(this.plugin.settings.serverPort) : "").onChange(async (value) => {
        const trimmed = value.trim();
        if (trimmed === "" || trimmed === "0") {
          this.plugin.settings.serverPort = null;
        } else {
          const num = parseInt(trimmed, 10);
          if (!isNaN(num) && num > 0 && num <= 65535) {
            this.plugin.settings.serverPort = num;
          }
        }
        await this.plugin.saveSettings();
      })
    );
    const updateSetting = new import_obsidian2.Setting(containerEl).setName("Server version").setDesc(this.plugin.settings.lilbeeVersion || "Unknown");
    updateSetting.addButton(
      (checkBtn) => checkBtn.setButtonText("Check for updates").onClick(async () => {
        checkBtn.setDisabled(true);
        checkBtn.setButtonText("Checking...");
        try {
          const result = await this.plugin.checkForUpdate();
          if (result.available && result.release) {
            const release = result.release;
            checkBtn.setButtonText(`Update to ${release.tag}`);
            checkBtn.setDisabled(false);
            checkBtn.onClick(async () => {
              checkBtn.setDisabled(true);
              checkBtn.setButtonText("Updating...");
              try {
                await this.plugin.updateServer(release, (msg) => {
                  checkBtn.setButtonText(msg);
                });
                new import_obsidian2.Notice(`lilbee: updated to ${release.tag}`);
                this.display();
              } catch (err) {
                new import_obsidian2.Notice("lilbee: update failed");
                console.error("[lilbee] update failed:", err);
                checkBtn.setButtonText("Check for updates");
                checkBtn.setDisabled(false);
              }
            });
          } else {
            new import_obsidian2.Notice("lilbee: already up to date");
            checkBtn.setButtonText("Check for updates");
            checkBtn.setDisabled(false);
          }
        } catch {
          new import_obsidian2.Notice("lilbee: could not check for updates");
          checkBtn.setButtonText("Check for updates");
          checkBtn.setDisabled(false);
        }
      })
    );
  }
  renderExternalSettings(containerEl) {
    const serverSetting = new import_obsidian2.Setting(containerEl).setName("Server URL").setDesc("Address of the lilbee HTTP server").addText(
      (text) => text.setPlaceholder("http://127.0.0.1:7433").setValue(this.plugin.settings.serverUrl).onChange(async (value) => {
        this.plugin.settings.serverUrl = value;
        await this.plugin.saveSettings();
      })
    );
    const serverStatusEl = serverSetting.settingEl.createEl("span", { cls: "lilbee-health-status" });
    serverSetting.addButton(
      (btn) => btn.setButtonText("Test").onClick(async () => {
        await this.checkEndpoint(`${this.plugin.settings.serverUrl}/api/health`, serverStatusEl);
      })
    );
    void this.checkEndpoint(`${this.plugin.settings.serverUrl}/api/health`, serverStatusEl);
    new import_obsidian2.Setting(containerEl).setName("Switch to managed server").setDesc("Stop using an external server and start the built-in one").addButton(
      (btn) => btn.setButtonText("Reset to managed").onClick(async () => {
        this.plugin.settings.serverMode = SERVER_MODE.MANAGED;
        this.plugin.settings.serverUrl = DEFAULT_SETTINGS.serverUrl;
        await this.plugin.saveSettings();
        this.display();
      })
    );
  }
  renderModelsSection(containerEl) {
    containerEl.createEl("h3", { text: "Models" });
    containerEl.createEl("p", {
      text: "Curated catalog \u2014 see ollama.com/library for the full model list. Requires the lilbee server.",
      cls: "setting-item-description"
    });
    const modelsContainer = containerEl.createDiv(CLS_MODELS_CONTAINER);
    new import_obsidian2.Setting(containerEl).setName("Refresh models").setDesc("Fetch available models from the server").addButton(
      (btn) => btn.setButtonText("Refresh").onClick(async () => {
        await this.loadModels(modelsContainer);
      })
    );
    this.loadModels(modelsContainer);
  }
  renderGeneralSettings(containerEl) {
    new import_obsidian2.Setting(containerEl).setName("Results count").setDesc("Number of search results to return").addSlider(
      (slider) => slider.setLimits(1, 20, 1).setValue(this.plugin.settings.topK).setDynamicTooltip().onChange(async (value) => {
        this.plugin.settings.topK = value;
        await this.plugin.saveSettings();
      })
    );
  }
  renderGenerationSettings(containerEl) {
    const details = containerEl.createEl("details", { cls: "lilbee-generation-details" });
    const modelLabel = this.plugin.activeModel || "no model selected";
    details.createEl("summary", { text: `Advanced settings (${modelLabel})` });
    this.genInputs.clear();
    const fields = [
      { key: "temperature", name: "Temperature", desc: "Controls randomness (0.0\u20132.0)", integer: false },
      { key: "top_p", name: "Top P", desc: "Nucleus sampling threshold (0.0\u20131.0)", integer: false },
      { key: "top_k_sampling", name: "Top K (sampling)", desc: "Limits token choices per step", integer: true },
      { key: "repeat_penalty", name: "Repeat penalty", desc: "Penalizes repeated tokens (1.0+)", integer: false },
      { key: "num_ctx", name: "Context length", desc: "Max context window in tokens", integer: true },
      { key: "seed", name: "Seed", desc: "Fixed seed for reproducible output", integer: true }
    ];
    for (const field of fields) {
      new import_obsidian2.Setting(details).setName(field.name).setDesc(field.desc).addText((text) => {
        text.setPlaceholder("Not set").setValue(this.plugin.settings[field.key] !== null ? String(this.plugin.settings[field.key]) : "").onChange(async (value) => {
          const trimmed = value.trim();
          if (trimmed === "") {
            this.plugin.settings[field.key] = null;
          } else {
            const num = field.integer ? parseInt(trimmed, 10) : parseFloat(trimmed);
            if (!isNaN(num)) {
              this.plugin.settings[field.key] = num;
            }
          }
          await this.plugin.saveSettings();
        });
        this.genInputs.set(field.key, text.inputEl);
      });
    }
  }
  loadModelDefaults() {
    const model = this.plugin.activeModel;
    if (!model) return;
    this.plugin.ollama.show(model).then((defaults) => {
      for (const [key, inputEl] of this.genInputs) {
        const ollamaKey = GEN_DEFAULTS_MAP[key];
        const val = defaults[ollamaKey];
        if (val !== void 0) {
          inputEl.placeholder = String(val);
        }
      }
    }).catch(() => {
    });
  }
  renderSyncSettings(containerEl) {
    new import_obsidian2.Setting(containerEl).setName("Sync mode").setDesc("How vault changes are synced to the knowledge base").addDropdown(
      (dropdown) => dropdown.addOption("manual", "Manual (command only)").addOption("auto", "Auto (watch for changes)").setValue(this.plugin.settings.syncMode).onChange(async (value) => {
        this.plugin.settings.syncMode = value;
        await this.plugin.saveSettings();
        this.display();
      })
    );
    if (this.plugin.settings.syncMode === "auto") {
      new import_obsidian2.Setting(containerEl).setName("Sync debounce").setDesc("Delay in ms before syncing after a change").addText(
        (text) => text.setPlaceholder("5000").setValue(String(this.plugin.settings.syncDebounceMs)).onChange(async (value) => {
          const num = parseInt(value, 10);
          if (!isNaN(num) && num >= 0) {
            this.plugin.settings.syncDebounceMs = num;
            await this.plugin.saveSettings();
          }
        })
      );
    }
  }
  async checkEndpoint(url, statusEl) {
    statusEl.empty();
    statusEl.classList.remove("lilbee-health-ok", "lilbee-health-error");
    const dot = statusEl.createDiv({ cls: "lilbee-health-dot" });
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), CHECK_TIMEOUT_MS);
      const response = await fetch(url, { signal: controller.signal });
      clearTimeout(timeout);
      const ok = response.ok;
      dot.classList.add(ok ? "is-ok" : "is-error");
      statusEl.classList.add(ok ? "lilbee-health-ok" : "lilbee-health-error");
    } catch {
      dot.classList.add("is-error");
      statusEl.classList.add("lilbee-health-error");
    }
  }
  async loadModels(container) {
    container.empty();
    try {
      const models = await this.plugin.api.listModels();
      this.renderModelSection(container, "Chat Model", models.chat, "chat");
      this.renderModelSection(container, "Vision Model", models.vision, "vision");
    } catch {
      container.createEl("p", {
        text: "Could not connect to lilbee server. Is it running?",
        cls: "mod-warning"
      });
    }
  }
  renderModelSection(container, label, catalog, type) {
    const section = container.createDiv("lilbee-model-section");
    section.createEl("h4", { text: label });
    const activeSetting = new import_obsidian2.Setting(section).setName(`Active ${type} model`).setDesc(catalog.active || (type === "vision" ? "Disabled" : "Not set"));
    const options = buildModelOptions(catalog, type);
    activeSetting.addDropdown(
      (dropdown) => dropdown.addOptions(options).setValue(catalog.active).onChange(async (value) => {
        if (value === SEPARATOR_KEY) return;
        await this.handleModelChange(value, catalog, label, type, container);
      })
    );
    const catalogEl = section.createDiv("lilbee-model-catalog");
    const table = catalogEl.createEl("table");
    const header = table.createEl("tr");
    header.createEl("th", { text: "Model" });
    header.createEl("th", { text: "Size" });
    header.createEl("th", { text: "Description" });
    header.createEl("th", { text: "" });
    for (const model of catalog.catalog) {
      this.renderCatalogRow(table, model, type);
    }
  }
  async handleModelChange(value, catalog, label, type, container) {
    const uninstalledCatalogModel = catalog.catalog.find(
      (m) => m.name === value && !m.installed
    );
    if (uninstalledCatalogModel) {
      await this.autoPullAndSet(uninstalledCatalogModel, type, container);
      return;
    }
    try {
      if (type === "chat") {
        await this.plugin.api.setChatModel(value);
      } else {
        await this.plugin.api.setVisionModel(value);
      }
      new import_obsidian2.Notice(`${label} set to ${value || "disabled"}`);
      this.display();
    } catch {
      new import_obsidian2.Notice(`Failed to set ${type} model`);
    }
  }
  async autoPullAndSet(model, type, container) {
    if (this.pulling) return;
    this.pulling = true;
    new import_obsidian2.Notice(`Pulling ${model.name}...`);
    const controller = new AbortController();
    const banner = container.createDiv("lilbee-pull-banner");
    const label = banner.createEl("span", { text: `Pulling ${model.name}...` });
    const cancelBtn = banner.createEl("button", { text: "Cancel", cls: "lilbee-pull-banner-cancel" });
    cancelBtn.addEventListener("click", () => controller.abort(), { once: true });
    try {
      for await (const progress of this.plugin.ollama.pull(
        model.name,
        controller.signal
      )) {
        if (progress.total && progress.completed !== void 0) {
          const pct = Math.round(progress.completed / progress.total * 100);
          label.textContent = `Pulling ${model.name} \u2014 ${pct}%`;
          if (this.plugin.statusBarEl) {
            this.plugin.statusBarEl.setText(
              `lilbee: pulling ${model.name} \u2014 ${pct}%`
            );
          }
        }
      }
      if (type === "chat") {
        await this.plugin.api.setChatModel(model.name);
      } else {
        await this.plugin.api.setVisionModel(model.name);
      }
      new import_obsidian2.Notice(`Model ${model.name} pulled and activated`);
      this.plugin.fetchActiveModel();
      this.display();
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        new import_obsidian2.Notice("Pull cancelled");
      } else {
        new import_obsidian2.Notice(`Failed to pull ${model.name}`);
      }
    } finally {
      banner.remove();
      this.pulling = false;
    }
  }
  renderCatalogRow(table, model, type) {
    const row = table.createEl("tr");
    row.createEl("td", { text: model.name });
    row.createEl("td", { text: `${model.size_gb} GB` });
    row.createEl("td", { text: model.description });
    const actionCell = row.createEl("td");
    if (model.installed) {
      actionCell.createEl("span", { text: "Installed", cls: "lilbee-installed" });
      const deleteBtn = actionCell.createEl("button", { cls: "lilbee-model-delete" });
      (0, import_obsidian2.setIcon)(deleteBtn, "trash-2");
      deleteBtn.setAttribute("aria-label", "Delete model");
      deleteBtn.addEventListener("click", () => this.deleteModel(deleteBtn, model, type));
    } else {
      const btn = actionCell.createEl("button", { text: "Pull" });
      btn.addEventListener("click", () => this.pullModel(btn, actionCell, model, type));
    }
  }
  async pullModel(btn, actionCell, model, type) {
    if (this.pulling) return;
    this.pulling = true;
    const controller = new AbortController();
    btn.textContent = "Cancel";
    btn.addEventListener("click", () => controller.abort(), { once: true });
    const progress = actionCell.createDiv("lilbee-pull-progress");
    try {
      for await (const p of this.plugin.ollama.pull(
        model.name,
        controller.signal
      )) {
        if (p.total && p.completed !== void 0) {
          const pct = Math.round(p.completed / p.total * 100);
          progress.textContent = `${pct}%`;
          if (this.plugin.statusBarEl) {
            this.plugin.statusBarEl.setText(`lilbee: pulling ${model.name} \u2014 ${pct}%`);
          }
        }
      }
      new import_obsidian2.Notice(`Model ${model.name} pulled successfully`);
      if (type === "chat") {
        await this.plugin.api.setChatModel(model.name);
      } else {
        await this.plugin.api.setVisionModel(model.name);
      }
      this.plugin.fetchActiveModel();
      this.display();
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        new import_obsidian2.Notice("Pull cancelled");
      } else {
        new import_obsidian2.Notice(`Failed to pull ${model.name}`);
      }
      btn.disabled = false;
      btn.textContent = "Pull";
    } finally {
      progress.remove();
      this.pulling = false;
    }
  }
  async deleteModel(btn, model, type) {
    btn.disabled = true;
    try {
      await this.plugin.ollama.delete(model.name);
      new import_obsidian2.Notice(`Deleted ${model.name}`);
      if (type === "chat" && model.name === this.plugin.activeModel) {
        await this.plugin.api.setChatModel("");
        this.plugin.activeModel = "";
      } else if (type === "vision" && model.name === this.plugin.activeVisionModel) {
        await this.plugin.api.setVisionModel("");
        this.plugin.activeVisionModel = "";
      }
      this.plugin.fetchActiveModel();
      const modelsContainer = this.containerEl.querySelector(`.${CLS_MODELS_CONTAINER}`);
      if (modelsContainer instanceof HTMLElement) {
        await this.loadModels(modelsContainer);
      }
    } catch {
      new import_obsidian2.Notice(`Failed to delete ${model.name}`);
      btn.disabled = false;
    }
  }
};

// src/views/chat-view.ts
var import_obsidian3 = require("obsidian");

// src/views/results.ts
var MAX_EXCERPT_CHARS = 200;
var MAX_EXCERPTS = 3;
function formatLocation(excerpt) {
  if (excerpt.page_start !== null) {
    return excerpt.page_end !== null && excerpt.page_end !== excerpt.page_start ? `pp. ${excerpt.page_start}\u2013${excerpt.page_end}` : `p. ${excerpt.page_start}`;
  }
  if (excerpt.line_start !== null) {
    return excerpt.line_end !== null && excerpt.line_end !== excerpt.line_start ? `lines ${excerpt.line_start}\u2013${excerpt.line_end}` : `line ${excerpt.line_start}`;
  }
  return null;
}
function truncate(text, maxLen) {
  if (text.length <= maxLen) return text;
  return text.slice(0, maxLen) + "...";
}
function renderDocumentResult(container, result, app) {
  const card = container.createDiv({ cls: "lilbee-document-card" });
  const header = card.createDiv({ cls: "lilbee-document-card-header" });
  const link = header.createEl("a", {
    text: result.source,
    cls: "lilbee-document-source"
  });
  link.addEventListener("click", (e) => {
    e.preventDefault();
    app.workspace.openLinkText(result.source, "");
  });
  header.createEl("span", {
    text: result.content_type,
    cls: "lilbee-content-badge"
  });
  const barContainer = card.createDiv({ cls: "lilbee-relevance-bar-container" });
  const bar = barContainer.createDiv({ cls: "lilbee-relevance-bar" });
  const pct = Math.round(Math.max(0, Math.min(1, result.best_relevance)) * 100);
  bar.style.width = `${pct}%`;
  const excerpts = result.excerpts.slice(0, MAX_EXCERPTS);
  for (const excerpt of excerpts) {
    const excerptEl = card.createDiv({ cls: "lilbee-excerpt" });
    excerptEl.createEl("p", { text: truncate(excerpt.content, MAX_EXCERPT_CHARS) });
    const loc = formatLocation(excerpt);
    if (loc) {
      excerptEl.createEl("span", { text: loc, cls: "lilbee-location" });
    }
  }
}
function renderSourceChip(container, source) {
  const chip = container.createEl("span", { cls: "lilbee-source-chip" });
  let label = source.source;
  const loc = formatLocation(source);
  if (loc) {
    label += ` (${loc})`;
  }
  chip.setText(label);
}

// src/views/chat-view.ts
var electronDialog = {
  /* v8 ignore start -- requires Electron runtime */
  showOpenDialog(opts) {
    const electron = require("electron");
    return electron.remote.dialog.showOpenDialog(opts);
  }
  /* v8 ignore stop */
};
var VIEW_TYPE_CHAT = "lilbee-chat";
function buildGenerationOptions(settings) {
  const opts = {};
  if (settings.temperature != null) opts.temperature = settings.temperature;
  if (settings.top_p != null) opts.top_p = settings.top_p;
  if (settings.top_k_sampling != null) opts.top_k = settings.top_k_sampling;
  if (settings.repeat_penalty != null) opts.repeat_penalty = settings.repeat_penalty;
  if (settings.num_ctx != null) opts.num_ctx = settings.num_ctx;
  if (settings.seed != null) opts.seed = settings.seed;
  return opts;
}
function extractString(data, field) {
  if (typeof data === "object" && data !== null && field in data) {
    return String(data[field]);
  }
  return String(data);
}
var ChatView = class extends import_obsidian3.ItemView {
  plugin;
  history = [];
  messagesEl = null;
  sendBtn = null;
  sending = false;
  streamController = null;
  pullController = null;
  progressCancelBtn = null;
  progressBanner = null;
  progressTopLabel = null;
  progressSubLabel = null;
  progressBar = null;
  chatCatalog = null;
  visionCatalog = null;
  chatSelectEl = null;
  visionSelectEl = null;
  constructor(leaf, plugin) {
    super(leaf);
    this.plugin = plugin;
  }
  getViewType() {
    return VIEW_TYPE_CHAT;
  }
  getDisplayText() {
    return "lilbee Chat";
  }
  getIcon() {
    return "message-circle";
  }
  async onOpen() {
    const container = this.containerEl.children[1];
    container.empty();
    container.addClass("lilbee-chat-container");
    this.createToolbar(container);
    this.createProgressBanner(container);
    this.messagesEl = container.createDiv({ cls: "lilbee-chat-messages" });
    this.createInputArea(container);
    this.plugin.onProgress = (event) => this.handleProgress(event);
  }
  async onClose() {
    this.streamController?.abort();
    this.pullController?.abort();
    if (this.plugin.onProgress) {
      this.plugin.onProgress = null;
    }
  }
  createToolbar(container) {
    const toolbar = container.createDiv({ cls: "lilbee-chat-toolbar" });
    const chatGroup = toolbar.createDiv({ cls: "lilbee-toolbar-group" });
    const chatIcon = chatGroup.createDiv({ cls: "lilbee-toolbar-icon" });
    (0, import_obsidian3.setIcon)(chatIcon, "message-circle");
    chatIcon.setAttribute("title", "Chat model");
    this.chatSelectEl = chatGroup.createEl("select", {
      cls: "lilbee-chat-model-select"
    });
    this.attachChatListener(this.chatSelectEl);
    const visionGroup = toolbar.createDiv({ cls: "lilbee-toolbar-group" });
    const visionIcon = visionGroup.createDiv({ cls: "lilbee-toolbar-icon" });
    (0, import_obsidian3.setIcon)(visionIcon, "eye");
    visionIcon.setAttribute("title", "Vision model");
    this.visionSelectEl = visionGroup.createEl("select", {
      cls: "lilbee-chat-vision-select"
    });
    this.attachVisionListener(this.visionSelectEl);
    this.fetchAndFillSelectors();
    toolbar.createDiv({ cls: "lilbee-toolbar-spacer" });
    const saveBtn = toolbar.createEl("button", { cls: "lilbee-chat-save" });
    (0, import_obsidian3.setIcon)(saveBtn, "save");
    saveBtn.setAttribute("aria-label", "Save to vault");
    saveBtn.addEventListener("click", () => this.saveToVault());
    const clearBtn = toolbar.createEl("button", {
      text: "Clear chat",
      cls: "lilbee-chat-clear"
    });
    clearBtn.addEventListener("click", () => this.clearChat());
  }
  createProgressBanner(container) {
    this.progressBanner = container.createDiv({ cls: "lilbee-progress-banner" });
    this.progressBanner.dataset.hidden = "";
    const row = this.progressBanner.createDiv({ cls: "lilbee-progress-row" });
    this.progressTopLabel = row.createDiv({ cls: "lilbee-progress-top-label" });
    this.progressCancelBtn = row.createEl("button", { cls: "lilbee-progress-cancel" });
    (0, import_obsidian3.setIcon)(this.progressCancelBtn, "x");
    this.progressCancelBtn.setAttribute("aria-label", "Cancel");
    this.progressCancelBtn.addEventListener("click", () => {
      this.pullController?.abort();
      this.plugin.cancelSync();
    });
    const barContainer = this.progressBanner.createDiv({ cls: "lilbee-progress-bar-container" });
    this.progressBar = barContainer.createDiv({ cls: "lilbee-progress-bar" });
    this.progressSubLabel = this.progressBanner.createDiv({ cls: "lilbee-progress-sub-label" });
  }
  createInputArea(container) {
    const inputArea = container.createDiv({ cls: "lilbee-chat-input" });
    const addBtn = inputArea.createEl("button", { cls: "lilbee-chat-add-file" });
    addBtn.setAttribute("aria-label", "Add file");
    (0, import_obsidian3.setIcon)(addBtn, "paperclip");
    addBtn.addEventListener("click", (e) => this.openFilePicker(e));
    const textarea = inputArea.createEl("textarea", {
      placeholder: "Ask something...",
      cls: "lilbee-chat-textarea"
    });
    this.sendBtn = inputArea.createEl("button", {
      text: "Send",
      cls: "lilbee-chat-send"
    });
    const handleSend = () => {
      const text = textarea.value.trim();
      if (!text) return;
      textarea.value = "";
      void this.sendMessage(text);
    };
    this.sendBtn.addEventListener("click", () => {
      if (this.sending) {
        this.streamController?.abort();
      } else {
        handleSend();
      }
    });
    textarea.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    });
  }
  fetchAndFillSelectors() {
    this.plugin.api.listModels().then((models) => {
      this.chatCatalog = models.chat;
      this.visionCatalog = models.vision;
      if (this.chatSelectEl) this.fillSelectOptions(this.chatSelectEl, models.chat, "chat");
      if (this.visionSelectEl) this.fillSelectOptions(this.visionSelectEl, models.vision, "vision");
    }).catch(() => {
      if (this.chatSelectEl) this.chatSelectEl.createEl("option", { text: "(offline)" });
      if (this.visionSelectEl) this.visionSelectEl.createEl("option", { text: "(offline)" });
    });
  }
  fillSelectOptions(selectEl, catalog, type) {
    const options = buildModelOptions(catalog, type);
    for (const [value, label] of Object.entries(options)) {
      const option = selectEl.createEl("option", { text: label });
      option.value = value;
      if (value === SEPARATOR_KEY) {
        option.disabled = true;
      }
      if (value === catalog.active) {
        option.selected = true;
      }
    }
  }
  attachChatListener(el) {
    el.addEventListener("change", () => {
      if (!el.value || el.value === SEPARATOR_KEY) return;
      const uninstalled = this.chatCatalog?.catalog.find(
        (m) => m.name === el.value && !m.installed
      );
      if (uninstalled) {
        this.autoPullAndSetChat(uninstalled);
        return;
      }
      this.plugin.api.setChatModel(el.value).then(() => {
        this.plugin.activeModel = el.value;
        this.plugin.fetchActiveModel();
      }).catch(() => {
        new import_obsidian3.Notice("lilbee: failed to switch model");
      });
    });
  }
  attachVisionListener(el) {
    el.addEventListener("change", () => {
      if (el.value === SEPARATOR_KEY) return;
      const uninstalled = this.visionCatalog?.catalog.find(
        (m) => m.name === el.value && !m.installed
      );
      if (uninstalled) {
        this.autoPullAndSetVision(uninstalled);
        return;
      }
      this.plugin.api.setVisionModel(el.value).then(() => {
        this.plugin.activeVisionModel = el.value;
        this.plugin.fetchActiveModel();
      }).catch(() => {
        new import_obsidian3.Notice("lilbee: failed to switch vision model");
      });
    });
  }
  autoPullAndSetChat(model) {
    this.autoPullAndSet(model, "chat");
  }
  autoPullAndSetVision(model) {
    this.autoPullAndSet(model, "vision");
  }
  autoPullAndSet(model, type) {
    new import_obsidian3.Notice(`Pulling ${model.name}...`);
    this.pullController = new AbortController();
    (async () => {
      try {
        for await (const progress of this.plugin.ollama.pull(
          model.name,
          this.pullController.signal
        )) {
          if (progress.total && progress.completed !== void 0) {
            const pct = Math.round(progress.completed / progress.total * 100);
            this.showPullProgress(
              `Pulling ${model.name} \u2014 ${pct}%`,
              progress.completed,
              progress.total
            );
          }
        }
        this.hideProgress();
        if (type === "chat") {
          await this.plugin.api.setChatModel(model.name);
          this.plugin.activeModel = model.name;
        } else {
          await this.plugin.api.setVisionModel(model.name);
          this.plugin.activeVisionModel = model.name;
        }
        this.plugin.fetchActiveModel();
        new import_obsidian3.Notice(`Model ${model.name} pulled and activated`);
        this.refreshModelSelector();
      } catch (err) {
        if (err instanceof Error && err.name === "AbortError") {
          new import_obsidian3.Notice("Pull cancelled");
        } else {
          new import_obsidian3.Notice(`Failed to pull ${model.name}`);
        }
        this.hideProgress();
      } finally {
        this.pullController = null;
      }
    })();
  }
  refreshModelSelector() {
    if (this.chatSelectEl) this.chatSelectEl.empty();
    if (this.visionSelectEl) this.visionSelectEl.empty();
    this.fetchAndFillSelectors();
  }
  clearChat() {
    this.history = [];
    if (this.messagesEl) this.messagesEl.empty();
  }
  async sendMessage(text) {
    if (!this.messagesEl || this.sending) return;
    this.sending = true;
    this.streamController = new AbortController();
    if (this.sendBtn) this.sendBtn.textContent = "Stop";
    const userBubble = this.messagesEl.createDiv({ cls: "lilbee-chat-message user" });
    userBubble.createEl("p", { text });
    this.history.push({ role: "user", content: text });
    const assistantBubble = this.messagesEl.createDiv({ cls: "lilbee-chat-message assistant" });
    const spinner = assistantBubble.createDiv({ cls: "lilbee-loading" });
    spinner.textContent = "Thinking...";
    const textEl = assistantBubble.createDiv({ cls: "lilbee-chat-content" });
    textEl.style.display = "none";
    this.messagesEl.scrollTop = this.messagesEl.scrollHeight;
    const state = { fullContent: "", sources: [], renderPending: false };
    const revealContent = () => {
      if (spinner.parentElement) spinner.remove();
      textEl.style.display = "";
    };
    const scrollToBottom = () => {
      if (this.messagesEl) this.messagesEl.scrollTop = this.messagesEl.scrollHeight;
    };
    const scheduleRender = () => {
      if (state.renderPending) return;
      state.renderPending = true;
      requestAnimationFrame(() => {
        state.renderPending = false;
        void this.renderMarkdown(textEl, state.fullContent).then(scrollToBottom);
      });
    };
    const genOpts = buildGenerationOptions(this.plugin.settings);
    try {
      for await (const event of this.plugin.api.chatStream(
        text,
        this.history.slice(0, -1),
        this.plugin.settings.topK,
        this.streamController.signal,
        genOpts
      )) {
        this.handleStreamEvent(event, textEl, assistantBubble, state, revealContent, scheduleRender);
      }
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        revealContent();
        if (state.fullContent) {
          void this.renderMarkdown(textEl, state.fullContent + "\n\n*(stopped)*");
          this.history.push({ role: "assistant", content: state.fullContent });
        } else {
          textEl.textContent = "(stopped)";
        }
      } else {
        revealContent();
        textEl.textContent = "Server unavailable \u2014 retries exhausted. Is lilbee running?";
        textEl.addClass("lilbee-chat-error");
      }
    } finally {
      this.sending = false;
      this.streamController = null;
      if (this.sendBtn) {
        this.sendBtn.textContent = "Send";
      }
    }
  }
  handleStreamEvent(event, textEl, assistantBubble, state, revealContent, scheduleRender) {
    switch (event.event) {
      case SSE_EVENT.TOKEN: {
        revealContent();
        state.fullContent += extractString(event.data, "token");
        scheduleRender();
        break;
      }
      case SSE_EVENT.SOURCES:
        state.sources.push(...event.data);
        break;
      case SSE_EVENT.DONE: {
        revealContent();
        void this.renderMarkdown(textEl, state.fullContent);
        if (state.sources.length > 0) this.renderSources(assistantBubble, state.sources);
        this.history.push({ role: "assistant", content: state.fullContent });
        break;
      }
      case SSE_EVENT.ERROR: {
        const errMsg = extractString(event.data, "message");
        revealContent();
        textEl.textContent = errMsg;
        textEl.addClass("lilbee-chat-error");
        new import_obsidian3.Notice(`lilbee: ${errMsg}`);
        break;
      }
    }
  }
  async renderMarkdown(el, markdown) {
    el.empty();
    await import_obsidian3.MarkdownRenderer.render(this.app, markdown, el, "", this.plugin);
    el.addClass("markdown-rendered");
  }
  openFilePicker(event) {
    const menu = new import_obsidian3.Menu();
    menu.addItem((item) => {
      item.setTitle("From vault").setIcon("vault").onClick(() => {
        new VaultFilePickerModal(this.app, this.plugin).open();
      });
    });
    menu.addItem((item) => {
      item.setTitle("Files from disk").setIcon("file-plus").onClick(() => this.openNativeFilePicker(false));
    });
    menu.addItem((item) => {
      item.setTitle("Folder from disk").setIcon("folder-plus").onClick(() => this.openNativeFilePicker(true));
    });
    menu.showAtMouseEvent(event);
  }
  openNativeFilePicker(directory) {
    const properties = directory ? ["openDirectory"] : ["openFile", "multiSelections"];
    electronDialog.showOpenDialog({ properties }).then((result) => {
      if (result.canceled || result.filePaths.length === 0) return;
      void this.plugin.addExternalFiles(result.filePaths);
    }).catch(() => {
      new import_obsidian3.Notice("lilbee: could not open file picker");
    });
  }
  handleProgress(event) {
    const data = event.data;
    switch (event.event) {
      case SSE_EVENT.FILE_START: {
        const fileIndex = Number(data.current_file ?? 0);
        const fileTotal = Number(data.total_files ?? 0);
        this.showFileProgress(`Syncing ${fileIndex}/${fileTotal} files`, fileIndex, fileTotal, "");
        break;
      }
      case SSE_EVENT.EXTRACT: {
        const page = Number(data.page ?? 0);
        const totalPages = Number(data.total_pages ?? 0);
        this.updateSubLabel(`Extracting page ${page}/${totalPages} \u2014 ${data.file ?? ""}`);
        break;
      }
      case SSE_EVENT.EMBED: {
        const chunk = Number(data.chunk ?? 0);
        const totalChunks = Number(data.total_chunks ?? 0);
        this.updateSubLabel(`Embedding chunk ${chunk}/${totalChunks} \u2014 ${data.file ?? ""}`);
        break;
      }
      case SSE_EVENT.PROGRESS: {
        const current = Number(data.current ?? 0);
        const total = Number(data.total ?? 0);
        this.showFileProgress(`Indexing ${current}/${total} \u2014 ${data.file ?? ""}`, current, total, "");
        break;
      }
      case SSE_EVENT.PULL: {
        const current = Number(data.current ?? 0);
        const total = Number(data.total ?? 0);
        const pct = total > 0 ? Math.round(current / total * 100) : 0;
        this.showFileProgress(`Pulling model \u2014 ${pct}%`, current, total, "");
        break;
      }
      case SSE_EVENT.DONE:
        this.hideProgress();
        break;
    }
  }
  showFileProgress(topLabel, current, total, subLabel) {
    if (!this.progressBanner || !this.progressTopLabel || !this.progressBar || !this.progressSubLabel) return;
    delete this.progressBanner.dataset.hidden;
    this.progressTopLabel.textContent = topLabel;
    this.progressBar.style.width = total > 0 ? `${Math.round(current / total * 100)}%` : "0%";
    this.progressSubLabel.textContent = subLabel;
  }
  showPullProgress(label, current, total) {
    this.showFileProgress(label, current, total, "");
  }
  updateSubLabel(text) {
    if (!this.progressSubLabel) return;
    this.progressSubLabel.textContent = text;
  }
  hideProgress() {
    if (!this.progressBanner || !this.progressBar || !this.progressSubLabel) return;
    this.progressBanner.dataset.hidden = "";
    this.progressBar.style.width = "0%";
    if (this.progressTopLabel) this.progressTopLabel.textContent = "";
    this.progressSubLabel.textContent = "";
  }
  async saveToVault() {
    if (this.history.length === 0) {
      new import_obsidian3.Notice("Nothing to save");
      return;
    }
    const now = /* @__PURE__ */ new Date();
    const pad = (n) => String(n).padStart(2, "0");
    const stamp = `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}-${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
    const filename = `chat-${stamp}.md`;
    const folder = "lilbee";
    const path = `${folder}/${filename}`;
    const lines = [`# lilbee Chat \u2014 ${now.toLocaleDateString()}`, ""];
    for (const msg of this.history) {
      const label = msg.role === "user" ? "User" : "Assistant";
      lines.push(`**${label}**: ${msg.content}`, "");
    }
    const content = lines.join("\n");
    try {
      const vault = this.app.vault;
      const existing = vault.getAbstractFileByPath(folder);
      if (!existing) {
        await vault.createFolder(folder);
      }
      await vault.create(path, content);
      new import_obsidian3.Notice(`Saved to ${path}`);
    } catch {
      new import_obsidian3.Notice("Failed to save chat");
    }
  }
  renderSources(container, sources) {
    const sourcesEl = container.createDiv({ cls: "lilbee-chat-sources" });
    const details = sourcesEl.createEl("details");
    details.createEl("summary", { text: "Sources" });
    const chipsEl = details.createDiv({ cls: "lilbee-chat-source-chips" });
    for (const source of sources) {
      renderSourceChip(chipsEl, source);
    }
  }
};
var VaultFilePickerModal = class extends import_obsidian3.FuzzySuggestModal {
  plugin;
  constructor(app, plugin) {
    super(app);
    this.plugin = plugin;
    this.setPlaceholder("Pick a vault file to add to lilbee...");
  }
  getItems() {
    return this.app.vault.getFiles();
  }
  getItemText(item) {
    return item.path;
  }
  onChooseItem(item) {
    void this.plugin.addToLilbee(item);
  }
};

// src/views/search-modal.ts
var import_obsidian4 = require("obsidian");
var SEARCH_DEBOUNCE_MS = 300;
var SearchModal = class extends import_obsidian4.Modal {
  plugin;
  mode;
  debounceTimer = null;
  resultsContainer = null;
  constructor(app, plugin, mode = "search") {
    super(app);
    this.plugin = plugin;
    this.mode = mode;
  }
  onOpen() {
    const { contentEl } = this;
    contentEl.empty();
    contentEl.addClass("lilbee-modal");
    const title = this.mode === "search" ? "Search knowledge base" : "Ask a question";
    contentEl.createEl("h2", { text: title });
    const input = contentEl.createEl("input", {
      type: "text",
      cls: "lilbee-search-input",
      placeholder: this.mode === "search" ? "Type to search..." : "Ask anything..."
    });
    this.resultsContainer = contentEl.createDiv({ cls: "lilbee-modal-results" });
    this.renderEmptyState("Enter a query to begin.");
    if (this.mode === "search") {
      input.addEventListener("input", () => {
        if (this.debounceTimer) clearTimeout(this.debounceTimer);
        this.debounceTimer = setTimeout(() => {
          this.runSearch(input.value.trim());
        }, SEARCH_DEBOUNCE_MS);
      });
    } else {
      input.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && input.value.trim()) {
          this.runAsk(input.value.trim());
        }
      });
    }
    setTimeout(() => input.focus(), 0);
  }
  onClose() {
    if (this.debounceTimer) clearTimeout(this.debounceTimer);
    const { contentEl } = this;
    contentEl.empty();
  }
  renderEmptyState(message) {
    if (!this.resultsContainer) return;
    this.resultsContainer.empty();
    this.resultsContainer.createEl("p", {
      text: message,
      cls: "lilbee-empty-state"
    });
  }
  renderLoading() {
    if (!this.resultsContainer) return;
    this.resultsContainer.empty();
    this.resultsContainer.createDiv({ cls: "lilbee-loading" });
  }
  async runSearch(query) {
    if (!query) {
      this.renderEmptyState("Enter a query to begin.");
      return;
    }
    this.renderLoading();
    try {
      const results = await this.plugin.api.search(
        query,
        this.plugin.settings.topK
      );
      if (!this.resultsContainer) return;
      this.resultsContainer.empty();
      if (results.length === 0) {
        this.renderEmptyState("No results found.");
        return;
      }
      for (const result of results) {
        renderDocumentResult(this.resultsContainer, result, this.app);
      }
    } catch {
      this.renderEmptyState("Error: could not connect to lilbee server.");
    }
  }
  async runAsk(question) {
    this.renderLoading();
    try {
      const response = await this.plugin.api.ask(question, this.plugin.settings.topK);
      if (!this.resultsContainer) return;
      this.resultsContainer.empty();
      this.resultsContainer.createEl("p", {
        text: response.answer,
        cls: "lilbee-ask-answer"
      });
      if (response.sources.length > 0) {
        const sourcesEl = this.resultsContainer.createDiv({ cls: "lilbee-ask-sources" });
        sourcesEl.createEl("span", { text: "Sources: " });
        for (const source of response.sources) {
          renderSourceChip(sourcesEl, source);
        }
      }
    } catch {
      this.renderEmptyState("Error: could not connect to lilbee server.");
    }
  }
};

// src/main.ts
function summarizeSyncResult(done) {
  const parts = [];
  if (done.added.length > 0) parts.push(`${done.added.length} added`);
  if (done.updated.length > 0) parts.push(`${done.updated.length} updated`);
  if (done.removed.length > 0) parts.push(`${done.removed.length} removed`);
  if (done.failed.length > 0) parts.push(`${done.failed.length} failed`);
  return parts.join(", ");
}
var LilbeePlugin = class extends import_obsidian5.Plugin {
  settings = { ...DEFAULT_SETTINGS };
  api = new LilbeeClient(DEFAULT_SETTINGS.serverUrl);
  ollama = new OllamaClient(DEFAULT_SETTINGS.ollamaUrl);
  activeModel = "";
  activeVisionModel = "";
  statusBarEl = null;
  onProgress = null;
  binaryManager = null;
  serverManager = null;
  syncController = null;
  syncTimeout = null;
  autoSyncRefs = [];
  previousServerMode = SERVER_MODE.MANAGED;
  startingServer = false;
  serverStartFailed = false;
  async onload() {
    await this.loadSettings();
    this.ollama = new OllamaClient(this.settings.ollamaUrl);
    this.statusBarEl = this.addStatusBarItem();
    this.registerView(VIEW_TYPE_CHAT, (leaf) => new ChatView(leaf, this));
    this.addSettingTab(new LilbeeSettingTab(this.app, this));
    this.registerCommands();
    this.registerEvent(
      this.app.workspace.on("file-menu", (menu, file) => {
        menu.addItem((item) => {
          item.setTitle("Add to lilbee").setIcon("plus-circle").onClick(() => this.addToLilbee(file));
        });
      })
    );
    if (this.settings.serverMode === SERVER_MODE.MANAGED) {
      void this.startManagedServer();
    } else {
      this.api = new LilbeeClient(this.settings.serverUrl);
      this.setStatusReady();
      this.fetchActiveModel();
    }
    if (this.settings.syncMode === "auto") {
      this.registerAutoSync();
    }
  }
  async startManagedServer() {
    if (this.startingServer) return;
    this.startingServer = true;
    this.serverStartFailed = false;
    try {
      const pluginDir = this.getPluginDir();
      this.binaryManager = new BinaryManager(pluginDir);
      const needsDownload = !this.binaryManager.binaryExists();
      if (needsDownload) {
        this.updateStatusBar("lilbee: downloading...");
        this.statusBarEl?.classList.add("lilbee-status-downloading");
      }
      let binaryPath;
      let downloadNotice = null;
      try {
        binaryPath = await this.binaryManager.ensureBinary((msg, url) => {
          this.updateStatusBar(`lilbee: ${msg}`);
          if (!downloadNotice && needsDownload) {
            const text = url ? `lilbee: ${msg}
${url}` : `lilbee: ${msg}`;
            downloadNotice = new import_obsidian5.Notice(text, 0);
          } else if (downloadNotice) {
            const text = url ? `lilbee: ${msg}
${url}` : `lilbee: ${msg}`;
            downloadNotice.setMessage(text);
          }
        });
        downloadNotice?.hide();
      } catch (err) {
        downloadNotice?.hide();
        this.showError("failed to download server", err);
        return;
      } finally {
        this.statusBarEl?.classList.remove("lilbee-status-downloading");
      }
      if (needsDownload && !this.settings.lilbeeVersion) {
        try {
          const release = await getLatestRelease();
          this.settings.lilbeeVersion = release.tag;
          await this.saveData(this.settings);
        } catch {
        }
      }
      try {
        this.serverManager = new ServerManager({
          binaryPath,
          dataDir: `${pluginDir}/server-data`,
          port: this.settings.serverPort,
          ollamaUrl: this.settings.ollamaUrl,
          onStateChange: (state) => this.handleServerStateChange(state),
          onRestartsExhausted: (stderr) => {
            if (this.serverStartFailed) return;
            const detail = stderr ? `
${stderr.split("\n").slice(-5).join("\n")}` : "";
            new import_obsidian5.Notice(`lilbee: server crashed after multiple restarts${detail}`, 0);
          }
        });
        this.updateStatusBar("lilbee: starting...");
        await this.serverManager.start();
        this.api = new LilbeeClient(this.serverManager.serverUrl);
        this.fetchActiveModel();
      } catch (err) {
        this.showError("failed to start server", err);
      }
    } finally {
      this.startingServer = false;
    }
  }
  async checkForUpdate() {
    const release = await getLatestRelease();
    if (checkForUpdate(this.settings.lilbeeVersion, release.tag)) {
      return { available: true, release };
    }
    return { available: false };
  }
  async updateServer(release, onProgress) {
    const pluginDir = this.getPluginDir();
    if (!this.binaryManager) {
      this.binaryManager = new BinaryManager(pluginDir);
    }
    if (this.serverManager) {
      onProgress?.("Stopping server...");
      await this.serverManager.stop();
      this.serverManager = null;
    }
    onProgress?.("Downloading...");
    await this.binaryManager.download(release.assetUrl, onProgress);
    this.settings.lilbeeVersion = release.tag;
    await this.saveData(this.settings);
    if (this.settings.serverMode === SERVER_MODE.MANAGED) {
      onProgress?.("Starting server...");
      await this.startManagedServer();
    }
    onProgress?.("Update complete.");
  }
  showError(label, err) {
    console.error(`[lilbee] ${label}:`, err);
    const stderr = this.serverManager?.lastStderr;
    if (stderr) console.error(`[lilbee] server stderr:
${stderr}`);
    const detail = err instanceof Error ? err.message : String(err);
    const stderrTail = stderr ? `
${stderr.split("\n").slice(-5).join("\n")}` : "";
    new import_obsidian5.Notice(`lilbee: ${label} \u2014 ${detail}${stderrTail}`, 8e3);
    this.updateStatusBar("lilbee: error");
    this.serverStartFailed = true;
  }
  handleServerStateChange(state) {
    switch (state) {
      case "ready":
        if (this.serverManager) {
          this.api = new LilbeeClient(this.serverManager.serverUrl);
        }
        this.setStatusReady();
        new import_obsidian5.Notice("lilbee: server ready", 3e3);
        break;
      case "starting":
        this.updateStatusBar("lilbee: starting...");
        break;
      case "error":
        this.updateStatusBar("lilbee: error");
        break;
      case "stopped":
        this.updateStatusBar("lilbee: stopped");
        break;
    }
  }
  getPluginDir() {
    const adapter = this.app.vault.adapter;
    return `${adapter.getBasePath()}/.obsidian/plugins/lilbee`;
  }
  registerCommands() {
    this.addCommand({
      id: "lilbee:search",
      name: "Search knowledge base",
      callback: () => new SearchModal(this.app, this).open()
    });
    this.addCommand({
      id: "lilbee:ask",
      name: "Ask a question",
      callback: () => new SearchModal(this.app, this, "ask").open()
    });
    this.addCommand({
      id: "lilbee:chat",
      name: "Open chat",
      callback: () => this.activateChatView()
    });
    this.addCommand({
      id: "lilbee:add-file",
      name: "Add current file to lilbee",
      checkCallback: (checking) => {
        const file = this.app.workspace.getActiveFile();
        if (!file) return false;
        if (!checking) void this.addToLilbee(file);
        return true;
      }
    });
    this.addCommand({
      id: "lilbee:add-folder",
      name: "Add current folder to lilbee",
      checkCallback: (checking) => {
        const file = this.app.workspace.getActiveFile();
        const folder = file?.parent;
        if (!folder) return false;
        if (!checking) void this.addToLilbee(folder);
        return true;
      }
    });
    this.addCommand({
      id: "lilbee:sync",
      name: "Sync vault",
      callback: () => this.triggerSync()
    });
    this.addCommand({
      id: "lilbee:status",
      name: "Show status",
      callback: async () => {
        try {
          const status = await this.api.status();
          new import_obsidian5.Notice(
            `lilbee: ${status.sources.length} documents, ${status.total_chunks} chunks`
          );
        } catch {
          new import_obsidian5.Notice("lilbee: cannot connect to server");
        }
      }
    });
  }
  onunload() {
    if (this.syncTimeout) {
      clearTimeout(this.syncTimeout);
    }
    void this.serverManager?.stop();
  }
  async loadSettings() {
    this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
    this.previousServerMode = this.settings.serverMode;
  }
  async saveSettings() {
    const previousMode = this.previousServerMode;
    this.previousServerMode = this.settings.serverMode;
    await this.saveData(this.settings);
    if (this.settings.serverMode === SERVER_MODE.MANAGED) {
      if (previousMode !== SERVER_MODE.MANAGED) {
        void this.startManagedServer();
      } else if (this.serverManager) {
        this.serverManager.updateOllamaUrl(this.settings.ollamaUrl);
        this.serverManager.updatePort(this.settings.serverPort);
        this.api = new LilbeeClient(this.serverManager.serverUrl);
      }
    } else {
      if (previousMode === SERVER_MODE.MANAGED) {
        void this.serverManager?.stop();
        this.serverManager = null;
        this.binaryManager = null;
      }
      this.api = new LilbeeClient(this.settings.serverUrl);
    }
    this.ollama = new OllamaClient(this.settings.ollamaUrl);
    this.updateAutoSync();
  }
  updateStatusBar(text) {
    if (!this.statusBarEl) return;
    const model = this.activeModel ? ` (${this.activeModel})` : "";
    this.statusBarEl.setText(`${text}${model}`);
  }
  setStatusReady() {
    const suffix = this.settings.serverMode === SERVER_MODE.EXTERNAL ? " [external]" : "";
    this.updateStatusBar(`lilbee: ready${suffix}`);
  }
  fetchActiveModel() {
    this.api.listModels().then((models) => {
      this.activeModel = models.chat.active;
      this.activeVisionModel = models.vision.active;
      this.setStatusReady();
    }).catch(() => {
    });
  }
  async addExternalFiles(paths) {
    if (!this.statusBarEl || paths.length === 0) return;
    const label = paths.length === 1 ? paths[0].split("/").pop() : `${paths.length} files`;
    new import_obsidian5.Notice(`lilbee: adding ${label}...`);
    await this.runAdd(paths);
  }
  async addToLilbee(file) {
    if (!this.statusBarEl) return;
    const adapter = this.app.vault.adapter;
    const absolutePath = `${adapter.getBasePath()}/${file.path}`;
    new import_obsidian5.Notice(`lilbee: adding ${file.name ?? file.path}...`);
    await this.runAdd([absolutePath]);
  }
  emitProgress(event) {
    if (this.onProgress) this.onProgress(event);
  }
  cancelSync() {
    this.syncController?.abort();
    this.syncController = null;
  }
  async runAdd(paths) {
    this.updateStatusBar("lilbee: adding...");
    this.syncController = new AbortController();
    try {
      let lastEvent = null;
      for await (const event of this.api.addFiles(paths, false, this.activeVisionModel || void 0, this.syncController.signal)) {
        this.emitProgress(event);
        if (event.event === SSE_EVENT.FILE_START) {
          const d = event.data;
          this.updateStatusBar(`lilbee: adding ${d.current_file}/${d.total_files}`);
        }
        lastEvent = event;
      }
      if (lastEvent?.event === SSE_EVENT.DONE) {
        this.emitProgress(lastEvent);
        const summary = summarizeSyncResult(lastEvent.data);
        new import_obsidian5.Notice(summary ? `lilbee: ${summary}` : "lilbee: nothing new to add");
      }
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        new import_obsidian5.Notice("lilbee: add cancelled");
      } else {
        console.error("[lilbee] add failed:", err);
        const msg = err instanceof Error ? err.message : "cannot connect to server";
        new import_obsidian5.Notice(`lilbee: add failed \u2014 ${msg}`);
      }
    } finally {
      this.syncController = null;
      this.setStatusReady();
    }
  }
  updateAutoSync() {
    if (this.settings.syncMode === "auto" && this.autoSyncRefs.length === 0) {
      this.registerAutoSync();
    } else if (this.settings.syncMode === "manual" && this.autoSyncRefs.length > 0) {
      this.unregisterAutoSync();
    }
  }
  unregisterAutoSync() {
    this.autoSyncRefs = [];
  }
  async activateChatView() {
    const existing = this.app.workspace.getLeavesOfType(VIEW_TYPE_CHAT);
    if (existing.length > 0) {
      this.app.workspace.revealLeaf(existing[0]);
      return;
    }
    const leaf = this.app.workspace.getRightLeaf(false);
    if (leaf) {
      await leaf.setViewState({ type: VIEW_TYPE_CHAT, active: true });
      this.app.workspace.revealLeaf(leaf);
    }
  }
  registerAutoSync() {
    const handler = () => this.debouncedSync();
    const vault = this.app.vault;
    const refs = [
      vault.on("create", handler),
      vault.on("modify", handler),
      vault.on("delete", handler),
      vault.on("rename", handler)
    ];
    for (const ref of refs) {
      this.autoSyncRefs.push(ref);
      this.registerEvent(ref);
    }
  }
  debouncedSync() {
    if (this.syncTimeout) {
      clearTimeout(this.syncTimeout);
    }
    this.syncTimeout = setTimeout(() => {
      this.triggerSync();
    }, this.settings.syncDebounceMs);
  }
  async triggerSync() {
    if (!this.statusBarEl) return;
    this.updateStatusBar("lilbee: syncing...");
    this.syncController = new AbortController();
    try {
      let lastEvent = null;
      for await (const event of this.api.syncStream(!!this.activeVisionModel, this.syncController.signal)) {
        this.emitProgress(event);
        if (event.event === SSE_EVENT.FILE_START) {
          const d = event.data;
          this.updateStatusBar(`lilbee: syncing ${d.current_file}/${d.total_files}`);
        }
        lastEvent = event;
      }
      if (lastEvent?.event === SSE_EVENT.DONE) {
        this.emitProgress(lastEvent);
        const summary = summarizeSyncResult(lastEvent.data);
        if (summary) new import_obsidian5.Notice(`lilbee: synced \u2014 ${summary}`);
      }
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        new import_obsidian5.Notice("lilbee: sync cancelled");
      } else {
        console.error("[lilbee] sync failed:", err);
        new import_obsidian5.Notice("lilbee: sync failed \u2014 cannot connect to server");
      }
    } finally {
      this.syncController = null;
      this.setStatusReady();
    }
  }
};
