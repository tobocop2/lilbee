export type ServerState = "stopped" | "starting" | "ready" | "error";

const PORT_BASE = 7433;
const PORT_RANGE = 500;
const HEALTH_ATTEMPTS = 5;
const HEALTH_INITIAL_MS = 200;
const SHUTDOWN_TIMEOUT_MS = 5000;

export function vaultPort(vaultPath: string): number {
    let hash = 0;
    for (const ch of vaultPath) hash = ((hash << 5) - hash + ch.charCodeAt(0)) | 0;
    return PORT_BASE + (Math.abs(hash) % PORT_RANGE);
}

export interface ServerManagerOpts {
    binaryPath: string;
    dataDir: string;
    host: string;
    port: number;
    onStateChange: (state: ServerState, detail?: string) => void;
}

/** Lazy-loaded Node builtins (esbuild externalizes them). */
export const node = {
    fs: () => require("fs") as typeof import("fs"),
    cp: () => require("child_process") as typeof import("child_process"),
    path: () => require("path") as typeof import("path"),
};

export function findBinary(configured: string): string {
    if (configured) return configured;
    try {
        return node.cp().execSync("which lilbee", { encoding: "utf-8" }).trim();
    } catch {
        throw new Error("lilbee not found. Install it or set the path in settings.");
    }
}

export function ensureDataDir(dataDir: string): void {
    const fs = node.fs();
    const path = node.path();
    fs.mkdirSync(path.join(dataDir, "documents"), { recursive: true });
    fs.mkdirSync(path.join(dataDir, "data"), { recursive: true });
    const gitignorePath = path.join(dataDir, ".gitignore");
    if (!fs.existsSync(gitignorePath)) {
        fs.writeFileSync(gitignorePath, "data/\n");
    }
}

export class ServerManager {
    private _state: ServerState = "stopped";
    private process: ReturnType<typeof import("child_process").spawn> | null = null;
    private stderr = "";
    private opts: ServerManagerOpts;

    constructor(opts: ServerManagerOpts) {
        this.opts = opts;
    }

    get state(): ServerState {
        return this._state;
    }

    private setState(state: ServerState, detail?: string): void {
        this._state = state;
        this.opts.onStateChange(state, detail);
    }

    async start(): Promise<void> {
        if (this._state === "starting" || this._state === "ready") return;

        const binary = findBinary(this.opts.binaryPath);
        ensureDataDir(this.opts.dataDir);
        this.setState("starting");
        this.stderr = "";

        this.process = node.cp().spawn(binary, [
            "serve",
            "--data-dir", this.opts.dataDir,
            "--host", this.opts.host,
            "--port", String(this.opts.port),
        ]);

        this.process.stderr?.on("data", (chunk: Buffer) => {
            this.stderr += chunk.toString();
        });

        this.process.on("close", (code: number | null) => {
            if (this._state === "starting" || this._state === "ready") {
                const detail = this.stderr.slice(0, 200) || `exited with code ${code}`;
                this.setState("error", detail);
            }
            this.process = null;
        });

        await this.pollHealth();
    }

    async stop(): Promise<void> {
        if (!this.process) return;
        const proc = this.process;
        this.process = null;

        proc.kill("SIGTERM");

        const exited = await Promise.race([
            new Promise<boolean>((resolve) => proc.on("close", () => resolve(true))),
            new Promise<boolean>((resolve) => setTimeout(() => resolve(false), SHUTDOWN_TIMEOUT_MS)),
        ]);

        if (!exited) {
            proc.kill("SIGKILL");
        }

        this.setState("stopped");
    }

    async restart(): Promise<void> {
        await this.stop();
        await this.start();
    }

    private async pollHealth(): Promise<void> {
        let delay = HEALTH_INITIAL_MS;
        for (let i = 0; i < HEALTH_ATTEMPTS; i++) {
            await new Promise((r) => setTimeout(r, delay));
            try {
                const res = await fetch(
                    `http://${this.opts.host}:${this.opts.port}/api/health`,
                );
                if (res.ok) {
                    if (this._state === "starting") {
                        this.setState("ready");
                    }
                    return;
                }
            } catch {
                // server not up yet
            }
            delay *= 2;
        }
        if (this._state === "starting") {
            this.setState("error", "server did not start in time");
        }
    }
}
