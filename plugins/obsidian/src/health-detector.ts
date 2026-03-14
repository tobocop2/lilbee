export type HealthState = "unknown" | "reachable" | "unreachable";

export const HEALTH_STATE = {
    UNKNOWN: "unknown",
    REACHABLE: "reachable",
    UNREACHABLE: "unreachable",
} as const satisfies Record<string, HealthState>;

const CHECK_TIMEOUT_MS = 5000;
const DEFAULT_INTERVAL_MS = 30000;
const FAILURES_BEFORE_UNREACHABLE = 1;

export interface HealthDetectorOpts {
    url: string;
    onStateChange: (state: HealthState) => void;
    intervalMs?: number;
    failThreshold?: number;
}

export class HealthDetector {
    private _state: HealthState = HEALTH_STATE.UNKNOWN;
    private consecutiveFailures = 0;
    private intervalId: ReturnType<typeof setInterval> | null = null;
    private readonly url: string;
    private readonly onStateChange: (state: HealthState) => void;
    private readonly intervalMs: number;
    private readonly failThreshold: number;

    constructor(opts: HealthDetectorOpts) {
        this.url = opts.url;
        this.onStateChange = opts.onStateChange;
        this.intervalMs = opts.intervalMs ?? DEFAULT_INTERVAL_MS;
        this.failThreshold = opts.failThreshold ?? FAILURES_BEFORE_UNREACHABLE;
    }

    get state(): HealthState {
        return this._state;
    }

    async check(): Promise<HealthState> {
        let ok = false;
        try {
            const controller = new AbortController();
            const timeout = setTimeout(() => controller.abort(), CHECK_TIMEOUT_MS);
            const response = await fetch(this.url, { signal: controller.signal });
            clearTimeout(timeout);
            ok = response.ok;
        } catch {
            ok = false;
        }

        if (ok) {
            this.consecutiveFailures = 0;
        } else {
            this.consecutiveFailures++;
        }

        const newState = ok
            ? HEALTH_STATE.REACHABLE
            : this.consecutiveFailures >= this.failThreshold
                ? HEALTH_STATE.UNREACHABLE
                : this._state;

        if (newState !== this._state) {
            this._state = newState;
            this.onStateChange(newState);
        }

        return newState;
    }

    startPolling(): void {
        if (this.intervalId !== null) return;
        this.check();
        this.intervalId = setInterval(() => this.check(), this.intervalMs);
    }

    stopPolling(): void {
        if (this.intervalId !== null) {
            clearInterval(this.intervalId);
            this.intervalId = null;
        }
    }
}
