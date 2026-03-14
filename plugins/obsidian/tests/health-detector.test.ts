import { vi, describe, it, expect, beforeEach, afterEach } from "vitest";
import { HealthDetector, type HealthState } from "../src/health-detector";

describe("HealthDetector", () => {
    let onStateChange: ReturnType<typeof vi.fn>;

    beforeEach(() => {
        onStateChange = vi.fn();
        vi.stubGlobal("fetch", vi.fn());
    });

    afterEach(() => {
        vi.useRealTimers();
        vi.unstubAllGlobals();
    });

    function makeDetector(opts?: { intervalMs?: number; url?: string }): HealthDetector {
        return new HealthDetector({
            url: opts?.url ?? "http://127.0.0.1:11434",
            onStateChange,
            intervalMs: opts?.intervalMs,
        });
    }

    describe("check()", () => {
        it("returns 'reachable' on 200 OK", async () => {
            (fetch as ReturnType<typeof vi.fn>).mockResolvedValue({ ok: true });
            const detector = makeDetector();
            const state = await detector.check();
            expect(state).toBe("reachable");
            expect(detector.state).toBe("reachable");
        });

        it("returns 'unreachable' on fetch throw (connection refused)", async () => {
            (fetch as ReturnType<typeof vi.fn>).mockRejectedValue(new Error("connection refused"));
            const detector = makeDetector();
            const state = await detector.check();
            expect(state).toBe("unreachable");
            expect(detector.state).toBe("unreachable");
        });

        it("returns 'unreachable' on non-200 response", async () => {
            (fetch as ReturnType<typeof vi.fn>).mockResolvedValue({ ok: false, status: 500 });
            const detector = makeDetector();
            const state = await detector.check();
            expect(state).toBe("unreachable");
        });

        it("returns 'unreachable' on AbortController timeout", async () => {
            vi.useFakeTimers();
            (fetch as ReturnType<typeof vi.fn>).mockImplementation(
                (_url: string, opts: { signal: AbortSignal }) =>
                    new Promise((_resolve, reject) => {
                        opts.signal.addEventListener("abort", () => reject(new DOMException("aborted")));
                    }),
            );
            const detector = makeDetector();
            const checkPromise = detector.check();
            vi.advanceTimersByTime(5000);
            const state = await checkPromise;
            expect(state).toBe("unreachable");
        });
    });

    describe("onStateChange", () => {
        it("fires on transition from unknown to reachable", async () => {
            (fetch as ReturnType<typeof vi.fn>).mockResolvedValue({ ok: true });
            const detector = makeDetector();
            await detector.check();
            expect(onStateChange).toHaveBeenCalledWith("reachable");
        });

        it("does NOT fire when state is unchanged", async () => {
            (fetch as ReturnType<typeof vi.fn>).mockResolvedValue({ ok: true });
            const detector = makeDetector();
            await detector.check();
            onStateChange.mockClear();
            await detector.check();
            expect(onStateChange).not.toHaveBeenCalled();
        });

        it("stays unknown until failThreshold is reached with custom threshold", async () => {
            (fetch as ReturnType<typeof vi.fn>).mockRejectedValue(new Error("down"));
            const detector = new HealthDetector({
                url: "http://127.0.0.1:11434",
                onStateChange,
                failThreshold: 3,
            });
            await detector.check();
            expect(detector.state).toBe("unknown");
            expect(onStateChange).not.toHaveBeenCalled();

            await detector.check();
            expect(detector.state).toBe("unknown");

            await detector.check();
            expect(detector.state).toBe("unreachable");
            expect(onStateChange).toHaveBeenCalledWith("unreachable");
        });

        it("fires on recovery from unreachable to reachable", async () => {
            const detector = makeDetector();

            (fetch as ReturnType<typeof vi.fn>).mockRejectedValue(new Error("down"));
            await detector.check();
            expect(onStateChange).toHaveBeenCalledWith("unreachable");

            onStateChange.mockClear();
            (fetch as ReturnType<typeof vi.fn>).mockResolvedValue({ ok: true });
            await detector.check();
            expect(onStateChange).toHaveBeenCalledWith("reachable");
        });
    });

    describe("state getter", () => {
        it("returns 'unknown' initially", () => {
            const detector = makeDetector();
            expect(detector.state).toBe("unknown");
        });

        it("returns current state after check", async () => {
            (fetch as ReturnType<typeof vi.fn>).mockResolvedValue({ ok: true });
            const detector = makeDetector();
            await detector.check();
            expect(detector.state).toBe("reachable");
        });
    });

    describe("startPolling()", () => {
        it("runs immediate check + sets interval", async () => {
            vi.useFakeTimers();
            (fetch as ReturnType<typeof vi.fn>).mockResolvedValue({ ok: true });
            const detector = makeDetector({ intervalMs: 1000 });

            detector.startPolling();
            // Immediate check
            expect(fetch).toHaveBeenCalledTimes(1);

            // After interval
            vi.advanceTimersByTime(1000);
            expect(fetch).toHaveBeenCalledTimes(2);

            detector.stopPolling();
        });

        it("is idempotent — calling twice does not create a second interval", async () => {
            vi.useFakeTimers();
            (fetch as ReturnType<typeof vi.fn>).mockResolvedValue({ ok: true });
            const detector = makeDetector({ intervalMs: 1000 });

            detector.startPolling();
            detector.startPolling();

            // Only one immediate check, not two
            expect(fetch).toHaveBeenCalledTimes(1);

            vi.advanceTimersByTime(1000);
            // Only one interval tick, not two
            expect(fetch).toHaveBeenCalledTimes(2);

            detector.stopPolling();
        });
    });

    describe("stopPolling()", () => {
        it("clears interval so no more checks run", async () => {
            vi.useFakeTimers();
            (fetch as ReturnType<typeof vi.fn>).mockResolvedValue({ ok: true });
            const detector = makeDetector({ intervalMs: 1000 });

            detector.startPolling();
            expect(fetch).toHaveBeenCalledTimes(1);

            detector.stopPolling();
            vi.advanceTimersByTime(5000);
            // No additional calls after stop
            expect(fetch).toHaveBeenCalledTimes(1);
        });

        it("is safe to call when not polling", () => {
            const detector = makeDetector();
            expect(() => detector.stopPolling()).not.toThrow();
        });
    });
});
