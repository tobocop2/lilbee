import { vi, describe, it, expect, beforeEach, afterEach } from "vitest";
import { EventEmitter } from "events";

import { vaultPort, findBinary, ensureDataDir, ServerManager, node } from "../src/server-manager";
import type { ServerManagerOpts } from "../src/server-manager";

let fetchMock: ReturnType<typeof vi.fn>;

const mockFs = {
    mkdirSync: vi.fn(),
    existsSync: vi.fn(),
    writeFileSync: vi.fn(),
};

const mockExecSync = vi.fn();
const mockSpawnFn = vi.fn();

function makeOpts(overrides: Partial<ServerManagerOpts> = {}): ServerManagerOpts {
    return {
        binaryPath: "/usr/bin/lilbee",
        dataDir: "/vault/.lilbee",
        host: "127.0.0.1",
        port: 7500,
        onStateChange: vi.fn(),
        ...overrides,
    };
}

interface MockProcess extends EventEmitter {
    stderr: EventEmitter;
    kill: ReturnType<typeof vi.fn>;
}

function createMockProcess(): MockProcess {
    const proc = new EventEmitter() as MockProcess;
    proc.stderr = new EventEmitter();
    proc.kill = vi.fn();
    mockSpawnFn.mockReturnValue(proc);
    return proc;
}

beforeEach(() => {
    vi.clearAllMocks();
    fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    // Stub the lazy-load accessors
    vi.spyOn(node, "fs").mockReturnValue(mockFs as any);
    vi.spyOn(node, "cp").mockReturnValue({ execSync: mockExecSync, spawn: mockSpawnFn } as any);
    vi.spyOn(node, "path").mockReturnValue({ join: (...parts: string[]) => parts.join("/") } as any);
});

afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
    vi.useRealTimers();
});

describe("node accessors", () => {
    it("fs() returns the fs module", () => {
        vi.restoreAllMocks();
        const fs = node.fs();
        expect(fs.existsSync).toBeTypeOf("function");
    });

    it("cp() returns the child_process module", () => {
        vi.restoreAllMocks();
        const cp = node.cp();
        expect(cp.execSync).toBeTypeOf("function");
    });

    it("path() returns the path module", () => {
        vi.restoreAllMocks();
        const p = node.path();
        expect(p.join).toBeTypeOf("function");
    });
});

describe("vaultPort()", () => {
    it("returns a deterministic port in the 7433–7932 range", () => {
        const port = vaultPort("/Users/tobias/vault");
        expect(port).toBeGreaterThanOrEqual(7433);
        expect(port).toBeLessThanOrEqual(7932);
    });

    it("returns the same port for the same path across calls", () => {
        expect(vaultPort("/my/vault")).toBe(vaultPort("/my/vault"));
    });

    it("returns different ports for different paths", () => {
        expect(vaultPort("/vault/a")).not.toBe(vaultPort("/vault/b"));
    });
});

describe("findBinary()", () => {
    it("returns configured path when provided", () => {
        expect(findBinary("/custom/lilbee")).toBe("/custom/lilbee");
    });

    it("auto-detects via 'which lilbee'", () => {
        mockExecSync.mockReturnValue("/usr/local/bin/lilbee\n");
        expect(findBinary("")).toBe("/usr/local/bin/lilbee");
    });

    it("throws when not found", () => {
        mockExecSync.mockImplementation(() => {
            throw new Error("not found");
        });
        expect(() => findBinary("")).toThrow("lilbee not found");
    });
});

describe("ensureDataDir()", () => {
    it("creates directories and writes .gitignore", () => {
        mockFs.existsSync.mockReturnValue(false);
        ensureDataDir("/vault/.lilbee");
        expect(mockFs.mkdirSync).toHaveBeenCalledWith("/vault/.lilbee/documents", { recursive: true });
        expect(mockFs.mkdirSync).toHaveBeenCalledWith("/vault/.lilbee/data", { recursive: true });
        expect(mockFs.writeFileSync).toHaveBeenCalledWith("/vault/.lilbee/.gitignore", "data/\n");
    });

    it("skips .gitignore if it already exists", () => {
        mockFs.existsSync.mockReturnValue(true);
        ensureDataDir("/vault/.lilbee");
        expect(mockFs.writeFileSync).not.toHaveBeenCalled();
    });
});

describe("ServerManager", () => {
    describe("start()", () => {
        it("spawns with correct args and transitions to ready on health success", async () => {
            vi.useFakeTimers();
            mockFs.existsSync.mockReturnValue(true);
            const proc = createMockProcess();
            fetchMock.mockResolvedValue({ ok: true });

            const onStateChange = vi.fn();
            const mgr = new ServerManager(makeOpts({ onStateChange }));

            const startPromise = mgr.start();
            await vi.advanceTimersByTimeAsync(200);
            await startPromise;

            expect(mockSpawnFn).toHaveBeenCalledWith("/usr/bin/lilbee", [
                "serve", "--data-dir", "/vault/.lilbee", "--host", "127.0.0.1", "--port", "7500",
            ]);
            expect(mgr.state).toBe("ready");
            expect(onStateChange).toHaveBeenCalledWith("starting", undefined);
            expect(onStateChange).toHaveBeenCalledWith("ready", undefined);
        });

        it("transitions to error when health poll times out", async () => {
            vi.useFakeTimers();
            mockFs.existsSync.mockReturnValue(true);
            createMockProcess();
            fetchMock.mockRejectedValue(new Error("ECONNREFUSED"));

            const onStateChange = vi.fn();
            const mgr = new ServerManager(makeOpts({ onStateChange }));

            const startPromise = mgr.start();
            await vi.advanceTimersByTimeAsync(6400);
            await startPromise;

            expect(mgr.state).toBe("error");
            expect(onStateChange).toHaveBeenCalledWith("error", "server did not start in time");
        });

        it("transitions to error on unexpected process exit", async () => {
            vi.useFakeTimers();
            mockFs.existsSync.mockReturnValue(true);
            const proc = createMockProcess();
            fetchMock.mockRejectedValue(new Error("ECONNREFUSED"));

            const onStateChange = vi.fn();
            const mgr = new ServerManager(makeOpts({ onStateChange }));

            const startPromise = mgr.start();
            await vi.advanceTimersByTimeAsync(100);
            proc.emit("close", 1);
            await vi.advanceTimersByTimeAsync(6400);
            await startPromise;

            expect(onStateChange).toHaveBeenCalledWith("error", "exited with code 1");
        });

        it("no-ops when already starting or ready", async () => {
            vi.useFakeTimers();
            mockFs.existsSync.mockReturnValue(true);
            createMockProcess();
            fetchMock.mockResolvedValue({ ok: true });

            const mgr = new ServerManager(makeOpts());
            const p1 = mgr.start();
            await vi.advanceTimersByTimeAsync(200);
            await p1;

            expect(mgr.state).toBe("ready");
            await mgr.start();
            expect(mockSpawnFn).toHaveBeenCalledTimes(1);
        });

        it("includes stderr in error detail on process close", async () => {
            vi.useFakeTimers();
            mockFs.existsSync.mockReturnValue(true);
            const proc = createMockProcess();
            fetchMock.mockRejectedValue(new Error("ECONNREFUSED"));

            const onStateChange = vi.fn();
            const mgr = new ServerManager(makeOpts({ onStateChange }));

            const startPromise = mgr.start();
            await vi.advanceTimersByTimeAsync(100);

            proc.stderr.emit("data", Buffer.from("Address already in use"));
            proc.emit("close", 1);

            await vi.advanceTimersByTimeAsync(6400);
            await startPromise;

            expect(onStateChange).toHaveBeenCalledWith("error", "Address already in use");
        });
    });

    describe("stop()", () => {
        it("sends SIGTERM and waits for process to exit", async () => {
            vi.useFakeTimers();
            mockFs.existsSync.mockReturnValue(true);
            const proc = createMockProcess();
            fetchMock.mockResolvedValue({ ok: true });

            const mgr = new ServerManager(makeOpts());
            const startP = mgr.start();
            await vi.advanceTimersByTimeAsync(200);
            await startP;

            proc.kill.mockImplementation(() => {
                setTimeout(() => proc.emit("close", 0), 10);
            });

            const stopP = mgr.stop();
            await vi.advanceTimersByTimeAsync(100);
            await stopP;

            expect(proc.kill).toHaveBeenCalledWith("SIGTERM");
            expect(mgr.state).toBe("stopped");
        });

        it("sends SIGKILL after timeout", async () => {
            vi.useFakeTimers();
            mockFs.existsSync.mockReturnValue(true);
            const proc = createMockProcess();
            fetchMock.mockResolvedValue({ ok: true });

            const mgr = new ServerManager(makeOpts());
            const startP = mgr.start();
            await vi.advanceTimersByTimeAsync(200);
            await startP;

            const stopP = mgr.stop();
            await vi.advanceTimersByTimeAsync(5100);
            await stopP;

            expect(proc.kill).toHaveBeenCalledWith("SIGTERM");
            expect(proc.kill).toHaveBeenCalledWith("SIGKILL");
            expect(mgr.state).toBe("stopped");
        });

        it("no-ops when already stopped", async () => {
            const mgr = new ServerManager(makeOpts());
            await mgr.stop();
            expect(mgr.state).toBe("stopped");
        });
    });

    describe("restart()", () => {
        it("calls stop then start", async () => {
            vi.useFakeTimers();
            mockFs.existsSync.mockReturnValue(true);
            const proc = createMockProcess();
            fetchMock.mockResolvedValue({ ok: true });

            const mgr = new ServerManager(makeOpts());
            const startP = mgr.start();
            await vi.advanceTimersByTimeAsync(200);
            await startP;

            proc.kill.mockImplementation(() => {
                setTimeout(() => proc.emit("close", 0), 10);
            });

            // After stop, new spawn for restart
            const proc2 = createMockProcess();

            const restartP = mgr.restart();
            await vi.advanceTimersByTimeAsync(300);
            await restartP;

            expect(proc.kill).toHaveBeenCalledWith("SIGTERM");
            expect(mgr.state).toBe("ready");
        });
    });

    describe("onStateChange fires on each transition", () => {
        it("fires starting then ready on success", async () => {
            vi.useFakeTimers();
            mockFs.existsSync.mockReturnValue(true);
            createMockProcess();
            fetchMock.mockResolvedValue({ ok: true });

            const onStateChange = vi.fn();
            const mgr = new ServerManager(makeOpts({ onStateChange }));

            const p = mgr.start();
            await vi.advanceTimersByTimeAsync(200);
            await p;

            const states = onStateChange.mock.calls.map((c: any[]) => c[0]);
            expect(states).toEqual(["starting", "ready"]);
        });
    });
});
