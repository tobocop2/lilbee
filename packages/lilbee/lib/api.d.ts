/**
 * Programmatic API of the lilbee npm launcher: resolve, download, and list
 * lilbee release binaries with the CLI's detection, verification, and cache
 * layout. Safe in a renderer: no console output, nothing blocks the event loop.
 */

export type Platform = "darwin" | "linux" | "win32";
export type Arch = "arm64" | "x64";

/** A release build. "default" is the plain build every host can run. */
export type Variant = "default" | "cu121" | "cu124" | "cu125" | "rocm" | "compat" | "compat-cu124" | "compat-rocm";

/** What this machine can run, resolved once and passed to every release query. */
export interface Host {
    platform: Platform;
    arch: Arch;
    /** The build detection picked; "default" when nothing better applies. */
    variant: Variant;
    /** The gfx targets of the host's AMD GPUs; empty when there are none. */
    amdGfxTargets: string[];
}

/**
 * Probe the host: NVIDIA driver CUDA level, AMD KFD topology, AVX2 baseline.
 * Never throws; detection failures fall back to the default build.
 */
export function detectHost(env?: NodeJS.ProcessEnv): Promise<Host>;

/** The release asset name for a host, e.g. "lilbee-linux-x86_64-cu124". Throws on an unsupported host. */
export function assetNameFor(platform: string, arch: string, variant?: Variant | ""): string;

/** The host and variant an asset name encodes, or null for an asset that is not a lilbee binary. */
export function parseAssetName(name: string): { platform: Platform; arch: Arch; variant: Variant } | null;

/** An in-development build, tagged with a trailing `.dev<n>` (e.g. v0.6.90b420.dev711). */
export function isDevBuild(tag: string): boolean;

/** Order release tags newest-first; numeric segments compare numerically. */
export function compareReleaseTags(a: string, b: string): number;

/** The minimum of the Fetch API the launcher uses: GET with headers and an abort signal. */
export type FetchLike = (
    url: string,
    init?: { headers?: Record<string, string>; signal?: AbortSignal },
) => Promise<FetchResponseLike>;

/** A fetch response whose body is a web ReadableStream or a Node Readable. */
export interface FetchResponseLike {
    ok: boolean;
    status: number;
    headers: { get(name: string): string | null };
    body: ReadableStream<Uint8Array> | NodeJS.ReadableStream | null;
    json(): Promise<unknown>;
    text(): Promise<string>;
}

export interface ReleaseQuery {
    /** GitHub "owner/repo"; defaults to tobocop2/lilbee. */
    repo?: string;
    /** Offer `.dev` builds as well as stable releases. Default false. */
    includeDev?: boolean;
    /** The host to resolve builds for; defaults to detectHost(). */
    host?: Host;
    /** Transport for the GitHub API and the asset manifests; defaults to global fetch. */
    fetch?: FetchLike;
    /** Read for GITHUB_TOKEN / GH_TOKEN (sent as a bearer to the GitHub API). Defaults to process.env. */
    env?: NodeJS.ProcessEnv;
}

/** A release with the build this host should run resolved. */
export interface ResolvedRelease {
    tag: string;
    dev: boolean;
    assetName: string;
    variant: Variant;
    /** Asset size in bytes as GitHub reports it. */
    size: number;
    /** GitHub's sha256 hex digest of the asset, or null on releases that carry none. */
    digest: string | null;
    url: string;
}

/**
 * Published releases that ship a build for the host, newest first: up to `limit` stable
 * releases plus up to `limit` dev builds when `includeDev` is set. Drafts, prereleases,
 * and releases without a build for the host are left out. Throws with a plain message
 * when GitHub rate-limits the request.
 */
export function listReleases(query?: ReleaseQuery & { limit?: number }): Promise<ResolvedRelease[]>;

/** The newest installable release on the chosen channel. Throws when none exists. */
export function latestRelease(query?: ReleaseQuery): Promise<ResolvedRelease>;

/** One release by tag, resolved for the host. Throws when the tag or the host's build is missing. */
export function releaseByTag(tag: string, query?: ReleaseQuery): Promise<ResolvedRelease>;

/** Where the launcher caches binaries: LILBEE_MCP_CACHE, else the platform cache dir. */
export function cacheDir(env?: NodeJS.ProcessEnv): string;

/** Path of a cached binary: `<cacheDir>/<release>/<assetName>`. */
export function cachedBinaryPath(cacheDir: string, release: string, assetName: string): string;

export interface InstalledBinary {
    path: string;
    release: string;
    assetName: string;
    variant: Variant;
}

/**
 * The newest cached binary for this host, or null when none is cached. Within a release,
 * a binary matching `host.variant` wins over any other build of the host's platform.
 */
export function installedBinary(options: { cacheDir: string; host: Host }): InstalledBinary | null;

export interface DownloadProgress {
    /** Bytes received so far. */
    done: number;
    /** Total bytes when known (Content-Length, else the release's asset size), or null. */
    total: number | null;
}

export interface EnsureOptions extends ReleaseQuery {
    cacheDir: string;
    /** An exact release tag to install; default: the cached binary, else the latest release. */
    release?: string;
    /** Re-resolve the latest release even when a binary is cached (what `lilbee prepare` does). */
    refresh?: boolean;
    /** Download even when the resolved release is already cached (reinstall). */
    force?: boolean;
    onProgress?: (progress: DownloadProgress) => void;
    /** Aborting rejects with DownloadCanceledError and discards the partial file. */
    signal?: AbortSignal;
    /** Human-readable download lifecycle lines; silent by default. */
    log?: (message: string) => void;
}

export interface EnsureResult extends InstalledBinary {
    source: "cache" | "download";
}

/**
 * Make a lilbee binary available in `cacheDir` and return it. Downloads stream to a temp
 * file, verify sha256 against the release digest, land by atomic rename, then remove
 * every other cached release and every other build of the same release. Rejects with
 * DownloadCanceledError when `signal` aborts; a stalled transfer (no bytes for 60 s)
 * is retried once, then rejected.
 */
export function ensureBinary(options: EnsureOptions): Promise<EnsureResult>;

/** Thrown when `signal` aborts an ensureBinary download. `name` is "AbortError". */
export class DownloadCanceledError extends Error {
    readonly name: "AbortError";
}

/** True for a DownloadCanceledError, or any error whose name is "AbortError". */
export function isDownloadCanceled(err: unknown): boolean;
