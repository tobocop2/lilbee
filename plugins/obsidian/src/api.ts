import { JSON_HEADERS, SSE_EVENT } from "./types";
import type {
    AskResponse,
    DocumentResult,
    Message,
    ModelsResponse,
    SSEEvent,
    StatusResponse,
} from "./types";

export class LilbeeClient {
    constructor(private baseUrl: string) {}

    private async assertOk(res: Response): Promise<Response> {
        if (!res.ok) {
            const text = await res.text().catch(() => "");
            throw new Error(`Server responded ${res.status}: ${text}`);
        }
        return res;
    }

    async health(): Promise<{ status: string; version: string }> {
        const res = await this.assertOk(await fetch(`${this.baseUrl}/api/health`));
        return res.json();
    }

    async status(): Promise<StatusResponse> {
        const res = await this.assertOk(await fetch(`${this.baseUrl}/api/status`));
        return res.json();
    }

    async search(query: string, topK?: number): Promise<DocumentResult[]> {
        const params = new URLSearchParams({ q: query });
        if (topK !== undefined) params.set("top_k", String(topK));
        const res = await this.assertOk(await fetch(`${this.baseUrl}/api/search?${params}`));
        return res.json();
    }

    async ask(question: string, topK?: number): Promise<AskResponse> {
        const res = await this.assertOk(
            await fetch(`${this.baseUrl}/api/ask`, {
                method: "POST",
                headers: JSON_HEADERS,
                body: JSON.stringify({ question, top_k: topK ?? 0 }),
            }),
        );
        return res.json();
    }

    async *askStream(question: string, topK?: number): AsyncGenerator<SSEEvent> {
        const res = await this.assertOk(
            await fetch(`${this.baseUrl}/api/ask/stream`, {
                method: "POST",
                headers: JSON_HEADERS,
                body: JSON.stringify({ question, top_k: topK ?? 0 }),
            }),
        );
        yield* this.parseSSE(res);
    }

    async chat(question: string, history: Message[], topK?: number): Promise<AskResponse> {
        const res = await this.assertOk(
            await fetch(`${this.baseUrl}/api/chat`, {
                method: "POST",
                headers: JSON_HEADERS,
                body: JSON.stringify({ question, history, top_k: topK ?? 0 }),
            }),
        );
        return res.json();
    }

    async *chatStream(
        question: string,
        history: Message[],
        topK?: number,
    ): AsyncGenerator<SSEEvent> {
        const res = await this.assertOk(
            await fetch(`${this.baseUrl}/api/chat/stream`, {
                method: "POST",
                headers: JSON_HEADERS,
                body: JSON.stringify({ question, history, top_k: topK ?? 0 }),
            }),
        );
        yield* this.parseSSE(res);
    }

    async *addFiles(
        paths: string[],
        force = false,
        visionModel?: string,
    ): AsyncGenerator<SSEEvent> {
        const body: Record<string, unknown> = { paths, force };
        if (visionModel) body.vision_model = visionModel;
        const res = await this.assertOk(
            await fetch(`${this.baseUrl}/api/add`, {
                method: "POST",
                headers: JSON_HEADERS,
                body: JSON.stringify(body),
            }),
        );
        yield* this.parseSSE(res);
    }

    async *syncStream(forceVision = false): AsyncGenerator<SSEEvent> {
        const res = await this.assertOk(
            await fetch(`${this.baseUrl}/api/sync`, {
                method: "POST",
                headers: JSON_HEADERS,
                body: JSON.stringify({ force_vision: forceVision }),
            }),
        );
        yield* this.parseSSE(res);
    }

    async listModels(): Promise<ModelsResponse> {
        const res = await this.assertOk(await fetch(`${this.baseUrl}/api/models`));
        return res.json();
    }

    async *pullModel(model: string): AsyncGenerator<SSEEvent> {
        const res = await this.assertOk(
            await fetch(`${this.baseUrl}/api/models/pull`, {
                method: "POST",
                headers: JSON_HEADERS,
                body: JSON.stringify({ model }),
            }),
        );
        yield* this.parseSSE(res);
    }

    async setChatModel(model: string): Promise<{ model: string }> {
        const res = await this.assertOk(
            await fetch(`${this.baseUrl}/api/models/chat`, {
                method: "PUT",
                headers: JSON_HEADERS,
                body: JSON.stringify({ model }),
            }),
        );
        return res.json();
    }

    async setVisionModel(model: string): Promise<{ model: string }> {
        const res = await this.assertOk(
            await fetch(`${this.baseUrl}/api/models/vision`, {
                method: "PUT",
                headers: JSON_HEADERS,
                body: JSON.stringify({ model }),
            }),
        );
        return res.json();
    }

    private async *parseSSE(response: Response): AsyncGenerator<SSEEvent> {
        if (!response.body) {
            throw new Error("Response body is null");
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let currentEvent: string = SSE_EVENT.MESSAGE;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });

            const lines = buffer.split("\n");
            buffer = lines.pop() ?? "";

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
}
