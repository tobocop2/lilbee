import { vi, describe, it, expect, beforeEach } from "vitest";
import { Notice, WorkspaceLeaf } from "../__mocks__/obsidian";
import { MockElement } from "../__mocks__/obsidian";
import { ChatView, VIEW_TYPE_CHAT } from "../../src/views/chat-view";
import type LilbeePlugin from "../../src/main";
import type { SSEEvent, Source } from "../../src/types";

function makeLeaf(): WorkspaceLeaf {
    return new WorkspaceLeaf();
}

/**
 * Creates a chatStream mock that yields the given events and returns a
 * promise that resolves when the generator is fully consumed.
 */
function makeStream(events: SSEEvent[]): {
    mockFn: ReturnType<typeof vi.fn>;
    done: Promise<void>;
} {
    let resolveStream!: () => void;
    const done = new Promise<void>((r) => { resolveStream = r; });
    const mockFn = vi.fn().mockReturnValue(
        (async function* () {
            for (const e of events) yield e;
            resolveStream();
        })(),
    );
    return { mockFn, done };
}

/** Flush one macrotask tick so async chains settle. */
function tick(): Promise<void> {
    return new Promise((r) => setTimeout(r, 0));
}

function makePlugin(): LilbeePlugin {
    return {
        api: { chatStream: vi.fn() },
        settings: { topK: 5 },
    } as unknown as LilbeePlugin;
}

function makeSource(overrides: Partial<Source> = {}): Source {
    return {
        source: "doc.pdf",
        chunk: "some chunk text",
        distance: 0.5,
        content_type: "pdf",
        page_start: null,
        page_end: null,
        line_start: null,
        line_end: null,
        ...overrides,
    };
}

describe("VIEW_TYPE_CHAT", () => {
    it("equals 'lilbee-chat'", () => {
        expect(VIEW_TYPE_CHAT).toBe("lilbee-chat");
    });
});

describe("ChatView metadata methods", () => {
    let view: ChatView;

    beforeEach(() => {
        Notice.clear();
        view = new ChatView(makeLeaf(), makePlugin());
    });

    it("getViewType returns 'lilbee-chat'", () => {
        expect(view.getViewType()).toBe("lilbee-chat");
    });

    it("getDisplayText returns 'lilbee Chat'", () => {
        expect(view.getDisplayText()).toBe("lilbee Chat");
    });

    it("getIcon returns 'message-circle'", () => {
        expect(view.getIcon()).toBe("message-circle");
    });
});

describe("ChatView.onClose", () => {
    it("resolves without error (empty method coverage)", async () => {
        const view = new ChatView(makeLeaf(), makePlugin());
        await expect(view.onClose()).resolves.toBeUndefined();
    });
});

describe("ChatView.onOpen — DOM structure", () => {
    let view: ChatView;
    let container: MockElement;

    beforeEach(async () => {
        Notice.clear();
        view = new ChatView(makeLeaf(), makePlugin());
        container = view.containerEl.children[1] as unknown as MockElement;
        await view.onOpen();
    });

    it("empties and adds class to the content container", () => {
        expect(container.classList.contains("lilbee-chat-container")).toBe(true);
    });

    it("creates a toolbar div with class 'lilbee-chat-toolbar'", () => {
        expect(container.find("lilbee-chat-toolbar")).not.toBeNull();
    });

    it("creates a clear button inside the toolbar", () => {
        const clearBtn = container.find("lilbee-chat-clear");
        expect(clearBtn).not.toBeNull();
        expect(clearBtn!.tagName).toBe("BUTTON");
        expect(clearBtn!.textContent).toBe("Clear chat");
    });

    it("creates a messages div with class 'lilbee-chat-messages'", () => {
        expect(container.find("lilbee-chat-messages")).not.toBeNull();
    });

    it("creates an input area div with class 'lilbee-chat-input'", () => {
        expect(container.find("lilbee-chat-input")).not.toBeNull();
    });

    it("creates a textarea with placeholder and class inside input area", () => {
        const textarea = container.find("lilbee-chat-textarea");
        expect(textarea).not.toBeNull();
        expect(textarea!.tagName).toBe("TEXTAREA");
        expect(textarea!.placeholder).toBe("Ask something...");
    });

    it("creates a send button with correct text and class", () => {
        const sendBtn = container.find("lilbee-chat-send");
        expect(sendBtn).not.toBeNull();
        expect(sendBtn!.tagName).toBe("BUTTON");
        expect(sendBtn!.textContent).toBe("Send");
    });
});

describe("ChatView.onOpen — send button empty text guard", () => {
    it("does not call chatStream when textarea is empty", async () => {
        Notice.clear();
        const plugin = makePlugin();
        const view = new ChatView(makeLeaf(), plugin);
        await view.onOpen();
        const container = view.containerEl.children[1] as unknown as MockElement;

        container.find("lilbee-chat-send")!.trigger("click");

        expect(plugin.api.chatStream).not.toHaveBeenCalled();
    });

    it("does not call chatStream when textarea has only whitespace", async () => {
        Notice.clear();
        const plugin = makePlugin();
        const view = new ChatView(makeLeaf(), plugin);
        await view.onOpen();
        const container = view.containerEl.children[1] as unknown as MockElement;
        const textarea = container.find("lilbee-chat-textarea")!;
        textarea.value = "   ";

        container.find("lilbee-chat-send")!.trigger("click");

        expect(plugin.api.chatStream).not.toHaveBeenCalled();
    });
});

describe("ChatView.onOpen — send button triggers send", () => {
    it("calls chatStream with trimmed text on send button click", async () => {
        Notice.clear();
        const plugin = makePlugin();
        const { mockFn, done } = makeStream([{ event: "done", data: {} }]);
        plugin.api.chatStream = mockFn;
        const view = new ChatView(makeLeaf(), plugin);
        await view.onOpen();
        const container = view.containerEl.children[1] as unknown as MockElement;
        const textarea = container.find("lilbee-chat-textarea")!;
        textarea.value = "  hello  ";

        container.find("lilbee-chat-send")!.trigger("click");
        await done;

        expect(plugin.api.chatStream).toHaveBeenCalledWith("hello", [], 5);
    });

    it("clears textarea value after send", async () => {
        Notice.clear();
        const plugin = makePlugin();
        const { mockFn } = makeStream([{ event: "done", data: {} }]);
        plugin.api.chatStream = mockFn;
        const view = new ChatView(makeLeaf(), plugin);
        await view.onOpen();
        const container = view.containerEl.children[1] as unknown as MockElement;
        const textarea = container.find("lilbee-chat-textarea")!;
        textarea.value = "hello";

        container.find("lilbee-chat-send")!.trigger("click");

        // textarea is cleared synchronously before async work begins
        expect(textarea.value).toBe("");
    });
});

describe("ChatView.onOpen — keydown on textarea", () => {
    it("Enter without Shift triggers send and calls preventDefault", async () => {
        Notice.clear();
        const plugin = makePlugin();
        const { mockFn, done } = makeStream([{ event: "done", data: {} }]);
        plugin.api.chatStream = mockFn;
        const view = new ChatView(makeLeaf(), plugin);
        await view.onOpen();
        const container = view.containerEl.children[1] as unknown as MockElement;
        const textarea = container.find("lilbee-chat-textarea")!;
        textarea.value = "hi there";

        const event = { key: "Enter", shiftKey: false, preventDefault: vi.fn() };
        textarea.trigger("keydown", event);
        await done;

        expect(event.preventDefault).toHaveBeenCalled();
        expect(plugin.api.chatStream).toHaveBeenCalled();
    });

    it("Shift+Enter does NOT trigger send", async () => {
        Notice.clear();
        const plugin = makePlugin();
        const view = new ChatView(makeLeaf(), plugin);
        await view.onOpen();
        const container = view.containerEl.children[1] as unknown as MockElement;
        const textarea = container.find("lilbee-chat-textarea")!;
        textarea.value = "hi";

        const event = { key: "Enter", shiftKey: true, preventDefault: vi.fn() };
        textarea.trigger("keydown", event);

        expect(event.preventDefault).not.toHaveBeenCalled();
        expect(plugin.api.chatStream).not.toHaveBeenCalled();
    });

    it("non-Enter key does NOT trigger send", async () => {
        Notice.clear();
        const plugin = makePlugin();
        const view = new ChatView(makeLeaf(), plugin);
        await view.onOpen();
        const container = view.containerEl.children[1] as unknown as MockElement;
        const textarea = container.find("lilbee-chat-textarea")!;
        textarea.value = "hi";

        const event = { key: "a", shiftKey: false, preventDefault: vi.fn() };
        textarea.trigger("keydown", event);

        expect(event.preventDefault).not.toHaveBeenCalled();
        expect(plugin.api.chatStream).not.toHaveBeenCalled();
    });
});

describe("ChatView.sendMessage — bubble structure", () => {
    it("creates user bubble and assistant bubble", async () => {
        Notice.clear();
        const plugin = makePlugin();
        const { mockFn, done } = makeStream([{ event: "done", data: {} }]);
        plugin.api.chatStream = mockFn;
        const view = new ChatView(makeLeaf(), plugin);
        await view.onOpen();
        const container = view.containerEl.children[1] as unknown as MockElement;
        const messagesEl = container.find("lilbee-chat-messages")!;
        const textarea = container.find("lilbee-chat-textarea")!;
        textarea.value = "question";

        container.find("lilbee-chat-send")!.trigger("click");
        await done;
        await tick();

        expect(messagesEl.children[0].classList.contains("user")).toBe(true);
        expect(messagesEl.children[1].classList.contains("assistant")).toBe(true);
    });

    it("user bubble contains a <p> with the message text", async () => {
        Notice.clear();
        const plugin = makePlugin();
        const { mockFn, done } = makeStream([{ event: "done", data: {} }]);
        plugin.api.chatStream = mockFn;
        const view = new ChatView(makeLeaf(), plugin);
        await view.onOpen();
        const container = view.containerEl.children[1] as unknown as MockElement;
        const messagesEl = container.find("lilbee-chat-messages")!;
        const textarea = container.find("lilbee-chat-textarea")!;
        textarea.value = "my question";

        container.find("lilbee-chat-send")!.trigger("click");
        await done;
        await tick();

        const userBubble = messagesEl.children[0];
        const p = userBubble.children.find((c) => c.tagName === "P");
        expect(p).toBeDefined();
        expect(p!.textContent).toBe("my question");
    });
});

describe("ChatView.sendMessage — token streaming", () => {
    it("streams tokens into assistant text element", async () => {
        Notice.clear();
        const plugin = makePlugin();
        const { mockFn, done } = makeStream([
            { event: "token", data: "Hello" },
            { event: "token", data: " world" },
            { event: "done", data: {} },
        ]);
        plugin.api.chatStream = mockFn;
        const view = new ChatView(makeLeaf(), plugin);
        await view.onOpen();
        const container = view.containerEl.children[1] as unknown as MockElement;
        const messagesEl = container.find("lilbee-chat-messages")!;
        const textarea = container.find("lilbee-chat-textarea")!;
        textarea.value = "stream test";

        container.find("lilbee-chat-send")!.trigger("click");
        await done;
        await tick();

        const assistantBubble = messagesEl.children[1];
        const textEl = assistantBubble.children.find((c) => c.tagName === "P");
        expect(textEl!.textContent).toBe("Hello world");
    });
});

describe("ChatView.sendMessage — sources", () => {
    it("renders sources section when sources event precedes done", async () => {
        Notice.clear();
        const plugin = makePlugin();
        const { mockFn, done } = makeStream([
            { event: "token", data: "Answer" },
            { event: "sources", data: [makeSource({ source: "notes.md" })] },
            { event: "done", data: {} },
        ]);
        plugin.api.chatStream = mockFn;
        const view = new ChatView(makeLeaf(), plugin);
        await view.onOpen();
        const container = view.containerEl.children[1] as unknown as MockElement;
        const messagesEl = container.find("lilbee-chat-messages")!;
        const textarea = container.find("lilbee-chat-textarea")!;
        textarea.value = "sources question";

        container.find("lilbee-chat-send")!.trigger("click");
        await done;
        await tick();

        const assistantBubble = messagesEl.children[1];
        expect(assistantBubble.find("lilbee-chat-sources")).not.toBeNull();
    });

    it("renders a details/summary inside sources section", async () => {
        Notice.clear();
        const plugin = makePlugin();
        const { mockFn, done } = makeStream([
            { event: "sources", data: [makeSource()] },
            { event: "done", data: {} },
        ]);
        plugin.api.chatStream = mockFn;
        const view = new ChatView(makeLeaf(), plugin);
        await view.onOpen();
        const container = view.containerEl.children[1] as unknown as MockElement;
        const messagesEl = container.find("lilbee-chat-messages")!;
        const textarea = container.find("lilbee-chat-textarea")!;
        textarea.value = "details test";

        container.find("lilbee-chat-send")!.trigger("click");
        await done;
        await tick();

        const assistantBubble = messagesEl.children[1];
        const sourcesEl = assistantBubble.find("lilbee-chat-sources")!;
        const details = sourcesEl.children.find((c) => c.tagName === "DETAILS");
        expect(details).toBeDefined();
        const summary = details!.children.find((c) => c.tagName === "SUMMARY");
        expect(summary).toBeDefined();
        expect(summary!.textContent).toBe("Sources");
    });

    it("renders source chips inside details element", async () => {
        Notice.clear();
        const plugin = makePlugin();
        const { mockFn, done } = makeStream([
            { event: "sources", data: [makeSource({ source: "chip-doc.md" })] },
            { event: "done", data: {} },
        ]);
        plugin.api.chatStream = mockFn;
        const view = new ChatView(makeLeaf(), plugin);
        await view.onOpen();
        const container = view.containerEl.children[1] as unknown as MockElement;
        const messagesEl = container.find("lilbee-chat-messages")!;
        const textarea = container.find("lilbee-chat-textarea")!;
        textarea.value = "chip test";

        container.find("lilbee-chat-send")!.trigger("click");
        await done;
        await tick();

        const assistantBubble = messagesEl.children[1];
        const chip = assistantBubble.find("lilbee-source-chip");
        expect(chip).not.toBeNull();
        expect(chip!.textContent).toBe("chip-doc.md");
    });

    it("does NOT render sources section when done has no prior sources", async () => {
        Notice.clear();
        const plugin = makePlugin();
        const { mockFn, done } = makeStream([
            { event: "token", data: "Answer" },
            { event: "done", data: {} },
        ]);
        plugin.api.chatStream = mockFn;
        const view = new ChatView(makeLeaf(), plugin);
        await view.onOpen();
        const container = view.containerEl.children[1] as unknown as MockElement;
        const messagesEl = container.find("lilbee-chat-messages")!;
        const textarea = container.find("lilbee-chat-textarea")!;
        textarea.value = "no sources test";

        container.find("lilbee-chat-send")!.trigger("click");
        await done;
        await tick();

        const assistantBubble = messagesEl.children[1];
        expect(assistantBubble.find("lilbee-chat-sources")).toBeNull();
    });

    it("pushes assistant message into history after done event", async () => {
        Notice.clear();
        const plugin = makePlugin();
        const { mockFn: mockFn1, done: done1 } = makeStream([
            { event: "token", data: "Reply" },
            { event: "done", data: {} },
        ]);
        plugin.api.chatStream = mockFn1;
        const view = new ChatView(makeLeaf(), plugin);
        await view.onOpen();
        const container = view.containerEl.children[1] as unknown as MockElement;
        const textarea = container.find("lilbee-chat-textarea")!;

        textarea.value = "first question";
        container.find("lilbee-chat-send")!.trigger("click");
        await done1;
        await tick();

        // Second send — verify history passed to chatStream includes prior exchange
        const { mockFn: mockFn2, done: done2 } = makeStream([{ event: "done", data: {} }]);
        plugin.api.chatStream = mockFn2;

        textarea.value = "second question";
        container.find("lilbee-chat-send")!.trigger("click");
        await done2;

        const historyArg = (mockFn2 as ReturnType<typeof vi.fn>).mock.calls[0][1] as Array<{
            role: string;
            content: string;
        }>;
        expect(historyArg.some((m) => m.role === "user" && m.content === "first question")).toBe(true);
        expect(historyArg.some((m) => m.role === "assistant" && m.content === "Reply")).toBe(true);
    });
});

describe("ChatView.sendMessage — error event", () => {
    it("shows a Notice when error event is received", async () => {
        Notice.clear();
        const plugin = makePlugin();
        const { mockFn, done } = makeStream([
            { event: "error", data: "something went wrong" },
        ]);
        plugin.api.chatStream = mockFn;
        const view = new ChatView(makeLeaf(), plugin);
        await view.onOpen();
        const container = view.containerEl.children[1] as unknown as MockElement;
        const textarea = container.find("lilbee-chat-textarea")!;
        textarea.value = "error question";

        container.find("lilbee-chat-send")!.trigger("click");
        await done;
        await tick();

        expect(Notice.instances[0].message).toBe("lilbee: something went wrong");
    });
});

describe("ChatView.sendMessage — API throws", () => {
    it("shows connection error Notice when chatStream throws", async () => {
        Notice.clear();
        const plugin = makePlugin();
        let resolveThrown!: () => void;
        const thrown = new Promise<void>((r) => { resolveThrown = r; });
        plugin.api.chatStream = vi.fn().mockReturnValue(
            (async function* () {
                resolveThrown();
                throw new Error("network");
            })(),
        );
        const view = new ChatView(makeLeaf(), plugin);
        await view.onOpen();
        const container = view.containerEl.children[1] as unknown as MockElement;
        const textarea = container.find("lilbee-chat-textarea")!;
        textarea.value = "throw question";

        container.find("lilbee-chat-send")!.trigger("click");
        await thrown;
        await tick();

        expect(Notice.instances[0].message).toBe(
            "lilbee: chat error — cannot connect to server",
        );
    });

    it("sets assistant text to error message when chatStream throws", async () => {
        Notice.clear();
        const plugin = makePlugin();
        let resolveThrown!: () => void;
        const thrown = new Promise<void>((r) => { resolveThrown = r; });
        plugin.api.chatStream = vi.fn().mockReturnValue(
            (async function* () {
                resolveThrown();
                throw new Error("network");
            })(),
        );
        const view = new ChatView(makeLeaf(), plugin);
        await view.onOpen();
        const container = view.containerEl.children[1] as unknown as MockElement;
        const messagesEl = container.find("lilbee-chat-messages")!;
        const textarea = container.find("lilbee-chat-textarea")!;
        textarea.value = "throw question";

        container.find("lilbee-chat-send")!.trigger("click");
        await thrown;
        await tick();

        const assistantBubble = messagesEl.children[1];
        const textEl = assistantBubble.children.find((c) => c.tagName === "P");
        expect(textEl!.textContent).toBe("Error: could not connect to the lilbee server.");
    });
});

describe("ChatView.sendMessage — messagesEl null guard", () => {
    it("returns immediately if messagesEl is null (onOpen not called)", async () => {
        Notice.clear();
        const plugin = makePlugin();
        const view = new ChatView(makeLeaf(), plugin);
        // Do NOT call onOpen — messagesEl stays null

        // Access the private method directly for coverage
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        await (view as any).sendMessage("test");

        expect(plugin.api.chatStream).not.toHaveBeenCalled();
    });
});

describe("ChatView.clearChat — via toolbar button", () => {
    it("empties messagesEl when clear button is clicked", async () => {
        Notice.clear();
        const plugin = makePlugin();
        const { mockFn, done } = makeStream([
            { event: "token", data: "hi" },
            { event: "done", data: {} },
        ]);
        plugin.api.chatStream = mockFn;
        const view = new ChatView(makeLeaf(), plugin);
        await view.onOpen();
        const container = view.containerEl.children[1] as unknown as MockElement;
        const messagesEl = container.find("lilbee-chat-messages")!;
        const textarea = container.find("lilbee-chat-textarea")!;

        textarea.value = "populate history";
        container.find("lilbee-chat-send")!.trigger("click");
        await done;
        await tick();

        expect(messagesEl.children.length).toBeGreaterThan(0);

        container.find("lilbee-chat-clear")!.trigger("click");

        expect(messagesEl.children).toHaveLength(0);
    });

    it("resets history so next chatStream receives empty history", async () => {
        Notice.clear();
        const plugin = makePlugin();
        const { mockFn: mockFn1, done: done1 } = makeStream([
            { event: "token", data: "hi" },
            { event: "done", data: {} },
        ]);
        plugin.api.chatStream = mockFn1;
        const view = new ChatView(makeLeaf(), plugin);
        await view.onOpen();
        const container = view.containerEl.children[1] as unknown as MockElement;
        const textarea = container.find("lilbee-chat-textarea")!;

        textarea.value = "first";
        container.find("lilbee-chat-send")!.trigger("click");
        await done1;
        await tick();

        // Clear history
        container.find("lilbee-chat-clear")!.trigger("click");

        // Send another message — history passed to chatStream must be empty
        const { mockFn: mockFn2, done: done2 } = makeStream([{ event: "done", data: {} }]);
        plugin.api.chatStream = mockFn2;
        textarea.value = "after clear";
        container.find("lilbee-chat-send")!.trigger("click");
        await done2;

        const historyArg = (mockFn2 as ReturnType<typeof vi.fn>).mock.calls[0][1] as Array<unknown>;
        expect(historyArg).toHaveLength(0);
    });
});
