/**
 * Minimal mock of the Obsidian API for testing.
 *
 * Obsidian extends HTMLElement with helpers like createDiv, createEl, empty, setText, addClass.
 * We simulate these on a lightweight MockElement tree so view/modal tests can
 * assert on the DOM structure without a real browser.
 */

import { vi } from "vitest";

export class MockElement {
    tagName: string;
    textContent: string = "";
    children: MockElement[] = [];
    classList: { list: string[]; add: (...classes: string[]) => void; remove: (...classes: string[]) => void; contains: (cls: string) => boolean };
    style: Record<string, string> = {};
    attributes: Record<string, string> = {};
    _listeners: Record<string, Function[]> = {};
    value: string = "";
    disabled: boolean = false;
    placeholder: string = "";
    parentElement: MockElement | null = null;

    constructor(tag = "div") {
        this.tagName = tag.toUpperCase();
        const list: string[] = [];
        this.classList = {
            list,
            add: (...classes: string[]) => { for (const cls of classes) { if (!list.includes(cls)) list.push(cls); } },
            remove: (...classes: string[]) => { for (const cls of classes) { const i = list.indexOf(cls); if (i >= 0) list.splice(i, 1); } },
            contains: (cls: string) => list.includes(cls),
        };
    }

    createDiv(opts?: string | { cls?: string; text?: string }): MockElement {
        const el = new MockElement("div");
        if (typeof opts === "string") {
            el.classList.add(opts);
        } else if (opts) {
            if (opts.cls) opts.cls.split(" ").forEach(c => el.classList.add(c));
            if (opts.text) el.textContent = opts.text;
        }
        el.parentElement = this;
        this.children.push(el);
        return el;
    }

    createEl(tag: string, opts?: { text?: string; cls?: string; type?: string; placeholder?: string; attr?: Record<string, string> }): MockElement {
        const el = new MockElement(tag);
        if (opts) {
            if (opts.text) el.textContent = opts.text;
            if (opts.cls) opts.cls.split(" ").forEach(c => el.classList.add(c));
            if (opts.type) el.attributes["type"] = opts.type;
            if (opts.placeholder) el.placeholder = opts.placeholder;
            if (opts.attr) Object.assign(el.attributes, opts.attr);
        }
        el.parentElement = this;
        this.children.push(el);
        return el;
    }

    empty(): void {
        this.children = [];
        this.textContent = "";
    }

    setText(text: string): void {
        this.textContent = text;
    }

    addClass(cls: string): void {
        cls.split(" ").forEach(c => this.classList.add(c));
    }

    removeClass(...classes: string[]): void {
        for (const cls of classes) {
            this.classList.remove(cls);
        }
    }

    addEventListener(event: string, handler: Function): void {
        if (!this._listeners[event]) this._listeners[event] = [];
        this._listeners[event].push(handler);
    }

    // Test helper: trigger an event
    trigger(event: string, ...args: unknown[]): void {
        for (const handler of this._listeners[event] ?? []) {
            handler(...args);
        }
    }

    // Test helper: find elements by class
    findAll(cls: string): MockElement[] {
        const found: MockElement[] = [];
        if (this.classList.contains(cls)) found.push(this);
        for (const child of this.children) {
            found.push(...child.findAll(cls));
        }
        return found;
    }

    // Test helper: find first element by class
    find(cls: string): MockElement | null {
        return this.findAll(cls)[0] ?? null;
    }

    querySelector(selector: string): MockElement | null {
        // Simple class selector support: ".some-class"
        if (selector.startsWith(".")) {
            return this.find(selector.slice(1));
        }
        return null;
    }

    remove(): void {
        if (this.parentElement) {
            const idx = this.parentElement.children.indexOf(this);
            if (idx >= 0) this.parentElement.children.splice(idx, 1);
            this.parentElement = null;
        }
    }

    scrollIntoView(_opts?: unknown): void { /* noop */ }
    focus(): void { /* noop */ }
    blur(): void { /* noop */ }

    // Support select element behavior
    get selectedIndex(): number { return 0; }
}

export interface TAbstractFile {
    path: string;
    name: string;
}

export class App {
    workspace = {
        openLinkText: vi.fn(),
        getLeavesOfType: vi.fn().mockReturnValue([]),
        getRightLeaf: vi.fn().mockReturnValue(null),
        revealLeaf: vi.fn(),
        on: vi.fn().mockReturnValue({ id: "mock-event" }),
        getActiveFile: vi.fn().mockReturnValue(null),
    };
    vault = {
        on: vi.fn().mockReturnValue({ id: "mock-vault-event" }),
        adapter: {
            getBasePath: vi.fn().mockReturnValue("/test/vault"),
        },
        getFiles: vi.fn().mockReturnValue([]),
    };
}

export interface TFile {
    path: string;
    name: string;
    parent: { path: string; name: string } | null;
}

export class Modal {
    app: App;
    contentEl: MockElement;

    constructor(app: App) {
        this.app = app;
        this.contentEl = new MockElement("div");
    }

    open(): void { this.onOpen(); }
    close(): void { this.onClose(); }
    onOpen(): void { /* override */ }
    onClose(): void { /* override */ }
}

export class FuzzySuggestModal<T> {
    app: App;

    constructor(app: App) {
        this.app = app;
    }

    setPlaceholder(_text: string): void { /* noop */ }
    open(): void { /* noop */ }
    close(): void { /* noop */ }
    getItems(): T[] { return []; }
    getItemText(_item: T): string { return ""; }
    onChooseItem(_item: T, _evt: unknown): void { /* override */ }
}

class MockMenuItem {
    setTitle(_title: string): this { return this; }
    setIcon(_icon: string): this { return this; }
    onClick(cb: () => void): this { cb(); return this; }
}

export class Menu {
    private _items: MockMenuItem[] = [];

    addItem(cb: (item: MockMenuItem) => void): this {
        const item = new MockMenuItem();
        cb(item);
        this._items.push(item);
        return this;
    }

    showAtMouseEvent(_event: unknown): void { /* noop */ }
}

export class ItemView {
    app: App;
    leaf: WorkspaceLeaf;
    containerEl: { children: MockElement[] };

    constructor(leaf: WorkspaceLeaf) {
        this.leaf = leaf;
        this.app = leaf.app ?? new App();
        const content = new MockElement("div");
        this.containerEl = { children: [new MockElement("div"), content] };
    }

    getViewType(): string { return ""; }
    getDisplayText(): string { return ""; }
    getIcon(): string { return ""; }

    registerEvent(_ref: unknown): void { /* noop */ }
}

export class WorkspaceLeaf {
    app: App;
    constructor(app?: App) {
        this.app = app ?? new App();
    }
    setViewState = vi.fn();
}

export class Plugin {
    app: App;
    manifest = { id: "lilbee", name: "lilbee", version: "0.1.0" };
    private _data: Record<string, unknown> = {};
    private _statusBarItems: MockElement[] = [];

    constructor(app?: App) {
        this.app = app ?? new App();
    }

    loadData = vi.fn(async () => this._data);
    saveData = vi.fn(async (data: Record<string, unknown>) => { this._data = data; });

    addCommand = vi.fn();
    addSettingTab = vi.fn();
    addStatusBarItem = vi.fn(() => {
        const el = new MockElement("div");
        this._statusBarItems.push(el);
        return el;
    });

    registerView = vi.fn();
    registerEvent = vi.fn();
}

export class PluginSettingTab {
    app: App;
    plugin: Plugin;
    containerEl: MockElement;

    constructor(app: App, plugin: Plugin) {
        this.app = app;
        this.plugin = plugin;
        this.containerEl = new MockElement("div");
    }

    display(): void { /* override */ }
    hide(): void { /* noop */ }
}

export const MarkdownRenderer = {
    async render(
        _app: App,
        markdown: string,
        el: MockElement,
        _sourcePath: string,
        _component: unknown,
    ): Promise<void> {
        el.empty();
        el.textContent = markdown;
        el.createEl("p", { text: markdown });
    },
};

export class Notice {
    message: string;
    duration: number | undefined;
    static instances: Notice[] = [];
    constructor(message: string, duration?: number) {
        this.message = message;
        this.duration = duration;
        Notice.instances.push(this);
    }
    static clear(): void { Notice.instances = []; }
}

export class Setting {
    private _el: MockElement;
    constructor(el: MockElement) {
        this._el = el;
    }
    setName(_name: string): this { return this; }
    setDesc(_desc: string): this { return this; }
    addText(cb: (text: MockTextComponent) => void): this {
        cb(new MockTextComponent());
        return this;
    }
    addSlider(cb: (slider: MockSliderComponent) => void): this {
        cb(new MockSliderComponent());
        return this;
    }
    addDropdown(cb: (dropdown: MockDropdownComponent) => void): this {
        cb(new MockDropdownComponent());
        return this;
    }
    addButton(cb: (btn: MockButtonComponent) => void): this {
        cb(new MockButtonComponent());
        return this;
    }
    addToggle(cb: (toggle: MockToggleComponent) => void): this {
        cb(new MockToggleComponent());
        return this;
    }
}

class MockTextComponent {
    private _onChange: ((v: string) => void) | null = null;
    setPlaceholder(_p: string): this { return this; }
    setValue(_v: string): this { return this; }
    onChange(cb: (v: string) => void): this { this._onChange = cb; return this; }
    triggerChange(v: string): void { this._onChange?.(v); }
}

class MockSliderComponent {
    private _onChange: ((v: number) => void) | null = null;
    setLimits(_min: number, _max: number, _step: number): this { return this; }
    setValue(_v: number): this { return this; }
    setDynamicTooltip(): this { return this; }
    onChange(cb: (v: number) => void): this { this._onChange = cb; return this; }
    triggerChange(v: number): void { this._onChange?.(v); }
}

class MockDropdownComponent {
    private _onChange: ((v: string) => void) | null = null;
    addOption(_value: string, _label: string): this { return this; }
    addOptions(_opts: Record<string, string>): this { return this; }
    setValue(_v: string): this { return this; }
    onChange(cb: (v: string) => void): this { this._onChange = cb; return this; }
    triggerChange(v: string): void { this._onChange?.(v); }
}

class MockToggleComponent {
    private _onChange: ((v: boolean) => void) | null = null;
    setValue(_v: boolean): this { return this; }
    onChange(cb: (v: boolean) => void): this { this._onChange = cb; return this; }
    triggerChange(v: boolean): void { this._onChange?.(v); }
}

class MockButtonComponent {
    private _onClick: (() => void) | null = null;
    setButtonText(_text: string): this { return this; }
    onClick(cb: () => void): this { this._onClick = cb; return this; }
    triggerClick(): void { this._onClick?.(); }
}
