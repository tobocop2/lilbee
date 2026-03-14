import { App, Notice, PluginSettingTab, Setting } from "obsidian";
import type LilbeePlugin from "./main";
import { SSE_EVENT } from "./types";
import type { ModelCatalog, ModelInfo, ModelsResponse, PullProgress } from "./types";

const CHECK_TIMEOUT_MS = 5000;
const CLS_MODELS_CONTAINER = "lilbee-models-container";
const SEPARATOR_KEY = "__separator__";
const SEPARATOR_LABEL = "\u2500\u2500 Other... \u2500\u2500";

/**
 * Remove `:latest` entries when a more specific tag of the same model exists.
 * e.g. if both `mistral:latest` and `mistral:7b` are present, drop `mistral:latest`.
 */
export function deduplicateLatest(models: string[]): string[] {
    const bases = new Set(
        models
            .filter((m) => !m.endsWith(":latest"))
            .map((m) => m.split(":")[0]),
    );
    return models.filter((m) => {
        if (!m.endsWith(":latest")) return true;
        return !bases.has(m.split(":")[0]);
    });
}

export function buildModelOptions(
    catalog: ModelCatalog,
    type: "chat" | "vision",
): Record<string, string> {
    const options: Record<string, string> = {};
    if (type === "vision") {
        options[""] = "Disabled";
    }

    const catalogNames = new Set(catalog.catalog.map((m) => m.name));
    const sortedCatalog = [...catalog.catalog].sort((a, b) => a.name.localeCompare(b.name));
    for (const model of sortedCatalog) {
        const suffix = model.installed ? "" : " (not installed)";
        options[model.name] = `${model.name}${suffix}`;
    }

    const otherInstalled = deduplicateLatest(
        catalog.installed.filter((name) => !catalogNames.has(name)),
    ).sort();

    if (otherInstalled.length > 0) {
        options[SEPARATOR_KEY] = SEPARATOR_LABEL;
        for (const name of otherInstalled) {
            options[name] = name;
        }
    }

    return options;
}

export { SEPARATOR_KEY, SEPARATOR_LABEL };

export class LilbeeSettingTab extends PluginSettingTab {
    plugin: LilbeePlugin;

    constructor(app: App, plugin: LilbeePlugin) {
        super(app, plugin);
        this.plugin = plugin;
    }

    display(): void {
        const { containerEl } = this;
        containerEl.empty();

        this.renderConnectionSettings(containerEl);
        this.renderModelsSection(containerEl);
        this.renderGeneralSettings(containerEl);
        this.renderSyncSettings(containerEl);
    }

    private renderConnectionSettings(containerEl: HTMLElement): void {
        const serverSetting = new Setting(containerEl)
            .setName("Server URL")
            .setDesc("Address of the lilbee HTTP server")
            .addText((text) =>
                text
                    .setPlaceholder("http://127.0.0.1:7433")
                    .setValue(this.plugin.settings.serverUrl)
                    .onChange(async (value) => {
                        this.plugin.settings.serverUrl = value;
                        await this.plugin.saveSettings();
                    }),
            );

        const serverStatusEl = serverSetting.settingEl.createEl("span", { cls: "lilbee-health-status" });
        void this.checkEndpoint(`${this.plugin.settings.serverUrl}/api/health`, serverStatusEl);

        serverSetting.addButton((btn) =>
            btn.setButtonText("Test").onClick(async () => {
                await this.checkEndpoint(`${this.plugin.settings.serverUrl}/api/health`, serverStatusEl);
            }),
        );

        const ollamaSetting = new Setting(containerEl)
            .setName("Ollama URL")
            .setDesc("Address of the Ollama server")
            .addText((text) =>
                text
                    .setPlaceholder("http://127.0.0.1:11434")
                    .setValue(this.plugin.settings.ollamaUrl)
                    .onChange(async (value) => {
                        this.plugin.settings.ollamaUrl = value;
                        await this.plugin.saveSettings();
                    }),
            );

        const ollamaStatusEl = ollamaSetting.settingEl.createEl("span", { cls: "lilbee-health-status" });
        void this.checkEndpoint(this.plugin.settings.ollamaUrl, ollamaStatusEl);

        ollamaSetting.addButton((btn) =>
            btn.setButtonText("Test").onClick(async () => {
                await this.checkEndpoint(this.plugin.settings.ollamaUrl, ollamaStatusEl);
            }),
        );
    }

    private renderModelsSection(containerEl: HTMLElement): void {
        containerEl.createEl("h3", { text: "Models" });
        containerEl.createEl("p", {
            text: "Manage chat and vision models. Requires the lilbee server to be running.",
            cls: "setting-item-description",
        });

        const modelsContainer = containerEl.createDiv(CLS_MODELS_CONTAINER);
        new Setting(containerEl)
            .setName("Refresh models")
            .setDesc("Fetch available models from the server")
            .addButton((btn) =>
                btn.setButtonText("Refresh").onClick(async () => {
                    await this.loadModels(modelsContainer);
                }),
            );

        this.loadModels(modelsContainer);
    }

    private renderGeneralSettings(containerEl: HTMLElement): void {
        new Setting(containerEl)
            .setName("Results count")
            .setDesc("Number of search results to return")
            .addSlider((slider) =>
                slider
                    .setLimits(1, 20, 1)
                    .setValue(this.plugin.settings.topK)
                    .setDynamicTooltip()
                    .onChange(async (value) => {
                        this.plugin.settings.topK = value;
                        await this.plugin.saveSettings();
                    }),
            );
    }

    private renderSyncSettings(containerEl: HTMLElement): void {
        new Setting(containerEl)
            .setName("Sync mode")
            .setDesc("How vault changes are synced to the knowledge base")
            .addDropdown((dropdown) =>
                dropdown
                    .addOption("manual", "Manual (command only)")
                    .addOption("auto", "Auto (watch for changes)")
                    .setValue(this.plugin.settings.syncMode)
                    .onChange(async (value) => {
                        this.plugin.settings.syncMode = value as "manual" | "auto";
                        await this.plugin.saveSettings();
                        this.display();
                    }),
            );

        if (this.plugin.settings.syncMode === "auto") {
            new Setting(containerEl)
                .setName("Sync debounce")
                .setDesc("Delay in ms before syncing after a change")
                .addText((text) =>
                    text
                        .setPlaceholder("5000")
                        .setValue(String(this.plugin.settings.syncDebounceMs))
                        .onChange(async (value) => {
                            const num = parseInt(value, 10);
                            if (!isNaN(num) && num >= 0) {
                                this.plugin.settings.syncDebounceMs = num;
                                await this.plugin.saveSettings();
                            }
                        }),
                );
        }
    }

    async checkEndpoint(url: string, statusEl: HTMLSpanElement): Promise<void> {
        statusEl.setText("checking...");
        statusEl.classList.remove("lilbee-health-ok", "lilbee-health-error");
        try {
            const controller = new AbortController();
            const timeout = setTimeout(() => controller.abort(), CHECK_TIMEOUT_MS);
            const response = await fetch(url, { signal: controller.signal });
            clearTimeout(timeout);
            statusEl.setText(response.ok ? " \u2713 reachable" : ` \u2717 ${response.status}`);
            statusEl.classList.add(response.ok ? "lilbee-health-ok" : "lilbee-health-error");
        } catch {
            statusEl.setText(" \u2717 not reachable");
            statusEl.classList.add("lilbee-health-error");
        }
    }

    private async loadModels(container: HTMLElement): Promise<void> {
        container.empty();
        try {
            const models = await this.plugin.api.listModels();
            this.renderModelSection(container, "Chat Model", models.chat, "chat");
            this.renderModelSection(container, "Vision Model", models.vision, "vision");
        } catch {
            container.createEl("p", {
                text: "Could not connect to lilbee server. Is it running?",
                cls: "mod-warning",
            });
        }
    }

    private renderModelSection(
        container: HTMLElement,
        label: string,
        catalog: ModelsResponse["chat"],
        type: "chat" | "vision",
    ): void {
        const section = container.createDiv("lilbee-model-section");
        section.createEl("h4", { text: label });

        const activeSetting = new Setting(section)
            .setName(`Active ${type} model`)
            .setDesc(catalog.active || (type === "vision" ? "Disabled" : "Not set"));

        const options = buildModelOptions(catalog, type);

        activeSetting.addDropdown((dropdown) =>
            dropdown
                .addOptions(options)
                .setValue(catalog.active)
                .onChange(async (value) => {
                    if (value === SEPARATOR_KEY) return;
                    await this.handleModelChange(value, catalog, label, type, container);
                }),
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

    private async handleModelChange(
        value: string,
        catalog: ModelCatalog,
        label: string,
        type: "chat" | "vision",
        container: HTMLElement,
    ): Promise<void> {
        const uninstalledCatalogModel = catalog.catalog.find(
            (m) => m.name === value && !m.installed,
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
            new Notice(`${label} set to ${value || "disabled"}`);
        } catch {
            new Notice(`Failed to set ${type} model`);
        }
    }

    private async autoPullAndSet(
        model: ModelInfo,
        type: "chat" | "vision",
        container: HTMLElement,
    ): Promise<void> {
        new Notice(`Pulling ${model.name}...`);
        try {
            for await (const event of this.plugin.api.pullModel(model.name)) {
                if (event.event === SSE_EVENT.PROGRESS) {
                    const data = event.data as PullProgress;
                    if (data.total > 0) {
                        const pct = Math.round((data.completed / data.total) * 100);
                        if (this.plugin.onProgress) {
                            this.plugin.onProgress({
                                event: SSE_EVENT.PULL,
                                data: {
                                    model: model.name,
                                    current: data.completed,
                                    total: data.total,
                                },
                            });
                        }
                        if (this.plugin.statusBarEl) {
                            this.plugin.statusBarEl.setText(
                                `lilbee: pulling ${model.name} — ${pct}%`,
                            );
                        }
                    }
                }
            }
            if (this.plugin.onProgress) {
                this.plugin.onProgress({ event: SSE_EVENT.DONE, data: {} });
            }
            if (type === "chat") {
                await this.plugin.api.setChatModel(model.name);
            } else {
                await this.plugin.api.setVisionModel(model.name);
            }
            new Notice(`Model ${model.name} pulled and activated`);
            this.plugin.fetchActiveModel();
            const modelsContainer = this.containerEl.querySelector(`.${CLS_MODELS_CONTAINER}`);
            if (modelsContainer instanceof HTMLElement) {
                await this.loadModels(modelsContainer);
            }
        } catch {
            new Notice(`Failed to pull ${model.name}`);
            if (this.plugin.onProgress) {
                this.plugin.onProgress({ event: SSE_EVENT.DONE, data: {} });
            }
        }
    }

    private renderCatalogRow(
        table: HTMLTableElement,
        model: ModelInfo,
        type: "chat" | "vision",
    ): void {
        const row = table.createEl("tr");
        row.createEl("td", { text: model.name });
        row.createEl("td", { text: `${model.size_gb} GB` });
        row.createEl("td", { text: model.description });
        const actionCell = row.createEl("td");

        if (model.installed) {
            actionCell.createEl("span", { text: "Installed", cls: "lilbee-installed" });
        } else {
            const btn = actionCell.createEl("button", { text: "Pull" });
            btn.addEventListener("click", () => this.pullModel(btn, actionCell, model, type));
        }
    }

    private async pullModel(
        btn: HTMLElement,
        actionCell: HTMLElement,
        model: ModelInfo,
        type: "chat" | "vision",
    ): Promise<void> {
        (btn as HTMLButtonElement).disabled = true;
        btn.textContent = "Pulling...";
        try {
            const progress = actionCell.createDiv("lilbee-pull-progress");
            for await (const event of this.plugin.api.pullModel(model.name)) {
                if (event.event === SSE_EVENT.PROGRESS) {
                    const data = event.data as PullProgress;
                    if (data.total > 0) {
                        const pct = Math.round((data.completed / data.total) * 100);
                        progress.textContent = `${pct}%`;
                        if (this.plugin.statusBarEl) {
                            this.plugin.statusBarEl.setText(`lilbee: pulling ${model.name} — ${pct}%`);
                        }
                    }
                }
            }
            new Notice(`Model ${model.name} pulled successfully`);
            if (type === "chat") {
                await this.plugin.api.setChatModel(model.name);
            } else {
                await this.plugin.api.setVisionModel(model.name);
            }
            this.plugin.fetchActiveModel();
            const modelsContainer = this.containerEl.querySelector(`.${CLS_MODELS_CONTAINER}`);
            if (modelsContainer instanceof HTMLElement) {
                await this.loadModels(modelsContainer);
            }
        } catch {
            new Notice(`Failed to pull ${model.name}`);
            (btn as HTMLButtonElement).disabled = false;
            btn.textContent = "Pull";
        }
    }
}
