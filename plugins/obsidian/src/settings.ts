import { App, Notice, PluginSettingTab, Setting } from "obsidian";
import type LilbeePlugin from "./main";
import { SSE_EVENT } from "./types";
import type { ModelInfo, ModelsResponse, PullProgress } from "./types";

export class LilbeeSettingTab extends PluginSettingTab {
    plugin: LilbeePlugin;

    constructor(app: App, plugin: LilbeePlugin) {
        super(app, plugin);
        this.plugin = plugin;
    }

    display(): void {
        const { containerEl } = this;
        containerEl.empty();

        // Server URL
        new Setting(containerEl)
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

        // Top K
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

        // Sync mode
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
                        this.display(); // refresh to show/hide debounce
                    }),
            );

        // Sync debounce (only when auto)
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

        // Model sections
        containerEl.createEl("h3", { text: "Models" });
        containerEl.createEl("p", {
            text: "Manage chat and vision models. Requires the lilbee server to be running.",
            cls: "setting-item-description",
        });

        // Load models button + model sections
        const modelsContainer = containerEl.createDiv("lilbee-models-container");
        new Setting(containerEl)
            .setName("Refresh models")
            .setDesc("Fetch available models from the server")
            .addButton((btn) =>
                btn.setButtonText("Refresh").onClick(async () => {
                    await this.loadModels(modelsContainer);
                }),
            );

        // Auto-load models
        this.loadModels(modelsContainer);
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

        // Active model dropdown
        const activeSetting = new Setting(section)
            .setName(`Active ${type} model`)
            .setDesc(catalog.active || (type === "vision" ? "Disabled" : "Not set"));

        const options: Record<string, string> = {};
        if (type === "vision") {
            options[""] = "Disabled";
        }
        for (const name of catalog.installed) {
            options[name] = name;
        }

        activeSetting.addDropdown((dropdown) =>
            dropdown
                .addOptions(options)
                .setValue(catalog.active)
                .onChange(async (value) => {
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
                }),
        );

        // Catalog table
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
            btn.addEventListener("click", async () => {
                btn.disabled = true;
                btn.textContent = "Pulling...";
                try {
                    const progress = actionCell.createDiv("lilbee-pull-progress");
                    for await (const event of this.plugin.api.pullModel(model.name)) {
                        if (event.event === SSE_EVENT.PROGRESS) {
                            const data = event.data as PullProgress;
                            if (data.total > 0) {
                                const pct = Math.round((data.completed / data.total) * 100);
                                progress.textContent = `${pct}%`;
                            }
                        }
                    }
                    new Notice(`Model ${model.name} pulled successfully`);
                    // Auto-select the pulled model
                    if (type === "chat") {
                        await this.plugin.api.setChatModel(model.name);
                    } else {
                        await this.plugin.api.setVisionModel(model.name);
                    }
                    // Refresh the models display
                    const modelsContainer = this.containerEl.querySelector(
                        ".lilbee-models-container",
                    );
                    if (modelsContainer instanceof HTMLElement) {
                        await this.loadModels(modelsContainer);
                    }
                } catch {
                    new Notice(`Failed to pull ${model.name}`);
                    btn.disabled = false;
                    btn.textContent = "Pull";
                }
            });
        }
    }
}
