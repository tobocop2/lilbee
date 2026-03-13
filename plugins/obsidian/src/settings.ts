import { App, Notice, PluginSettingTab, Setting } from "obsidian";
import { OLLAMA_STATE } from "./ollama-detector";
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

        // Ollama warning banner
        if (this.plugin.ollamaDetector?.state === OLLAMA_STATE.UNREACHABLE) {
            const warning = containerEl.createDiv({ cls: "lilbee-ollama-warning" });
            warning.createEl("p", {
                text: "Ollama is not running. Sync, ask, and chat will not work.",
            });
            warning.createEl("p", {
                text: "Start Ollama or install it from ollama.com",
            });
        }

        // Connection settings
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

        new Setting(containerEl)
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

        // Models section — promoted to top after connection settings
        containerEl.createEl("h3", { text: "Models" });
        containerEl.createEl("p", {
            text: "Manage chat and vision models. Requires the lilbee server to be running.",
            cls: "setting-item-description",
        });

        const modelsContainer = containerEl.createDiv("lilbee-models-container");
        new Setting(containerEl)
            .setName("Refresh models")
            .setDesc("Fetch available models from the server")
            .addButton((btn) =>
                btn.setButtonText("Refresh").onClick(async () => {
                    await this.loadModels(modelsContainer);
                }),
            );

        this.loadModels(modelsContainer);

        // General settings
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

        // Manage server
        new Setting(containerEl)
            .setName("Manage server")
            .setDesc("Start and stop the lilbee server with Obsidian")
            .addToggle((toggle) =>
                toggle.setValue(this.plugin.settings.manageServer).onChange(async (value) => {
                    this.plugin.settings.manageServer = value;
                    await this.plugin.saveSettings();
                    this.display();
                }),
            );

        if (this.plugin.settings.manageServer) {
            new Setting(containerEl)
                .setName("Binary path")
                .setDesc("Path to the lilbee binary (leave empty to auto-detect)")
                .addText((text) =>
                    text
                        .setPlaceholder("auto-detect")
                        .setValue(this.plugin.settings.binaryPath)
                        .onChange(async (value) => {
                            this.plugin.settings.binaryPath = value;
                            await this.plugin.saveSettings();
                        }),
                );

            new Setting(containerEl)
                .setName("Restart server")
                .setDesc("Restart the managed lilbee server")
                .addButton((btn) =>
                    btn.setButtonText("Restart").onClick(async () => {
                        await this.plugin.restartServer();
                    }),
                );
        }

        // Sync settings
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
                                if (this.plugin.statusBarEl) {
                                    this.plugin.statusBarEl.setText(
                                        `lilbee: pulling ${model.name} — ${pct}%`,
                                    );
                                }
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
                    // Refresh status bar with new model
                    this.plugin.fetchActiveModel();
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
