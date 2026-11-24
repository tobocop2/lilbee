# Installing lilbee

Use a package manager if you have one, so updates come through it. If you have none, the install script fetches the right build for your hardware.

```sh
# macOS / Linux
curl -fsSL https://lilbee.sh/install.sh | sh
```

```powershell
# Windows
irm https://lilbee.sh/install.ps1 | iex
```

Then check that it runs:

```bash
lilbee self-check    # ~90 MB download; runs an inference and an embedding
lilbee               # start the terminal app and pick your models
```

## Two kinds of install

- **A self-contained bundle** (start here). This is the standalone binary, the install script, and the Homebrew, AUR, Nix, Docker, Flatpak, Snap, Scoop and npm builds. Each one carries its own Python, `llama.cpp` and all the optional extras. It is one large download, and the first start unpacks it once.
- **Into your own Python** with `pip` or `uv`. This is the developer route. You manage the environment, and the model engine is a separate extra from a separate package index. See [pip and uv](#pip-and-uv).

## Every channel

| How                   | Command                                                                                  | Notes |
| --------------------- | ---------------------------------------------------------------------------------------- | ----- |
| **Homebrew**          | `brew tap tobocop2/lilbee && brew install lilbee`                                        | macOS (Apple silicon and Intel) and Linux x86_64. Clears the macOS quarantine flag for you. |
| **AUR**               | `paru -S lilbee`                                                                         | Arch Linux. Works with `yay`, `pacaur` or any helper. |
| **Docker**            | `docker run --rm -v lilbee-data:/home/lilbee/data ghcr.io/tobocop2/lilbee:latest --help` | GHCR image, tagged by version and `latest`. Mount a volume at `/home/lilbee/data`. The `:cuda` and `:rocm` tags carry the vendor builds. |
| **Nix**               | `nix run github:tobocop2/lilbee`                                                         | NixOS, nix-darwin, or any host with nix. On Linux the flake bundles `glibc`, `libgomp` and `vulkan-loader`, so it runs on bare NixOS. |
| **Flatpak**           | `flatpak remote-add --if-not-exists lilbee https://tobocop2.github.io/flatpak-lilbee/lilbee.flatpakrepo && flatpak install lilbee io.github.tobocop2.lilbee` | Linux x86_64. Needs the [Flathub remote](https://flathub.org/setup) for the runtime. Run it with `flatpak run io.github.tobocop2.lilbee`, or add `alias lilbee='flatpak run io.github.tobocop2.lilbee'` to your shell rc. Data lives under `~/.var/app/io.github.tobocop2.lilbee/`. A single-file `lilbee.flatpakref` is on the [release page](https://github.com/tobocop2/lilbee/releases/latest). |
| **Snap**              | `curl -LO https://github.com/tobocop2/lilbee/releases/latest/download/lilbee-linux-x86_64.snap && sudo snap install ./lilbee-linux-x86_64.snap --dangerous --classic` | Linux x86_64. The snap is sideloaded, so snapd calls it `--dangerous` (unsigned) and it does not auto-update. Run the same command again to upgrade. |
| **Scoop**             | `scoop install git; scoop bucket add lilbee https://github.com/tobocop2/scoop-lilbee; scoop install lilbee` | Windows x86_64. Scoop needs git to add a bucket. The install reads your NVIDIA driver and picks the CUDA build (555.85+ gets cu125, 551.61+ gets cu124), else the Vulkan build. If you added the bucket before 2026-09-08, run `scoop bucket rm lilbee` and add it again. |
| **npm**               | `npm install -g lilbee`                                                                  | macOS, Linux x86_64 and Windows, Node 18+. A small launcher: on first run it detects your hardware, downloads the matching binary, verifies it and caches it. `lilbee prepare` does this ahead of time. `npx -y lilbee mcp` runs the MCP server for agent hosts. |
| **Install script**    | `curl -fsSL https://lilbee.sh/install.sh \| sh`<br>`irm https://lilbee.sh/install.ps1 \| iex` | Detects your hardware (CUDA, ROCm, Metal or the AVX baseline) and refuses a download whose SHA-256 does not match the release's `SHA256SUMS`. Installs one file to `~/.local/bin` (Windows: `%LOCALAPPDATA%\Programs\lilbee`) and adds it to your PATH. `LILBEE_NO_MODIFY_PATH=1` skips the PATH change. `LILBEE_INSTALL_DIR`, `LILBEE_RELEASE` and `LILBEE_VARIANT` set the folder, the release and the build. Run it again to upgrade. To uninstall, delete the file. |
| **Standalone binary** | [download for your platform](https://github.com/tobocop2/lilbee/releases/latest)        | One file with its own Python. Linux needs glibc 2.34+. The macOS and Windows builds are unsigned: if Gatekeeper blocks a browser download, run `xattr -d com.apple.quarantine ./lilbee-macos-arm64`. |

## NVIDIA (CUDA)

The default Vulkan build works on NVIDIA cards. The CUDA build is faster, and it avoids the Vulkan loader crash on Windows machines with both an iGPU and a dGPU. The CUDA runtime is bundled, so you need only the NVIDIA driver.

|              | Command |
| ------------ | ------- |
| **Homebrew** | `brew install tobocop2/lilbee/lilbee-cuda` |
| **AUR**      | `paru -S lilbee-cuda` |
| **Nix**      | `nix run github:tobocop2/lilbee#lilbee-cuda` |
| **Flatpak**  | `flatpak install lilbee io.github.tobocop2.lilbee.cuda` |
| **Scoop**    | `scoop install lilbee-cuda` (cu125 in one download, driver 555.85+. The plain `lilbee` install also picks CUDA, but downloads the Vulkan build first.) |
| **Binary**   | [`lilbee-linux-x86_64-cu125`](https://github.com/tobocop2/lilbee/releases/latest) or [`lilbee-windows-x86_64-cu125.exe`](https://github.com/tobocop2/lilbee/releases/latest) |
| **pip**      | `pip install --pre 'lilbee[engine]' --extra-index-url https://lilbee.sh/cu125/` |
| **uv**       | `uv tool install --prerelease=allow 'lilbee[engine]' --extra-index-url https://lilbee.sh/cu125/` |

If the regular `lilbee` is already installed: on AUR, `paru -S lilbee-cuda` replaces it. On Homebrew and Scoop, uninstall `lilbee` first. For an older driver, `cu124` and `cu121` ship on the matching package indexes and as Linux binaries on the release page.

## AMD (ROCm)

The default Vulkan build works on AMD cards and stays the fallback when ROCm is not set up. The ROCm build is faster. It bundles the ROCm userspace, so you need only the amdgpu kernel driver. Linux only.

|              | Command |
| ------------ | ------- |
| **Homebrew** | `brew install tobocop2/lilbee/lilbee-rocm` |
| **AUR**      | `paru -S lilbee-rocm` |
| **Nix**      | `nix run github:tobocop2/lilbee#lilbee-rocm` |
| **Flatpak**  | `flatpak install lilbee io.github.tobocop2.lilbee.rocm` |
| **Binary**   | [`lilbee-linux-x86_64-rocm`](https://github.com/tobocop2/lilbee/releases/latest) |
| **pip**      | `pip install --pre 'lilbee[engine]' --extra-index-url https://lilbee.sh/rocm/` |
| **uv**       | `uv tool install --prerelease=allow 'lilbee[engine]' --extra-index-url https://lilbee.sh/rocm/` |

Supported cards: MI100, MI200, MI300, MI350, RDNA2, RDNA3, RDNA3.5 APUs and RDNA4. ROCm 7 has no GEMM kernels for the MI50, so the MI50 needs the Vulkan build. The ROCm userspace and the kernels for each card ship inside the build.

## Older CPUs (pre-AVX2)

Intel CPUs before 2013 and AMD CPUs before Zen lack [AVX2](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#Advanced_Vector_Extensions_2), and the normal build crashes on them at start. The `lilbee-compat` build runs on any x86-64 CPU back to about 2008.

|              | Command |
| ------------ | ------- |
| **Homebrew** | `brew install tobocop2/lilbee/lilbee-compat` |
| **AUR**      | `paru -S lilbee-compat` |
| **Nix**      | `nix run github:tobocop2/lilbee#lilbee-compat` |
| **Scoop**    | `scoop install lilbee-compat` |
| **Flatpak**  | `flatpak install lilbee io.github.tobocop2.lilbee.compat` |
| **Snap**     | `curl -LO https://github.com/tobocop2/lilbee/releases/latest/download/lilbee-compat-linux-x86_64.snap && sudo snap install ./lilbee-compat-linux-x86_64.snap --dangerous --classic` |
| **Binary**   | [`lilbee-compat-linux-x86_64`](https://github.com/tobocop2/lilbee/releases/latest), [`lilbee-compat-windows-x86_64.exe`](https://github.com/tobocop2/lilbee/releases/latest), or [`lilbee-compat-macos-x86_64`](https://github.com/tobocop2/lilbee/releases/latest) (pre-AVX2 Intel Macs, for example the 2013 Mac Pro) |
| **Binary + GPU** | [`lilbee-compat-linux-x86_64-cu124`](https://github.com/tobocop2/lilbee/releases/latest) (NVIDIA, driver 550+) or [`lilbee-compat-linux-x86_64-rocm`](https://github.com/tobocop2/lilbee/releases/latest) (AMD). These need a CPU with AVX (2011+). |
| **pip**      | `pip install --pre lilbee 'lancedb==0.40.0+compat' --extra-index-url https://lilbee.sh/compat/` (no engine, see below) |

Use a bundled build if you can. The compat package index carries the patched lancedb and the old-macOS builds below, but no engine, so the `pip` row installs lilbee without an engine. Every model call fails until you set `LILBEE_LLAMA_SERVER_PATH` to a `llama-server` that runs on your CPU.

For an older CPU with an NVIDIA GPU (AVX, 2011+), `lilbee-compat-cuda` is the compat build with CUDA 12.4. It needs NVIDIA driver 550+ on Linux and 551.61+ on Windows.

|              | Command |
| ------------ | ------- |
| **Homebrew** | `brew install tobocop2/lilbee/lilbee-compat-cuda` |
| **AUR**      | `paru -S lilbee-compat-cuda` |
| **Nix**      | `nix run github:tobocop2/lilbee#lilbee-compat-cuda` |
| **Scoop**    | `scoop install lilbee-compat-cuda` |
| **Flatpak**  | `flatpak install lilbee io.github.tobocop2.lilbee.compat-cuda` |
| **Snap**     | `curl -LO https://github.com/tobocop2/lilbee/releases/latest/download/lilbee-compat-linux-x86_64-cu124.snap && sudo snap install ./lilbee-compat-linux-x86_64-cu124.snap --dangerous --classic` |
| **Docker**   | `docker run --rm --gpus all ghcr.io/tobocop2/lilbee:compat-cuda --help` |
| **Binary**   | [`lilbee-compat-linux-x86_64-cu124`](https://github.com/tobocop2/lilbee/releases/latest) or [`lilbee-compat-windows-x86_64-cu124.exe`](https://github.com/tobocop2/lilbee/releases/latest) |

The crash comes from [lancedb](https://lancedb.github.io/lancedb/)'s AVX2 wheels. The compat build uses a [lancedb fork](https://github.com/tobocop2/lance) that picks instructions at run time. A 👍 on the upstream [lance PR](https://github.com/lance-format/lance/pull/6630) helps it land.

## On older macOS

The main Mac build needs macOS 14 on Apple silicon. On macOS 12.3 to 13, use `lilbee-compat-macos-arm64`. No build runs on an older macOS on Apple silicon. Homebrew and the npm launcher pick it for you. It bundles old-macOS builds of the crawler and the model client, which the compat index at `https://lilbee.sh/compat/` serves. Pip users on an old Mac add that index and pin the builds:

```bash
pip install --only-binary crawlberg,liter-llm 'crawlberg==1.13.0+compat' 'liter-llm==2.7.0+compat' --extra-index-url https://lilbee.sh/compat/
```

A Mac on a newer macOS never needs the index.

## Linux runtime requirements

The Linux wheel and binary bundle the Vulkan loader, so they start on any distro. GPU detection still reads adapters through the system loader: without `libvulkan1`, lilbee treats the machine as CPU-only. Most desktop distros (Ubuntu 22.04+, Pop!\_OS, Mint) ship it. On a bare image, install it once:

- Arch, Manjaro: `sudo pacman -S vulkan-icd-loader`
- Fedora, RHEL: `sudo dnf install vulkan-loader`
- Debian, Ubuntu: `sudo apt-get install libvulkan1`

## OCR

For scanned PDFs and images, install [Tesseract](https://github.com/tesseract-ocr/tesseract) (`brew install tesseract` or `apt install tesseract-ocr`), or use a [GGUF vision model](usage.md#vision-models).

## pip and uv

Use this route to work on lilbee, or to import it as a library. Python 3.11 to 3.14.

| How             | Command | Notes |
| --------------- | ------- | ----- |
| **pip**         | `pip install --pre 'lilbee[engine]' --extra-index-url https://lilbee.sh/cpu/` | Runs on any x86_64 CPU with AVX2 (2013+). Swap the index for your hardware, below. |
| **uv**          | `uv tool install --prerelease=allow 'lilbee[engine]' --extra-index-url https://lilbee.sh/cpu/` | Same wheels as pip; fetches a Python for you if you need one. |
| **From source** | `git clone https://github.com/tobocop2/lilbee && cd lilbee && uv sync && uv run lilbee` | Needs `git` and `uv`. |

`uv sync` resolves the in-repo engine package, but a source checkout has no engine binaries, because CI fills them in. Install the `[engine]` extra from an index below, or set `LILBEE_LLAMA_SERVER_PATH` to your own `llama-server`.

### The engine index

`llama-server` runs every model. It ships as the `[engine]` extra, and it is not on PyPI: the CUDA and ROCm wheels are 444 MiB to 863 MiB, above [PyPI's 100 MiB file limit](https://pypi.org/help/#file-size-limit). lilbee publishes them on its own [PEP 503](https://peps.python.org/pep-0503/) index at `lilbee.sh`. The index you pick is the build you get: `cpu` is compiled with every GPU backend off, `metal` publishes only a macOS arm64 wheel, and `vulkan` only Linux and Windows wheels.

| Hardware          | Index    | Command |
| ----------------- | -------- | ------- |
| **NVIDIA (CUDA)** | `cu125`  | `pip install --pre 'lilbee[engine]' --extra-index-url https://lilbee.sh/cu125/` |
| **AMD (ROCm)**    | `rocm`   | `pip install --pre 'lilbee[engine]' --extra-index-url https://lilbee.sh/rocm/` |
| **Apple silicon** | `metal`  | `pip install --pre 'lilbee[engine]' --extra-index-url https://lilbee.sh/metal/` |
| **Other GPUs**    | `vulkan` | `pip install --pre 'lilbee[engine]' --extra-index-url https://lilbee.sh/vulkan/` |
| **No GPU**        | `cpu`    | `pip install --pre 'lilbee[engine]' --extra-index-url https://lilbee.sh/cpu/` |

Without the extra, lilbee installs and starts. The first model call then fails with an error that names the command for your hardware.

### Optional extras

Add extras in brackets, for example `pip install --pre 'lilbee[engine,crawler,providers]'`. The bundled builds already include all four.

| Extra         | What it adds |
| ------------- | ------------ |
| `[engine]`    | The bundled `llama-server` that runs your models. Without it, no model loads. |
| `[crawler]`   | Crawl a docs site or wiki to markdown and search it offline. |
| `[providers]` | Hosted model providers for chat, vision or embeddings while other roles stay local, and Ollama or LM Studio as a backend. The TUI flags when a hosted role is active. |
| `[graph]`     | Concept-graph search: finds related ideas that keyword search misses. No extra model calls. |

The [usage guide](usage.md#optional-extras) covers how to configure each one.

## Hardware requirements

**Minimum:** an Apple silicon Mac, a 64-bit Intel or AMD CPU from 2013 or later, or an ARMv8 Linux machine, with 8 GB RAM and 2 GB disk. For older CPUs, see [Older CPUs](#older-cpus-pre-avx2).

| Platform           | Minimum | Recommended |
| ------------------ | ------- | ----------- |
| **macOS arm64**    | Apple silicon (M1 or newer), macOS 11+ | M-series Pro, Max or Ultra |
| **macOS x86_64**   | Intel Mac with AVX (Ivy Bridge or newer), macOS 12+ | Quad-core Intel with 16+ GB RAM |
| **Linux x86_64**   | 64-bit Intel or AMD from 2013+ ([`x86-64-v3`](https://en.wikipedia.org/wiki/X86-64#Microarchitecture_levels)) | Modern CPU with an NVIDIA, AMD or Intel Arc GPU |
| **Windows x86_64** | 64-bit Intel or AMD from 2013+ (`x86-64-v3`), Windows 10/11 | Modern CPU with a GPU |
| **Linux ARM64**    | ARMv8 with NEON (Raspberry Pi 4+, AWS Graviton, Ampere Altra) | Modern ARM server with 16+ GB RAM |

| Resource | Minimum | Recommended |
| -------- | ------- | ----------- |
| **RAM**  | 8 GB | 16 to 32 GB to keep chat, embedding, rerank and vision models warm together. The real footprint depends on the model sizes and quantizations you pick. |
| **GPU**  | none (CPU-only works) | Apple silicon (Metal), NVIDIA, AMD or Intel Arc (Vulkan), or the CUDA and ROCm builds |
| **Disk** | 2 GB | 10+ GB for several models |
