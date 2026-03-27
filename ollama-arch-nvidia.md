# Ollama on Arch Linux with NVIDIA RTX 2070

## Prerequisites

```bash
# Ensure system is up to date
sudo pacman -Syu
```

## Step 1: Install NVIDIA drivers

The RTX 2070 uses the Turing architecture (compute capability 7.5). You need the proprietary NVIDIA driver.

```bash
# Install the NVIDIA driver and CUDA toolkit
sudo pacman -S nvidia nvidia-utils cuda

# Reboot to load the new driver
sudo reboot
```

After reboot, verify the driver is working:

```bash
nvidia-smi
```

You should see your RTX 2070 listed with driver version and CUDA version.

## Step 2: Install Ollama

```bash
# Option A: Official install script (recommended — handles systemd service setup)
curl -fsSL https://ollama.com/install.sh | sh

# Option B: From AUR (if you prefer AUR packages)
# Using your AUR helper of choice, e.g.:
yay -S ollama
# or
paru -S ollama
```

## Step 3: Enable and start the service

```bash
sudo systemctl enable ollama
sudo systemctl start ollama
```

## Step 4: Verify GPU detection

```bash
ollama serve  # if not using systemd, or check logs:
journalctl -u ollama -f
```

Look for a line like:

```
msg="inference compute" ... library=cuda compute=7.5 driver=... name="NVIDIA GeForce RTX 2070"
```

This confirms Ollama detected your GPU and will use CUDA acceleration.

## Step 5: Pull a model and test

```bash
# Pull the default model used by lilbee
ollama pull mistral
ollama pull nomic-embed-text

# Quick test
ollama run mistral "Hello, are you using my GPU?"
```

While the model is running, you can verify GPU usage in another terminal:

```bash
nvidia-smi
# You should see ollama_llama_server (or similar) in the process list using GPU memory
```

## Troubleshooting

**`nvidia-smi` not found or shows no devices:**
- Ensure you rebooted after installing the driver
- Check that the `nvidia` kernel module is loaded: `lsmod | grep nvidia`
- If using a custom kernel, you may need `nvidia-dkms` instead of `nvidia`

**Ollama doesn't detect GPU (no CUDA in logs):**
- Make sure `nvidia-utils` is installed (provides the CUDA runtime libraries Ollama needs)
- Check that `/dev/nvidia*` devices exist: `ls /dev/nvidia*`
- Try restarting the ollama service: `sudo systemctl restart ollama`

**Out of memory errors:**
- RTX 2070 has 8GB VRAM. Models up to ~7B parameters (like mistral) fit comfortably
- For larger models, use quantized versions (e.g., `ollama pull llama3:8b-q4_0`)
- Check VRAM usage: `nvidia-smi`

**Secure boot issues:**
- If secure boot is enabled, unsigned NVIDIA modules won't load
- Either disable secure boot in BIOS or sign the modules with `mokutil`
