# Raspberry Pi Integration

## Hardware Requirements

- Raspberry Pi 4 Model B with **8 GB RAM** (minimum)
- OS: **Ubuntu Server 22.04 LTS (x64)** — newer versions do not have full support yet
- Storage: **USB SSD or USB drive** — do not use an SD card. SD cards are not designed for continuous OS-level writes and will fail prematurely

[How to install Ubuntu on a Raspberry Pi](https://ubuntu.com/download/raspberry-pi)

---

## Initial SSH Access

After installation, access the Pi via SSH. If you have a static IP reservation configured on your router (recommended):

```bash
ssh dobby@raspberrypi.local
```

> **Mesh network note:** mDNS (`hostname.local`) is unreliable on mesh networks because multicast packets are often not forwarded between nodes. If it doesn't resolve, add an entry to your local `/etc/hosts` pointing to the static IP:
> ```
> 192.168.68.56   raspberrypi.local
> ```

---

## Base Dependencies

Install Git and Docker:

```bash
# Git
sudo apt-get install git-all

# Docker (official install script)
curl -fsSL https://get.docker.com | sh

# Add current user to docker group (re-login required to take effect)
sudo usermod -aG docker $USER
```

Log out and back in, then verify: `docker ps`

---

## OpenClaw

OpenClaw handles long-running or complex task delegation from the bridge. It runs natively on the Pi via Node.js — no Docker required.

### Install Node.js 22 via nvm

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash
source ~/.bashrc
nvm install 22
corepack enable
```

### Install OpenClaw

```bash
npm install -g openclaw@latest
```

Verify: `openclaw --version`

### Install channel plugins

The WhatsApp channel is distributed as a separate plugin (not bundled in the npm package):

```bash
openclaw plugins install @openclaw/whatsapp
```

Restart the gateway after installing plugins.

### Configure environment

Create the native environment file:

```bash
mkdir -p ~/.config/openclaw
nano ~/.config/openclaw/native.env
```

Minimum required variables:

```env
OPENAI_API_KEY=sk-...
OPENCLAW_GATEWAY_TOKEN=<your-token>        # generate with: openssl rand -hex 32
OPENCLAW_GATEWAY_PORT=18789
OPENCLAW_TZ=America/Sao_Paulo
OPENCLAW_SANDBOX=0
```

**Save the gateway token** — it goes into `dobby-the-claw/.env` as `OPENCLAW_BEARER_TOKEN`.

> OpenClaw stores its config and workspace at `~/.openclaw/`. This directory is created automatically on first run. To configure skills, exec settings, and model preferences, edit `~/.openclaw/openclaw.json` directly or use the web UI after connecting.

### Run as a systemd service

Create the service unit:

```bash
sudo nano /etc/systemd/system/openclaw.service
```

Paste:

```ini
[Unit]
Description=OpenClaw Gateway
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=dobby
WorkingDirectory=/home/dobby/.openclaw
EnvironmentFile=/home/dobby/.config/openclaw/native.env
Environment="PATH=/home/dobby/.nvm/versions/node/v22.22.3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStart=/home/dobby/.nvm/versions/node/v22.22.3/bin/openclaw gateway
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=openclaw

[Install]
WantedBy=multi-user.target
```

> Update the Node.js version path if you install a different version: `nvm which 22` shows the exact binary path.

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable openclaw
sudo systemctl start openclaw

# Verify
curl http://localhost:18789/healthz   # should return {"ok":true,"status":"live"}
journalctl -u openclaw -f            # follow logs
```

### Update OpenClaw

```bash
npm install -g openclaw@latest
sudo systemctl restart openclaw
```

### Web UI access

The web UI requires a secure browser context (HTTPS or localhost) to generate the device identity key used for authentication. Use the SSH tunnel so the browser sees `localhost`:

```bash
# From your development machine (tunnel already included in ./scripts/tunnel.sh)
ssh -N -L 18789:127.0.0.1:18789 dobby@192.168.68.56
```

Then open: `http://127.0.0.1:18789`

**First-time connection:** open in an **incognito/private window** so the browser generates a fresh device identity. Enter the gateway token from `~/.openclaw/openclaw.json` → `gateway.auth.token` in the Gateway Token field, or append it as a URL fragment:

```
http://127.0.0.1:18789/#token=<your-gateway-token>
```

After the first successful connection, the device identity is stored in the browser's IndexedDB and subsequent connections in regular windows work without the fragment.

---

## Home Assistant

Run Home Assistant using the provided Docker Compose file. The config maps `/dev/ttyUSB0` for a Zigbee dongle (ZBDongle-E); remove that line if not needed.

```bash
docker compose -f docker-compose-ha.yml up -d
```

Access the web UI via SSH tunnel:

```bash
ssh -fN -L 8123:127.0.0.1:8123 dobby@raspberrypi.local
# Then open: http://localhost:8123
```

**Install HACS** (Home Assistant Community Store):

```bash
sudo mkdir -p ~/homeassistant/config/custom_components
sudo chown -R dobby:dobby ~/homeassistant/config/custom_components
wget -O - https://get.hacs.xyz | bash -
```

After setup, generate a **Long-Lived Access Token** in the HA UI (Profile -> Security) and save it as `HOME_ASSISTANT_TOKEN` in your `.env`.

---

## Reachy Mini SDK (Native Install)

The Reachy daemon requires direct access to USB, microphone, and camera hardware. Native installation is simpler and more reliable than Docker for this component.

This repository also ships a small daemon wrapper at `scripts/reachy-mini-daemon-wrapper.py`. Keep using it on Raspberry Pi for now: current `reachy-mini` camera detection can miss the V4L2 camera on this Ubuntu setup unless the GLib loop runs briefly and `device.path` is normalized to `api.v4l2.path` before startup.

```bash
# Step 1 — System dependencies
sudo apt install git git-lfs libportaudio2
git lfs install

# Step 2 — Install uv (Python version manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Step 3 — Install Python 3.12
uv python install 3.12 --default

# Step 4 — Create virtualenv and install the SDK
uv venv reachy_mini_env --python 3.12
source reachy_mini_env/bin/activate
uv pip install "reachy-mini"

# Step 5 — USB permissions for Reachy hardware
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", MODE="0666", GROUP="dialout"
SUBSYSTEM=="usb", ATTRS{idVendor}=="38fb", ATTRS{idProduct}=="1001", MODE="0666", GROUP="dialout"' \
| sudo tee /etc/udev/rules.d/99-reachy-mini.rules

sudo udevadm control --reload-rules && sudo udevadm trigger
sudo usermod -aG dialout $USER

# Step 6 — GStreamer base packages
sudo apt-get update
sudo apt-get install -y \
    libgstreamer-plugins-bad1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer1.0-dev \
    libglib2.0-dev \
    libssl-dev \
    libgirepository1.0-dev \
    libcairo2-dev \
    libportaudio2 \
    libnice10 \
    gstreamer1.0-plugins-good \
    gstreamer1.0-alsa \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-nice \
    python3-gi \
    python3-gi-cairo

# Step 7 — Upgrade GStreamer to 1.24 (required on Ubuntu 22.04)
sudo add-apt-repository ppa:savoury1/multimedia
sudo apt update
sudo apt install \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-good1.0-dev \
    libgstreamer-plugins-bad1.0-dev

# Step 8 — Install Rust (required to compile the WebRTC plugin)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Step 9 — Compile the GStreamer WebRTC plugin
git clone https://gitlab.freedesktop.org/gstreamer/gst-plugins-rs.git
cd gst-plugins-rs
git checkout 0.14.1
cargo install cargo-c
sudo mkdir -p /opt/gst-plugins-rs
sudo chown $USER /opt/gst-plugins-rs
cargo cinstall -p gst-plugin-webrtc --prefix=/opt/gst-plugins-rs --release

# Step 10 — Add plugin path to shell profile (ARM64)
echo 'export GST_PLUGIN_PATH=/opt/gst-plugins-rs/lib/aarch64-linux-gnu:$GST_PLUGIN_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Reachy Daemon as a systemd Service

Clone this repository on the Pi so the service can use the tracked wrapper:

```bash
git clone <repo-url> ~/dobby-the-claw
```

To ensure the daemon starts automatically on boot and restarts on failure:

```bash
sudo nano /etc/systemd/system/reachy-mini-daemon.service
```

Paste the following content:

```ini
[Unit]
Description=Reachy Mini Daemon
After=network.target
StartLimitIntervalSec=0

[Service]
Type=simple
User=dobby
WorkingDirectory=/home/dobby
Environment="PATH=/home/dobby/reachy_mini_env/bin:/opt/gstreamer/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="GST_PLUGIN_PATH=/opt/gst-plugins-rs/lib/aarch64-linux-gnu:/opt/gstreamer/lib/aarch64-linux-gnu/gstreamer-1.0"
Environment="LD_LIBRARY_PATH=/opt/gstreamer/lib/aarch64-linux-gnu"
ExecStart=/home/dobby/reachy_mini_env/bin/python3 /home/dobby/dobby-the-claw/scripts/reachy-mini-daemon-wrapper.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable reachy-mini-daemon
sudo systemctl start reachy-mini-daemon

# Check status
sudo systemctl status reachy-mini-daemon

# View logs
journalctl -u reachy-mini-daemon -f
```

### Persisting Maximum Speaker Volume

Install ALSA tools once so the Raspberry can manage the Reachy USB audio mixer:

```bash
sudo apt-get update
sudo apt-get install -y alsa-utils
```

Set the bridge startup volume to maximum in `~/dobby-the-claw/.env`:

```bash
REACHY_OUTPUT_VOLUME=100
```

To force the Reachy speaker mixer back to `100%` on every Raspberry reboot, install the tracked helper script and systemd unit:

```bash
chmod +x ~/dobby-the-claw/scripts/set-reachy-audio-max.sh
sudo cp ~/dobby-the-claw/scripts/reachy-audio-max.service /etc/systemd/system/reachy-audio-max.service
sudo systemctl daemon-reload
sudo systemctl enable --now reachy-audio-max.service
```

The unit waits for the Reachy USB audio card and then drives both playback mixer controls (`PCM` and `PCM,1`) to `100%`.

---

## Deploying the Bridge

If you haven't cloned the repository yet:

```bash
git clone <repo-url> ~/dobby-the-claw
cd ~/dobby-the-claw
cp .env.example .env
nano .env  # Fill in OPENAI_API_KEY, REACHY_BRIDGE_URL, OPENCLAW_*, HA_* etc.
```

On Raspberry Pi, set `REACHY_OUTPUT_VOLUME=100` in `.env` so the bridge re-applies max output volume whenever it starts.

Run via Docker:

```bash
docker compose up --build -d

# View logs
docker compose logs -f
```

On Raspberry Pi, the bridge container uses two host integrations for local media:
- `/tmp` is mounted so the Reachy local IPC camera socket remains visible inside Docker.
- `/dev/snd` is exposed so the SDK can use ALSA for microphone and speaker access.

At container startup, the bridge now writes `/root/.asoundrc` automatically so ALSA defaults point to the Reachy USB audio card (`Audio`). This avoids Docker-local GStreamer device discovery issues where the SDK cannot enumerate the microphone/speaker directly but can still use the correct card through the default ALSA route.

The bridge also defaults `REACHY_DIRECT_ALSA_AUDIO=1` on Linux. On Raspberry Pi this bypasses the SDK's local GStreamer audio path entirely for microphone capture and speaker playback, because the custom Reachy GStreamer runtime falls back to fake audio source/sink elements instead of real hardware backends.

If you restart `reachy-mini-daemon`, wait until the service is fully `active` again before restarting the bridge container. The Docker bridge relies on the local IPC camera socket at `/tmp/reachymini_camera_socket`, and restarting the bridge while that socket is still absent can make container startup fail.

The `REACHY_BRIDGE_URL` in `.env` should point to the local Reachy daemon WebSocket (typically `ws://localhost:<port>`).
