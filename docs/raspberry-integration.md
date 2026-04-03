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

Use the containerized (Docker) version. OpenClaw handles long-running or complex task delegation from the bridge.

```bash
git clone https://github.com/openclaw/openclaw.git ~/openclaw
cd ~/openclaw

export OPENCLAW_IMAGE="ghcr.io/openclaw/openclaw:latest"
./scripts/docker/setup.sh
```

**Save the bearer token** shown during setup — it goes into your `.env` as `OPENCLAW_BEARER_TOKEN`.

To access the OpenClaw web UI, forward the port via SSH tunnel:

```bash
ssh -fN -L 18789:127.0.0.1:18789 dobby@raspberrypi.local
# Then open: http://localhost:18789
```

To access the OpenClaw CLI:

```bash
docker ps                              # find the container name/ID
docker exec -it <container_name> bash
```

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

After setup, generate a **Long-Lived Access Token** in the HA UI (Profile → Security) and save it as `HA_TOKEN` in your `.env`.

---

## Reachy Mini SDK (Native Install)

The Reachy daemon requires direct access to USB, microphone, and camera hardware. Native installation is simpler and more reliable than Docker for this component.

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
ExecStart=/home/dobby/reachy_mini_env/bin/reachy-mini-daemon
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
```

---

## Deploying the Bridge

Clone the repository and configure the environment:

```bash
git clone <repo-url> ~/dobby-the-claw
cd ~/dobby-the-claw
cp .env.example .env
nano .env  # Fill in OPENAI_API_KEY, REACHY_BRIDGE_URL, OPENCLAW_*, HA_* etc.
```

Run via Docker:

```bash
docker compose up --build -d

# View logs
docker compose logs -f
```

The `REACHY_BRIDGE_URL` in `.env` should point to the local Reachy daemon WebSocket (typically `ws://localhost:<port>`).
