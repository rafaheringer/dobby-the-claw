# Raspberry  Pi integration

## Setup Raspberry Pi
É necessário pelo menos um Raspberry Pi 4 Model com 8G de RAM. Idealmente rodaddo o Ubuntu Server 22.04 LTS (x64). Versões mais novas não possuem suporte pleno para este Raspberry. [How to install Ubuntu o a Raspberry Pi](https://ubuntu.com/download/raspberry-pi).

Atenção: não use cartão SD. SD cards foram projetados para armazenamento, não para uso como disco de sistema operacional com escrita contínua.

Após a instalação, acesse o Sistema Operacional através do SSH:
```bash
ssh <user>@<ip-do-raspberry>
```

Iremos precisar do [Docker](https://docs.docker.com/engine/install/ubuntu/) e GIT:

```bash
#GIT
sudo apt-get install git-all

#Docker
curl -fsSL https://get.docker.com | sh

# Add user to docker group
sudo usermod -aG docker $USER
```


### Open Claw
Use a versão containerizada (Docker). Com docker instalado, vamos baixar o repositório do Open Claw e utilizar o script pronto para instalar.

```bash
# Git clone
git clone https://github.com/openclaw/openclaw.git ~/openclaw
cd ~/openclaw

# Install
export OPENCLAW_IMAGE="ghcr.io/openclaw/openclaw:latest"
./scripts/docker/setup.sh
```

Para acessar o CLI do OpenClaw:
```bash
# Descubra o nome/ID do container
docker ps

# Entre no container e use o CLI
docker exec -it <nome_do_container> bash
```

Para acessar a inteface do OpenClaw no navegador, basta mapear a porta remota. Lembre-se de guadar o token informado na configuração no Open Claw.
```bash
ssh -fN -L 18789:127.0.0.1:18789 <user>@<ip-do-raspberry>
```

### Home Assistant
Podemos usar a versão Docker do Home Assistant. Para isso, basta rodar o docker compose utilizando o arquivo `docker-compose-ha.yml`. Neste exemplo já estamos mapeando o device "ttyUSB0" (neste projeto é um ZBDongle-E), caso não seja necessário, remova.

```bash
docker compose up -f docker-compose-ha.yml  -d
```

Para acessar a interface, vamos precisar criar um túnel:
```bash
ssh -fN -L 8123:127.0.0.1:8123 <user>@<ip-do-raspberry>
```

E para instalar o HAC:
```bash
sudo mkdir -p ~/homeassistant/config/custom_components
sudo chown -R dobby:dobby ~/homeassistant/config/custom_components
wget -O - https://get.hacs.xyz | bash -
```

## Reachy Mini SDK
Para o daemon do Reachy, a instalação nativa pode ser mais simples já que ele precisa de acesso direto a dispositivos de hardware (USB, microfone, câmera) — Docker com múltiplos devices passthrough pode ficar mais complicado.

```bash
# Passo 1 — Instalar dependências
sudo apt install git git-lfs libportaudio2
git lfs install

# Passo 2 — Instalar o uv (gerenciador de Python)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Passo 3 — Instalar o Python 3.12
uv python install 3.12 --default

# Passo 4 — Criar o ambiente virtual e instalar o SDK
uv venv reachy_mini_env --python 3.12
source reachy_mini_env/bin/activate
uv pip install "reachy-mini"

# Passo 5 — Permissões USB para o Reachy
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", MODE="0666", GROUP="dialout"
SUBSYSTEM=="usb", ATTRS{idVendor}=="38fb", ATTRS{idProduct}=="1001", MODE="0666", GROUP="dialout"' \
| sudo tee /etc/udev/rules.d/99-reachy-mini.rules

sudo udevadm control --reload-rules && sudo udevadm trigger
sudo usermod -aG dialout $USER

# Passo 6 — Instalar o GStreamer base
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

# Passo 7 — Atualizar GStreamer para 1.24 (necessário no Ubuntu 22.04)
sudo add-apt-repository ppa:savoury1/multimedia
sudo apt update
sudo apt install \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-good1.0-dev \
    libgstreamer-plugins-bad1.0-dev

# Passo 8 — Instalar o Rust (necessário para o plugin WebRTC)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Passo 9 — Compilar o plugin WebRTC
git clone https://gitlab.freedesktop.org/gstreamer/gst-plugins-rs.git
cd gst-plugins-rs
git checkout 0.14.1
cargo install cargo-c
sudo mkdir -p /opt/gst-plugins-rs
sudo chown $USER /opt/gst-plugins-rs
cargo cinstall -p gst-plugin-webrtc --prefix=/opt/gst-plugins-rs --release

# Passo 10 — Configurar o PATH (ARM64)
echo 'export GST_PLUGIN_PATH=/opt/gst-plugins-rs/lib/aarch64-linux-gnu:$GST_PLUGIN_PATH' >> ~/.bashrc
source ~/.bashrc
```

E agora precisamos garantir que o Daemon sempre funcione mesmo após o reinicio do Raspberry ou quando ocorre isso, para isso:

```bash
# Crie o arquivo de serviço
sudo nano /etc/systemd/system/reachy-mini-daemon.service

# Com o conteúdo abaixo
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

# Reinicia systemctl
sudo systemctl daemon-reload
sudo systemctl enable reachy-mini-daemon
sudo systemctl start reachy-mini-daemon
```