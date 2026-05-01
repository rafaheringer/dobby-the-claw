#!/usr/bin/env bash
# Setup script for OpenClaw on a fresh Raspberry Pi.
#
# Run from the dobby-the-claw project root after filling in .env:
#   ./scripts/setup-openclaw.sh
#
# What it does:
#   1. Rebuilds the OpenClaw image with Docker CLI (required for sandbox exec)
#   2. Installs the home-assistant skill
#   3. Downloads jq into the persistent workspace
#   4. Creates the HA config file in the workspace
#   5. Applies exec and approval settings in openclaw.json / exec-approvals.json
#   6. Copies docker-compose.override.yml and restarts the gateway

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOBBY_ENV="$PROJECT_ROOT/.env"
OPENCLAW_DIR="$HOME/openclaw"
OPENCLAW_CONFIG_DIR="${OPENCLAW_CONFIG_DIR:-$HOME/.openclaw}"
OPENCLAW_WORKSPACE_DIR="${OPENCLAW_WORKSPACE_DIR:-$HOME/.openclaw/workspace}"

# ── helpers ──────────────────────────────────────────────────────────────────

fail() { echo "ERROR: $*" >&2; exit 1; }
info() { echo "▸ $*"; }

read_env() {
  local key="$1"
  grep -E "^${key}=" "$DOBBY_ENV" 2>/dev/null | head -1 | cut -d= -f2- | tr -d '"'
}

# ── pre-checks ───────────────────────────────────────────────────────────────

[[ -f "$DOBBY_ENV" ]] || fail ".env not found at $DOBBY_ENV — copy .env.example and fill it in."
[[ -d "$OPENCLAW_DIR" ]] || fail "OpenClaw not found at $OPENCLAW_DIR — clone it first."
command -v docker >/dev/null 2>&1 || fail "docker not found."
command -v python3 >/dev/null 2>&1 || fail "python3 not found."
command -v curl >/dev/null 2>&1 || fail "curl not found."

HOME_ASSISTANT_URL="$(read_env HOME_ASSISTANT_URL)"
HOME_ASSISTANT_TOKEN="$(read_env HOME_ASSISTANT_TOKEN)"

[[ -n "$HOME_ASSISTANT_URL" ]] || fail "HOME_ASSISTANT_URL is empty in .env"
[[ -n "$HOME_ASSISTANT_TOKEN" ]] || fail "HOME_ASSISTANT_TOKEN is empty in .env"

# ── 1. rebuild image with Docker CLI ─────────────────────────────────────────

info "Rebuilding OpenClaw image with Docker CLI support (this takes a few minutes)..."
DOCKER_GID="$(stat -c '%g' /var/run/docker.sock 2>/dev/null || echo 999)"
DOCKER_BUILDKIT=1 docker build \
  --build-arg OPENCLAW_INSTALL_DOCKER_CLI=1 \
  -t openclaw:local \
  "$OPENCLAW_DIR"

# ── 2. install home-assistant skill ──────────────────────────────────────────

SKILL_DIR="$OPENCLAW_WORKSPACE_DIR/skills/home-assistant"
if [[ -d "$SKILL_DIR" ]]; then
  info "home-assistant skill already installed, skipping."
else
  info "Installing home-assistant skill from clawhub..."
  TMP_ZIP="$(mktemp /tmp/ha-skill-XXXX.zip)"
  curl -sL "https://wry-manatee-359.convex.site/api/v1/download?slug=home-assistant" -o "$TMP_ZIP"
  mkdir -p "$SKILL_DIR"
  unzip -o "$TMP_ZIP" -d "$SKILL_DIR"
  chmod +x "$SKILL_DIR/scripts/ha.sh"
  rm "$TMP_ZIP"
  info "home-assistant skill installed."
fi

# ── 3. download jq to persistent workspace ───────────────────────────────────

JQ_BIN="$OPENCLAW_WORKSPACE_DIR/.bin/jq"
if [[ -x "$JQ_BIN" ]]; then
  info "jq already present, skipping."
else
  info "Downloading jq for $(uname -m)..."
  mkdir -p "$(dirname "$JQ_BIN")"
  ARCH="$(uname -m)"
  case "$ARCH" in
    aarch64) JQ_URL="https://github.com/jqlang/jq/releases/download/jq-1.7.1/jq-linux-arm64" ;;
    x86_64)  JQ_URL="https://github.com/jqlang/jq/releases/download/jq-1.7.1/jq-linux-amd64" ;;
    *) fail "Unsupported architecture: $ARCH" ;;
  esac
  curl -sL "$JQ_URL" -o "$JQ_BIN"
  chmod +x "$JQ_BIN"
  info "jq installed at $JQ_BIN"
fi

# ── 4. HA config file in workspace ───────────────────────────────────────────

HA_CONFIG_DIR="$OPENCLAW_WORKSPACE_DIR/.config/home-assistant"
HA_CONFIG_FILE="$HA_CONFIG_DIR/config.json"
info "Writing HA config to workspace..."
mkdir -p "$HA_CONFIG_DIR"
python3 -c "
import json
cfg = {'url': '$HOME_ASSISTANT_URL', 'token': '$HOME_ASSISTANT_TOKEN'}
with open('$HA_CONFIG_FILE', 'w') as f:
    json.dump(cfg, f, indent=2)
"
info "HA config written: $HA_CONFIG_FILE"

# ── 5. openclaw.json — exec settings ─────────────────────────────────────────

OPENCLAW_JSON="$OPENCLAW_CONFIG_DIR/openclaw.json"
if [[ -f "$OPENCLAW_JSON" ]]; then
  info "Applying exec settings to openclaw.json..."
  python3 - "$OPENCLAW_JSON" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    cfg = json.load(f)
exec_cfg = cfg.setdefault("tools", {}).setdefault("exec", {})
exec_cfg["host"] = "gateway"
exec_cfg["security"] = "full"
exec_cfg["ask"] = "on-miss"
cfg["agents"]["defaults"]["sandbox"]["mode"] = "off"
with open(sys.argv[1], "w") as f:
    json.dump(cfg, f, indent=4)
print("  tools.exec:", json.dumps(exec_cfg))
PY
else
  info "openclaw.json not found yet — exec settings will be applied on first run."
fi

# ── 6. exec-approvals.json — ask fallback ────────────────────────────────────

APPROVALS_JSON="$OPENCLAW_CONFIG_DIR/exec-approvals.json"
if [[ -f "$APPROVALS_JSON" ]]; then
  info "Applying askFallback=full to exec-approvals.json..."
  python3 - "$APPROVALS_JSON" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    cfg = json.load(f)
cfg.setdefault("defaults", {})["askFallback"] = "full"
cfg["defaults"]["autoAllowSkills"] = True
with open(sys.argv[1], "w") as f:
    json.dump(cfg, f, indent=4)
print("  askFallback:", cfg["defaults"]["askFallback"])
PY
else
  info "exec-approvals.json not found yet — will be created on first run."
fi

# ── 7. openclaw .env — sandbox + HA vars ─────────────────────────────────────

OPENCLAW_ENV="$OPENCLAW_DIR/.env"
if [[ -f "$OPENCLAW_ENV" ]]; then
  info "Updating ~/openclaw/.env with HA and sandbox vars..."

  set_env_var() {
    local key="$1" val="$2"
    if grep -qE "^${key}=" "$OPENCLAW_ENV"; then
      sed -i "s|^${key}=.*|${key}=${val}|" "$OPENCLAW_ENV"
    else
      echo "${key}=${val}" >> "$OPENCLAW_ENV"
    fi
  }

  set_env_var "HOME_ASSISTANT_URL" "$HOME_ASSISTANT_URL"
  set_env_var "HOME_ASSISTANT_TOKEN" "$HOME_ASSISTANT_TOKEN"
  set_env_var "OPENCLAW_SANDBOX" "1"
  set_env_var "DOCKER_GID" "$DOCKER_GID"
  set_env_var "OPENCLAW_INSTALL_DOCKER_CLI" "1"
fi

# ── 8. docker-compose override ───────────────────────────────────────────────

OVERRIDE_SRC="$PROJECT_ROOT/docker-compose.openclaw.yml"
OVERRIDE_DST="$OPENCLAW_DIR/docker-compose.override.yml"
info "Copying docker-compose.override.yml to $OPENCLAW_DIR..."
cp "$OVERRIDE_SRC" "$OVERRIDE_DST"

# ── 9. restart gateway ───────────────────────────────────────────────────────

info "Restarting OpenClaw gateway..."
cd "$OPENCLAW_DIR"
docker compose up -d --force-recreate openclaw-gateway

info ""
info "Done! OpenClaw gateway restarting. It takes ~2 minutes to become healthy."
info "Check with: curl http://localhost:18789/healthz"
