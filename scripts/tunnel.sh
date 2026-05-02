#!/usr/bin/env bash
# tunnel.sh — Opens SSH tunnels to the Raspberry Pi for local development.
#
# Services forwarded:
#   localhost:7447  -> Pi:7447  (Zenoh router — Reachy SDK control)
#   localhost:8000  -> Pi:8000  (Reachy HTTP API)
#   localhost:8443  -> Pi:8443  (Reachy WebRTC signaling)
#   localhost:18789 -> Pi:18789 (OpenClaw UI)
#   localhost:8123  -> Pi:8123  (Home Assistant UI)
#   localhost:61208 -> Pi:61208 (Glances system monitor)
#   Pi:18800        <- localhost:18800  (Dobby notification webhook — reverse tunnel)
#
# Usage:
#   ./scripts/tunnel.sh          # start tunnels
#   ./scripts/tunnel.sh stop     # kill all tunnels

PI="dobby@raspberrypi.local"
PID_FILE="/tmp/dobby-tunnel.pid"

start() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "Tunnels already running (PID $(cat "$PID_FILE"))."
        exit 0
    fi

    echo "Opening tunnels to $PI ..."
    nohup ssh -N \
        -L 7447:127.0.0.1:7447 \
        -L 8000:127.0.0.1:8000 \
        -L 8443:127.0.0.1:8443 \
        -L 18789:127.0.0.1:18789 \
        -L 8123:127.0.0.1:8123 \
        -L 61208:127.0.0.1:61208 \
        -R 18800:127.0.0.1:18800 \
        "$PI" > /dev/null 2>&1 &
    SSH_PID=$!
    echo "$SSH_PID" > "$PID_FILE"

    echo ""
    echo "Tunnels open (PID $SSH_PID):"
    echo "  Reachy Zenoh          -> tcp://localhost:7447"
    echo "  Reachy HTTP API       -> http://localhost:8000"
    echo "  Reachy WebRTC         -> wss://localhost:8443"
    echo "  OpenClaw              -> http://localhost:18789"
    echo "  Home Assistant        -> http://localhost:8123"
    echo "  Glances               -> http://localhost:61208"
    echo "  Notification webhook  <- http://127.0.0.1:18800 (reverse tunnel)"
    echo ""
    echo "Stop with: ./scripts/tunnel.sh stop"
}

stop() {
    if [ ! -f "$PID_FILE" ]; then
        echo "No active tunnels found."
        exit 0
    fi
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        kill "$PID"
        rm -f "$PID_FILE"
        echo "Tunnels closed (PID $PID)."
    else
        rm -f "$PID_FILE"
        echo "No active tunnels found (stale PID file removed)."
    fi
}

case "${1:-start}" in
    start) start ;;
    stop)  stop  ;;
    *)     echo "Usage: $0 [start|stop]"; exit 1 ;;
esac
