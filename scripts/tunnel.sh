#!/usr/bin/env bash
# tunnel.sh — Opens SSH tunnels to the Raspberry Pi for local development.
#
# Services forwarded:
#   localhost:7447  -> Pi:7447  (Zenoh router — Reachy SDK control)
#   localhost:8000  -> Pi:8000  (Reachy HTTP API)
#   localhost:8443  -> Pi:8443  (Reachy WebRTC signaling)
#   localhost:18789 -> Pi:18789 (OpenClaw UI)
#   localhost:8123  -> Pi:8123  (Home Assistant UI)
#
# Usage:
#   ./scripts/tunnel.sh          # start tunnels
#   ./scripts/tunnel.sh stop     # kill all tunnels

PI="dobby@raspberrypi.local"
CONTROL_SOCKET="/tmp/dobby-tunnel.sock"

start() {
    if ssh -O check -S "$CONTROL_SOCKET" "$PI" &>/dev/null; then
        echo "Tunnels already running."
        exit 0
    fi

    echo "Opening tunnels to $PI ..."
    ssh -fN -M -S "$CONTROL_SOCKET" \
        -L 7447:127.0.0.1:7447 \
        -L 8000:127.0.0.1:8000 \
        -L 8443:127.0.0.1:8443 \
        -L 18789:127.0.0.1:18789 \
        -L 8123:127.0.0.1:8123 \
        "$PI"

    echo ""
    echo "Tunnels open:"
    echo "  Reachy Zenoh     -> tcp://localhost:7447"
    echo "  Reachy HTTP API  -> http://localhost:8000"
    echo "  Reachy WebRTC    -> wss://localhost:8443"
    echo "  OpenClaw         -> http://localhost:18789"
    echo "  Home Assistant   -> http://localhost:8123"
    echo ""
    echo "Stop with: ./scripts/tunnel.sh stop"
}

stop() {
    if ssh -O check -S "$CONTROL_SOCKET" "$PI" &>/dev/null; then
        ssh -O exit -S "$CONTROL_SOCKET" "$PI"
        echo "Tunnels closed."
    else
        echo "No active tunnels found."
    fi
}

case "${1:-start}" in
    start) start ;;
    stop)  stop  ;;
    *)     echo "Usage: $0 [start|stop]"; exit 1 ;;
esac
