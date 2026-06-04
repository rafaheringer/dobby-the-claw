#!/usr/bin/env sh
set -eu

find_reachy_card_name() {
    for id_file in /sys/class/sound/card*/id; do
        [ -f "$id_file" ] || continue
        card_name=$(cat "$id_file" 2>/dev/null || true)
        case "$card_name" in
            Audio|audio|Array|array)
                printf '%s\n' "$card_name"
                return 0
                ;;
        esac
    done
    return 1
}

if card_name="$(find_reachy_card_name)"; then
    cat > /root/.asoundrc <<EOF
pcm.!default {
    type hw
    card $card_name
}

ctl.!default {
    type hw
    card $card_name
}
EOF
    echo "Configured ALSA default card for bridge container: $card_name"
fi

exec "$@"