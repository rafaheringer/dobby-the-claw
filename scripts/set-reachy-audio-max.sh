#!/usr/bin/env bash
set -euo pipefail

target_volume="${1:-100%}"

find_reachy_card() {
    if amixer -c Audio scontrols >/dev/null 2>&1; then
        printf 'Audio\n'
        return 0
    fi

    local match
    match="$(aplay -l 2>/dev/null | grep -i 'Reachy Mini Audio' | head -n 1 || true)"
    if [[ -z "$match" ]]; then
        return 1
    fi

    if [[ "$match" =~ card[[:space:]]+([0-9]+): ]]; then
        printf '%s\n' "${BASH_REMATCH[1]}"
        return 0
    fi

    return 1
}

for _ in $(seq 1 20); do
    if card="$(find_reachy_card)"; then
        amixer -c "$card" sset PCM "$target_volume" >/dev/null
        amixer -c "$card" sset 'PCM,1' "$target_volume" >/dev/null
        exit 0
    fi
    sleep 1
done

echo 'Reachy Mini audio card not found' >&2
exit 1