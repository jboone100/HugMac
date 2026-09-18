#!/bin/zsh
# Render LocalLab's window to a PNG, for checking layouts without a human at the keyboard.
#   scripts/snapshot.sh out.png [file-to-open] [WxH]
# Needs Screen Recording permission for the terminal running it.
set -e
out=${1:?usage: snapshot.sh out.png [file] [WxH]}
file=${2:-}
size=${3:-1180x1560}
app="$(cd "$(dirname "$0")/.." && pwd)/.build/xcode/Build/Products/Debug/LocalLab.app"
rm -f "$out" "$out.windowid"
args=(--env LOCALLAB_SNAPSHOT="$out" --env LOCALLAB_SNAPSHOT_SIZE="$size"
      --env LOCALLAB_EXPAND_REASONS="${LOCALLAB_EXPAND_REASONS:-0}" --env LOCALLAB_SNAPSHOT_DELAY=5)
[[ -n "$file" ]] && args+=(--env LOCALLAB_OPEN="$file")
open -n "${args[@]}" "$app"
for i in {1..60}; do [[ -f "$out.windowid" ]] && break; sleep 0.25; done
sleep 0.5
screencapture -x -o -l "$(cat "$out.windowid")" "$out"
rm -f "$out.windowid"
echo "$out"
