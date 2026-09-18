#!/bin/zsh
# Resize a 1024 px icon into App/Assets.xcassets/AppIcon.appiconset at every macOS size.
set -euo pipefail
src=${1:?usage: make-icon-set.sh icon-1024.png}
set_dir="$(dirname $0)/../App/Assets.xcassets/AppIcon.appiconset"
mkdir -p "$set_dir"
for size in 16 32 128 256 512; do
  sips -z $size $size "$src" --out "$set_dir/icon_${size}x${size}.png" >/dev/null
  sips -z $((size * 2)) $((size * 2)) "$src" --out "$set_dir/icon_${size}x${size}@2x.png" >/dev/null
done
