#!/usr/bin/env bash
# Build the train_detect .eap with the official ACAP Native SDK image.
# Usage (from WSL or Linux): tools/build.sh [aarch64|armv7hf]
set -euo pipefail
ARCH="${1:-aarch64}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="$ROOT/model/out/traindet.tflite"
mkdir -p "$ROOT/acap/app/model" "$ROOT/build"
if [ ! -f "$MODEL" ]; then
    echo "missing $MODEL - run the model pipeline first (model/README.md)" >&2
    exit 1
fi
cp "$MODEL" "$ROOT/acap/app/model/traindet.tflite"
cp "$ROOT/model/yamnet/yamnet_class_map.csv" "$ROOT/acap/app/model/yamnet_class_map.csv"
# Packaged license = MIT + third-party notices + Apache-2.0 text (required for YAMNet)
{ cat "$ROOT/LICENSE"; printf '\n\n'; cat "$ROOT/THIRD-PARTY-NOTICES.md"; printf '\n\n'; cat "$ROOT/LICENSE-APACHE-2.0"; } > "$ROOT/acap/app/LICENSE"

# --no-cache: BuildKit on WSL can reuse stale COPY layers from /mnt/c (see learnings)
docker build --no-cache --build-arg ARCH="$ARCH" -t "train_detect:$ARCH" "$ROOT/acap"
cid=$(docker create "train_detect:$ARCH")
trap 'docker rm -f "$cid" >/dev/null' EXIT
rm -f "$ROOT"/build/*_"$ARCH".eap
docker cp "$cid:/opt/app/." "$ROOT/build/tmp_$ARCH"
mv "$ROOT"/build/tmp_"$ARCH"/*.eap "$ROOT/build/"
rm -rf "$ROOT/build/tmp_$ARCH"
ls -la "$ROOT"/build/*.eap
