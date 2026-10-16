#!/usr/bin/env bash
# Compile and run the host unit tests (detector + WAV parser) with ASan/UBSan.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if ! docker image inspect traindet-hosttest >/dev/null 2>&1; then
    printf 'FROM debian:bookworm-slim\nRUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*\n' |
        docker build -q -t traindet-hosttest -
fi
docker run --rm --security-opt seccomp=unconfined -v "$ROOT/acap:/src" -w /src traindet-hosttest bash -c \
  "gcc -std=gnu11 -O1 -g -Wall -Wextra -Werror -fsanitize=address,undefined -fno-omit-frame-pointer \
     tests/test_host.c app/detector.c app/fusion.c app/wav.c -lm -o /tmp/test_host && setarch $(uname -m) -R /tmp/test_host"
