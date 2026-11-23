#!/bin/sh
set -eu

base=184f6536024668e8908ab8c86033d56708e62b40
if [ "$#" -ne 1 ]; then
    echo "Usage: sh patches/apply-llama.sh <llama.cpp checkout>" >&2
    exit 2
fi
patches=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
checkout=$(git -C "$1" rev-parse --show-toplevel)
if [ "$(git -C "$checkout" rev-parse HEAD)" != "$base" ]; then
    echo "Check out llama.cpp $base before applying the patches." >&2
    exit 1
fi
if [ -n "$(git -C "$checkout" status --porcelain --untracked-files=no)" ]; then
    echo "Apply the patches to a clean llama.cpp checkout." >&2
    exit 1
fi
git -C "$checkout" apply --check "$patches"/llama.cpp/*.patch
git -C "$checkout" apply "$patches"/llama.cpp/*.patch
