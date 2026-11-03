# lanternfish: multi-stage image
#   1. llama.cpp shared libraries (CPU), pinned to a commit
#   2. Svelte UI
#   3. Rust binary (links stage 1, embeds stage 2)
#   4. slim runtime
#
# Build:  docker build -t lanternfish .
#         docker build --build-arg LLAMA_CPP_REF=<sha> -t lanternfish .
# Run:    docker run -p 8791:8791 -v /path/to/models:/models:ro -v lanternfish-data:/data lanternfish

ARG LLAMA_CPP_REPO=https://github.com/ggml-org/llama.cpp
# Upstream commit with qwen4exp support (llama.cpp build 11429, 2026-10-05).
ARG LLAMA_CPP_REF=d81235049384534c167caea52b85a694f6103d14

FROM debian:trixie-slim AS llama
ARG LLAMA_CPP_REPO
ARG LLAMA_CPP_REF
# GGML_NATIVE=OFF + CPU_ALL_VARIANTS: one image that runs on any x86-64 CPU and
# picks AVX2 / AVX-512 kernels at run time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN git init -q llama.cpp && cd llama.cpp \
    && git remote add origin "$LLAMA_CPP_REPO" \
    && git fetch -q --depth 1 origin "$LLAMA_CPP_REF" && git checkout -q FETCH_HEAD
RUN cd llama.cpp && cmake -B build \
        -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
        -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF \
        -DLLAMA_BUILD_COMMON=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_CURL=OFF \
    && cmake --build build -j"$(nproc)" \
    && cmake --install build --prefix /opt/llama \
    && cp -n build/bin/libggml*.so /opt/llama/lib/ 2>/dev/null; ls /opt/llama/lib

FROM node:22-bookworm-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

FROM rust:1-trixie AS rust
RUN apt-get update && apt-get install -y --no-install-recommends libclang-dev clang \
    && rm -rf /var/lib/apt/lists/*
COPY --from=llama /opt/llama /opt/llama
ENV LLAMA_INCLUDE_DIR=/opt/llama/include LLAMA_LIB_DIR=/opt/llama/lib
WORKDIR /app
COPY Cargo.toml Cargo.lock build.rs ./
COPY src ./src
COPY --from=web /web/dist ./web/dist
RUN cargo build --release --locked && cp target/release/lanternfish /usr/local/bin/

FROM debian:trixie-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --home /data lanternfish \
    && mkdir -p /data && chown lanternfish:lanternfish /data
COPY --from=llama /opt/llama/lib /opt/llama/lib
COPY --from=rust /usr/local/bin/lanternfish /usr/local/bin/lanternfish
ENV LD_LIBRARY_PATH=/opt/llama/lib \
    LANTERNFISH_BACKEND_DIR=/opt/llama/lib \
    LANTERNFISH_HOME=/data \
    LANTERNFISH_MODELS=/models
VOLUME ["/data"]
EXPOSE 8791
USER lanternfish
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD curl -fsS http://127.0.0.1:8791/api/health || exit 1
ENTRYPOINT ["lanternfish"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8791"]
