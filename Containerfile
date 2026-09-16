# syntax=docker/dockerfile:1

# ── builder ──────────────────────────────────────────────────────────────
FROM docker.io/library/rust:1-slim-bookworm AS builder
WORKDIR /src
COPY . .
# The binaries the container ships. ewm-cache is a library, not a bin.
RUN cargo build --release \
    -p ewm-app \
    -p ewm-ops \
    -p ewm-scene \
    -p ewm-sm-explore \
    -p ewm-flux-host \
    -p ewm-git

# ── runtime ──────────────────────────────────────────────────────────────
FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgcc-s1 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m ewm \
    && mkdir -p /var/lib/ewm \
    && chown ewm:ewm /var/lib/ewm
WORKDIR /home/ewm

COPY --from=builder \
    /src/target/release/ewm-app \
    /src/target/release/ewm-ops \
    /src/target/release/ewm-scene \
    /src/target/release/ewm-sm-explore \
    /src/target/release/ewm-flux-host \
    /src/target/release/ewm-git \
    /usr/local/bin/

USER ewm
VOLUME /var/lib/ewm

# Bare `podman run` boots the operational graph CLI; override with any other
# binary: `podman run --rm ewm-state-machine ewm-app --help`
CMD ["ewm-ops", "--help"]
