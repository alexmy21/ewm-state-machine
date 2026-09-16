# Podman packaging — ewm-state-machine

The repository ships a multi-stage `Containerfile` that builds all six
binaries (`ewm-app`, `ewm-ops`, `ewm-scene`, `ewm-sm-explore`,
`ewm-flux-host`, `ewm-git`) in a Rust builder image and copies them into a
small `debian:bookworm-slim` runtime running as a non-root `ewm` user.

## Build

```bash
podman build -t ewm-state-machine .
```

## Run

Bare `podman run` boots the operational graph CLI (`ewm-ops`):

```bash
podman run --rm ewm-state-machine                 # ewm-ops --help
podman run --rm ewm-state-machine ewm-app --help  # any other binary
```

## Boot with a persisted state volume

The image declares `/var/lib/ewm` as its data volume (boot store, boot log,
ewm-git repo, Arrow cache). Two mount styles work under rootless Podman:

**Named volume (simplest — Podman initializes it with the image's `ewm`
ownership):**

```bash
podman volume create ewm-data

cat > /tmp/boot.ops <<'EOF'
value a apple
value b banana cherry
def union2 ( 2 -- 1 ) union
link value:@a -> in:union2.0
link value:@b -> in:union2.1
stack @a @b
EOF

# install the boot file into the volume once (`-i` keeps stdin open)
podman run -i --rm -v ewm-data:/var/lib/ewm ewm-state-machine \
    sh -c 'cat > /var/lib/ewm/boot.ops' < /tmp/boot.ops

# first boot: install the boot file, run, persist state
podman run --rm -v ewm-data:/var/lib/ewm ewm-state-machine \
    ewm-ops --store /var/lib/ewm --boot /var/lib/ewm/boot.ops

# second boot: picks up the state from the top of the stack (no --boot)
podman run --rm -v ewm-data:/var/lib/ewm ewm-state-machine \
    ewm-ops --store /var/lib/ewm

# boot log, list, and rollback
podman run --rm -v ewm-data:/var/lib/ewm ewm-state-machine \
    ewm-ops log --store /var/lib/ewm
podman run --rm -v ewm-data:/var/lib/ewm ewm-state-machine \
    ewm-ops list --store /var/lib/ewm
podman run --rm -v ewm-data:/var/lib/ewm ewm-state-machine \
    ewm-ops prev --store /var/lib/ewm
```

**Host directory** — add `--userns=keep-id` so the container's `ewm` uid maps
to the host user (and `:Z` on SELinux hosts):

```bash
mkdir -p /tmp/ewm-podman
cp /tmp/boot.ops /tmp/ewm-podman/
podman run --rm --userns=keep-id -v /tmp/ewm-podman:/var/lib/ewm:Z ewm-state-machine \
    ewm-ops --store /var/lib/ewm --boot /var/lib/ewm/boot.ops
podman run --rm --userns=keep-id -v /tmp/ewm-podman:/var/lib/ewm:Z ewm-state-machine \
    ewm-ops --store /var/lib/ewm
```

## Notes

- The container is **rootless-friendly**: it runs as UID 1000 (`ewm`) and
  only writes inside `/var/lib/ewm`.
- `ewm-app` persists its commit DAG wherever `--repo` points; put it on the
  same volume for durable commits.
- The build context ignores `target/`, `.git/`, notebook assets, and PDFs
  (`.containerignore`).
- The image is derived-data-only: any volume content can be rebuilt from
  the commit log, so containers are disposable.
