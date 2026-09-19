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

## Notebook 08 with the container

`notebooks/08_multi_llm_sidecar.ipynb` runs on the **host** Python/Jupyter by
default, using the host `target/debug/ewm-scene` binary. The container image
is binaries-only (no Python), so it is not the notebook runtime; it either
supplies the `ewm-scene` binary to the host notebook, or a Python-capable
image is derived from it for a fully containerized run.

### Option A — host notebook, containerized ewm-scene (recommended)

Keep the notebook and its data on the host. Replace the notebook's
`ewm_scene()` helper with a podman wrapper that bind-mounts the WORK
directory into the image and calls the release `ewm-scene` binary:

```python
def ewm_scene(*args):
    # All notebook JSON/JSONL files live in one WORK dir, so mapping each
    # path to /work/<basename> is enough.
    mapped = [("/work/" + os.path.basename(a)) if a.endswith((".jsonl", ".json"))
              else a for a in args]
    r = subprocess.run(
        ["podman", "run", "--rm", "--userns=keep-id",
         "-v", f"{WORK}:/work:Z", "ewm-state-machine", "ewm-scene", *mapped],
        capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"ewm-scene failed: {r.stderr[-800:]}")
    return json.loads(r.stdout.strip())
```

Notes:

- `--userns=keep-id` keeps host-file ownership correct under rootless
  Podman; drop the `:Z` on non-SELinux hosts.
- This replaces the ~25 direct `ewm-scene` calls with ~25 `podman run`
  spawns — slower than the host binary, but fine for a notebook.

### Option B — fully containerized notebook run

Derive a Python-capable image from the binaries image, then run the notebook
headless with `jupyter nbconvert`. The canonical notebook stays on the host
(repo); the container only executes a bind-mounted copy.

```dockerfile
# Containerfile.notebook
FROM localhost/ewm-state-machine
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-numpy python3-matplotlib jupyter-nbconvert \
    && rm -rf /var/lib/apt/lists/*
USER ewm
```

```bash
podman build -t ewm-state-machine-notebook -f Containerfile.notebook .
podman run --rm --userns=keep-id \
  -v "$PWD/notebooks":/home/ewm/notebooks:Z \
  -v /home/alexmy/.cache/ewm-multi-llm:/home/ewm/.cache/ewm-multi-llm:Z \
  ewm-state-machine-notebook \
  jupyter nbconvert --execute --to notebook \
  /home/ewm/notebooks/08_multi_llm_sidecar.ipynb
```

Inside the container the notebook runs from `/home/ewm/notebooks`; the two
paths at the top of the notebook need to point at the container layout:

```python
EWM_SCENE_BIN = "ewm-scene"                 # already on PATH in the image
WORK = "/home/ewm/.cache/ewm-multi-llm"
```

(or make both constants environment-driven so the same notebook works in
both modes).

## Notes

- The container is **rootless-friendly**: it runs as UID 1000 (`ewm`) and
  only writes inside `/var/lib/ewm`.
- `ewm-app` persists its commit DAG wherever `--repo` points; put it on the
  same volume for durable commits.
- The build context ignores `target/`, `.git/`, notebook assets, and PDFs
  (`.containerignore`).
- The image is derived-data-only: any volume content can be rebuilt from
  the commit log, so containers are disposable.
