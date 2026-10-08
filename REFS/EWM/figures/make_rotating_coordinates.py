"""Generate the rotating-base-coordinate trajectory figure for the TECS draft.

For every commit t we snapshot the ring basis B(t-1) *before* the new original
enters it, and read the commit's TOP state U(t) -- the union of the frame's
perceptron sketches, which for the single-stream clip is the frame sketch X(t)
-- in those coordinates,

    w_i(t) = |U(t) ∩ B_i(t-1)| / |B_i(t-1)|,

so the heat-map is read with base coordinates on the Y axis and commits on the
X axis: column t is the top state of commit t, expressed in the base
coordinates that were current when it arrived. The lower panel adds the span
dimension and the pre-commit residual |res_{B(t-1)}(U(t))|.

Note (see the paper, Remark on the cumulative union): projecting the
*cumulative* union is uninformative -- every generator is an XOR of originals,
hence a subset of the cumulative union, so every projection is exactly 1. The
informative object is the per-commit top state read in the rotating
coordinates, or its novel part.

Output: REFS/EWM/figures/rotating_coordinates.pdf / .png
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mmh3
import numpy as np

OUT = pathlib.Path(__file__).resolve().parent
P, M, BITS = 10, 1024, 32


def token_position(tok):
    h = mmh3.hash64(tok.encode(), seed=0, signed=False)[0]
    rem = h >> P
    tz = 31 if rem == 0 else min((rem & -rem).bit_length() - 1, 31)
    return (h & (M - 1)) * BITS + tz


def hllset(tokens):
    m = 0
    for t in tokens:
        m |= 1 << token_position(t)
    return m


popcount = int.bit_count


class BoolBasis:
    def __init__(self):
        self.basis, self.pivots, self.last_rotation = [], [], 0

    def dimension(self):
        return len(self.basis)

    def reduce(self, x):
        used = []
        while x:
            p = (x & -x).bit_length() - 1
            if p in self.pivots:
                j = self.pivots.index(p)
                x ^= self.basis[j]
                used.append(j)
            else:
                return x, used
        return 0, used

    def residual(self, x):
        return self.reduce(x)[0]

    def insert(self, x):
        res, _ = self.reduce(x)
        if res == 0:
            self.last_rotation = 0
            return False
        p = (res & -res).bit_length() - 1
        rot = 0
        for i in range(len(self.basis)):
            if (self.basis[i] >> p) & 1:
                self.basis[i] ^= res
                rot += 1
        self.pivots.append(p)
        self.basis.append(res)
        self.last_rotation = rot
        return True


# ── the same 10 x 10 scene clip used in the paper's experiments ─────────────
SCENES, FRAMES, BASE, DRIFT = 10, 10, 200, 56
frames = []
for s in range(SCENES):
    base = [f"s{s}_w{j}" for j in range(BASE)]
    for f in range(FRAMES):
        t = s * FRAMES + f
        frames.append(hllset(base + [f"d{t}_{j}" for j in range(DRIFT)]))
T = len(frames)
CUTS = set(range(0, T, FRAMES)) - {0}

# ── walk the commits: read U(t) in B(t-1), then push ────────────────────────
basis = BoolBasis()
prev_snaps, dim_before, dims, residual, rotations = [], [], [], [], []
for t, x in enumerate(frames):
    prev_snaps.append(list(basis.basis))             # B(t-1)
    dim_before.append(basis.dimension())
    residual.append(popcount(basis.residual(x)))     # res_{B(t-1)}(U(t))
    basis.insert(x)
    dims.append(basis.dimension())
    rotations.append(basis.last_rotation)

K = max(dim_before)
W = np.full((T, K), np.nan)                          # W[t, i] = w_i(t)
for t in range(T):
    for i, b in enumerate(prev_snaps[t]):
        W[t, i] = popcount(frames[t] & b) / popcount(b)

residual = np.array(residual)
cut_mask = np.array([t in CUTS for t in range(T)])
in_span = residual == 0
rot_total = sum(rotations)

fig = plt.figure(figsize=(7.2, 4.9))
gs = fig.add_gridspec(2, 1, height_ratios=[2.7, 1.0], hspace=0.16)
ax0 = fig.add_subplot(gs[0])

cmap = plt.get_cmap("viridis").copy()
cmap.set_bad("white")
im = ax0.imshow(np.ma.masked_invalid(W.T), aspect="auto", origin="lower",
                cmap=cmap, interpolation="nearest",
                extent=(0.5, T + 0.5, 0.5, K + 0.5))
ax0.plot(np.arange(1, T + 1), dim_before, color="white", lw=1.0,
         label="span dimension before the commit (basis frontier)")
for c in CUTS:
    ax0.axvline(c + 0.5, color="w", ls=":", lw=0.7)
ax0.set_ylabel("base coordinate  $i$")
ax0.set_xlim(0.5, T + 0.5)
ax0.set_title("Top state read in rotating base coordinates:  "
              r"$w_i(t)=|U(t)\cap B_i(t{-}1)|\,/\,|B_i(t{-}1)|$", fontsize=8.5)
ax0.legend(fontsize=7, loc="lower right", framealpha=0.85)
cb = fig.colorbar(im, ax=ax0, pad=0.015)
cb.set_label("projection $w_i(t)$", fontsize=8)
cb.ax.tick_params(labelsize=7)

ax1 = fig.add_subplot(gs[1], sharex=ax0)
ax1.plot(np.arange(1, T + 1), residual, color="tab:red", lw=1.0,
         label=r"pre-commit residual $|\mathrm{res}_{B(t-1)}(U(t))|$")
ax1.plot(np.arange(1, T + 1), dims, color="tab:green", lw=1.0,
         label="span dimension after the commit")
for c in CUTS:
    ax1.axvline(c + 0.5, color="gray", ls=":", lw=0.7)
ax1.set_xlabel("commit  $t$")
ax1.set_ylabel("popcount", fontsize=8)
ax1.tick_params(labelsize=7)
ax1.legend(fontsize=7, loc="upper left")

fig.tight_layout()
(out := OUT / "rotating_coordinates.pdf")
fig.savefig(out, bbox_inches="tight")
fig.savefig(OUT / "rotating_coordinates.png", dpi=160, bbox_inches="tight")

print("wrote", out)
print("T=%d  K=%d  final dim=%d" % (T, K, dims[-1]))
print("mean projection over live cells: %.3f  min %.3f  max %.3f"
      % (np.nanmean(W), np.nanmin(W), np.nanmax(W)))
print("top state fully in prior span: %d/%d" % (in_span.sum(), T))
print("pre-commit residual: cut mean %.1f | in-scene mean %.1f | max %d"
      % (residual[cut_mask].mean(), residual[~cut_mask].mean(), residual.max()))
if (~cut_mask).any():
    print("  separation: %.2fx"
          % ((residual[cut_mask].mean()) / (residual[~cut_mask].mean())))
print("basis rotations over run: %d (mean %.2f/commit)" % (rot_total, rot_total / T))
