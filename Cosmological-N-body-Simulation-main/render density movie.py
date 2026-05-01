#!/usr/bin/env python3
"""
render_3d_movie.py
-------------------
Reads frame_XXXX.npz snapshots produced by nbody_movie_sim.py
and renders a 3D particle + halo movie.

Output:  movies/nbody_3d_movie.mp4

Requirements:
    pip install matplotlib numpy ffmpeg-python
    System: ffmpeg must be installed (conda install ffmpeg  or  apt install ffmpeg)

Features:
  - 3D scatter of all particles (subsampled for speed)
  - Halo positions overlaid as large coloured spheres (reads halo_catalogue_<z>
    if available, otherwise skips halos for that frame)
  - Rotating camera — box slowly rotates 360° over the full movie
  - Redshift / scale factor label on every frame
  - Colour of particles encodes local density (CIC)

  Supports 32³ and 128³ (and other grid sizes) automatically — grid size
  is inferred from the first snapshot rather than being hard-coded.
"""

import os
import glob
import struct
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless rendering
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FFMpegWriter
from scipy.ndimage import gaussian_filter

# =====================================================================
# CONFIG — adjust to taste
# =====================================================================
SNAP_DIR        = "snapshots"
HALOES_DIR      = "halos"
OUTPUT_DIR      = "movies"
OUTPUT_FILE     = os.path.join(OUTPUT_DIR, "nbody_3d_movie.mp4")

# Grid size and box scale — set to None to auto-detect from snapshots.
# Override manually if your snapshots don't store these values:
#   GRID_N  = 128   (number of cells per side)
#   LL_BOX  = 0.14  (Mpc per grid cell)
GRID_N          = None   # auto-detect
LL_BOX          = 0.14   # Mpc per grid cell (assumed constant)

# Rendering
FPS             = 30               # frames per second in output video
DPI             = 120              # increase to 180 for HD
PARTICLE_ALPHA  = 0.25
HALO_SIZE_SCALE = 60               # halo marker size = HALO_SIZE_SCALE * log10(M)
CAMERA_ELEV     = 25               # fixed elevation angle
CAMERA_AZIM_START = 30             # starting azimuth
CAMERA_AZIM_FULL_ROTATIONS = 1    # full 360° rotations over entire movie

# Colour map — particles coloured by log density
CMAP_PARTICLES  = "plasma"
CMAP_HALOS      = "cool"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================================
# HELPERS
# =====================================================================
def load_snapshot(path):
    d = np.load(path)
    rra = d["rra"]
    aa  = float(d["aa"])
    z   = float(d["z"])
    # Optionally stored grid size
    grid_n = int(d["grid_n"]) if "grid_n" in d else None
    return rra, aa, z, grid_n


def infer_grid_n(snap_files):
    """
    Infer grid size N from particle positions in the first snapshot.
    Particles live in [0, N), so N ≈ ceil(max position) rounded to a
    power-of-two or common value.  Falls back to 32 if detection fails.
    """
    rra, _, _, stored_n = load_snapshot(snap_files[0])
    if stored_n is not None:
        return stored_n
    max_pos = rra.max()
    # Round up to nearest power of two ≥ max_pos
    n = 1
    while n < max_pos:
        n *= 2
    print(f"  Auto-detected grid size N={n} (max particle coord={max_pos:.1f})")
    return n


def particle_render_params(grid_n):
    """
    Return (subsample_factor, dot_size) scaled so rendering is fast and
    visually consistent across different grid sizes.

    For 32³  (~32k particles):   subsample=1,  size=0.8
    For 64³  (~262k particles):  subsample=2,  size=0.5
    For 128³ (~2M particles):    subsample=8,  size=0.2
    For 256³ (~16M particles):   subsample=32, size=0.1
    """
    n_particles = grid_n ** 3
    target_displayed = 65_000   # aim to plot this many points per frame
    subsample = max(1, int(np.ceil(n_particles / target_displayed)))
    # Dot size: smaller grids benefit from larger dots
    dot_size = max(0.1, 1.2 / (subsample ** 0.4))
    return subsample, dot_size


def set_camera_distance(ax, grid_n):
    """
    Push the camera further out for larger grids so the whole box is visible.

    Matplotlib's default camera distance (10 in axes units) is calibrated
    for a ~1-unit cube.  When axes limits span [0, N], we need to scale the
    distance proportionally.  The magic number 4.5 gives a comfortable framing
    for N=32; we scale linearly.
    """
    scale = grid_n / 32.0
    # ax.dist controls the zoom-out level (higher = more zoomed out).
    # Default is ~10; we set it relative to grid scale.
    try:
        ax.dist = 4.5 * scale          # mpl ≥ 3.3
    except AttributeError:
        pass
    # Also set the box aspect so X/Y/Z axes stay equal
    try:
        ax.set_box_aspect([1, 1, 1])   # mpl ≥ 3.3
    except AttributeError:
        pass


def compute_density_colours(rra, Nx, Ny, Nz, subsample, global_lo, global_hi, cmap):
    """CIC density on coarse grid → colour each (subsampled) particle."""
    pos = rra[::subsample]

    ix = np.clip(pos[:, 0].astype(int), 0, Nx - 1)
    iy = np.clip(pos[:, 1].astype(int), 0, Ny - 1)
    iz = np.clip(pos[:, 2].astype(int), 0, Nz - 1)

    density = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    np.add.at(density, (ix, iy, iz), 1.0)
    density = gaussian_filter(density, sigma=1.0)

    rho_log  = np.log10(density[ix, iy, iz] + 1e-3)
    c_norm   = np.clip((rho_log - global_lo) / (global_hi - global_lo + 1e-10), 0, 1)
    return pos, cmap(c_norm)


def find_nearest_halo_catalogue(z_target, haloes_dir, tol=0.3):
    """Return path to halo catalogue closest in redshift to z_target."""
    pattern = os.path.join(haloes_dir, "halo_catalogue_*")
    files   = glob.glob(pattern)
    if not files:
        return None
    best, best_dz = None, 1e9
    for f in files:
        try:
            z_cat = float(os.path.basename(f).replace("halo_catalogue_", ""))
            dz    = abs(z_cat - z_target)
            if dz < best_dz:
                best_dz, best = dz, f
        except ValueError:
            pass
    return best if best_dz < tol else None


def read_halo_catalogue(fname, ll_box=0.14):
    """Returns mass (10^10 Msun/h) and positions (grid units)."""
    try:
        with open(fname, "rb") as f:
            f.read(4); f.read(256); f.read(4)
            f.read(4)
            Nc = struct.unpack("<i", f.read(4))[0]
            f.read(4)
            if Nc == 0:
                return np.array([]), np.zeros((0, 3))
            f.read(4)
            raw = f.read(Nc * 7 * 4)
            f.read(4)
        data = np.frombuffer(raw, dtype=np.float32).reshape(Nc, 7)
        mass = data[:, 0]
        pos  = data[:, 1:4]   # already in grid units (output_flag=1)
        return mass, pos
    except Exception as e:
        print(f"  [halo reader] {e}")
        return np.array([]), np.zeros((0, 3))


# =====================================================================
# DISCOVER SNAPSHOTS
# =====================================================================
snap_files = sorted(glob.glob(os.path.join(SNAP_DIR, "frame_*.npz")))
if not snap_files:
    raise FileNotFoundError(f"No snapshots found in '{SNAP_DIR}/'.\n"
                            "Run nbody_movie_sim.py first.")

N_FRAMES = len(snap_files)
print(f"Found {N_FRAMES} snapshots → rendering at {FPS} fps")
print(f"Estimated movie duration: {N_FRAMES / FPS:.1f} s")

# =====================================================================
# AUTO-DETECT GRID SIZE
# =====================================================================
if GRID_N is None:
    GRID_N = infer_grid_n(snap_files)

Nx = Ny = Nz = GRID_N
PARTICLE_SUBSAMPLE, PARTICLE_SIZE = particle_render_params(GRID_N)

box_mpc = GRID_N * LL_BOX
print(f"Grid: {GRID_N}³   Box: {box_mpc:.2f} Mpc   "
      f"Subsample: 1/{PARTICLE_SUBSAMPLE}   Dot size: {PARTICLE_SIZE:.2f}")

# =====================================================================
# PRE-COMPUTE DENSITY COLOUR RANGE ACROSS ALL FRAMES
# (sample every ~50th frame to set a global colour scale)
# =====================================================================
print("Pre-computing global colour scale...")
all_rho_samples = []
for f in snap_files[::max(1, N_FRAMES // 50)]:
    rra, _, _, _ = load_snapshot(f)
    pos = rra[::PARTICLE_SUBSAMPLE]
    ix  = np.clip(pos[:, 0].astype(int), 0, Nx - 1)
    iy  = np.clip(pos[:, 1].astype(int), 0, Ny - 1)
    iz  = np.clip(pos[:, 2].astype(int), 0, Nz - 1)
    density = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    np.add.at(density, (ix, iy, iz), 1.0)
    density = gaussian_filter(density, sigma=1.0)
    all_rho_samples.append(np.log10(density[ix, iy, iz] + 1e-3))

all_rho_flat = np.concatenate(all_rho_samples)
GLOBAL_RHO_LO = np.percentile(all_rho_flat, 2)
GLOBAL_RHO_HI = np.percentile(all_rho_flat, 98)
print(f"  log10(rho) range: [{GLOBAL_RHO_LO:.2f}, {GLOBAL_RHO_HI:.2f}]")

cmap_p = plt.get_cmap(CMAP_PARTICLES)

# =====================================================================
# HALO COLOUR NORMALISER (by log10 mass)
# =====================================================================
halo_norm = mcolors.Normalize(vmin=0, vmax=3)   # 10^10 to 10^13 M_sun/h
cmap_h    = plt.get_cmap(CMAP_HALOS)

# =====================================================================
# RENDER LOOP
# =====================================================================
fig = plt.figure(figsize=(8, 7), facecolor="black")
ax  = fig.add_subplot(111, projection="3d", facecolor="black")

# Apply camera distance scaling once before the loop;
# it will be re-applied after each ax.cla() inside the loop too.
set_camera_distance(ax, GRID_N)

writer = FFMpegWriter(
    fps=FPS,
    metadata=dict(title="N-body 3D movie", artist="nbodysim"),
    extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "18", "-preset", "fast"]
)

print(f"Writing {OUTPUT_FILE} ...")
with writer.saving(fig, OUTPUT_FILE, dpi=DPI):
    for fi, snap_path in enumerate(snap_files):

        rra, aa, z, _ = load_snapshot(snap_path)

        # ---- particle positions + density colours ----
        pos_sub, colours = compute_density_colours(
            rra, Nx, Ny, Nz,
            PARTICLE_SUBSAMPLE,
            GLOBAL_RHO_LO, GLOBAL_RHO_HI,
            cmap_p
        )

        # ---- halos (if catalogue exists near this redshift) ----
        halo_cat = find_nearest_halo_catalogue(z, HALOES_DIR, tol=0.5)
        halo_mass, halo_pos = (np.array([]), np.zeros((0, 3)))
        if halo_cat:
            halo_mass, halo_pos = read_halo_catalogue(halo_cat, LL_BOX)

        # ---- rotating camera ----
        azim = (CAMERA_AZIM_START
                + fi / N_FRAMES * 360.0 * CAMERA_AZIM_FULL_ROTATIONS) % 360

        # ---- draw frame ----
        ax.cla()
        ax.set_facecolor("black")
        ax.set_xlim(0, Nx); ax.set_ylim(0, Ny); ax.set_zlim(0, Nz)
        ax.set_xlabel("x", color="#888888", labelpad=2, fontsize=8)
        ax.set_ylabel("y", color="#888888", labelpad=2, fontsize=8)
        ax.set_zlabel("z", color="#888888", labelpad=2, fontsize=8)
        ax.tick_params(colors="#555555", labelsize=6)
        for spine in [ax.xaxis, ax.yaxis, ax.zaxis]:
            spine.pane.fill  = False
            spine.pane.set_edgecolor("#222222")
        ax.grid(False)

        # Camera must be reset after cla()
        ax.view_init(elev=CAMERA_ELEV, azim=azim)
        set_camera_distance(ax, GRID_N)

        # Particles
        ax.scatter(
            pos_sub[:, 0], pos_sub[:, 1], pos_sub[:, 2],
            c=colours,
            s=PARTICLE_SIZE,
            alpha=PARTICLE_ALPHA,
            linewidths=0,
            depthshade=False,
            rasterized=True
        )

        # Halos
        if len(halo_mass) > 0:
            log_m   = np.log10(np.maximum(halo_mass, 1e-3))
            h_sizes = HALO_SIZE_SCALE * np.clip(log_m, 0, 3)
            h_cols  = cmap_h(halo_norm(log_m))
            ax.scatter(
                halo_pos[:, 0], halo_pos[:, 1], halo_pos[:, 2],
                c=h_cols,
                s=h_sizes,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.3,
                depthshade=False,
                zorder=10,
                label=f"Halos (N={len(halo_mass)})"
            )

        # Labels
        ax.set_title(
            f"L = {box_mpc:.2f} Mpc  |  "
            f"{GRID_N}³ particles  |  "
            f"z = {z:.2f}  a = {aa:.4f}",
            color="white", fontsize=9, pad=4
        )

        # Redshift ticker in corner
        fig.text(0.02, 0.02,
                 f"z = {z:6.2f}",
                 color="cyan", fontsize=13, fontweight="bold",
                 transform=fig.transFigure)

        writer.grab_frame()

        if fi % 100 == 0 or fi == N_FRAMES - 1:
            print(f"  Frame {fi+1:4d}/{N_FRAMES}  z={z:.3f}  a={aa:.4f}")

print(f"\nMovie saved: {OUTPUT_FILE}")
print(f"  {N_FRAMES} frames  |  {FPS} fps  |  "
      f"{N_FRAMES/FPS:.1f} s  |  DPI={DPI}")