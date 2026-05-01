#!/usr/bin/env python3
"""
nbody_movie_renderer.py
------------------------
3D movie renderer for N-body simulation snapshots with halo tracking.

Halo display strategy (two layers):
  1. FOF catalogue markers  — cyan dots, from halo_catalogue binary files
  2. Overdensity markers    — orange dots, detected directly from particle
                              density field using a smoothed 3D grid peak-finder
                              with PERIODIC boundary conditions.
     This catches clusters that straddle box boundaries and are missed by FOF
     due to the left_flag/right_flag dual-boundary bug in fof_main.c.

Particle style : warm yellow on black, alpha modulated by local density.
Positions      : grid units 0-128 throughout.

Binary layout of halo_catalogue files (sizeof(io_header)=256, sizeof(long)=4):
    [4B garbage][256B io_header][4B garbage]
    [4B garbage][4B  totcluster][4B garbage]
    [4B garbage][N*7*4B floats ][4B garbage]
    columns: mass, x, y, z, vx, vy, vz
"""

import os, sys, re, glob, argparse, struct
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.animation import FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter, maximum_filter
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="N-body 3D movie renderer with halos")
    p.add_argument("--snap-dir",        default="snapshots",       help="Dir with frame_XXXX.npz")
    p.add_argument("--halo-dir",        default="halos",          help="Dir with halo_catalogue_X.XXX")
    p.add_argument("--output",          default="nbody_movie.mp4", help="Output .mp4 or .gif")
    p.add_argument("--frames",          type=int,   default=None,  help="Max frames (default: all)")
    p.add_argument("--dpi",             type=int,   default=150)
    p.add_argument("--fps",             type=int,   default=30)
    p.add_argument("--trail-len",       type=int,   default=10,    help="Halo trail length in frames")
    p.add_argument("--no-trail",        action="store_true")
    p.add_argument("--png-only",        action="store_true",       help="Save PNGs instead of video")
    p.add_argument("--view-elev",       type=float, default=25.0)
    p.add_argument("--view-azim",       type=float, default=None,  help="Fixed azim (None=rotate)")
    p.add_argument("--Nbox",            type=int,   default=128)
    p.add_argument("--halo-z-tol",      type=float, default=0.05,
                   help="Max |dz| between frame and FOF catalogue to show FOF halos")
    p.add_argument("--od-bins",         type=int,   default=96,
                   help="Grid bins for overdensity peak finder (default 96 for 128-box)")
    p.add_argument("--od-sigma",        type=float, default=1.2,
                   help="Gaussian smoothing sigma for density field (default 1.2)")
    p.add_argument("--od-threshold",    type=float, default=4.0,
                   help="Density threshold in units of mean density (default 4.0)")
    p.add_argument("--od-min-z",        type=float, default=5.0,
                   help="Only show overdensity markers below this redshift (default 5.0)")
    p.add_argument("--diagnose-halos",  action="store_true",
                   help="Print binary structure of halo files and exit")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# HALO BINARY READER
# ─────────────────────────────────────────────────────────────────────────────
SIZEOF_IO_HEADER  = 256
SIZEOF_LONG       = 4
SIZEOF_DUMMY      = 4
OFFSET_TOTCLUSTER = SIZEOF_DUMMY + SIZEOF_IO_HEADER + SIZEOF_DUMMY + SIZEOF_DUMMY  # 268
OFFSET_DATA       = OFFSET_TOTCLUSTER + SIZEOF_LONG + SIZEOF_DUMMY + SIZEOF_DUMMY  # 280


def read_halo_catalogue_binary(filepath):
    EMPTY = np.zeros((0, 7), dtype=np.float32)
    if os.path.getsize(filepath) < OFFSET_DATA:
        return EMPTY
    try:
        with open(filepath, "rb") as f:
            f.seek(OFFSET_TOTCLUSTER)
            raw_n = f.read(SIZEOF_LONG)
            if len(raw_n) < SIZEOF_LONG:
                return EMPTY
            n = struct.unpack("<i", raw_n)[0]
            if n <= 0:
                return EMPTY
            f.seek(OFFSET_DATA)
            raw = f.read(n * 28)
        if len(raw) < 28:
            return EMPTY
        n   = len(raw) // 28
        arr = np.frombuffer(raw[:n*28], dtype=np.float32).reshape(n, 7)
        return arr[np.isfinite(arr[:, 0]) & (arr[:, 0] > 0)]
    except Exception as e:
        print(f"  [halos] Error reading {os.path.basename(filepath)}: {e}")
        return None


def parse_z(fname):
    m = re.search(r"(\d+\.\d+)$", os.path.basename(fname))
    return float(m.group(1)) if m else None


def load_all_halo_catalogues(halo_dir):
    catalogues = {}
    n_empty = n_err = 0
    if not os.path.isdir(halo_dir):
        print(f"  [halos] '{halo_dir}' not found — no FOF halos will be shown.")
        return catalogues
    files = sorted(glob.glob(os.path.join(halo_dir, "halo_catalogue_*")))
    print(f"  [halos] Scanning {len(files)} catalogue files ...")
    for fp in files:
        z = parse_z(fp)
        if z is None:
            continue
        data = read_halo_catalogue_binary(fp)
        if data is None:
            n_err += 1
        elif len(data) == 0:
            n_empty += 1
        else:
            catalogues[z] = data
    loaded = len(catalogues)
    print(f"  [halos] {loaded} epochs with halos | "
          f"{n_empty} empty (no halos yet) | {n_err} unreadable")
    if loaded:
        zv = sorted(catalogues.keys())
        print(f"  [halos] FOF redshift range: z={zv[-1]:.3f} -> z={zv[0]:.3f}")
    return catalogues


def get_halos_for_frame(cats, z, z_tol=0.05):
    if not cats:
        return None, None
    zarr = np.array(list(cats.keys()))
    idx  = int(np.argmin(np.abs(zarr - z)))
    zk   = zarr[idx]
    if abs(zk - z) > z_tol:
        return None, None
    return cats[zk], float(zk)


# ─────────────────────────────────────────────────────────────────────────────
# PARTICLE-BASED OVERDENSITY PEAK FINDER  (periodic boundaries)
# ─────────────────────────────────────────────────────────────────────────────

def find_overdensity_peaks(rra, Nbox, bins=96, sigma=1.2, threshold=4.0):
    """
    Detect density peaks directly from particle positions using a smoothed
    3D density field with PERIODIC boundary conditions.

    Returns array of shape (N_peaks, 3) with peak positions in grid units,
    and array of shape (N_peaks,) with relative density at each peak.

    Uses np.pad with mode='wrap' for periodic smoothing so clusters
    straddling the box boundary are detected correctly.
    """
    edges  = np.linspace(0, Nbox, bins + 1)
    H, _   = np.histogramdd(rra, bins=[edges] * 3)
    H      = H.astype(np.float32)
    mean_d = H.mean()
    if mean_d == 0:
        return np.zeros((0, 3)), np.zeros(0)

    # Periodic smoothing: pad by sigma*3 cells on each side with wrap
    pad = max(1, int(np.ceil(sigma * 3)))
    Hp  = np.pad(H, pad, mode='wrap')
    Hs  = gaussian_filter(Hp, sigma=sigma)
    # Unpad
    Hs  = Hs[pad:-pad, pad:-pad, pad:-pad]

    # Normalise by mean
    Hn = Hs / (mean_d + 1e-30)

    # Find local maxima: a peak is a cell that equals its neighbourhood max
    footprint = np.ones((3, 3, 3), dtype=bool)
    local_max = maximum_filter(Hn, footprint=footprint, mode='wrap')
    peak_mask = (Hn == local_max) & (Hn > threshold)

    if not peak_mask.any():
        return np.zeros((0, 3)), np.zeros(0)

    ix, iy, iz = np.where(peak_mask)
    # Convert bin indices to grid-unit positions (cell centres)
    cell = Nbox / bins
    px   = (ix + 0.5) * cell
    py   = (iy + 0.5) * cell
    pz   = (iz + 0.5) * cell
    dens = Hn[ix, iy, iz]

    # Sort by density descending
    order = np.argsort(-dens)
    return np.column_stack([px[order], py[order], pz[order]]), dens[order]


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────
def diagnose_halo_files(halo_dir):
    files = sorted(glob.glob(os.path.join(halo_dir, "halo_catalogue_*")))
    if not files:
        print("No halo catalogue files found."); return
    print(f"\n{'='*65}\nHALO CATALOGUE BINARY DIAGNOSTICS")
    print(f"  OFFSET_TOTCLUSTER={OFFSET_TOTCLUSTER}  OFFSET_DATA={OFFSET_DATA}\n{'='*65}")
    for fp in files[:10]:
        sz = os.path.getsize(fp)
        print(f"\n{os.path.basename(fp)}  ({sz} bytes)")
        with open(fp, "rb") as f:
            raw = f.read(min(sz, OFFSET_DATA + 56))
        if sz >= OFFSET_TOTCLUSTER + 4:
            n = struct.unpack("<i", raw[OFFSET_TOTCLUSTER:OFFSET_TOTCLUSTER+4])[0]
            print(f"  totcluster={n}  expected_data={n*28}  actual={sz-OFFSET_DATA-4}")
            if n > 0 and sz >= OFFSET_DATA + 28:
                row = np.frombuffer(raw[OFFSET_DATA:OFFSET_DATA+28], dtype=np.float32)
                print(f"  halo[0]: mass={row[0]:.4e} x={row[1]:.3f} "
                      f"y={row[2]:.3f} z={row[3]:.3f}")
    print(f"\n{'='*65}\n")


# ─────────────────────────────────────────────────────────────────────────────
# SNAPSHOT LOADER
# ─────────────────────────────────────────────────────────────────────────────
def load_snapshot(path):
    d   = np.load(path)
    rra = d["rra"].astype(np.float32)
    aa  = float(d["aa"])
    z   = float(d["z"])
    emb = None
    if "halo_xyz" in d and "halo_mass" in d:
        xyz  = d["halo_xyz"].astype(np.float32)
        mass = d["halo_mass"].astype(np.float32)
        if len(mass) > 0:
            emb = np.column_stack([mass, xyz,
                                   np.zeros((len(mass), 3), dtype=np.float32)])
    return rra, aa, z, emb


# ─────────────────────────────────────────────────────────────────────────────
# PARTICLE COLOURS  (warm yellow, alpha from density)
# ─────────────────────────────────────────────────────────────────────────────
_P_RGB = np.array([1.00, 0.97, 0.78], dtype=np.float32)


def particle_colors(rra, Nbox):
    bins  = max(8, Nbox // 4)
    edges = np.linspace(0, Nbox, bins + 1)
    H, _  = np.histogramdd(rra, bins=[edges] * 3)
    H     = gaussian_filter(H.astype(np.float32), sigma=0.5)
    ix = np.clip((rra[:, 0] / Nbox * bins).astype(int), 0, bins - 1)
    iy = np.clip((rra[:, 1] / Nbox * bins).astype(int), 0, bins - 1)
    iz = np.clip((rra[:, 2] / Nbox * bins).astype(int), 0, bins - 1)
    d  = np.log1p(H[ix, iy, iz])
    d -= d.min()
    if d.max() > 0:
        d /= d.max()
    alpha = (0.10 + d * 0.70).astype(np.float32)
    rgba  = np.empty((len(rra), 4), dtype=np.float32)
    rgba[:, 0] = _P_RGB[0]
    rgba[:, 1] = _P_RGB[1]
    rgba[:, 2] = _P_RGB[2]
    rgba[:, 3] = alpha
    return rgba


# ─────────────────────────────────────────────────────────────────────────────
# TRAIL BUFFER
# ─────────────────────────────────────────────────────────────────────────────
class TrailBuf:
    def __init__(self, n):
        self.n    = n
        self.hist = []

    def push(self, xyz):
        self.hist.append(xyz.copy() if xyz is not None else None)
        if len(self.hist) > self.n:
            self.hist.pop(0)

    def frames(self):
        for i, e in enumerate(self.hist[:-1]):
            if e is not None:
                yield e, (i + 1) / len(self.hist) * 0.35


# ─────────────────────────────────────────────────────────────────────────────
# AXIS STYLE
# ─────────────────────────────────────────────────────────────────────────────
def style_ax(ax, N):
    ax.set_facecolor("black")
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("none")
    ax.grid(False)
    for sp in [ax.xaxis, ax.yaxis, ax.zaxis]:
        sp.line.set_color("none")
    ax.tick_params(colors="none")
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
    ax.set_xlim(0, N); ax.set_ylim(0, N); ax.set_zlim(0, N)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RENDER
# ─────────────────────────────────────────────────────────────────────────────
def render_movie(args):
    if args.diagnose_halos:
        diagnose_halo_files(args.halo_dir); sys.exit(0)

    snaps = sorted(glob.glob(os.path.join(args.snap_dir, "frame_*.npz")))
    if not snaps:
        sys.exit(f"No snapshots found in '{args.snap_dir}'")
    if args.frames:
        snaps = snaps[:args.frames]
    NF   = len(snaps)
    Nbox = args.Nbox
    print(f"Found {NF} snapshots")

    cats  = load_all_halo_catalogues(args.halo_dir)
    trail = TrailBuf(args.trail_len)   # trails for overdensity peaks

    rra0, aa0, z0, _ = load_snapshot(snaps[0])
    print(f"First frame: a={aa0:.5f}  z={z0:.3f}  particles={len(rra0):,}")
    print(f"Overdensity finder: bins={args.od_bins}  sigma={args.od_sigma}  "
          f"threshold={args.od_threshold}x mean  active below z={args.od_min_z}")

    fig = plt.figure(figsize=(10, 9), facecolor="black")
    ax  = fig.add_subplot(111, projection="3d")
    style_ax(ax, Nbox)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    ttxt = fig.text(0.03, 0.96, "", color="white",   fontsize=13,
                    fontfamily="monospace", va="top")
    htxt = fig.text(0.03, 0.91, "", color="#00ffff",  fontsize=10,
                    fontfamily="monospace", va="top")
    otxt = fig.text(0.03, 0.87, "", color="#ff9900",  fontsize=10,
                    fontfamily="monospace", va="top")

    # Legend
    leg_patches = [
        mpatches.Patch(facecolor="cyan",    edgecolor="none", label="FOF halo  (cyan)"),
        mpatches.Patch(facecolor="#ff9900", edgecolor="none",
                       label="Density peak  (orange)\nincl. boundary clusters"),
    ]
    fig.legend(handles=leg_patches, loc="lower right",
               bbox_to_anchor=(0.97, 0.03),
               framealpha=0.15, labelcolor="white", fontsize=7)

    png_dir = writer = None
    if args.png_only:
        png_dir = args.output.replace(".mp4", "_frames").replace(".gif", "_frames")
        os.makedirs(png_dir, exist_ok=True)
        print(f"PNG mode -> '{png_dir}/'")
    else:
        writer = (PillowWriter(fps=args.fps) if args.output.endswith(".gif")
                  else FFMpegWriter(fps=args.fps,
                                    extra_args=["-vcodec", "libx264",
                                                "-crf", "18",
                                                "-pix_fmt", "yuv420p"]))
        writer.setup(fig, args.output, dpi=args.dpi)
        print(f"Writing '{args.output}'  fps={args.fps}  dpi={args.dpi}")

    azim_rate = 360.0 / NF

    for fi, sp in enumerate(tqdm(snaps, desc="Rendering")):
        rra, aa, z, emb = load_snapshot(sp)

        # ── FOF halos ──────────────────────────────────────────────────────
        if emb is not None and len(emb):
            fof_hn, fof_mz = emb, z
        elif cats:
            fof_hn, fof_mz = get_halos_for_frame(cats, z, args.halo_z_tol)
        else:
            fof_hn = fof_mz = None

        # ── Overdensity peaks (periodic, catches boundary clusters) ────────
        od_xyz = od_dens = None
        if z <= args.od_min_z:
            od_xyz, od_dens = find_overdensity_peaks(
                rra, Nbox,
                bins=args.od_bins,
                sigma=args.od_sigma,
                threshold=args.od_threshold
            )
            if len(od_xyz) == 0:
                od_xyz = od_dens = None

        trail.push(od_xyz)
        cols = particle_colors(rra, Nbox)

        ax.cla()
        style_ax(ax, Nbox)

        # ── Particles ──────────────────────────────────────────────────────
        ax.scatter(rra[:, 0], rra[:, 1], rra[:, 2],
                   c=cols, s=0.4,
                   linewidths=0, rasterized=True, depthshade=False)

        # ── Overdensity peak trails ────────────────────────────────────────
        if not args.no_trail:
            for txyz, talpha in trail.frames():
                ax.scatter(txyz[:, 0], txyz[:, 1], txyz[:, 2],
                           c="#ff9900", s=8, alpha=talpha,
                           linewidths=0, depthshade=False)

        # ── Set view angle ─────────────────────────────────────────────────
        azim = (args.view_azim if args.view_azim is not None
                else (30 + fi * azim_rate) % 360)
        ax.view_init(elev=args.view_elev, azim=azim)

        # ── FOF halo dots (cyan) ───────────────────────────────────────────
        n_fof = 0
        if fof_hn is not None and len(fof_hn):
            ax.scatter(fof_hn[:, 1], fof_hn[:, 2], fof_hn[:, 3],
                       c="cyan", s=30, marker="o",
                       linewidths=0, depthshade=False, zorder=5)
            n_fof = len(fof_hn)

        # ── Overdensity peak dots (orange) ─────────────────────────────────
        n_od = 0
        if od_xyz is not None and len(od_xyz):
            ax.scatter(od_xyz[:, 0], od_xyz[:, 1], od_xyz[:, 2],
                       c="#ff9900", s=30, marker="o",
                       linewidths=0, depthshade=False, zorder=5)
            n_od = len(od_xyz)

        ttxt.set_text(f"N-body  z={z:7.3f}   a={aa:.5f}   frame {fi+1:04d}/{NF}")
        htxt.set_text(f"FOF halos : {n_fof:>4d}" +
                      (f"  (cat z={fof_mz:.3f})" if fof_mz and n_fof else
                       "  (none yet)" if not n_fof else ""))
        otxt.set_text(f"OD peaks  : {n_od:>4d}" if z <= args.od_min_z
                      else "OD peaks  : (z too high)")

        if args.png_only:
            plt.savefig(os.path.join(png_dir, f"frame_{fi:04d}.png"),
                        dpi=args.dpi, facecolor="black")
        else:
            writer.grab_frame()

    if not args.png_only and writer:
        writer.finish()
        print(f"\nMovie saved -> '{args.output}'")
    elif args.png_only:
        print(f"\nFrames saved -> '{png_dir}/'")
        print(f"  ffmpeg -framerate {args.fps} -i '{png_dir}/frame_%04d.png' "
              f"-c:v libx264 -crf 18 -pix_fmt yuv420p nbody_movie.mp4")

    plt.close(fig)


if __name__ == "__main__":
    render_movie(parse_args())