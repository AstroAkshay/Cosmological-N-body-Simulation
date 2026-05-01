import numpy as np
import pyvista as pv
import glob
import re

# ---------- READ FUNCTION ----------
def read_hi_map(filename):
    with open(filename, "rb") as f:
        N1 = np.fromfile(f, dtype=np.int32, count=1)[0]
        N2 = np.fromfile(f, dtype=np.int32, count=1)[0]
        N3 = np.fromfile(f, dtype=np.int32, count=1)[0]

        data = np.fromfile(f, dtype=np.float32)
        data = data.reshape((N1, N2, N3))

    return data


# ---------- REDSHIFT ----------
def extract_z(filename):
    match = re.search(r'HI_map_(\d+\.\d+)', filename)
    return float(match.group(1)) if match else -1


# ---------- LOAD FILES ----------
files = sorted(glob.glob("ionz_out/HI_map_*"), key=extract_z, reverse=True)

if len(files) == 0:
    raise RuntimeError("No HI_map_* files found")

print(f"Found {len(files)} files")


# =========================================================
# 🔥 STEP 1: COMPUTE GLOBAL MIN/MAX (LOG SPACE)
# =========================================================
global_min = np.inf
global_max = -np.inf

print("Computing global min/max...")

for f in files:
    cube = read_hi_map(f)

    cube = cube[::2, ::2, ::2]
    cube = np.nan_to_num(cube, nan=0.0, posinf=0.0, neginf=0.0)

    cube_log = np.log10(cube + 1e-10)

    global_min = min(global_min, cube_log.min())
    global_max = max(global_max, cube_log.max())

print(f"Global log10 range: {global_min:.3f} to {global_max:.3f}")


# =========================================================
# 🎬 STEP 2: PLOTTING WITH FIXED COLORBAR
# =========================================================
plotter = pv.Plotter(off_screen=True)
plotter.open_movie("HI_slices_normalized.mp4")


for f in files:
    z = extract_z(f)
    print(f"Processing z = {z:.3f}")

    cube = read_hi_map(f)

    cube = cube[::2, ::2, ::2]
    cube = np.nan_to_num(cube, nan=0.0, posinf=0.0, neginf=0.0)

    cube = np.log10(cube + 1e-10)

    # ---------- GRID ----------
    grid = pv.ImageData()
    grid.dimensions = np.array(cube.shape) + 1
    grid.cell_data["values"] = cube.flatten(order="F")

    plotter.clear()

    # ---------- SLICES ----------
    slices = grid.slice_orthogonal(
        x=0.1,
        y=0.1,
        z=0.1
    )

    plotter.add_mesh(
        slices,
        cmap="plasma",
        clim=[global_min, global_max],  # 🔥 FIXED SCALE
        show_scalar_bar=True
    )

    # ---------- OUTLINE ----------
    #plotter.add_mesh(grid.outline(), color="white", line_width=1)

    # ---------- LABEL ----------
    plotter.add_text(f"z = {z:.2f}", font_size=12)

    plotter.write_frame()


plotter.close()

print("Saved: HI_slices_normalized.mp4")