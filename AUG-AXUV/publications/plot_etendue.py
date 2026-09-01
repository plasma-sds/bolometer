# %%
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from axuv.io import AXUV_DATAFILE, PROJECT_ROOT
from axuv.plotting import set_plt_rcparams

# Update default matplotlib parameters for LaTeX plotting for articles
set_plt_rcparams()

project_dir = PROJECT_ROOT
data_dir = project_dir / "axuv" / "data"

# %% Load etendue data
ETENDUE_PATH_S5 = data_dir / "etendue_S5.h5"
ETENDUE_PATH_S16 = data_dir / "etendue_S16.h5"
with h5py.File(ETENDUE_PATH_S5, "r") as h5f:
    raytraced_etendue_s5 = np.asarray(h5f["raytraced_etendue"])
    raytraced_error_s5 = np.asarray(h5f["raytraced_error"])
    diode_names_s5 = np.asarray(h5f["diode_names"], dtype=str)
print(f"Loaded raytraced etendue from {ETENDUE_PATH_S5}")

with h5py.File(ETENDUE_PATH_S16, "r") as h5f:
    raytraced_etendue_s16 = np.asarray(h5f["raytraced_etendue"])
    raytraced_error_s16 = np.asarray(h5f["raytraced_error"])
    diode_names_s16 = np.asarray(h5f["diode_names"], dtype=str)
print(f"Loaded raytraced etendue from {ETENDUE_PATH_S16}")

# Concatenate S16 data to S5 data
raytraced_etendue = np.concatenate([raytraced_etendue_s5, raytraced_etendue_s16])
raytraced_error = np.concatenate([raytraced_error_s5, raytraced_error_s16])
diode_names = np.concatenate([diode_names_s5, diode_names_s16])

# We only need the etendue and the corresponding diode names
df = pd.read_csv(AXUV_DATAFILE, sep=r"\s+", engine="python").drop(
    columns=[
        "chan",
        "act",
        "con",
        "R_Kabel",
        "U_Gen.",
        "f_Blende",
        "f_Folie",
        "d(Folie-Blende)",
        "delta",
        "gamma",
        "alpha",
        "R_start",
        "z_start",
        "Phi_start",
        "R_end",
        "z_end",
        "Phi_end",
        "F",
        "Foil_ID",
    ]
)
print(f"Loaded AXUV etendue datafile '{AXUV_DATAFILE}' with shape {df.shape}")

# Build typed dicts — names stay str, numerics stay float64
raytraced = {
    "names": np.array([name.split(" ")[-1] for name in diode_names]),
    "etendue": raytraced_etendue.astype(np.float64),
    "error": raytraced_error.astype(np.float64),
}
aug = {
    "names": df["RAW"].values,
    "etendue": df["Faktor"].values.astype(np.float64),
}

# Sort both alphabetically by diode name
rt_order = np.argsort(raytraced["names"])
aug_order = np.argsort(aug["names"])
for key in ("names", "etendue", "error"):
    raytraced[key] = raytraced[key][rt_order]
aug["names"] = aug["names"][aug_order]
aug["etendue"] = aug["etendue"][aug_order]

# Filter AUG to only the diodes present in the raytraced set
mask = np.isin(aug["names"], raytraced["names"])
aug["names"] = aug["names"][mask]
aug["etendue"] = aug["etendue"][mask]
print(f"Filtered AUG data to {aug['names'].size} diodes")

# Sanity checks
print(f"Raytraced diodes: {raytraced['names'].size}, AUG diodes: {aug['names'].size}")
print(
    f"First 2 raytraced: {list(zip(raytraced['names'][:2], raytraced['etendue'][:2]))}"
)
print(f"First 2 AUG:       {list(zip(aug['names'][:2], aug['etendue'][:2]))}")

# %%
raytraced_etendue = raytraced["etendue"] / (4 * np.pi)
raytraced_error = raytraced["error"] / (4 * np.pi)
aug_etendue = aug["etendue"]
diode_indices = range(len(raytraced_etendue))

fig, ax = plt.subplots(figsize=(10, 4.5))

ax.plot(diode_indices, raytraced_etendue, label="Raytraced")
ax.errorbar(
    diode_indices,
    raytraced_etendue,
    yerr=raytraced_error,
    fmt="none",
    ecolor="gray",
    capsize=3,
)
ax.plot(diode_indices, aug_etendue, label="AUG analytical")
ax.set_xlabel("Diode index")
ax.set_ylabel(r"Etendue [m$^2$]")

quarter_labels = ["DVC", "DHC", "D16", "DHT"]
quarter_centers = [0.125, 0.375, 0.625, 0.875]

for x, label in zip(quarter_centers, quarter_labels):
    ax.text(
        x,
        0.02,  # x: quarter center, y: near bottom
        label,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=20,
        color="black",
    )

ax.legend()
for i in range(3):
    ax.axvline(i * 48 + 47.5, linestyle="--", color="gray", linewidth=1)
plt.ylim(0.3e-9, 1e-9)
plt.xlim(0, 192)
plt.savefig(project_dir / "output" / "etendue_comparison.png")
plt.savefig(project_dir / "output" / "etendue_comparison.pdf", format="PDF")

# %%
# Also print the relative error between raytraced and AUG
relative_error = np.abs(raytraced_etendue - aug_etendue) / aug_etendue
print(f"Relative error percentage: {np.mean(relative_error) * 100:.2f}%")
print(f"Max relative error: {np.max(relative_error) * 100:.2f}%")
