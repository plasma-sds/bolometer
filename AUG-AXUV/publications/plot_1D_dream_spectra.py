# %%
from scipy.constants import alpha
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

from axuv.io import PROJECT_ROOT
from axuv.plotting import set_plt_rcparams
from axuv.responsivity import degraded_sensitivity_function, sensitivity_function, worst_sensitivity

set_plt_rcparams()

axuv_data_dir = PROJECT_ROOT / "axuv" / "data"
data_dir = PROJECT_ROOT / "output" / "1D_data"
output_dir = PROJECT_ROOT / "output" / "1D_plots"
os.makedirs(output_dir, exist_ok=True)

# %% Figure 1: time evolution of η_eff

times = []
eta_effs_degraded = []
eta_effs_nominal = []
eta_effs_worst = []

for hdf5_path in sorted(data_dir.glob("highres_*.h5")):
    with h5py.File(hdf5_path, "r") as h5f:
        spectral_power = np.asarray(h5f["spectral_power_mean"])
        photon_energies = np.asarray(h5f["photon_energies_eV"])
        t = float(h5f.attrs["observation_time_s"])

    R = degraded_sensitivity_function(photon_energies)
    R_nominal = sensitivity_function(photon_energies)
    R_worst = worst_sensitivity(photon_energies)

    eta_effs_degraded.append(np.sum(spectral_power * R) / np.sum(spectral_power))
    eta_effs_nominal.append(np.sum(spectral_power * R_nominal) / np.sum(spectral_power))
    eta_effs_worst.append(np.sum(spectral_power * R_worst) / np.sum(spectral_power))
    times.append(t)

times = np.array(times)
eta_effs_degraded = np.array(eta_effs_degraded)
eta_effs_nominal = np.array(eta_effs_nominal)
eta_effs_worst = np.array(eta_effs_worst)

fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(times, eta_effs_nominal, color="k", marker="o", markersize=4, label="Nominal")
ax.plot(times, eta_effs_degraded, color="r", ls="--", marker="o", markersize=4, label="Degraded")
ax.plot(times, eta_effs_worst, color="b", ls="-.", marker="o", markersize=4, label="Worst estimated")
ax.fill_between(times, eta_effs_nominal, eta_effs_worst, hatch="//", facecolor="none", alpha=0.3, edgecolor="grey")

ax.axvline(2.301, linestyle="--", color="blue", linewidth=2)
ax.axvline(2.3035, linestyle="--", color="magenta", linewidth=2)
ax.set_xlabel("Time (s)")
ax.set_ylabel(r"$\eta_\text{eff}$ (A W$^{-1}$)")
ax.set_ylim(0, 0.3)
ax.set_xlim(2.3, 2.31)
ax.legend(fontsize="small", ncols=2, framealpha=1.0)
plt.savefig(output_dir / "eta_eff_time_evolution.png")
plt.savefig(output_dir / "eta_eff_time_evolution.pdf", format="PDF")
plt.show()
plt.close()
print("Saved eta_eff time evolution")

# %% Figure 2: spectra at two time steps

_SPECTRA = [
    (2.3010, "blue",    r"$t = 2.301\,\mathrm{s}$"),
    (2.3035, "magenta", r"$t = 2.3035\,\mathrm{s}$"),
]

fig, ax = plt.subplots(figsize=(8, 4.5))
ax2 = ax.twinx()

for t_val, color, label in _SPECTRA:
    fname = data_dir / f"highres_{t_val:.4f}s.h5"
    with h5py.File(fname, "r") as h5f:
        power = np.asarray(h5f["spectral_power_mean"])
        energies = np.asarray(h5f["photon_energies_eV"])
        max_energy = 1239.8 / float(h5f.attrs["min_wavelength_nm"])

    # sort ascending in photon energy (HDF5 stores ascending wavelength → descending energy)
    order = np.argsort(energies)
    energies = energies[order]
    power = power[order]

    # Add another datapoint after the largest energy so that the plotting doesn't get cut off
    # due to using plt.step(..., where="mid")
    energies = np.append(energies, max_energy)
    power = np.append(power, power[-1])

    ax.step(energies, power, where="mid", color=color, label=label, linewidth=1)

ax.set_yscale("log")

# overlay degraded responsivity on right axis using energy grid from last file
R = degraded_sensitivity_function(energies)
ax2.plot(energies, R, color="k", linestyle="--", linewidth=2,
         label="Degraded responsivity")

ax.set_xscale("log")
ax.set_xlim(1, 5e3)
ax.set_ylim(1e-4, 1e9)
ax.set_xlabel("Photon energy (eV)")
ax.set_ylabel(r"Spectral power (W eV$^{-1}$)")
ax2.set_ylabel(r"Responsivity (A W$^{-1}$)")

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=12)

plt.savefig(output_dir / "spectra_2t.png")
plt.savefig(output_dir / "spectra_2t.pdf", format="PDF")
plt.show()
plt.close()
print("Saved two-spectra figure to", output_dir)

# %% Figure 3: DREAM average electron temperature and plasma current time evolution

savefolder = PROJECT_ROOT / "output" / "1D_plots"
os.makedirs(savefolder, exist_ok=True)

# Data is a DREAM hdf5 output file: axuv/data/dream_output.h5
# read file with h5py, get datasets from "eqsys": T_cold, I_p
# from "grid": t
with h5py.File(PROJECT_ROOT / "axuv" / "data" / "dream_output.h5", "r") as h5f:
    T_cold = np.asarray(h5f["eqsys"]["T_cold"])
    I_plasma = np.asarray(h5f["eqsys"]["I_p"])
    times = np.asarray(h5f["grid"]["t"]) + 2.3
# electron temperature is 2D (r, t), plasma current is 1D (t)
average_T_cold = np.mean(T_cold, axis=1)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(times, average_T_cold, color="r", label=r"$<T_{\mathrm{e}}>$ [eV]")

ax.set_yscale("log")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Electron temperature [eV]")
ax.set_xlim(2.3, 2.31)

ax2 = ax.twinx()
ax2.plot(times, I_plasma / 1e3, color="k", label=r"I$_{\mathrm{p}}$ [kA]")
ax2.set_ylim(0, 900)
ax2.set_ylabel("Plasma current [kA]")

# add two vertical dashed lines where the spectra were taken
# first after shard arrival, second during TQ
ax2.axvline(2.301, color="b", linestyle="--", label=r"1$^{\mathrm{st}}$ spectrum" + "\nshard arrival")
ax2.axvline(2.3035, color="magenta", linestyle="--", label=r"2$^{\mathrm{nd}}$ spectrum" + "\nduring TQ")

# show one combined legend for the two axes
plt.legend(ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0],
           ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1], loc="upper right")

plt.savefig(PROJECT_ROOT / "output" / "1D_plots" / "dream_time_evolution.png", dpi=300, bbox_inches="tight")
plt.savefig(PROJECT_ROOT / "output" / "1D_plots" / "dream_time_evolution.pdf", format="PDF", bbox_inches="tight")
plt.show()
