# %%
import h5py
import numpy as np
import matplotlib.pyplot as plt

from axuv.io import PROJECT_ROOT
from axuv.plotting import set_plt_rcparams
from axuv.responsivity import degraded_sensitivity_function

set_plt_rcparams()

data_dir = PROJECT_ROOT / "output" / "1D_data"
output_dir = PROJECT_ROOT / "output"

# %% Figure 1: time evolution of η_eff

times = []
eta_effs = []

for hdf5_path in sorted(data_dir.glob("highres_*.h5")):
    with h5py.File(hdf5_path, "r") as h5f:
        spectral_power = np.asarray(h5f["spectral_power_mean"])
        photon_energies = np.asarray(h5f["photon_energies_eV"])
        t = float(h5f.attrs["observation_time_s"])

    R = degraded_sensitivity_function(photon_energies)
    eta_effs.append(np.sum(spectral_power * R) / np.sum(spectral_power))
    times.append(t)

times = np.array(times)
eta_effs = np.array(eta_effs)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(times, eta_effs, color="k", marker="o", markersize=4)
ax.axvline(2.301, linestyle="--", color="blue", linewidth=2)
ax.axvline(2.3035, linestyle="--", color="magenta", linewidth=2)
ax.set_xlabel("Time (s)")
ax.set_ylabel(r"$\eta_\text{eff}$ (A W$^{-1}$)")
plt.savefig(output_dir / "eta_eff_time_evolution.png")
plt.savefig(output_dir / "eta_eff_time_evolution.eps", format="EPS")
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

    # sort ascending in photon energy (HDF5 stores ascending wavelength → descending energy)
    order = np.argsort(energies)
    energies = energies[order]
    power = power[order]

    ax.step(energies, power, where="mid", color=color, label=label, linewidth=1)

ax.set_yscale("log")

# overlay degraded responsivity on right axis using energy grid from last file
R = degraded_sensitivity_function(energies)
ax2.plot(energies, R, color="k", linestyle="--", linewidth=2,
         label="Degraded responsivity")

ax.set_xscale("log")
ax.set_xlim(1, 2.5e3)
ax.set_ylim(1e-4, 1e9)
ax.set_xlabel("Photon energy (eV)")
ax.set_ylabel(r"Spectral power (W nm$^{-1}$)")
ax2.set_ylabel(r"Responsivity (A W$^{-1}$)")

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=12)

plt.savefig(output_dir / "spectra_2t.png")
plt.savefig(output_dir / "spectra_2t.eps", format="EPS")
plt.close()
print("Saved two-spectra figure")
