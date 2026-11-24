"""Sightline spectra for the DREAM SPI cases with and without the W + B background.

Six runs: per shot (40737, 40949), the case without impurities and the case
with impurities for W couplings cl and ca (B uses its default variant, llu).
Steps lie on a uniform t_rel grid per shot (axuv.dream.sample_grid): from the
earliest fragment arrival minus two steps to eight steps after the thermal
quench onset, 0.5 ms apart, the same grid for all runs of a shot. Each step
takes the nearest saved DREAM step and records its actual time.

One HDF5 file per step, written under a temporary name and renamed when
complete; a restart skips the steps already done. Steps run with time on the
outside and the runs of a shot on the inside, so an interruption leaves
complete with/without comparisons. Create the file STOP in the output root to
stop after the steps in progress.

    python publications/run_dream_spi_sightlines.py [--workers N]
"""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import h5py
import numpy as np

from axuv.dream import load_case, sample_grid
from axuv.io import OUTPUT_DIR

DREAM_DIR = Path.home() / "Data" / "AUG-AXUV" / "DREAM_SPI"
SHOTS = ("40737", "40949")
# (impurity case, W variant)
RUNS = (("noImpurities", None), ("withImpurities", "cl"), ("withImpurities", "ca"))
PIXEL_SAMPLES = 1


def run_name(shot, impurities, w_variant):
    return f"{shot}_{impurities}" + (f"_W{w_variant}" if w_variant else "")


def step_name(t_rel):
    return f"trel{t_rel * 1e3:+07.3f}ms.h5"


def _case_dir(shot, impurities):
    return DREAM_DIR / f"{shot}_nr11_{impurities}"


def run_step(shot, impurities, w_variant, t_rel_grid, output):
    """Observe one step and write its HDF5 file (worker process)."""
    sys.path.insert(0, str(Path(__file__).parent))
    from sightline_DREAMoutput_Ne_SPI import build_scene, model_parts, observe

    from axuv.responsivity import degraded_sensitivity_function, sensitivity_function, worst_sensitivity

    started = time.time()
    case = load_case(_case_dir(shot, impurities))
    index = case.index_at(t_rel_grid)
    los, pipeline, _, plasma = build_scene(case, index, PIXEL_SAMPLES)
    variants = {"W": w_variant} if w_variant else None
    parts = model_parts(plasma, variants)
    total, variance = observe(los, pipeline, plasma, [m for ms in parts.values() for m in ms])
    part_spectra = {}
    if len(parts) > 1:
        for name, models in parts.items():
            part_spectra[name] = observe(los, pipeline, plasma, models)[0]
    wavelengths = np.array(pipeline.wavelengths)
    if not (np.all(np.isfinite(total)) and np.all(total >= 0)):
        raise ValueError(f"{run_name(shot, impurities, w_variant)} t_rel {t_rel_grid}: spectrum not finite or negative")
    if part_spectra:
        mismatch = np.abs(sum(part_spectra.values()) - total).max() / total.max()
        if mismatch > 1e-9:
            raise ValueError(f"parts do not sum to the total (relative {mismatch:.2e})")

    energies = 1239.8 / wavelengths
    bin_width = (wavelengths[-1] - wavelengths[0]) / (len(wavelengths) - 1)
    power = float(total.sum() * bin_width)
    eta = {
        name: float(np.sum(total * function(energies)) / np.sum(total))
        for name, function in (("nominal", sensitivity_function),
                               ("degraded", degraded_sensitivity_function),
                               ("worst", worst_sensitivity))
    }
    source_file, source_index = case.source[index]
    temporary = output.with_suffix(".tmp")
    with h5py.File(temporary, "w") as h5f:
        h5f.create_dataset("spectral_power_mean", data=total)
        h5f.create_dataset("spectral_power_variance", data=variance)
        h5f.create_dataset("wavelengths_nm", data=wavelengths)
        h5f.create_dataset("photon_energies_eV", data=energies)
        for name, spectrum in part_spectra.items():
            h5f.create_dataset(f"parts/{name}", data=spectrum)
            h5f.attrs[f"share_{name}"] = float(spectrum.sum() / total.sum())
        h5f.attrs["shot"] = shot
        h5f.attrs["impurities"] = impurities
        h5f.attrs["w_variant"] = w_variant or ""
        h5f.attrs["b_variant"] = "llu (default)" if w_variant else ""
        h5f.attrs["t_rel_grid_s"] = t_rel_grid
        h5f.attrs["t_rel_s"] = float(case.t_rel[index])
        h5f.attrs["t_joined_s"] = float(case.t[index])
        h5f.attrs["t_onset_joined_s"] = case.t_onset
        h5f.attrs["t_arrival_rel_s"] = case.t_arrival - case.t_onset
        h5f.attrs["source_file"] = source_file
        h5f.attrs["source_index"] = source_index
        h5f.attrs["total_power_W"] = power
        for name, value in eta.items():
            h5f.attrs[f"eta_eff_{name}_A_per_W"] = value
        h5f.attrs["pixel_samples"] = PIXEL_SAMPLES
        h5f.attrs["spectral_bins"] = len(wavelengths)
        h5f.attrs["flags"] = "\n".join(case.flags)
        h5f.attrs["parts_note"] = ("parts/W and parts/B: line emission of that element; parts/rest: "
                                   "all other lines and Bremsstrahlung of all species")
    os.replace(temporary, output)
    return time.time() - started


def _tasks(root):
    """All steps in run order: time outside, runs of each shot inside."""
    grids = {}
    for shot in SHOTS:
        cases = [load_case(_case_dir(shot, impurities)) for impurities in ("noImpurities", "withImpurities")]
        grids[shot] = sample_grid(cases)
    tasks = []
    for k in range(max(len(g) for g in grids.values())):
        for shot in SHOTS:
            if k >= len(grids[shot]):
                continue
            t_rel = float(grids[shot][k])
            for impurities, w_variant in RUNS:
                output = root / run_name(shot, impurities, w_variant) / step_name(t_rel)
                tasks.append((shot, impurities, w_variant, t_rel, output))
    return grids, tasks


def _install_first(tasks):
    """Install the atomic data of every run serially before parallel work starts."""
    sys.path.insert(0, str(Path(__file__).parent))
    from sightline_DREAMoutput_Ne_SPI import build_scene, model_parts

    seen = set()
    for shot, impurities, w_variant, t_rel, _ in tasks:
        if (impurities, w_variant) in seen:
            continue
        seen.add((impurities, w_variant))
        case = load_case(_case_dir(shot, impurities))
        _, _, _, plasma = build_scene(case, case.index_at(t_rel), PIXEL_SAMPLES)
        model_parts(plasma, {"W": w_variant} if w_variant else None)


def _record_code(root):
    """Save the code state next to the outputs (the runs may use uncommitted code)."""
    repo = Path(__file__).resolve().parents[1]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    diff = subprocess.run(["git", "diff", "HEAD"], cwd=repo, capture_output=True, text=True).stdout
    status = subprocess.run(["git", "status", "--short"], cwd=repo, capture_output=True, text=True).stdout
    code = root / "code"
    code.mkdir(parents=True, exist_ok=True)
    (code / "commit.txt").write_text(commit + "\n" + status)
    (code / "diff.patch").write_text(diff)
    for path in ("axuv/dream.py", "publications/sightline_DREAMoutput_Ne_SPI.py",
                 "publications/run_dream_spi_sightlines.py"):
        (code / Path(path).name).write_text((repo / path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_DIR / "1D_DREAM_SPI")
    args = parser.parse_args()

    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    stop = root / "STOP"
    grids, tasks = _tasks(root)
    (root / "grids.json").write_text(json.dumps({s: list(g) for s, g in grids.items()}, indent=1))
    _record_code(root)
    todo = [task for task in tasks if not task[-1].exists()]
    print(f"{len(tasks)} steps, {len(tasks) - len(todo)} done, {len(todo)} to run; stop with: touch {stop}",
          flush=True)
    for task in todo:
        task[-1].parent.mkdir(parents=True, exist_ok=True)
    _install_first(todo)

    failed = 0
    with ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=1) as pool:
        pending = {}
        queue = list(todo)
        while queue or pending:
            while queue and len(pending) < args.workers and not stop.exists():
                task = queue.pop(0)
                pending[pool.submit(run_step, *task)] = task
            if not pending:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                shot, impurities, w_variant, t_rel, output = pending.pop(future)
                label = f"{run_name(shot, impurities, w_variant)} t_rel {t_rel * 1e3:+.1f} ms"
                try:
                    print(f"done {label} in {future.result():.0f} s", flush=True)
                except Exception:
                    failed += 1
                    print(f"FAILED {label}\n{traceback.format_exc()}", flush=True)
            if stop.exists() and queue:
                print(f"STOP found; finishing {len(pending)} steps in progress", flush=True)
                queue = []
    remaining = sum(not task[-1].exists() for task in tasks)
    print(f"finished: {remaining} steps not done, {failed} failed", flush=True)


if __name__ == "__main__":
    main()
