## This was used to run the 1D (1 LOS) synthetic diagnostics

Forced locale to `C` so that the dots survive in the numbers.

```bash
for time in $(LC_ALL=C seq -f "%.4f" 2.3000 0.0005 2.3100); do
    echo "=== Observing at t=${time} s ==="
    python sightline_DREAMoutput_Ne_SPI.py "$time" --pickle
done
```

## These options were run for the sensitivity calculation

- All with the default resolution of NR=60 and NZ=110. 
- Note the `nice` and `taskset` commands are machine-specific restrictions.
- The file in `--jorek-file axuv/data/step02410_out.h5` is used to mask the poloidal cross-section to the plasma cross-section. 

### S5 reflections
```bash
python axuv/raytransfer_sensitivity.py --sectors S5 --reflections --pixel-samples 100000 --ray-max-depth 5 --jorek-file axuv/data/step02410_out.h5 --observe-processes 10
```

### S16 reflections
```bash
python axuv/raytransfer_sensitivity.py --sectors S16 --reflections --pixel-samples 100000 --ray-max-depth 5 --jorek-file axuv/data/step02410_out.h5 --observe-processes 10
```

### S5 no reflections
```bash
python axuv/raytransfer_sensitivity.py --sectors S5 --jorek-file axuv/data/step02410_out.h5 --observe-processes 10
```

### S16 no reflections
```bash
python axuv/raytransfer_sensitivity.py --sectors S16 --jorek-file axuv/data/step02410_out.h5 --observe-processes 10
```

### S5 etendue
```bash
python axuv/raytransfer_sensitivity.py --sectors S5 --observe-processes 10 --etendue-mode
```

### S16 etendue
```bash
python axuv/raytransfer_sensitivity.py --sectors S16 --observe-processes 10 --etendue-mode
```

## To verify the `RayTransferPipeline` against the `PowerPipeline` for sector 16
```bash
python publications/verify_uniform_emitter.py --sectors S16 --processes 10
```

## High spatial resolution --- 1 by 1 cm

### No reflections
```bash
nice -n 5 taskset -c 0-9 python axuv/raytransfer_sensitivity.py --sectors S5 --jorek-file axuv/data/step02410_out.h5 --observe-processes 10 --resolution-r 120 --resolution-z 220 -o axuv/data/raytransfer_S5_highres_norefl.h5

nice -n 5 taskset -c 10-19 python axuv/raytransfer_sensitivity.py --sectors S16 --jorek-file axuv/data/step02410_out.h5 --observe-processes 10 --resolution-r 120 --resolution-z 220 -o axuv/data/raytransfer_S16_highres_norefl.h5
```

### Reflections
```bash
nice -n 5 taskset -c 20-29 python axuv/raytransfer_sensitivity.py --sectors S5 --jorek-file axuv/data/step02410_out.h5 --observe-processes 10 --resolution-r 120 --resolution-z 220 --reflections -o axuv/data/raytransfer_S5_highres_refl.h5

nice -n 5 taskset -c 30-39 python axuv/raytransfer_sensitivity.py --sectors S16 --jorek-file axuv/data/step02410_out.h5 --observe-processes 10 --resolution-r 120 --resolution-z 220 --reflections -o axuv/data/raytransfer_S16_highres_refl.h5
```

## Threshold-related plotting (W_th vertical lines)

The fractional `W_th` levels at which vertical lines are drawn are defined in a
single place — `WTH_THRESHOLDS` in `axuv/io.py`. `publications/plot_time_traces.py`
is the producer: it computes the threshold crossing times for each shot and
writes them to `output/threshold_times.json` (the `"sim"` times are used by the
synthetic figures, the `"exp"` times by the experimental ones). The two wrapper
scripts below run that step first, then the relevant plotting script. Both
resolve the repo root from their own location, set `MPLBACKEND=Agg`, and put the
repo root on `PYTHONPATH`, so they run on a headless server from any checkout.

To change the threshold levels, edit `WTH_THRESHOLDS` once and delete
`output/threshold_times.json` before re-running (cached entries are not
recomputed).

### Synthetic figures — wherever the synthetic data lives
Runs `plot_time_traces.py`, then `emission_to_measurement.py` on the four
dataset directories.
```bash
axuv/shell_scripts/plot_and_save_synthetic_measurements.sh
```

### Experimental figures — on the AUG shotfile server
Runs `plot_time_traces.py`, then `plotting_from_shotfiles.py`.
```bash
axuv/shell_scripts/plot_and_save_experimental_measurements.sh
```

### Running a single synthetic dataset by hand
`emission_to_measurement.py` takes the emissions directory as a positional
argument. Vertical lines are drawn by default (levels from `WTH_THRESHOLDS`);
override or disable them per run:
```bash
# Default thresholds
python axuv/emission_to_measurement.py axuv/data/40673/P1/output

# Custom threshold levels for this run only
python axuv/emission_to_measurement.py axuv/data/40673/P1/output --thresholds 0.8 0.2

# No vertical lines (only the no-line figures are saved)
python axuv/emission_to_measurement.py axuv/data/40673/P1/output --no-thresholds
```
