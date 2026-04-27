## These options were run for the sensitivity calculation

- All with the default resolution of NR=60 and NZ=110. 
- Note the `nice` and `taskset` commands are machine-specific restrictions.
- The file in `--jorek-file axuv/data/step02410_out.h5` is used to mask the poloidal cross-section to the plasma cross-section. 

### S5 reflections
```bash
nice -n 5 taskset -c 0-9 python axuv/raytransfer_sensitivity.py --sectors S5 --reflections --pixel-samples 100000 --ray-max-depth 5 --jorek-file axuv/data/step02410_out.h5
```

### S16 reflections
```bash
nice -n 5 taskset -c 10-19 python axuv/raytransfer_sensitivity.py --sectors S16 --reflections --pixel-samples 100000 --ray-max-depth 5 --jorek-file axuv/data/step02410_out.h5
```

### S5 no reflections
```bash
nice -n 5 taskset -c 20-29 python axuv/raytransfer_sensitivity.py --sectors S5 --jorek-file axuv/data/step02410_out.h5
```

### S16 no reflections
```bash
nice -n 5 taskset -c 30-39 python axuv/raytransfer_sensitivity.py --sectors S16 --jorek-file axuv/data/step02410_out.h5
```
