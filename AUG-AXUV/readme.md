# ASDEX Upgrade AXUV synthetic diagnostics and experimental signal plotting
- Using Cherab [GitHub](https://github.com/cherab/core) and Raysect [GitHub](https://github.com/raysect/source)
- Option to use external input profiles from plasma simulations

## AXUV directory
Contains main 3D synthetic diagnostics setup with AUG wall CAD file importing and reflections together with plotting files.

## measurement directory
Contains tools for experimental signal analysis and plotting files.

### For Ne lines
Lines 150-151 in `populate()` in `cherab/core/cherab/openadas/repository/create.py` can be used as an example and be expanded to Ne 9+ and then the `populate()` function can be called to download Ne data from OpenADAS.

### `sightline_DREAMoutput_Ne_SPI.py` 
A simple LOS modell using DREAM SPI simulation output
