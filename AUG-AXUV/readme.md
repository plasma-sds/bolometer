# ASDEX Upgrade AXUV synthetic diagnostics and experimental signal plotting
- Using Cherab [GitHub](https://github.com/cherab/core) and Raysect [GitHub](https://github.com/raysect/source)
- Option to use external input profiles from DREAM [GitHub](https://github.com/chalmersplasmatheory/DREAM)

## AXUV directory
Contains main 3D synthetic diagnostics setup with AUG wall CAD file importing and reflections together with plotting files.

## measurement directory
Contains tools for experimental signal analysis and plotting files.

### For Ne lines
`populate()` in `create.py` can be used as an example in to download Ne data from OpenADAS, the file is located in `cherab/core/cherab/openadas/repository/`.

### `sightline_DREAMoutput_Ne_SPI.py` 
A simple LOS modell using DREAM SPI simulation output
