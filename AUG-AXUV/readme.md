# ASDEX Upgrade (AUG) AXUV synthetic diagnostics and experimental signal plotting
- Using Cherab [GitHub](https://github.com/cherab/core) and Raysect [GitHub](https://github.com/raysect/source)
- Option to use external input profiles from DREAM [GitHub](https://github.com/chalmersplasmatheory/DREAM)
- Option to use external input profiles from JOREK 2D or 3D simulations [website](https://www.jorek.eu/) - 2D profiles or 2D slices from 3D simulations is supported for now

## Data requirements
### For 3D synthetic diagnostics if reflections are NOT needed
- Depending on plasma composition, the photon emissivity coefficients (PEC) are needed to be downloaded by Cherab in either ADAS or OpenADAS
  - Ne lines are provided as example, see below
- the AXUV diode sensitivity is provided in `/data` as specified by the manufacturer, as well as a degraded measured sensitivity (only at a part of the spectrum)
  - these are set up for usage by functions
- AXUV diode geometry datafile from AUG
- preferably some simulation input for AUG (electron temperature, plasma composition with electron density and ion charge state densities), however it works with arbitrary data too

### For 3D synthetic diagnostics if reflections ARE needed
- everything from above
- AUG wall geometry CAD files
  - NOTE: depending on the machine configuration, some CAD files might have components that fully or partially cover diode lines of sight, int his case it is recommended to either source the appropriate CAD files (recommended) or edit the CAD files
 
### For experimental signal plotting
- in principle, must be run on AUG IPP server with the user being a part of the AUG group, thereby having access to `aug_sfutils`
  - some functions work outside of the IPP server

## AXUV directory
Contains main 3D synthetic diagnostics setup with AUG wall CAD file importing and reflections together with plotting files.

## measurement directory
Contains tools for experimental signal analysis and plotting files. There is an intermediate step between getting AXUV data from the AUG shotfiles and plotting it where relevant parts of the data is smoothed and saved to HDF5 files (see comments in functions).

## For Ne lines
`populate()` in `create.py` can be used as an example in to download Ne data from OpenADAS, the file is located in `cherab/core/cherab/openadas/repository/`.

## `sightline_DREAMoutput_Ne_SPI.py` 
A simple LOS modell using DREAM SPI simulation output
