#!/bin/python3

import sys
import json
import pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py
import bisect
import numpy as np
from scipy.interpolate import CubicSpline

from scipy.constants import electron_mass, atomic_mass

from raysect.core import Point3D, Vector3D, translate, rotate, Point2D, rotate_basis
from raysect.optical import World
from raysect.primitive import Cylinder
from raysect.optical.observer import SightLine, SpectralPowerPipeline0D, SpectralRadiancePipeline0D
from raysect.optical.material.emitter.inhomogeneous import NumericalIntegrator

from cherab.core import Species, Maxwellian, Plasma, Line
from cherab.core.atomic.elements import deuterium, neon
from cherab.core.model import ExcitationLine, GaussianLine, RecombinationLine, Bremsstrahlung
from cherab.core.math import Constant3D, ConstantVector3D, sample3d, Function3D
from cherab.openadas import OpenADAS
from cherab.tools.plasmas import GaussianVolume
from cherab.tools.equilibrium import EFITEquilibrium


# change matplotlib renderer so that when calling observe() on observers the script does not get interrupted
matplotlib.use('Agg')

# Convenient constants
XAXIS = Vector3D(1, 0, 0)
YAXIS = Vector3D(0, 1, 0)
ZAXIS = Vector3D(0, 0, 1)
ORIGIN = Point3D(0, 0, 0)

# read time dependent densities from DREAM output file
with h5py.File("dream_output.h5") as do:
    do_ions = do["eqsys/n_i"][()]
    do_n_cold = do["eqsys/n_cold"][()]
    do_n_hot = do["eqsys/n_hot"][()]
    do_n_re = do["eqsys/n_re"][()]
    do_n_free = do_n_cold + do_n_hot + do_n_re
    do_time = 2.3 + do["grid/t"][()]
    do_minor_radius = do["grid/r"][()]
    do_minor_radius_edges = do["grid/r_f"][()]
    do_temperature = do["eqsys/T_cold"][()]
    do_radiated_power = do["other/fluid/Tcold_radiation"][()]
    d0_density = do_ions[:, 0, :]
    d0i_density = do_ions[:, 2, :]
    d1_density = do_ions[:, 1, :]
    d1i_density = do_ions[:, 3, :]
    ne_densities = do_ions[:, 4:, :]

# pass the time from the terminal when to observe the plasma
ti = bisect.bisect_left(do_time, float(sys.argv[1]))

# The plasma will be a cylindrical plasma
MAJOR_RADIUS = 1.7365249  # from DREAM
MINOR_RADIUS = do_minor_radius_edges[-1]
CENTRE_Z = 0
CYLINDER_HEIGHT = 2


class CustomFunction:
    """3D function with data interpolation for temperature and densities"""
    def __init__(self, radialgrid, data):
        self.data = data
        self.radialgrid = radialgrid
        self.dr = abs(self.radialgrid[-1] - self.radialgrid[-2])
        self.interpolated = CubicSpline(self.radialgrid, self.data)

    def __call__(self, x, y, z):
        if abs(z) < (CYLINDER_HEIGHT / 2):
            xnew = x - MAJOR_RADIUS
            ynew = y
            radius = np.sqrt(xnew ** 2 + ynew ** 2)
            if radius <= (self.radialgrid[-1] + (self.dr / 2)):
                return self.interpolated(radius)
            else:
                return 0.
        else:
            return 0.
        
world = World()

# SightLine geometry with transform: will look in the direction of XAXIS, with "up" being the ZAXIS
los = SightLine(parent=world, transform=rotate_basis(XAXIS, ZAXIS)*translate(0, 0, 0))
sppipeline = SpectralPowerPipeline0D()  # default for sightline
sradpipeline = SpectralRadiancePipeline0D()
los.pipelines = [sppipeline, sradpipeline]  # this can be later accessed
los.spectral_rays = 1
los.spectral_bins = 2500
los.ray_extinction_prob = 0.01
los.min_wavelength = 0.25
los.max_wavelength = 1240
los.pixel_samples = 3e4

###################################################
# Produce a radiating, cylindrical plasma with Neon
###################################################

# atomic data source
adas = OpenADAS(permit_extrapolation=True)

# PLASMA ##########################################
sigma = 0.25
plasma = Plasma(parent=world)
plasma.atomic_data = adas
plasma.geometry = Cylinder(radius=MAJOR_RADIUS + MINOR_RADIUS, height=CYLINDER_HEIGHT,
                           transform=translate(0, 0, CENTRE_Z - CYLINDER_HEIGHT / 2))
plasma.geometry_transform = None
plasma.integrator = NumericalIntegrator(step=sigma / 5.0)

# define basic distributions
bulk_velocity = Vector3D(0, 0, 0)

deuterium_mass = deuterium.atomic_weight * atomic_mass
neon_mass = neon.atomic_weight * atomic_mass

temperature = CustomFunction(do_minor_radius, do_temperature[ti, :])

electron_density = CustomFunction(do_minor_radius, do_n_free[ti, :])

fd0_density = CustomFunction(do_minor_radius, d0_density[ti, :] + d0i_density[ti, :])
fd1_density = CustomFunction(do_minor_radius, d1_density[ti, :] + d1i_density[ti, :])

fne0_d = CustomFunction(do_minor_radius, ne_densities[ti, 0, :])
fne1_d = CustomFunction(do_minor_radius, ne_densities[ti, 1, :])
fne2_d = CustomFunction(do_minor_radius, ne_densities[ti, 2, :])
fne3_d = CustomFunction(do_minor_radius, ne_densities[ti, 3, :])
fne4_d = CustomFunction(do_minor_radius, ne_densities[ti, 4, :])
fne5_d = CustomFunction(do_minor_radius, ne_densities[ti, 5, :])
fne6_d = CustomFunction(do_minor_radius, ne_densities[ti, 6, :])
fne7_d = CustomFunction(do_minor_radius, ne_densities[ti, 7, :])
fne8_d = CustomFunction(do_minor_radius, ne_densities[ti, 8, :])
fne9_d = CustomFunction(do_minor_radius, ne_densities[ti, 9, :])
fne10_d = CustomFunction(do_minor_radius, ne_densities[ti, 10, :])

e_distribution = Maxwellian(electron_density, temperature, bulk_velocity, electron_mass)

d0_distribution = Maxwellian(fd0_density, temperature, bulk_velocity, deuterium_mass)
d1_distribution = Maxwellian(fd1_density, temperature, bulk_velocity, deuterium_mass)

ne0_distribution = Maxwellian(fne0_d, temperature, bulk_velocity, neon_mass)
ne1_distribution = Maxwellian(fne1_d, temperature, bulk_velocity, neon_mass)
ne2_distribution = Maxwellian(fne2_d, temperature, bulk_velocity, neon_mass)
ne3_distribution = Maxwellian(fne3_d, temperature, bulk_velocity, neon_mass)
ne4_distribution = Maxwellian(fne4_d, temperature, bulk_velocity, neon_mass)
ne5_distribution = Maxwellian(fne5_d, temperature, bulk_velocity, neon_mass)
ne6_distribution = Maxwellian(fne6_d, temperature, bulk_velocity, neon_mass)
ne7_distribution = Maxwellian(fne7_d, temperature, bulk_velocity, neon_mass)
ne8_distribution = Maxwellian(fne8_d, temperature, bulk_velocity, neon_mass)
ne9_distribution = Maxwellian(fne9_d, temperature, bulk_velocity, neon_mass)
ne10_distribution = Maxwellian(fne10_d, temperature, bulk_velocity, neon_mass)

d0_s = Species(deuterium, 0, d0_distribution)
d1_s = Species(deuterium, 1, d1_distribution)

ne0_s = Species(neon, 0, ne0_distribution)
ne1_s = Species(neon, 1, ne1_distribution)
ne2_s = Species(neon, 2, ne2_distribution)
ne3_s = Species(neon, 3, ne3_distribution)
ne4_s = Species(neon, 4, ne4_distribution)
ne5_s = Species(neon, 5, ne5_distribution)
ne6_s = Species(neon, 6, ne6_distribution)
ne7_s = Species(neon, 7, ne7_distribution)
ne8_s = Species(neon, 8, ne8_distribution)
ne9_s = Species(neon, 9, ne9_distribution)
ne10_s = Species(neon, 10, ne10_distribution)

# define species
plasma.b_field = Vector3D(1.0, 1.0, 1.0)  # not sure about this
plasma.electron_distribution = e_distribution  # this is the free electron distribution
plasma.composition = [d0_s, d1_s, ne0_s, ne1_s, ne2_s, ne3_s, ne4_s, ne5_s, ne6_s, ne7_s, ne8_s, ne9_s, ne10_s]

# setup two Balmer lines, need more? probably not, but could add more later
hydrogen_I_410 = Line(deuterium, 0, (6, 2))
hydrogen_I_396 = Line(deuterium, 0, (7, 2))

########################################################################
# Get Neon lines from Photon Emissivity Coefficients datafiles
# This "ne" folder would normally be located at ~/.cherab/openadas/repository/pec/exctitaion/ne 
# or at .../pec/recombination/ne The JSON files are the same in both directories
########################################################################
neon_lines = []
for i in range(10):
    with open("ne/" + str(i) + ".json") as f:
        data = json.load(f)
        keys = data.keys()
        for key in keys:
            splitkey = key.split(" -> ")
            if " " not in splitkey[0]:
                part1 = int(splitkey[0])
                part2 = int(splitkey[1])
            else:
                part1 = splitkey[0]
                part2 = splitkey[1]

            neon_lines.append(ExcitationLine(Line(neon, i, (part1, part2)), lineshape=GaussianLine))
            neon_lines.append(RecombinationLine(Line(neon, i, (part1, part2)), lineshape=GaussianLine))

# add all lines to the plasma
plasma.models = [
    ExcitationLine(hydrogen_I_410, lineshape=GaussianLine),
    RecombinationLine(hydrogen_I_410, lineshape=GaussianLine),
    ExcitationLine(hydrogen_I_396, lineshape=GaussianLine),
    RecombinationLine(hydrogen_I_396, lineshape=GaussianLine),
    *neon_lines,
    Bremsstrahlung()
]

########################################################################
# Observe the radiation spectrum along the LOS
########################################################################

los.observe()

photon_energies = 1239.8 / sppipeline.wavelengths
datatosave = np.array([sppipeline.samples.mean, 
                       sppipeline.samples.variance, 
                       sppipeline.wavelengths,
                       photon_energies])
np.save("saved_data_new/highres_{:.4f}s.npy".format(do_time[ti]), datatosave)

# # Optionally dump the spectrum objects with pickle
# spectrum = sradpipeline.to_spectrum()
# filehandler = open("saved_data_new/highres_{:.4f}s_spectrum.obj".format(do_time[ti]), 'wb') 
# pickle.dump(spectrum, filehandler)
print("Saved array and dumped object")

