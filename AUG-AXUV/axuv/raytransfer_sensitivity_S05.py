# This script is used for calculating sensitivity matrices for all AXUV didoes in DHC and DVC
# Sector 5 of AUG

import os
import h5py
import shapely
import numpy as np

from scipy.spatial import ConvexHull

from raysect.core import translate
from cherab.tools.raytransfer import RayTransferCylinder, RayTransferPipeline0D

from axuv.interpolation import interpolate_parameters


if __name__ == "__main__":
    timestep = "02410"
    # loading JOREK output data
    print(timestep)
    fname = "step" + timestep + "_out.h5"
    filepath = "data/" + fname
    with h5py.File(filepath, "r") as f:
        majorR = f["R"][()][108:, np.newaxis]
        zaxis = f["Z"][()][108:, np.newaxis]

    neon0 = f["Ne0"][()][108:, np.newaxis]
    neon1 = f["Ne1"][()][108:, np.newaxis]
    neon2 = f["Ne2"][()][108:, np.newaxis]
    neon3 = f["Ne3"][()][108:, np.newaxis]
    neon4 = f["Ne4"][()][108:, np.newaxis]
    neon5 = f["Ne5"][()][108:, np.newaxis]
    neon6 = f["Ne6"][()][108:, np.newaxis]
    neon7 = f["Ne7"][()][108:, np.newaxis]
    neon8 = f["Ne8"][()][108:, np.newaxis]
    neon9 = f["Ne9"][()][108:, np.newaxis]
    neon10 = f["Ne10"][()][108:, np.newaxis]
    
    eTemp = f["Te"][()][108:, np.newaxis]
    eDens = f["ne"][()][108:, np.newaxis]

    neonlist = [neon0, neon1, neon2, neon3, neon4, neon5, neon6, neon7, neon8, neon9, neon10]
    
    # Setting up interpolation of JOREK data
    # In this case the vertical and horizontal distances between the gridpoints will be the same
    # Later the voxel grid will have the same dimensions, but will be masked where there is no plasma
    # this results in ~4 GB memory allocation for the creation of the ~4000 element voxel grid 
    # NOTE Doubling the total number of grid points results in a 2^2=4 times increase in the memory needed!
    # NOTE Doubling the resolution in both directions results in a (2*2)^2=16 times increase!
    resolution_R = 60
    resolution_z = 110
    
    plasma_res_R: int=int(resolution_R)
    plasma_res_z: int=int(resolution_z)
    
    # The points at which the JOREK data is defined
    points = np.hstack([majorR, zaxis])
    
    i_neon, i_eTemp, i_eDens = interpolate_parameters(points, plasma_res_R, plasma_res_z, neonlist,
                                                        eTemp, eDens, method="linear")
    
    # A convex hull is created around the JOREK datapoints to be used as boundary for the voxel grid later
    hull = ConvexHull(points)
    convex_hull = points[hull.vertices]
    polygon_minimum = shapely.geometry.Polygon(convex_hull)
    polygon = polygon_minimum.buffer(0.1, join_style=2)  # make the polygon a bit bigger (by 10%)
    
    # Dummy world for the voxels
    world, cameras = create_observable_world_S5(cad_mesh=USE_CAD_MESH, show_plots=False)
    
    
    ########################################################################
    # Produce a voxel grid
    ########################################################################
    print("Producing the voxel grid...")
    # Define the centres of each voxel, as an (nx, ny, 2) array
    nx = resolution_R
    ny = resolution_z
    cell_r, cell_dx = np.linspace(POLOIDAL_RMIN, POLOIDAL_RMAX, nx, retstep=True)
    cell_z, cell_dz = np.linspace(POLOIDAL_ZMIN, POLOIDAL_ZMAX, ny, retstep=True)
    cell_r_grid, cell_z_grid = np.broadcast_arrays(cell_r[:, None], cell_z[None, :])
    cell_centres = np.stack((cell_r_grid, cell_z_grid), axis=-1)  # (nx, ny, 2) array
    
    # Define the positions of the vertices of the voxels
    cell_vertices_r = np.linspace(cell_r[0] - 0.5 * cell_dx, cell_r[-1] + 0.5 * cell_dx, nx + 1)
    cell_vertices_z = np.linspace(cell_z[0] - 0.5 * cell_dz, cell_z[-1] + 0.5 * cell_dz, ny + 1)
    
    # Build a mask, only including cells within the wall
    # The inversions will be performed on the emission profile used in the
    # radiation_function.py demo, so we'll trim the voxel grid down to the
    # emitting region using the shapely polygon
    
    grid_mask = np.empty(shape=(nx, ny), dtype=bool)
    for ix in range(nx):
        for iy in range(ny):
            point1 = shapely.geometry.Point([cell_vertices_r[ix], cell_vertices_z[iy]])
            point2 = shapely.geometry.Point([cell_vertices_r[ix+1], cell_vertices_z[iy]])
            point3 = shapely.geometry.Point([cell_vertices_r[ix], cell_vertices_z[iy+1]])
            point4 = shapely.geometry.Point([cell_vertices_r[ix+1], cell_vertices_z[iy+1]])
            if polygon.contains(point1) or polygon.contains(point2) or polygon.contains(point3) or polygon.contains(point4):
                grid_mask[ix, iy] = True
            else:
                grid_mask[ix, iy] = False
    
    # The RayTransferCylinder object is fully 3D, but for simplicity we're only
    # working in 2D as this case is axisymmetric. It is easy enough to pass 3D
    # views of our 2D data into the RayTransferCylinder object: we just ues a
    # numpy.newaxis (or equivalently, None) for the toroidal dimension.
    grid_mask = grid_mask[:, None, :]
    
    num_cells = grid_mask.sum()
    
    ray_transfer_grid = RayTransferCylinder(
        radius_outer=cell_vertices_r[-1],
        radius_inner=cell_vertices_r[0],
        height=cell_vertices_z[-1] - cell_vertices_z[0],
        n_radius=nx, n_height=ny, mask=grid_mask, n_polar=1,
        transform=translate(0, 0, cell_vertices_z[0])
    )
    
    ########################################################################
    # Produce a regularisation operator for inversions
    ########################################################################
    # We'll use simple isotropic smoothing here, in which case an ND second
    # derivative operator (the laplacian operator) is appropriate. This can be
    # produced in the same way as in the geometry matrix with voxels demo, but we
    # show a faster vectorised method here.
    
    # Pad the voxel map with a 1-cell-wide border.
    voxel_map_with_borders = - np.ones((nx + 2, ny + 2), dtype=int)
    voxel_map_with_borders[1:-1, 1:-1] = ray_transfer_grid.voxel_map[:, 0, :]
    inverted_voxel_map = ray_transfer_grid.invert_voxel_map()
    grid_laplacian = np.zeros((num_cells, num_cells))
    
    
    for ith_cell in range(num_cells):
        # get the 2D mesh coordinates of this cell
        ix, _, iy = inverted_voxel_map[ith_cell]
        # we didn't map multiple cells into the same light source,
        # so ix and iy are single-element arrays
        ix = ix[0]
        iy = iy[0]
    
        neighbours_2d = ([ix, ix, ix, ix + 1, ix + 1, ix + 2, ix + 2, ix + 2],
                        [iy, iy + 1, iy + 2, iy, iy + 2, iy, iy + 1, iy + 2])
    
        neighbours_1d = voxel_map_with_borders[neighbours_2d]
        neighbours_1d = neighbours_1d[neighbours_1d > -1]
    
        grid_laplacian[ith_cell, neighbours_1d] = -1
        grid_laplacian[ith_cell, ith_cell] = neighbours_1d.size
    
    
    ########################################################################
    # Calculate the geometry matrix for the grid
    ########################################################################
    print("Calculating the geometry matrix...")
    # The ray transfer object must be in the same world as the bolometers
    ray_transfer_grid.parent = world
    
    NUM_OF_DIODES = 96
    
    def get_spectrum_part(part):
        SPECTRAL_BINS = 100
                                            # Approx photon energies in eV
        MIN_WAVELENGTHS = [0.25, 12.4, 124]   # 5000, 100, 10
        MAX_WAVELENGTHS = [12.4, 124, 1240]   # 100, 10, 1
    
        return np.linspace(MIN_WAVELENGTHS[part], MAX_WAVELENGTHS[part], SPECTRAL_BINS)
    
    wavelengths = np.unique(np.array([*get_spectrum_part(0),*get_spectrum_part(1),*get_spectrum_part(2)]))
    energies_eV = 1239.8 / wavelengths
    total_wavelength_bins = len(wavelengths) - 1
    
    HDF5_PATH = "raytransfer_S05_reflections_lowres.h5"
    
    # === One-time file setup ===
    if not os.path.exists(HDF5_PATH):
        with h5py.File(HDF5_PATH, 'w') as h5f:
            h5f.create_dataset("sensitivity_matrix", shape=(NUM_OF_DIODES, num_cells, total_wavelength_bins), dtype='f8')
            h5f.create_dataset("grid_centres", data=cell_centres)
            h5f.create_dataset("voxel_map", data=ray_transfer_grid.voxel_map)
            h5f.create_dataset("inverse_voxel_map", data=ray_transfer_grid.invert_voxel_map())
            h5f.create_dataset("laplacian", data=grid_laplacian)
            h5f.create_dataset("mask", data=ray_transfer_grid.mask)
            h5f.create_dataset("completed_bins", shape=(total_wavelength_bins,), dtype='i1')  # 0 = not done, 1 = done
            h5f.create_dataset("wavelength_bin_edges", data=wavelengths)
            h5f.create_dataset("energy_bin_edges_eV", data=energies_eV)
    
    # === Main computation loop ===
    sensitivity_matrix = np.zeros([NUM_OF_DIODES, num_cells, total_wavelength_bins])
    
    for j in range(total_wavelength_bins):
        # Check if bin j is already completed
        with h5py.File(HDF5_PATH, 'r') as h5f:
            if h5f["completed_bins"][j]:
                print(f"Skipping wavelength bin {j+1} (already completed)")
                continue
    
        print(f"Calculating for wavelength bin {j+1}/{total_wavelength_bins}")
        i = 0
        for camera in cameras:
            for foil in camera:
                print(f"{j+1}/{total_wavelength_bins} Calculating sensitivity for {foil.name}...", end="\n")
                foil.pipelines = [RayTransferPipeline0D(kind=foil.units)]
                foil.max_wavelength = 2000  # just to be on the safe side, but this will be overwritten
                foil.min_wavelength = wavelengths[j]
                foil.max_wavelength = wavelengths[j+1]
                foil.spectral_bins = ray_transfer_grid.bins
                foil.spectral_rays = 1
                foil.pixel_samples = 1e6
                foil.ray_max_depth = 10
                foil.observe()
                sensitivity_matrix[i, :, j] = foil.pipelines[0].matrix
                i += 1
                foil.max_wavelength = wavelengths[j+1] + 10
    
        # Open file just to save results for this bin
        with h5py.File(HDF5_PATH, 'r+') as h5f:
            h5f["sensitivity_matrix"][:, :, j] = sensitivity_matrix[:, :, j]
            h5f["completed_bins"][j] = 1
            h5f.flush()


