
import matplotlib.patches as patches



import torch
from torch.utils.data import DataLoader
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(128)
import torch.nn as nn
import os
import configparser
import initial_conditions
import shutil
import numpy as np
from cached_property import cached_property
from devito import *
import matplotlib.pyplot as plt
from matplotlib import cm
from sympy import pprint
import torch
from devito.builtins import initialize_function
import PINNs
import FD_devito
import sys
import time as timer
import mixture_model
import pickle
import pandas as pd
import random
import matplotlib.gridspec as gridspec
from scipy.ndimage import zoom


def interpolate_solution(coarse, fine_shape):
    """
    Interpolates the coarse solution array to match the shape of the fine solution.

    Parameters:
    - coarse: numpy array, the coarser solution to be interpolated.
    - fine_shape: tuple, the shape of the finer resolution solution.

    Returns:
    - interpolated: numpy array, the interpolated solution with the shape of the fine solution.
    """
    # Calculate the zoom factors for each dimension
    zoom_factors = [n / c for n, c in zip(fine_shape, coarse.shape)]

    # Interpolate using the calculated zoom factors
    interpolated = zoom(coarse, zoom_factors, order=1)  # Linear interpolation
    return interpolated


def relative_l2_error(true, pred):
    # Ensure true and pred are numpy arrays for correct operations
    true = np.array(true)
    pred = np.array(pred)

    # Calculate the L2 norm of the error
    error_norm = np.linalg.norm(pred - true)

    # Calculate the L2 norm of the true (observed) values
    true_norm = np.linalg.norm(true)

    # Compute the Relative L2 Error
    relative_error = error_norm / true_norm

    return relative_error

plt.rcParams['text.usetex'] = True

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.manual_seed(42)





#Changable parameters:




def get_devito_solution(numpoints_sqrt,n_different_sources):
    result_list = []

    # Number of points along x and y axis, total number of points will be numpoints_sqrt**2
    # Number of points along x and y axis, total number of points will be numpoints_sqrt**2
    dt = 1.095e-3

    enlarge_factor = 3
    pinn_domain_extent_x = [-1.0, 1.0]
    pinn_domain_extent_y = [-1.0, 1.0]

    FD_prep_0 = timer.time()

    pinn_length_x = pinn_domain_extent_x[1] - pinn_domain_extent_x[0]
    pinn_length_y = pinn_domain_extent_y[1] - pinn_domain_extent_y[0]

    devito_length_x = enlarge_factor * pinn_length_x
    devito_length_y = enlarge_factor * pinn_length_y
    extent = (devito_length_x, devito_length_y)
    devito_center = [devito_length_x / 2.0, devito_length_y / 2.0]
    shape = (numpoints_sqrt * enlarge_factor, numpoints_sqrt * enlarge_factor)
    x = SpaceDimension(name='x', spacing=Constant(name='h_x', value=extent[0] / (shape[0] - 1)))
    y = SpaceDimension(name='y', spacing=Constant(name='h_y', value=extent[1] / (shape[1] - 1)))
    grid = Grid(extent=extent, shape=shape, dimensions=(x, y))
    spacing = (extent[0] / shape[0], extent[0] / shape[0])
    X, Y = np.meshgrid(np.linspace(0, numpoints_sqrt * spacing[0], numpoints_sqrt),
                       np.linspace(0, numpoints_sqrt * spacing[1], numpoints_sqrt))

    FD_prep_1 = timer.time()

    # Generate mixtures for mu and lambda
    mu_mixture = FD_devito.generate_mixture().numpy()
    lambda_mixture = FD_devito.generate_mixture().numpy()
    lambda_solid = FD_devito.compute_param_np(X, Y, lambda_mixture)
    mu_solid = FD_devito.compute_param_np(X, Y, mu_mixture)

    # Create a larger array (7x7)
    X_large, Y_large = numpoints_sqrt * enlarge_factor, numpoints_sqrt * enlarge_factor
    lambda_large = np.full((X_large, Y_large), 0.0)
    # Calculate the position to place the smaller array at the center of the larger array
    x_offset = (X_large - numpoints_sqrt) // 2
    y_offset = (Y_large - numpoints_sqrt) // 2

    # Place the smaller array in the center of the larger array
    lambda_large[x_offset:x_offset + numpoints_sqrt, y_offset:y_offset + numpoints_sqrt] = lambda_solid
    lambda_large[0:x_offset, :] = lambda_large[x_offset, :]
    lambda_large[x_offset + numpoints_sqrt:, :] = lambda_large[x_offset + (numpoints_sqrt - 1), :]
    lambda_large[:, 0:y_offset] = lambda_large[:, y_offset][:, np.newaxis]
    lambda_large[:, y_offset + numpoints_sqrt:] = lambda_large[:, y_offset + (numpoints_sqrt - 1)][:,
                                                  np.newaxis]

    lambda_solid = lambda_large

    Xp, Yp = np.meshgrid(
        np.linspace(0, (numpoints_sqrt * enlarge_factor) * spacing[0], numpoints_sqrt * enlarge_factor),
        np.linspace(0, (numpoints_sqrt * enlarge_factor) * spacing[1], numpoints_sqrt * enlarge_factor))

    mu_large = np.full((X_large, Y_large), 0.0)
    mu_large[x_offset:x_offset + numpoints_sqrt, y_offset:y_offset + numpoints_sqrt] = mu_solid
    mu_large[0:x_offset, :] = mu_large[x_offset, :]
    mu_large[x_offset + numpoints_sqrt:, :] = mu_large[x_offset + (numpoints_sqrt - 1), :]
    mu_large[:, 0:y_offset] = mu_large[:, y_offset][:, np.newaxis]
    mu_large[:, y_offset + numpoints_sqrt:] = mu_large[:, y_offset + (numpoints_sqrt - 1)][:, np.newaxis]
    mu_solid = mu_large

    # Using both ux and uy as separate field for easier comparison to PINNs and easier pde formulation
    ux_devito = TimeFunction(name='ux_devito', grid=grid, space_order=4, time_order=2)
    uy_devito = TimeFunction(name='uy_devito', grid=grid, space_order=4, time_order=2)

    lambda_ = Function(name='lambda_f', grid=grid, space_order=4)
    mu_ = Function(name='mu_f', grid=grid, space_order=4)
    initialize_function(lambda_, lambda_solid, nbl=0)
    initialize_function(mu_, mu_solid, nbl=0)

    rho_solid = 100.0

    div_stress_ux = (
                            lambda_ + 2.0 * mu_) * ux_devito.dx2 + mu_ * ux_devito.dy2 + lambda_ * uy_devito.dy.dx + mu_ * uy_devito.dx.dy
    div_stress_uy = (
                            lambda_ + 2.0 * mu_) * uy_devito.dy2 + mu_ * uy_devito.dx2 + lambda_ * ux_devito.dx.dy + mu_ * ux_devito.dy.dx
    pde_x = rho_solid * ux_devito.dt2 - div_stress_ux
    pde_y = rho_solid * uy_devito.dt2 - div_stress_uy
    # BOundary conditions:
    x, y = grid.dimensions
    t = grid.stepping_dim
    ny, nx = shape[0], shape[1]
    bc = [Eq(ux_devito[t + 1, x, 0], 0.)]
    bc += [Eq(ux_devito[t + 1, x, ny - 1], 0.)]
    bc += [Eq(ux_devito[t + 1, 0, y], 0.)]
    bc += [Eq(ux_devito[t + 1, nx - 1, y], 0.)]

    bc += [Eq(ux_devito[t + 1, x, 1], 0.)]
    bc += [Eq(ux_devito[t + 1, x, ny - 2], 0.)]
    bc += [Eq(ux_devito[t + 1, 1, y], 0.)]
    bc += [Eq(ux_devito[t + 1, nx - 2, y], 0.)]

    bc += [Eq(uy_devito[t + 1, x, 0], 0.)]
    bc += [Eq(uy_devito[t + 1, x, ny - 1], 0.)]
    bc += [Eq(uy_devito[t + 1, 0, y], 0.)]
    bc += [Eq(uy_devito[t + 1, nx - 1, y], 0.)]

    bc += [Eq(uy_devito[t + 1, x, 1], 0.)]
    bc += [Eq(uy_devito[t + 1, x, ny - 2], 0.)]
    bc += [Eq(uy_devito[t + 1, 1, y], 0.)]
    bc += [Eq(uy_devito[t + 1, nx - 2, y], 0.)]

    # Formulating stencil to solve for u forward
    stencil_x = Eq(ux_devito.forward, solve(pde_x, ux_devito.forward))
    stencil_y = Eq(uy_devito.forward, solve(pde_y, uy_devito.forward))
    op = Operator([stencil_x] + [stencil_y] + bc)

    n = 102
    dt = 1.096e-3
    time_list = np.linspace(0, 1, n).tolist()
    # Determine the grid size
    grid_side = int(np.sqrt(n_different_sources))  # Calculate side length of the square grid
    actual_n_sources = grid_side ** 2  # Actual number of sources used

    # Generate grid points
    x_grid = np.linspace(-0.45, -0.3, grid_side)
    y_grid = np.linspace(-0.4, 0.3, grid_side)

    t0 = timer.time()
    for mu_quake_x in np.linspace(-0.45, -0.3, grid_side):
        for mu_quake_y in np.linspace(-0.4, 0.3, grid_side):

            mu_quake_devito = [devito_center[0] + mu_quake_x, devito_center[1] + mu_quake_y]

            u0x_devito, u0y_devito = FD_devito.initial_condition_double_gaussian_derivative(
                sigma=float(0.1), mu_quake=mu_quake_devito, nx=shape[0], ny=shape[1],
                dx=spacing[0], dy=spacing[1])

            ux_devito.data[0] = u0x_devito
            uy_devito.data[0] = u0y_devito
            ux_devito.data[1] = u0x_devito
            uy_devito.data[1] = u0y_devito

            index = 0
            FD_prep_1 = timer.time()

            res_list_devito_x = []
            res_list_devito_y = []
            res_list_devito_u = []

            for i in time_list:
                res_list_devito_x.append(np.transpose(
                    ux_devito.data[1][x_offset:x_offset + numpoints_sqrt, y_offset:y_offset + numpoints_sqrt]).copy())
                res_list_devito_y.append(np.transpose(
                    uy_devito.data[1][x_offset:x_offset + numpoints_sqrt, y_offset:y_offset + numpoints_sqrt]).copy())
                res_list_devito_u.append(np.sqrt(np.transpose(uy_devito.data[1][x_offset:x_offset + numpoints_sqrt,
                                                              y_offset:y_offset + numpoints_sqrt]).copy() ** 2 + np.transpose(
                    ux_devito.data[1][x_offset:x_offset + numpoints_sqrt,
                    y_offset:y_offset + numpoints_sqrt]).copy() ** 2))
                op(time_M=10, dt=dt)
            result_list.append(res_list_devito_u)

    return result_list


solution_64 = get_devito_solution(64,10)
#solution_256 = get_devito_solution(256,1)
solution_512 = get_devito_solution(512,10)
diff = 0.0



# Assuming solution_64 and solution_512 are available
# Interpolate solution_64 to match the grid of solution_512
solution_64_interp = interpolate_solution(np.array(solution_64), np.array(solution_512).shape)
#solution_256_interp = interpolate_solution(np.array(solution_256), np.array(solution_512).shape)


# Now you can compute the relative L2 error between solution_512 and the interpolated solution_64
diff = 0.0
for i in range(len(solution_512)):
    current_relative_error_64 = relative_l2_error(solution_512[i], solution_64_interp[i])
    #current_relative_error_256 = relative_l2_error(solution_512[i], solution_256_interp[i])

    print("relative_error for 64 solution = ", 100*current_relative_error_64,"%")
   # print("relative_error for 256 solution = ", 100 * current_relative_error_256, "%")