
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

plt.rcParams['text.usetex'] = True

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.manual_seed(42)
config = configparser.ConfigParser()



#Changable parameters:

#Number of points along x and y axis, total number of points will be numpoints_sqrt**2
numpoints_sqrt = 512
dt=1.095e-3
#mu_quake_x = 0.11
#mu_quake_y = -0.134


# Define the range for random numbers
min_val, max_val = -0.45, 0.45

# Generate random numbers for each list
#mu_quake_x_list = [-0.44,-0.33,-0.22,-0.11,-0.01,0.01,0.11,0.22,0.33,0.44]  # 10 random numbers for mu_quake_x
#mu_quake_y_list = [-0.44,-0.33,-0.22,-0.11,-0.01,0.01,0.11,0.22,0.33,0.44]  # 10 random numbers for mu_quake_y
mu_quake_x_list = [0.0]
mu_quake_y_list = [0.0]
my_devito_time = 0.0

#mu_quake_x_list, mu_quake_y_list

#increase Devito domain to avoid Boundary issues (reflections)
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


model_type = 'constant'


if model_type == "mixture":
    # Generate mixtures for mu and lambda
    mu_mixture = FD_devito.generate_mixture().numpy()
    lambda_mixture = FD_devito.generate_mixture().numpy()
    lambda_solid = FD_devito.compute_param_np(X, Y, lambda_mixture)
    mu_solid = FD_devito.compute_param_np(X, Y, mu_mixture)
elif model_type == "constant":
    lambda_solid = np.full(X.shape, 20.0)
    mu_solid = np.full(X.shape, 30.0)
elif model_type == 'layered':
    lambda_solid, mu_solid = mixture_model.compute_lambda_mu_layers_accoustic_slow(torch.tensor(X), torch.tensor(Y), 5)
    lambda_solid = lambda_solid.numpy()
    mu_solid = mu_solid.numpy()
elif model_type == 'layered_sine' or model_type == 'layered_sine_simpler_3':
    lambda_solid, mu_solid = mixture_model.create_sine_layer_map(torch.tensor(X), torch.tensor(Y))
else:
    lambda_solid = np.full(X.shape, 20.0)
    mu_solid = np.full(X.shape, 30.0)
    # raise Exception("model type {} not implemented".format(model_type))

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
plt.scatter(Xp, Yp, c=lambda_solid)
plt.show()
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
print(mu_.data, mu_.data.shape)
plt.scatter(Xp, Yp, c=lambda_.data)
plt.show()
plt.scatter(Xp, Yp, c=mu_.data)
plt.show()

rho_solid = 100.0

mu_quake_devito = [devito_center[0] + 0.0, devito_center[1] + 0.0]

# The initial condition that also was used for The PINN, mimics a perfectly isotropic explosion
u0x_devito, u0y_devito = FD_devito.initial_condition_double_gaussian_derivative(
    sigma=0.1, mu_quake=mu_quake_devito, nx=shape[0], ny=shape[1],
    dx=spacing[0], dy=spacing[1])

ux_devito.data[0] = u0x_devito
uy_devito.data[0] = u0y_devito
ux_devito.data[1] = u0x_devito
uy_devito.data[1] = u0y_devito

FD_devito.plot_field(ux_devito.data[0])
FD_devito.plot_field(uy_devito.data[0])
# Plot from initial field look good


tm0 = timer.time()

print("start stress calculation")
# Divergence of stress tensor for ux
div_stress_ux = (
                        lambda_ + 2.0 * mu_) * ux_devito.dx2 + mu_ * ux_devito.dy2 + lambda_ * uy_devito.dy.dx + mu_ * uy_devito.dx.dy

# Divergence of stress tensor for uy
div_stress_uy = (
                        lambda_ + 2.0 * mu_) * uy_devito.dy2 + mu_ * uy_devito.dx2 + lambda_ * ux_devito.dx.dy + mu_ * ux_devito.dy.dx

print("stress calculated")
# Elastic wave equation for ux and uy
# print("model damp = ",model.damp,model.damp.data)
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
pprint(stencil_x)
pprint(stencil_y)
op = Operator([stencil_x] + [stencil_y] + bc)
tm1 = timer.time()
my_devito_time = my_devito_time + (tm1 - tm0)

print("start")
res_list_devito = []

index = 0
print("start")
FD_prep_1 = timer.time()

# res_list_devito = []
devito_time = 0.0

inferece_time = 0.0
FD_time = 0.0

grid_x, grid_y = torch.meshgrid(torch.linspace(-1.0, 1.0, numpoints_sqrt),
                                torch.linspace(-1.0, 1.0, numpoints_sqrt))
grid_x = torch.reshape(grid_x, (-1,))
grid_y = torch.reshape(grid_y, (-1,))

res_list_ux = []
res_list_uy = []
res_list_u = []
res_list_devito_x = []
res_list_devito_y = []
res_list_devito_u = []
n = 102
dt = 1.096e-3

time_list = np.linspace(0, 1, n).tolist()

for i in time_list:
    tm0 = timer.time()
    FD_t1 = timer.time()

    res_list_devito_x.append(np.transpose(
        ux_devito.data[1][x_offset:x_offset + numpoints_sqrt, y_offset:y_offset + numpoints_sqrt]).copy())
    res_list_devito_y.append(np.transpose(
        uy_devito.data[1][x_offset:x_offset + numpoints_sqrt, y_offset:y_offset + numpoints_sqrt]).copy())
    res_list_devito_u.append(np.sqrt(np.transpose(uy_devito.data[1][x_offset:x_offset + numpoints_sqrt,
                                                  y_offset:y_offset + numpoints_sqrt]).copy() ** 2 + np.transpose(
        ux_devito.data[1][x_offset:x_offset + numpoints_sqrt,
        y_offset:y_offset + numpoints_sqrt]).copy() ** 2))

    op(time_M=10, dt=dt)
    tm1 = timer.time()
    my_devito_time = my_devito_time + (tm1 - tm0)
    print(my_devito_time)
    FD_t2 = timer.time()
    FD_time = FD_time + (FD_t2 - FD_t1)
