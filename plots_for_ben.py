
pth1 = 'PINN_new_source_100000_200_1.0_100_100_100_5_128_tanh_0.1_PINN_1_constant_0.1'
pth2 = 'PINN_new_source_100000_200_1.0_100_100_100_4_32_tanh_0.1_FCN_ALL_PARAMS_WAVELET_FCN_128_4_64_constant_0.1'

pth1 = 'PINN_layered_and_sigma_300000_400_2.0_200_200_100_5_115_tanh_0.2_PINNS_1_layered_sine_0.06'
pth2 = 'PINN_layered_and_sigma_300000_400_2.0_200_200_100_4_50_tanh_0.2_FCN_ALL_PARAMS_PLANEWAVE_FCN_80_3_28_layered_sine_0.06'

pth1 = 'PINN_mixture_thesis_100000_200_1.0_100_100_100_5_128_tanh_0.1_PINN_1_constant_0.1'
pth2 = 'PINN_mixture_thesis_100000_200_1.0_100_100_100_4_32_tanh_0.07_FCN_ALL_PARAMS_WAVELET_FCN_128_4_64_constant_0.1'

pth1 = 'PINN_mixture_thesis_100000_200_1.0_100_100_100_6_128_tanh_0.2_PINN_1_mixture_0.1'
pth2 = 'PINN_mixture_thesis_100000_200_1.0_100_100_100_4_32_tanh_0.1_FCN_ALL_PARAMS_WAVELET_FCN_128_4_64_mixture_0.1'

pth1 = 'PINN_layered_thesis_100000_400_2.0_200_200_100_5_128_tanh_0.2_PINN_1_layered_sine_simpler_3_0.1'
pth2= 'PINN_layered_thesis_100000_400_2.0_200_200_100_4_32_tanh_0.07_FCN_ALL_PARAMS_PLANEWAVE_FCN_128_3_8_layered_sine_simpler_3_0.1'

pth1 = 'Global_FullDomain_Conditioned_Pinns_FCN_ALL_PARAMS_PLANEWAVE_FCN_500000_400_1.0_200_200_100_4_64_tanh_0.2_FCN_ALL_PARAMS_PLANEWAVE_FCN_135_3_16_mixture'
pth2 = 'Global_FullDomain_Conditioned_Pinns_PINN_500000_400_1.0_200_200_100_5_128_tanh_0.2_PINN_1_mixture'

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


#Changable parameters:

#Number of points along x and y axis, total number of points will be numpoints_sqrt**2
numpoints_sqrt = 512
dt=1.095e-3


# Generate random numbers for each list
#mu_quake_x_list = [-0.44,-0.33,-0.22,-0.11,-0.01,0.01,0.11,0.22,0.33,0.44]  # 10 random numbers for mu_quake_x
#mu_quake_y_list = [-0.44,-0.33,-0.22,-0.11,-0.01,0.01,0.11,0.22,0.33,0.44]  # 10 random numbers for mu_quake_y
mu_quake_x_list = [0.0]
mu_quake_y_list = [0.0]

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
config.read("../pre_trained_models/"+pth1+"/config.ini")

model_type = config['parameters']['model_type']


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
    lambda_solid = np.full(X.shape, config["parameters"]["lambda_solid"])
    mu_solid = np.full(X.shape, config["parameters"]["mu_solid"])
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


mu_quake = torch.tensor([-0.25,-0.4])

for test_model in ['Vanilla', 'PLANEWAVE']:

    if test_model == 'Vanilla':
        save_path = '../../Results/Elastic/mixture_thesis_test/[{}_{}]/Vanilla/'.format(str(0),
                                                                                        str(0))
        os.makedirs(save_path, exist_ok=True)
        input_folder = pth2
    else:
        save_path = '../../Results/Elastic/mixture_thesis_test/[{}_{}]/WAVELET/'.format(str(0),
                                                                                        str(0))
        os.makedirs(save_path, exist_ok=True)
        input_folder = pth1

    config.read("../pre_trained_models/" + input_folder + "/config.ini")
    t1 = float(config['initial_condition']['t1'])

    devito_prep_time = 0.0

    # Parameters
    rho_solid = float(config['parameters']['rho_solid'])

    model_path = "../pre_trained_models/" + input_folder + "/model.pth"
    print("model path = ", model_path)
    str_path = model_path.replace("../pre_trained_models/", "")
    model_type = config["parameters"]["model_type"]

    mu_quake_devito = [devito_center[0] + mu_quake[0].numpy(), devito_center[1] + mu_quake[1].numpy()]

    # The initial condition that also was used for The PINN, mimics a perfectly isotropic explosion
    u0x_devito, u0y_devito = FD_devito.initial_condition_double_gaussian_derivative(
        sigma=float(config["parameters"]["sigma_quake"]), mu_quake=mu_quake_devito, nx=shape[0], ny=shape[1],
        dx=spacing[0], dy=spacing[1])

    #u0y_devito = u0y_devito * 0.0

    ux_devito.data[0] = u0x_devito
    uy_devito.data[0] = u0y_devito
    ux_devito.data[1] = u0x_devito
    uy_devito.data[1] = u0y_devito

    FD_devito.plot_field(ux_devito.data[0])
    FD_devito.plot_field(uy_devito.data[0])
    # Plot from initial field look good
    print(ux_devito.shape, uy_devito.shape)

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

    print("start")
    res_list_devito = []

    index = 0
    print("start")
    FD_prep_1 = timer.time()
    devito_prep_time = devito_prep_time + (FD_prep_1 - FD_prep_0)

    print("Model type not specified, reading from config file instead")
    model_type = config['Network']['model_type']
    if model_type == "Pinns":
        pinn = PINNs.Pinns(int(config['Network']['n_points']), wandb_on=False, config=config)
    elif model_type == "Global_NSources_Conditioned_Pinns":
        pinn = PINNs.Global_NSources_Conditioned_Pinns(int(config['Network']['n_points']), wandb_on=False,
                                                       config=config)
    elif model_type == "Relative_Distance_NSources_Conditioned_Pinns":
        pinn = PINNs.Relative_Distance_NSources_Conditioned_Pinns(int(config['Network']['n_points']), wandb_on=False,
                                                                  config=config)
    elif model_type == "Relative_Distance_FullDomain_Conditioned_Pinns":
        pinn = PINNs.Relative_Distance_FullDomain_Conditioned_Pinns(int(config['Network']['n_points']), wandb_on=False,
                                                                    config=config)
    elif model_type == "Global_NSources_Conditioned_Lame_Pinns":
        pinn = PINNs.Global_NSources_Conditioned_Lame_Pinns(int(config['Network']['n_points']), wandb_on=False,
                                                            config=config)
    elif model_type == "Global_FullDomain_Conditioned_Pinns":
        pinn = PINNs.Global_FullDomain_Conditioned_Pinns(int(config['Network']['n_points']), wandb_on=False,
                                                         config=config)
    else:
        raise Exception("Model type {} not supported".format(model_type))

    tag = str_path.replace(".pth", "")
    tag = tag.replace("/model", "")

    dir = "../images/" + tag
    # Define the path to the directory
    dir_path = os.path.join(os.getcwd(), dir)
    print(dir_path)

    # Check if the directory already exists
    if os.path.exists(dir_path):
        # Remove the existing directory and all its subdirectories
        shutil.rmtree(dir_path)

    # Create the directory
    os.mkdir(dir_path)

    # Create the subdirectories
    subdirs = ['x', 'y', 'u']
    for subdir in subdirs:
        os.mkdir(os.path.join(dir_path, subdir))

    print(f"Directory '{tag}' created with subdirectories 'x', 'y', and 'u'.")

    n_neurons = int(config['Network']['n_neurons'])
    n_hidden_layers = int(config['Network']['n_hidden_layers'])
    print(n_neurons, n_hidden_layers)
    if config['Network']['activation'] == 'tanh':
        activation = nn.Tanh()
    elif config['Network']['activation'] == 'sin':
        activation = PINNs.SinActivation()
    elif config['Network']['activation'] == 'wavelet':
        activation = PINNs.WaveletActivation(int(config['Network']['n_neurons']))
    elif config['Network']['activation'] == 'wavelet2':
        activation = PINNs.WaveletActivation_2(int(config['Network']['n_neurons']))
    else:
        print("unknown activation function", config['Network']['activation'])
        exit()

    if config['Network']['nn_type'] == 'FCN_ALL_PARAMS_WAVELET_FCN':
        print("correct")
        my_network = PINNs.FCN_all_params_Wavelet_FCN(input_dimension=5, output_dimension=2,
                                                      n_hidden_layers=int(
                                                          config['Network']['n_hidden_layers']),
                                                      neurons=int(config['Network']['n_neurons']),
                                                      regularization_param=0.,
                                                      regularization_exp=2.,
                                                      retrain_seed=3, activation=activation,
                                                      n_1=int(config['Network']['n_1']),
                                                      n_hidden_layers_after=int(
                                                          config['Network']['n_hidden_layers_after']),
                                                      n_neurons_after=int(
                                                          config['Network']['n_neurons_after']))
    elif config['Network']['nn_type'] == 'FCN_ALL_PARAMS_PLANEWAVE_FCN':
        print("it's a plane")
        my_network = PINNs.FCN_all_params_Planewave_FCN(input_dimension=5, output_dimension=2,
                                                        n_hidden_layers=int(
                                                            config['Network']['n_hidden_layers']),
                                                        neurons=int(config['Network']['n_neurons']),
                                                        regularization_param=0.,
                                                        regularization_exp=2.,
                                                        retrain_seed=3, activation=activation,
                                                        n_1=int(config['Network']['n_1']),
                                                        n_hidden_layers_after=int(
                                                            config['Network']['n_hidden_layers_after']),
                                                        n_neurons_after=int(
                                                            config['Network']['n_neurons_after']))
    else:
        my_network = PINNs.NeuralNet(input_dimension=5, output_dimension=2,
                                     n_hidden_layers=int(config['Network']['n_hidden_layers']),
                                     neurons=int(config['Network']['n_neurons']),
                                     regularization_param=0.,
                                     regularization_exp=2.,
                                     retrain_seed=3, activation=activation)

    print(model_path)
    my_network.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    pinn.approximate_solution = my_network

    # res_list_devito = []
    devito_time = 0.0

    inferece_time = 0.0
    FD_time = 0.0

    soboleng = torch.quasirandom.SobolEngine(dimension=pinn.domain_extrema.shape[0])
    inputs = soboleng.draw(int(pow(numpoints_sqrt, 2)))

    grid_x, grid_y = torch.meshgrid(torch.linspace(-1.0, 1.0, numpoints_sqrt),
                                    torch.linspace(-1.0, 1.0, numpoints_sqrt))
    grid_x = torch.reshape(grid_x, (-1,))
    grid_y = torch.reshape(grid_y, (-1,))

    inputs[:, 1] = grid_x
    inputs[:, 2] = grid_y
    inputs[:, 3] = mu_quake[0]
    inputs[:, 4] = mu_quake[1]

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
        time = i
        inputs[:, 0] = time

        NN_time_0 = timer.time()
        ux = pinn.pinn_model_eval(inputs)[:, 0]
        uy = pinn.pinn_model_eval(inputs)[:, 1]
        ux_out = ux.detach()
        uy_out = uy.detach()

        np_ux_out = ux_out.numpy()
        np_uy_out = uy_out.numpy()

        B_ux = np.reshape(np_ux_out, (-1, int(np.sqrt(np_ux_out.shape[0]))))
        B_uy = np.reshape(np_uy_out, (-1, int(np.sqrt(np_uy_out.shape[0]))))
        B = np.sqrt(B_uy ** 2 + B_ux ** 2)
        res_list_ux.append(B_ux)
        res_list_uy.append(B_uy)
        res_list_u.append(B)
        NN_time_1 = timer.time()
        inferece_time = inferece_time + (NN_time_1 - NN_time_0)

    res_ux = np.dstack(res_list_ux)
    res_uy = np.dstack(res_list_uy)
    res_ux = np.rollaxis(res_ux, -1)
    res_uy = np.rollaxis(res_uy, -1)
    res_u = np.dstack(res_list_u)
    res_u = np.rollaxis(res_u, -1)

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

    if test_model == 'Vanilla':
        res_uy_model1 = res_uy
        res_ux_model1 = res_ux
        res_u_model1 = res_u
        res_list_devito_x_model1 = res_list_devito_x
        res_list_devito_y_model1 = res_list_devito_y
        res_list_devito_u_model1 = res_list_devito_u
    else:
        res_uy_model2 = res_uy
        res_ux_model2 = res_ux
        res_u_model2 = res_u
        res_list_devito_x_model2 = res_list_devito_x
        res_list_devito_y_model2 = res_list_devito_y
        res_list_devito_u_model2 = res_list_devito_u



s = 6 * np.mean((np.abs(res_list_devito_x[0])))

# Create an empty DataFrame with specified columns
df = pd.DataFrame(columns=['time', 'relative_l2', 'RRMSE'])


# Define the specific time steps to plot
time_steps_to_plot = [
    0,
    int(len(res_list_uy) / 4),
    int(len(res_list_uy) / 2),
    int(3 * int(len(res_list_uy) / 4)),
    len(res_list_uy) - 2
]
time_steps_to_plot = list(range(0,len(res_list_uy) - 2,1))
print(time_steps_to_plot)
time_values_to_plot = [time_list[h] for h in time_steps_to_plot]
title_positions = [0.288, 0.715]  # Fine-tuned positions for model names

# Create figure and GridSpec with appropriate layout
n_columns = len(time_steps_to_plot)
f = plt.figure(figsize=(10, 15))  # Adjust figure size as needed
f.subplots_adjust(left=.1, right=.9)
main_gs = gridspec.GridSpec(3, n_columns, hspace=0.1, wspace=0)  # Adjust hspace for spacing between groups

# Set row titles for the leftmost column
titles = ["$u_x$", "$u_y$", r"$|u|$"]
model_names = ["$\it{wED}$-PINN",r"PINN-tanh",]


s = 6 * np.mean(np.abs(res_list_devito_x[0]))

for h, time_value in enumerate(time_steps_to_plot):
    h = time_value
    fig = plt.figure(figsize=(24, 8))  # Adjust size as needed
    gs = gridspec.GridSpec(1, 2, figure=fig)

    # Loop over both models
    for model_idx in range(2):
        model_gs = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs[model_idx], hspace=0.0, wspace=-0.78)

        # Adjust these to use your model-specific data
        res_uy_model = res_uy_model1 if model_idx == 0 else res_uy_model2
        res_ux_model = res_ux_model1 if model_idx == 0 else res_ux_model2
        res_u_model = res_u_model1 if model_idx == 0 else res_u_model2
        res_list_devito_x_model = res_list_devito_x_model1 if model_idx == 0 else res_list_devito_x_model2
        res_list_devito_y_model = res_list_devito_y_model1 if model_idx == 0 else res_list_devito_y_model2
        res_list_devito_u_model = res_list_devito_u_model1 if model_idx == 0 else res_list_devito_u_model2

        for group_idx in range(3):  # Loop over groups (u_x, u_y, magnitude)
            for col_idx in range(3):  # Within each group, create subplots for PINN, DEVITO, difference
                ax = fig.add_subplot(model_gs[group_idx, col_idx])

                # Determine which data to plot based on group_idx and col_idx
                if group_idx == 0:  # u_x
                    data = [res_uy_model[h, :, :], res_list_devito_x_model[h],
                            res_uy_model[h, :, :] - res_list_devito_x_model[h]][col_idx]
                elif group_idx == 1:  # u_y
                    data = [res_ux_model[h, :, :], res_list_devito_y_model[h],
                            res_ux_model[h, :, :] - res_list_devito_y_model[h]][col_idx]
                else:  # magnitude
                    data = [res_u_model[h, :, :], res_list_devito_u_model[h],
                            res_u_model[h, :, :] - res_list_devito_u_model[h]][col_idx]

                ax.imshow(data, cmap='bwr', vmin=-s, vmax=s)
                ax.set_xticks([])
                ax.set_yticks([])
                if col_idx == 0:
                    ax.set_ylabel(titles[group_idx], fontsize=25, rotation=0, labelpad=10)
                if group_idx == 0:
                    ax.set_title(['PINN', 'DEVITO', 'Difference'][col_idx], fontsize=20)


                if group_idx == 2 and col_idx == 2:
                    # Determine the size of the rectangle
                    rect_width, rect_height = 0.25, 0.25  # Adjust these values as needed

                    # Calculate the position of the bottom-right corner
                    rect_x = 1 - rect_width
                    rect_y = 0

                    # Create and add the rectangle
                    rect = patches.Rectangle((rect_x, rect_y), rect_width, rect_height,
                                             transform=ax.transAxes, color='black', clip_on=False)
                    ax.add_patch(rect)
                    relative_l2_err = relative_l2_error(res_list_devito_u_model[h],res_u_model[h, :, :])
                    # Add L2 error text inside the rectangle
                    error_text = f"{round(float(relative_l2_err * 100), 1)}\%"  # Format the L2 error
                    ax.text(rect_x + rect_width / 2, rect_y + rect_height / 2, error_text,
                            transform=ax.transAxes, fontsize=12, color='white', va='center', ha='center')


        plt.figtext(title_positions[model_idx], 0.95, model_names[model_idx], ha='center', va='top', fontsize=25)

    # Adjust figure layout with the time as the overall title
    fig.suptitle(f'Time: {(time_value/100):.2f}s', fontsize=30)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the figure
    #plt.show()
    plt.savefig(f"PRESI_CONDITIONED/time_step_{h}_comparison_{mu_quake[0]}_{mu_quake[1]}.png")
    plt.close(fig)
