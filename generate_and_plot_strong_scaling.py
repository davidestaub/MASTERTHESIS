import matplotlib.patches as patches
import torch
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(128)
import numpy as np
import matplotlib.pyplot as plt
import time as timer
import pandas as pd
import matplotlib.gridspec as gridspec
import torch
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(128)
import torch.nn as nn
import os
import configparser
from devito import *
from sympy import pprint
import torch
from devito.builtins import initialize_function
import PINNs
import FD_devito
import mixture_model


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


vanilla_folder = '../pre_trained_models/strong_scaling/strong_scaling_vanilla/'
vsubfolders  = [ f.path for f in os.scandir(vanilla_folder) if f.is_dir() ]


wavelet_folder = '../pre_trained_models/strong_scaling/strong_scaling_WAVELET/'
wsubfolders = [ f.path for f in os.scandir(wavelet_folder) if f.is_dir() ]


numpoints_sqrt = 256
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

lambda_solid = np.full(X.shape, 20.0)
mu_solid = np.full(X.shape, 30.0)

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



for input_folder in vsubfolders:
#input_folder = '../pre_trained_models/constant_thesis/PINN_constant_thesis_100000_200_1.0_100_100_100_4_32_tanh_0.07_FCN_ALL_PARAMS_WAVELET_FCN_128_4_8_constant_0.1'



    dt=1.095e-3
    # Define the range for random numbers
    min_val, max_val = -0.45, 0.45

    mu_quake_x_list = [0.0]
    mu_quake_y_list = [0.0]


    FD_prep_1 = timer.time()
    config.read("../pre_trained_models/"+input_folder+"/config.ini")

    model_type = config['parameters']['model_type']



    mu_quake = [0.0, 0.0]
    mu_quake = torch.tensor(mu_quake)

    input_str = str(input_folder)
    if 'vanilla' in input_str:
        str_path = input_str.replace("../pre_trained_models/strong_scaling_vanilla/", "")
        save_path = '../../Results/Elastic/strong_scaling/256/vanilla/[{}_{}]/{}/'.format(str(mu_quake[0]),
                                                                                      str(mu_quake[1]), str_path)

    elif 'WAVELET' in input_str:
        str_path = input_str.replace("../pre_trained_models/strong_sclaing_WAVELET/", "")
        save_path = '../../Results/Elastic/strong_scaling/256/wavelet/[{}_{}]/{}/'.format(str(mu_quake[0]),
                                                                                      str(mu_quake[1]), str_path)

    else:
        print("unknong input str",input_str)
        raise Exception

    os.makedirs(save_path, exist_ok=True)

    sigma = float(str_path.split("_")[-1])

    config.read(input_folder + "/config.ini")
    t1 = float(config['initial_condition']['t1'])

    devito_prep_time = 0.0

    # Parameters
    rho_solid = float(config['parameters']['rho_solid'])

    model_path =  input_folder + "/model.pth"
    print("model path = ", model_path)

    model_type = config["parameters"]["model_type"]

    mu_quake_devito=[devito_center[0]+ mu_quake[0].numpy(),devito_center[1]+ mu_quake[1].numpy()]

    # The initial condition that also was used for The PINN, mimics a perfectly isotropic explosion
    u0x_devito, u0y_devito = FD_devito.initial_condition_double_gaussian_derivative(
        sigma=sigma, mu_quake=mu_quake_devito, nx=shape[0], ny=shape[1],
        dx=spacing[0], dy=spacing[1])

    # tmp = u0y_devito
    # u0y_devito = u0x_devito
    # u0x_devito = tmp

    # Initialize the VectorTimeFunction with the initial values
    # Setting both u(t0,x,y) and u(t0+dt,x,y) to u0 to guarantee du0/dt=0
    # and because 2nd order time discretization is neeed
    ux_devito.data[0] = u0x_devito
    uy_devito.data[0] = u0y_devito
    ux_devito.data[1] = u0x_devito
    uy_devito.data[1] = u0y_devito

    FD_devito.plot_field(ux_devito.data[0])
    FD_devito.plot_field(uy_devito.data[0])
    # Plot from initial field look good
    print(ux_devito.shape, uy_devito.shape)

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

    print("start")
    res_list_devito = []

    index = 0
    print("start")
    FD_prep_1 = timer.time()
    devito_prep_time = devito_prep_time + (FD_prep_1 - FD_prep_0)



    print("Model type not specified, reading from config file instead")
    network_type = config['Network']['model_type']
    if network_type == "Pinns":
        pinn = PINNs.Pinns(int(config['Network']['n_points']), wandb_on=False, config=config)
    elif network_type == "Global_NSources_Conditioned_Pinns":
        pinn = PINNs.Global_NSources_Conditioned_Pinns(int(config['Network']['n_points']), wandb_on=False, config=config)
    elif network_type == "Relative_Distance_NSources_Conditioned_Pinns":
        pinn = PINNs.Relative_Distance_NSources_Conditioned_Pinns(int(config['Network']['n_points']), wandb_on=False,
                                                                  config=config)
    elif network_type == "Relative_Distance_FullDomain_Conditioned_Pinns":
        pinn = PINNs.Relative_Distance_FullDomain_Conditioned_Pinns(int(config['Network']['n_points']), wandb_on=False,
                                                                    config=config)
    elif network_type == "Global_NSources_Conditioned_Lame_Pinns":
        pinn = PINNs.Global_NSources_Conditioned_Lame_Pinns(int(config['Network']['n_points']), wandb_on=False,
                                                            config=config)
    elif network_type == "Global_FullDomain_Conditioned_Pinns":
        pinn = PINNs.Global_FullDomain_Conditioned_Pinns(int(config['Network']['n_points']), wandb_on=False,
                                                         config=config)
    else:
        raise Exception("Model type {} not supported".format(network_type))

    tag = str_path.replace(".pth", "")
    tag = tag.replace("/model", "")

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
        raise Exception

    if config['Network']['nn_type'] == 'SIREN':
        # self, input_dimension, output_dimension, n_hidden_layers, neurons,
        # regularization_param, regularization_exp, retrain_seed, activation, n_1
        my_network = PINNs.SIREN_NeuralNet(input_dimension=3, output_dimension=2,
                                                    n_hidden_layers=int(config['Network']['n_hidden_layers']),
                                                    neurons=int(config['Network']['n_neurons']),
                                                    regularization_param=0.,
                                                    regularization_exp=2.,
                                                    retrain_seed=3, activation=activation,
                                                    n_1=int(config['Network']['n_1']))
    elif config['Network']['nn_type'] == 'PLANE_WAVE_FCN':
        my_network = PINNs.Planewave_and_FCN_NeuralNet(input_dimension=3, output_dimension=2,
                                                                n_hidden_layers=int(config['Network']['n_hidden_layers']),
                                                                neurons=int(config['Network']['n_neurons']),
                                                                regularization_param=0.,
                                                                regularization_exp=2.,
                                                                retrain_seed=3, activation=activation,
                                                                n_1=int(config['Network']['n_1']))
    elif config['Network']['nn_type'] == 'MORLET_WAVELET_FCN':
        my_network = PINNs.MorletWavelet_and_FCN(input_dimension=3, output_dimension=2,
                                                          n_hidden_layers=int(
                                                              config['Network']['n_hidden_layers']),
                                                          neurons=int(config['Network']['n_neurons']),
                                                          regularization_param=0.,
                                                          regularization_exp=2.,
                                                          retrain_seed=3, activation=activation,
                                                          n_1=int(config['Network']['n_1']))
    elif config['Network']['nn_type'] == 'FCN_AMPLITUDE_PLANEWAVE':
        my_network = PINNs.FCN_Amplitude_Planewave(input_dimension=3, output_dimension=2,
                                                            n_hidden_layers=int(
                                                                config['Network']['n_hidden_layers']),
                                                            neurons=int(config['Network']['n_neurons']),
                                                            regularization_param=0.,
                                                            regularization_exp=2.,
                                                            retrain_seed=3, activation=activation,
                                                            n_1=int(config['Network']['n_1']))
    elif config['Network']['nn_type'] == 'FCN_AMPLITUDE_WAVELET':
        my_network = PINNs.FCN_Amplitude_Wavelet(input_dimension=3, output_dimension=2,
                                                          n_hidden_layers=int(
                                                              config['Network']['n_hidden_layers']),
                                                          neurons=int(config['Network']['n_neurons']),
                                                          regularization_param=0.,
                                                          regularization_exp=2.,
                                                          retrain_seed=3, activation=activation,
                                                          n_1=int(config['Network']['n_1']))
    elif config['Network']['nn_type'] == 'FCN_ALL_PARAMS_WAVELET':
        my_network = PINNs.FCN_all_params_Wavelet_Modulation(input_dimension=3, output_dimension=2,
                                                                      n_hidden_layers=int(
                                                                          config['Network']['n_hidden_layers']),
                                                                      neurons=int(config['Network']['n_neurons']),
                                                                      regularization_param=0.,
                                                                      regularization_exp=2.,
                                                                      retrain_seed=3, activation=activation,
                                                                      n_1=int(config['Network']['n_1']))
    elif config['Network']['nn_type'] == 'FCN_ALL_PARAMS_PLANEWAVE':
        my_network = PINNs.FCN_all_params_Planewave_Modulation(input_dimension=3,
                                                                        output_dimension=2,
                                                                        n_hidden_layers=int(
                                                                            config['Network']['n_hidden_layers']),
                                                                        neurons=int(config['Network']['n_neurons']),
                                                                        regularization_param=0.,
                                                                        regularization_exp=2.,
                                                                        retrain_seed=3, activation=activation,
                                                                        n_1=int(config['Network']['n_1']))
    elif config['Network']['nn_type'] == 'FCN_ALL_PARAMS_WAVELET_FCN':
        print("correct")
        my_network = PINNs.FCN_all_params_Wavelet_FCN(input_dimension=3, output_dimension=2,
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
        my_network = PINNs.FCN_all_params_Planewave_FCN(input_dimension=3, output_dimension=2,
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
        my_network = PINNs.NeuralNet(input_dimension=3, output_dimension=2,
                                     n_hidden_layers=int(config['Network']['n_hidden_layers']),
                                     neurons=int(config['Network']['n_neurons']),
                                     regularization_param=0.,
                                     regularization_exp=2.,
                                     retrain_seed=3, activation=activation)

    my_network.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    pinn.approximate_solution = my_network

    if model_type == 'constant':
        numpoints_sqrt = 256
    elif model_type == 'mixture':
        numpoints_sqrt = 256
    elif (model_type == 'layered') or (model_type == 'layered_sine') or (model_type == 'layered_sine_simpler') or (
            model_type == 'layered_sine_simpler_3'):
        numpoints_sqrt = 256
    else:
        print("unsuported model type", model_type)
        raise Exception

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
    # inputs[:, 3] = mu_quake[0]
    # inputs[:, 4] = mu_quake[1]

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

    RMSE = 0
    RRMSE = 0
    index = 1
    total_relative_l2_error = 0.0
    total_RRMSE = 0.0
    relative_l2_err = 0.0

    error_file = save_path + 'running_errors.csv'

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
    time_values_to_plot = [time_list[h] for h in time_steps_to_plot]

    # Create figure and GridSpec with appropriate layout
    n_columns = len(time_steps_to_plot)
    f = plt.figure(figsize=(10, 20))  # Adjust figure size as needed
    f.subplots_adjust(left=.1, right=.9)
    main_gs = gridspec.GridSpec(3, n_columns, hspace=0.1, wspace=0)  # Adjust hspace for spacing between groups

    # Set row titles for the leftmost column
    titles = ["PINN $u_x$", "FD $u_x$", r"$\triangle u_x$", r"PINN $u_y$", r"FD $u_y$", r"$\triangle u_y$",
              r"PINN $|u|$", r"FD $|u|$", r"$\triangle |u|$"]

    for i in time_list:
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
        FD_t2 = timer.time()
        FD_time = FD_time + (FD_t2 - FD_t1)

    for h in range(0, len(res_list_uy)):
        diffx = ((res_uy[h, :, :]) - (res_list_devito_x[h]))
        diffx[0:9, :] = 0
        diffx[:, 0:9] = 0
        diffx[:, numpoints_sqrt - 9:numpoints_sqrt] = 0
        diffx[numpoints_sqrt - 9:numpoints_sqrt:, :] = 0

        diffy = (res_ux[h, :, :]) - (res_list_devito_y[h])
        diffy[0:9, :] = 0
        diffy[:, 0:9] = 0
        diffy[:, numpoints_sqrt - 9:numpoints_sqrt] = 0
        diffy[numpoints_sqrt - 9:numpoints_sqrt:, :] = 0

        diffu = ((res_u[h, :, :]) - (res_list_devito_u[h]))
        diffu[0:9, :] = 0
        diffu[:, 0:9] = 0
        diffu[:, numpoints_sqrt - 9:numpoints_sqrt] = 0
        diffu[numpoints_sqrt - 9:numpoints_sqrt:, :] = 0

        RMSE = np.sqrt(np.mean(diffu ** 2))
        RRMSE = np.sqrt(np.mean(diffu ** 2)) / np.sqrt(np.mean(res_list_devito_u[h] ** 2))
        # print("RMSE:", RMSE, "RRMSE:", RRMSE * 100.0, "%")
        relative_l2_err = relative_l2_error(res_list_devito_u[h], res_u[h, :, :])
        # print("Relative L2 error: ", relative_l2_err * 100.0, "%")
        total_relative_l2_error = total_relative_l2_error + relative_l2_err
        total_RRMSE = total_RRMSE + RRMSE
        index = index + 1
        new_row = {'time': f'time_{h}', 'relative_l2': relative_l2_err, 'RRMSE': RRMSE}
        df = df.append(new_row, ignore_index=True)

    s = 12 * np.mean(np.abs(res_list_devito_x[0]))

    # Plot the data for each selected time step
    for idx, (h, time_value) in enumerate(zip(time_steps_to_plot, time_values_to_plot)):
        # Calculate differences for time step 'h'
        diffx = res_uy[h, :, :] - res_list_devito_x[h]
        diffy = res_ux[h, :, :] - res_list_devito_y[h]
        diffu = res_u[h, :, :] - res_list_devito_u[h]

        relative_l2_err = relative_l2_error(res_list_devito_u[h], res_u[h, :, :])

        # Set the time title for the top of each column
        title_gs = main_gs[0, idx].get_gridspec()
        ax = f.add_subplot(title_gs.new_subplotspec((0, idx)))
        ax.set_title(f"Time: {time_value:.2f}s", fontsize=15, pad=20)
        ax.axis('off')  # Hide this axis

        # Apply zero-padding to differences
        for diff in [diffx, diffy, diffu]:
            diff[0:9, :] = 0
            diff[:, 0:9] = 0
            diff[:, numpoints_sqrt - 9:numpoints_sqrt] = 0
            diff[numpoints_sqrt - 9:numpoints_sqrt:, :] = 0

        for group_idx in range(3):
            # Create a subgrid for each group in the current column (time step)
            sub_gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=main_gs[group_idx, idx], hspace=0.0,
                                                      wspace=0.0)

            for row_idx in range(3):
                ax = f.add_subplot(sub_gs[row_idx, 0])

                # Calculate the overall row index
                i = group_idx * 3 + row_idx

                # Select data to plot
                if i < 3:  # First group (PINN)
                    data = res_uy[h, :, :] if i == 0 else res_list_devito_x[h] if i == 1 else diffx
                elif i < 6:  # Second group (FD)
                    data = res_ux[h, :, :] if i == 3 else res_list_devito_y[h] if i == 4 else diffy
                else:  # Third group (Difference)
                    data = res_u[h, :, :] if i == 6 else res_list_devito_u[h] if i == 7 else diffu

                # Plot data
                ax.imshow(data, cmap='bwr', vmin=-s, vmax=s)

                # Remove axis ticks
                ax.set_xticks([])
                ax.set_yticks([])

                # Set the row title for the first column
                if idx == 0:
                    ax.text(-0.15, 0.5, titles[i], transform=ax.transAxes, fontsize=15, va='center', ha='right',
                            rotation=0)

                # In the bottom row of each column, add the rectangle with L2 error text
                if group_idx == 2 and row_idx == 2:
                    # Determine the size of the rectangle
                    rect_width, rect_height = 0.25, 0.25  # Adjust these values as needed

                    # Calculate the position of the bottom-right corner
                    rect_x = 1 - rect_width
                    rect_y = 0

                    # Create and add the rectangle
                    rect = patches.Rectangle((rect_x, rect_y), rect_width, rect_height,
                                             transform=ax.transAxes, color='black', clip_on=False)
                    ax.add_patch(rect)

                    # Add L2 error text inside the rectangle
                    error_text = f"{round(float(relative_l2_err * 100), 1)}\%"  # Format the L2 error
                    ax.text(rect_x + rect_width / 2, rect_y + rect_height / 2, error_text,
                            transform=ax.transAxes, fontsize=12, color='white', va='center', ha='center')

    # Create an axes for the colorbar
    cbar_ax = f.add_axes([0.25, 0.05, 0.5, 0.02])  # [left, bottom, width, height]

    # Create a ScalarMappable with the same colormap and limits as your plots
    sm = plt.cm.ScalarMappable(cmap='bwr', norm=plt.Normalize(vmin=-s, vmax=s))

    # Create the colorbar
    f.colorbar(sm, cax=cbar_ax, orientation='horizontal')

    f.tight_layout()
    f.savefig(save_path + "selected_time_steps_comparison.png")
    f.show()

    df.to_csv(error_file, index=False)
    RMSE = RMSE / len(res_list_uy)
    RRMSE = RRMSE / len(res_list_uy)
    print(len(res_list_uy))
    print("Average relative l2 error:", (total_relative_l2_error / len(res_list_uy)) * 100.0, "%", " dt = ", dt, "n = ",
          n)
    print("Average RRMSE:", (total_RRMSE / len(res_list_uy)) * 100.0, "%", "dt = ", dt)

    average_file = save_path + 'averages.csv'
    # Create an empty DataFrame with specified columns
    df_a = pd.DataFrame(columns=['average_l2', 'average_RRMSE'])
    new_row_a = {'average_l2': (total_relative_l2_error / len(res_list_uy)) * 100.0,
                 'average_RRMSE': (total_RRMSE / len(res_list_uy)) * 100.0}
    df_a = df_a.append(new_row_a, ignore_index=True)
    df_a.to_csv(average_file, index=False)

