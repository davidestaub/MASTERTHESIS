
pth1 = 'Global_FullDomain_Conditioned_Pinns_accoustic_conditioned_1000000_400_1.0_200_200_100_4_64_tanh_0.2_FCN_ALL_PARAMS_PLANEWAVE_FCN_135_3_16_mixture_0.1'
pth2 = 'Global_FullDomain_Conditioned_Pinns_accoustic_1000000_400_1.0_200_200_100_7_128_tanh_0.2_PINN_1_mixture_0.1'


input_folder = pth1



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
import matplotlib.pyplot as plt

import sys

# Check if two arguments were passed (excluding the script name)
if len(sys.argv) == 3:
    # Convert arguments to integers or floats as needed
    start = float(sys.argv[1])  # Converting the first number
    stop = float(sys.argv[2])  # Converting the second number

else:
    print("This script requires exactly two numbers as arguments.")
    sys.exit(1)  # Exit the script with an error code




os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.manual_seed(42)
config = configparser.ConfigParser()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

numpoints_sqrt = 512

dt = 1.095e-3
# Define the range for random numbers
min_val, max_val = -0.45, 0.45


mu_quake_x_list = np.linspace(-0.4,0.4,40)
mu_quake_y_list = np.linspace(-0.4,0.4,40)

FD_prep_1 = timer.time()
config.read( input_folder + "/config.ini")

model_type = config['parameters']['model_type']


input_str = str(input_folder)
if 'layered' in input_str:
    str_path = input_str.replace("../pre_trained_models/layered_thesis/", "")
elif 'mixture' in input_str:
    str_path = input_str.replace("../pre_trained_models/mixture_thesis/", "")
elif 'constant' in input_str:
    str_path = input_str.replace("../pre_trained_models/hyper/", "")
else:
    print("unknong input str", input_str)
    raise Exception



t1 = float(config['initial_condition']['t1'])

devito_prep_time = 0.0

# Parameters
rho_solid = float(config['parameters']['rho_solid'])



pinn_domain_extent_x = [float(config["domain"]["xmin"]), float(config["domain"]["xmax"])]
pinn_domain_extent_y = [float(config["domain"]["ymin"]), float(config["domain"]["ymax"])]

FD_prep_0 = timer.time()

pinn_length_x = pinn_domain_extent_x[1] - pinn_domain_extent_x[0]
pinn_length_y = pinn_domain_extent_y[1] - pinn_domain_extent_y[0]

enlarge_factor = 3
devito_length_x = enlarge_factor * pinn_length_x
devito_length_y = enlarge_factor * pinn_length_y
extent = (devito_length_x, devito_length_y)
devito_center = [devito_length_x / 2.0, devito_length_y / 2.0]


# increase Devito domain to avoid Boundary issues (reflections)
shape = (numpoints_sqrt * enlarge_factor, numpoints_sqrt * enlarge_factor)
x = SpaceDimension(name='x', spacing=Constant(name='h_x', value=extent[0] / (shape[0] - 1)))
y = SpaceDimension(name='y', spacing=Constant(name='h_y', value=extent[1] / (shape[1] - 1)))
grid = Grid(extent=extent, shape=shape, dimensions=(x, y))
spacing = (extent[0] / shape[0], extent[0] / shape[0])
X, Y = np.meshgrid(np.linspace(0, numpoints_sqrt * spacing[0], numpoints_sqrt),
                   np.linspace(0, numpoints_sqrt * spacing[1], numpoints_sqrt))
print(X.shape, type(X), config["parameters"]["lambda_solid"])

FD_prep_1 = timer.time()
devito_prep_time = devito_prep_time + (FD_prep_1 - FD_prep_0)

if model_type == "mixture":
    # Generate mixtures for mu and lambda
    velocity_mixture = FD_devito.generate_acoustic_mixture_np()
    velocity = FD_devito.compute_param_np(X, Y, velocity_mixture)
elif model_type == "constant":
    velocity = np.full(X.shape, 1.0)
elif model_type == 'layered':
    velocity, _ = mixture_model.compute_lambda_mu_layers_accoustic_slow(torch.tensor(X), torch.tensor(Y), 5)
    velocity = velocity.numpy()
elif model_type == 'layered_sine':
    velocity, _ = mixture_model.create_sine_layer_map(torch.tensor(X), torch.tensor(Y))
else:
    raise Exception("model type {} not implemented".format(model_type))

# Create a larger array (7x7)
X_large, Y_large = numpoints_sqrt * enlarge_factor, numpoints_sqrt * enlarge_factor
velocity_large = np.full((X_large, Y_large), 0.0)
# Calculate the position to place the smaller array at the center of the larger array
x_offset = (X_large - numpoints_sqrt) // 2
y_offset = (Y_large - numpoints_sqrt) // 2

# Place the smaller array in the center of the larger array
velocity_large[x_offset:x_offset + numpoints_sqrt, y_offset:y_offset + numpoints_sqrt] = velocity
velocity_large[0:x_offset, :] = velocity_large[x_offset, :]
velocity_large[x_offset + numpoints_sqrt:, :] = velocity_large[x_offset + (numpoints_sqrt - 1), :]
velocity_large[:, 0:y_offset] = velocity_large[:, y_offset][:, np.newaxis]
velocity_large[:, y_offset + numpoints_sqrt:] = velocity_large[:, y_offset + (numpoints_sqrt - 1)][:, np.newaxis]

velocity = velocity_large

Xp, Yp = np.meshgrid(
    np.linspace(0, (numpoints_sqrt * enlarge_factor) * spacing[0], numpoints_sqrt * enlarge_factor),
    np.linspace(0, (numpoints_sqrt * enlarge_factor) * spacing[1], numpoints_sqrt * enlarge_factor))
# plt.show()

# t_prep = time.time()
# Using both ux and uy as separate field for easier comparison to PINNs and easier pde formulation
FD_prep_0 = timer.time()
u_devito = TimeFunction(name='ux_devito', grid=grid, space_order=4, time_order=2)

velocity_ = Function(name='velocity_f', grid=grid, space_order=4)

initialize_function(velocity_, velocity, nbl=0)




for mu_quake_x in mu_quake_x_list[int(start):int(stop)]:
    for mu_quake_y in mu_quake_y_list:


        mu_quake = [mu_quake_x, mu_quake_y]
        mu_quake = torch.tensor(mu_quake)

        for test_model in ['Vanilla','PLANEWAVE']:

            if test_model == 'Vanilla':
                save_path = 'Results/Acoustic/conditioned_thesis2/[{}_{}]/Vanilla/'.format(str(mu_quake_x),str(mu_quake_y))
                os.makedirs(save_path, exist_ok=True)
                input_folder = pth2
            else:
                save_path = 'Results/Acoustic/conditioned_thesis2/[{}_{}]/WAVELET/'.format(str(mu_quake_x),str(mu_quake_y))
                os.makedirs(save_path, exist_ok=True)
                input_folder = pth1

            config.read(input_folder+"/config.ini")
            t1 = float(config['initial_condition']['t1'])

            devito_prep_time = 0.0


            # Parameters
            rho_solid = float(config['parameters']['rho_solid'])

            model_path = input_folder+"/model.pth"
            print("model path = ",model_path)
            str_path = model_path
            model_type = config["parameters"]["model_type"]

            mu_quake_devito=[devito_center[0]+ mu_quake[0].numpy(),devito_center[1]+ mu_quake[1].numpy()]

            u0_devito = FD_devito.initial_condition_simple_gaussian(
                sigma=0.1, mu_quake=mu_quake_devito, nx=shape[0], ny=shape[1],
                dx=spacing[0], dy=spacing[1])

            # Initialize the VectorTimeFunction with the initial values
            # Setting both u(t0,x,y) and u(t0+dt,x,y) to u0 to guarantee du0/dt=0
            # and because 2nd order time discretization is neeed
            u_devito.data[0] = u0_devito
            u_devito.data[1] = u0_devito

            print("start laplacian calculation")
            laplacian = u_devito.dx2 + u_devito.dy2
            print("laplacian calculated")

            # Accoustic wave equation
            pde = laplacian - (1.0 / velocity_ ** 2) * u_devito.dt2

            # BOundary conditions:
            x, y = grid.dimensions
            t = grid.stepping_dim
            ny, nx = shape[0], shape[1]
            bc = [Eq(u_devito[t + 1, x, 0], 0.)]
            bc += [Eq(u_devito[t + 1, x, ny - 1], 0.)]
            bc += [Eq(u_devito[t + 1, 0, y], 0.)]
            bc += [Eq(u_devito[t + 1, nx - 1, y], 0.)]

            bc += [Eq(u_devito[t + 1, x, 1], 0.)]
            bc += [Eq(u_devito[t + 1, x, ny - 2], 0.)]
            bc += [Eq(u_devito[t + 1, 1, y], 0.)]
            bc += [Eq(u_devito[t + 1, nx - 2, y], 0.)]

            # Formulating stencil to solve for u forward
            print(type(u_devito.forward), type(u_devito), type(pde), u_devito)
            stencil_x = Eq(u_devito.forward, solve(pde, u_devito.forward))
            op = Operator([stencil_x] + bc)

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
                pinn = PINNs.Global_NSources_Conditioned_Pinns(int(config['Network']['n_points']), wandb_on=False,
                                                               config=config)
            elif network_type == "Relative_Distance_NSources_Conditioned_Pinns":
                pinn = PINNs.Relative_Distance_NSources_Conditioned_Pinns(int(config['Network']['n_points']),
                                                                          wandb_on=False,
                                                                          config=config)
            elif network_type == "Relative_Distance_FullDomain_Conditioned_Pinns":
                pinn = PINNs.Relative_Distance_FullDomain_Conditioned_Pinns(int(config['Network']['n_points']),
                                                                            wandb_on=False,
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

            if config['Network']['nn_type'] == 'FCN_ALL_PARAMS_WAVELET_FCN':
                print("correct")
                my_network = PINNs.FCN_all_params_Wavelet_FCN(input_dimension=5, output_dimension=1,
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
                my_network = PINNs.FCN_all_params_Planewave_FCN(input_dimension=5, output_dimension=1,
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
                my_network = PINNs.NeuralNet(input_dimension=5, output_dimension=1,
                                             n_hidden_layers=int(config['Network']['n_hidden_layers']),
                                             neurons=int(config['Network']['n_neurons']),
                                             regularization_param=0.,
                                             regularization_exp=2.,
                                             retrain_seed=3, activation=activation)

            my_network.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            my_network = my_network.to(device)
            pinn.approximate_solution = my_network


            #res_list_devito = []
            devito_time = 0.0

            inferece_time = 0.0
            FD_time = 0.0

            soboleng = torch.quasirandom.SobolEngine(dimension=pinn.domain_extrema.shape[0])
            inputs = soboleng.draw(int(pow(numpoints_sqrt, 2)))
            inputs = inputs.to(device)

            grid_x, grid_y = torch.meshgrid(torch.linspace(-1.0, 1.0, numpoints_sqrt),
                                            torch.linspace(-1.0, 1.0, numpoints_sqrt))
            grid_x = torch.reshape(grid_x, (-1,))
            grid_y = torch.reshape(grid_y, (-1,))

            inputs[:, 1] = grid_x
            inputs[:, 2] = grid_y
            inputs[:, 3] = mu_quake[0]
            inputs[:, 4] = mu_quake[1]

            res_list = []
            n=102
            dt = 1.096e-3

            time_list = np.linspace(0, 1, n).tolist()
            for i in time_list:
                time = i
                inputs[:, 0] = time

                NN_time_0 = timer.time()
                u = pinn.pinn_model_eval_accoustic(inputs)
                NN_time_1 = timer.time()
                inferece_time = inferece_time + (NN_time_1 - NN_time_0)
                u_out = u.detach()

                np_u_out = u_out.cpu().numpy()

                B_u = np.reshape(np_u_out, (-1, int(np.sqrt(np_u_out.shape[0]))))
                res_list.append(B_u)

            res_u = np.dstack(res_list)
            res = np.rollaxis(res_u, -1)

            for i in time_list:
                FD_t1 = timer.time()

                res_list_devito.append(np.transpose(
                    u_devito.data[1][x_offset:x_offset + numpoints_sqrt, y_offset:y_offset + numpoints_sqrt]).copy())
                op(time_M=10, dt=dt)
                FD_t2 = timer.time()
                FD_time = FD_time + (FD_t2 - FD_t1)

            RMSE = 0
            RRMSE = 0
            index = 1
            total_relative_l2_error = 0.0
            total_RRMSE = 0.0
            relative_l2_err = 0.0
            s = 12 * np.mean((np.abs(res_list_devito[0])))

            error_file = save_path+'running_errors.csv'

            # Create an empty DataFrame with specified columns
            df = pd.DataFrame(columns=['time', 'relative_l2', 'RRMSE'])

            for h in range(0, len(res_list)):
                diffu = ((res[h, :, :]) - (res_list_devito[h]))


                # Plot data
                if h == 0 or h == int(len(res_list)/4) or h==int(len(res_list)/2) or h==3*int(len(res_list)/4) or h == len(res_list)-1:
                    plt.imshow(res[h, :, :], cmap='bwr', vmin=-s, vmax=s)
                    plt.title('PINN')
                    plt.show()
                    plt.imshow(res_list_devito[h], cmap='bwr', vmin=-s, vmax=s)
                    plt.title('FD')
                    plt.show()
                diffu[0:9, :] = 0
                diffu[:, 0:9] = 0
                diffu[:, numpoints_sqrt - 9:numpoints_sqrt] = 0
                diffu[numpoints_sqrt - 9:numpoints_sqrt:, :] = 0

                RMSE = np.sqrt(np.mean(diffu ** 2))
                RRMSE = np.sqrt(np.mean(diffu ** 2)) / np.sqrt(np.mean(res_list_devito[h] ** 2))
                # print("RMSE:", RMSE, "RRMSE:", RRMSE * 100.0, "%")
                relative_l2_err = relative_l2_error(res_list_devito[h], res[h, :, :])
                # print("Relative L2 error: ", relative_l2_err * 100.0, "%")
                total_relative_l2_error = total_relative_l2_error + relative_l2_err
                total_RRMSE = total_RRMSE + RRMSE
                index = index + 1
                new_row = {'time': "time_{}".format(h), 'relative_l2': relative_l2_err, 'RRMSE': RRMSE}
                df = df.append(new_row, ignore_index=True)

            print("Average relative l2 error:", (total_relative_l2_error / len(res_list)) * 100.0, "%", " dt = ", dt, "n = ", n)
            print("Average RRMSE:", (total_RRMSE / len(res_list)) * 100.0, "%", "dt = ", dt)

            average_file = save_path + 'averages.csv'
            # Create an empty DataFrame with specified columns
            df_a = pd.DataFrame(columns=['average_l2', 'average_RRMSE'])
            new_row_a = {'average_l2': (total_relative_l2_error / len(res_list)) * 100.0,
                         'average_RRMSE': (total_RRMSE / len(res_list)) * 100.0}
            df_a = df_a.append(new_row_a, ignore_index=True)
            df_a.to_csv(average_file, index=False)