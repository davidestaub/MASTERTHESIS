pth = 'Global_FullDomain_Conditioned_Pinns_FCN_ALL_PARAMS_PLANEWAVE_FCN_500000_400_1.0_200_200_100_4_64_tanh_0.2_FCN_ALL_PARAMS_PLANEWAVE_FCN_135_3_16_mixture'

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
torch.set_printoptions(threshold=10_000)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.manual_seed(42)
config = configparser.ConfigParser()

input_folder = pth

config.read("../pre_trained_models/"+input_folder+"/config.ini")
t1 = float(config['initial_condition']['t1'])

# Parameters
rho_solid = float(config['parameters']['rho_solid'])

model_path = "../pre_trained_models/" + input_folder + "/model.pth"
print("model path = ", model_path)
str_path = model_path.replace("../pre_trained_models/", "")
model_type = config["parameters"]["model_type"]

print("Model type not specified, reading from config file instead")
model_type = config['Network']['model_type']

pinn = PINNs.Global_FullDomain_Conditioned_Pinns(int(config['Network']['n_points']), wandb_on=False,
                                                     config=config)

n_neurons = int(config['Network']['n_neurons'])
n_hidden_layers = int(config['Network']['n_hidden_layers'])
print(n_neurons, n_hidden_layers)
activation = nn.Tanh()



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

my_network.eval()
my_network.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
pinn.approximate_solution = my_network
total_time = 0


# Parameters
for i in range(0,1):
    with torch.no_grad():
        n_receivers = 63
        y_receiver_line = 1
        n_different_sources = 100  # Total number of different source locations
        total_points = n_receivers * n_different_sources
        points_per_source = 100  # Number of points per source for sampling

        # Determine the grid size
        grid_side = int(np.sqrt(n_different_sources))  # Calculate side length of the square grid
        actual_n_sources = grid_side ** 2  # Actual number of sources used

        # Generate grid points for source locations
        x_grid = np.linspace(-0.3, 0.3, grid_side)
        y_grid = np.linspace(-0.3, 0.3, grid_side)
        xx, yy = np.meshgrid(x_grid, y_grid)
        print("xx,yy = ",xx,yy)
        source_locations = np.vstack([xx.ravel(), yy.ravel()]).T  # Shape: (actual_n_sources, 2)
        print("source locations = ",source_locations)


        # Prepare receiver locations along y=0.99
        #receiver_x_locations = np.linspace(x_min, x_max, n_receivers)
        #print("receiver_loications = ",receiver_x_locations)

        # Sobol sequence generator for quasi-random sampling
        soboleng = torch.quasirandom.SobolEngine(dimension=5)

        receiver_x_locations = np.linspace(-1, 1, n_receivers)
        print("receiver_loications = ", receiver_x_locations)

        # Generate the base input tensor
        base_inputs = soboleng.draw(points_per_source * actual_n_sources)

        # Setup the inputs for all receivers, times, and source locations
        inputs = torch.zeros((n_receivers, points_per_source * actual_n_sources, 5))

        for i, (sx, sy) in enumerate(source_locations):
            start_idx = i * points_per_source
            end_idx = start_idx + points_per_source
            for j in range(n_receivers):
                inputs[j, start_idx:end_idx, 0] = torch.linspace(0.0, 1.0, points_per_source)
                inputs[j, start_idx:end_idx, 1] = receiver_x_locations[j]
                inputs[j, start_idx:end_idx, 2] = y_receiver_line
                inputs[j, start_idx:end_idx, 3] = sx
                inputs[j, start_idx:end_idx, 4] = sy

        # Flatten inputs to fit the model's expected input shape
        inputs_flattened = inputs.view(-1, 5)
        print(inputs_flattened)

        # Model evaluation in a single pass
        t0 = timer.time()
        u = pinn.pinn_model_eval(inputs_flattened)
        print(u)
        u_magnitude = torch.sqrt(u[:,0]**2 + u[:,1]**2)
        print("magnitude = ",u_magnitude)
        plt.plot(u_magnitude)
        plt.show()
        t1 = timer.time()
        print(f"Took {t1 - t0} seconds")
        total_time =total_time+t1 - t0
print("Total time = ",total_time)

# Process the output to find the maximum pressure for each receiver location
u_reshaped = u_magnitude.view(n_receivers, -1)  # Assuming u can be reshaped appropriately

# Calculate the absolute values of the tensor
u_abs = torch.abs(u_reshaped)


# Get the maximum of the absolute values along the specified dimension
max_pressures = u_abs.max(dim=1)[0].cpu().detach().numpy()



# Plotting
plt.figure(figsize=(10, 6))
plt.plot(receiver_x_locations, max_pressures, '-o', markersize=4, label='Max Pressure')
plt.title("Maximum Pressure at Different Receiver Locations")
plt.xlabel("Receiver Location X")
plt.ylabel("Maximum Pressure [Pa]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Calculate bar width
bar_width = receiver_x_locations[1] - receiver_x_locations[0]  # Assuming uniform spacing

# IEEE figure size guidelines: column width: 3.5 inches, page width: 7 inches
fig_width = 3.5  # or 7 for full-page width
fig_height = fig_width / 1.618  # Golden ratio to determine height

plt.figure(figsize=(fig_width, fig_height), dpi=300)

# Adjust font size and type
plt.rcParams.update({'font.size': 8, 'font.family': 'sans-serif', 'font.sans-serif': ['Arial']})

# Create bar plot
plt.bar(receiver_x_locations, max_pressures, width=0.1, color='b', edgecolor='k', alpha=0.7)

# Add labels, title, and grid
plt.title("Potential peak ground movement", fontsize=10)
plt.xlabel("Receiver Location X", fontsize=9)
plt.ylabel("Maximum Displacement Amplitude", fontsize=9)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Adjust line width and marker size
for line in plt.gca().lines:
    line.set_linewidth(1.5)

plt.tight_layout()
plt.show()
print("done")

